//! SPDX-License-Identifier: GPL-3.0-or-later
//! Model Context Protocol (MCP) JSON-RPC 2.0 server (`vouch mcp` and `POST /mcp`).
//!
//! Exposes configured `vouch` routes dynamically as MCP tools (`vouch_ask`
//! plus one `route_<name>` tool per configured model). Every invocation routes
//! through [`crate::router::Router`] so circuit breakers, semaphores, sticky
//! sessions, worktree isolation, metrics, and secret redaction apply identically
//! to HTTP ingress.

use std::{collections::HashSet, net::SocketAddr, sync::Arc, time::Duration};

use axum::{
    Json, Router as AxumRouter,
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::{Map, Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    auth::Principal,
    cli::{McpArgs, McpTransport},
    config::GatewayConfig,
    driver::{ChatMessage, ExecutionRequest, WorkspaceMode, render_prompt},
    error::{GatewayError, Result},
    ingress::{completion, ops},
    metrics,
    router::derive_session_key,
    state::AppState,
    telemetry,
};

const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// Run `vouch mcp` in stdio or HTTP transport mode.
pub async fn run(args: &McpArgs) -> Result<()> {
    let config = GatewayConfig::load(&args.config)?;
    let log_level = config.server.log_level.clone();
    let secrets = config.secrets();
    telemetry::init_stderr(&log_level, secrets)?;

    let state = AppState::new(args.config.clone(), config)?;
    state.start_background_tasks();

    let result = match args.transport {
        McpTransport::Stdio => run_stdio(state.clone()).await,
        McpTransport::Http => run_http(state.clone(), args.bind).await,
    };

    state.shutdown().await;
    result
}

async fn run_stdio(state: Arc<AppState>) -> Result<()> {
    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin);
    let mut stdout = tokio::io::stdout();
    let mut line = String::new();

    loop {
        line.clear();
        let max_bytes = state.config().server.max_body_bytes.max(1024);
        let read = tokio::select! {
            _ = tokio::signal::ctrl_c() => break,
            res = reader.read_line(&mut line) => res.map_err(GatewayError::Io)?,
        };
        if read == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.len() > max_bytes {
            let err = json_rpc_error(Value::Null, -32600, "request exceeds max_body_bytes");
            write_rpc_line(&mut stdout, &err).await?;
            continue;
        }
        let response = match serde_json::from_str::<Value>(trimmed) {
            Ok(payload) => handle_rpc(&state, &Principal::local(), payload).await,
            Err(error) => Some(json_rpc_error(
                Value::Null,
                -32700,
                &format!("parse error: {error}"),
            )),
        };
        if let Some(reply) = response {
            write_rpc_line(&mut stdout, &reply).await?;
        }
    }
    Ok(())
}

async fn write_rpc_line(stdout: &mut tokio::io::Stdout, value: &Value) -> Result<()> {
    let serialized = serde_json::to_string(value)?;
    stdout.write_all(serialized.as_bytes()).await?;
    stdout.write_all(b"\n").await?;
    stdout.flush().await?;
    Ok(())
}

async fn run_http(state: Arc<AppState>, bind: SocketAddr) -> Result<()> {
    let app = AxumRouter::new()
        .route("/mcp", post(http_mcp))
        .route("/healthz", get(ops::healthz))
        .route("/readyz", get(ops::readyz))
        .route("/metrics", get(ops::metrics))
        .with_state(state.clone());

    let listener = tokio::net::TcpListener::bind(bind).await.map_err(|error| {
        GatewayError::Io(std::io::Error::new(
            error.kind(),
            format!("cannot bind MCP HTTP listener {bind}: {error}"),
        ))
    })?;
    tracing::info!(address = %bind, "MCP HTTP listener ready");

    let shutdown = state.shutdown_token();
    let ctrl_c_state = state.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        ctrl_c_state.shutdown_token().cancel();
    });

    axum::serve(listener, app)
        .with_graceful_shutdown(async move { shutdown.cancelled().await })
        .await
        .map_err(|error| GatewayError::Internal(format!("MCP HTTP server failed: {error}")))
}

/// Axum handler for `POST /mcp` (used by both `vouch serve` and `vouch mcp --transport http`).
pub async fn http_mcp(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let principal = match state.tokens().authenticate(&headers) {
        Ok(principal) => principal,
        Err(error) => return error.into_openai_response(),
    };
    let payload: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(error) => {
            let reply = json_rpc_error(Value::Null, -32700, &format!("parse error: {error}"));
            return (StatusCode::BAD_REQUEST, Json(reply)).into_response();
        }
    };
    match handle_rpc(&state, &principal, payload).await {
        Some(reply) => (StatusCode::OK, Json(reply)).into_response(),
        None => StatusCode::ACCEPTED.into_response(),
    }
}

/// Process a single MCP JSON-RPC 2.0 request or notification.
///
/// Returns `None` for notifications (requests without an `id` field).
pub async fn handle_rpc(state: &AppState, principal: &Principal, payload: Value) -> Option<Value> {
    let id = payload.get("id").cloned();
    let Some(method) = payload.get("method").and_then(Value::as_str) else {
        return id.map(|id| json_rpc_error(id, -32600, "missing JSON-RPC method"));
    };

    if payload.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return id.map(|id| json_rpc_error(id, -32600, "jsonrpc must equal \"2.0\""));
    }

    // Notifications such as `notifications/initialized` produce no response.
    let id = id?;
    let params = payload.get("params").cloned().unwrap_or_else(|| json!({}));

    let result = match method {
        "initialize" => {
            let protocol = params
                .get("protocolVersion")
                .and_then(Value::as_str)
                .unwrap_or(MCP_PROTOCOL_VERSION);
            Ok(json!({
                "protocolVersion": protocol,
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "vouch",
                    "version": crate::VERSION,
                }
            }))
        }
        "ping" => Ok(json!({})),
        "tools/list" => Ok(tools_list_payload(state)),
        "tools/call" => handle_tool_call(state, principal, &params).await,
        other => Err((-32601, format!("method not found: {other}"))),
    };

    Some(match result {
        Ok(result) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        }),
        Err((code, message)) => json_rpc_error(id, code, &message),
    })
}

fn tools_list_payload(state: &AppState) -> Value {
    let models = state.router().models();
    let route_tools = build_route_tool_map(&models);

    let mut tools = Vec::with_capacity(route_tools.len() + 1);
    let mut model_schema = Map::new();
    model_schema.insert("type".into(), json!("string"));
    model_schema.insert(
        "description".into(),
        json!("Configured vouch public model route"),
    );
    if !models.is_empty() {
        model_schema.insert("enum".into(), json!(models));
    }

    tools.push(json!({
        "name": "vouch_ask",
        "description": "Send a prompt to any configured vouch model route with failover, circuit breaking, and session affinity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model": Value::Object(model_schema),
                "prompt": {
                    "type": "string",
                    "description": "User prompt to send to the model"
                },
                "system": {
                    "type": "string",
                    "description": "Optional system instruction"
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional sticky session identifier"
                },
                "workspace_mode": {
                    "type": "string",
                    "enum": ["auto", "worktree", "none"],
                    "description": "Optional git worktree isolation override"
                }
            },
            "required": if state.router().default_model().is_some() {
                json!(["prompt"])
            } else {
                json!(["model", "prompt"])
            }
        }
    }));

    for (tool_name, route_name) in route_tools {
        tools.push(json!({
            "name": tool_name,
            "description": format!("Invoke vouch route `{route_name}`"),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "User prompt to send to this route"
                    },
                    "system": {
                        "type": "string",
                        "description": "Optional system instruction"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional sticky session identifier"
                    },
                    "workspace_mode": {
                        "type": "string",
                        "enum": ["auto", "worktree", "none"],
                        "description": "Optional git worktree isolation override"
                    }
                },
                "required": ["prompt"]
            }
        }));
    }

    json!({ "tools": tools })
}

async fn handle_tool_call(
    state: &AppState,
    principal: &Principal,
    params: &Value,
) -> std::result::Result<Value, (i32, String)> {
    let tool_name = params
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| (-32602, "tools/call requires non-empty `name`".to_string()))?;

    let args = params
        .get("arguments")
        .and_then(Value::as_object)
        .ok_or_else(|| (-32602, "tools/call requires object `arguments`".to_string()))?;

    let models = state.router().models();
    let route_map = build_route_tool_map(&models);

    let model = if tool_name == "vouch_ask" {
        let requested = args.get("model").and_then(Value::as_str);
        state
            .resolve_model(requested)
            .map_err(|error| (-32602, error.to_string()))?
    } else if let Some((_, route_name)) = route_map.into_iter().find(|(name, _)| name == tool_name)
    {
        route_name
    } else {
        return Err((-32602, format!("unknown MCP tool: {tool_name}")));
    };

    if !principal.allows_model(&model) {
        return Err((
            -32003,
            format!(
                "token {:?} is not allowed to use model {model:?}",
                principal.name
            ),
        ));
    }

    let prompt_text = args
        .get("prompt")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| {
            (
                -32602,
                "tools/call requires non-empty `arguments.prompt`".to_string(),
            )
        })?
        .to_string();

    let system = args
        .get("system")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(ToString::to_string);

    let explicit_session = args.get("session_id").and_then(Value::as_str);
    let session_key = derive_session_key(explicit_session, system.as_deref(), &prompt_text);

    let workspace_mode = match args.get("workspace_mode").and_then(Value::as_str) {
        Some(raw) => WorkspaceMode::parse(raw).ok_or_else(|| {
            (
                -32602,
                format!("invalid workspace_mode {raw:?}; expected auto, worktree, or none"),
            )
        })?,
        None => WorkspaceMode::Auto,
    };

    let messages = vec![ChatMessage::text("user", prompt_text)];
    let rendered = render_prompt(system.as_deref(), &messages);
    let request = ExecutionRequest {
        request_id: format!("mcp-{}", Uuid::new_v4().simple()),
        session_key,
        model,
        system,
        messages,
        prompt: rendered,
        stream: false,
        temperature: None,
        max_tokens: None,
        workspace_mode,
        workdir: None,
        cancel: CancellationToken::new(),
        tools: Vec::new(),
        tool_choice: None,
        extra: Map::new(),
    };

    let started = std::time::Instant::now();
    let timeout = Duration::from_secs(state.config().server.request_timeout_secs.max(1));
    let routed = tokio::time::timeout(
        timeout,
        state
            .router()
            .execute(request, workspace_mode, state.workspaces().as_ref()),
    )
    .await;

    let outcome = match routed {
        Ok(Ok(routed)) => match completion(routed.response).await {
            Ok(done) => Ok((routed.backend_id, done)),
            Err(error) => Err(error),
        },
        Ok(Err(error)) => Err(error),
        Err(_) => Err(GatewayError::Timeout(timeout)),
    };

    match outcome {
        Ok((backend_id, done)) => {
            metrics::record_request("mcp", "/mcp", 200, started.elapsed());
            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": done.content,
                }],
                "isError": false,
                "_meta": {
                    "backend": backend_id,
                    "usage": {
                        "prompt_tokens": done.usage.prompt_tokens,
                        "completion_tokens": done.usage.completion_tokens,
                        "total_tokens": done.usage.total_tokens,
                    }
                }
            }))
        }
        Err(error) => {
            metrics::record_request("mcp", "/mcp", error.status().as_u16(), started.elapsed());
            let redacted = telemetry::redact_text(&error.to_string());
            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": redacted,
                }],
                "isError": true,
            }))
        }
    }
}

/// Build deterministic, collision-free `route_<sanitized>` tool names.
pub fn build_route_tool_map(models: &[String]) -> Vec<(String, String)> {
    let mut used = HashSet::new();
    let mut entries = Vec::with_capacity(models.len());
    for model in models {
        let base = sanitize_route_name(model);
        let mut candidate = base.clone();
        let mut suffix = 2usize;
        while !used.insert(candidate.clone()) {
            candidate = format!("{base}_{suffix}");
            suffix += 1;
        }
        entries.push((candidate, model.clone()));
    }
    entries
}

fn sanitize_route_name(model: &str) -> String {
    let mut slug = String::from("route_");
    let mut prev_underscore = true;
    for ch in model.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            prev_underscore = false;
        } else if !prev_underscore {
            slug.push('_');
            prev_underscore = true;
        }
    }
    while slug.ends_with('_') && slug.len() > "route_".len() {
        slug.pop();
    }
    if slug == "route_" {
        slug.push_str("default");
    }
    slug
}

fn json_rpc_error(id: Value, code: i32, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": telemetry::redact_text(message),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizes_route_names_and_avoids_collisions() {
        let map = build_route_tool_map(&[
            "coding/default".into(),
            "coding.default".into(),
            "hermes/agent".into(),
        ]);
        assert_eq!(
            map[0],
            ("route_coding_default".into(), "coding/default".into())
        );
        assert_eq!(
            map[1],
            ("route_coding_default_2".into(), "coding.default".into())
        );
        assert_eq!(map[2], ("route_hermes_agent".into(), "hermes/agent".into()));
    }
}
