//! SPDX-License-Identifier: GPL-3.0-or-later
//! Model Context Protocol (MCP) JSON-RPC 2.0 server (`vouch mcp` and `POST /mcp`).
//!
//! Supports both the legacy handshake (`2024-11-05` `initialize` /
//! `notifications/initialized`) and the stateless `2026-07-28` specification
//! (`server/discover`, `Mcp-Method`/`Mcp-Name` header validation, `tools/list`
//! caching metadata, W3C Trace Context `_meta` extraction, and async `tasks/*`
//! execution).

use std::{
    collections::{BTreeMap, HashSet},
    net::SocketAddr,
    process::Stdio,
    sync::{Arc, atomic::Ordering},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router as AxumRouter,
    body::Bytes,
    extract::State,
    http::{
        HeaderMap, StatusCode,
        header::{CONTENT_TYPE, HeaderName, HeaderValue},
    },
    response::{IntoResponse, Response},
    routing::{get, post},
};
use dashmap::DashMap;
use serde_json::{Map, Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;
use tracing::Instrument as _;
use uuid::Uuid;

use crate::{
    auth::Principal,
    cli::{McpArgs, McpTransport},
    config::{GatewayConfig, McpUpstreamConfig, McpUpstreamTransport},
    driver::{ChatMessage, ExecutionRequest, WorkspaceMode, render_prompt},
    error::{GatewayError, Result},
    ingress::{completion, ops},
    metrics,
    router::derive_session_key,
    state::AppState,
    telemetry::{self, SpanKind, TraceContext},
};

const MCP_PROTOCOL_LEGACY: &str = "2024-11-05";
const MCP_PROTOCOL_STATELESS: &str = "2026-07-28";
const TOOLS_LIST_TTL_MS: u64 = 60_000;

/// Lifecycle state for an async MCP task (`tasks/create`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpTaskStatus {
    /// Execution in progress.
    Working,
    /// Execution succeeded.
    Completed,
    /// Execution failed with a gateway or backend error.
    Failed,
    /// Cancelled via `tasks/cancel` or gateway shutdown.
    Cancelled,
}

impl McpTaskStatus {
    /// Wire label defined by the MCP `Tasks` extension.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Working => "working",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone)]
struct McpTaskRecord {
    task_id: String,
    owner: String,
    status: McpTaskStatus,
    created_at_unix: u64,
    updated_at_unix: u64,
    touched: Instant,
    result: Option<Value>,
    error_message: Option<String>,
    cancel: CancellationToken,
    done: Arc<tokio::sync::Notify>,
}

/// Thread-safe in-memory registry backing MCP async `tasks/*` calls.
#[derive(Debug, Default)]
pub struct McpTaskStore {
    tasks: DashMap<String, McpTaskRecord>,
}

impl McpTaskStore {
    /// Create an empty task store.
    pub fn new() -> Self {
        Self::default()
    }

    fn insert_working(
        &self,
        owner: &str,
        cancel: CancellationToken,
    ) -> (String, u64, Arc<tokio::sync::Notify>) {
        let task_id = format!("task_{}", Uuid::new_v4().simple());
        let now_unix = unix_now_secs();
        let done = Arc::new(tokio::sync::Notify::new());
        self.tasks.insert(
            task_id.clone(),
            McpTaskRecord {
                task_id: task_id.clone(),
                owner: owner.to_string(),
                status: McpTaskStatus::Working,
                created_at_unix: now_unix,
                updated_at_unix: now_unix,
                touched: Instant::now(),
                result: None,
                error_message: None,
                cancel,
                done: done.clone(),
            },
        );
        (task_id, now_unix, done)
    }

    fn finish_success(&self, task_id: &str, result: Value) {
        if let Some(mut entry) = self.tasks.get_mut(task_id) {
            if entry.status != McpTaskStatus::Working {
                return;
            }
            entry.status = McpTaskStatus::Completed;
            entry.result = Some(result);
            entry.updated_at_unix = unix_now_secs();
            entry.touched = Instant::now();
            entry.done.notify_waiters();
        }
    }

    fn finish_error(&self, task_id: &str, message: String) {
        if let Some(mut entry) = self.tasks.get_mut(task_id) {
            if entry.status != McpTaskStatus::Working {
                return;
            }
            entry.status = McpTaskStatus::Failed;
            entry.error_message = Some(message);
            entry.updated_at_unix = unix_now_secs();
            entry.touched = Instant::now();
            entry.done.notify_waiters();
        }
    }

    fn cancel_task(&self, task_id: &str, owner: &str) -> std::result::Result<Value, (i32, String)> {
        let Some(mut entry) = self.tasks.get_mut(task_id) else {
            return Err((-32602, format!("unknown taskId: {task_id}")));
        };
        if entry.owner != owner {
            return Err((
                -32003,
                format!("taskId {task_id:?} belongs to another caller"),
            ));
        }
        entry.cancel.cancel();
        if entry.status == McpTaskStatus::Working {
            entry.status = McpTaskStatus::Cancelled;
            entry.updated_at_unix = unix_now_secs();
            entry.touched = Instant::now();
            entry.done.notify_waiters();
        }
        Ok(json!({
            "taskId": entry.task_id,
            "status": entry.status.as_str(),
            "createdAt": entry.created_at_unix,
            "updatedAt": entry.updated_at_unix,
        }))
    }

    /// Cancel every in-flight task during daemon shutdown.
    pub fn cancel_all(&self) {
        for mut entry in self.tasks.iter_mut() {
            entry.cancel.cancel();
            if entry.status == McpTaskStatus::Working {
                entry.status = McpTaskStatus::Cancelled;
                entry.updated_at_unix = unix_now_secs();
                entry.done.notify_waiters();
            }
        }
    }

    /// Remove expired completed/failed/cancelled tasks older than `ttl`.
    pub fn evict_expired(&self, ttl: Duration) {
        let expired: Vec<String> = self
            .tasks
            .iter()
            .filter(|entry| entry.status != McpTaskStatus::Working && entry.touched.elapsed() > ttl)
            .map(|entry| entry.key().clone())
            .collect();
        for key in expired {
            self.tasks.remove(&key);
        }
    }

    /// Spawn a background TTL eviction task.
    pub fn spawn_eviction_task(
        self: &Arc<Self>,
        ttl_secs: Arc<std::sync::atomic::AtomicU64>,
        shutdown: CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
        let store = self.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(30));
            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        let ttl = Duration::from_secs(ttl_secs.load(Ordering::Relaxed).max(1));
                        store.evict_expired(ttl);
                    }
                    _ = shutdown.cancelled() => break,
                }
            }
        })
    }
}

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
    if let Err(err_reply) = validate_stateless_headers(&headers, &payload) {
        return (StatusCode::BAD_REQUEST, Json(err_reply)).into_response();
    }
    match handle_rpc_with_headers(&state, &principal, Some(&headers), payload).await {
        Some(reply) => (StatusCode::OK, Json(reply)).into_response(),
        None => StatusCode::ACCEPTED.into_response(),
    }
}

/// Validate MCP `2026-07-28` `Mcp-Method` and `Mcp-Name` headers when supplied.
fn validate_stateless_headers(
    headers: &HeaderMap,
    payload: &Value,
) -> std::result::Result<(), Value> {
    let id = payload.get("id").cloned().unwrap_or(Value::Null);
    if let Some(header_val) = headers.get("mcp-method") {
        let expected_method = header_val.to_str().map(str::trim).unwrap_or("");
        let actual_method = payload.get("method").and_then(Value::as_str).unwrap_or("");
        if expected_method.is_empty() || expected_method != actual_method {
            return Err(json_rpc_error(
                id,
                -32600,
                "Invalid Request: Mcp-Method header does not match JSON-RPC method",
            ));
        }
    }
    if let Some(header_val) = headers.get("mcp-name") {
        let expected_name = header_val.to_str().map(str::trim).unwrap_or("");
        let actual_name = payload
            .pointer("/params/name")
            .or_else(|| payload.pointer("/params/tool"))
            .or_else(|| payload.pointer("/params/request/params/name"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if expected_name.is_empty() || expected_name != actual_name {
            return Err(json_rpc_error(
                id,
                -32600,
                "Invalid Request: Mcp-Name header does not match tool name",
            ));
        }
    }
    Ok(())
}

/// Process a single MCP JSON-RPC 2.0 request or notification.
///
/// Returns `None` for notifications (requests without an `id` field).
pub async fn handle_rpc(
    state: &Arc<AppState>,
    principal: &Principal,
    payload: Value,
) -> Option<Value> {
    handle_rpc_with_headers(state, principal, None, payload).await
}

/// Process an MCP JSON-RPC 2.0 request with optional HTTP ingress headers
/// (used to propagate W3C `traceparent`/`tracestate` headers as fallback to `_meta`).
pub async fn handle_rpc_with_headers(
    state: &Arc<AppState>,
    principal: &Principal,
    headers: Option<&HeaderMap>,
    payload: Value,
) -> Option<Value> {
    let id = payload.get("id").cloned();
    let Some(method) = payload
        .get("method")
        .and_then(Value::as_str)
        .map(ToString::to_string)
    else {
        return id.map(|id| json_rpc_error(id, -32600, "missing JSON-RPC method"));
    };

    if payload.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return id.map(|id| json_rpc_error(id, -32600, "jsonrpc must equal \"2.0\""));
    }

    // Notifications such as `notifications/initialized` produce no response.
    let id = id?;
    let params = payload.get("params").cloned().unwrap_or_else(|| json!({}));

    let meta = params.get("_meta").or_else(|| payload.get("_meta"));
    let trace_ctx = TraceContext::from_headers_and_meta(headers, meta);
    let traceparent = trace_ctx.traceparent();
    let tracestate = trace_ctx.tracestate.as_deref().unwrap_or("");
    let client_name = meta
        .and_then(|m| {
            m.get("client")
                .or_else(|| m.get("clientInfo").and_then(|c| c.get("name")))
                .and_then(Value::as_str)
        })
        .unwrap_or("");

    let span = tracing::info_span!(
        "mcp_rpc",
        method = %method,
        traceparent = %traceparent,
        tracestate = %tracestate,
        client = %client_name,
    );

    async move {
        let result = match method.as_str() {
            "initialize" => {
                let protocol = params
                    .get("protocolVersion")
                    .and_then(Value::as_str)
                    .unwrap_or(MCP_PROTOCOL_LEGACY);
                Ok(json!({
                    "protocolVersion": protocol,
                    "capabilities": {
                        "tools": { "listChanged": false },
                        "resources": { "subscribe": false, "listChanged": false },
                        "prompts": { "listChanged": false },
                        "tasks": { "list": true, "cancel": true }
                    },
                    "serverInfo": {
                        "name": "vouch",
                        "version": crate::VERSION,
                    }
                }))
            }
            "server/discover" => Ok(json!({
                "protocolVersion": MCP_PROTOCOL_STATELESS,
                "serverInfo": {
                    "name": "vouch",
                    "version": crate::VERSION,
                },
                "capabilities": {
                    "tools": { "listChanged": false },
                    "resources": { "subscribe": false, "listChanged": false },
                    "prompts": { "listChanged": false },
                    "tasks": { "list": true, "cancel": true }
                },
                "stateless": true,
            })),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(tools_list_payload(state).await),
            "tools/call" => {
                if params.get("task").is_some()
                    || params.pointer("/_meta/task").and_then(Value::as_bool) == Some(true)
                {
                    handle_task_create(state, principal, &params, trace_ctx)
                } else {
                    handle_tool_call(state, principal, &params, trace_ctx).await
                }
            }
            "resources/list" => Ok(resources_list_payload()),
            "resources/read" => handle_resources_read(state, &params),
            "prompts/list" => Ok(prompts_list_payload(state)),
            "prompts/get" => handle_prompts_get(state, &params),
            "tasks/create" => handle_task_create(state, principal, &params, trace_ctx),
            "tasks/get" => handle_task_get(state, principal, &params),
            "tasks/result" => handle_task_result(state, principal, &params).await,
            "tasks/cancel" => handle_task_cancel(state, principal, &params),
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
    .instrument(span)
    .await
}

async fn tools_list_payload(state: &AppState) -> Value {
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

    let upstreams = state.config().mcp_upstreams.clone();
    for upstream in &upstreams {
        match fetch_upstream_tools(upstream).await {
            Ok(remote_tools) => {
                for mut tool in remote_tools {
                    let Some(obj) = tool.as_object_mut() else {
                        continue;
                    };
                    let Some(orig_name) = obj
                        .get("name")
                        .and_then(Value::as_str)
                        .map(ToString::to_string)
                    else {
                        continue;
                    };
                    obj.insert(
                        "name".into(),
                        json!(format!("{}__{orig_name}", upstream.id)),
                    );
                    if let Some(desc) = obj
                        .get("description")
                        .and_then(Value::as_str)
                        .map(ToString::to_string)
                    {
                        obj.insert(
                            "description".into(),
                            json!(format!("[{}] {desc}", upstream.id)),
                        );
                    }
                    tools.push(tool);
                }
            }
            Err(error) => {
                tracing::warn!(
                    upstream = %upstream.id,
                    error = %error,
                    "failed discovering upstream MCP tools"
                );
            }
        }
    }

    json!({
        "tools": tools,
        "ttlMs": TOOLS_LIST_TTL_MS,
        "cacheScope": "private",
    })
}

fn prepare_tool_execution(
    state: &AppState,
    principal: &Principal,
    params: &Value,
    cancel: CancellationToken,
    trace_ctx: TraceContext,
) -> std::result::Result<(ExecutionRequest, WorkspaceMode), (i32, String)> {
    let tool_params = params.pointer("/request/params").unwrap_or(params);

    let tool_name = tool_params
        .get("name")
        .or_else(|| tool_params.get("tool"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or("vouch_ask");

    let args = tool_params
        .get("arguments")
        .and_then(Value::as_object)
        .or_else(|| tool_params.as_object())
        .ok_or_else(|| (-32602, "requires object arguments".to_string()))?;

    let models = state.router().models();
    let route_map = build_route_tool_map(&models);

    let (requested_label, model) = if tool_name == "vouch_ask" {
        let requested = args
            .get("model")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|m| !m.is_empty())
            .map(ToString::to_string);
        let resolved = state
            .resolve_model(requested.as_deref())
            .map_err(|error| (-32602, error.to_string()))?;
        (requested.unwrap_or_else(|| resolved.clone()), resolved)
    } else if let Some((_, alias_or_route)) =
        route_map.into_iter().find(|(name, _)| name == tool_name)
    {
        let resolved = state
            .resolve_model(Some(&alias_or_route))
            .map_err(|error| (-32602, error.to_string()))?;
        (alias_or_route, resolved)
    } else {
        return Err((-32602, format!("unknown MCP tool: {tool_name}")));
    };

    if !principal.allows_model_or_resolved(&requested_label, &model) {
        return Err((
            -32003,
            format!(
                "token {:?} is not allowed to use model {requested_label:?}",
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
                "tool call requires non-empty `arguments.prompt`".to_string(),
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
        cancel,
        tools: Vec::new(),
        tool_choice: None,
        response_format: None,
        backend_session_id: None,
        delta_messages: None,
        session_capture: Arc::new(parking_lot::Mutex::new(None)),
        trace_context: Some(trace_ctx),
        profile_id: None,
        profile_env: BTreeMap::new(),
        allowed_backends: None,
        extra: Map::new(),
    };
    Ok((request, workspace_mode))
}

async fn handle_tool_call(
    state: &AppState,
    principal: &Principal,
    params: &Value,
    trace_ctx: TraceContext,
) -> std::result::Result<Value, (i32, String)> {
    let tool_name = params
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| (-32602, "tools/call requires non-empty `name`".to_string()))?;
    let arguments = params
        .get("arguments")
        .and_then(Value::as_object)
        .ok_or_else(|| (-32602, "tools/call requires object `arguments`".to_string()))?;

    if let Some((upstream_id, remote_tool)) = tool_name.split_once("__") {
        let upstream = state
            .config()
            .mcp_upstreams
            .iter()
            .find(|u| u.id == upstream_id)
            .cloned();
        if let Some(upstream) = upstream {
            return proxy_upstream_tool_call(&upstream, remote_tool, arguments)
                .await
                .map_err(|error| (-32001, telemetry::redact_text(&error.to_string())));
        }
    }

    let (request, workspace_mode) = prepare_tool_execution(
        state,
        principal,
        params,
        CancellationToken::new(),
        trace_ctx.clone(),
    )?;
    let mut ingress_span = state
        .otel()
        .start_span(trace_ctx, "ingress", SpanKind::Server);
    ingress_span.set_str("vouch.route", &request.model);
    ingress_span.set_str("http.route", "/mcp");
    let started = Instant::now();
    let outcome = execute_mcp_request(state, request, workspace_mode).await;

    match outcome {
        Ok((backend_id, fallback, done)) => {
            let elapsed = started.elapsed();
            metrics::record_request("mcp", "/mcp", 200, elapsed);
            principal.record_tokens(done.usage.total_tokens);
            state
                .usage()
                .record_success(&principal.name, &backend_id, done.usage, elapsed);
            ingress_span.set_str("vouch.backend", &backend_id);
            ingress_span.set_bool("vouch.fallback", fallback);
            ingress_span.set_int("gen_ai.usage.input_tokens", done.usage.prompt_tokens as i64);
            ingress_span.set_int(
                "gen_ai.usage.output_tokens",
                done.usage.completion_tokens as i64,
            );
            ingress_span.finish_ok();
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
            let elapsed = started.elapsed();
            metrics::record_request("mcp", "/mcp", error.status().as_u16(), elapsed);
            state.usage().record_error(&principal.name, None, elapsed);
            let redacted = telemetry::redact_text(&error.to_string());
            ingress_span.finish_err(&redacted);
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

async fn execute_mcp_request(
    state: &AppState,
    request: ExecutionRequest,
    workspace_mode: WorkspaceMode,
) -> Result<(String, bool, crate::driver::Completion)> {
    let timeout = Duration::from_secs(state.config().server.request_timeout_secs.max(1));
    let routed = tokio::time::timeout(
        timeout,
        state
            .router()
            .execute(request, workspace_mode, state.workspaces().as_ref()),
    )
    .await;

    match routed {
        Ok(Ok(routed)) => {
            let done = completion(routed.response).await?;
            Ok((routed.backend_id, routed.fallback_used, done))
        }
        Ok(Err(error)) => Err(error),
        Err(_) => Err(GatewayError::Timeout(timeout)),
    }
}

fn handle_task_create(
    state: &Arc<AppState>,
    principal: &Principal,
    params: &Value,
    trace_ctx: TraceContext,
) -> std::result::Result<Value, (i32, String)> {
    let cancel = CancellationToken::new();
    let (request, workspace_mode) =
        prepare_tool_execution(state, principal, params, cancel.clone(), trace_ctx.clone())?;
    let mut ingress_span = state
        .otel()
        .start_span(trace_ctx, "ingress", SpanKind::Server);
    ingress_span.set_str("vouch.route", &request.model);
    ingress_span.set_str("http.route", "/mcp#tasks");
    let (task_id, created_at, _done) = state.mcp_tasks().insert_working(&principal.name, cancel);

    let bg_state = state.clone();
    let bg_principal = principal.clone();
    let bg_task_id = task_id.clone();
    tokio::spawn(async move {
        let started = Instant::now();
        match execute_mcp_request(&bg_state, request, workspace_mode).await {
            Ok((backend_id, fallback, done)) => {
                let elapsed = started.elapsed();
                metrics::record_request("mcp", "/mcp#tasks", 200, elapsed);
                bg_principal.record_tokens(done.usage.total_tokens);
                bg_state.usage().record_success(
                    &bg_principal.name,
                    &backend_id,
                    done.usage,
                    elapsed,
                );
                ingress_span.set_str("vouch.backend", &backend_id);
                ingress_span.set_bool("vouch.fallback", fallback);
                ingress_span.set_int("gen_ai.usage.input_tokens", done.usage.prompt_tokens as i64);
                ingress_span.set_int(
                    "gen_ai.usage.output_tokens",
                    done.usage.completion_tokens as i64,
                );
                ingress_span.finish_ok();
                bg_state.mcp_tasks().finish_success(
                    &bg_task_id,
                    json!({
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
                    }),
                );
            }
            Err(error) => {
                let elapsed = started.elapsed();
                metrics::record_request("mcp", "/mcp#tasks", error.status().as_u16(), elapsed);
                bg_state
                    .usage()
                    .record_error(&bg_principal.name, None, elapsed);
                let redacted = telemetry::redact_text(&error.to_string());
                ingress_span.finish_err(&redacted);
                bg_state.mcp_tasks().finish_error(&bg_task_id, redacted);
            }
        }
    });

    Ok(json!({
        "task": {
            "taskId": task_id,
            "status": McpTaskStatus::Working.as_str(),
            "createdAt": created_at,
            "pollIntervalMs": 200,
        },
        "taskId": task_id,
        "status": McpTaskStatus::Working.as_str(),
    }))
}

fn handle_task_get(
    state: &AppState,
    principal: &Principal,
    params: &Value,
) -> std::result::Result<Value, (i32, String)> {
    let task_id = params
        .get("taskId")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| (-32602, "tasks/get requires non-empty `taskId`".to_string()))?;

    let Some(entry) = state.mcp_tasks().tasks.get(task_id) else {
        return Err((-32602, format!("unknown taskId: {task_id}")));
    };
    if entry.owner != principal.name {
        return Err((
            -32003,
            format!("taskId {task_id:?} belongs to another caller"),
        ));
    }

    Ok(json!({
        "taskId": entry.task_id,
        "status": entry.status.as_str(),
        "createdAt": entry.created_at_unix,
        "updatedAt": entry.updated_at_unix,
    }))
}

async fn handle_task_result(
    state: &AppState,
    principal: &Principal,
    params: &Value,
) -> std::result::Result<Value, (i32, String)> {
    let task_id = params
        .get("taskId")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            (
                -32602,
                "tasks/result requires non-empty `taskId`".to_string(),
            )
        })?;

    let wait_ms = params
        .get("waitMs")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .min(10_000);

    if wait_ms > 0
        && let Some(notify) = state.mcp_tasks().tasks.get(task_id).and_then(|e| {
            if e.owner == principal.name && e.status == McpTaskStatus::Working {
                Some(e.done.clone())
            } else {
                None
            }
        })
    {
        let _ = tokio::time::timeout(Duration::from_millis(wait_ms), notify.notified()).await;
    }

    let Some(entry) = state.mcp_tasks().tasks.get(task_id) else {
        return Err((-32602, format!("unknown taskId: {task_id}")));
    };
    if entry.owner != principal.name {
        return Err((
            -32003,
            format!("taskId {task_id:?} belongs to another caller"),
        ));
    }

    match entry.status {
        McpTaskStatus::Working => Ok(json!({
            "taskId": entry.task_id,
            "status": McpTaskStatus::Working.as_str(),
        })),
        McpTaskStatus::Completed => {
            let mut res = entry.result.clone().unwrap_or_else(|| json!({}));
            if let Some(obj) = res.as_object_mut() {
                obj.insert("taskId".into(), json!(entry.task_id));
                obj.insert("status".into(), json!(McpTaskStatus::Completed.as_str()));
            }
            Ok(res)
        }
        McpTaskStatus::Failed => Ok(json!({
            "taskId": entry.task_id,
            "status": McpTaskStatus::Failed.as_str(),
            "isError": true,
            "content": [{
                "type": "text",
                "text": entry.error_message.clone().unwrap_or_else(|| "task failed".into()),
            }],
        })),
        McpTaskStatus::Cancelled => Ok(json!({
            "taskId": entry.task_id,
            "status": McpTaskStatus::Cancelled.as_str(),
            "isError": true,
            "content": [{
                "type": "text",
                "text": "task cancelled",
            }],
        })),
    }
}

fn handle_task_cancel(
    state: &AppState,
    principal: &Principal,
    params: &Value,
) -> std::result::Result<Value, (i32, String)> {
    let task_id = params
        .get("taskId")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            (
                -32602,
                "tasks/cancel requires non-empty `taskId`".to_string(),
            )
        })?;

    state.mcp_tasks().cancel_task(task_id, &principal.name)
}

fn resources_list_payload() -> Value {
    json!({
        "resources": [
            {
                "uri": "vouch://health",
                "name": "Gateway Health",
                "description": "Live daemon readiness, circuit breaker status, version, and uptime.",
                "mimeType": "application/json",
            },
            {
                "uri": "vouch://routes",
                "name": "Configured Model Routes",
                "description": "Public model routes, aliases, scheduling policies, and backend chains.",
                "mimeType": "application/json",
            },
            {
                "uri": "vouch://usage",
                "name": "Usage & Cost Analytics",
                "description": "Cumulative per-token and per-backend BPE tokens, latency percentiles, and estimated USD spend.",
                "mimeType": "application/json",
            }
        ]
    })
}

fn handle_resources_read(
    state: &AppState,
    params: &Value,
) -> std::result::Result<Value, (i32, String)> {
    let uri = params
        .get("uri")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|u| !u.is_empty())
        .ok_or_else(|| {
            (
                -32602,
                "resources/read requires non-empty `uri`".to_string(),
            )
        })?;

    let payload = match uri {
        "vouch://health" => {
            let config = state.config();
            let readiness = state.router().readiness();
            json!({
                "status": "ok",
                "version": crate::VERSION,
                "uptime_secs": state.uptime_secs(),
                "ready": readiness.ready,
                "tokens_enforced": state.tokens().enabled(),
                "workspace_enabled": config.workspace.enabled,
                "backends": readiness.backends,
            })
        }
        "vouch://routes" => {
            let config = state.config();
            let models = state.router().model_info();
            json!({
                "models": models,
                "aliases": config.aliases,
                "route_policies": config.route_policies,
            })
        }
        "vouch://usage" => {
            let budgets = state.tokens().budget_statuses();
            let report = state.usage().report(state.uptime_secs(), &budgets);
            serde_json::to_value(&report)
                .map_err(|error| (-32603, format!("cannot serialize usage report: {error}")))?
        }
        other => return Err((-32602, format!("unknown resource uri: {other:?}"))),
    };

    let text = serde_json::to_string_pretty(&payload)
        .map_err(|error| (-32603, format!("cannot encode resource JSON: {error}")))?;
    Ok(json!({
        "contents": [{
            "uri": uri,
            "mimeType": "application/json",
            "text": text,
        }]
    }))
}

fn prompts_list_payload(state: &AppState) -> Value {
    let prompts: Vec<Value> = state
        .config()
        .mcp_prompts
        .iter()
        .map(|prompt| {
            let args: Vec<Value> = prompt
                .arguments
                .iter()
                .map(|arg| {
                    json!({
                        "name": arg.name,
                        "description": arg.description,
                        "required": arg.required,
                    })
                })
                .collect();
            json!({
                "name": prompt.name,
                "description": prompt.description,
                "arguments": args,
            })
        })
        .collect();
    json!({ "prompts": prompts })
}

fn handle_prompts_get(
    state: &AppState,
    params: &Value,
) -> std::result::Result<Value, (i32, String)> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .ok_or_else(|| (-32602, "prompts/get requires non-empty `name`".to_string()))?;

    let cfg = state.config();
    let Some(prompt) = cfg.mcp_prompts.iter().find(|p| p.name == name) else {
        return Err((-32602, format!("unknown MCP prompt {name:?}")));
    };
    let supplied = params
        .get("arguments")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    for decl in &prompt.arguments {
        if decl.required {
            let present = supplied
                .get(&decl.name)
                .is_some_and(|v| !v.is_null() && v.as_str().map(|s| !s.is_empty()).unwrap_or(true));
            if !present {
                return Err((
                    -32602,
                    format!("prompt {name:?} missing required argument {:?}", decl.name),
                ));
            }
        }
    }

    let mut rendered = prompt.template.clone();
    for (k, v) in &supplied {
        let replacement = match v {
            Value::String(s) => s.clone(),
            Value::Null => String::new(),
            other => other.to_string(),
        };
        rendered = rendered.replace(&format!("{{{{{k}}}}}"), &replacement);
    }

    Ok(json!({
        "description": prompt.description,
        "messages": [{
            "role": "user",
            "content": {
                "type": "text",
                "text": rendered,
            }
        }]
    }))
}

async fn fetch_upstream_tools(upstream: &McpUpstreamConfig) -> Result<Vec<Value>> {
    let resp = call_upstream_mcp(upstream, "tools/list", json!({})).await?;
    let tools = resp
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    Ok(tools)
}

async fn proxy_upstream_tool_call(
    upstream: &McpUpstreamConfig,
    tool_name: &str,
    arguments: &Map<String, Value>,
) -> Result<Value> {
    call_upstream_mcp(
        upstream,
        "tools/call",
        json!({
            "name": tool_name,
            "arguments": arguments,
        }),
    )
    .await
}

async fn call_upstream_mcp(
    upstream: &McpUpstreamConfig,
    method: &str,
    params: Value,
) -> Result<Value> {
    let timeout = Duration::from_secs(upstream.timeout_secs.max(1));
    match upstream.transport {
        McpUpstreamTransport::Http => {
            let base_url = upstream.base_url.as_deref().ok_or_else(|| {
                GatewayError::Config(format!("mcp_upstream {:?} missing base_url", upstream.id))
            })?;
            let client = reqwest::Client::builder()
                .timeout(timeout)
                .build()
                .map_err(|error| {
                    GatewayError::Internal(format!("cannot build MCP HTTP client: {error}"))
                })?;
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            for (k, v) in &upstream.extra_headers {
                let name = HeaderName::from_bytes(k.as_bytes()).map_err(|error| {
                    GatewayError::Config(format!(
                        "mcp_upstream {:?} invalid header {k:?}: {error}",
                        upstream.id
                    ))
                })?;
                let val = HeaderValue::from_str(v).map_err(|error| {
                    GatewayError::Config(format!(
                        "mcp_upstream {:?} invalid header value for {k:?}: {error}",
                        upstream.id
                    ))
                })?;
                headers.insert(name, val);
            }
            let body = json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params,
            });
            let response = client
                .post(base_url)
                .headers(headers)
                .json(&body)
                .send()
                .await
                .map_err(|error| {
                    GatewayError::Upstream(format!(
                        "mcp_upstream {:?} HTTP call failed: {error}",
                        upstream.id
                    ))
                })?;
            let status = response.status();
            let text = response.text().await.map_err(|error| {
                GatewayError::Upstream(format!(
                    "mcp_upstream {:?} read failed: {error}",
                    upstream.id
                ))
            })?;
            if !status.is_success() {
                return Err(GatewayError::Upstream(format!(
                    "mcp_upstream {:?} returned HTTP {status}: {}",
                    upstream.id,
                    telemetry::redact_text(&text)
                )));
            }
            let parsed: Value = serde_json::from_str(&text).map_err(|error| {
                GatewayError::Upstream(format!(
                    "mcp_upstream {:?} returned invalid JSON: {error}",
                    upstream.id
                ))
            })?;
            extract_rpc_result(&upstream.id, parsed)
        }
        McpUpstreamTransport::Stdio => {
            let command = upstream.command.as_deref().ok_or_else(|| {
                GatewayError::Config(format!("mcp_upstream {:?} missing command", upstream.id))
            })?;
            let mut cmd = Command::new(command);
            cmd.args(&upstream.args)
                .envs(&upstream.env)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true);
            #[cfg(unix)]
            cmd.process_group(0);

            let mut child = cmd.spawn().map_err(|error| {
                GatewayError::Upstream(format!(
                    "cannot spawn mcp_upstream {:?}: {error}",
                    upstream.id
                ))
            })?;
            let pid = child.id().unwrap_or(0);
            #[cfg(unix)]
            let guard = if pid > 0 {
                Some(crate::driver::ProcessGroupGuard::new(
                    pid,
                    upstream.id.clone(),
                ))
            } else {
                None
            };

            let req_future = async {
                let mut stdin = child
                    .stdin
                    .take()
                    .ok_or_else(|| GatewayError::Internal("upstream stdio missing stdin".into()))?;
                let stdout = child.stdout.take().ok_or_else(|| {
                    GatewayError::Internal("upstream stdio missing stdout".into())
                })?;

                let init_msg = json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": MCP_PROTOCOL_LEGACY,
                        "capabilities": {},
                        "clientInfo": { "name": "vouch", "version": crate::VERSION }
                    }
                });
                let notify_msg = json!({
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized"
                });
                let call_msg = json!({
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": method,
                    "params": params,
                });
                let payload = format!("{init_msg}\n{notify_msg}\n{call_msg}\n");
                stdin.write_all(payload.as_bytes()).await?;
                let _ = stdin.shutdown().await;

                let mut lines = BufReader::new(stdout).lines();
                while let Some(line) = lines.next_line().await? {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if let Ok(val) = serde_json::from_str::<Value>(trimmed)
                        && val.get("id").and_then(Value::as_i64) == Some(2)
                    {
                        return extract_rpc_result(&upstream.id, val);
                    }
                }
                Err(GatewayError::Upstream(format!(
                    "mcp_upstream {:?} closed stdout before responding to {method}",
                    upstream.id
                )))
            };

            let res = tokio::time::timeout(timeout, req_future).await;
            #[cfg(unix)]
            if let Some(g) = guard {
                g.terminate();
            }
            let _ = tokio::time::timeout(Duration::from_secs(1), child.wait()).await;
            match res {
                Ok(inner) => inner,
                Err(_) => Err(GatewayError::Timeout(timeout)),
            }
        }
    }
}

fn extract_rpc_result(upstream_id: &str, response: Value) -> Result<Value> {
    if let Some(error) = response.get("error") {
        let msg = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("unknown MCP error");
        return Err(GatewayError::Upstream(format!(
            "mcp_upstream {upstream_id:?} returned error: {}",
            telemetry::redact_text(msg)
        )));
    }
    response.get("result").cloned().ok_or_else(|| {
        GatewayError::Upstream(format!(
            "mcp_upstream {upstream_id:?} response missing `result`"
        ))
    })
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

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
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
