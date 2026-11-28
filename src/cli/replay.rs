//! SPDX-License-Identifier: GPL-3.0-or-later
//! Offline flight-log replay and regression benchmark (`vouch replay`).

use std::{
    io::Write,
    sync::Arc,
    time::{Duration, Instant},
};

use serde::Serialize;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::{
    cli::ReplayArgs,
    config::GatewayConfig,
    driver::workspace::WorkspaceManager,
    driver::{
        DriverResponse, ExecutionRequest, ResponseFormat, WorkspaceMode, collect_stream_full,
        render_prompt, tool_bridge::enforce_response_format,
    },
    error::{GatewayError, Result},
    recorder::{FlightRecord, load_flight_records, resolve_flight_path},
    router::{Router, SessionAffinity},
};

/// Result of replaying one recorded request against the current configuration.
#[derive(Debug, Clone, Serialize)]
pub struct ReplayItemResult {
    /// 1-based record index within the replay slice.
    pub index: usize,
    /// Protocol of the original call.
    pub protocol: String,
    /// Route used during replay.
    pub route: String,
    /// Backend that handled the original call.
    pub original_backend: String,
    /// Backend that won during replay.
    pub replay_backend: Option<String>,
    /// Original total duration in milliseconds.
    pub original_ms: u64,
    /// Replay duration in milliseconds.
    pub replay_ms: u64,
    /// True if schema/output validation succeeded (or was not constrained).
    pub schema_valid: bool,
    /// Overall status (`"ok"` or error description).
    pub status: String,
}

/// Replay flight records through the configured router and print a comparison report.
pub async fn run(args: &ReplayArgs) -> Result<()> {
    let config = GatewayConfig::load(&args.config)?;
    let path = args.file.clone().unwrap_or_else(|| {
        resolve_flight_path(
            &config.flight_recorder,
            config.defaults.state_dir.as_deref(),
        )
    });
    let mut records = load_flight_records(&path)?;
    let limit = args.limit.unwrap_or(20).max(1);
    if records.len() > limit {
        let start = records.len() - limit;
        records = records.split_off(start);
    }

    let max_entries = config.defaults.max_session_entries.max(1);
    let sessions = Arc::new(SessionAffinity::with_max_entries(max_entries));
    let router = Router::new(&config, sessions)?;
    let workspaces = WorkspaceManager::new(config.workspace.clone());
    let timeout = Duration::from_secs(config.server.request_timeout_secs.max(1));

    let mut results = Vec::with_capacity(records.len());
    for (idx, rec) in records.iter().enumerate() {
        let item = replay_one(
            idx + 1,
            rec,
            &config,
            &router,
            &workspaces,
            args.route.as_deref(),
            timeout,
        )
        .await;
        results.push(item);
    }

    if config.workspace.cleanup_on_shutdown {
        workspaces.cleanup_all().await;
    }

    if args.json {
        let passed = results.iter().filter(|r| r.status == "ok").count();
        let failed = results.len().saturating_sub(passed);
        let payload = json!({
            "file": path.display().to_string(),
            "total": results.len(),
            "passed": passed,
            "failed": failed,
            "results": results,
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(
        stdout,
        "vouch replay — {} ({} record(s))",
        path.display(),
        results.len()
    );
    let _ = writeln!(
        stdout,
        "{:<5} {:<20} {:<14} {:<14} {:>8} {:>9} {:<7} STATUS",
        "#", "ROUTE", "ORIG_BACKEND", "REPLAY_BACKEND", "ORIG_MS", "REPLAY_MS", "SCHEMA"
    );
    for row in &results {
        let _ = writeln!(
            stdout,
            "{:<5} {:<20} {:<14} {:<14} {:>8} {:>9} {:<7} {}",
            row.index,
            row.route,
            row.original_backend,
            row.replay_backend.as_deref().unwrap_or("-"),
            row.original_ms,
            row.replay_ms,
            if row.schema_valid { "ok" } else { "FAIL" },
            row.status
        );
    }
    Ok(())
}

async fn replay_one(
    index: usize,
    rec: &FlightRecord,
    config: &GatewayConfig,
    router: &Router,
    workspaces: &WorkspaceManager,
    route_override: Option<&str>,
    timeout: Duration,
) -> ReplayItemResult {
    let target_model = route_override.unwrap_or(&rec.resolved_route).to_string();
    let resolved_route = config
        .resolve_route_name(&target_model)
        .unwrap_or(&target_model)
        .to_string();

    let response_format = rec.response_format.as_ref().and_then(parse_replay_format);
    let prompt = render_prompt(rec.system.as_deref(), &rec.normalized_messages);
    let request = ExecutionRequest {
        request_id: format!("replay-{index}"),
        session_key: format!("replay-{index}"),
        model: resolved_route.clone(),
        system: rec.system.clone(),
        messages: rec.normalized_messages.clone(),
        prompt,
        stream: false,
        temperature: None,
        max_tokens: None,
        workspace_mode: WorkspaceMode::None,
        workdir: None,
        cancel: CancellationToken::new(),
        tools: rec.tools.clone(),
        tool_choice: None,
        response_format: response_format.clone(),
        backend_session_id: None,
        delta_messages: None,
        session_capture: Arc::new(parking_lot::Mutex::new(None)),
        trace_context: None,
        profile_id: None,
        profile_env: Default::default(),
        allowed_backends: None,
        extra: Default::default(),
    };

    let started = Instant::now();
    let routed = match tokio::time::timeout(
        timeout,
        router.execute(request, WorkspaceMode::None, workspaces),
    )
    .await
    {
        Ok(Ok(r)) => r,
        Ok(Err(error)) => {
            return ReplayItemResult {
                index,
                protocol: rec.protocol.clone(),
                route: resolved_route,
                original_backend: rec.winning_backend.clone(),
                replay_backend: None,
                original_ms: rec.total_ms,
                replay_ms: started.elapsed().as_millis() as u64,
                schema_valid: false,
                status: format!("error: {error}"),
            };
        }
        Err(_) => {
            return ReplayItemResult {
                index,
                protocol: rec.protocol.clone(),
                route: resolved_route,
                original_backend: rec.winning_backend.clone(),
                replay_backend: None,
                original_ms: rec.total_ms,
                replay_ms: started.elapsed().as_millis() as u64,
                schema_valid: false,
                status: "error: replay timed out".into(),
            };
        }
    };

    let replay_backend = routed.backend_id.clone();
    let completion = match routed.response {
        DriverResponse::Complete(c) => Ok(c),
        DriverResponse::Stream(stream) => match collect_stream_full(stream, 16 * 1024 * 1024).await
        {
            Ok(collected) => Ok(crate::driver::Completion {
                content: collected.content,
                finish_reason: collected.finish_reason.unwrap_or_else(|| "stop".into()),
                usage: collected.usage.unwrap_or_default(),
                raw: None,
                tool_calls: collected.tool_calls,
                thinking: collected.thinking,
            }),
            Err(error) => Err(error),
        },
    };
    let replay_ms = started.elapsed().as_millis() as u64;

    match completion {
        Ok(c) => {
            let schema_valid = if c.tool_calls.is_empty()
                && let Some(rf) = response_format.as_ref()
            {
                enforce_response_format(&c.content, rf).is_ok()
            } else {
                true
            };
            let status = if schema_valid {
                "ok".into()
            } else {
                "schema validation failed".into()
            };
            ReplayItemResult {
                index,
                protocol: rec.protocol.clone(),
                route: resolved_route,
                original_backend: rec.winning_backend.clone(),
                replay_backend: Some(replay_backend),
                original_ms: rec.total_ms,
                replay_ms,
                schema_valid,
                status,
            }
        }
        Err(error) => ReplayItemResult {
            index,
            protocol: rec.protocol.clone(),
            route: resolved_route,
            original_backend: rec.winning_backend.clone(),
            replay_backend: Some(replay_backend),
            original_ms: rec.total_ms,
            replay_ms,
            schema_valid: false,
            status: format!("error: {error}"),
        },
    }
}

fn parse_replay_format(val: &Value) -> Option<ResponseFormat> {
    let kind = val.get("type").and_then(Value::as_str)?;
    match kind {
        "json_object" => Some(ResponseFormat::JsonObject),
        "json_schema" => {
            let js = val.get("json_schema")?;
            let schema = js.get("schema")?.clone();
            let name = js
                .get("name")
                .and_then(Value::as_str)
                .map(ToString::to_string);
            let strict = js.get("strict").and_then(Value::as_bool).unwrap_or(false);
            Some(ResponseFormat::JsonSchema {
                name,
                schema,
                strict,
                raw: val.clone(),
            })
        }
        _ => None,
    }
}

/// Convenience error check helper when parsing flight files fails.
pub fn validate_replay_file(path: &std::path::Path) -> Result<usize> {
    load_flight_records(path)
        .map(|v| v.len())
        .map_err(|err| match err {
            GatewayError::Io(e) => GatewayError::Io(e),
            other => other,
        })
}
