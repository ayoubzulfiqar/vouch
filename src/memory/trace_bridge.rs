//! SPDX-License-Identifier: GPL-3.0-or-later
//! Pooled stdio MCP client bridge to `trace` (`https://github.com/ayoubzulfiqar/trace`).
//!
//! Spawns a persistent `trace serve` child per normalized workspace root supervised
//! by [`ProcessGroupGuard`], executes bounded JSON-RPC 2.0 tool queries without any
//! LLM calls, and degrades gracefully when `trace` is unavailable.

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use arc_swap::ArcSwap;
use dashmap::DashMap;
use serde::Serialize;
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::Mutex,
};

use crate::{
    config::{TraceBridgeConfig, WorkspaceConfig},
    driver::ProcessGroupGuard,
    error::{GatewayError, Result},
    memory::{
        handoff::truncate_to_bpe_budget,
        ledger::{extract_sanitized_repo_paths, sanitize_relative_repo_path},
    },
    metrics, telemetry,
    usage::count_tokens,
};

const MCP_PROTOCOL_VERSION: &str = "2024-11-05";
const MAX_RPC_LINE_BYTES: usize = 2 * 1024 * 1024;
const UNSUPPORTED_BACKOFF: Duration = Duration::from_secs(2);

/// Structured status reported by `vouch doctor` and `vouch_session_status`.
#[derive(Debug, Clone, Serialize)]
pub struct TraceProbeStatus {
    /// Whether `[trace].enabled = true`.
    pub enabled: bool,
    /// Whether `[trace].required = true`.
    pub required: bool,
    /// Effective workspace root probed.
    pub workspace_root: String,
    /// Configured executable command.
    pub command: String,
    /// Whether the probe succeeded.
    pub ok: bool,
    /// Number of repository rules returned by `list_rules`, if reachable.
    pub rules_count: Option<usize>,
    /// Diagnostic detail or error summary.
    pub detail: String,
}

#[derive(Debug, Default)]
struct TraceWorkerState {
    process: Option<TraceProcess>,
    backoff_until: Option<Instant>,
    warned_unavailable: bool,
}

#[derive(Debug)]
struct TraceProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    guard: ProcessGroupGuard,
}

impl TraceProcess {
    async fn terminate_and_reap(mut self) {
        self.guard.terminate();
        let _ = tokio::time::timeout(Duration::from_secs(2), self.child.wait()).await;
    }
}

/// Thread-safe pool of model-free `trace serve` stdio workers keyed by workspace root.
#[derive(Debug)]
pub struct TraceBridge {
    config: ArcSwap<TraceBridgeConfig>,
    workspace_config: ArcSwap<WorkspaceConfig>,
    workers: DashMap<PathBuf, Arc<Mutex<TraceWorkerState>>>,
    next_id: AtomicU64,
}

impl TraceBridge {
    /// Create a bridge from initial configuration.
    pub fn new(config: TraceBridgeConfig, workspace_config: WorkspaceConfig) -> Self {
        Self {
            config: ArcSwap::from_pointee(config),
            workspace_config: ArcSwap::from_pointee(workspace_config),
            workers: DashMap::new(),
            next_id: AtomicU64::new(1),
        }
    }

    /// Atomically apply hot-reloaded config and close stale workers if command/args changed.
    pub fn update_config(&self, config: TraceBridgeConfig, workspace_config: WorkspaceConfig) {
        let prev = self.config.load();
        let changed = prev.enabled != config.enabled
            || prev.command != config.command
            || prev.args != config.args
            || prev.workspace_root != config.workspace_root;
        self.config.store(Arc::new(config));
        self.workspace_config.store(Arc::new(workspace_config));
        if changed {
            self.workers.clear();
        }
    }

    /// Effective configuration snapshot.
    pub fn config(&self) -> Arc<TraceBridgeConfig> {
        self.config.load_full()
    }

    /// Resolve effective workspace root for requests without an explicit header override.
    pub fn default_workspace_root(&self) -> PathBuf {
        let cfg = self.config.load();
        let ws = self.workspace_config.load();
        effective_base_root(&cfg, &ws)
    }

    /// Gather bounded, priority-ordered repository facts from `trace` with zero LLM calls.
    ///
    /// Priority order within `max_bpe_tokens`:
    /// 1. `list_rules`
    /// 2. `search_decisions`
    /// 3. `get_symbol_outline` + `find_callers` (up to 5 existing verified files)
    /// 4. `get_recent_history` (limit 5)
    pub async fn gather_repo_context(
        &self,
        workspace_root_override: Option<&Path>,
        model: &str,
        candidate_files: &[String],
        prompt_text: &str,
        max_bpe_tokens: usize,
    ) -> Option<String> {
        let cfg = self.config.load();
        if !cfg.enabled || max_bpe_tokens == 0 {
            return None;
        }
        let ws_root = workspace_root_override
            .map(Path::to_path_buf)
            .unwrap_or_else(|| self.default_workspace_root());
        let verified_files = verify_existing_files(&ws_root, candidate_files, prompt_text, 5);

        let rules_text = match self.call_tool(&ws_root, "list_rules", json!({})).await {
            Ok(val) => format_mcp_tool_text(&val),
            Err(TraceCallErr::Unavailable) => return None,
            Err(TraceCallErr::ToolError) => String::new(),
        };

        let query_terms = build_decision_query(&verified_files, prompt_text);
        let decisions_text = match self
            .call_tool(
                &ws_root,
                "search_decisions",
                json!({ "query": query_terms }),
            )
            .await
        {
            Ok(val) => format_mcp_tool_text(&val),
            Err(TraceCallErr::Unavailable) => {
                if rules_text.is_empty() {
                    return None;
                }
                String::new()
            }
            Err(TraceCallErr::ToolError) => String::new(),
        };

        let mut outline_sections = Vec::new();
        let mut combined_outline_raw = String::new();
        for rel in &verified_files {
            if let Ok(val) = self
                .call_tool(&ws_root, "get_symbol_outline", json!({ "path": rel }))
                .await
            {
                let text = format_mcp_tool_text(&val);
                if !text.trim().is_empty() {
                    combined_outline_raw.push_str(&text);
                    combined_outline_raw.push('\n');
                    outline_sections.push(format!("- {rel}:\n{}", indent_block(text.trim(), 2)));
                }
            }
        }

        let symbols = extract_candidate_symbols(&combined_outline_raw, prompt_text);
        let mut caller_sections = Vec::new();
        for sym in symbols.into_iter().take(3) {
            if let Ok(val) = self
                .call_tool(&ws_root, "find_callers", json!({ "symbol": sym }))
                .await
            {
                let text = format_mcp_tool_text(&val);
                if !text.trim().is_empty() {
                    caller_sections.push(format!("- {sym}: {}", compact_one_line(text.trim())));
                }
            }
        }

        let history_text = match self
            .call_tool(&ws_root, "get_recent_history", json!({ "limit": 5 }))
            .await
        {
            Ok(val) => format_mcp_tool_text(&val),
            Err(_) => String::new(),
        };

        let mut sections: Vec<(String, String)> = Vec::new();
        if !rules_text.trim().is_empty() {
            sections.push(("Trace Architectural Rules".into(), rules_text.trim().into()));
        }
        if !decisions_text.trim().is_empty() {
            sections.push(("Trace Decisions".into(), decisions_text.trim().into()));
        }
        if !outline_sections.is_empty() || !caller_sections.is_empty() {
            let mut combined = outline_sections.join("\n");
            if !caller_sections.is_empty() {
                if !combined.is_empty() {
                    combined.push_str("\nCallers:\n");
                }
                combined.push_str(&caller_sections.join("\n"));
            }
            sections.push(("Trace Symbol Outlines & Callers".into(), combined));
        }
        if !history_text.trim().is_empty() {
            sections.push(("Trace Recent History".into(), history_text.trim().into()));
        }

        if sections.is_empty() {
            return None;
        }

        let assembled = assemble_budgeted_sections(model, &sections, max_bpe_tokens);
        (!assembled.trim().is_empty()).then_some(assembled)
    }

    /// Fire-and-forget bounded checkpoint write to `trace` via `record_session`.
    pub fn record_session_checkpoint_async(
        self: &Arc<Self>,
        workspace_root_override: Option<PathBuf>,
        summary: String,
        files_modified: Vec<String>,
    ) {
        let cfg = self.config.load();
        if !cfg.enabled {
            return;
        }
        let bridge = self.clone();
        let clean_summary = telemetry::redact_text(&summary);
        let ws_root = workspace_root_override.unwrap_or_else(|| self.default_workspace_root());
        let clean_files: Vec<String> = files_modified
            .into_iter()
            .filter_map(|p| sanitize_relative_repo_path(&p, Some(&ws_root)))
            .take(32)
            .collect();
        tokio::spawn(async move {
            let _ = bridge
                .call_tool(
                    &ws_root,
                    "record_session",
                    json!({
                        "summary": clean_summary,
                        "files_modified": clean_files,
                    }),
                )
                .await;
        });
    }

    /// Synchronously invoke `record_session` (used by `POST /v1/sessions/{id}/checkpoint` and MCP `vouch_checkpoint`).
    pub async fn record_session_checkpoint_now(
        &self,
        workspace_root_override: Option<&Path>,
        summary: &str,
        files_modified: &[String],
    ) -> Result<bool> {
        let cfg = self.config.load();
        if !cfg.enabled {
            return Ok(false);
        }
        let ws_root = workspace_root_override
            .map(Path::to_path_buf)
            .unwrap_or_else(|| self.default_workspace_root());
        let clean_summary = telemetry::redact_text(summary);
        let clean_files: Vec<String> = files_modified
            .iter()
            .filter_map(|p| sanitize_relative_repo_path(p, Some(&ws_root)))
            .take(32)
            .collect();
        match self
            .call_tool(
                &ws_root,
                "record_session",
                json!({
                    "summary": clean_summary,
                    "files_modified": clean_files,
                }),
            )
            .await
        {
            Ok(_) => Ok(true),
            Err(TraceCallErr::Unavailable) if !cfg.required => Ok(false),
            Err(_) => Err(GatewayError::Upstream(
                "trace record_session invocation failed".into(),
            )),
        }
    }

    /// Probe `trace serve` health (`initialize` + `list_rules`) for `vouch doctor`.
    pub async fn doctor_probe(&self, workspace_root_override: Option<&Path>) -> TraceProbeStatus {
        let cfg = self.config.load();
        let ws_root = workspace_root_override
            .map(Path::to_path_buf)
            .unwrap_or_else(|| self.default_workspace_root());
        if !cfg.enabled {
            return TraceProbeStatus {
                enabled: false,
                required: cfg.required,
                workspace_root: ws_root.display().to_string(),
                command: cfg.command.clone(),
                ok: true,
                rules_count: None,
                detail: "trace integration disabled".into(),
            };
        }

        // Force fresh attempt in doctor regardless of previous short backoff.
        let key = normalize_root_key(&ws_root);
        if let Some(worker) = self.workers.get(&key) {
            let mut state = worker.lock().await;
            state.backoff_until = None;
        }

        match self.call_tool(&ws_root, "list_rules", json!({})).await {
            Ok(val) => {
                let text = format_mcp_tool_text(&val);
                let rules_count = count_items_in_result(&val, &text);
                TraceProbeStatus {
                    enabled: true,
                    required: cfg.required,
                    workspace_root: ws_root.display().to_string(),
                    command: cfg.command.clone(),
                    ok: true,
                    rules_count: Some(rules_count),
                    detail: format!("trace serve initialized ({rules_count} rule entry(s))"),
                }
            }
            Err(err) => TraceProbeStatus {
                enabled: true,
                required: cfg.required,
                workspace_root: ws_root.display().to_string(),
                command: cfg.command.clone(),
                ok: false,
                rules_count: None,
                detail: match err {
                    TraceCallErr::Unavailable => {
                        format!(
                            "cannot start `{}` against {}",
                            cfg.command,
                            ws_root.display()
                        )
                    }
                    TraceCallErr::ToolError => {
                        format!("`{}` returned an RPC error on list_rules", cfg.command)
                    }
                },
            },
        }
    }

    /// Terminate all pooled `trace serve` subprocesses during graceful shutdown.
    pub async fn shutdown(&self) {
        let keys: Vec<PathBuf> = self.workers.iter().map(|e| e.key().clone()).collect();
        for key in keys {
            if let Some((_, worker)) = self.workers.remove(&key) {
                let mut guard = worker.lock().await;
                if let Some(proc) = guard.process.take() {
                    proc.terminate_and_reap().await;
                }
            }
        }
    }

    async fn call_tool(
        &self,
        workspace_root: &Path,
        tool_name: &str,
        arguments: Value,
    ) -> std::result::Result<Value, TraceCallErr> {
        let cfg = self.config.load();
        if !cfg.enabled {
            return Err(TraceCallErr::Unavailable);
        }
        let key = normalize_root_key(workspace_root);
        let worker_arc = self
            .workers
            .entry(key)
            .or_insert_with(|| Arc::new(Mutex::new(TraceWorkerState::default())))
            .clone();

        let mut worker = worker_arc.lock().await;
        if let Some(until) = worker.backoff_until
            && Instant::now() < until
        {
            metrics::record_trace_query(tool_name, "unavailable");
            return Err(TraceCallErr::Unavailable);
        }

        // Check if existing child exited unexpectedly.
        if let Some(proc) = worker.process.as_mut()
            && !matches!(proc.child.try_wait(), Ok(None))
            && let Some(dead) = worker.process.take()
        {
            dead.terminate_and_reap().await;
        }

        if worker.process.is_none() {
            match self.spawn_and_init(workspace_root, &cfg).await {
                Ok(proc) => {
                    worker.process = Some(proc);
                    worker.backoff_until = None;
                    worker.warned_unavailable = false;
                }
                Err(msg) => {
                    worker.backoff_until = Some(Instant::now() + UNSUPPORTED_BACKOFF);
                    if !worker.warned_unavailable {
                        worker.warned_unavailable = true;
                        tracing::warn!(
                            command = %cfg.command,
                            workspace = %workspace_root.display(),
                            error = %telemetry::redact_text(&msg),
                            "trace bridge unavailable; degrading gracefully to canonical shared memory"
                        );
                    }
                    metrics::record_trace_query(tool_name, "unavailable");
                    return Err(TraceCallErr::Unavailable);
                }
            }
        }

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let req = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            }
        });
        let timeout = Duration::from_millis(cfg.query_timeout_ms.max(1));
        let Some(proc) = worker.process.as_mut() else {
            metrics::record_trace_query(tool_name, "unavailable");
            return Err(TraceCallErr::Unavailable);
        };

        match rpc_round_trip(proc, id, &req, timeout).await {
            Ok(Ok(result)) => {
                metrics::record_trace_query(tool_name, "ok");
                Ok(result)
            }
            Ok(Err(rpc_err)) => {
                metrics::record_trace_query(tool_name, "error");
                tracing::debug!(tool = tool_name, error = %telemetry::redact_text(&rpc_err), "trace tool returned JSON-RPC error");
                Err(TraceCallErr::ToolError)
            }
            Err(io_or_timeout) => {
                let status = if io_or_timeout == "timeout" {
                    "timeout"
                } else {
                    "error"
                };
                metrics::record_trace_query(tool_name, status);
                if let Some(dead) = worker.process.take() {
                    dead.terminate_and_reap().await;
                }
                worker.backoff_until = Some(Instant::now() + UNSUPPORTED_BACKOFF);
                Err(TraceCallErr::Unavailable)
            }
        }
    }

    async fn spawn_and_init(
        &self,
        workspace_root: &Path,
        cfg: &TraceBridgeConfig,
    ) -> std::result::Result<TraceProcess, String> {
        let root_str = workspace_root.display().to_string();
        let mut cmd = Command::new(&cfg.command);
        for arg in &cfg.args {
            cmd.arg(arg.replace("{root}", &root_str));
        }
        cmd.env("VOUCH_WORKSPACE_ROOT", &root_str)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        if workspace_root.is_dir() {
            cmd.current_dir(workspace_root);
        }
        #[cfg(unix)]
        cmd.process_group(0);

        let mut child = cmd
            .spawn()
            .map_err(|error| format!("cannot spawn {:?}: {error}", cfg.command))?;
        let pid = child.id().ok_or_else(|| "missing pid".to_string())?;
        let guard = ProcessGroupGuard::new(pid, "trace-bridge");
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "missing stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "missing stdout".to_string())?;
        let mut proc = TraceProcess {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            guard,
        };

        let init_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let init_req = json!({
            "jsonrpc": "2.0",
            "id": init_id,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": { "name": "vouch", "version": crate::VERSION },
            }
        });
        let timeout = Duration::from_millis(cfg.startup_timeout_ms.max(1));
        match rpc_round_trip(&mut proc, init_id, &init_req, timeout).await {
            Ok(Ok(_)) => {}
            Ok(Err(err)) => {
                proc.terminate_and_reap().await;
                return Err(format!("trace initialize error: {err}"));
            }
            Err(err) => {
                proc.terminate_and_reap().await;
                return Err(format!("trace initialize failed: {err}"));
            }
        }

        let notif = json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        });
        let mut line = notif.to_string();
        line.push('\n');
        if let Err(err) = tokio::time::timeout(timeout, async {
            proc.stdin.write_all(line.as_bytes()).await?;
            proc.stdin.flush().await
        })
        .await
        {
            proc.terminate_and_reap().await;
            return Err(format!("trace initialized notify timed out: {err}"));
        }
        Ok(proc)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TraceCallErr {
    Unavailable,
    ToolError,
}

async fn rpc_round_trip(
    proc: &mut TraceProcess,
    id: u64,
    payload: &Value,
    timeout: Duration,
) -> std::result::Result<std::result::Result<Value, String>, &'static str> {
    let mut encoded = payload.to_string();
    encoded.push('\n');
    let io_res = tokio::time::timeout(timeout, async {
        proc.stdin
            .write_all(encoded.as_bytes())
            .await
            .map_err(|_| "io")?;
        proc.stdin.flush().await.map_err(|_| "io")?;

        let mut line = String::new();
        loop {
            line.clear();
            let n = proc.stdout.read_line(&mut line).await.map_err(|_| "io")?;
            if n == 0 {
                return Err("eof");
            }
            if line.len() > MAX_RPC_LINE_BYTES {
                return Err("line_limit");
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let Ok(msg) = serde_json::from_str::<Value>(trimmed) else {
                continue;
            };
            if msg.get("id").and_then(Value::as_u64) != Some(id) {
                continue;
            }
            if let Some(error) = msg.get("error") {
                let err_msg = error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown trace error")
                    .to_string();
                return Ok(Err(err_msg));
            }
            let result = msg.get("result").cloned().unwrap_or_else(|| json!({}));
            return Ok(Ok(result));
        }
    })
    .await;

    match io_res {
        Ok(inner) => inner,
        Err(_) => Err("timeout"),
    }
}

fn effective_base_root(cfg: &TraceBridgeConfig, ws: &WorkspaceConfig) -> PathBuf {
    cfg.workspace_root
        .clone()
        .or_else(|| ws.repo_path.clone())
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

fn normalize_root_key(root: &Path) -> PathBuf {
    std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf())
}

/// Resolve and security-check an optional per-request `X-Workspace-Root` override header.
///
/// The override must not contain traversal segments and must stay within the configured
/// allowed root (`trace.workspace_root` -> `workspace.repo_path` -> current directory).
pub fn resolve_workspace_root_header(
    trace_cfg: &TraceBridgeConfig,
    ws_cfg: &WorkspaceConfig,
    header_val: Option<&str>,
) -> Result<Option<PathBuf>> {
    let Some(raw) = header_val.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    if raw.contains('\0') {
        return Err(GatewayError::Forbidden(
            "invalid X-Workspace-Root header".into(),
        ));
    }
    let req_path = Path::new(raw);
    if req_path
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(GatewayError::Forbidden(format!(
            "X-Workspace-Root {raw:?} must not contain parent traversal segments"
        )));
    }
    let base_root = effective_base_root(trace_cfg, ws_cfg);
    let canon_base = std::fs::canonicalize(&base_root).map_err(|error| {
        GatewayError::Forbidden(format!(
            "cannot resolve base workspace root {}: {error}",
            base_root.display()
        ))
    })?;
    let candidate = if req_path.is_absolute() {
        req_path.to_path_buf()
    } else {
        canon_base.join(req_path)
    };
    let canon_req = std::fs::canonicalize(&candidate).map_err(|error| {
        GatewayError::Forbidden(format!("cannot resolve X-Workspace-Root {raw:?}: {error}"))
    })?;
    if !canon_req.starts_with(&canon_base) {
        return Err(GatewayError::Forbidden(format!(
            "X-Workspace-Root {raw:?} escapes the configured workspace root"
        )));
    }
    Ok(Some(canon_req))
}

fn verify_existing_files(
    workspace_root: &Path,
    touched_files: &[String],
    prompt_text: &str,
    limit: usize,
) -> Vec<String> {
    let Ok(canon_root) = std::fs::canonicalize(workspace_root) else {
        return Vec::new();
    };
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();

    let mut consider = |raw: &str| {
        if candidates.len() >= limit {
            return;
        }
        let Some(rel) = sanitize_relative_repo_path(raw, Some(workspace_root)) else {
            return;
        };
        let joined = canon_root.join(&rel);
        let Ok(canon_file) = std::fs::canonicalize(&joined) else {
            return;
        };
        if canon_file.starts_with(&canon_root) && canon_file.is_file() && seen.insert(rel.clone()) {
            candidates.push(rel);
        }
    };

    for f in touched_files {
        consider(f);
    }
    for extracted in extract_sanitized_repo_paths(prompt_text, Some(workspace_root)) {
        consider(&extracted);
    }
    candidates
}

fn format_mcp_tool_text(result: &Value) -> String {
    if let Some(items) = result.get("content").and_then(Value::as_array) {
        let joined: Vec<&str> = items
            .iter()
            .filter_map(|item| item.get("text").and_then(Value::as_str))
            .collect();
        if !joined.is_empty() {
            return telemetry::redact_text(&joined.join("\n"));
        }
    }
    if let Some(sc) = result.get("structuredContent") {
        return telemetry::redact_text(&sc.to_string());
    }
    if result.is_null() {
        return String::new();
    }
    let rendered = result.to_string();
    if rendered == "{}" {
        String::new()
    } else {
        telemetry::redact_text(&rendered)
    }
}

fn indent_block(text: &str, spaces: usize) -> String {
    let pad = " ".repeat(spaces);
    text.lines()
        .map(|l| format!("{pad}{l}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn compact_one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn build_decision_query(files: &[String], prompt_text: &str) -> String {
    let mut terms = Vec::new();
    for f in files.iter().take(3) {
        if let Some(stem) = Path::new(f).file_stem().and_then(|s| s.to_str()) {
            terms.push(stem.to_string());
        }
    }
    if terms.is_empty() {
        for word in prompt_text
            .split_whitespace()
            .filter(|w| w.len() >= 4)
            .take(4)
        {
            let clean: String = word
                .chars()
                .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !clean.is_empty() {
                terms.push(clean);
            }
        }
    }
    terms.join(" ")
}

/// Deterministically extract up to 5 likely function/struct symbol identifiers from
/// `get_symbol_outline` output or the prompt for `find_callers` lookup.
pub fn extract_candidate_symbols(outline_text: &str, prompt_text: &str) -> Vec<String> {
    let mut syms = Vec::new();
    let mut seen = HashSet::new();

    for line in outline_text.lines() {
        let trimmed = line.trim();
        for keyword in [
            "fn ",
            "struct ",
            "enum ",
            "trait ",
            "function ",
            "class ",
            "def ",
        ] {
            if let Some(rest) = trimmed.split(keyword).nth(1) {
                let ident: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                if ident.len() >= 3 && seen.insert(ident.clone()) {
                    syms.push(ident);
                    if syms.len() >= 5 {
                        return syms;
                    }
                }
            }
        }
    }

    for token in prompt_text.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
        let looks_ident = (token.contains('_') && token.len() >= 5)
            || (token.chars().any(char::is_uppercase)
                && token.chars().any(char::is_lowercase)
                && token.len() >= 4);
        if looks_ident && seen.insert(token.to_string()) {
            syms.push(token.to_string());
            if syms.len() >= 5 {
                break;
            }
        }
    }
    syms
}

fn assemble_budgeted_sections(
    model: &str,
    sections: &[(String, String)],
    max_bpe_tokens: usize,
) -> String {
    let mut out = String::new();
    let mut used = 0u64;
    let budget = max_bpe_tokens as u64;

    for (title, body) in sections {
        if used >= budget {
            break;
        }
        let section = format!("[{title}]\n{body}");
        let section_tokens = count_tokens(model, &section);
        let remaining = budget.saturating_sub(used);
        if section_tokens <= remaining {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&section);
            used = count_tokens(model, &out);
        } else {
            let trimmed = truncate_to_bpe_budget(model, &section, remaining as usize);
            if !trimmed.is_empty() {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(&trimmed);
            }
            break;
        }
    }
    out
}

fn count_items_in_result(val: &Value, fallback_text: &str) -> usize {
    if let Some(arr) = val
        .pointer("/structuredContent/rules")
        .or_else(|| val.pointer("/rules"))
        .and_then(Value::as_array)
    {
        return arr.len();
    }
    if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(fallback_text.trim()) {
        return arr.len();
    }
    if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(fallback_text.trim())
        && let Some(arr) = map.get("rules").and_then(Value::as_array)
    {
        return arr.len();
    }
    fallback_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .count()
}
