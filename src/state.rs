//! Shared application state.
//!
//! [`AppState`] owns the hot-reloadable configuration and every long-lived
//! subsystem derived from it. Handlers receive an `Arc<AppState>`; a reload
//! swaps the router table, token policies, workspace settings, and redaction
//! list without dropping in-flight requests, which keep using the snapshot
//! they started with.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex as ParkingMutex;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::{
    auth::TokenRegistry,
    cache::ResponseCache,
    cli::mcp::McpTaskStore,
    config::{ConfigStore, GatewayConfig},
    driver::{Usage, workspace::WorkspaceManager},
    error::{GatewayError, Result},
    ingress::responses::ResponseStore,
    memory::{
        SessionLedgerStore, SharedMemoryEngine, TraceBridge, ledger::format_trace_session_summary,
    },
    metrics::Metrics,
    persistence::{GatewaySnapshot, load_snapshot, save_snapshot},
    recorder::FlightRecorder,
    router::{ResolvedRoute, Router, RoutingInput, SessionAffinity},
    security::EgressGuard,
    telemetry::{self, OtelExporter, TraceRingBuffer},
    usage::{UsageTracker, calculate_cost_usd_with_profile},
};

/// Process-wide gateway state.
pub struct AppState {
    config: Arc<ConfigStore>,
    router: Arc<Router>,
    tokens: Arc<TokenRegistry>,
    workspaces: Arc<WorkspaceManager>,
    responses: Arc<ResponseStore>,
    cache: Arc<ResponseCache>,
    mcp_tasks: Arc<McpTaskStore>,
    metrics: Metrics,
    usage: Arc<UsageTracker>,
    otel: Arc<OtelExporter>,
    egress_guard: Arc<EgressGuard>,
    flight_recorder: Arc<FlightRecorder>,
    traces: Arc<TraceRingBuffer>,
    memory: Arc<SharedMemoryEngine>,
    shutdown_token: CancellationToken,
    reload_lock: tokio::sync::Mutex<()>,
    started_at: Instant,
    background: ParkingMutex<Vec<JoinHandle<()>>>,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("config", &self.config.path())
            .field("models", &self.router.models())
            .field("tokens_enforced", &self.tokens.enabled())
            .field("workspace_enabled", &self.workspaces.enabled())
            .finish()
    }
}

impl AppState {
    /// Build state from an already validated configuration.
    ///
    /// The configuration path is retained so that `SIGHUP` and file watchers
    /// can reload from disk.
    pub fn new(path: PathBuf, config: GatewayConfig) -> Result<Arc<Self>> {
        telemetry::set_global_secrets(config.secrets());
        let metrics = Metrics::install()?;
        let otel = OtelExporter::new(&config.telemetry);
        let usage = Arc::new(UsageTracker::new());
        let max_entries = config.defaults.max_session_entries.max(1);
        let ttl = Duration::from_secs(config.defaults.session_ttl_secs.max(1));

        let sessions = Arc::new(SessionAffinity::with_max_entries(max_entries));
        let router = Arc::new(Router::new(&config, sessions.clone())?);
        router.set_otel(otel.clone());
        let tokens = Arc::new(TokenRegistry::new(&config.tokens)?);
        let workspaces = Arc::new(WorkspaceManager::new(config.workspace.clone()));
        let responses = Arc::new(ResponseStore::with_max_entries(max_entries));
        let cache = Arc::new(ResponseCache::new(config.cache.clone()));
        let mcp_tasks = Arc::new(McpTaskStore::new());
        let egress_guard = Arc::new(EgressGuard::new(&config.security.egress_guard)?);
        let flight_recorder = FlightRecorder::new(
            config.flight_recorder.clone(),
            config.defaults.state_dir.clone(),
        );
        let traces = Arc::new(TraceRingBuffer::new(100));
        let shared_ledger = Arc::new(SessionLedgerStore::new(
            max_entries,
            config.shared_memory.max_turns_per_session,
        ));
        let trace_bridge = Arc::new(TraceBridge::new(
            config.trace.clone(),
            config.workspace.clone(),
        ));
        let memory = Arc::new(SharedMemoryEngine::new(
            config.shared_memory.clone(),
            shared_ledger,
            trace_bridge,
        ));
        router.set_memory(memory.clone());

        if let Some(state_dir) = config.defaults.state_dir.as_deref()
            && let Some(snapshot) = load_snapshot(state_dir)
        {
            sessions.restore_snapshot(&snapshot.session_affinity, &snapshot.backend_sessions, ttl);
            responses.restore_snapshot(&snapshot.responses, ttl);
            tokens.restore_daily_state(&snapshot.daily_token_budgets);
            usage.restore(&snapshot.usage);
            cache.restore_snapshot(&snapshot.cached_responses);
            memory.ledger().restore_snapshot(
                &snapshot.shared_sessions,
                snapshot.total_handoffs,
                ttl,
            );
            tracing::info!(
                dir = %state_dir.display(),
                sessions = sessions.len(),
                responses = responses.len(),
                cached = cache.len(),
                shared_sessions = memory.ledger().len(),
                "restored persistent state snapshot"
            );
        }

        let config = Arc::new(ConfigStore::new(path, config));

        Ok(Arc::new(Self {
            config,
            router,
            tokens,
            workspaces,
            responses,
            cache,
            mcp_tasks,
            metrics,
            usage,
            otel,
            egress_guard,
            flight_recorder,
            traces,
            memory,
            shutdown_token: CancellationToken::new(),
            reload_lock: tokio::sync::Mutex::new(()),
            started_at: Instant::now(),
            background: ParkingMutex::new(Vec::new()),
        }))
    }

    /// Cumulative usage and latency analytics tracker.
    pub fn usage(&self) -> &Arc<UsageTracker> {
        &self.usage
    }

    /// Deterministic response cache.
    pub fn cache(&self) -> &Arc<ResponseCache> {
        &self.cache
    }

    /// Calculate USD cost for `usage` using `[pricing]` overrides.
    pub fn calculate_cost_usd(&self, route: &str, backend_id: &str, usage: Usage) -> f64 {
        calculate_cost_usd_with_profile(&self.config.load().pricing, route, backend_id, None, usage)
    }

    /// Profile-aware USD pricing calculation (`backend:profile` -> `backend` -> `route`).
    pub fn calculate_cost_usd_for_profile(
        &self,
        route: &str,
        backend_id: &str,
        profile_id: Option<&str>,
        usage: Usage,
    ) -> f64 {
        calculate_cost_usd_with_profile(
            &self.config.load().pricing,
            route,
            backend_id,
            profile_id,
            usage,
        )
    }

    /// Non-blocking OpenTelemetry trace span exporter.
    pub fn otel(&self) -> &Arc<OtelExporter> {
        &self.otel
    }

    /// Outbound secret/PII egress guardrail engine.
    pub fn egress_guard(&self) -> &Arc<EgressGuard> {
        &self.egress_guard
    }

    /// Opt-in redacted JSONL flight recorder.
    pub fn flight_recorder(&self) -> &Arc<FlightRecorder> {
        &self.flight_recorder
    }

    /// Bounded ring buffer of recent completed request traces (`GET /v1/traces`).
    pub fn traces(&self) -> &Arc<TraceRingBuffer> {
        &self.traces
    }

    /// Cross-agent shared memory ledger and model-free `trace` bridge.
    pub fn memory(&self) -> &Arc<SharedMemoryEngine> {
        &self.memory
    }

    /// Stateful `/v1/responses` chaining store.
    pub fn responses(&self) -> &Arc<ResponseStore> {
        &self.responses
    }

    /// Async MCP `tasks/*` execution store.
    pub fn mcp_tasks(&self) -> &Arc<McpTaskStore> {
        &self.mcp_tasks
    }

    /// Path of the active configuration file.
    pub fn config_path(&self) -> &Path {
        self.config.path()
    }

    /// Current immutable configuration snapshot.
    pub fn config(&self) -> Arc<GatewayConfig> {
        self.config.load()
    }

    /// Model router.
    pub fn router(&self) -> &Arc<Router> {
        &self.router
    }

    /// Virtual-token registry.
    pub fn tokens(&self) -> &Arc<TokenRegistry> {
        &self.tokens
    }

    /// Worktree manager.
    pub fn workspaces(&self) -> &Arc<WorkspaceManager> {
        &self.workspaces
    }

    /// Prometheus renderer.
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    /// Cancellation token shared by listeners and background tasks.
    pub fn shutdown_token(&self) -> CancellationToken {
        self.shutdown_token.clone()
    }

    /// Seconds since the process started serving.
    pub fn uptime_secs(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }

    /// Resolve an optional client-supplied model against the configured
    /// default and `[aliases]`. Empty strings are treated as absent.
    pub fn resolve_model(&self, requested: Option<&str>) -> Result<String> {
        if let Some(model) = requested.map(str::trim).filter(|model| !model.is_empty()) {
            let cfg = self.config();
            return Ok(cfg.resolve_route_name(model).unwrap_or(model).to_string());
        }
        self.router.default_model().ok_or_else(|| {
            GatewayError::BadRequest(
                "model is required because no default model is configured".into(),
            )
        })
    }

    /// Evaluate `[[routing_rules]]`, `[routes]`, and `[aliases]` using full
    /// request traits and HTTP headers.
    pub fn resolve_route(&self, input: &RoutingInput<'_>) -> Result<ResolvedRoute> {
        self.router.resolve_route(input)
    }

    /// Re-read the configuration file and atomically install every derived
    /// component.
    ///
    /// The token registry is replaced first because it can fail while
    /// compiling globs; the router is rebuilt next because driver construction
    /// can fail. If either step fails, the previous configuration remains
    /// active and the previous token set is restored.
    pub async fn reload(&self) -> Result<Arc<GatewayConfig>> {
        let _guard = self.reload_lock.lock().await;
        let next = GatewayConfig::load(self.config.path())?;
        let previous = self.config.load();

        self.tokens.reload(&next.tokens)?;
        if let Err(error) = self.egress_guard.update(&next.security.egress_guard) {
            let _ = self.tokens.reload(&previous.tokens);
            return Err(error);
        }
        if let Err(error) = self.router.rebuild(&next) {
            let _ = self.egress_guard.update(&previous.security.egress_guard);
            let _ = self.tokens.reload(&previous.tokens);
            return Err(error);
        }

        self.responses
            .set_max_entries(next.defaults.max_session_entries);
        self.cache.update_config(next.cache.clone());
        self.workspaces.update_config(next.workspace.clone());
        self.flight_recorder.update_config(
            next.flight_recorder.clone(),
            next.defaults.state_dir.clone(),
        );
        self.otel.reconfigure(&next.telemetry);
        self.memory.update_config(
            next.shared_memory.clone(),
            next.trace.clone(),
            next.workspace.clone(),
            next.defaults.max_session_entries,
        );
        telemetry::set_global_secrets(next.secrets());

        if next.server.tcp_listen != previous.server.tcp_listen {
            tracing::warn!(
                previous = %previous.server.tcp_listen,
                next = %next.server.tcp_listen,
                "server.tcp_listen changed; restart vouch to rebind the TCP listener"
            );
        }
        if next.server.unix_socket != previous.server.unix_socket {
            tracing::warn!(
                previous = ?previous.server.unix_socket,
                next = ?next.server.unix_socket,
                "server.unix_socket changed; restart vouch to rebind the Unix listener"
            );
        }

        let next = Arc::new(next);
        self.config.store_snapshot(next.clone());
        tracing::info!(
            path = %self.config.path().display(),
            models = next.routes.len(),
            backends = next.backends.len(),
            tokens_enforced = !next.tokens.is_empty(),
            "configuration reloaded"
        );
        Ok(next)
    }

    /// Start session, response, task, worktree, persistence, and OTLP tasks. Safe to call once.
    pub fn start_background_tasks(self: &Arc<Self>) {
        let mut background = self.background.lock();
        if !background.is_empty() {
            return;
        }
        let sessions = self.router.sessions().clone();
        background.push(
            sessions.spawn_eviction_task(self.router.ttl_handle(), self.shutdown_token.clone()),
        );
        background.push(
            self.responses
                .spawn_eviction_task(self.router.ttl_handle(), self.shutdown_token.clone()),
        );
        background.push(
            self.mcp_tasks
                .spawn_eviction_task(self.router.ttl_handle(), self.shutdown_token.clone()),
        );
        background.push(
            self.workspaces
                .spawn_eviction_task(self.shutdown_token.clone()),
        );
        background.push(self.cache.spawn_eviction_task(self.shutdown_token.clone()));
        let memory_for_evict = self.memory.clone();
        background.push(self.memory.ledger().spawn_eviction_task(
            self.router.ttl_handle(),
            self.shutdown_token.clone(),
            move |evicted| {
                if !memory_for_evict.config().sync_to_trace {
                    return;
                }
                for entry in evicted {
                    let summary = format_trace_session_summary(&entry);
                    memory_for_evict.trace().record_session_checkpoint_async(
                        None,
                        summary,
                        entry.touched_files,
                    );
                }
            },
        ));
        if let Some(otel_handle) = self.otel.start_worker(self.shutdown_token.clone()) {
            background.push(otel_handle);
        }
        if let Some(rec_handle) = self
            .flight_recorder
            .start_worker(self.shutdown_token.clone())
        {
            background.push(rec_handle);
        }

        let state_for_persist = self.clone();
        let shutdown = self.shutdown_token.clone();
        background.push(tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(15));
            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        if let Err(error) = state_for_persist.persist_state() {
                            tracing::warn!(error = %error, "periodic state snapshot write failed");
                        }
                    }
                    _ = shutdown.cancelled() => break,
                }
            }
        }));
    }

    /// Persist sessions, `/v1/responses`, daily token budgets, and usage analytics
    /// to `defaults.state_dir` when configured (atomically with mode `0600`).
    pub fn persist_state(&self) -> Result<()> {
        let cfg = self.config();
        let Some(state_dir) = cfg.defaults.state_dir.as_deref() else {
            return Ok(());
        };
        let ttl = Duration::from_secs(cfg.defaults.session_ttl_secs.max(1));
        let (affinity, backend_sessions) = self.router.sessions().export_snapshot(ttl);
        let responses = self.responses.export_snapshot(ttl);
        let budgets = self.tokens.export_daily_state();
        let usage = self.usage.snapshot();
        let cached = self.cache.export_snapshot();
        let (shared_sessions, total_handoffs) = self.memory.ledger().export_snapshot(ttl);
        let snapshot = GatewaySnapshot::new(affinity, backend_sessions, responses, budgets, usage)
            .with_cached_responses(cached)
            .with_shared_sessions(shared_sessions, total_handoffs);
        save_snapshot(state_dir, &snapshot)
    }

    /// Cancel background work, wait briefly for tasks, persist state, and clean worktrees.
    pub async fn shutdown(&self) {
        self.flight_recorder.flush().await;
        if let Err(error) = self.persist_state() {
            tracing::warn!(error = %error, "failed to flush state snapshot before shutdown");
        }
        self.shutdown_token.cancel();
        self.mcp_tasks.cancel_all();
        let handles: Vec<JoinHandle<()>> = std::mem::take(&mut *self.background.lock());
        for handle in handles {
            let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
        }
        self.memory.trace().shutdown().await;
        if self.config.load().workspace.cleanup_on_shutdown {
            self.workspaces.cleanup_all().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(model: &str) -> GatewayConfig {
        GatewayConfig::from_toml_str(&format!(
            r#"
[server]
tcp_listen = "127.0.0.1:0"

[[backends]]
id = "mock"
kind = "cli"
command = "/bin/echo"
stdin = true

[routes]
"{model}" = ["mock"]
"#
        ))
        .expect("config")
    }

    #[test]
    fn resolves_default_and_explicit_models() {
        let temp = tempfile::tempdir().expect("temp");
        let path = temp.path().join("vouch.toml");
        std::fs::write(&path, "").expect("write");
        let state = AppState::new(path, config("coding/default")).expect("state");
        assert_eq!(
            state.resolve_model(None).expect("default"),
            "coding/default"
        );
        assert_eq!(
            state
                .resolve_model(Some("explicit/model"))
                .expect("explicit"),
            "explicit/model"
        );
        assert!(state.resolve_model(Some("  ")).is_ok());
    }

    #[test]
    fn exposes_readiness_and_models() {
        let temp = tempfile::tempdir().expect("temp");
        let path = temp.path().join("vouch.toml");
        std::fs::write(&path, "").expect("write");
        let state = AppState::new(path, config("coding/default")).expect("state");
        assert_eq!(state.router().models(), vec!["coding/default".to_string()]);
        assert!(state.router().readiness().ready);
        assert_eq!(state.uptime_secs(), 0);
    }
}
