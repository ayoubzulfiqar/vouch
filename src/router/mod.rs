//! Model routing, ordered fallback, circuit breaking, and sticky affinity.
//!
//! A public model maps to an ordered chain of backend handles. The router
//! tries the sticky backend first when the session already has affinity, then
//! walks the chain. A backend is skipped when its circuit is open or when its
//! concurrency permit cannot be acquired; a failure before meaningful stream
//! output begins transparently fails over to the next candidate.

pub mod circuit_breaker;

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use arc_swap::ArcSwap;
use axum::http::HeaderMap;
use dashmap::DashMap;
use futures::StreamExt;
use globset::{Glob, GlobMatcher};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::{
    config::{BackendConfig, BackendKind, GatewayConfig},
    driver::{
        ChatMessage, ChunkStream, Driver, DriverResponse, ExecutionRequest, WorkspaceMode,
        cli_pipe::CliPipeDriver, http_proxy::HttpProxyDriver, prime_stream,
        pty_runner::PtyRunnerDriver, workspace::WorkspaceManager,
    },
    error::{GatewayError, Result},
    metrics,
    persistence::{
        PersistedAffinityEntry, PersistedBackendSession, evict_oldest_if_full, instant_from_unix,
        unix_now_secs,
    },
    telemetry::{ActiveSpan, OtelExporter, SpanKind, TraceContext},
};

use self::circuit_breaker::{BreakerSettings, CircuitBreaker, CircuitState, FailureDetails};

/// Sticky backend selected for a session.
#[derive(Debug, Clone)]
struct AffinityEntry {
    backend_id: String,
    last_used: Instant,
    updated_unix_secs: u64,
    seq: u64,
}

/// Native multi-turn session state tracked per `(session_key, backend_id)`.
#[derive(Debug, Clone)]
pub struct BackendSessionState {
    /// Captured native CLI session handle (when `session_id_regex` matched).
    pub backend_session_id: Option<String>,
    /// True after turn 1 completed successfully on this backend.
    pub has_history: bool,
    last_used: Instant,
    updated_unix_secs: u64,
    seq: u64,
}

/// Thread-safe sticky-session map with TTL eviction, bounded capacity, and
/// native backend session handles for delta continuation.
#[derive(Debug)]
pub struct SessionAffinity {
    entries: DashMap<String, AffinityEntry>,
    backend_sessions: DashMap<(String, String), BackendSessionState>,
    max_entries: AtomicUsize,
    seq: AtomicU64,
}

impl Default for SessionAffinity {
    fn default() -> Self {
        Self::with_max_entries(10_000)
    }
}

impl SessionAffinity {
    /// Create an empty map with the default capacity (`10_000`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty map bounded to `max_entries` entries.
    pub fn with_max_entries(max_entries: usize) -> Self {
        Self {
            entries: DashMap::new(),
            backend_sessions: DashMap::new(),
            max_entries: AtomicUsize::new(max_entries.max(1)),
            seq: AtomicU64::new(1),
        }
    }

    /// Update the configured capacity ceiling (evicts oldest entries if shrinking).
    pub fn set_max_entries(&self, max_entries: usize) {
        let cap = max_entries.max(1);
        self.max_entries.store(cap, Ordering::Relaxed);
        if self.entries.len() > cap {
            evict_oldest_if_full(&self.entries, cap.saturating_add(1), |e| e.seq);
        }
        if self.backend_sessions.len() > cap {
            evict_oldest_if_full(&self.backend_sessions, cap.saturating_add(1), |e| e.seq);
        }
    }

    fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::Relaxed)
    }

    /// Look up a live affinity entry.
    pub fn get(&self, key: &str, ttl: Duration) -> Option<String> {
        let entry = self.entries.get(key)?;
        if entry.last_used.elapsed() > ttl {
            drop(entry);
            self.entries.remove(key);
            None
        } else {
            Some(entry.backend_id.clone())
        }
    }

    /// Record a successful backend for a session, evicting the oldest entry if full.
    pub fn set(&self, key: &str, backend_id: &str) {
        let cap = self.max_entries.load(Ordering::Relaxed).max(1);
        if !self.entries.contains_key(key) {
            evict_oldest_if_full(&self.entries, cap, |e| e.seq);
        }
        let seq = self.next_seq();
        self.entries.insert(
            key.to_string(),
            AffinityEntry {
                backend_id: backend_id.to_string(),
                last_used: Instant::now(),
                updated_unix_secs: unix_now_secs(),
                seq,
            },
        );
    }

    /// Look up native CLI session handle/history for `(session_key, backend_id)`.
    pub fn get_backend_session(
        &self,
        session_key: &str,
        backend_id: &str,
        ttl: Duration,
    ) -> Option<BackendSessionState> {
        let composite = (session_key.to_string(), backend_id.to_string());
        let entry = self.backend_sessions.get(&composite)?;
        if entry.last_used.elapsed() > ttl {
            drop(entry);
            self.backend_sessions.remove(&composite);
            None
        } else {
            Some(entry.clone())
        }
    }

    /// Persist or refresh native CLI session handle after a successful turn.
    pub fn record_backend_session(
        &self,
        session_key: &str,
        backend_id: &str,
        captured_id: Option<String>,
    ) {
        let composite = (session_key.to_string(), backend_id.to_string());
        let cap = self.max_entries.load(Ordering::Relaxed).max(1);
        if !self.backend_sessions.contains_key(&composite) {
            evict_oldest_if_full(&self.backend_sessions, cap, |e| e.seq);
        }
        let seq = self.next_seq();
        let now_unix = unix_now_secs();
        let mut entry =
            self.backend_sessions
                .entry(composite)
                .or_insert_with(|| BackendSessionState {
                    backend_session_id: None,
                    has_history: false,
                    last_used: Instant::now(),
                    updated_unix_secs: now_unix,
                    seq,
                });
        entry.has_history = true;
        entry.last_used = Instant::now();
        entry.updated_unix_secs = now_unix;
        entry.seq = seq;
        if let Some(id) = captured_id {
            entry.backend_session_id = Some(id);
        }
    }

    /// Snapshot live affinity & backend continuation state for persistence.
    pub fn export_snapshot(
        &self,
        ttl: Duration,
    ) -> (Vec<PersistedAffinityEntry>, Vec<PersistedBackendSession>) {
        self.evict_expired(ttl);
        let affinity = self
            .entries
            .iter()
            .map(|entry| PersistedAffinityEntry {
                key: entry.key().clone(),
                backend_id: entry.backend_id.clone(),
                updated_unix_secs: entry.updated_unix_secs,
            })
            .collect();
        let sessions = self
            .backend_sessions
            .iter()
            .map(|entry| PersistedBackendSession {
                session_key: entry.key().0.clone(),
                backend_id: entry.key().1.clone(),
                backend_session_id: entry.backend_session_id.clone(),
                has_history: entry.has_history,
                updated_unix_secs: entry.updated_unix_secs,
            })
            .collect();
        (affinity, sessions)
    }

    /// Restore non-expired affinity & backend continuation entries from disk.
    pub fn restore_snapshot(
        &self,
        affinity: &[PersistedAffinityEntry],
        sessions: &[PersistedBackendSession],
        ttl: Duration,
    ) {
        let now_unix = unix_now_secs();
        let cap = self.max_entries.load(Ordering::Relaxed).max(1);
        for item in affinity {
            let Some(last_used) = instant_from_unix(item.updated_unix_secs, now_unix, ttl) else {
                continue;
            };
            if !self.entries.contains_key(&item.key) {
                evict_oldest_if_full(&self.entries, cap, |e| e.seq);
            }
            let seq = self.next_seq();
            self.entries.insert(
                item.key.clone(),
                AffinityEntry {
                    backend_id: item.backend_id.clone(),
                    last_used,
                    updated_unix_secs: item.updated_unix_secs,
                    seq,
                },
            );
        }
        for item in sessions {
            let Some(last_used) = instant_from_unix(item.updated_unix_secs, now_unix, ttl) else {
                continue;
            };
            let composite = (item.session_key.clone(), item.backend_id.clone());
            if !self.backend_sessions.contains_key(&composite) {
                evict_oldest_if_full(&self.backend_sessions, cap, |e| e.seq);
            }
            let seq = self.next_seq();
            self.backend_sessions.insert(
                composite,
                BackendSessionState {
                    backend_session_id: item.backend_session_id.clone(),
                    has_history: item.has_history,
                    last_used,
                    updated_unix_secs: item.updated_unix_secs,
                    seq,
                },
            );
        }
    }

    /// Remove expired entries.
    pub fn evict_expired(&self, ttl: Duration) {
        let expired: Vec<String> = self
            .entries
            .iter()
            .filter(|entry| entry.last_used.elapsed() > ttl)
            .map(|entry| entry.key().clone())
            .collect();
        for key in expired {
            self.entries.remove(&key);
        }

        let expired_bs: Vec<(String, String)> = self
            .backend_sessions
            .iter()
            .filter(|entry| entry.last_used.elapsed() > ttl)
            .map(|entry| entry.key().clone())
            .collect();
        for key in expired_bs {
            self.backend_sessions.remove(&key);
        }
    }

    /// Number of tracked sessions.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no sessions are tracked.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Spawn a background eviction task.
    pub fn spawn_eviction_task(
        self: &Arc<Self>,
        ttl_secs: Arc<AtomicU64>,
        shutdown: CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
        let sessions = self.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(30));
            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        let ttl = Duration::from_secs(ttl_secs.load(Ordering::Relaxed).max(1));
                        sessions.evict_expired(ttl);
                    }
                    _ = shutdown.cancelled() => break,
                }
            }
        })
    }
}

/// Derive a stable session key.
///
/// An explicit `X-Session-ID` is preserved when it is reasonably short and
/// safe; otherwise it is hashed. Without a header, the SHA-256 prefix of the
/// system prompt plus the first user turn gives multi-turn orchestrators a
/// deterministic key.
pub fn derive_session_key(
    explicit: Option<&str>,
    system: Option<&str>,
    first_user: &str,
) -> String {
    if let Some(explicit) = explicit.map(str::trim).filter(|value| !value.is_empty()) {
        let safe = explicit.len() <= 128
            && explicit
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':'));
        if safe {
            return explicit.to_string();
        }
        return hash_key(explicit);
    }
    hash_key(&format!("{}\n{}", system.unwrap_or(""), first_user))
}

fn hash_key(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    hex::encode(&digest[..8])
}

/// Extract only messages appended after the last assistant turn for CLI delta
/// resumption.
pub fn slice_delta_messages(messages: &[ChatMessage]) -> Vec<ChatMessage> {
    match messages.iter().rposition(|m| m.role == "assistant") {
        Some(idx) if idx + 1 < messages.len() => messages[idx + 1..].to_vec(),
        Some(_) => Vec::new(),
        None => messages.to_vec(),
    }
}

/// One backend with its driver and breaker.
pub struct BackendHandle {
    /// Backend configuration snapshot.
    pub config: BackendConfig,
    /// Egress implementation.
    pub driver: Arc<dyn Driver>,
    /// Circuit breaker.
    pub breaker: Arc<CircuitBreaker>,
}

impl std::fmt::Debug for BackendHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackendHandle")
            .field("id", &self.config.id)
            .field("kind", &self.config.kind)
            .field("max_concurrency", &self.config.max_concurrency)
            .finish()
    }
}

#[derive(Debug, Clone)]
struct CompiledRoutingRule {
    label: String,
    matcher: GlobMatcher,
    has_tools: Option<bool>,
    has_images: Option<bool>,
    min_prompt_tokens: Option<u64>,
    headers: Vec<(String, String)>,
    target_route: String,
}

/// Request traits inspected when evaluating `[[routing_rules]]`.
#[derive(Debug, Clone)]
pub struct RoutingInput<'a> {
    /// Raw model name or alias sent by the caller (optional).
    pub requested_model: Option<&'a str>,
    /// True if tool/function definitions are present on the request.
    pub has_tools: bool,
    /// True if any turn contains image parts or URLs.
    pub has_images: bool,
    /// Estimated BPE prompt token count.
    pub prompt_tokens: u64,
    /// HTTP ingress headers checked against `header_equals`.
    pub headers: &'a HeaderMap,
}

/// Result of resolving requested model + routing rules + aliases.
#[derive(Debug, Clone)]
pub struct ResolvedRoute {
    /// Requested model/alias name (after default substitution if omitted).
    pub requested: String,
    /// Canonical route key under `[routes]`.
    pub route: String,
    /// Rule label if redirected by `[[routing_rules]]`.
    pub matched_rule: Option<String>,
}

/// Immutable routing snapshot.
#[derive(Debug, Clone, Default)]
pub struct RouterTable {
    routes: HashMap<String, Vec<Arc<BackendHandle>>>,
    aliases: BTreeMap<String, String>,
    rules: Vec<CompiledRoutingRule>,
    order: Vec<Arc<BackendHandle>>,
    default_model: Option<String>,
    default_requested: Option<String>,
}

/// Public backend health record.
#[derive(Debug, Clone, Serialize)]
pub struct BackendStatus {
    /// Backend id.
    pub id: String,
    /// Transport.
    pub kind: BackendKind,
    /// Concurrency limit.
    pub max_concurrency: usize,
    /// Circuit state.
    pub circuit: CircuitState,
    /// Consecutive failures.
    pub consecutive_failures: u32,
    /// Remaining cooldown when open.
    pub open_remaining_secs: u64,
    /// True when the breaker opened due to upstream auth expiry / rejection.
    pub auth_expired: bool,
}

/// One public model and its fallback chain.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    /// Public model name.
    pub id: String,
    /// Ordered backend health records.
    pub backends: Vec<BackendStatus>,
}

/// Readiness summary.
#[derive(Debug, Clone, Serialize)]
pub struct ReadinessReport {
    /// True when at least one backend circuit is not open.
    pub ready: bool,
    /// Per-backend status.
    pub backends: Vec<BackendStatus>,
}

/// Result of a routed execution.
pub struct RoutedResponse {
    /// Backend that accepted the request.
    pub backend_id: String,
    /// True when an earlier candidate was skipped or failed.
    pub fallback_used: bool,
    /// Driver response.
    pub response: DriverResponse,
    /// Human-readable attempt log.
    pub attempts: Vec<String>,
}

/// Model router.
#[derive(Debug)]
pub struct Router {
    table: ArcSwap<RouterTable>,
    sessions: Arc<SessionAffinity>,
    breakers: DashMap<String, Arc<CircuitBreaker>>,
    ttl_secs: Arc<AtomicU64>,
    otel: ArcSwap<Option<Arc<OtelExporter>>>,
}

impl Router {
    /// Build a router from validated configuration.
    pub fn new(config: &GatewayConfig, sessions: Arc<SessionAffinity>) -> Result<Self> {
        let router = Self {
            table: ArcSwap::from_pointee(RouterTable::default()),
            sessions,
            breakers: DashMap::new(),
            ttl_secs: Arc::new(AtomicU64::new(config.defaults.session_ttl_secs.max(1))),
            otel: ArcSwap::from_pointee(None),
        };
        router.rebuild(config)?;
        Ok(router)
    }

    /// Attach the OpenTelemetry exporter used to emit `router.dispatch`,
    /// `workspace.worktree`, and `driver.exec` child spans.
    pub fn set_otel(&self, otel: Arc<OtelExporter>) {
        self.otel.store(Arc::new(Some(otel)));
    }

    /// Atomically replace drivers and routes. Breakers are reused by backend
    /// id so that cooldown state survives a configuration reload.
    pub fn rebuild(&self, config: &GatewayConfig) -> Result<()> {
        let settings = BreakerSettings::from(&config.resilience);
        let mut handles: HashMap<String, Arc<BackendHandle>> = HashMap::new();
        let mut order = Vec::with_capacity(config.backends.len());
        let mut active = HashSet::with_capacity(config.backends.len());

        for backend in &config.backends {
            active.insert(backend.id.clone());
            let breaker = match self.breakers.get(&backend.id) {
                Some(existing) => {
                    existing.reconfigure(settings.clone());
                    existing.clone()
                }
                None => {
                    let breaker =
                        Arc::new(CircuitBreaker::new(backend.id.clone(), settings.clone()));
                    self.breakers.insert(backend.id.clone(), breaker.clone());
                    breaker
                }
            };
            let driver: Arc<dyn Driver> = match backend.kind {
                BackendKind::Http => {
                    Arc::new(HttpProxyDriver::new(backend.clone(), &config.defaults)?)
                }
                BackendKind::Cli => Arc::new(
                    CliPipeDriver::new(backend.clone(), &config.defaults)?
                        .with_attachments(config.attachments.clone()),
                ),
                BackendKind::Pty => Arc::new(
                    PtyRunnerDriver::new(backend.clone(), &config.defaults)?
                        .with_attachments(config.attachments.clone()),
                ),
            };
            let handle = Arc::new(BackendHandle {
                config: backend.clone(),
                driver,
                breaker,
            });
            handles.insert(backend.id.clone(), handle.clone());
            order.push(handle);
        }

        self.breakers.retain(|id, _| active.contains(id));

        let mut routes = HashMap::with_capacity(config.routes.len());
        for (model, chain) in &config.routes {
            let mut backends = Vec::with_capacity(chain.len());
            for id in chain {
                if let Some(handle) = handles.get(id) {
                    backends.push(handle.clone());
                }
            }
            routes.insert(model.clone(), backends);
        }

        let mut compiled_rules = Vec::with_capacity(config.routing_rules.len());
        for (idx, rule) in config.routing_rules.iter().enumerate() {
            let matcher = Glob::new(&rule.model_pattern)
                .map_err(|error| {
                    GatewayError::Config(format!(
                        "invalid routing rule glob {:?}: {error}",
                        rule.model_pattern
                    ))
                })?
                .compile_matcher();
            compiled_rules.push(CompiledRoutingRule {
                label: rule
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("rule-{idx}:{}", rule.target_route)),
                matcher,
                has_tools: rule.has_tools,
                has_images: rule.has_images,
                min_prompt_tokens: rule.min_prompt_tokens,
                headers: rule
                    .header_equals
                    .iter()
                    .map(|(k, v)| (k.to_ascii_lowercase(), v.clone()))
                    .collect(),
                target_route: rule.target_route.clone(),
            });
        }

        let default_requested = config.defaults.model.clone().or_else(|| {
            if config.routes.len() == 1 {
                config.routes.keys().next().cloned()
            } else {
                None
            }
        });

        self.sessions
            .set_max_entries(config.defaults.max_session_entries);
        self.ttl_secs
            .store(config.defaults.session_ttl_secs.max(1), Ordering::Relaxed);
        self.table.store(Arc::new(RouterTable {
            routes,
            aliases: config.aliases.clone(),
            rules: compiled_rules,
            order,
            default_model: config.default_model().map(ToString::to_string),
            default_requested,
        }));
        Ok(())
    }

    /// Resolve a requested model/alias and request traits against `[[routing_rules]]`,
    /// `[routes]`, and `[aliases]`.
    pub fn resolve_route(&self, input: &RoutingInput<'_>) -> Result<ResolvedRoute> {
        let table = self.table.load();
        let requested = match input
            .requested_model
            .map(str::trim)
            .filter(|model| !model.is_empty())
        {
            Some(model) => model.to_string(),
            None => table.default_requested.clone().ok_or_else(|| {
                GatewayError::BadRequest(
                    "model is required because no default model is configured".into(),
                )
            })?,
        };

        for rule in &table.rules {
            if !rule.matcher.is_match(&requested) {
                continue;
            }
            if let Some(req_tools) = rule.has_tools
                && req_tools != input.has_tools
            {
                continue;
            }
            if let Some(req_images) = rule.has_images
                && req_images != input.has_images
            {
                continue;
            }
            if let Some(min_tokens) = rule.min_prompt_tokens
                && input.prompt_tokens < min_tokens
            {
                continue;
            }
            let headers_match = rule.headers.iter().all(|(key, expected)| {
                input
                    .headers
                    .get(key.as_str())
                    .and_then(|value| value.to_str().ok())
                    .map(str::trim)
                    == Some(expected.as_str())
            });
            if !headers_match {
                continue;
            }
            return Ok(ResolvedRoute {
                requested,
                route: rule.target_route.clone(),
                matched_rule: Some(rule.label.clone()),
            });
        }

        let route = if table.routes.contains_key(&requested) {
            requested.clone()
        } else if let Some(target) = table.aliases.get(&requested) {
            target.clone()
        } else {
            requested.clone()
        };

        Ok(ResolvedRoute {
            requested,
            route,
            matched_rule: None,
        })
    }

    /// Execute a request with fallback and affinity.
    pub async fn execute(
        &self,
        request: ExecutionRequest,
        workspace_mode: WorkspaceMode,
        workspaces: &WorkspaceManager,
    ) -> Result<RoutedResponse> {
        let otel = self.otel.load_full().as_ref().clone();
        let parent_trace = request
            .trace_context
            .clone()
            .unwrap_or_else(|| TraceContext::from_headers(&HeaderMap::new()));
        let mut dispatch_span = otel
            .as_ref()
            .map(|o| o.start_span(parent_trace.child(), "router.dispatch", SpanKind::Internal));
        if let Some(span) = &mut dispatch_span {
            span.set_str("vouch.route", &request.model);
        }
        let dispatch_ctx = dispatch_span
            .as_ref()
            .map(|s| s.context().clone())
            .unwrap_or_else(|| parent_trace.child());

        let table = self.table.load_full();
        let Some(chain) = table.routes.get(&request.model).cloned() else {
            let err = GatewayError::NotFound(format!(
                "model {:?} is not configured; available models: {}",
                request.model,
                table.routes.keys().cloned().collect::<Vec<_>>().join(", ")
            ));
            if let Some(span) = dispatch_span {
                span.finish_err(&err.to_string());
            }
            return Err(err);
        };
        if chain.is_empty() {
            let err = GatewayError::NoHealthyBackend(format!(
                "route {:?} has no backends",
                request.model
            ));
            if let Some(span) = dispatch_span {
                span.finish_err(&err.to_string());
            }
            return Err(err);
        }

        let ttl = Duration::from_secs(self.ttl_secs.load(Ordering::Relaxed).max(1));
        let mut candidates = chain;
        if let Some(sticky) = self.sessions.get(&request.session_key, ttl)
            && let Some(position) = candidates
                .iter()
                .position(|backend| backend.config.id == sticky)
        {
            let backend = candidates.remove(position);
            candidates.insert(0, backend);
        }

        let mut attempts = Vec::new();
        let mut last_error: Option<GatewayError> = None;
        let mut previous_backend: Option<String> = None;

        for (index, backend) in candidates.iter().enumerate() {
            if !backend.breaker.allow_request() {
                attempts.push(format!(
                    "{}: circuit {}",
                    backend.config.id,
                    backend.breaker.state().as_str()
                ));
                previous_backend = Some(backend.config.id.clone());
                continue;
            }

            let mut attempt = request.clone();
            let attempt_capture = Arc::new(parking_lot::Mutex::new(None));
            attempt.session_capture = attempt_capture.clone();

            if backend.config.kind != BackendKind::Http
                && !backend.config.resume_args.is_empty()
                && let Some(prior) =
                    self.sessions
                        .get_backend_session(&request.session_key, &backend.config.id, ttl)
            {
                let can_resume = prior.has_history
                    && (backend.config.session_id_regex.is_none()
                        || prior.backend_session_id.is_some());
                if can_resume {
                    attempt.backend_session_id = prior.backend_session_id;
                    let delta = slice_delta_messages(&attempt.messages);
                    if !delta.is_empty() {
                        attempt.delta_messages = Some(delta);
                    }
                }
            }

            let isolate = match workspace_mode {
                WorkspaceMode::Worktree => true,
                WorkspaceMode::Auto => backend.config.isolate_workspace,
                WorkspaceMode::None => false,
            };
            if isolate && backend.config.kind != BackendKind::Http {
                let mut ws_span = otel.as_ref().map(|o| {
                    o.start_span(
                        dispatch_ctx.child(),
                        "workspace.worktree",
                        SpanKind::Internal,
                    )
                });
                if let Some(span) = &mut ws_span {
                    span.set_str("vouch.backend", &backend.config.id);
                }
                match workspaces.acquire(&request.session_key).await {
                    Ok(entry) => {
                        attempt.workdir = Some(entry.path.clone());
                        if let Some(span) = ws_span {
                            span.finish_ok();
                        }
                    }
                    Err(error) => {
                        if let Some(span) = ws_span {
                            span.finish_err(&error.to_string());
                        }
                        backend.breaker.record_neutral();
                        attempts.push(format!("{}: workspace error: {error}", backend.config.id));
                        previous_backend = Some(backend.config.id.clone());
                        last_error = Some(error);
                        continue;
                    }
                }
            } else if attempt.workdir.is_none() {
                attempt.workdir = backend.config.workdir.clone();
            }

            let mut exec_span = otel
                .as_ref()
                .map(|o| o.start_span(dispatch_ctx.child(), "driver.exec", SpanKind::Client));
            if let Some(span) = &mut exec_span {
                span.set_str("vouch.route", &request.model);
                span.set_str("vouch.backend", &backend.config.id);
                attempt.trace_context = Some(span.context().clone());
            } else {
                attempt.trace_context = Some(dispatch_ctx.child());
            }

            let exec_result = match backend.driver.execute(attempt).await {
                Ok(DriverResponse::Stream(stream)) => {
                    prime_stream(stream).await.map(DriverResponse::Stream)
                }
                other => other,
            };

            match exec_result {
                Ok(response) => {
                    let fallback_used = index > 0;
                    if let Some(span) = &mut exec_span {
                        span.set_bool("vouch.fallback", fallback_used);
                    }
                    if let Some(span) = &mut dispatch_span {
                        span.set_str("vouch.backend", &backend.config.id);
                        span.set_bool("vouch.fallback", fallback_used);
                    }
                    let final_response = match response {
                        DriverResponse::Stream(stream) => {
                            let sessions = self.sessions.clone();
                            let session_key = request.session_key.clone();
                            let backend_id = backend.config.id.clone();
                            let capture = attempt_capture.clone();
                            DriverResponse::Stream(wrap_stream_session_commit(
                                stream,
                                sessions,
                                session_key,
                                backend_id,
                                capture,
                                exec_span,
                                dispatch_span,
                            ))
                        }
                        DriverResponse::Complete(completion) => {
                            let captured = attempt_capture.lock().take();
                            self.sessions.record_backend_session(
                                &request.session_key,
                                &backend.config.id,
                                captured,
                            );
                            if let Some(mut span) = exec_span {
                                span.set_int(
                                    "gen_ai.usage.input_tokens",
                                    completion.usage.prompt_tokens as i64,
                                );
                                span.set_int(
                                    "gen_ai.usage.output_tokens",
                                    completion.usage.completion_tokens as i64,
                                );
                                span.finish_ok();
                            }
                            if let Some(mut span) = dispatch_span {
                                span.set_int(
                                    "gen_ai.usage.input_tokens",
                                    completion.usage.prompt_tokens as i64,
                                );
                                span.set_int(
                                    "gen_ai.usage.output_tokens",
                                    completion.usage.completion_tokens as i64,
                                );
                                span.finish_ok();
                            }
                            DriverResponse::Complete(completion)
                        }
                    };
                    backend.breaker.record_success();
                    self.sessions.set(&request.session_key, &backend.config.id);
                    if let Some(previous) = previous_backend.as_deref() {
                        metrics::record_fallback(previous, &backend.config.id);
                    }
                    return Ok(RoutedResponse {
                        backend_id: backend.config.id.clone(),
                        fallback_used,
                        response: final_response,
                        attempts,
                    });
                }
                Err(error) => {
                    if let Some(span) = exec_span {
                        span.finish_err(&error.to_string());
                    }
                    if error.counts_as_backend_failure() {
                        backend.breaker.record_failure_detailed(
                            error.to_string(),
                            FailureDetails {
                                rate_limited: error.is_rate_limited(),
                                auth_expired: error.is_auth_expired(),
                                retry_after: error.upstream_rate_limit_retry_after(),
                            },
                        );
                    } else {
                        backend.breaker.record_neutral();
                    }
                    attempts.push(format!("{}: {error}", backend.config.id));
                    if !error.is_retryable() {
                        if let Some(span) = dispatch_span {
                            span.finish_err(&error.to_string());
                        }
                        return Err(error);
                    }
                    previous_backend = Some(backend.config.id.clone());
                    last_error = Some(error);
                }
            }
        }

        let final_err = match last_error {
            Some(error) => error,
            None => GatewayError::NoHealthyBackend(format!(
                "no eligible backend for model {:?}: {}",
                request.model,
                attempts.join("; ")
            )),
        };
        if let Some(span) = dispatch_span {
            span.finish_err(&final_err.to_string());
        }
        Err(final_err)
    }

    /// Default model, when configured or unambiguous.
    pub fn default_model(&self) -> Option<String> {
        self.table.load().default_model.clone()
    }

    /// Canonical `[routes]` names plus `[aliases]` exposed by `GET /v1/models`.
    pub fn models(&self) -> Vec<String> {
        let table = self.table.load();
        let mut models: Vec<String> = table
            .routes
            .keys()
            .chain(table.aliases.keys())
            .cloned()
            .collect();
        models.sort();
        models.dedup();
        models
    }

    /// Canonical configured route names (excluding aliases).
    pub fn route_names(&self) -> Vec<String> {
        let table = self.table.load();
        let mut routes: Vec<String> = table.routes.keys().cloned().collect();
        routes.sort();
        routes
    }

    /// Health and chain metadata for every public model.
    pub fn model_info(&self) -> Vec<ModelInfo> {
        let table = self.table.load();
        let mut models: Vec<ModelInfo> = table
            .routes
            .iter()
            .map(|(model, chain)| ModelInfo {
                id: model.clone(),
                backends: chain
                    .iter()
                    .map(|backend| backend_status(backend))
                    .collect(),
            })
            .collect();
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models
    }

    /// Readiness probe based on circuit state.
    pub fn readiness(&self) -> ReadinessReport {
        let table = self.table.load();
        let backends: Vec<BackendStatus> = table
            .order
            .iter()
            .map(|backend| backend_status(backend))
            .collect();
        let ready = !backends.is_empty()
            && backends
                .iter()
                .any(|backend| backend.circuit != CircuitState::Open);
        ReadinessReport { ready, backends }
    }

    /// All configured backends in configuration order.
    pub fn backends(&self) -> Vec<Arc<BackendHandle>> {
        self.table.load().order.clone()
    }

    /// Session affinity map.
    pub fn sessions(&self) -> &Arc<SessionAffinity> {
        &self.sessions
    }

    /// Shared handle to the current session TTL, updated on reload.
    pub fn ttl_handle(&self) -> Arc<AtomicU64> {
        self.ttl_secs.clone()
    }

    /// Current session TTL.
    pub fn session_ttl_secs(&self) -> u64 {
        self.ttl_secs.load(Ordering::Relaxed).max(1)
    }
}

fn wrap_stream_session_commit(
    mut inner: ChunkStream,
    sessions: Arc<SessionAffinity>,
    session_key: String,
    backend_id: String,
    capture: Arc<parking_lot::Mutex<Option<String>>>,
    mut exec_span: Option<ActiveSpan>,
    dispatch_span: Option<ActiveSpan>,
) -> ChunkStream {
    Box::pin(async_stream::stream! {
        let mut error_msg: Option<String> = None;
        let mut last_usage = None;
        while let Some(item) = inner.next().await {
            match &item {
                Ok(chunk) => {
                    if let Some(usage) = chunk.usage {
                        last_usage = Some(usage);
                    }
                }
                Err(error) => {
                    error_msg = Some(error.to_string());
                }
            }
            yield item;
        }
        if error_msg.is_none() {
            let captured = capture.lock().take();
            sessions.record_backend_session(&session_key, &backend_id, captured);
            if let Some(mut span) = exec_span.take() {
                if let Some(usage) = last_usage {
                    span.set_int("gen_ai.usage.input_tokens", usage.prompt_tokens as i64);
                    span.set_int("gen_ai.usage.output_tokens", usage.completion_tokens as i64);
                }
                span.finish_ok();
            }
            if let Some(mut span) = dispatch_span {
                if let Some(usage) = last_usage {
                    span.set_int("gen_ai.usage.input_tokens", usage.prompt_tokens as i64);
                    span.set_int("gen_ai.usage.output_tokens", usage.completion_tokens as i64);
                }
                span.finish_ok();
            }
        } else if let Some(msg) = error_msg {
            if let Some(span) = exec_span.take() {
                span.finish_err(&msg);
            }
            if let Some(span) = dispatch_span {
                span.finish_err(&msg);
            }
        }
    })
}

fn backend_status(backend: &BackendHandle) -> BackendStatus {
    let snapshot = backend.breaker.snapshot();
    BackendStatus {
        id: backend.config.id.clone(),
        kind: backend.config.kind,
        max_concurrency: backend.config.max_concurrency,
        circuit: snapshot.state,
        consecutive_failures: snapshot.consecutive_failures,
        open_remaining_secs: snapshot.open_remaining_secs,
        auth_expired: snapshot.auth_expired,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_session_ids_are_preserved_and_sanitized() {
        assert_eq!(
            derive_session_key(Some("my-session"), None, "hi"),
            "my-session"
        );
        let hashed = derive_session_key(Some("bad session with spaces"), None, "hi");
        assert_eq!(hashed.len(), 16);
    }

    #[test]
    fn prompt_hash_is_deterministic() {
        let first = derive_session_key(None, Some("system"), "hello");
        let second = derive_session_key(None, Some("system"), "hello");
        assert_eq!(first, second);
        assert_ne!(first, derive_session_key(None, Some("system"), "other"));
    }

    #[test]
    fn affinity_expires() {
        let sessions = SessionAffinity::new();
        sessions.set("s", "backend");
        assert_eq!(
            sessions.get("s", Duration::from_secs(60)).as_deref(),
            Some("backend")
        );
        std::thread::sleep(Duration::from_millis(10));
        assert_eq!(sessions.get("s", Duration::from_millis(1)), None);
    }

    #[test]
    fn delta_slices_after_last_assistant_turn() {
        let msgs = vec![
            ChatMessage::text("user", "turn 1"),
            ChatMessage::text("assistant", "reply 1"),
            ChatMessage::text("user", "turn 2"),
        ];
        let delta = slice_delta_messages(&msgs);
        assert_eq!(delta.len(), 1);
        assert_eq!(delta[0].content.as_plain_text(), "turn 2");
    }

    #[test]
    fn bounded_session_affinity_evicts_oldest() {
        let sessions = SessionAffinity::with_max_entries(2);
        sessions.set("s1", "b1");
        sessions.set("s2", "b2");
        sessions.set("s3", "b3");
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions.get("s1", Duration::from_secs(60)), None);
        assert_eq!(
            sessions.get("s2", Duration::from_secs(60)).as_deref(),
            Some("b2")
        );
        assert_eq!(
            sessions.get("s3", Duration::from_secs(60)).as_deref(),
            Some("b3")
        );
    }
}
