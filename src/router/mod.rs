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
    config::{BackendConfig, BackendKind, GatewayConfig, RoutePolicyConfig, RouteStrategy},
    driver::{
        ChatMessage, ChunkStream, Driver, DriverResponse, ExecutionRequest, WorkspaceMode,
        cli_pipe::CliPipeDriver, http_proxy::HttpProxyDriver, prime_stream,
        pty_runner::PtyRunnerDriver, tool_bridge::is_schema_validation_error,
        workspace::WorkspaceManager,
    },
    error::{GatewayError, Result},
    memory::{HandoffReason, SharedMemoryEngine, ledger::format_trace_session_summary},
    metrics,
    persistence::{
        PersistedAffinityEntry, PersistedBackendSession, evict_oldest_if_full, instant_from_unix,
        unix_now_secs,
    },
    telemetry::{ActiveSpan, OtelExporter, SpanKind, TraceContext},
};

use self::circuit_breaker::{BreakerSettings, CircuitBreaker, CircuitState, FailureDetails};

const TTFT_EWMA_ALPHA: f64 = 0.3;

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
    profile_affinity: DashMap<(String, String), AffinityEntry>,
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
            profile_affinity: DashMap::new(),
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
        if self.profile_affinity.len() > cap {
            evict_oldest_if_full(&self.profile_affinity, cap.saturating_add(1), |e| e.seq);
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

    /// Look up sticky credential profile for `(session_key, backend_id)`.
    pub fn get_profile(
        &self,
        session_key: &str,
        backend_id: &str,
        ttl: Duration,
    ) -> Option<String> {
        let composite = (session_key.to_string(), backend_id.to_string());
        let entry = self.profile_affinity.get(&composite)?;
        if entry.last_used.elapsed() > ttl {
            drop(entry);
            self.profile_affinity.remove(&composite);
            None
        } else {
            Some(entry.backend_id.clone())
        }
    }

    /// Record sticky credential profile for `(session_key, backend_id)`.
    pub fn set_profile(&self, session_key: &str, backend_id: &str, profile_id: &str) {
        let composite = (session_key.to_string(), backend_id.to_string());
        let cap = self.max_entries.load(Ordering::Relaxed).max(1);
        if !self.profile_affinity.contains_key(&composite) {
            evict_oldest_if_full(&self.profile_affinity, cap, |e| e.seq);
        }
        let seq = self.next_seq();
        self.profile_affinity.insert(
            composite,
            AffinityEntry {
                backend_id: profile_id.to_string(),
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

        let expired_profiles: Vec<(String, String)> = self
            .profile_affinity
            .iter()
            .filter(|entry| entry.last_used.elapsed() > ttl)
            .map(|entry| entry.key().clone())
            .collect();
        for key in expired_profiles {
            self.profile_affinity.remove(&key);
        }
    }

    /// Remove all affinity and continuation state associated with `session_key`.
    pub fn remove_session(&self, session_key: &str) -> bool {
        let mut removed = self.entries.remove(session_key).is_some();
        let bs_keys: Vec<(String, String)> = self
            .backend_sessions
            .iter()
            .filter(|entry| entry.key().0 == session_key)
            .map(|entry| entry.key().clone())
            .collect();
        for key in bs_keys {
            if self.backend_sessions.remove(&key).is_some() {
                removed = true;
            }
        }
        let prof_keys: Vec<(String, String)> = self
            .profile_affinity
            .iter()
            .filter(|entry| entry.key().0 == session_key)
            .map(|entry| entry.key().clone())
            .collect();
        for key in prof_keys {
            if self.profile_affinity.remove(&key).is_some() {
                removed = true;
            }
        }
        removed
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
    /// Weighted round-robin cursor across configured credential profiles.
    profile_cursor: AtomicUsize,
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
    policies: BTreeMap<String, RoutePolicyConfig>,
    schema_repair_retries: u8,
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
    /// Credential profile selected for the winning execution, if any.
    pub profile_id: Option<String>,
    /// True when an earlier candidate was skipped or failed.
    pub fallback_used: bool,
    /// Driver response.
    pub response: DriverResponse,
    /// Human-readable attempt log.
    pub attempts: Vec<String>,
    /// Previous backend id if execution transitioned from another agent in the same session.
    pub handoff_from: Option<String>,
    /// Handoff trigger reason (`"failover"`, `"policy_reorder"`, `"model_switch"`), if any.
    pub handoff_reason: Option<String>,
}

/// Result of a routed `POST /v1/embeddings` execution.
pub struct RoutedEmbeddings {
    /// Backend that produced the embeddings.
    pub backend_id: String,
    /// Credential profile used, if any.
    pub profile_id: Option<String>,
    /// True when an earlier candidate failed or was skipped.
    pub fallback_used: bool,
    /// Embedding vectors and usage.
    pub result: crate::driver::EmbeddingResult,
    /// Human-readable attempt summary.
    pub attempts: Vec<String>,
}

/// Model router.
#[derive(Debug)]
pub struct Router {
    table: ArcSwap<RouterTable>,
    sessions: Arc<SessionAffinity>,
    breakers: DashMap<String, Arc<CircuitBreaker>>,
    profile_breakers: DashMap<String, Arc<CircuitBreaker>>,
    ttft_ewma_ms: DashMap<String, f64>,
    ttl_secs: Arc<AtomicU64>,
    otel: ArcSwap<Option<Arc<OtelExporter>>>,
    memory: ArcSwap<Option<Arc<SharedMemoryEngine>>>,
}

impl Router {
    /// Build a router from validated configuration.
    pub fn new(config: &GatewayConfig, sessions: Arc<SessionAffinity>) -> Result<Self> {
        let router = Self {
            table: ArcSwap::from_pointee(RouterTable::default()),
            sessions,
            breakers: DashMap::new(),
            profile_breakers: DashMap::new(),
            ttft_ewma_ms: DashMap::new(),
            ttl_secs: Arc::new(AtomicU64::new(config.defaults.session_ttl_secs.max(1))),
            otel: ArcSwap::from_pointee(None),
            memory: ArcSwap::from_pointee(None),
        };
        router.rebuild(config)?;
        Ok(router)
    }

    /// Attach the shared memory & `trace` bridge engine for cross-agent handoffs.
    pub fn set_memory(&self, memory: Arc<SharedMemoryEngine>) {
        self.memory.store(Arc::new(Some(memory)));
    }

    /// Manually reset the circuit breaker for `backend_id` (and all its profiles).
    pub fn reset_circuit(&self, backend_id: &str) -> bool {
        let Some(breaker) = self.breakers.get(backend_id) else {
            return false;
        };
        breaker.reset();
        let prefix = format!("{backend_id}::profile::");
        for entry in self.profile_breakers.iter() {
            if entry.key().starts_with(&prefix) {
                entry.value().reset();
            }
        }
        true
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
        let mut active_profiles = HashSet::new();

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
            for profile in &backend.profiles {
                let pkey = profile_breaker_key(&backend.id, &profile.id);
                active_profiles.insert(pkey.clone());
                match self.profile_breakers.get(&pkey) {
                    Some(existing) => existing.reconfigure(settings.clone()),
                    None => {
                        let pb = Arc::new(CircuitBreaker::new(pkey.clone(), settings.clone()));
                        self.profile_breakers.insert(pkey, pb);
                    }
                }
            }
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
                profile_cursor: AtomicUsize::new(0),
            });
            handles.insert(backend.id.clone(), handle.clone());
            order.push(handle);
        }

        self.breakers.retain(|id, _| active.contains(id));
        self.profile_breakers
            .retain(|key, _| active_profiles.contains(key));
        self.ttft_ewma_ms.retain(|id, _| active.contains(id));

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
            policies: config.route_policies.clone(),
            schema_repair_retries: config.defaults.schema_repair_retries,
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

    /// Execute a request with fallback, strategy policies, multi-profile rotation, and affinity.
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

        let mut candidates = chain;
        if let Some(allowlist) = &request.allowed_backends {
            candidates.retain(|backend| {
                allowlist
                    .iter()
                    .any(|allowed| allowed == &backend.config.id)
            });
            if candidates.is_empty() {
                let err = GatewayError::Forbidden(format!(
                    "token is not allowed to use any backend for model {:?}",
                    request.model
                ));
                if let Some(span) = dispatch_span {
                    span.finish_err(&err.to_string());
                }
                return Err(err);
            }
        }

        let policy = table
            .policies
            .get(&request.model)
            .cloned()
            .unwrap_or(RoutePolicyConfig {
                strategy: RouteStrategy::Ordered,
                hedge_delay_ms: 2500,
            });
        let ttl = Duration::from_secs(self.ttl_secs.load(Ordering::Relaxed).max(1));

        match policy.strategy {
            RouteStrategy::LeastLatency => {
                let mut indexed: Vec<(usize, Arc<BackendHandle>)> =
                    candidates.into_iter().enumerate().collect();
                indexed.sort_by(|(idx_a, a), (idx_b, b)| {
                    let lat_a = self
                        .ttft_ewma_ms
                        .get(&a.config.id)
                        .map(|v| *v)
                        .unwrap_or(0.0);
                    let lat_b = self
                        .ttft_ewma_ms
                        .get(&b.config.id)
                        .map(|v| *v)
                        .unwrap_or(0.0);
                    lat_a.total_cmp(&lat_b).then_with(|| idx_a.cmp(idx_b))
                });
                candidates = indexed.into_iter().map(|(_, b)| b).collect();
            }
            RouteStrategy::Ordered | RouteStrategy::Hedged => {
                if let Some(sticky) = self.sessions.get(&request.session_key, ttl)
                    && let Some(position) = candidates
                        .iter()
                        .position(|backend| backend.config.id == sticky)
                {
                    let backend = candidates.remove(position);
                    candidates.insert(0, backend);
                }
            }
        }

        let mut attempts = Vec::new();
        let mut last_error: Option<GatewayError> = None;
        let mut previous_backend: Option<String> = None;
        let schema_repair_retries = table.schema_repair_retries;

        let seq_start_idx = if policy.strategy == RouteStrategy::Hedged && candidates.len() >= 2 {
            let mut circuit_ok = Vec::new();
            let mut deferred = 0usize;
            for backend in &candidates {
                if backend.breaker.allow_request() {
                    circuit_ok.push(backend.clone());
                } else {
                    attempts.push(format!(
                        "{}: circuit {}",
                        backend.config.id,
                        backend.breaker.state().as_str()
                    ));
                    previous_backend = Some(backend.config.id.clone());
                }
                deferred += 1;
                if circuit_ok.len() == 2 {
                    break;
                }
            }

            if circuit_ok.len() == 2 {
                let c1 = circuit_ok[0].clone();
                let c2 = circuit_ok[1].clone();
                if let Some(outcome) = self
                    .try_hedged_pair(
                        c1,
                        c2,
                        &request,
                        policy.hedge_delay_ms,
                        workspace_mode,
                        workspaces,
                        &dispatch_ctx,
                        ttl,
                        schema_repair_retries,
                        &otel,
                        &candidates,
                        &mut attempts,
                        &mut last_error,
                        &mut previous_backend,
                    )
                    .await?
                {
                    return Ok(self.finalize_winner(
                        outcome,
                        &request.session_key,
                        previous_backend.as_deref(),
                        attempts,
                        dispatch_span,
                    ));
                }
                deferred
            } else {
                // Consume the single circuit-checked candidate if only 1 was healthy before sequential scan.
                if let Some(single) = circuit_ok.into_iter().next() {
                    match self
                        .execute_single_backend(
                            &single,
                            &request,
                            workspace_mode,
                            workspaces,
                            &dispatch_ctx,
                            ttl,
                            schema_repair_retries,
                            &otel,
                            true,
                            previous_backend.as_deref(),
                            &candidates,
                            policy.strategy,
                            &mut attempts,
                        )
                        .await
                    {
                        Ok(outcome) => {
                            return Ok(self.finalize_winner(
                                outcome,
                                &request.session_key,
                                previous_backend.as_deref(),
                                attempts,
                                dispatch_span,
                            ));
                        }
                        Err(error) => {
                            if !error.is_retryable() {
                                if let Some(span) = dispatch_span {
                                    span.finish_err(&error.to_string());
                                }
                                return Err(error);
                            }
                            previous_backend = Some(single.config.id.clone());
                            last_error = Some(error);
                        }
                    }
                }
                deferred
            }
        } else {
            0
        };

        for (offset, backend) in candidates.iter().skip(seq_start_idx).enumerate() {
            if !backend.breaker.allow_request() {
                attempts.push(format!(
                    "{}: circuit {}",
                    backend.config.id,
                    backend.breaker.state().as_str()
                ));
                previous_backend = Some(backend.config.id.clone());
                continue;
            }
            let fallback_used = seq_start_idx + offset > 0 || previous_backend.is_some();
            match self
                .execute_single_backend(
                    backend,
                    &request,
                    workspace_mode,
                    workspaces,
                    &dispatch_ctx,
                    ttl,
                    schema_repair_retries,
                    &otel,
                    fallback_used,
                    previous_backend.as_deref(),
                    &candidates,
                    policy.strategy,
                    &mut attempts,
                )
                .await
            {
                Ok(outcome) => {
                    return Ok(self.finalize_winner(
                        outcome,
                        &request.session_key,
                        previous_backend.as_deref(),
                        attempts,
                        dispatch_span,
                    ));
                }
                Err(error) => {
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

    fn record_ttft(&self, backend_id: &str, elapsed: Duration) {
        let sample_ms = elapsed.as_secs_f64() * 1000.0;
        self.ttft_ewma_ms
            .entry(backend_id.to_string())
            .and_modify(|ewma| {
                *ewma = (*ewma * (1.0 - TTFT_EWMA_ALPHA)) + (sample_ms * TTFT_EWMA_ALPHA);
            })
            .or_insert(sample_ms);
    }

    fn ordered_profiles(
        &self,
        backend: &BackendHandle,
        session_key: &str,
        ttl: Duration,
    ) -> Vec<ProfileSlot> {
        let profiles = &backend.config.profiles;
        if profiles.is_empty() {
            return vec![ProfileSlot {
                id: None,
                env: BTreeMap::new(),
                breaker: backend.breaker.clone(),
            }];
        }

        let mut weighted_indices = Vec::new();
        for (idx, profile) in profiles.iter().enumerate() {
            let weight = profile.weight.clamp(1, 64) as usize;
            for _ in 0..weight {
                weighted_indices.push(idx);
            }
        }
        let start = backend.profile_cursor.fetch_add(1, Ordering::Relaxed) % weighted_indices.len();
        let mut seen = vec![false; profiles.len()];
        let mut ordered_indices = Vec::with_capacity(profiles.len());
        for offset in 0..weighted_indices.len() {
            let idx = weighted_indices[(start + offset) % weighted_indices.len()];
            if !seen[idx] {
                seen[idx] = true;
                ordered_indices.push(idx);
            }
        }

        if let Some(sticky_id) = self
            .sessions
            .get_profile(session_key, &backend.config.id, ttl)
            && let Some(pos) = ordered_indices
                .iter()
                .position(|&idx| profiles[idx].id == sticky_id)
        {
            let sticky_idx = ordered_indices.remove(pos);
            ordered_indices.insert(0, sticky_idx);
        }

        ordered_indices
            .into_iter()
            .map(|idx| {
                let profile = &profiles[idx];
                let pkey = profile_breaker_key(&backend.config.id, &profile.id);
                let breaker = self
                    .profile_breakers
                    .get(&pkey)
                    .map(|b| b.clone())
                    .unwrap_or_else(|| backend.breaker.clone());
                ProfileSlot {
                    id: Some(profile.id.clone()),
                    env: profile.env.clone(),
                    breaker,
                }
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    async fn prepare_backend_attempt(
        &self,
        backend: &BackendHandle,
        base: &ExecutionRequest,
        workspace_mode: WorkspaceMode,
        workspaces: &WorkspaceManager,
        dispatch_ctx: &TraceContext,
        otel: &Option<Arc<OtelExporter>>,
        previous_backend_failed: Option<&str>,
        route_candidates: &[Arc<BackendHandle>],
        strategy: RouteStrategy,
    ) -> std::result::Result<
        (
            ExecutionRequest,
            Arc<parking_lot::Mutex<Option<String>>>,
            Option<ActiveSpan>,
            Option<String>,
            Option<HandoffReason>,
        ),
        GatewayError,
    > {
        let ttl = Duration::from_secs(self.ttl_secs.load(Ordering::Relaxed).max(1));
        let mut attempt = base.clone();
        let attempt_capture = Arc::new(parking_lot::Mutex::new(None));
        attempt.session_capture = attempt_capture.clone();

        if backend.config.kind != BackendKind::Http
            && !backend.config.resume_args.is_empty()
            && let Some(prior) =
                self.sessions
                    .get_backend_session(&base.session_key, &backend.config.id, ttl)
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
            match workspaces.acquire(&base.session_key).await {
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
                    return Err(error);
                }
            }
        } else if attempt.workdir.is_none() {
            attempt.workdir = backend.config.workdir.clone();
        }

        let mut exec_span = otel
            .as_ref()
            .map(|o| o.start_span(dispatch_ctx.child(), "driver.exec", SpanKind::Client));
        if let Some(span) = &mut exec_span {
            span.set_str("vouch.route", &base.model);
            span.set_str("vouch.backend", &backend.config.id);
            attempt.trace_context = Some(span.context().clone());
        } else {
            attempt.trace_context = Some(dispatch_ctx.child());
        }

        let (handoff_from, handoff_reason) = if let Some(mem) = self.memory.load().as_ref() {
            let plan = mem
                .hydrate_attempt_for_candidate(
                    base,
                    &backend.config,
                    &mut attempt,
                    previous_backend_failed,
                    |id| route_candidates.iter().any(|c| c.config.id == id),
                    strategy,
                    ttl,
                )
                .await;
            (plan.handoff_from, plan.reason)
        } else {
            (None, None)
        };

        Ok((
            attempt,
            attempt_capture,
            exec_span,
            handoff_from,
            handoff_reason,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_single_backend(
        &self,
        backend: &Arc<BackendHandle>,
        base_request: &ExecutionRequest,
        workspace_mode: WorkspaceMode,
        workspaces: &WorkspaceManager,
        dispatch_ctx: &TraceContext,
        ttl: Duration,
        schema_repair_retries: u8,
        otel: &Option<Arc<OtelExporter>>,
        fallback_used: bool,
        previous_backend_failed: Option<&str>,
        route_candidates: &[Arc<BackendHandle>],
        strategy: RouteStrategy,
        attempts: &mut Vec<String>,
    ) -> Result<CandidateOutcome> {
        let (base_attempt, capture, mut exec_span, handoff_from, handoff_reason) = match self
            .prepare_backend_attempt(
                backend,
                base_request,
                workspace_mode,
                workspaces,
                dispatch_ctx,
                otel,
                previous_backend_failed,
                route_candidates,
                strategy,
            )
            .await
        {
            Ok(prep) => prep,
            Err(error) => {
                attempts.push(format!("{}: workspace error: {error}", backend.config.id));
                return Err(error);
            }
        };

        let profiles = self.ordered_profiles(backend, &base_request.session_key, ttl);
        let has_profiles = profiles.first().is_some_and(|p| p.id.is_some());
        let mut last_profile_err: Option<GatewayError> = None;

        for slot in profiles {
            if has_profiles && !slot.breaker.allow_request() {
                attempts.push(format!(
                    "{}:{}: circuit {}",
                    backend.config.id,
                    slot.id.as_deref().unwrap_or("default"),
                    slot.breaker.state().as_str()
                ));
                continue;
            }

            let mut attempt = base_attempt.clone();
            attempt.profile_id = slot.id.clone();
            attempt.profile_env = slot.env.clone();

            let started = Instant::now();
            let exec_result = execute_with_repair(
                &backend.driver,
                attempt,
                schema_repair_retries,
                backend.config.kind,
            )
            .await;

            match exec_result {
                Ok(response) => {
                    let elapsed = started.elapsed();
                    self.record_ttft(&backend.config.id, elapsed);
                    slot.breaker.record_success();
                    backend.breaker.record_success();
                    if let Some(pid) = &slot.id {
                        self.sessions.set_profile(
                            &base_request.session_key,
                            &backend.config.id,
                            pid,
                        );
                    }
                    return Ok(CandidateOutcome {
                        backend: backend.clone(),
                        profile_id: slot.id.clone(),
                        response,
                        capture,
                        exec_span,
                        fallback_used,
                        handoff_from,
                        handoff_reason,
                    });
                }
                Err(error) => {
                    if base_request.cancel.is_cancelled() {
                        if let Some(span) = exec_span.take() {
                            span.finish_err("cancelled");
                        }
                        slot.breaker.record_neutral();
                        backend.breaker.record_neutral();
                        return Err(error);
                    }
                    let prof_label = slot.id.as_deref().unwrap_or("default");
                    if has_profiles {
                        attempts.push(format!("{}:{}: {error}", backend.config.id, prof_label));
                    } else {
                        attempts.push(format!("{}: {error}", backend.config.id));
                    }
                    if error.counts_as_backend_failure() {
                        slot.breaker.record_failure_detailed(
                            error.to_string(),
                            FailureDetails {
                                rate_limited: error.is_rate_limited(),
                                auth_expired: error.is_auth_expired(),
                                retry_after: error.upstream_rate_limit_retry_after(),
                            },
                        );
                    } else {
                        slot.breaker.record_neutral();
                    }
                    let retryable = error.is_retryable();
                    last_profile_err = Some(error);
                    if !retryable || !has_profiles {
                        break;
                    }
                }
            }
        }

        let err = last_profile_err.unwrap_or_else(|| {
            GatewayError::NoHealthyBackend(format!(
                "backend {:?} has no healthy credential profiles",
                backend.config.id
            ))
        });
        if has_profiles {
            if err.counts_as_backend_failure() {
                backend.breaker.record_failure_detailed(
                    err.to_string(),
                    FailureDetails {
                        rate_limited: err.is_rate_limited(),
                        auth_expired: err.is_auth_expired(),
                        retry_after: err.upstream_rate_limit_retry_after(),
                    },
                );
            } else {
                backend.breaker.record_neutral();
            }
        }
        if let Some(span) = exec_span {
            span.finish_err(&err.to_string());
        }
        Err(err)
    }

    #[allow(clippy::too_many_arguments)]
    async fn try_hedged_pair(
        &self,
        c1: Arc<BackendHandle>,
        c2: Arc<BackendHandle>,
        base_request: &ExecutionRequest,
        hedge_delay_ms: u64,
        workspace_mode: WorkspaceMode,
        workspaces: &WorkspaceManager,
        dispatch_ctx: &TraceContext,
        ttl: Duration,
        schema_repair_retries: u8,
        otel: &Option<Arc<OtelExporter>>,
        route_candidates: &[Arc<BackendHandle>],
        attempts: &mut Vec<String>,
        last_error: &mut Option<GatewayError>,
        previous_backend: &mut Option<String>,
    ) -> Result<Option<CandidateOutcome>> {
        let mut req1 = base_request.clone();
        let cancel1 = base_request.cancel.child_token();
        req1.cancel = cancel1.clone();

        let (mut attempt1, capture1, exec_span1, handoff_from1, handoff_reason1) = match self
            .prepare_backend_attempt(
                &c1,
                &req1,
                workspace_mode,
                workspaces,
                dispatch_ctx,
                otel,
                previous_backend.as_deref(),
                route_candidates,
                RouteStrategy::Hedged,
            )
            .await
        {
            Ok(prep) => prep,
            Err(error) => {
                attempts.push(format!("{}: workspace error: {error}", c1.config.id));
                *previous_backend = Some(c1.config.id.clone());
                let retryable = error.is_retryable();
                *last_error = Some(error);
                c2.breaker.record_neutral();
                if !retryable {
                    return Err(last_error
                        .take()
                        .unwrap_or_else(|| GatewayError::Upstream("non-retryable error".into())));
                }
                if !c2.breaker.allow_request() {
                    attempts.push(format!(
                        "{}: circuit {}",
                        c2.config.id,
                        c2.breaker.state().as_str()
                    ));
                    return Ok(None);
                }
                match self
                    .execute_single_backend(
                        &c2,
                        base_request,
                        workspace_mode,
                        workspaces,
                        dispatch_ctx,
                        ttl,
                        schema_repair_retries,
                        otel,
                        true,
                        previous_backend.as_deref(),
                        route_candidates,
                        RouteStrategy::Hedged,
                        attempts,
                    )
                    .await
                {
                    Ok(win) => return Ok(Some(win)),
                    Err(err2) => {
                        *previous_backend = Some(c2.config.id.clone());
                        *last_error = Some(err2);
                        return Ok(None);
                    }
                }
            }
        };

        let profiles1 = self.ordered_profiles(&c1, &base_request.session_key, ttl);
        let slot1 = profiles1
            .into_iter()
            .find(|s| s.id.is_none() || s.breaker.allow_request());
        let Some(slot1) = slot1 else {
            c2.breaker.record_neutral();
            attempts.push(format!("{}: no healthy profile", c1.config.id));
            *previous_backend = Some(c1.config.id.clone());
            if !c2.breaker.allow_request() {
                attempts.push(format!(
                    "{}: circuit {}",
                    c2.config.id,
                    c2.breaker.state().as_str()
                ));
                return Ok(None);
            }
            match self
                .execute_single_backend(
                    &c2,
                    base_request,
                    workspace_mode,
                    workspaces,
                    dispatch_ctx,
                    ttl,
                    schema_repair_retries,
                    otel,
                    true,
                    previous_backend.as_deref(),
                    route_candidates,
                    RouteStrategy::Hedged,
                    attempts,
                )
                .await
            {
                Ok(win) => return Ok(Some(win)),
                Err(err2) => {
                    *previous_backend = Some(c2.config.id.clone());
                    *last_error = Some(err2);
                    return Ok(None);
                }
            }
        };
        attempt1.profile_id = slot1.id.clone();
        attempt1.profile_env = slot1.env.clone();

        let drv1 = c1.driver.clone();
        let kind1 = c1.config.kind;
        let started1 = Instant::now();
        let mut handle1 = tokio::spawn(async move {
            let res = execute_with_repair(&drv1, attempt1, schema_repair_retries, kind1).await;
            (res, started1.elapsed())
        });

        let delay = tokio::time::sleep(Duration::from_millis(hedge_delay_ms.max(1)));
        tokio::pin!(delay);

        enum FirstPhase {
            Finished1(
                std::result::Result<(Result<DriverResponse>, Duration), tokio::task::JoinError>,
            ),
            DelayElapsed,
        }

        let phase = tokio::select! {
            res = &mut handle1 => FirstPhase::Finished1(res),
            _ = &mut delay => FirstPhase::DelayElapsed,
        };

        match phase {
            FirstPhase::Finished1(Ok((Ok(response), elapsed))) => {
                c2.breaker.record_neutral();
                self.record_ttft(&c1.config.id, elapsed);
                slot1.breaker.record_success();
                c1.breaker.record_success();
                if let Some(pid) = &slot1.id {
                    self.sessions
                        .set_profile(&base_request.session_key, &c1.config.id, pid);
                }
                return Ok(Some(CandidateOutcome {
                    backend: c1,
                    profile_id: slot1.id.clone(),
                    response,
                    capture: capture1,
                    exec_span: exec_span1,
                    fallback_used: false,
                    handoff_from: handoff_from1.clone(),
                    handoff_reason: handoff_reason1,
                }));
            }
            FirstPhase::Finished1(join_res) => {
                c2.breaker.record_neutral();
                if let Some(span) = exec_span1 {
                    span.finish_err("primary failed before hedge delay");
                }
                let err = match join_res {
                    Ok((Err(e), _)) => e,
                    Err(e) => GatewayError::Internal(format!("hedged task failed: {e}")),
                    _ => unreachable!(),
                };
                if base_request.cancel.is_cancelled() {
                    slot1.breaker.record_neutral();
                    c1.breaker.record_neutral();
                    return Err(err);
                }
                record_candidate_failure(&c1, Some(&slot1), &err, attempts);
                *previous_backend = Some(c1.config.id.clone());
                let retryable = err.is_retryable();
                *last_error = Some(err);
                if !retryable {
                    return Err(last_error
                        .take()
                        .unwrap_or_else(|| GatewayError::Upstream("non-retryable error".into())));
                }
                match self
                    .execute_single_backend(
                        &c2,
                        base_request,
                        workspace_mode,
                        workspaces,
                        dispatch_ctx,
                        ttl,
                        schema_repair_retries,
                        otel,
                        true,
                        previous_backend.as_deref(),
                        route_candidates,
                        RouteStrategy::Hedged,
                        attempts,
                    )
                    .await
                {
                    Ok(win) => return Ok(Some(win)),
                    Err(err2) => {
                        *previous_backend = Some(c2.config.id.clone());
                        *last_error = Some(err2);
                        return Ok(None);
                    }
                }
            }
            FirstPhase::DelayElapsed => {}
        }

        // Speculatively launch c2 concurrently while c1 continues.
        let mut req2 = base_request.clone();
        let cancel2 = base_request.cancel.child_token();
        req2.cancel = cancel2.clone();
        let (mut attempt2, capture2, exec_span2, handoff_from2, handoff_reason2) = match self
            .prepare_backend_attempt(
                &c2,
                &req2,
                workspace_mode,
                workspaces,
                dispatch_ctx,
                otel,
                previous_backend.as_deref(),
                route_candidates,
                RouteStrategy::Hedged,
            )
            .await
        {
            Ok(prep) => prep,
            Err(error) => {
                attempts.push(format!("{}: workspace error: {error}", c2.config.id));
                match handle1.await {
                    Ok((Ok(response), elapsed)) => {
                        self.record_ttft(&c1.config.id, elapsed);
                        slot1.breaker.record_success();
                        c1.breaker.record_success();
                        if let Some(pid) = &slot1.id {
                            self.sessions.set_profile(
                                &base_request.session_key,
                                &c1.config.id,
                                pid,
                            );
                        }
                        return Ok(Some(CandidateOutcome {
                            backend: c1,
                            profile_id: slot1.id.clone(),
                            response,
                            capture: capture1,
                            exec_span: exec_span1,
                            fallback_used: false,
                            handoff_from: handoff_from1.clone(),
                            handoff_reason: handoff_reason1,
                        }));
                    }
                    Ok((Err(e), _)) => {
                        if let Some(span) = exec_span1 {
                            span.finish_err(&e.to_string());
                        }
                        record_candidate_failure(&c1, Some(&slot1), &e, attempts);
                        *previous_backend = Some(c1.config.id.clone());
                        *last_error = Some(e);
                        return Ok(None);
                    }
                    Err(e) => {
                        if let Some(span) = exec_span1 {
                            span.finish_err(&e.to_string());
                        }
                        *previous_backend = Some(c1.config.id.clone());
                        *last_error =
                            Some(GatewayError::Internal(format!("hedged task panicked: {e}")));
                        return Ok(None);
                    }
                }
            }
        };

        let profiles2 = self.ordered_profiles(&c2, &base_request.session_key, ttl);
        let slot2 = match profiles2
            .into_iter()
            .find(|s| s.id.is_none() || s.breaker.allow_request())
        {
            Some(s) => s,
            None => {
                if let Some(span) = exec_span2 {
                    span.finish_err("no healthy profile");
                }
                attempts.push(format!("{}: no healthy profile", c2.config.id));
                match handle1.await {
                    Ok((Ok(response), elapsed)) => {
                        self.record_ttft(&c1.config.id, elapsed);
                        slot1.breaker.record_success();
                        c1.breaker.record_success();
                        if let Some(pid) = &slot1.id {
                            self.sessions.set_profile(
                                &base_request.session_key,
                                &c1.config.id,
                                pid,
                            );
                        }
                        return Ok(Some(CandidateOutcome {
                            backend: c1,
                            profile_id: slot1.id.clone(),
                            response,
                            capture: capture1,
                            exec_span: exec_span1,
                            fallback_used: false,
                            handoff_from: handoff_from1.clone(),
                            handoff_reason: handoff_reason1,
                        }));
                    }
                    Ok((Err(e), _)) => {
                        if let Some(span) = exec_span1 {
                            span.finish_err(&e.to_string());
                        }
                        record_candidate_failure(&c1, Some(&slot1), &e, attempts);
                        *previous_backend = Some(c1.config.id.clone());
                        *last_error = Some(e);
                        return Ok(None);
                    }
                    Err(e) => {
                        if let Some(span) = exec_span1 {
                            span.finish_err(&e.to_string());
                        }
                        *previous_backend = Some(c1.config.id.clone());
                        *last_error =
                            Some(GatewayError::Internal(format!("hedged task panicked: {e}")));
                        return Ok(None);
                    }
                }
            }
        };
        attempt2.profile_id = slot2.id.clone();
        attempt2.profile_env = slot2.env.clone();

        let drv2 = c2.driver.clone();
        let kind2 = c2.config.kind;
        let started2 = Instant::now();
        let handle2 = tokio::spawn(async move {
            let res = execute_with_repair(&drv2, attempt2, schema_repair_retries, kind2).await;
            (res, started2.elapsed())
        });

        let mut opt1 = Some(handle1);
        let mut opt2 = Some(handle2);
        let mut span1 = exec_span1;
        let mut span2 = exec_span2;

        loop {
            enum RaceEvt {
                One(
                    std::result::Result<(Result<DriverResponse>, Duration), tokio::task::JoinError>,
                ),
                Two(
                    std::result::Result<(Result<DriverResponse>, Duration), tokio::task::JoinError>,
                ),
            }
            let evt = match (&mut opt1, &mut opt2) {
                (Some(h1), Some(h2)) => tokio::select! {
                    r1 = h1 => {
                        opt1 = None;
                        RaceEvt::One(r1)
                    }
                    r2 = h2 => {
                        opt2 = None;
                        RaceEvt::Two(r2)
                    }
                },
                (Some(_), None) => {
                    let Some(h1) = opt1.take() else {
                        break;
                    };
                    RaceEvt::One(h1.await)
                }
                (None, Some(_)) => {
                    let Some(h2) = opt2.take() else {
                        break;
                    };
                    RaceEvt::Two(h2.await)
                }
                (None, None) => break,
            };

            match evt {
                RaceEvt::One(Ok((Ok(response), elapsed))) => {
                    self.record_ttft(&c1.config.id, elapsed);
                    slot1.breaker.record_success();
                    c1.breaker.record_success();
                    if let Some(pid) = &slot1.id {
                        self.sessions
                            .set_profile(&base_request.session_key, &c1.config.id, pid);
                    }
                    if let Some(h2) = opt2.take() {
                        cancel2.cancel();
                        c2.breaker.record_neutral();
                        slot2.breaker.record_neutral();
                        if let Some(s) = span2.take() {
                            s.finish_err("cancelled (hedged race lost)");
                        }
                        tokio::spawn(async move {
                            if let Ok(Ok((Ok(loser_response), _))) =
                                tokio::time::timeout(Duration::from_secs(5), h2).await
                            {
                                drop(loser_response);
                            }
                        });
                        metrics::record_hedged_win(&c1.config.id, &c2.config.id);
                    }
                    return Ok(Some(CandidateOutcome {
                        backend: c1,
                        profile_id: slot1.id.clone(),
                        response,
                        capture: capture1,
                        exec_span: span1.take(),
                        fallback_used: false,
                        handoff_from: handoff_from1.clone(),
                        handoff_reason: handoff_reason1,
                    }));
                }
                RaceEvt::Two(Ok((Ok(response), elapsed))) => {
                    self.record_ttft(&c2.config.id, elapsed);
                    slot2.breaker.record_success();
                    c2.breaker.record_success();
                    if let Some(pid) = &slot2.id {
                        self.sessions
                            .set_profile(&base_request.session_key, &c2.config.id, pid);
                    }
                    if let Some(h1) = opt1.take() {
                        cancel1.cancel();
                        c1.breaker.record_neutral();
                        slot1.breaker.record_neutral();
                        if let Some(s) = span1.take() {
                            s.finish_err("cancelled (hedged race lost)");
                        }
                        tokio::spawn(async move {
                            if let Ok(Ok((Ok(loser_response), _))) =
                                tokio::time::timeout(Duration::from_secs(5), h1).await
                            {
                                drop(loser_response);
                            }
                        });
                        metrics::record_hedged_win(&c2.config.id, &c1.config.id);
                    }
                    return Ok(Some(CandidateOutcome {
                        backend: c2,
                        profile_id: slot2.id.clone(),
                        response,
                        capture: capture2,
                        exec_span: span2.take(),
                        fallback_used: true,
                        handoff_from: handoff_from2.clone(),
                        handoff_reason: handoff_reason2,
                    }));
                }
                RaceEvt::One(join_res) => {
                    if let Some(s) = span1.take() {
                        s.finish_err("hedged primary error");
                    }
                    let err = match join_res {
                        Ok((Err(e), _)) => e,
                        Err(e) => GatewayError::Internal(format!("hedged task failed: {e}")),
                        _ => unreachable!(),
                    };
                    if !base_request.cancel.is_cancelled() {
                        record_candidate_failure(&c1, Some(&slot1), &err, attempts);
                        *previous_backend = Some(c1.config.id.clone());
                        *last_error = Some(err);
                    } else {
                        c1.breaker.record_neutral();
                        slot1.breaker.record_neutral();
                        *last_error = Some(err);
                    }
                }
                RaceEvt::Two(join_res) => {
                    if let Some(s) = span2.take() {
                        s.finish_err("hedged backup error");
                    }
                    let err = match join_res {
                        Ok((Err(e), _)) => e,
                        Err(e) => GatewayError::Internal(format!("hedged task failed: {e}")),
                        _ => unreachable!(),
                    };
                    if !base_request.cancel.is_cancelled() {
                        record_candidate_failure(&c2, Some(&slot2), &err, attempts);
                        *previous_backend = Some(c2.config.id.clone());
                        *last_error = Some(err);
                    } else {
                        c2.breaker.record_neutral();
                        slot2.breaker.record_neutral();
                        *last_error = Some(err);
                    }
                }
            }
        }
        Ok(None)
    }

    fn finalize_winner(
        &self,
        outcome: CandidateOutcome,
        session_key: &str,
        previous_backend: Option<&str>,
        attempts: Vec<String>,
        dispatch_span: Option<ActiveSpan>,
    ) -> RoutedResponse {
        let backend_id = outcome.backend.config.id.clone();
        let profile_id = outcome.profile_id;
        let fallback_used = outcome.fallback_used;
        let mut exec_span = outcome.exec_span;
        let mut dispatch_span = dispatch_span;
        if let Some(span) = &mut exec_span {
            span.set_bool("vouch.fallback", fallback_used);
            if let Some(pid) = &profile_id {
                span.set_str("vouch.profile", pid);
            }
        }
        if let Some(span) = &mut dispatch_span {
            span.set_str("vouch.backend", &backend_id);
            span.set_bool("vouch.fallback", fallback_used);
            if let Some(pid) = &profile_id {
                span.set_str("vouch.profile", pid);
            }
        }

        let final_response = match outcome.response {
            DriverResponse::Stream(stream) => DriverResponse::Stream(wrap_stream_session_commit(
                stream,
                self.sessions.clone(),
                session_key.to_string(),
                backend_id.clone(),
                outcome.capture,
                exec_span,
                dispatch_span,
            )),
            DriverResponse::Complete(completion) => {
                let captured = outcome.capture.lock().take();
                self.sessions
                    .record_backend_session(session_key, &backend_id, captured);
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
        self.sessions.set(session_key, &backend_id);
        if let Some(previous) = previous_backend {
            metrics::record_fallback(previous, &backend_id);
        }
        let handoff_from = outcome.handoff_from;
        let handoff_reason = outcome.handoff_reason.map(|r| r.as_str().to_string());
        if let (Some(from), Some(reason)) = (handoff_from.as_deref(), outcome.handoff_reason) {
            metrics::record_handoff(from, &backend_id, reason.as_str());
            if let Some(mem) = self.memory.load().as_ref() {
                mem.ledger()
                    .record_handoff(session_key, from, &backend_id, reason);
                let ttl = Duration::from_secs(self.session_ttl_secs());
                if mem.config().sync_to_trace
                    && let Some(entry) = mem.ledger().get(session_key, ttl)
                {
                    let summary = format_trace_session_summary(&entry);
                    mem.trace()
                        .record_session_checkpoint_async(None, summary, entry.touched_files);
                }
            }
        }
        RoutedResponse {
            backend_id,
            profile_id,
            fallback_used,
            response: final_response,
            attempts,
            handoff_from,
            handoff_reason,
        }
    }

    /// Execute an embeddings request across the resolved candidate chain with
    /// circuit-breaker awareness and fallback.
    pub async fn execute_embeddings(
        &self,
        model: &str,
        allowed_backends: Option<&[String]>,
        request: crate::driver::EmbeddingRequest,
    ) -> Result<RoutedEmbeddings> {
        let table = self.table.load_full();
        let entry = table
            .routes
            .get(model)
            .ok_or_else(|| GatewayError::NotFound(format!("unknown model route {model:?}")))?;
        let candidates: Vec<Arc<BackendHandle>> = if let Some(allowed) = allowed_backends {
            let compiled = crate::config::ModelAllowlist::compile(allowed)?;
            entry
                .iter()
                .filter(|b| compiled.matches(&b.config.id))
                .cloned()
                .collect()
        } else {
            entry.clone()
        };
        if candidates.is_empty() {
            return Err(GatewayError::Forbidden(format!(
                "token is not allowed to access any backend for model {model:?}"
            )));
        }

        let mut attempts = Vec::new();
        let mut last_error: Option<GatewayError> = None;
        let mut fallback_used = false;
        let mut previous_backend: Option<String> = None;

        for (index, backend) in candidates.iter().enumerate() {
            if !backend.breaker.allow_request() {
                attempts.push(format!(
                    "{}: circuit {}",
                    backend.config.id,
                    backend.breaker.state().as_str()
                ));
                fallback_used = true;
                continue;
            }
            let started = Instant::now();
            match backend.driver.embed(request.clone()).await {
                Ok(result) => {
                    self.record_ttft(&backend.config.id, started.elapsed());
                    backend.breaker.record_success();
                    if let Some(prev) = previous_backend.as_deref() {
                        metrics::record_fallback(prev, &backend.config.id);
                    }
                    return Ok(RoutedEmbeddings {
                        backend_id: backend.config.id.clone(),
                        profile_id: None,
                        fallback_used: fallback_used || index > 0,
                        result,
                        attempts,
                    });
                }
                Err(error) => {
                    record_candidate_failure(backend, None, &error, &mut attempts);
                    let retryable = error.is_retryable();
                    last_error = Some(error);
                    if !retryable {
                        break;
                    }
                    previous_backend = Some(backend.config.id.clone());
                    fallback_used = true;
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            GatewayError::NoHealthyBackend(format!(
                "all backends for embeddings model {model:?} failed ({})",
                attempts.join("; ")
            ))
        }))
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

fn profile_breaker_key(backend_id: &str, profile_id: &str) -> String {
    format!("{backend_id}::profile::{profile_id}")
}

struct ProfileSlot {
    id: Option<String>,
    env: BTreeMap<String, String>,
    breaker: Arc<CircuitBreaker>,
}

struct CandidateOutcome {
    backend: Arc<BackendHandle>,
    profile_id: Option<String>,
    response: DriverResponse,
    capture: Arc<parking_lot::Mutex<Option<String>>>,
    exec_span: Option<ActiveSpan>,
    fallback_used: bool,
    handoff_from: Option<String>,
    handoff_reason: Option<HandoffReason>,
}

fn record_candidate_failure(
    backend: &BackendHandle,
    profile: Option<&ProfileSlot>,
    error: &GatewayError,
    attempts: &mut Vec<String>,
) {
    let prof_label = profile.and_then(|s| s.id.as_deref());
    if let Some(pl) = prof_label {
        attempts.push(format!("{}:{}: {error}", backend.config.id, pl));
    } else {
        attempts.push(format!("{}: {error}", backend.config.id));
    }
    if error.counts_as_backend_failure() {
        let details = FailureDetails {
            rate_limited: error.is_rate_limited(),
            auth_expired: error.is_auth_expired(),
            retry_after: error.upstream_rate_limit_retry_after(),
        };
        if let Some(slot) = profile
            && slot.id.is_some()
        {
            slot.breaker
                .record_failure_detailed(error.to_string(), details);
        } else {
            backend
                .breaker
                .record_failure_detailed(error.to_string(), details);
        }
    } else {
        if let Some(slot) = profile {
            slot.breaker.record_neutral();
        }
        backend.breaker.record_neutral();
    }
}

async fn execute_with_repair(
    driver: &Arc<dyn Driver>,
    mut request: ExecutionRequest,
    schema_repair_retries: u8,
    kind: BackendKind,
) -> Result<DriverResponse> {
    let can_repair =
        !request.stream && request.response_format.is_some() && kind != BackendKind::Http;
    let first = match driver.execute(request.clone()).await {
        Ok(DriverResponse::Stream(stream)) => {
            prime_stream(stream).await.map(DriverResponse::Stream)
        }
        other => other,
    };
    let Err(mut last_err) = first else {
        return first;
    };
    if !can_repair || schema_repair_retries == 0 || !is_schema_validation_error(&last_err) {
        return Err(last_err);
    }

    for _ in 0..schema_repair_retries.min(2) {
        if request.cancel.is_cancelled() {
            return Err(GatewayError::ClientDisconnected);
        }
        request.backend_session_id = None;
        request.delta_messages = None;
        request.messages.push(ChatMessage::text(
            "user",
            format!(
                "Your previous response failed structured output schema validation with the following error:\n{last_err}\n\nPlease reply again with ONLY valid JSON conforming strictly to the requested format/schema, without markdown fences or prose."
            ),
        ));
        request.prompt = crate::driver::render_prompt(request.system.as_deref(), &request.messages);

        match driver.execute(request.clone()).await {
            Ok(DriverResponse::Stream(stream)) => {
                match prime_stream(stream).await.map(DriverResponse::Stream) {
                    Ok(resp) => return Ok(resp),
                    Err(err) => {
                        last_err = err;
                        if !is_schema_validation_error(&last_err) {
                            return Err(last_err);
                        }
                    }
                }
            }
            Ok(resp) => return Ok(resp),
            Err(err) => {
                last_err = err;
                if !is_schema_validation_error(&last_err) {
                    return Err(last_err);
                }
            }
        }
    }
    Err(last_err)
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
