//! Model routing, ordered fallback, circuit breaking, and sticky affinity.
//!
//! A public model maps to an ordered chain of backend handles. The router
//! tries the sticky backend first when the session already has affinity, then
//! walks the chain. A backend is skipped when its circuit is open or when its
//! concurrency permit cannot be acquired; a failure before a stream starts
//! transparently fails over to the next candidate.

pub mod circuit_breaker;

use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use arc_swap::ArcSwap;
use dashmap::DashMap;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::{
    config::{BackendConfig, BackendKind, GatewayConfig},
    driver::{
        Driver, DriverResponse, ExecutionRequest, WorkspaceMode, cli_pipe::CliPipeDriver,
        http_proxy::HttpProxyDriver, pty_runner::PtyRunnerDriver, workspace::WorkspaceManager,
    },
    error::{GatewayError, Result},
    metrics,
};

use self::circuit_breaker::{BreakerSettings, CircuitBreaker, CircuitState};

/// Sticky backend selected for a session.
#[derive(Debug, Clone)]
struct AffinityEntry {
    backend_id: String,
    last_used: Instant,
}

/// Thread-safe sticky-session map with TTL eviction.
#[derive(Debug, Default)]
pub struct SessionAffinity {
    entries: DashMap<String, AffinityEntry>,
}

impl SessionAffinity {
    /// Create an empty map.
    pub fn new() -> Self {
        Self::default()
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

    /// Record a successful backend for a session.
    pub fn set(&self, key: &str, backend_id: &str) {
        self.entries.insert(
            key.to_string(),
            AffinityEntry {
                backend_id: backend_id.to_string(),
                last_used: Instant::now(),
            },
        );
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

/// Immutable routing snapshot.
#[derive(Debug, Clone, Default)]
pub struct RouterTable {
    routes: HashMap<String, Vec<Arc<BackendHandle>>>,
    order: Vec<Arc<BackendHandle>>,
    default_model: Option<String>,
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
}

impl Router {
    /// Build a router from validated configuration.
    pub fn new(config: &GatewayConfig, sessions: Arc<SessionAffinity>) -> Result<Self> {
        let router = Self {
            table: ArcSwap::from_pointee(RouterTable::default()),
            sessions,
            breakers: DashMap::new(),
            ttl_secs: Arc::new(AtomicU64::new(config.defaults.session_ttl_secs.max(1))),
        };
        router.rebuild(config)?;
        Ok(router)
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
                BackendKind::Cli => {
                    Arc::new(CliPipeDriver::new(backend.clone(), &config.defaults)?)
                }
                BackendKind::Pty => {
                    Arc::new(PtyRunnerDriver::new(backend.clone(), &config.defaults)?)
                }
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

        self.ttl_secs
            .store(config.defaults.session_ttl_secs.max(1), Ordering::Relaxed);
        self.table.store(Arc::new(RouterTable {
            routes,
            order,
            default_model: config.default_model().map(ToString::to_string),
        }));
        Ok(())
    }

    /// Execute a request with fallback and affinity.
    pub async fn execute(
        &self,
        request: ExecutionRequest,
        workspace_mode: WorkspaceMode,
        workspaces: &WorkspaceManager,
    ) -> Result<RoutedResponse> {
        let table = self.table.load_full();
        let chain = table.routes.get(&request.model).cloned().ok_or_else(|| {
            GatewayError::NotFound(format!(
                "model {:?} is not configured; available models: {}",
                request.model,
                table.routes.keys().cloned().collect::<Vec<_>>().join(", ")
            ))
        })?;
        if chain.is_empty() {
            return Err(GatewayError::NoHealthyBackend(format!(
                "route {:?} has no backends",
                request.model
            )));
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
            let isolate = match workspace_mode {
                WorkspaceMode::Worktree => true,
                WorkspaceMode::Auto => backend.config.isolate_workspace,
                WorkspaceMode::None => false,
            };
            if isolate && backend.config.kind != BackendKind::Http {
                match workspaces.acquire(&request.session_key).await {
                    Ok(entry) => attempt.workdir = Some(entry.path.clone()),
                    Err(error) => {
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

            match backend.driver.execute(attempt).await {
                Ok(response) => {
                    backend.breaker.record_success();
                    self.sessions.set(&request.session_key, &backend.config.id);
                    let fallback_used = index > 0;
                    if let Some(previous) = previous_backend.as_deref() {
                        metrics::record_fallback(previous, &backend.config.id);
                    }
                    return Ok(RoutedResponse {
                        backend_id: backend.config.id.clone(),
                        fallback_used,
                        response,
                        attempts,
                    });
                }
                Err(error) => {
                    if error.counts_as_backend_failure() {
                        backend.breaker.record_failure(
                            error.to_string(),
                            matches!(error, GatewayError::UpstreamRateLimited(_)),
                        );
                    } else {
                        backend.breaker.record_neutral();
                    }
                    attempts.push(format!("{}: {error}", backend.config.id));
                    if !error.is_retryable() {
                        return Err(error);
                    }
                    previous_backend = Some(backend.config.id.clone());
                    last_error = Some(error);
                }
            }
        }

        match last_error {
            Some(error) => Err(error),
            None => Err(GatewayError::NoHealthyBackend(format!(
                "no eligible backend for model {:?}: {}",
                request.model,
                attempts.join("; ")
            ))),
        }
    }

    /// Default model, when configured or unambiguous.
    pub fn default_model(&self) -> Option<String> {
        self.table.load().default_model.clone()
    }

    /// Available public models.
    pub fn models(&self) -> Vec<String> {
        let table = self.table.load();
        let mut models: Vec<String> = table.routes.keys().cloned().collect();
        models.sort();
        models
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

fn backend_status(backend: &BackendHandle) -> BackendStatus {
    let snapshot = backend.breaker.snapshot();
    BackendStatus {
        id: backend.config.id.clone(),
        kind: backend.config.kind,
        max_concurrency: backend.config.max_concurrency,
        circuit: snapshot.state,
        consecutive_failures: snapshot.consecutive_failures,
        open_remaining_secs: snapshot.open_remaining_secs,
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
}
