//! Typed configuration schema, validation, and hot-reload storage.
//!
//! The on-disk format is TOML. Unknown fields are rejected so that a typo in a
//! production configuration fails loudly instead of being silently ignored.
//! [`ConfigStore`] wraps the active configuration in an [`arc_swap::ArcSwap`],
//! allowing readers to keep using a consistent snapshot while a reload swaps
//! in a new one.

use std::{
    collections::{BTreeMap, HashSet},
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
};

use arc_swap::ArcSwap;
use globset::{Glob, GlobSet, GlobSetBuilder};
use serde::{Deserialize, Serialize};

use crate::error::{GatewayError, Result};

/// Root configuration document.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GatewayConfig {
    /// Listener, timeout, and logging settings.
    #[serde(default)]
    pub server: ServerConfig,
    /// Values inherited by backends and requests.
    #[serde(default)]
    pub defaults: DefaultsConfig,
    /// Circuit-breaker tuning.
    #[serde(default)]
    pub resilience: ResilienceConfig,
    /// Optional git worktree isolation.
    #[serde(default)]
    pub workspace: WorkspaceConfig,
    /// Local credential/CLI discovery settings used by `vouch doctor`.
    #[serde(default)]
    pub discovery: DiscoveryConfig,
    /// Egress backends.
    #[serde(default)]
    pub backends: Vec<BackendConfig>,
    /// Public model name to ordered backend-id fallback chain.
    #[serde(default)]
    pub routes: BTreeMap<String, Vec<String>>,
    /// Optional virtual ingress tokens.
    #[serde(default)]
    pub tokens: Vec<TokenConfig>,
}

/// Network listener and process-wide runtime settings.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    /// TCP address to bind. Defaults to loopback only.
    #[serde(default = "default_tcp_listen")]
    pub tcp_listen: SocketAddr,
    /// Optional Unix domain socket path. Created with mode `0600`.
    #[serde(default)]
    pub unix_socket: Option<PathBuf>,
    /// Deadline for non-streaming requests, including workspace setup.
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
    /// Maximum lifetime of a single streaming response.
    #[serde(default = "default_stream_timeout_secs")]
    pub stream_timeout_secs: u64,
    /// Interval between SSE keep-alive comments. Zero disables keep-alives.
    #[serde(default = "default_keep_alive_secs")]
    pub keep_alive_interval_secs: u64,
    /// Maximum accepted request body size in bytes.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
    /// Tracing filter directive. `RUST_LOG` takes precedence when set.
    #[serde(default = "default_log_level")]
    pub log_level: String,
    /// Whether to allow cross-origin browser clients. Defaults to `false`.
    #[serde(default)]
    pub cors_permissive: bool,
    /// Serve the embedded read-only operational dashboard at `GET /ui`. Defaults to `false`.
    #[serde(default)]
    pub enable_ui: bool,
}

/// Defaults applied when a backend or request omits a value.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DefaultsConfig {
    /// Model used when a request omits `model`.
    #[serde(default)]
    pub model: Option<String>,
    /// Idle lifetime of sticky sessions and isolated workspaces.
    #[serde(default = "default_session_ttl_secs")]
    pub session_ttl_secs: u64,
    /// Default per-backend concurrency limit.
    #[serde(default = "default_max_concurrency")]
    pub max_concurrency: usize,
    /// Default time to wait for a backend permit before failing over.
    #[serde(default = "default_queue_timeout_ms")]
    pub queue_timeout_ms: u64,
}

/// Circuit-breaker tuning shared by every backend.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResilienceConfig {
    /// Consecutive backend failures required to open a circuit.
    #[serde(default = "default_failure_threshold")]
    pub failure_threshold: u32,
    /// Cooldown before an open circuit admits a half-open probe.
    #[serde(default = "default_reset_timeout_secs")]
    pub reset_timeout_secs: u64,
    /// Maximum concurrent probes while half-open.
    #[serde(default = "default_half_open_max_probes")]
    pub half_open_max_probes: u32,
    /// Open immediately when an upstream returns HTTP 429.
    #[serde(default = "default_true")]
    pub open_on_rate_limit: bool,
}

/// Optional git worktree isolation for CLI/PTY backends.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceConfig {
    /// Master switch. Enabling requires `repo_path`.
    #[serde(default)]
    pub enabled: bool,
    /// Repository from which worktrees are created.
    #[serde(default)]
    pub repo_path: Option<PathBuf>,
    /// Directory that holds ephemeral worktrees. Defaults to the system temp
    /// directory under `vouch-worktrees`.
    #[serde(default)]
    pub root: Option<PathBuf>,
    /// Idle time after which an unused worktree is removed.
    #[serde(default = "default_workspace_ttl_secs")]
    pub ttl_secs: u64,
    /// Git executable.
    #[serde(default = "default_git_binary")]
    pub git_binary: String,
    /// Remove all worktrees during graceful shutdown.
    #[serde(default = "default_true")]
    pub cleanup_on_shutdown: bool,
}

/// `vouch doctor` discovery settings.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryConfig {
    /// Inspect known local auth files.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// TCP-probe configured HTTP backends.
    #[serde(default = "default_true")]
    pub probe_ports: bool,
    /// Additional credential files to report on.
    #[serde(default)]
    pub extra_auth_paths: Vec<PathBuf>,
    /// Per-backend timeout in seconds for CLI/PTY smoke probes in `vouch doctor`.
    #[serde(default = "default_probe_timeout_secs")]
    pub probe_timeout_secs: u64,
}

/// Egress backend definition.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BackendConfig {
    /// Stable identifier referenced by routes.
    pub id: String,
    /// Transport implementation.
    pub kind: BackendKind,
    /// Maximum simultaneous executions. Protects consumer subscriptions from
    /// parallel rate-limit bans.
    #[serde(default = "default_max_concurrency")]
    pub max_concurrency: usize,
    /// Per-execution deadline.
    #[serde(default = "default_backend_timeout_secs")]
    pub timeout_secs: u64,
    /// How long to wait for a permit before treating the backend as
    /// saturated. Inherits `defaults.queue_timeout_ms` when absent.
    #[serde(default)]
    pub queue_timeout_ms: Option<u64>,
    /// Create an isolated git worktree for every session.
    #[serde(default)]
    pub isolate_workspace: bool,
    /// Model name sent upstream. Defaults to the requested public model.
    #[serde(default)]
    pub model: Option<String>,

    // ---- HTTP proxy settings -------------------------------------------------
    /// Base URL of the local daemon, e.g. `http://127.0.0.1:8645`.
    #[serde(default)]
    pub base_url: Option<String>,
    /// Chat path appended to `base_url`. Defaults to
    /// `/v1/chat/completions`.
    #[serde(default)]
    pub upstream_path: Option<String>,
    /// Health path appended to `base_url` for `vouch doctor`.
    #[serde(default)]
    pub health_path: Option<String>,
    /// Environment variable holding a bearer token for the upstream.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Additional upstream headers.
    #[serde(default)]
    pub extra_headers: BTreeMap<String, String>,

    // ---- CLI / PTY settings --------------------------------------------------
    /// Executable name or absolute path.
    #[serde(default)]
    pub command: Option<String>,
    /// Arguments. `{prompt}`, `{model}`, and `{session_id}` are substituted.
    #[serde(default)]
    pub args: Vec<String>,
    /// Pipe the rendered prompt to stdin.
    #[serde(default)]
    pub stdin: bool,
    /// Environment variables injected into the child.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Working directory. Ignored when workspace isolation is active.
    #[serde(default)]
    pub workdir: Option<PathBuf>,
    /// How stdout is interpreted.
    #[serde(default)]
    pub output_mode: OutputMode,
    /// For PTY backends that read stdin, send EOT after the prompt.
    #[serde(default = "default_true")]
    pub pty_send_eof: bool,
    /// Prompts larger than this (when `stdin = false`) spill to a `0600` temp
    /// file to avoid `ARG_MAX` (`E2BIG`) execve failures. Default: `65536`.
    #[serde(default = "default_max_inline_prompt_bytes")]
    pub max_inline_prompt_bytes: usize,
    /// Optional idle stall watchdog timeout in seconds (reset on every
    /// stdout/stderr byte received). Default: unset.
    #[serde(default)]
    pub idle_timeout_secs: Option<u64>,
    /// PTY width in columns (`portable-pty`). Default: `160`.
    #[serde(default = "default_pty_cols")]
    pub pty_cols: u16,
    /// PTY height in rows (`portable-pty`). Default: `48`.
    #[serde(default = "default_pty_rows")]
    pub pty_rows: u16,
    /// Automatic answers injected into PTY stdin when matching interactive
    /// prompts appear in PTY output.
    #[serde(default)]
    pub pty_auto_respond: Vec<AutoRespondRule>,
    /// Alternative argument template for continuing native multi-turn sessions
    /// (`{prompt}`, `{backend_session_id}`, `{session_id}`, `{model}`,
    /// `{request_id}`).
    #[serde(default)]
    pub resume_args: Vec<String>,
    /// Regex to capture a backend-native session identifier from stdout/stderr
    /// during turn 1 (prefers capture group 1 if present, otherwise full match).
    #[serde(default)]
    pub session_id_regex: Option<String>,
}

/// Interactive prompt auto-response rule for PTY backends.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AutoRespondRule {
    /// Substring or regular expression matched against stripped PTY output.
    pub pattern: String,
    /// Text sent to PTY stdin when `pattern` matches.
    pub response: String,
}

/// Egress transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    /// HTTP/SSE proxy to a local daemon.
    Http,
    /// Headless stdin/stdout subprocess.
    Cli,
    /// Pseudo-terminal subprocess.
    Pty,
}

impl BackendKind {
    /// Human-readable label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Http => "http",
            Self::Cli => "cli",
            Self::Pty => "pty",
        }
    }
}

/// Interpretation of a CLI/PTY backend's stdout.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputMode {
    /// Emit stdout as plain text (ANSI stripped).
    #[default]
    Text,
    /// Parse the complete stdout as a JSON object.
    Json,
    /// Parse newline-delimited JSON objects.
    JsonLines,
}

/// Virtual ingress credential.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TokenConfig {
    /// Secret presented by the client.
    pub token: String,
    /// Label used in logs and metrics. Defaults to a stable hash prefix.
    #[serde(default)]
    pub name: Option<String>,
    /// Sliding-window requests-per-minute limit.
    #[serde(default)]
    pub rpm: Option<u32>,
    /// Glob patterns of models this token may call. Empty means all.
    #[serde(default)]
    pub allowed_models: Vec<String>,
}

/// Compiled model allowlist.
#[derive(Debug)]
pub struct ModelAllowlist {
    set: GlobSet,
    patterns: Vec<String>,
}

impl ModelAllowlist {
    /// Compile glob patterns.
    pub fn compile(patterns: &[String]) -> Result<Self> {
        let mut builder = GlobSetBuilder::new();
        for pattern in patterns {
            let glob = Glob::new(pattern).map_err(|error| {
                GatewayError::Config(format!("invalid allowed_models glob {pattern:?}: {error}"))
            })?;
            builder.add(glob);
        }
        let set = builder.build().map_err(|error| {
            GatewayError::Config(format!("invalid allowed_models glob set: {error}"))
        })?;
        Ok(Self {
            set,
            patterns: patterns.to_vec(),
        })
    }

    /// Whether `model` matches this allowlist.
    pub fn matches(&self, model: &str) -> bool {
        self.set.is_match(model)
    }

    /// Original patterns, for diagnostics.
    pub fn patterns(&self) -> &[String] {
        &self.patterns
    }
}

impl GatewayConfig {
    /// Load and validate a TOML configuration file.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|error| {
            GatewayError::Config(format!("cannot read {}: {error}", path.display()))
        })?;
        Self::from_toml_str(&text)
    }

    /// Parse and validate a TOML document.
    pub fn from_toml_str(text: &str) -> Result<Self> {
        let config: Self = toml::from_str(text)
            .map_err(|error| GatewayError::Config(format!("invalid TOML: {error}")))?;
        config.validate()?;
        Ok(config)
    }

    /// Validate cross-references and invariants.
    pub fn validate(&self) -> Result<()> {
        if self.server.max_body_bytes == 0 {
            return Err(GatewayError::Config(
                "server.max_body_bytes must be greater than zero".into(),
            ));
        }
        if self.server.request_timeout_secs == 0 {
            return Err(GatewayError::Config(
                "server.request_timeout_secs must be greater than zero".into(),
            ));
        }
        if self.server.stream_timeout_secs == 0 {
            return Err(GatewayError::Config(
                "server.stream_timeout_secs must be greater than zero".into(),
            ));
        }
        if self.backends.is_empty() {
            return Err(GatewayError::Config(
                "at least one [[backends]] entry is required".into(),
            ));
        }
        if self.routes.is_empty() {
            return Err(GatewayError::Config(
                "at least one [routes] entry is required".into(),
            ));
        }
        if self.defaults.max_concurrency == 0 {
            return Err(GatewayError::Config(
                "defaults.max_concurrency must be greater than zero".into(),
            ));
        }
        if self.resilience.failure_threshold == 0 {
            return Err(GatewayError::Config(
                "resilience.failure_threshold must be greater than zero".into(),
            ));
        }
        if self.resilience.reset_timeout_secs == 0 {
            return Err(GatewayError::Config(
                "resilience.reset_timeout_secs must be greater than zero".into(),
            ));
        }
        if self.resilience.half_open_max_probes == 0 {
            return Err(GatewayError::Config(
                "resilience.half_open_max_probes must be greater than zero".into(),
            ));
        }
        if self.discovery.probe_timeout_secs == 0 {
            return Err(GatewayError::Config(
                "discovery.probe_timeout_secs must be greater than zero".into(),
            ));
        }

        let mut ids = HashSet::new();
        for backend in &self.backends {
            backend.validate(self)?;
            if !ids.insert(backend.id.clone()) {
                return Err(GatewayError::Config(format!(
                    "duplicate backend id {:?}",
                    backend.id
                )));
            }
        }

        for (model, chain) in &self.routes {
            if model.trim().is_empty() {
                return Err(GatewayError::Config(
                    "route model names must not be empty".into(),
                ));
            }
            if chain.is_empty() {
                return Err(GatewayError::Config(format!(
                    "route {model:?} has an empty backend chain"
                )));
            }
            let mut seen = HashSet::new();
            for id in chain {
                if !ids.contains(id) {
                    return Err(GatewayError::Config(format!(
                        "route {model:?} references unknown backend {id:?}"
                    )));
                }
                if !seen.insert(id) {
                    return Err(GatewayError::Config(format!(
                        "route {model:?} lists backend {id:?} more than once"
                    )));
                }
            }
        }

        if let Some(model) = &self.defaults.model
            && !self.routes.contains_key(model)
        {
            return Err(GatewayError::Config(format!(
                "defaults.model {model:?} is not a configured route"
            )));
        }

        if self.workspace.enabled {
            if self.workspace.repo_path.is_none() {
                return Err(GatewayError::Config(
                    "workspace.enabled requires workspace.repo_path".into(),
                ));
            }
            if self.workspace.ttl_secs == 0 {
                return Err(GatewayError::Config(
                    "workspace.ttl_secs must be greater than zero".into(),
                ));
            }
        } else if self
            .backends
            .iter()
            .any(|backend| backend.isolate_workspace)
        {
            return Err(GatewayError::Config(
                "a backend sets isolate_workspace but workspace.enabled is false".into(),
            ));
        }

        let mut tokens = HashSet::new();
        for token in &self.tokens {
            if token.token.len() < 8 {
                return Err(GatewayError::Config(
                    "virtual tokens must be at least 8 characters".into(),
                ));
            }
            if !tokens.insert(token.token.clone()) {
                return Err(GatewayError::Config("duplicate virtual token value".into()));
            }
            if token.rpm == Some(0) {
                return Err(GatewayError::Config(
                    "token rpm must be greater than zero when set".into(),
                ));
            }
            ModelAllowlist::compile(&token.allowed_models)?;
        }

        Ok(())
    }

    /// Resolve the model used when a request omits one.
    pub fn default_model(&self) -> Option<&str> {
        self.defaults.model.as_deref().or_else(|| {
            if self.routes.len() == 1 {
                self.routes.keys().next().map(String::as_str)
            } else {
                None
            }
        })
    }

    /// Locate a backend by id.
    pub fn backend(&self, id: &str) -> Option<&BackendConfig> {
        self.backends.iter().find(|backend| backend.id == id)
    }

    /// Values that must never appear in logs or error messages.
    pub fn secrets(&self) -> Vec<String> {
        let mut secrets = Vec::new();
        for token in &self.tokens {
            secrets.push(token.token.clone());
        }
        for backend in &self.backends {
            for (key, value) in &backend.env {
                let key = key.to_ascii_uppercase();
                let looks_secret = key.contains("KEY")
                    || key.contains("TOKEN")
                    || key.contains("SECRET")
                    || key.contains("PASSWORD")
                    || key.contains("AUTH");
                if looks_secret || looks_like_secret(value) {
                    secrets.push(value.clone());
                }
            }
            for (key, value) in &backend.extra_headers {
                let key = key.to_ascii_lowercase();
                if key.contains("authorization")
                    || key.contains("api-key")
                    || key.contains("token")
                    || looks_like_secret(value)
                {
                    secrets.push(value.clone());
                }
            }
            if let Some(env) = &backend.api_key_env
                && let Ok(value) = std::env::var(env)
                && !value.is_empty()
            {
                secrets.push(value);
            }
        }
        secrets.sort();
        secrets.dedup();
        secrets
    }
}

impl BackendConfig {
    fn validate(&self, config: &GatewayConfig) -> Result<()> {
        if self.id.trim().is_empty() {
            return Err(GatewayError::Config("backend id must not be empty".into()));
        }
        if self.max_concurrency == 0 {
            return Err(GatewayError::Config(format!(
                "backend {:?}: max_concurrency must be greater than zero",
                self.id
            )));
        }
        if self.timeout_secs == 0 {
            return Err(GatewayError::Config(format!(
                "backend {:?}: timeout_secs must be greater than zero",
                self.id
            )));
        }
        if self.queue_timeout_ms == Some(0) && config.defaults.queue_timeout_ms == 0 {
            return Err(GatewayError::Config(format!(
                "backend {:?}: queue timeout must be greater than zero",
                self.id
            )));
        }
        if self.max_inline_prompt_bytes == 0 {
            return Err(GatewayError::Config(format!(
                "backend {:?}: max_inline_prompt_bytes must be greater than zero",
                self.id
            )));
        }
        if self.idle_timeout_secs == Some(0) {
            return Err(GatewayError::Config(format!(
                "backend {:?}: idle_timeout_secs must be greater than zero when set",
                self.id
            )));
        }
        if self.pty_cols == 0 {
            return Err(GatewayError::Config(format!(
                "backend {:?}: pty_cols must be greater than zero",
                self.id
            )));
        }
        if self.pty_rows == 0 {
            return Err(GatewayError::Config(format!(
                "backend {:?}: pty_rows must be greater than zero",
                self.id
            )));
        }
        for rule in &self.pty_auto_respond {
            if rule.pattern.is_empty() {
                return Err(GatewayError::Config(format!(
                    "backend {:?}: pty_auto_respond.pattern must not be empty",
                    self.id
                )));
            }
        }
        if let Some(pattern) = &self.session_id_regex {
            regex::Regex::new(pattern).map_err(|error| {
                GatewayError::Config(format!(
                    "backend {:?}: invalid session_id_regex {pattern:?}: {error}",
                    self.id
                ))
            })?;
        }

        match self.kind {
            BackendKind::Http => {
                let base = self.base_url.as_deref().ok_or_else(|| {
                    GatewayError::Config(format!(
                        "backend {:?}: HTTP backends require base_url",
                        self.id
                    ))
                })?;
                let url = reqwest::Url::parse(base).map_err(|error| {
                    GatewayError::Config(format!(
                        "backend {:?}: invalid base_url {base:?}: {error}",
                        self.id
                    ))
                })?;
                if url.scheme() != "http" && url.scheme() != "https" {
                    return Err(GatewayError::Config(format!(
                        "backend {:?}: base_url must use http or https",
                        self.id
                    )));
                }
                if self.command.is_some() {
                    return Err(GatewayError::Config(format!(
                        "backend {:?}: HTTP backends must not set command",
                        self.id
                    )));
                }
                if !self.resume_args.is_empty() || self.session_id_regex.is_some() {
                    return Err(GatewayError::Config(format!(
                        "backend {:?}: HTTP backends must not set resume_args or session_id_regex",
                        self.id
                    )));
                }
            }
            BackendKind::Cli | BackendKind::Pty => {
                if self
                    .command
                    .as_deref()
                    .map(str::trim)
                    .unwrap_or_default()
                    .is_empty()
                {
                    return Err(GatewayError::Config(format!(
                        "backend {:?}: {} backends require command",
                        self.id,
                        self.kind.as_str()
                    )));
                }
                if self.base_url.is_some() {
                    return Err(GatewayError::Config(format!(
                        "backend {:?}: {} backends must not set base_url",
                        self.id,
                        self.kind.as_str()
                    )));
                }
                let templated = self.args.iter().any(|arg| arg.contains("{prompt}"));
                if !templated && !self.stdin {
                    return Err(GatewayError::Config(format!(
                        "backend {:?}: set stdin = true or include {{prompt}} in args",
                        self.id
                    )));
                }
                if !self.resume_args.is_empty()
                    && !self.stdin
                    && !self.resume_args.iter().any(|arg| arg.contains("{prompt}"))
                {
                    return Err(GatewayError::Config(format!(
                        "backend {:?}: resume_args must include {{prompt}} when stdin = false",
                        self.id
                    )));
                }
            }
        }
        Ok(())
    }
}

fn looks_like_secret(value: &str) -> bool {
    value.starts_with("sk-")
        || value.starts_with("Bearer ")
        || value.starts_with("vouch-")
        || value.len() >= 40
}

/// Thread-safe holder for the active configuration snapshot.
#[derive(Debug)]
pub struct ConfigStore {
    path: PathBuf,
    current: ArcSwap<GatewayConfig>,
}

impl ConfigStore {
    /// Create a store from an already validated configuration.
    pub fn new(path: PathBuf, config: GatewayConfig) -> Self {
        Self {
            path,
            current: ArcSwap::from_pointee(config),
        }
    }

    /// Path of the backing TOML file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Current configuration snapshot.
    pub fn load(&self) -> Arc<GatewayConfig> {
        self.current.load_full()
    }

    /// Read, validate, and atomically install a new configuration.
    ///
    /// On failure the previous snapshot remains active.
    pub fn reload(&self) -> Result<Arc<GatewayConfig>> {
        let config = Arc::new(GatewayConfig::load(&self.path)?);
        self.current.store(config.clone());
        Ok(config)
    }

    /// Install an already validated snapshot.
    ///
    /// [`crate::state::AppState`] uses this after rebuilding the router and
    /// token registry so that readers never observe a configuration whose
    /// dependants failed to compile.
    pub fn store_snapshot(&self, config: Arc<GatewayConfig>) {
        self.current.store(config);
    }
}

fn default_tcp_listen() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 8080))
}
fn default_request_timeout_secs() -> u64 {
    300
}
fn default_stream_timeout_secs() -> u64 {
    900
}
fn default_keep_alive_secs() -> u64 {
    15
}
fn default_max_body_bytes() -> usize {
    8 * 1024 * 1024
}
fn default_log_level() -> String {
    "info".into()
}
fn default_session_ttl_secs() -> u64 {
    3600
}
fn default_max_concurrency() -> usize {
    4
}
fn default_queue_timeout_ms() -> u64 {
    250
}
fn default_failure_threshold() -> u32 {
    3
}
fn default_reset_timeout_secs() -> u64 {
    30
}
fn default_half_open_max_probes() -> u32 {
    1
}
fn default_workspace_ttl_secs() -> u64 {
    3600
}
fn default_git_binary() -> String {
    "git".into()
}
fn default_backend_timeout_secs() -> u64 {
    300
}
fn default_true() -> bool {
    true
}
fn default_probe_timeout_secs() -> u64 {
    30
}
fn default_max_inline_prompt_bytes() -> usize {
    65_536
}
fn default_pty_cols() -> u16 {
    160
}
fn default_pty_rows() -> u16 {
    48
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            tcp_listen: default_tcp_listen(),
            unix_socket: None,
            request_timeout_secs: default_request_timeout_secs(),
            stream_timeout_secs: default_stream_timeout_secs(),
            keep_alive_interval_secs: default_keep_alive_secs(),
            max_body_bytes: default_max_body_bytes(),
            log_level: default_log_level(),
            cors_permissive: false,
            enable_ui: false,
        }
    }
}

impl Default for DefaultsConfig {
    fn default() -> Self {
        Self {
            model: None,
            session_ttl_secs: default_session_ttl_secs(),
            max_concurrency: default_max_concurrency(),
            queue_timeout_ms: default_queue_timeout_ms(),
        }
    }
}

impl Default for ResilienceConfig {
    fn default() -> Self {
        Self {
            failure_threshold: default_failure_threshold(),
            reset_timeout_secs: default_reset_timeout_secs(),
            half_open_max_probes: default_half_open_max_probes(),
            open_on_rate_limit: true,
        }
    }
}

impl Default for WorkspaceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            repo_path: None,
            root: None,
            ttl_secs: default_workspace_ttl_secs(),
            git_binary: default_git_binary(),
            cleanup_on_shutdown: true,
        }
    }
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            probe_ports: true,
            extra_auth_paths: Vec::new(),
            probe_timeout_secs: default_probe_timeout_secs(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal(backend_extra: &str) -> String {
        format!(
            r#"
[server]
tcp_listen = "127.0.0.1:0"

[[backends]]
id = "mock"
kind = "cli"
command = "/bin/echo"
stdin = true
{backend_extra}

[routes]
"coding/default" = ["mock"]
"#
        )
    }

    #[test]
    fn parses_minimal_config() {
        let config = GatewayConfig::from_toml_str(&minimal("")).expect("valid config");
        assert_eq!(config.default_model(), Some("coding/default"));
        assert_eq!(config.backends.len(), 1);
    }

    #[test]
    fn rejects_unknown_backend_reference() {
        let text = minimal("").replace(
            r#""coding/default" = ["mock"]"#,
            r#""coding/default" = ["nope"]"#,
        );
        let error = GatewayConfig::from_toml_str(&text).expect_err("must fail");
        assert!(error.to_string().contains("unknown backend"));
    }

    #[test]
    fn rejects_cli_without_prompt_transport() {
        let text = minimal("").replace("stdin = true", "");
        let error = GatewayConfig::from_toml_str(&text).expect_err("must fail");
        assert!(error.to_string().contains("stdin"));
    }

    #[test]
    fn rejects_unknown_fields() {
        let text = minimal("surprise = true");
        let error = GatewayConfig::from_toml_str(&text).expect_err("must fail");
        assert!(error.to_string().contains("surprise"));
    }

    #[test]
    fn secrets_include_tokens_and_env() {
        let text = minimal("").replace(
            "\n[routes]",
            "\n[[tokens]]\ntoken = \"vouch-super-secret-token\"\n\n[routes]",
        );
        let mut config = GatewayConfig::from_toml_str(&text).expect("valid");
        config
            .backends
            .first_mut()
            .expect("backend")
            .env
            .insert("API_KEY".into(), "sk-abcdefghijklmnop".into());
        let secrets = config.secrets();
        assert!(secrets.contains(&"vouch-super-secret-token".to_string()));
        assert!(secrets.contains(&"sk-abcdefghijklmnop".to_string()));
    }

    #[test]
    fn parses_v1_defaults_and_overrides() {
        let base = GatewayConfig::from_toml_str(&minimal("")).expect("valid");
        assert!(!base.server.enable_ui);
        assert_eq!(base.discovery.probe_timeout_secs, 30);

        let text = format!(
            "{}\n[discovery]\nprobe_timeout_secs = 45\n",
            minimal("").replace(
                "tcp_listen = \"127.0.0.1:0\"",
                "tcp_listen = \"127.0.0.1:0\"\nenable_ui = true"
            )
        );
        let custom = GatewayConfig::from_toml_str(&text).expect("valid custom");
        assert!(custom.server.enable_ui);
        assert_eq!(custom.discovery.probe_timeout_secs, 45);
    }

    #[test]
    fn rejects_zero_probe_timeout() {
        let text = format!("{}\n[discovery]\nprobe_timeout_secs = 0\n", minimal(""));
        let error = GatewayConfig::from_toml_str(&text).expect_err("must fail");
        assert!(error.to_string().contains("probe_timeout_secs"));
    }

    #[test]
    fn parses_v2_defaults_and_overrides() {
        let base = GatewayConfig::from_toml_str(&minimal("")).expect("valid v1 config");
        let b = &base.backends[0];
        assert_eq!(b.max_inline_prompt_bytes, 65_536);
        assert_eq!(b.idle_timeout_secs, None);
        assert_eq!(b.pty_cols, 160);
        assert_eq!(b.pty_rows, 48);
        assert!(b.pty_auto_respond.is_empty());
        assert!(b.resume_args.is_empty());
        assert_eq!(b.session_id_regex, None);

        let override_text = minimal(
            r#"
max_inline_prompt_bytes = 4096
idle_timeout_secs = 15
pty_cols = 200
pty_rows = 60
resume_args = ["--resume", "{backend_session_id}", "{prompt}"]
session_id_regex = "session=([a-z0-9-]+)"

[[backends.pty_auto_respond]]
pattern = "Continue? [y/N]"
response = "y"
"#,
        );
        let custom = GatewayConfig::from_toml_str(&override_text).expect("valid v2 overrides");
        let b2 = &custom.backends[0];
        assert_eq!(b2.max_inline_prompt_bytes, 4096);
        assert_eq!(b2.idle_timeout_secs, Some(15));
        assert_eq!(b2.pty_cols, 200);
        assert_eq!(b2.pty_rows, 60);
        assert_eq!(b2.pty_auto_respond.len(), 1);
        assert_eq!(b2.resume_args.len(), 3);
        assert_eq!(b2.session_id_regex.as_deref(), Some("session=([a-z0-9-]+)"));
    }

    #[test]
    fn rejects_invalid_v2_fields() {
        let err_idle = GatewayConfig::from_toml_str(&minimal("idle_timeout_secs = 0"))
            .expect_err("zero idle timeout must fail");
        assert!(err_idle.to_string().contains("idle_timeout_secs"));

        let err_regex = GatewayConfig::from_toml_str(&minimal("session_id_regex = \"[unclosed\""))
            .expect_err("bad regex must fail");
        assert!(err_regex.to_string().contains("session_id_regex"));

        let err_pty =
            GatewayConfig::from_toml_str(&minimal("pty_cols = 0")).expect_err("zero pty cols");
        assert!(err_pty.to_string().contains("pty_cols"));

        let err_resume = GatewayConfig::from_toml_str(
            &minimal("resume_args = [\"--resume\", \"foo\"]").replace("stdin = true", ""),
        );
        assert!(err_resume.is_err());
    }
}
