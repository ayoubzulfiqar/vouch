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
    /// Per-route scheduling strategy (`ordered`, `least_latency`, `hedged`).
    #[serde(default)]
    pub route_policies: BTreeMap<String, RoutePolicyConfig>,
    /// Public model aliases pointing to canonical `[routes]` entries.
    #[serde(default)]
    pub aliases: BTreeMap<String, String>,
    /// Trait-based routing rules evaluated before static `[routes]`/`[aliases]`.
    #[serde(default)]
    pub routing_rules: Vec<RoutingRuleConfig>,
    /// Deterministic response cache (`[cache]`).
    #[serde(default)]
    pub cache: CacheConfig,
    /// Per-route or per-backend token pricing (`[pricing.<id>]`).
    #[serde(default)]
    pub pricing: BTreeMap<String, PricingConfig>,
    /// Configurable prompt templates exposed via MCP `prompts/list` & `prompts/get`.
    #[serde(default)]
    pub mcp_prompts: Vec<McpPromptConfig>,
    /// External stdio/HTTP MCP servers aggregated under `<id>__<tool>`.
    #[serde(default)]
    pub mcp_upstreams: Vec<McpUpstreamConfig>,
    /// Remote multimodal image attachment fetching & size limits.
    #[serde(default)]
    pub attachments: AttachmentsConfig,
    /// Optional OpenTelemetry OTLP/HTTP trace export configuration.
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    /// Outbound security controls (`[security]`).
    #[serde(default)]
    pub security: SecurityConfig,
    /// Opt-in redacted JSONL flight recorder (`[flight_recorder]`).
    #[serde(default)]
    pub flight_recorder: FlightRecorderConfig,
    /// Cross-agent shared session ledger & handoff settings (`[shared_memory]`).
    #[serde(default)]
    pub shared_memory: SharedMemoryConfig,
    /// Model-free `trace` stdio MCP codebase intelligence bridge (`[trace]`).
    #[serde(default)]
    pub trace: TraceBridgeConfig,
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
    /// Maximum time `vouch serve` waits for in-flight connections to drain on
    /// `SIGINT`/`SIGTERM` before forcibly cancelling active tasks. Default: `15`.
    #[serde(default = "default_shutdown_drain_secs")]
    pub shutdown_drain_secs: u64,
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
    /// Enable runtime administrative control endpoints (`/admin/*`). Defaults to `false`.
    #[serde(default)]
    pub enable_admin: bool,
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
    /// Maximum number of entries retained in session affinity and `/v1/responses` stores.
    #[serde(default = "default_max_session_entries")]
    pub max_session_entries: usize,
    /// Optional directory for atomic `0600` persistence of sessions, responses,
    /// daily token budgets, and usage analytics across daemon restarts.
    #[serde(default)]
    pub state_dir: Option<PathBuf>,
    /// Automatic corrective schema repair attempts (`0..=2`) on non-streaming
    /// `cli`/`pty` `response_format` JSON Schema failures. Default: `0`.
    #[serde(default)]
    pub schema_repair_retries: u8,
}

/// Candidate scheduling strategy under `[route_policies.<model>]`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteStrategy {
    /// Walk the configured chain in static order (after sticky affinity).
    #[default]
    Ordered,
    /// Dynamically prefer the candidate with the lowest EWMA Time-To-First-Token.
    LeastLatency,
    /// Speculatively launch candidate #2 if candidate #1 misses `hedge_delay_ms`.
    Hedged,
}

/// Optional route scheduling policy (`[route_policies.<model>]`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RoutePolicyConfig {
    /// Candidate selection / speculative racing strategy.
    #[serde(default)]
    pub strategy: RouteStrategy,
    /// Delay before dispatching the hedged backup candidate when `strategy = "hedged"`.
    #[serde(default = "default_hedge_delay_ms")]
    pub hedge_delay_ms: u64,
}

/// Deterministic response cache (`[cache]`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CacheConfig {
    /// Enable SHA-256 deterministic response caching. Default: `false`.
    #[serde(default)]
    pub enabled: bool,
    /// Maximum number of cached completions before oldest-entry eviction. Default: `1000`.
    #[serde(default = "default_cache_max_entries")]
    pub max_entries: usize,
    /// Cache entry time-to-live in seconds. Default: `3600`.
    #[serde(default = "default_cache_ttl_secs")]
    pub ttl_secs: u64,
    /// Requests above this sampling temperature bypass the cache. Default: `0.2`.
    #[serde(default = "default_cache_max_temperature")]
    pub max_temperature: f64,
    /// Persist cache entries into `defaults.state_dir` when set. Default: `false`.
    #[serde(default)]
    pub persist: bool,
    /// Deduplicate concurrent identical deterministic non-streaming requests via
    /// in-flight singleflight coalescing. Defaults to `true` when `cache.enabled = true`.
    #[serde(default)]
    pub coalesce_inflight: Option<bool>,
    /// Collapse consecutive whitespace in prompt text before hashing cache/singleflight keys.
    #[serde(default)]
    pub normalize_whitespace: bool,
    /// Maximum concurrent singleflight waiters queued behind one in-flight leader. Default: `64`.
    #[serde(default = "default_cache_max_coalesce_waiters")]
    pub max_coalesce_waiters: usize,
}

impl CacheConfig {
    /// Whether in-flight singleflight coalescing is enabled.
    pub fn coalesce_enabled(&self) -> bool {
        self.coalesce_inflight.unwrap_or(self.enabled)
    }
}

/// USD pricing per million tokens for `[pricing.<route_or_backend>]`.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PricingConfig {
    /// Input cost in USD per 1,000,000 prompt tokens.
    #[serde(default)]
    pub input_per_1m_usd: f64,
    /// Output cost in USD per 1,000,000 completion tokens.
    #[serde(default)]
    pub output_per_1m_usd: f64,
}

/// Configurable prompt argument under `[[mcp_prompts]]`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct McpPromptArgument {
    /// Argument placeholder key (substituted as `{{name}}`).
    pub name: String,
    /// Human-readable description surfaced via `prompts/list`.
    #[serde(default)]
    pub description: String,
    /// Whether callers must supply this argument to `prompts/get`.
    #[serde(default)]
    pub required: bool,
}

/// Prompt template exposed via MCP `prompts/list` and `prompts/get`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct McpPromptConfig {
    /// Unique prompt identifier.
    pub name: String,
    /// Short prompt summary.
    #[serde(default)]
    pub description: String,
    /// Template body supporting `{{argument_name}}` placeholders.
    pub template: String,
    /// Declared template arguments.
    #[serde(default)]
    pub arguments: Vec<McpPromptArgument>,
}

/// Transport protocol for an aggregated upstream MCP server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum McpUpstreamTransport {
    /// JSON-RPC 2.0 over child stdin/stdout lines.
    Stdio,
    /// JSON-RPC 2.0 over HTTP `POST`.
    Http,
}

/// External MCP server aggregated under `<id>__<tool>` (`[[mcp_upstreams]]`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct McpUpstreamConfig {
    /// Unique namespace prefix (`<id>__<tool>`).
    pub id: String,
    /// Transport (`stdio` or `http`).
    pub transport: McpUpstreamTransport,
    /// Command to spawn when `transport = "stdio"`.
    #[serde(default)]
    pub command: Option<String>,
    /// Arguments passed to `command`.
    #[serde(default)]
    pub args: Vec<String>,
    /// Environment injected into the child MCP process.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// HTTP JSON-RPC endpoint URL when `transport = "http"`.
    #[serde(default)]
    pub base_url: Option<String>,
    /// Extra headers attached to upstream HTTP MCP requests.
    #[serde(default)]
    pub extra_headers: BTreeMap<String, String>,
    /// Call timeout in seconds. Default: `30`.
    #[serde(default = "default_mcp_upstream_timeout_secs")]
    pub timeout_secs: u64,
}

/// Remote image attachment fetching and size controls.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AttachmentsConfig {
    /// Fetch remote `http://` and `https://` `image_url` inputs on CLI/PTY backends.
    #[serde(default)]
    pub fetch_remote_urls: bool,
    /// Maximum allowed remote image payload in bytes. Default: `10_485_760` (10 MiB).
    #[serde(default = "default_max_image_bytes")]
    pub max_image_bytes: usize,
    /// Allow fetching remote URLs pointing at loopback/private/link-local IPs.
    /// Defaults to `false` to prevent SSRF.
    #[serde(default)]
    pub allow_private_ips: bool,
}

/// Trait-based routing rule evaluated before static `[routes]` lookup.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingRuleConfig {
    /// Optional human-readable label for logs and telemetry.
    #[serde(default)]
    pub name: Option<String>,
    /// Glob pattern matched against the requested public model or alias.
    pub model_pattern: String,
    /// Require tool definitions (`true`) or their absence (`false`).
    #[serde(default)]
    pub has_tools: Option<bool>,
    /// Require image inputs (`true`) or their absence (`false`).
    #[serde(default)]
    pub has_images: Option<bool>,
    /// Minimum estimated BPE prompt token count to match this rule.
    #[serde(default)]
    pub min_prompt_tokens: Option<u64>,
    /// Required exact HTTP request header matches (`header_name -> value`).
    #[serde(default)]
    pub header_equals: BTreeMap<String, String>,
    /// Target route key in `[routes]` selected when all conditions match.
    pub target_route: String,
}

/// OpenTelemetry OTLP/HTTP trace export settings.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetryConfig {
    /// Optional OTLP/HTTP JSON traces endpoint, e.g. `http://127.0.0.1:4318/v1/traces`.
    #[serde(default)]
    pub otlp_endpoint: Option<String>,
    /// Service name exported in OTLP `service.name` resource attribute.
    #[serde(default = "default_service_name")]
    pub service_name: String,
    /// Optional extra headers sent with OTLP trace export requests.
    #[serde(default)]
    pub otlp_headers: BTreeMap<String, String>,
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
    /// How CLI stdout bytes are buffered before ANSI stripping & streaming (`chunk` or `line`).
    #[serde(default)]
    pub read_mode: ReadMode,
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
    /// Optional per-profile credential/environment rotation pool for `cli`/`pty` backends.
    #[serde(default)]
    pub profiles: Vec<BackendProfileConfig>,
    /// Regex output filters applied to `cli`/`pty` output before yielding downstream.
    #[serde(default)]
    pub output_filters: Vec<OutputFilterConfig>,
}

/// Regex replacement rule applied to `cli`/`pty` backend text output (`[[backends.output_filters]]`).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OutputFilterConfig {
    /// Regular expression pattern matched against backend output.
    pub pattern: String,
    /// Replacement string (`regex::Regex::replace_all` syntax). Default: `""`.
    #[serde(default)]
    pub replace: String,
}

/// Action taken when the outbound egress guardrail detects a secret or credential.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EgressGuardMode {
    /// Disable outbound egress secret inspection.
    #[default]
    Off,
    /// Replace detected secret patterns with `[REDACTED_EGRESS_SECRET]` before dispatching.
    Redact,
    /// Reject matching requests with HTTP `403` (`egress_policy_blocked`) before dispatching.
    Block,
}

/// Outbound security controls (`[security]`).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SecurityConfig {
    /// Outbound prompt/message secret & credential scanner (`[security.egress_guard]`).
    #[serde(default)]
    pub egress_guard: EgressGuardConfig,
}

/// Outbound secret/PII egress guardrail settings (`[security.egress_guard]`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EgressGuardConfig {
    /// Enforcement mode (`off`, `redact`, or `block`). Default: `"off"`.
    #[serde(default)]
    pub mode: EgressGuardMode,
    /// Enable built-in detectors for PEM private keys, AWS `AKIA`, GitHub tokens, and Stripe live keys. Default: `true`.
    #[serde(default = "default_true")]
    pub block_private_keys: bool,
    /// Additional custom regular expressions flagged as egress secrets.
    #[serde(default)]
    pub extra_patterns: Vec<String>,
}

/// Opt-in redacted JSONL flight recorder settings (`[flight_recorder]`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FlightRecorderConfig {
    /// Enable asynchronous redacted JSONL flight recording. Default: `false`.
    #[serde(default)]
    pub enabled: bool,
    /// Maximum records retained in the flight log file (rotates/truncates oldest entries when exceeded). Default: `500`.
    #[serde(default = "default_flight_recorder_max_records")]
    pub max_records: usize,
    /// Custom JSONL path. Defaults to `<defaults.state_dir>/flight.jsonl` when `state_dir` is configured,
    /// or `std::env::temp_dir().join("vouch-flight.jsonl")`.
    #[serde(default)]
    pub path: Option<PathBuf>,
}

/// Cross-agent canonical session ledger & deterministic handoff settings (`[shared_memory]`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SharedMemoryConfig {
    /// Enable canonical session memory tracking and `<vouch_shared_context>` handoffs. Default: `false`.
    #[serde(default)]
    pub enabled: bool,
    /// Maximum canonical turns kept per session before oldest turns drop. Default: `32`.
    #[serde(default = "default_shared_memory_max_turns")]
    pub max_turns_per_session: usize,
    /// Strict BPE token budget for injected `<vouch_shared_context>` handoff envelopes. Default: `1500`.
    #[serde(default = "default_shared_memory_max_context_tokens")]
    pub max_context_tokens: usize,
    /// Inject repository context from `[trace]` on turn 1 of a new session. Default: `true`.
    #[serde(default = "default_true")]
    pub inject_on_new_session: bool,
    /// Inject `<vouch_shared_context>` whenever execution moves to a different agent or loses native handle. Default: `true`.
    #[serde(default = "default_true")]
    pub inject_on_handoff: bool,
    /// Write structured session checkpoints back to `trace` via `record_session`. Default: `true`.
    #[serde(default = "default_true")]
    pub sync_to_trace: bool,
}

/// Model-free `trace` stdio MCP intelligence bridge settings (`[trace]`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TraceBridgeConfig {
    /// Enable spawning pooled `trace serve` stdio MCP workers. Default: `false`.
    #[serde(default)]
    pub enabled: bool,
    /// Executable command used to launch the `trace` MCP server. Default: `"trace"`.
    #[serde(default = "default_trace_command")]
    pub command: String,
    /// Command-line arguments passed to `command` (supports `{root}`). Default: `["serve"]`.
    #[serde(default = "default_trace_args")]
    pub args: Vec<String>,
    /// Optional workspace root passed to `trace`; falls back to `workspace.repo_path` or cwd.
    #[serde(default)]
    pub workspace_root: Option<PathBuf>,
    /// Startup + `initialize` handshake timeout in milliseconds. Default: `3000`.
    #[serde(default = "default_trace_startup_timeout_ms")]
    pub startup_timeout_ms: u64,
    /// Hard per-RPC deadline in milliseconds so `trace` queries never stall requests. Default: `1500`.
    #[serde(default = "default_trace_query_timeout_ms")]
    pub query_timeout_ms: u64,
    /// BPE sub-budget reserved inside `shared_memory.max_context_tokens` for `trace` facts. Default: `900`.
    #[serde(default = "default_trace_max_tokens")]
    pub max_trace_tokens: usize,
    /// When `true`, `vouch doctor` fails with exit `2` if `trace` is unreachable. Default: `false`.
    #[serde(default)]
    pub required: bool,
}

/// One credential profile in a `cli` or `pty` backend rotation pool.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BackendProfileConfig {
    /// Profile identifier unique within its parent backend.
    pub id: String,
    /// Environment variables overlaid when this profile is selected (e.g. `HOME`).
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Selection weight for weighted round-robin rotation. Default: `1`.
    #[serde(default = "default_profile_weight")]
    pub weight: u32,
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

/// How raw bytes from a CLI subprocess's stdout are read into text chunks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadMode {
    /// Stream raw UTF-8 byte chunks immediately without waiting for newlines.
    #[default]
    Chunk,
    /// Legacy newline-delimited buffering (`\n`).
    Line,
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
    /// Optional rolling UTC day prompt + completion BPE token limit.
    #[serde(default)]
    pub daily_token_budget: Option<u64>,
    /// Optional sliding 60-second BPE token limit (TPM).
    #[serde(default)]
    pub tpm: Option<u64>,
    /// Optional maximum concurrent active requests for this token.
    #[serde(default)]
    pub max_concurrent: Option<usize>,
    /// Optional RFC-3339 expiration timestamp (`YYYY-MM-DDTHH:MM:SSZ`).
    #[serde(default)]
    pub expires_at: Option<String>,
    /// Glob patterns of models this token may call. Empty means all.
    #[serde(default)]
    pub allowed_models: Vec<String>,
    /// Optional glob allowlist restricting MCP tool names (`tools/call`, `tasks/create`).
    #[serde(default)]
    pub allowed_mcp_tools: Option<Vec<String>>,
    /// Optional backend id/glob restriction; `None` permits all backends.
    #[serde(default)]
    pub allowed_backends: Option<Vec<String>>,
    /// Grant administrative control access (`/admin/*`). Defaults to `false`.
    #[serde(default)]
    pub admin: bool,
}

/// Compiled model allowlist.
#[derive(Debug)]
pub struct ModelAllowlist {
    set: GlobSet,
    patterns: Vec<String>,
}

impl ModelAllowlist {
    /// Compile glob patterns for `allowed_models`.
    pub fn compile(patterns: &[String]) -> Result<Self> {
        Self::compile_named("allowed_models", patterns)
    }

    /// Compile glob patterns for a named configuration allowlist field.
    pub fn compile_named(field: &str, patterns: &[String]) -> Result<Self> {
        let mut builder = GlobSetBuilder::new();
        for pattern in patterns {
            let glob = Glob::new(pattern).map_err(|error| {
                GatewayError::Config(format!("invalid {field} glob {pattern:?}: {error}"))
            })?;
            builder.add(glob);
        }
        let set = builder
            .build()
            .map_err(|error| GatewayError::Config(format!("invalid {field} glob set: {error}")))?;
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
        if let Some(socket) = &self.server.unix_socket {
            let socket_len = socket.to_string_lossy().len();
            if socket_len == 0 || socket_len >= 100 {
                return Err(GatewayError::Config(format!(
                    "server.unix_socket {:?} must be non-empty and under 100 bytes for macOS/BSD sockaddr_un compatibility (got {socket_len} bytes)",
                    socket.display()
                )));
            }
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
        if self.defaults.max_session_entries == 0 {
            return Err(GatewayError::Config(
                "defaults.max_session_entries must be greater than zero".into(),
            ));
        }
        if self.defaults.schema_repair_retries > 2 {
            return Err(GatewayError::Config(
                "defaults.schema_repair_retries must be at most 2".into(),
            ));
        }
        if self.cache.max_entries == 0 {
            return Err(GatewayError::Config(
                "cache.max_entries must be greater than zero".into(),
            ));
        }
        if self.cache.ttl_secs == 0 {
            return Err(GatewayError::Config(
                "cache.ttl_secs must be greater than zero".into(),
            ));
        }
        if !self.cache.max_temperature.is_finite() || self.cache.max_temperature < 0.0 {
            return Err(GatewayError::Config(
                "cache.max_temperature must be a finite number >= 0.0".into(),
            ));
        }
        if self.cache.max_coalesce_waiters == 0 {
            return Err(GatewayError::Config(
                "cache.max_coalesce_waiters must be greater than zero".into(),
            ));
        }
        if self.attachments.max_image_bytes == 0 {
            return Err(GatewayError::Config(
                "attachments.max_image_bytes must be greater than zero".into(),
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

        for (alias, target) in &self.aliases {
            if alias.trim().is_empty() {
                return Err(GatewayError::Config(
                    "model alias names must not be empty".into(),
                ));
            }
            if self.routes.contains_key(alias) {
                return Err(GatewayError::Config(format!(
                    "alias {alias:?} conflicts with an explicit [routes] entry"
                )));
            }
            if !self.routes.contains_key(target) {
                return Err(GatewayError::Config(format!(
                    "alias {alias:?} references unknown route {target:?}"
                )));
            }
        }

        for (idx, rule) in self.routing_rules.iter().enumerate() {
            if rule.model_pattern.trim().is_empty() {
                return Err(GatewayError::Config(format!(
                    "routing_rules[{idx}].model_pattern must not be empty"
                )));
            }
            ModelAllowlist::compile(std::slice::from_ref(&rule.model_pattern))?;
            if !self.routes.contains_key(&rule.target_route) {
                return Err(GatewayError::Config(format!(
                    "routing_rules[{idx}].target_route {:?} is not a configured route",
                    rule.target_route
                )));
            }
            for header_name in rule.header_equals.keys() {
                if header_name.trim().is_empty()
                    || axum::http::header::HeaderName::from_bytes(header_name.as_bytes()).is_err()
                {
                    return Err(GatewayError::Config(format!(
                        "routing_rules[{idx}].header_equals has invalid header name {header_name:?}"
                    )));
                }
            }
        }

        if let Some(model) = &self.defaults.model
            && self.resolve_route_name(model).is_none()
        {
            return Err(GatewayError::Config(format!(
                "defaults.model {model:?} is not a configured route or alias"
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

        for (model, policy) in &self.route_policies {
            if !self.routes.contains_key(model) {
                return Err(GatewayError::Config(format!(
                    "route_policies.{model:?} does not reference a configured [routes] key"
                )));
            }
            if policy.hedge_delay_ms == 0 {
                return Err(GatewayError::Config(format!(
                    "route_policies.{model:?}.hedge_delay_ms must be greater than zero"
                )));
            }
        }

        for (target, pricing) in &self.pricing {
            if target.trim().is_empty() {
                return Err(GatewayError::Config(
                    "pricing keys must not be empty".into(),
                ));
            }
            if !pricing.input_per_1m_usd.is_finite()
                || pricing.input_per_1m_usd < 0.0
                || !pricing.output_per_1m_usd.is_finite()
                || pricing.output_per_1m_usd < 0.0
            {
                return Err(GatewayError::Config(format!(
                    "pricing.{target:?} rates must be finite non-negative USD values"
                )));
            }
        }

        let mut prompt_names = HashSet::new();
        for prompt in &self.mcp_prompts {
            if prompt.name.trim().is_empty() {
                return Err(GatewayError::Config(
                    "mcp_prompts.name must not be empty".into(),
                ));
            }
            if !prompt_names.insert(prompt.name.clone()) {
                return Err(GatewayError::Config(format!(
                    "duplicate mcp_prompts.name {:?}",
                    prompt.name
                )));
            }
            let mut arg_names = HashSet::new();
            for arg in &prompt.arguments {
                if arg.name.trim().is_empty() || !arg_names.insert(arg.name.clone()) {
                    return Err(GatewayError::Config(format!(
                        "mcp_prompts {:?} has invalid or duplicate argument name {:?}",
                        prompt.name, arg.name
                    )));
                }
            }
        }

        let mut upstream_ids = HashSet::new();
        for upstream in &self.mcp_upstreams {
            if upstream.id.trim().is_empty() || upstream.id.contains("__") {
                return Err(GatewayError::Config(format!(
                    "mcp_upstreams.id {:?} must be non-empty and must not contain '__'",
                    upstream.id
                )));
            }
            if !upstream_ids.insert(upstream.id.clone()) {
                return Err(GatewayError::Config(format!(
                    "duplicate mcp_upstreams.id {:?}",
                    upstream.id
                )));
            }
            if upstream.timeout_secs == 0 {
                return Err(GatewayError::Config(format!(
                    "mcp_upstreams {:?}: timeout_secs must be greater than zero",
                    upstream.id
                )));
            }
            match upstream.transport {
                McpUpstreamTransport::Stdio => {
                    if upstream
                        .command
                        .as_deref()
                        .map(str::trim)
                        .unwrap_or("")
                        .is_empty()
                    {
                        return Err(GatewayError::Config(format!(
                            "mcp_upstreams {:?}: stdio transport requires command",
                            upstream.id
                        )));
                    }
                    if upstream.base_url.is_some() {
                        return Err(GatewayError::Config(format!(
                            "mcp_upstreams {:?}: stdio transport forbids base_url",
                            upstream.id
                        )));
                    }
                }
                McpUpstreamTransport::Http => {
                    let url = upstream.base_url.as_deref().ok_or_else(|| {
                        GatewayError::Config(format!(
                            "mcp_upstreams {:?}: http transport requires base_url",
                            upstream.id
                        ))
                    })?;
                    let parsed = reqwest::Url::parse(url).map_err(|error| {
                        GatewayError::Config(format!(
                            "mcp_upstreams {:?}: invalid base_url {url:?}: {error}",
                            upstream.id
                        ))
                    })?;
                    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
                        return Err(GatewayError::Config(format!(
                            "mcp_upstreams {:?}: base_url must use http:// or https://",
                            upstream.id
                        )));
                    }
                    if upstream.command.is_some() {
                        return Err(GatewayError::Config(format!(
                            "mcp_upstreams {:?}: http transport forbids command",
                            upstream.id
                        )));
                    }
                    for header_name in upstream.extra_headers.keys() {
                        if header_name.trim().is_empty()
                            || axum::http::header::HeaderName::from_bytes(header_name.as_bytes())
                                .is_err()
                        {
                            return Err(GatewayError::Config(format!(
                                "mcp_upstreams {:?}: invalid header name {header_name:?}",
                                upstream.id
                            )));
                        }
                    }
                }
            }
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
            if token.daily_token_budget == Some(0) {
                return Err(GatewayError::Config(
                    "token daily_token_budget must be greater than zero when set".into(),
                ));
            }
            if token.tpm == Some(0) {
                return Err(GatewayError::Config(
                    "token tpm must be greater than zero when set".into(),
                ));
            }
            if token.max_concurrent == Some(0) {
                return Err(GatewayError::Config(
                    "token max_concurrent must be greater than zero when set".into(),
                ));
            }
            if let Some(exp) = &token.expires_at {
                parse_rfc3339_unix_secs(exp)?;
            }
            if let Some(allowed_backends) = &token.allowed_backends {
                ModelAllowlist::compile_named("allowed_backends", allowed_backends)?;
            }
            if let Some(allowed_mcp_tools) = &token.allowed_mcp_tools {
                ModelAllowlist::compile_named("allowed_mcp_tools", allowed_mcp_tools)?;
            }
            ModelAllowlist::compile(&token.allowed_models)?;
        }

        if self.telemetry.service_name.trim().is_empty() {
            return Err(GatewayError::Config(
                "telemetry.service_name must not be empty".into(),
            ));
        }
        if let Some(endpoint) = &self.telemetry.otlp_endpoint {
            let parsed = reqwest::Url::parse(endpoint).map_err(|error| {
                GatewayError::Config(format!(
                    "invalid telemetry.otlp_endpoint {endpoint:?}: {error}"
                ))
            })?;
            if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
                return Err(GatewayError::Config(format!(
                    "telemetry.otlp_endpoint {endpoint:?} must be an http:// or https:// URL with a host"
                )));
            }
        }
        for header_name in &self.telemetry.otlp_headers {
            if header_name.0.trim().is_empty()
                || axum::http::header::HeaderName::from_bytes(header_name.0.as_bytes()).is_err()
            {
                return Err(GatewayError::Config(format!(
                    "telemetry.otlp_headers contains invalid header name {:?}",
                    header_name.0
                )));
            }
        }
        for pattern in &self.security.egress_guard.extra_patterns {
            if pattern.is_empty() {
                return Err(GatewayError::Config(
                    "security.egress_guard.extra_patterns must not contain empty patterns".into(),
                ));
            }
            regex::Regex::new(pattern).map_err(|error| {
                GatewayError::Config(format!(
                    "security.egress_guard.extra_patterns invalid regex {pattern:?}: {error}"
                ))
            })?;
        }
        if self.flight_recorder.max_records == 0 {
            return Err(GatewayError::Config(
                "flight_recorder.max_records must be greater than zero".into(),
            ));
        }
        if self.shared_memory.max_turns_per_session == 0 {
            return Err(GatewayError::Config(
                "shared_memory.max_turns_per_session must be greater than zero".into(),
            ));
        }
        if self.shared_memory.max_context_tokens == 0 {
            return Err(GatewayError::Config(
                "shared_memory.max_context_tokens must be greater than zero".into(),
            ));
        }
        if self.trace.enabled && self.trace.command.trim().is_empty() {
            return Err(GatewayError::Config(
                "trace.command must not be empty when trace.enabled is true".into(),
            ));
        }
        if self.trace.startup_timeout_ms == 0 {
            return Err(GatewayError::Config(
                "trace.startup_timeout_ms must be greater than zero".into(),
            ));
        }
        if self.trace.query_timeout_ms == 0 {
            return Err(GatewayError::Config(
                "trace.query_timeout_ms must be greater than zero".into(),
            ));
        }
        if self.trace.max_trace_tokens == 0 {
            return Err(GatewayError::Config(
                "trace.max_trace_tokens must be greater than zero".into(),
            ));
        }

        Ok(())
    }

    /// Resolve a requested public model name or alias to its canonical route key.
    pub fn resolve_route_name<'a>(&'a self, requested: &'a str) -> Option<&'a str> {
        if self.routes.contains_key(requested) {
            Some(requested)
        } else {
            self.aliases.get(requested).map(String::as_str)
        }
    }

    /// Resolve the model used when a request omits one (expanding aliases).
    pub fn default_model(&self) -> Option<&str> {
        self.defaults
            .model
            .as_deref()
            .map(|name| self.resolve_route_name(name).unwrap_or(name))
            .or_else(|| {
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
                collect_secret_kv(&mut secrets, key, value);
            }
            for profile in &backend.profiles {
                for (key, value) in &profile.env {
                    collect_secret_kv(&mut secrets, key, value);
                }
            }
            for (key, value) in &backend.extra_headers {
                collect_header_secret(&mut secrets, key, value);
            }
            if let Some(env) = &backend.api_key_env
                && let Ok(value) = std::env::var(env)
                && !value.is_empty()
            {
                secrets.push(value);
            }
        }
        for upstream in &self.mcp_upstreams {
            for (key, value) in &upstream.env {
                collect_secret_kv(&mut secrets, key, value);
            }
            for (key, value) in &upstream.extra_headers {
                collect_header_secret(&mut secrets, key, value);
            }
        }
        for (key, value) in &self.telemetry.otlp_headers {
            collect_header_secret(&mut secrets, key, value);
        }
        secrets.sort();
        secrets.dedup();
        secrets
    }
}

fn collect_secret_kv(secrets: &mut Vec<String>, key: &str, value: &str) {
    let key = key.to_ascii_uppercase();
    let looks_secret = key.contains("KEY")
        || key.contains("TOKEN")
        || key.contains("SECRET")
        || key.contains("PASSWORD")
        || key.contains("AUTH");
    if looks_secret || looks_like_secret(value) {
        secrets.push(value.to_string());
    }
}

fn collect_header_secret(secrets: &mut Vec<String>, key: &str, value: &str) {
    let key = key.to_ascii_lowercase();
    if key.contains("authorization")
        || key.contains("api-key")
        || key.contains("token")
        || key.contains("secret")
        || looks_like_secret(value)
    {
        secrets.push(value.to_string());
    }
}

/// Parse an RFC-3339 timestamp string (`YYYY-MM-DDTHH:MM:SSZ` or with offset) into UNIX seconds.
pub fn parse_rfc3339_unix_secs(raw: &str) -> Result<u64> {
    let s = raw.trim();
    let sep_idx = s.find(['T', 't']).ok_or_else(|| {
        GatewayError::Config(format!("invalid RFC-3339 timestamp {raw:?}: missing 'T'"))
    })?;
    let (date_str, time_and_tz) = s.split_at(sep_idx);
    let time_and_tz = &time_and_tz[1..];
    let mut date_parts = date_str.split('-');
    let year: i64 = date_parts
        .next()
        .and_then(|v| if v.len() == 4 { v.parse().ok() } else { None })
        .ok_or_else(|| GatewayError::Config(format!("invalid RFC-3339 date in {raw:?}")))?;
    let month: u32 = date_parts
        .next()
        .and_then(|v| if v.len() == 2 { v.parse().ok() } else { None })
        .ok_or_else(|| GatewayError::Config(format!("invalid RFC-3339 month in {raw:?}")))?;
    let day: u32 = date_parts
        .next()
        .and_then(|v| if v.len() == 2 { v.parse().ok() } else { None })
        .ok_or_else(|| GatewayError::Config(format!("invalid RFC-3339 day in {raw:?}")))?;
    if date_parts.next().is_some() || !(1..=12).contains(&month) {
        return Err(GatewayError::Config(format!(
            "invalid RFC-3339 date in {raw:?}"
        )));
    }
    let max_day = days_in_month(year, month);
    if !(1..=max_day).contains(&day) {
        return Err(GatewayError::Config(format!(
            "invalid RFC-3339 calendar day in {raw:?}"
        )));
    }

    let (time_str, tz_offset_secs): (&str, i64) = if let Some(bare) = time_and_tz
        .strip_suffix('Z')
        .or_else(|| time_and_tz.strip_suffix('z'))
    {
        (bare, 0)
    } else if let Some(idx) = time_and_tz.rfind(['+', '-']) {
        let (t, off) = time_and_tz.split_at(idx);
        let sign = if off.starts_with('-') { -1i64 } else { 1i64 };
        let hm = &off[1..];
        let (oh_s, om_s) = hm
            .split_once(':')
            .ok_or_else(|| GatewayError::Config(format!("invalid RFC-3339 offset in {raw:?}")))?;
        let oh: i64 = oh_s
            .parse()
            .map_err(|_| GatewayError::Config(format!("invalid RFC-3339 offset in {raw:?}")))?;
        let om: i64 = om_s
            .parse()
            .map_err(|_| GatewayError::Config(format!("invalid RFC-3339 offset in {raw:?}")))?;
        if !(0..=23).contains(&oh) || !(0..=59).contains(&om) {
            return Err(GatewayError::Config(format!(
                "RFC-3339 offset out of range in {raw:?}"
            )));
        }
        (t, sign * (oh * 3600 + om * 60))
    } else {
        return Err(GatewayError::Config(format!(
            "invalid RFC-3339 timezone in {raw:?}"
        )));
    };

    let main_time = time_str.split('.').next().unwrap_or(time_str);
    let mut tp = main_time.split(':');
    let hour: u64 = tp
        .next()
        .and_then(|v| if v.len() == 2 { v.parse().ok() } else { None })
        .ok_or_else(|| GatewayError::Config(format!("invalid RFC-3339 hour in {raw:?}")))?;
    let min: u64 = tp
        .next()
        .and_then(|v| if v.len() == 2 { v.parse().ok() } else { None })
        .ok_or_else(|| GatewayError::Config(format!("invalid RFC-3339 minute in {raw:?}")))?;
    let sec: u64 = tp
        .next()
        .and_then(|v| if v.len() == 2 { v.parse().ok() } else { None })
        .ok_or_else(|| GatewayError::Config(format!("invalid RFC-3339 second in {raw:?}")))?;
    if tp.next().is_some() || hour > 23 || min > 59 || sec > 60 {
        return Err(GatewayError::Config(format!(
            "invalid RFC-3339 time in {raw:?}"
        )));
    }

    let days = days_from_civil(year, month, day);
    let local_secs = days * 86_400 + (hour as i64) * 3600 + (min as i64) * 60 + (sec as i64);
    let utc_secs = local_secs - tz_offset_secs;
    u64::try_from(utc_secs).map_err(|_| {
        GatewayError::Config(format!("RFC-3339 timestamp {raw:?} predates UNIX epoch"))
    })
}

fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            let leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
            if leap { 29 } else { 28 }
        }
        _ => 0,
    }
}

fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let m = i64::from(month);
    let d = i64::from(day);
    let y = year - if m <= 2 { 1 } else { 0 };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
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
        for filter in &self.output_filters {
            if filter.pattern.is_empty() {
                return Err(GatewayError::Config(format!(
                    "backend {:?}: output_filters.pattern must not be empty",
                    self.id
                )));
            }
            regex::Regex::new(&filter.pattern).map_err(|error| {
                GatewayError::Config(format!(
                    "backend {:?}: invalid output_filters regex {:?}: {error}",
                    self.id, filter.pattern
                ))
            })?;
        }
        if !self.profiles.is_empty() {
            if self.kind == BackendKind::Http {
                return Err(GatewayError::Config(format!(
                    "backend {:?}: credential profiles are only supported on cli and pty backends",
                    self.id
                )));
            }
            let mut profile_ids = HashSet::new();
            for profile in &self.profiles {
                if profile.id.trim().is_empty() {
                    return Err(GatewayError::Config(format!(
                        "backend {:?}: profile id must not be empty",
                        self.id
                    )));
                }
                if !profile_ids.insert(profile.id.clone()) {
                    return Err(GatewayError::Config(format!(
                        "backend {:?}: duplicate profile id {:?}",
                        self.id, profile.id
                    )));
                }
                if profile.weight == 0 {
                    return Err(GatewayError::Config(format!(
                        "backend {:?} profile {:?}: weight must be greater than zero",
                        self.id, profile.id
                    )));
                }
            }
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
fn default_max_session_entries() -> usize {
    10_000
}
fn default_max_image_bytes() -> usize {
    10_485_760
}
fn default_service_name() -> String {
    "vouch".into()
}
fn default_hedge_delay_ms() -> u64 {
    2500
}
fn default_cache_max_entries() -> usize {
    1000
}
fn default_cache_ttl_secs() -> u64 {
    3600
}
fn default_cache_max_temperature() -> f64 {
    0.2
}
fn default_cache_max_coalesce_waiters() -> usize {
    64
}
fn default_profile_weight() -> u32 {
    1
}
fn default_mcp_upstream_timeout_secs() -> u64 {
    30
}
fn default_shutdown_drain_secs() -> u64 {
    15
}
fn default_flight_recorder_max_records() -> usize {
    500
}
fn default_shared_memory_max_turns() -> usize {
    32
}
fn default_shared_memory_max_context_tokens() -> usize {
    1500
}
fn default_trace_command() -> String {
    "trace".into()
}
fn default_trace_args() -> Vec<String> {
    vec!["serve".into()]
}
fn default_trace_startup_timeout_ms() -> u64 {
    3000
}
fn default_trace_query_timeout_ms() -> u64 {
    1500
}
fn default_trace_max_tokens() -> usize {
    900
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            tcp_listen: default_tcp_listen(),
            unix_socket: None,
            request_timeout_secs: default_request_timeout_secs(),
            stream_timeout_secs: default_stream_timeout_secs(),
            shutdown_drain_secs: default_shutdown_drain_secs(),
            keep_alive_interval_secs: default_keep_alive_secs(),
            max_body_bytes: default_max_body_bytes(),
            log_level: default_log_level(),
            cors_permissive: false,
            enable_ui: false,
            enable_admin: false,
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
            max_session_entries: default_max_session_entries(),
            state_dir: None,
            schema_repair_retries: 0,
        }
    }
}

impl Default for RoutePolicyConfig {
    fn default() -> Self {
        Self {
            strategy: RouteStrategy::Ordered,
            hedge_delay_ms: default_hedge_delay_ms(),
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_entries: default_cache_max_entries(),
            ttl_secs: default_cache_ttl_secs(),
            max_temperature: default_cache_max_temperature(),
            persist: false,
            coalesce_inflight: None,
            normalize_whitespace: false,
            max_coalesce_waiters: default_cache_max_coalesce_waiters(),
        }
    }
}

impl Default for EgressGuardConfig {
    fn default() -> Self {
        Self {
            mode: EgressGuardMode::Off,
            block_private_keys: true,
            extra_patterns: Vec::new(),
        }
    }
}

impl Default for FlightRecorderConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_records: default_flight_recorder_max_records(),
            path: None,
        }
    }
}

impl Default for SharedMemoryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_turns_per_session: default_shared_memory_max_turns(),
            max_context_tokens: default_shared_memory_max_context_tokens(),
            inject_on_new_session: true,
            inject_on_handoff: true,
            sync_to_trace: true,
        }
    }
}

impl Default for TraceBridgeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            command: default_trace_command(),
            args: default_trace_args(),
            workspace_root: None,
            startup_timeout_ms: default_trace_startup_timeout_ms(),
            query_timeout_ms: default_trace_query_timeout_ms(),
            max_trace_tokens: default_trace_max_tokens(),
            required: false,
        }
    }
}

impl Default for AttachmentsConfig {
    fn default() -> Self {
        Self {
            fetch_remote_urls: false,
            max_image_bytes: default_max_image_bytes(),
            allow_private_ips: false,
        }
    }
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            otlp_endpoint: None,
            service_name: default_service_name(),
            otlp_headers: BTreeMap::new(),
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

    #[test]
    fn parses_v3_defaults_and_overrides() {
        let base = GatewayConfig::from_toml_str(&minimal("")).expect("valid v2 config");
        assert_eq!(base.defaults.max_session_entries, 10_000);
        assert_eq!(base.defaults.state_dir, None);
        assert_eq!(base.backends[0].read_mode, ReadMode::Chunk);
        assert!(!base.attachments.fetch_remote_urls);
        assert_eq!(base.attachments.max_image_bytes, 10_485_760);
        assert!(!base.attachments.allow_private_ips);
        assert_eq!(base.telemetry.service_name, "vouch");
        assert_eq!(base.telemetry.otlp_endpoint, None);

        let override_text = r#"
[server]
tcp_listen = "127.0.0.1:0"

[defaults]
model = "fast-coder"
max_session_entries = 512
state_dir = "/tmp/vouch-state"

[attachments]
fetch_remote_urls = true
max_image_bytes = 2097152
allow_private_ips = true

[telemetry]
otlp_endpoint = "http://127.0.0.1:4318/v1/traces"
service_name = "vouch-dev"
otlp_headers = { Authorization = "Bearer otel-secret-123" }

[[backends]]
id = "mock"
kind = "cli"
command = "/bin/echo"
stdin = true
read_mode = "line"

[routes]
"coding/default" = ["mock"]

[aliases]
"fast-coder" = "coding/default"

[[routing_rules]]
name = "tools-to-default"
model_pattern = "fast-*"
has_tools = true
min_prompt_tokens = 128
header_equals = { "x-tier" = "pro" }
target_route = "coding/default"

[[tokens]]
token = "vouch-live-secret-token"
daily_token_budget = 50000
"#;
        let custom = GatewayConfig::from_toml_str(override_text).expect("valid v3 config");
        assert_eq!(custom.default_model(), Some("coding/default"));
        assert_eq!(custom.defaults.max_session_entries, 512);
        assert_eq!(
            custom.defaults.state_dir.as_deref(),
            Some(Path::new("/tmp/vouch-state"))
        );
        assert_eq!(custom.backends[0].read_mode, ReadMode::Line);
        assert!(custom.attachments.fetch_remote_urls);
        assert_eq!(custom.attachments.max_image_bytes, 2_097_152);
        assert!(custom.attachments.allow_private_ips);
        assert_eq!(custom.telemetry.service_name, "vouch-dev");
        assert_eq!(custom.routing_rules.len(), 1);
        assert_eq!(custom.tokens[0].daily_token_budget, Some(50_000));
        assert!(
            custom
                .secrets()
                .contains(&"Bearer otel-secret-123".to_string())
        );
    }

    #[test]
    fn rejects_invalid_v3_fields() {
        let zero_entries = minimal("").replace(
            "[[backends]]",
            "[defaults]\nmax_session_entries = 0\n\n[[backends]]",
        );
        assert!(
            GatewayConfig::from_toml_str(&zero_entries)
                .expect_err("zero max_session_entries")
                .to_string()
                .contains("max_session_entries")
        );

        let zero_budget = minimal("").replace(
            "[routes]",
            "[[tokens]]\ntoken = \"vouch-secret-token\"\ndaily_token_budget = 0\n\n[routes]",
        );
        assert!(
            GatewayConfig::from_toml_str(&zero_budget)
                .expect_err("zero budget")
                .to_string()
                .contains("daily_token_budget")
        );

        let bad_alias = format!(
            "{}\n[aliases]\n\"coding/default\" = \"coding/default\"\n",
            minimal("")
        );
        assert!(GatewayConfig::from_toml_str(&bad_alias).is_err());

        let unknown_alias_target =
            format!("{}\n[aliases]\n\"fast\" = \"missing/route\"\n", minimal(""));
        assert!(GatewayConfig::from_toml_str(&unknown_alias_target).is_err());

        let bad_rule = format!(
            "{}\n[[routing_rules]]\nmodel_pattern = \"[unclosed\"\ntarget_route = \"coding/default\"\n",
            minimal("")
        );
        assert!(GatewayConfig::from_toml_str(&bad_rule).is_err());

        let bad_otlp = format!(
            "{}\n[telemetry]\notlp_endpoint = \"ftp://localhost:4318/traces\"\n",
            minimal("")
        );
        assert!(GatewayConfig::from_toml_str(&bad_otlp).is_err());
    }

    #[test]
    fn parses_v4_defaults_and_overrides() {
        let base =
            GatewayConfig::from_toml_str(&minimal("")).expect("v3 config backwards compatible");
        assert!(!base.server.enable_admin);
        assert_eq!(base.defaults.schema_repair_retries, 0);
        assert!(!base.cache.enabled);
        assert_eq!(base.cache.max_entries, 1000);
        assert_eq!(base.cache.ttl_secs, 3600);
        assert!(base.route_policies.is_empty());
        assert!(base.pricing.is_empty());

        let v4_text = r#"
[server]
tcp_listen = "127.0.0.1:0"
enable_admin = true

[defaults]
model = "coding/default"
schema_repair_retries = 2

[cache]
enabled = true
max_entries = 500
ttl_secs = 1200
max_temperature = 0.1
persist = true

[[backends]]
id = "mock"
kind = "cli"
command = "/bin/echo"
stdin = true

[[backends.profiles]]
id = "acct-a"
weight = 3
env = { HOME = "/tmp/home-a", API_TOKEN = "sk-profile-secret-123" }

[[backends.profiles]]
id = "acct-b"
weight = 1
env = { HOME = "/tmp/home-b" }

[routes]
"coding/default" = ["mock"]

[route_policies."coding/default"]
strategy = "hedged"
hedge_delay_ms = 1500

[pricing."coding/default"]
input_per_1m_usd = 3.0
output_per_1m_usd = 15.0

[[mcp_prompts]]
name = "code_review"
description = "Review code"
template = "Review {{lang}}: {{code}}"

[[mcp_prompts.arguments]]
name = "lang"
required = true

[[mcp_prompts.arguments]]
name = "code"
required = true

[[mcp_upstreams]]
id = "fs"
transport = "stdio"
command = "echo"
args = ["hello"]

[[tokens]]
token = "vouch-live-admin-secret"
name = "ops"
rpm = 60
tpm = 100000
max_concurrent = 4
expires_at = "2099-12-31T23:59:59Z"
allowed_backends = ["mock"]
admin = true
"#;
        let parsed = GatewayConfig::from_toml_str(v4_text).expect("valid v4 config");
        assert!(parsed.server.enable_admin);
        assert_eq!(parsed.defaults.schema_repair_retries, 2);
        assert!(parsed.cache.enabled);
        assert_eq!(
            parsed.route_policies["coding/default"].strategy,
            RouteStrategy::Hedged
        );
        assert_eq!(parsed.backends[0].profiles.len(), 2);
        assert_eq!(parsed.tokens[0].tpm, Some(100_000));
        assert!(parsed.tokens[0].admin);
        assert!(
            parsed
                .secrets()
                .contains(&"sk-profile-secret-123".to_string())
        );
    }

    #[test]
    fn rejects_invalid_v4_fields() {
        let too_many_repairs = minimal("").replace(
            "[[backends]]",
            "[defaults]\nschema_repair_retries = 3\n\n[[backends]]",
        );
        assert!(GatewayConfig::from_toml_str(&too_many_repairs).is_err());

        let bad_expiry = minimal("").replace(
            "[routes]",
            "[[tokens]]\ntoken = \"vouch-secret-token\"\nexpires_at = \"not-a-date\"\n\n[routes]",
        );
        assert!(GatewayConfig::from_toml_str(&bad_expiry).is_err());

        let zero_tpm = minimal("").replace(
            "[routes]",
            "[[tokens]]\ntoken = \"vouch-secret-token\"\ntpm = 0\n\n[routes]",
        );
        assert!(GatewayConfig::from_toml_str(&zero_tpm).is_err());
    }
}
