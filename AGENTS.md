# AGENTS.md — contributing to `vouch`

`vouch` is a local authentication multiplexer and process-supervised gateway for
AI orchestrators. It exposes OpenAI `GET /v1/models` and `POST
/v1/chat/completions`, plus Anthropic `POST /v1/messages`, over TCP and an
optional Unix socket. It routes public model names to ordered fallback chains
of HTTP/SSE daemons, headless CLIs, and PTY subprocesses.

Rust 2024, single binary crate with a library target. No database. All durable
state is the TOML file plus ephemeral worktrees.

## Architecture map

| Module | Responsibility |
| --- | --- |
| `src/config.rs` | Typed TOML schema, validation, `ConfigStore` (ArcSwap snapshot) |
| `src/error.rs` | The single `GatewayError` type and `Result` alias |
| `src/memory/mod.rs` | Shared cross-agent memory ledger (`SessionLedgerStore`), deterministic `<vouch_shared_context>` handoff (`SharedMemoryEngine`), and `[trace]` MCP bridge (`TraceBridge`) |
| `src/cache.rs` | Deterministic SHA-256 response cache (`ResponseCache`) with TTL, eviction, & persistence |
| `src/usage.rs` | `tiktoken-rs` BPE token counter (`cl100k_base` / `o200k_base`), usage, cost & latency analytics |
| `src/persistence.rs` | Lock-guarded atomic `0600` snapshots for sessions, responses, cache, budgets, and usage |
| `src/recorder.rs` | Opt-in bounded redacted JSONL flight recorder (`[flight_recorder]`) |
| `src/security/` | Outbound secret/PII egress guardrails (`[security.egress_guard]`) |
| `src/cli/mod.rs` | clap definitions for `serve`, `routes`, `doctor`, `service`, `mcp`, `top`, `token`, `bench`, `stats`, `init`, `replay`, `config` |
| `src/cli/config_cmd.rs` | Semantic config diff subcommand (`vouch config diff`) |
| `src/cli/replay.rs` | Offline flight-log replay & schema benchmark runner (`vouch replay`) |
| `src/cli/mcp.rs` | Dual-spec MCP JSON-RPC 2.0 server (`2024-11-05` + `2026-07-28`, `resources/*`, `prompts/*`, upstreams, async `tasks/*`) |
| `src/cli/top.rs` | Live terminal TUI (`vouch top`) polling `/healthz`, `/readyz`, `/metrics`, and `/v1/usage` |
| `src/cli/token.rs` | CSPRNG virtual token generator (`vouch token create`) |
| `src/cli/bench.rs` | End-to-end route latency/TTFT benchmark runner (`vouch bench`) |
| `src/cli/stats.rs` | CLI query & formatter for `GET /v1/usage` (`vouch stats`) |
| `src/cli/init.rs` | Host CLI/port/docker0 auto-discovery & config scaffolder (`vouch init`) |
| `src/ingress/mod.rs` | Shared auth/egress-guard/singleflight/caching/`StreamUsageGuard`/routing/affinity/trace/multimodal/SSE plumbing |
| `src/ingress/openai.rs` | OpenAI Chat Completions wire-format adapter (`tools`, `response_format`, multimodal) |
| `src/ingress/embeddings.rs` | OpenAI `POST /v1/embeddings` (`float` & `base64` vectors) adapter |
| `src/ingress/responses.rs` | OpenAI Responses (`POST` + `WS /v1/responses`) adapter with stateful `previous_response_id` store |
| `src/ingress/anthropic.rs` | Anthropic Messages adapter (`tools`, `tool_use`, `thinking`, `cache_control`, multimodal) |
| `src/ingress/gemini.rs` | Google Gemini API adapter (`POST /v1beta/models/{model}:{generateContent,streamGenerateContent}`, `thought` & streaming `functionCall`) |
| `src/ingress/ollama.rs` | Ollama API adapter (`GET /api/tags`, `POST /api/chat`, `POST /api/generate`, `tool_calls` & `options`) |
| `src/ingress/sessions.rs` | Cross-agent shared session ledger API (`GET/DELETE /v1/sessions`, `POST /v1/sessions/{id}/checkpoint`) |
| `src/ingress/admin.rs` | Opt-in `server.enable_admin` control plane (`POST /admin/reload`, circuit reset, `DELETE /admin/cache`) |
| `src/ingress/ui.rs` | Optional `server.enable_ui`-gated embedded Console v2 (`GET /ui`: topology, playground, waterfall) |
| `src/ingress/ops.rs` | `/healthz`, `/readyz`, `/metrics` (unauthenticated) and `/v1/usage`, `/v1/traces` (authenticated) |
| `src/auth/tokens.rs` | Virtual tokens, constant-time compare, RPM/TPM/concurrency limits, expiry, model/backend allowlists, budgets |
| `src/auth/discovery.rs` | Read-only local binary/auth-file/daemon discovery for `doctor` |
| `src/router/mod.rs` | Route table, policies (`ordered`/`least_latency`/`hedged`), profiles, repair retries, sticky sessions |
| `src/router/circuit_breaker.rs` | Per-backend and per-profile closed/half-open/open state machine (`auth_expired`, `Retry-After`) |
| `src/driver/mod.rs` | `Driver` trait, `Utf8ChunkDecoder`, multimodal `MessageContent`, `prime_stream`, permits, PGID guard |
| `src/driver/attachments.rs` | RAII `0600` temp spillover, SSRF-safe remote image fetch, & multimodal image cleanup |
| `src/driver/tool_bridge.rs` | Parallel `StreamingToolParser`, `StreamingJsonFenceStripper`, extended JSON Schema validator |
| `src/driver/http_proxy.rs` | HTTP/SSE proxy backend |
| `src/driver/cli_pipe.rs` | stdin/stdout subprocess backend (`read_mode` chunk/line, stall watchdog, spill files, profile env) |
| `src/driver/pty_runner.rs` | `portable-pty` subprocess backend with configurable dimensions, auto-respond rules, and profile env |
| `src/driver/workspace.rs` | git worktree lifecycle |
| `src/telemetry.rs` | JSON tracing, escaped-quote/base64 secret redaction, W3C `TraceContext`, and OTLP/HTTP exporter |
| `src/metrics.rs` | Prometheus recorder and all `vouch_*` instruments |
| `src/state.rs` | `AppState`: config, router, cache, tokens, workspaces, usage, otel, persistence, shutdown, reload |

### Request lifecycle

1. axum middleware records protocol/endpoint/status/latency.
2. The adapter deserializes its payload; `ingress::prepare` authenticates the
   token, resolves the public model, applies `X-Session-ID` (or hashes system +
   first user turn) for affinity, and parses `X-Workspace-Mode`.
3. `ExecutionRequest` is normalized; `server.request_timeout_secs` wraps the
   whole routed execution.
4. `Router::execute` loads an immutable table snapshot, promotes the sticky
   backend to the front of the chain, and walks it. Circuit-open, saturated,
   or failing backends are skipped; a failure before a stream starts fails over
   transparently.
5. A driver acquires a bounded concurrency permit, executes, and returns a
   complete `Completion` or a bounded `ChunkStream`.
6. The adapter renders OpenAI or Anthropic JSON/SSE and attaches
   `x-vouch-backend`, `x-vouch-fallback`, `x-vouch-request-id`.

## Conventions that must hold

### No `unwrap`/`expect` in runtime paths

Fallible work returns `Result<T>` with `GatewayError` variants. Convert with
`ok_or_else`, `map_err`, `?`, or `unwrap_or(_default)` for genuinely benign
fallbacks. `expect`/`unwrap` are acceptable only in `#[cfg(test)]` code and for
static, provably infallible initialization (for example the `LazyLock` regexes
in `telemetry.rs`). Never let a malformed request, bad upstream, or missing
env var panic the daemon. Client-visible error strings go through
`telemetry::redact_text`.

### Bound every channel and buffer

- Stream bridges use `mpsc::channel::<Result<StreamChunk>>(32)`, never an
  unbounded channel. A slow SSE consumer applies backpressure through the
  `PermitGuard` held for the stream's lifetime.
- Subprocess stdout is capped at 16 MiB, captured stderr at 8 KiB, collected
  streams at 16 MiB, unclosed `<vouch_tool_call>` blocks at 1 MiB, single-line
  `read_mode = "line"` / `json_lines` buffers at 2 MiB, `/v1/responses` chains at
  64 hops, singleflight waiters by `cache.max_coalesce_waiters` (default 64),
  and request bodies by `server.max_body_bytes`.
- `PermitGuard::acquire` waits at most `queue_timeout_ms`, honors client
  cancellation, releases on `Drop` (held by subprocess drivers until child reap),
  and updates the active-permit gauge.

### PGID & Job Object supervision for every child process

On Unix, subprocess backends spawn with `process_group(0)` so PGID == PID; on
Windows they attach to a `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` Win32 Job Object.
Move a `ProcessGroupGuard` into the task that owns the child: `disarm()` on
clean exit, `terminate()` on timeout, I/O error, cancellation, or dropped stream.
Unix termination sends `SIGTERM` to the whole group, waits 500 ms, then escalates
to `SIGKILL` if members remain — grandchildren must never outlive a request.
PTY work is blocking (`portable-pty`) so it runs inside `spawn_blocking` and
communicates over bounded channels with the same guard (`EIO` on Unix handled as
clean EOF).

### Hot reload via ArcSwap (never tearing down in-flight work)

`ConfigStore`, `Router::table`, `TokenRegistry::policies`, and
`RedactionRegistry::secrets` are `ArcSwap`-backed. A reload is atomic:
validate the file, rebuild token policies, rebuild the router table, update
workspace settings and redaction secrets, then `store_snapshot` last. If any
step fails, keep the previous snapshot. Circuit breakers are reused by backend
id so cooldown state survives reload. Changing `server.tcp_listen` or
`server.unix_socket` requires a restart — log that explicitly. Readers keep
their `Arc` snapshot for the duration of a request.

### reqwest feature flag

`Cargo.toml` pins `reqwest` with `default-features = false` and features
`["json", "stream", "rustls"]`. Keep `rustls` — do **not** switch to
`rustls-tls` or enable default TLS. The gateway is a local daemon and must not
silently depend on system OpenSSL.

### License headers

`vouch` is `GPL-3.0-or-later`; see [LICENSE](LICENSE), [NOTICE](NOTICE), and
"Choosing a license" in [README.md](README.md). Add
`//! SPDX-License-Identifier: GPL-3.0-or-later` to new source files. Never copy
in code whose license is incompatible with GPLv3, and never paste credentials,
user prompts, or third-party snippets with unknown provenance into source,
tests, docs, or fixtures.

### Config is a strict contract

Every schema struct uses `#[serde(deny_unknown_fields)]`. Adding a field means
updating the struct, its `Default`/default fn, `vouch.example.toml`, and this
repo's docs together. Re-validate the example with
`GatewayConfig::from_toml_str` after any change. Validation invariants live in
`GatewayConfig::validate` / `BackendConfig::validate`: non-empty backends and
routes, unique backend ids, known/no-duplicate route ids, valid globs, tokens
at least 8 chars, and workspace consistency.

### Secrets

`GatewayConfig::secrets()` collects token values, secret-looking `env` values,
secret-looking `extra_headers`, and the value of `api_key_env`. Feed these to
`telemetry::set_global_secrets` whenever config installs or reloads. Never log
a token, prompt body, or upstream key. `/healthz`, `/readyz`, and `/metrics`
expose only ids and circuit state.

## Working on this repo

- Keep clap argument definitions in `src/cli/mod.rs`. `src/main.rs` dispatches
  those commands and owns the serve/listener/signal/hot-reload wiring:
  SIGINT/SIGTERM drain gracefully, SIGHUP and config-file changes reload
  atomically, and `doctor` exits `2` when a check is unhealthy.
- Keep adapters protocol-only: auth, model resolution, affinity, and fallback
  belong in shared code, not duplicated per wire format.
- Tests are colocated in `#[cfg(test)]` modules; `tempfile` is available as a
  dev-dependency. Run `cargo test` and `cargo build` before handing work off.
- Documentation lives in `README.md`; the runnable schema reference is
  `vouch.example.toml`. Keep both in sync with `src/config.rs`.
