# Changelog

All notable changes to `vouch` are documented in this file.

## [6.0.0] - 2026-11-28

### Added
- **Cross-Agent Shared Memory Ledger (`[shared_memory]`, `src/memory/ledger.rs`, `src/memory/handoff.rs`):**
  - Added a bounded, secret-redacted `SessionLedgerStore` tracking canonical turns, active backend/profile, backend handoff history, sanitized repository `touched_files`, checkpoints, and token totals per `session_id`, persisted cleanly inside `[defaults].state_dir` (`gateway_state.json`).
  - Added deterministic zero-LLM-token `<vouch_shared_context>` synthesis on cross-agent transitions (`failover`, `policy_reorder`, `model_switch`) strictly bounded to `shared_memory.max_context_tokens` BPE tokens via `tiktoken`, clearing native CLI delta state (`delta_messages = None`) cleanly and emitting `X-Vouch-Handoff: true` + `X-Vouch-Handoff-From` across streaming and buffered protocols.
- **Model-Free `trace` MCP Codebase Bridge (`[trace]`, `src/memory/trace_bridge.rs`):**
  - Added pooled `trace serve` stdio JSON-RPC 2.0 workers keyed by normalized workspace root with `ProcessGroupGuard` process-tree cleanup, strict startup/query timeouts, graceful non-blocking degradation when `trace.required = false`, and `list_rules` / `search_decisions` / `get_symbol_outline` / `find_callers` / `get_recent_history` grounding plus async `record_session` checkpoints.
  - Enforced canonical traversal/containment checks on `X-Workspace-Root` (`403 Forbidden` on `..` or out-of-root escapes) and sanitized relative repository file extraction (`sanitize_relative_repo_path`).
- **Session Control Endpoints, MCP Tools & Observability (`src/ingress/sessions.rs`, `src/cli/`):**
  - Added authenticated `GET /v1/sessions`, `GET /v1/sessions/{session_id}`, `POST /v1/sessions/{session_id}/checkpoint`, and `DELETE /v1/sessions/{session_id}` (clearing ledger, sticky affinity, and git worktree).
  - Added `vouch_session_status` and `vouch_checkpoint` built-in MCP tools (scoped by `[[tokens]].allowed_mcp_tools`), `[trace]` health probing in `vouch doctor`, `shared_sessions` + `handoffs_total` in `GET /healthz` and `vouch top`, and Prometheus counters `vouch_handoffs_total{from,to,reason}` and `vouch_trace_queries_total{tool,status}`.
- **Cross-Platform CI & Multi-Distro Releases (`.github/workflows/`, `packaging/`):**
  - Added Windows CI checks and `x86_64-pc-windows-msvc` `.zip` release artifacts (plus `cargo-binstall` ZIP override).
  - Added automated Linux native package builds via `packaging/nfpm.yaml` for Debian/Ubuntu (`.deb`), Fedora/RHEL (`.rpm`), and Arch Linux (`.pkg.tar.zst`) across `x86_64`/`amd64` and `aarch64`/`arm64`, alongside `packaging/arch/PKGBUILD` and `packaging/homebrew/vouch.rb`.

## [5.0.0] - 2026-11-28

### Added
- **Singleflight In-Flight Request Coalescing (`src/cache.rs`, `src/ingress/mod.rs`):**
  - Added `cache.coalesce_inflight` (defaults to `true` when `cache.enabled = true`, configurable independently) and `cache.normalize_whitespace` (default `false`).
  - Deduplicates concurrent identical deterministic non-streaming requests through a `tokio::sync::watch` singleflight slot so only one backend execution runs; secondary callers receive the shared response with `X-Vouch-Coalesced: true` and increment `vouch_coalesced_requests_total`.
- **Outbound Egress Secret Guardrails & Output Filters (`src/security/egress_guard.rs`, `src/driver/`):**
  - Added `[security.egress_guard]` (`mode = "off" | "redact" | "block"`, `block_private_keys = true`, `extra_patterns`) inspecting prompts, messages, tool arguments, and `/v1/embeddings` inputs before dispatching upstream, emitting `vouch_egress_guard_actions_total{action}` and returning `403 egress_policy_blocked` on `block`.
  - Added `[[backends.output_filters]]` (`pattern`, `replace`) regex filtering across buffered and streaming `cli`/`pty` outputs.
- **OpenAI `/v1/embeddings` Ingress (`src/ingress/embeddings.rs`):**
  - Added `POST /v1/embeddings` (`string` / `string[]` inputs, `"float"` and IEEE-754 little-endian `"base64"` formats) backed by `Driver::embed` across OpenAI `/v1/embeddings`, Ollama `/api/embed`, and `cli` JSON vector backends.
- **Opt-In Redacted Flight Recorder & `vouch replay` (`src/recorder.rs`, `src/cli/replay.rs`):**
  - Added `[flight_recorder]` (`enabled`, `max_records = 500`, `path`) asynchronously writing secret-scrubbed JSONL logs with rotation.
  - Added `vouch replay --config vouch.toml [--file ...] [--route ...] [--limit 20] [--json]` to replay captured requests and validate schema/tool contracts.
- **Self-Healing `vouch doctor --fix` & `vouch config diff` (`src/cli/doctor.rs`, `src/cli/config_cmd.rs`):**
  - Added `vouch doctor --fix [--yes]` to repair missing CLI paths and unreachable local daemon URLs cleanly with schema validation and atomic `0600` writes.
  - Added `vouch config diff --current ... --candidate ... [--json]` for semantic TOML diffing without exposing raw secrets.
- **Interactive Console v2 (`GET /ui`) & Trace Ring Buffer (`GET /v1/traces`):**
  - Added an authenticated `GET /v1/traces` ring buffer (`100` recent redacted entries) and upgraded `GET /ui` with **Topology & Health** (plus one-click circuit reset), **Route Playground**, and **Trace Waterfall** tabs.

### Fixed
- **Hedged Routing Attribution & Non-Blocking Loser Cancellation:** Tracked winning `profile_id` alongside `backend_id` through `CandidateOutcome` / `RoutedResponse`, `StreamUsageGuard`, `[pricing]` (`backend:profile` lookup), and trace spans while cancelling losing hedged tasks cleanly in background tasks.
- **Pre-Execution Token Budget Reservations:** Added atomic `TokenBudgetReservation` (`TokenPolicy::check_and_reserve_tokens` / `settle_tokens`) across `daily_token_budget` and 60 s `tpm` so concurrent bursts cannot overshoot quotas before streams finish.
- **Crash-Orphan Temp/Worktree Sweep & Graceful Drain:** Tagged `vouch_att_*`, `vouch_prompt_*`, and `.vouch.pid` worktree sidecars with creator PIDs, swept orphaned artifacts on `vouch serve` startup, and added `server.shutdown_drain_secs` (default `15`) to stop accepting new connections before cancelling active tasks.
- **Cross-Protocol Tool & Multimodal Fidelity:** Preserved `tool_call_id` and image placeholders in `tool_bridge::format_message`, mapped Gemini `thought: true` parts and incremental streaming `functionCall` deltas, and mapped Ollama `/api/chat` inbound `tool_calls`, `tool_call_id`, and flexible `options`.
- **Pre-Release Security, Memory & Cross-Platform Hardening:** Enforced `Content-Type: application/json` on `POST` API endpoints (`415`) and cross-origin `Origin` rejection on `/v1/responses` WebSocket upgrades (`403`), tightened SSRF redirect cap (`3`) and `0.0.0.0/8` + IPv6 blocklists, blocked worktree traversal session keys, added `--` option terminators in `init` CLI templates, added `[[tokens]].allowed_mcp_tools` scoping and upstream MCP `error.data` redaction, held hedged loser permits until child reap finishes, capped unclosed `<vouch_tool_call>` blocks (1 MiB) and single-line subprocess buffers (2 MiB) with `ProcessFailure`, bounded `previous_response_id` depth (`64`) with cycle detection, added `cache.max_coalesce_waiters` (default `64`), added Win32 Job Object process-tree termination + `.cmd`/`.bat` spill + Windows service guidance, enforced `< 100`-byte Unix socket validation with live-listener probing, and handled PTY `EIO` as clean EOF on Unix/macOS/BSD.

## [4.0.0] - 2026-11-27

### Added
- **Speculative Hedged Routing & EWMA Latency Scheduling (`[route_policies.<model>]`, `src/router/mod.rs`):**
  - Added `strategy = "ordered" | "least_latency" | "hedged"` and `hedge_delay_ms` (default `2500`).
  - `least_latency` dynamically ranks healthy backends using per-backend EWMA Time-To-First-Token measurements (`alpha = 0.3`).
  - `hedged` speculatively dispatches the backup candidate after `hedge_delay_ms` if the primary has not returned content, cancels/reaps the slower loser via its PGID without faulting its circuit breaker, and exports `vouch_hedged_wins_total{winner,loser}`.
- **Deterministic SHA-256 Response Cache (`[cache]`, `src/cache.rs`):**
  - Added `ResponseCache` with `enabled` (default `false`), `max_entries` (default `1000`), `ttl_secs` (default `3600`), `max_temperature` (default `0.2`), and optional `state_dir` snapshot `persist` (`false`).
  - Keys normalized `(route, system, messages, tools, response_format, temperature, max_tokens)`, bypasses on `Cache-Control: no-cache` / `no-store`, high temperature, or git worktree isolation, replays buffered or synthesized SSE/NDJSON streams cleanly (`X-Vouch-Cache: HIT|MISS|BYPASS`), and emits `vouch_cache_requests_total{outcome}`.
- **Google Gemini & Ollama Ingress Compatibility (`src/ingress/gemini.rs`, `src/ingress/ollama.rs`):**
  - Added `POST /v1beta/models/{model}:generateContent` and `:streamGenerateContent` (supporting `?alt=sse`, chunked JSON arrays, `x-goog-api-key`, `?key=`, `systemInstruction`, `inlineData`, `functionCall`/`functionResponse`, and `generationConfig`).
  - Added `GET /api/tags`, `POST /api/chat`, and `POST /api/generate` with NDJSON streaming (`stream: true` default), `images`, `format`, `options`, and `tools`.
- **Multi-Profile Credential Rotation (`[[backends.profiles]]`):**
  - Supports multiple credential profiles (`id`, `env`, `weight`) per `cli`/`pty` backend with weighted round-robin, sticky session profile affinity, `VOUCH_PROFILE_ID` / `{profile_id}` injection, and independent per-profile circuit breakers failing over cleanly on `429`/auth expiry before tripping the parent backend.
- **Per-Token TPM, Expiry, Concurrency Caps, Backend Allowlists & Pricing (`[pricing.<id>]`):**
  - Added `tpm`, `max_concurrent` (`TokenConcurrencyGuard`), `expires_at` (RFC-3339), `allowed_backends`, and `admin` on `[[tokens]]`.
  - Added `[pricing.<route_or_backend>]` (`input_per_1m_usd`, `output_per_1m_usd`) computing cumulative `total_cost_usd` in `GET /v1/usage`, `vouch stats`, and `vouch top [--token]`.
- **Opt-In Runtime Admin Plane (`server.enable_admin`, `src/ingress/admin.rs`):**
  - Added `POST /admin/reload`, `POST /admin/circuits/{backend}/reset`, and `DELETE /admin/cache` gated behind `server.enable_admin` and restricted to `admin = true` tokens (or loopback TCP / Unix sockets when tokens are disabled).
- **MCP Resources, Prompts & Upstream Aggregation (`src/cli/mcp.rs`):**
  - Added `resources/list` + `resources/read` (`vouch://health`, `vouch://routes`, `vouch://usage`), configurable `prompts/list` + `prompts/get` (`[[mcp_prompts]]`), and `[[mcp_upstreams]]` (`stdio`/`http`) tool discovery & proxying under `<upstream>__<tool>`.

### Fixed
- **Stream-Drop Token Accounting Guard (`StreamUsageGuard`):** Wrapped every OpenAI, Anthropic, Responses (HTTP + WS), Gemini, and Ollama stream in an RAII `Drop` guard so mid-stream client disconnects and stream timeouts count emitted tokens with `tiktoken` and debit `UsageTracker`, UTC daily budgets, and 60 s TPM windows exactly once.
- **Extended JSON Schema Validation & Auto-Repair Retries:** Validates nested `type` (single/union), `required`, `properties`, `enum`, `items`, and `additionalProperties`; added `defaults.schema_repair_retries` (`0..=2`, default `0`) for automated CLI/PTY validation feedback retries.
- **Parallel Multi-Tool Call Streaming:** Verified `StreamingToolParser` streams back-to-back, chunk-split, and array `<vouch_tool_call>` blocks with distinct `call_...` IDs and sequential indices (`0`, `1`, ...).
- **Advisory File Locking & Crash-Safe `state_dir` Writes:** Serialized snapshot reads/writes via `gateway_state.lock` (`lock_shared` / exclusive `lock`) alongside `File::sync_all` + parent directory `sync_all`.
- **Hardened Secret & Multimodal Payload Redaction:** Scrubs `vouch_live_` keys, credentials inside JSON-escaped quotes (`\"...\"`), and truncates `data:image/<subtype>;base64,<payload>` blobs to `[REDACTED_BYTES]`.

## [3.0.0] - 2026-11-27

### Added
- **WebSocket `/v1/responses` (`src/ingress/responses.rs`):**
  - Enabled Axum WebSocket upgrades on `GET /v1/responses` with virtual-token handshake authentication, `X-Session-ID` / `X-Workspace-Mode` support, `response.create` & `response.cancel` client frames, real-time server JSON frames (`response.created`, `response.output_text.delta`, `response.function_call_arguments.delta`, `response.completed`, `error`), and immediate child PGID reaping on cancellation or socket disconnect.
- **Model aliases & dynamic trait-based routing rules (`src/config.rs`, `src/router/mod.rs`):**
  - Added `[aliases]` mapping friendly public names (e.g. `"gpt-4o"`) to canonical `[routes]` entries, exposing aliases in `GET /v1/models` and checking `allowed_models` against both requested alias and resolved route.
  - Added `[[routing_rules]]` evaluated prior to static routes (`model_pattern`, `has_tools`, `has_images`, `min_prompt_tokens`, `header_equals` → `target_route`).
- **Per-token daily BPE budgets & cumulative usage analytics (`src/auth/tokens.rs`, `src/usage.rs`, `src/cli/stats.rs`):**
  - Added `daily_token_budget` on `[[tokens]]` enforcing rolling UTC-day prompt + completion BPE token caps (`429 rate_limit_error`, code `daily_token_budget_exceeded`, `Retry-After` until UTC midnight).
  - Added `UsageTracker`, `GET /v1/usage`, and `vouch stats [--url ... | --unix-socket ...] [--token ...] [--json]` reporting cumulative per-token/per-backend requests, errors, token counts, and `p50`/`p95`/`p99` latency quantiles without exposing raw secrets.
- **Durable `0600` state persistence & bounded stores (`src/persistence.rs`, `src/state.rs`):**
  - Added `defaults.max_session_entries` (default `10000`, oldest-entry eviction) across sticky session affinity and `/v1/responses` history stores.
  - Added `defaults.state_dir` writing atomic `0600` snapshots (`gateway_state.json`) periodically and on shutdown, reloading non-expired sessions, responses, daily token budgets, and usage stats on startup.
- **SSRF-safe remote `image_url` materialization (`[attachments]`):**
  - Added `fetch_remote_urls` (default `false`), `max_image_bytes` (default `10485760`), and `allow_private_ips` (default `false`), validating resolved/redirected IPs, enforcing bounded downloads, pinning DNS addresses, checking image magic bytes, and cleaning temp files via `TempAttachmentGuard`.
- **OpenTelemetry OTLP/HTTP trace export (`[telemetry]`, `src/telemetry.rs`):**
  - Added `otlp_endpoint`, `service_name` (default `"vouch"`), and `otlp_headers`, extracting W3C `traceparent`/`tracestate` across HTTP & MCP ingress and exporting non-blocking redacted spans (`ingress`, `router.dispatch`, `driver.exec`, `workspace.worktree`).
- **`vouch init` scaffolder & distribution packaging (`src/cli/init.rs`, `packaging/homebrew/vouch.rb`, `Cargo.toml`):**
  - Added `vouch init [--output vouch.toml] [--docker-bridge] [--force]` scanning host CLIs (`hermes`, `opencode`, `kilocode`, `claude`, `codex`, `gemini`) and local daemons (`:8645`, `:4096`), detecting `docker0` IPv4, generating CSPRNG tokens, and emitting a validated config.
  - Added `[package.metadata.binstall]` in `Cargo.toml`, Homebrew formula `packaging/homebrew/vouch.rb`, and release workflow SHA-256 sync.

### Fixed
- **Real-time UTF-8 chunk streaming (`read_mode = "chunk" | "line"`):** Replaced buffered whole-turn waiting with `Utf8ChunkDecoder` carry-over across reads and immediate chunk streaming in `src/driver/cli_pipe.rs` and `src/driver/pty_runner.rs`, resetting idle watchdogs on every byte received without waiting for `\n`.
- **Split-tag tool calls & streaming Markdown fence stripping (`src/driver/tool_bridge.rs`):** Added `StreamingToolParser` (buffering `<vouch_tool_call>` prefixes across chunk boundaries without leaking XML) and `StreamingJsonFenceStripper` (stripping ```` ```json ```` / ```` ``` ```` across chunks and validating JSON/Schema before terminal `stop`).

## [2.0.0] - 2026-11-27

### Added
- **Multimodal content parts across OpenAI, Anthropic, Responses & CLI drivers (`src/driver/attachments.rs`):**
  - Replaced plain-string message content with structured `MessageContent` (`Text` | `Parts(Vec<ContentPart>)`) supporting `Text`, `ImageBase64`, and `ImageUrl`.
  - HTTP backends forward structured multimodal arrays intact. CLI/PTY backends materialize base64 images to private `0600` temp files (`vouch_att_<uuid>.<ext>`) with RAII `Drop` cleanup and append `[Attached image file: <path>]` markers.
- **Structured outputs (`response_format` & `text.format`):**
  - Parses OpenAI `response_format` (`json_object` and `json_schema`) and Responses `text.format`. Forwards to HTTP upstreams, appends strict schema instructions to CLI/PTY prompts, strips accidental Markdown code fences, and validates JSON + schema structure (`502 upstream_error` on failure).
- **Native CLI multi-turn session delta continuation:**
  - Added `resume_args` and `session_id_regex` to CLI/PTY backend configuration. On turns 2+ sharing `X-Session-ID`, invokes `resume_args` with `{backend_session_id}` and only messages after the last assistant turn.
- **Stateful `/v1/responses` chaining (`previous_response_id` & `store`):**
  - Added an in-memory TTL-evicted `ResponseStore` backing `previous_response_id` history chaining (including `function_call` / `function_call_output` turns, `404` on unknown ids, and `store = false` Zero Data Retention).
- **Dual-spec MCP `2026-07-28` server & async tasks (`src/cli/mcp.rs`):**
  - Added stateless `server/discover`, `Mcp-Method` / `Mcp-Name` HTTP header validation, W3C Trace Context `_meta` extraction (`traceparent`, `tracestate`), `tools/list` caching metadata (`ttlMs`, `cacheScope: "private"`), and background `tasks/create`, `tasks/get`, `tasks/result`, and `tasks/cancel`.
- **Anthropic extended thinking & prompt caching:**
  - Parses and forwards `thinking` (`enabled` / `budget_tokens`) and `cache_control` metadata, rendering `thinking`, `redacted_thinking`, `thinking_delta`, and `signature_delta` blocks from upstream traces.
- **Idle stall watchdog, PTY sizing & auto-respond (`src/config.rs`, `src/driver/pty_runner.rs`):**
  - Added `idle_timeout_secs`, `pty_cols` (default `160`), `pty_rows` (default `48`), and `[[backends.pty_auto_respond]]` interactive prompt rules.
- **Operational CLI subcommands & extended redaction:**
  - Added `vouch token create` (`src/cli/token.rs`) generating 256-bit `vouch_live_<64_hex>` TOML blocks via OS CSPRNG.
  - Added `vouch bench` (`src/cli/bench.rs`) for TTFT, p50/p95 latency, throughput, and fallback benchmarking.
  - Extended log redaction (`src/telemetry.rs`) for URL query parameters (`?key=`, `?api_key=`, `?token=`) and `access_token` / `refresh_token` JSON fields.

### Fixed
- **Large prompt `ARG_MAX` (`E2BIG`) protection:** Added RAII `PromptSpillFile` (`max_inline_prompt_bytes`, default `65536`) spilling oversized non-stdin CLI/PTY prompts to `0600` temp files and cleaning up automatically on drop.
- **Pre-content stream buffering (`prime_stream`):** Buffers leading empty/whitespace/ANSI noise chunks before committing to a streaming backend so early non-zero exits or errors transparently fail over to the next candidate.
- **Auth-expiry vs. transient rate-limit classification:** Distinctly classifies `401`/`403` and CLI auth-expiry stderr (`UpstreamAuthExpired`), surfaces `auth_expired` in `/readyz`, `GET /ui`, and `vouch top` (`AUTH_EXPIRED`), and parses HTTP `Retry-After` headers on `429` responses to override breaker cooldown dynamically.

## [1.0.0] - 2026-11-27

### Added
- **Structured tool calling across all drivers (`http`, `cli`, `pty`):**
  - Unified `tools`, `tool_choice`, `tool_calls`, and `tool` / `tool_result` conversation turn representations across ingress adapters and drivers.
  - HTTP proxy driver forwards `tools` and `tool_choice` upstream and streams `tool_calls` deltas transparently.
  - `src/driver/tool_bridge.rs` translates tool schemas into deterministic CLI/PTY prompt instructions and extracts `<vouch_tool_call>...</vouch_tool_call>` blocks into OpenAI `tool_calls` (`finish_reason: "tool_calls"`), OpenAI Responses `function_call` items, or Anthropic `tool_use` blocks (`stop_reason: "tool_use"`).
- **Model Context Protocol (MCP) ingress (`vouch mcp` + `POST /mcp`):**
  - Added `vouch mcp --config vouch.toml [--transport stdio|http] [--bind 127.0.0.1:8081]` subcommand plus `POST /mcp` on the main daemon.
  - Supports `initialize`, `notifications/initialized`, `ping`, `tools/list` (`vouch_ask` + per-route `route_<name>` tools), and `tools/call` routed through the standard fallback, circuit-breaker, affinity, and telemetry pipeline.
  - `stdio` mode routes all tracing logs strictly to `stderr` so `stdout` carries pure JSON-RPC 2.0 lines.
- **OpenAI Responses API (`POST /v1/responses`):**
  - Accepts `input` (string or structured items), `instructions`, `tools`, and `stream`, emitting standard Responses API JSON objects or SSE lifecycle events (`response.created`, `response.output_item.added`, `response.output_text.delta`, `response.output_text.done`, `response.function_call_arguments.delta`, `response.completed`).
- **Accurate BPE token accounting (`tiktoken-rs`):**
  - Replaced character heuristics for CLI/PTY backends with cached `cl100k_base` / `o200k_base` BPE token counting (with panic-free char/4 fallback), while preserving upstream `usage` blocks on HTTP backends.
- **Live TUI (`vouch top`) & opt-in HTML status dashboard (`GET /ui`):**
  - Added `vouch top [--url ... | --unix-socket ...] [--interval-ms ...]` built with `ratatui` + `crossterm`, displaying uptime, backend readiness, circuit-breaker states, active permits, requests, errors, fallbacks, and token throughput.
  - Added `[server].enable_ui` (default `false`) to optionally serve a read-only HTML dashboard at `GET /ui`.
- **Configurable `doctor` smoke probe timeout:**
  - Added `[discovery].probe_timeout_secs` (default `30`, validated `> 0`) so slow cold-start Node/TypeScript CLIs complete smoke probes cleanly while HTTP probes stay at 5 s.
- **GitHub Actions CI/CD:**
  - Added `.github/workflows/ci.yml` (`ubuntu-latest` and `macos-latest`: `fmt`, `clippy`, `test`, `build`) and `.github/workflows/release.yml` (tagged cross-platform release tarballs + `SHA256SUMS.txt`).

### Changed
- Standardized macOS LaunchAgent identifier and filename to `com.ayoubzulfiqar.vouch` (`~/Library/LaunchAgents/com.ayoubzulfiqar.vouch.plist`).
- Documented Docker-to-host bridge networking (`docker0`, `extra_hosts: ["host.docker.internal:host-gateway"]`, and orchestrator SSRF private-endpoint allowlists).

## [0.1.0] - 2026-11-26

### Added
- Initial release of `vouch`: OpenAI (`/v1/models`, `/v1/chat/completions`) and Anthropic (`/v1/messages`) ingress adapters, HTTP/CLI/PTY egress drivers, ordered fallback routing, circuit breakers, sticky session affinity, git worktree isolation, virtual ingress tokens, Prometheus `/metrics`, secret-redacted JSON logging, `vouch doctor`, `vouch routes`, `vouch service`, and hot reload (`SIGHUP` + file watcher).
