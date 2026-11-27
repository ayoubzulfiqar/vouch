# Changelog

All notable changes to `vouch` are documented in this file.

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
