# Changelog

All notable changes to `vouch` are documented in this file.

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
