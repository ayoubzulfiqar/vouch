# vouch

A local authentication multiplexer and process-supervised gateway for AI orchestrators. Point any OpenAI- or Anthropic-speaking client at `vouch` and it inherits the authenticated CLI sessions and local daemons already present on the host — `hermes`, `opencode`, `claude`, `codex`, or your own commands — instead of asking you to paste raw API keys into another tool.

`vouch` runs under your host user account, binds only to loopback (and an optional `0600` Unix socket by default), and adds policy in the middle: routing, fallback chains, circuit breakers, sticky sessions, per-token quotas, worktree isolation, redacted logs, and process-group supervision.

- OpenAI-compatible: `GET /v1/models`, `POST /v1/chat/completions` (JSON and SSE), `POST /v1/responses` (JSON and SSE)
- Anthropic-compatible: `POST /v1/messages` (JSON and SSE)
- Structured tool calling: `tools` / `tool_calls` / `tool_use` bridged across HTTP, CLI, and PTY backends
- MCP ingress: `vouch mcp` (`stdio` / HTTP) and `POST /mcp` exposing every route as an MCP tool
- Operations & UI: `GET /healthz`, `GET /readyz`, `GET /metrics`, `vouch top` TUI, and opt-in `GET /ui` (`server.enable_ui = true`)
- Egress: pooled HTTP/SSE proxy, headless stdin/stdout subprocess, and PTY subprocess
- Reload: `SIGHUP` or config-file change atomically swaps configuration without dropping in-flight requests

---

## Contents

1. [How it works](#how-it-works)
2. [Requirements](#requirements)
3. [Quick start](#quick-start)
4. [Installation](#installation)
5. [Configuration](#configuration)
   - [Minimal configuration](#minimal-configuration)
   - [Server, defaults, resilience, workspace, discovery](#server-defaults-resilience-workspace-discovery)
   - [Backends](#backends)
   - [Routes](#routes)
   - [Tokens](#tokens)
   - [Validation rules](#validation-rules)
   - [Hot reload](#hot-reload)
6. [Transports, endpoints, and headers](#transports-endpoints-and-headers)
7. [API reference](#api-reference)
   - [OpenAI chat completions](#openai-chat-completions)
   - [OpenAI responses](#openai-responses)
   - [Anthropic messages](#anthropic-messages)
   - [Structured tool calling](#structured-tool-calling)
   - [MCP server ingress](#mcp-server-ingress)
   - [Error format](#error-format)
8. [Orchestrator integration](#orchestrator-integration)
   - [Docker containers and host bridges](#docker-containers-and-host-bridges)
9. [Egress drivers and process supervision](#egress-drivers-and-process-supervision)
10. [Resilience and session affinity](#resilience-and-session-affinity)
11. [Security](#security)
12. [Observability](#observability)
13. [CLI reference](#cli-reference)
14. [Running as a service](#running-as-a-service)
15. [Development](#development)
16. [Troubleshooting](#troubleshooting)
17. [Community, contributing, and security reporting](#community-contributing-and-security-reporting)
18. [License](#license)
    - [Choosing a license](#choosing-a-license)

---

## How it works

```
 orchestrator (OpenAI / Anthropic SDK)
        |
        | HTTP over 127.0.0.1:PORT  or  Unix socket (0600)
        v
 +-------------------------------------------------------------+
 | vouch ingress                                               |
 |   auth -> model resolve -> affinity -> body limit -> metrics |
 +-------------------------------------------------------------+
        |
        v
 +-------------------------------------------------------------+
 | router                                                      |
 |   sticky backend first -> ordered fallback chain             |
 |   skip circuit-open / saturated -> record outcome           |
 +-------------------------------------------------------------+
        |
        +--> http_proxy  : hermes proxy :8645 / opencode serve :4096
        +--> cli_pipe    : headless subprocess, stdin or {prompt} args
        +--> pty_runner  : portable-pty subprocess for TTY-bound CLIs
        |
        +--> workspace   : per-session git worktree (optional)
```

Every request is normalized into one `ExecutionRequest`, so the public wire format is independent of the backend that services it. A driver returns either a complete JSON object or a bounded stream of chunks; the adapter renders it back into the protocol the caller used.

Processes are supervised, not just spawned. Each child runs in its own process group; on timeout, upstream error, or client disconnect the whole group receives `SIGTERM`, then `SIGKILL` after a 500 ms grace period. Grandchildren cannot outlive a request.

## Requirements

- Rust 2024 toolchain (`rustup toolchain install stable` and `rustup update`)
- Linux, macOS, or another Unix for full process-group supervision (non-Unix builds omit the Unix-only signal/PGID paths)
- At least one configured backend: a reachable local HTTP daemon, or an executable on `PATH` (or an absolute path)
- Optional: `git` for workspace isolation, `systemd --user` or `launchd` for service installation

`reqwest` is built with `default-features = false` and the `rustls` feature — no system OpenSSL is required. Keep it that way; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick start

```bash
git clone https://github.com/ayoubzulfiqar/vouch.git vouch
cd vouch
cargo build --release
cp vouch.example.toml vouch.toml
$EDITOR vouch.toml

# Validate config and references before starting anything:
./target/release/vouch routes --config vouch.toml
./target/release/vouch doctor --config vouch.toml

# Run in the foreground:
./target/release/vouch serve --config vouch.toml
```

Point an orchestrator at it:

```bash
curl -s --unix-socket /run/user/1000/vouch.sock \
  -H 'content-type: application/json' \
  -d '{"model":"coding/default","messages":[{"role":"user","content":"hello"}]}' \
  http://localhost/v1/chat/completions
```

With no `[[tokens]]` entries, `vouch` runs in zero-friction local mode and accepts any credential. Once you add a token, every ingress request needs `Authorization: Bearer *** or `x-api-key`.

## Installation

### From source

```bash
cargo build --release
install -D -m 0755 target/release/vouch ~/.local/bin/vouch
```

Add `~/.local/bin` to `PATH`, then install the user service:

```bash
vouch service install --config "$HOME/.config/vouch/vouch.toml"
systemctl --user status vouch.service          # Linux
# launchctl print gui/$(id -u)/io.github.ayoubzulfiqar.vouch   # macOS
```

`service install` writes a unit under `$HOME` (never system-wide), defaults the executable to the current binary, and accepts `--binary` and `--no-start`. `service uninstall` stops/disables the unit unless `--no-stop` is given. See [Running as a service](#running-as-a-service).

### Verify the installation

```bash
vouch --version
vouch routes -c vouch.toml
vouch doctor -c vouch.toml --json
curl -s http://127.0.0.1:8080/healthz
```

## Configuration

The on-disk format is TOML. Every schema struct uses `#[serde(deny_unknown_fields)]`, so a typo fails validation at startup rather than being silently ignored. The authoritative, fully commented reference is [vouch.example.toml](vouch.example.toml); it parses cleanly with the crate's own loader.

### Minimal configuration

```toml
[defaults]
model = "coding/default"

[[backends]]
id = "local-agent"
kind = "cli"
command = "codex"
args = ["exec", "{prompt}"]
# prompt transport is satisfied by {prompt} in args, or by stdin = true

[routes]
"coding/default" = ["local-agent"]
```

Everything else has a default. Start from the example config in production — it documents each field, including the security-relevant ones.

### Server, defaults, resilience, workspace, discovery

| Section | Field | Default | Meaning |
| --- | --- | --- | --- |
| `server` | `tcp_listen` | `127.0.0.1:8080` | TCP bind address |
| | `unix_socket` | unset | Optional Unix socket path, created `0600` |
| | `request_timeout_secs` | `300` | Deadline for non-streaming requests, including workspace setup |
| | `stream_timeout_secs` | `900` | Maximum lifetime of one streaming response |
| | `keep_alive_interval_secs` | `15` | SSE keep-alive comment interval; `0` disables |
| | `max_body_bytes` | `8388608` | Maximum request body; must be > 0 |
| | `log_level` | `"info"` | Tracing filter; `RUST_LOG` wins |
| | `cors_permissive` | `false` | Permissive browser CORS |
| | `enable_ui` | `false` | Serve read-only HTML status dashboard at `GET /ui` |
| `defaults` | `model` | unset | Default public model; must be a route when set |
| | `session_ttl_secs` | `3600` | Idle TTL for sticky sessions and worktrees |
| | `max_concurrency` | `4` | Default per-backend concurrency |
| | `queue_timeout_ms` | `250` | Permit wait before failover |
| `resilience` | `failure_threshold` | `3` | Consecutive failures before a circuit opens |
| | `reset_timeout_secs` | `30` | Cooldown before a half-open probe |
| | `half_open_max_probes` | `1` | Concurrent half-open probes |
| | `open_on_rate_limit` | `true` | Open immediately on upstream `429` |
| `workspace` | `enabled` | `false` | Enable git worktree isolation |
| | `repo_path` | unset | Source repository; required when enabled |
| | `root` | `<temp>/vouch-worktrees` | Parent directory for worktrees |
| | `ttl_secs` | `3600` | Idle worktree lifetime |
| | `git_binary` | `"git"` | Git executable |
| | `cleanup_on_shutdown` | `true` | Remove all worktrees on shutdown |
| `discovery` | `enabled` | `true` | Inspect known auth files in `doctor` |
| | `probe_ports` | `true` | TCP-probe HTTP backends |
| | `probe_timeout_secs` | `30` | Live CLI/PTY smoke probe deadline in `doctor` (HTTP stays `5s`) |
| | `extra_auth_paths` | `[]` | Extra credential paths reported by `doctor` (metadata only) |

### Backends

At least one `[[backends]]` is required and ids must be unique.

| Field | Default | Notes |
| --- | --- | --- |
| `id` | required | Referenced by routes; stable across reloads |
| `kind` | required | `"http"`, `"cli"`, or `"pty"` |
| `max_concurrency` | `4` | Must be > 0; bounds parallel executions |
| `timeout_secs` | `300` | Per-execution deadline; must be > 0 |
| `queue_timeout_ms` | inherits `defaults` | Wait for a concurrency permit before failover |
| `isolate_workspace` | `false` | Requires `workspace.enabled = true` |
| `model` | requested model | Model name sent upstream |
| `base_url` | unset | **Required for `http`**, forbidden for `cli`/`pty` |
| `upstream_path` | `/v1/chat/completions` | HTTP chat path |
| `health_path` | unset | HTTP path used by `doctor` |
| `api_key_env` | unset | Environment variable holding the upstream bearer key |
| `extra_headers` | `{}` | Additional upstream headers |
| `command` | unset | **Required for `cli`/`pty`**, forbidden for `http` |
| `args` | `[]` | Substitutes `{prompt}`, `{model}`, `{session_id}`, `{request_id}` |
| `stdin` | `false` | Pipe the prompt to stdin; required unless `args` contains `{prompt}` |
| `env` | `{}` | Extra child environment variables |
| `workdir` | unset | Ignored while workspace isolation is active |
| `output_mode` | `"text"` | `"text"` (ANSI-stripped), `"json"`, or `"json_lines"` |
| `pty_send_eof` | `true` | Send EOT after the prompt for PTY readers |

CLI/PTY children also receive `VOUCH_REQUEST_ID`, `VOUCH_SESSION_ID`, and `VOUCH_MODEL`.

Prompts are templated as `arg.replace("{prompt}", prompt).replace("{model}", model)...`. For `output_mode = "json"`, stdout must be a JSON object with a `choices[0].message.content` or `content` field; `json_lines` reads newline-delimited objects and concatenates their text.

### Routes

A public model name maps to an **ordered** list of backend ids. The sticky backend for the session is moved to the front, then the chain is tried in order. Circuit-open, saturated, or failing backends are skipped; a failure before streaming begins fails over transparently.

```toml
[routes]
"coding/default" = ["codex-cli", "local-hermes", "interactive-pty"]
"coding/fast"    = ["local-hermes", "codex-cli"]
"general"        = ["interactive-pty", "local-hermes"]
```

At least one route is required, ids must resolve to configured backends, and a chain may not repeat an id.

### Tokens

| Field | Default | Notes |
| --- | --- | --- |
| `token` | required | At least 8 characters and unique |
| `name` | `token-<hash-prefix>` | Label used in logs and metrics |
| `rpm` | unlimited | Sliding 60-second request cap; must be > 0 when set |
| `allowed_models` | all models | Glob patterns, e.g. `["coding/*", "general"]` |

```toml
[[tokens]]
token = "replace-with-a-long-random-value"
name = "laptop-agent"
rpm = 60
allowed_models = ["coding/*", "general"]
```

Credentials are SHA-256 hashed and compared in constant time against every configured policy (so the matching index is not leaked by timing). Accepted forms: `Authorization: Bearer *** lowercase `bearer `), a bare token in `Authorization` with no spaces, and `x-api-key`.

**Important:** the RPM slot is consumed when the credential is authenticated, before the model allowlist is checked. A request rejected with `403 Forbidden` still counts toward that token's quota.

### Validation rules

`vouch routes -c vouch.toml` (and every `serve` startup) rejects:

- unknown fields anywhere in the document
- zero backends or zero routes
- duplicate backend ids or duplicate tokens
- route entries referencing unknown backend ids
- a chain that repeats the same backend
- `defaults.model` that does not name a route
- `cli`/`pty` backends without a prompt transport (`stdin = true` or `{prompt}` in `args`)
- `http` backends without `base_url`, or with a `command`
- `workspace.enabled = true` without `repo_path`, and `isolate_workspace = true` while workspace isolation is disabled
- tokens shorter than 8 characters, `rpm = 0`, or invalid glob patterns

### Hot reload

Send `SIGHUP` or edit the config file. A `notify` watcher debounces file events; both paths call the same atomic reload:

1. Parse and validate the new file.
2. Rebuild token policies and the routing table.
3. Update workspace settings and redaction secrets.
4. Publish the new snapshot last (`ArcSwap`).

If any step fails, the previous snapshot stays active and the error is logged; no request is dropped. Circuit breakers are reused by backend id so cooldown state survives. Changing `server.tcp_listen` or `server.unix_socket` is logged as requiring a restart — listeners are not rebound on reload.

## Transports, endpoints, and headers

Both transports expose the same router:

```bash
# TCP (default)
curl -s http://127.0.0.1:8080/healthz

# Unix domain socket
curl -s --unix-socket /run/user/1000/vouch.sock http://localhost/healthz
```

CLI overrides: `vouch serve -c vouch.toml --tcp-listen 127.0.0.1:9090 --unix-socket /tmp/vouch.sock`. The socket is created with mode `0600` and removed on graceful shutdown. A stale socket at that path is removed at startup.

| Header | Direction | Meaning |
| --- | --- | --- |
| `Authorization: Bearer *** / `x-api-key` | in | Credential when tokens are configured |
| `X-Session-ID` (or `X-Vouch-Session-ID`) | in | Sticky affinity key |
| `X-Workspace-Mode` | in | `auto` (default), `worktree`/`isolated`/`isolate`, or `none`/`off`/`disabled` |
| `x-request-id` | in | Safe correlation id (≤128 chars, ASCII alphanumeric or `-_.::`) |
| `x-vouch-backend` | out | Backend that served the request |
| `x-vouch-fallback` | out | `"true"` when an earlier candidate was skipped or failed |
| `x-vouch-request-id` | out | Effective correlation id |

An explicit session id is preserved verbatim when it is ≤128 characters and uses only ASCII alphanumeric, `-`, `_`, `.`, or `:`. Longer or unsafe values — and the absence of a header — are replaced with the first 8 bytes of a SHA-256 hash of the system prompt plus first user turn. This gives multi-turn orchestrators a deterministic affinity key without leaking prompt text into headers.

SSE responses use `text/event-stream`, `cache-control: no-cache`, `x-accel-buffering: no`, periodic `: vouch` keep-alive comments (when enabled), and a terminal `data: [DONE]`.

## API reference

### OpenAI chat completions

`POST /v1/chat/completions` accepts `model`, `messages`, `stream`, `temperature`, `max_tokens` / `max_completion_tokens`, and passes additional JSON fields through to the backend. At least one non-system message is required.

Non-streaming:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"coding/default","messages":[{"role":"user","content":"hello"}]}'
```

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1795688955,
  "model": "coding/default",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 2, "completion_tokens": 6, "total_tokens": 8}
}
```

Streaming (`"stream": true`) emits `chat.completion.chunk` events and ends with `data: [DONE]`:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":"hello"}}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

`GET /v1/models` returns `{"object":"list","data":[{"id":"coding/default","object":"model","owned_by":"vouch","created":0}, ...]}` with one entry per configured route.

### OpenAI responses

`POST /v1/responses` accepts `model`, `input` (string or array of message/function-call items), `instructions`, `stream`, `temperature`, `max_output_tokens`, `tools`, and `tool_choice`. At least one non-system item is required.

Non-streaming returns a `response` object (`status: "completed"`, `output` array containing `output_text` and/or `function_call` items, and `usage` with `input_tokens` / `output_tokens` / `total_tokens`). Streaming (`"stream": true`) emits SSE lifecycle events (`response.created`, `response.output_item.added`, `response.content_part.added`, `response.output_text.delta`, `response.function_call_arguments.delta`, `response.output_text.done`, `response.completed`).

### Anthropic messages

`POST /v1/messages` requires `max_tokens > 0` and at least one message. `system` may be a string or an array of text blocks, and `tools`, `tool_choice`, `tool_use`, and `tool_result` blocks are translated cleanly across drivers.

```bash
curl -s http://127.0.0.1:8080/v1/messages \
  -H 'content-type: application/json' \
  -d '{"model":"coding/default","max_tokens":256,"messages":[{"role":"user","content":"hello"}]}'
```

Streaming emits `message_start`, `content_block_start`, `content_block_delta` (`text_delta` or `input_json_delta`), `content_block_stop`, `message_delta`, and `message_stop`. Response headers are the same as for OpenAI (`x-vouch-backend`, `x-vouch-fallback`, `x-vouch-request-id`).

### Structured tool calling

Clients can pass `tools` and `tool_choice` to `/v1/chat/completions`, `/v1/responses`, or `/v1/messages` regardless of driver:

- **HTTP proxy (`kind = "http"`):** forwards `tools` and `tool_choice` upstream and relays non-streaming `tool_calls` and streaming `delta.tool_calls` plus upstream `usage` untouched.
- **CLI & PTY (`kind = "cli"` / `"pty"`):** appends deterministic tool schemas and `<vouch_tool_call>{"name":"...","arguments":{...}}</vouch_tool_call>` instructions to the rendered prompt, formats prior `tool_calls` and `tool` / `tool_result` turns in history, and extracts emitted `<vouch_tool_call>` blocks back into native OpenAI `tool_calls` (`finish_reason: "tool_calls"`), OpenAI Responses `function_call` items, or Anthropic `tool_use` blocks (`stop_reason: "tool_use"`). Requests without `tools` incur zero prompt overhead.

CLI/PTY `usage` counters are computed with `tiktoken-rs` (`cl100k_base` / `o200k_base`).

### MCP server ingress

`vouch` exposes configured routes as MCP (Model Context Protocol, JSON-RPC 2.0) tools over `stdio` or HTTP:

- `vouch mcp --config vouch.toml` runs over `stdin`/`stdout` (with all logs sent strictly to `stderr` so `stdout` stays pure JSON-RPC).
- `vouch mcp --config vouch.toml --transport http --bind 127.0.0.1:8081` or `POST /mcp` on the main `vouch serve` daemon exposes the same JSON-RPC handler over HTTP with bearer token enforcement when `[[tokens]]` is configured.
- Supported methods: `initialize`, `notifications/initialized`, `ping`, `tools/list` (`vouch_ask` plus one `route_<name>` tool per route), and `tools/call` routed through the standard fallback, circuit-breaker, affinity, worktree, and telemetry pipeline.

### Error format

Errors are mapped to each protocol's shape. OpenAI:

```json
{"error":{"message":"...","type":"invalid_request_error","param":null,"code":"model_not_found"}}
```

Anthropic returns `{"type":"error","error":{"type":"...","message":"..."}}`, with `type` one of `invalid_request_error`, `authentication_error`, `permission_error`, `not_found_error`, `rate_limit_error`, `overloaded_error`, or `api_error`.

| Condition | HTTP | OpenAI `type` / `code` |
| --- | --- | --- |
| Malformed request | 400 | `invalid_request_error` / `invalid_request` |
| Missing or invalid credential | 401 | `authentication_error` / `invalid_api_key` |
| Model outside token scope | 403 | `permission_error` / `model_not_allowed` |
| Unknown model | 404 | `not_found_error` / `model_not_found` |
| Token RPM exhausted | 429 | `rate_limit_error` / `rate_limit_exceeded` |
| Backend exited nonzero | 502 | `api_error` / `upstream_error` |
| All candidates saturated or open | 503 | `service_unavailable_error` / `no_healthy_backend` |
| Backend deadline exceeded | 504 | `api_error` / `upstream_timeout` |
| Client disconnected | 408 | `api_error` / `client_disconnected` |
| Configuration / internal fault | 500 | `server_error` / `internal_error` |

`429` and `503` responses include a `retry-after` header.

## Orchestrator integration

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="replace-with-a-configured-token",
    default_headers={
        "X-Session-ID": "orchestrator-run-42",
        "X-Workspace-Mode": "worktree",
    },
)

stream = client.chat.completions.create(
    model="coding/default",
    messages=[{"role": "user", "content": "Fix the failing test"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

### Anthropic Python SDK

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://127.0.0.1:8080",   # SDK appends /v1/messages
    api_key="replace-with-a-configured-token",
    default_headers={"X-Session-ID": "orchestrator-run-42"},
)

message = client.messages.create(
    model="coding/default",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Summarize this diff"}],
)
print(message.content[0].text)
```

### JavaScript / TypeScript

```js
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:8080/v1",
  apiKey: process.env.VOUCH_TOKEN,
  defaultHeaders: { "X-Session-ID": "my-agent-session" },
});

const result = await client.chat.completions.create({
  model: "coding/default",
  messages: [{ role: "user", content: "hello" }],
});
console.log(result.choices[0].message.content);
```

### LangGraph / CrewAI-style tools

Use the OpenAI adapter with a custom `base_url`; set `X-Session-ID` to the graph/thread id so multi-turn runs reuse the same backend session. Set `X-Workspace-Mode: worktree` when the agent writes code and you want each run isolated in its own git worktree.

### Unix socket via curl

```bash
curl -N --unix-socket /run/user/1000/vouch.sock \
  -X POST http://localhost/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"coding/default","stream":true,"messages":[{"role":"user","content":"hello"}]}'
```

### Docker containers and host bridges

To let a Docker Compose orchestrator (such as ARTEX, Open WebUI, or LiteLLM) reach host-bound `vouch` without binding `0.0.0.0`:

1. Find the Linux Docker bridge IPv4 address (`ip -4 addr show docker0`, typically `172.17.0.1`), configure `server.tcp_listen = "172.17.0.1:8080"`, and always define at least one `[[tokens]]` entry so the bridge listener is authenticated.
2. Add `extra_hosts` to the container in `docker-compose.yml` so `host.docker.internal` resolves reliably on Linux:

   ```yaml
   services:
     artex:
       extra_hosts:
         - "host.docker.internal:host-gateway"
   ```

3. Point the orchestrator's OpenAI base URL at `http://host.docker.internal:8080/v1` (or `http://172.17.0.1:8080/v1`).
4. If the orchestrator blocks private/RFC1918/Docker-bridge IP addresses for SSRF protection, explicitly allowlist the `vouch` host endpoint (for example in ARTEX: `ARTEX_LLM_PRIVATE_ENDPOINTS=http://host.docker.internal:8080/v1,http://172.17.0.1:8080/v1`).
5. Ensure the host firewall (if active, e.g., `ufw`/`firewalld`) permits traffic on `docker0` to port `8080`.

## Egress drivers and process supervision

**HTTP proxy (`kind = "http"`).** Connection-pooled `reqwest` client with per-backend timeouts, optional bearer key from `api_key_env`, extra headers, and upstream SSE delta extraction. Point it at `hermes proxy` (`http://127.0.0.1:8645`) or `opencode serve` (`http://127.0.0.1:4096`). `doctor` probes `health_path` or the chat path.

**Headless CLI (`kind = "cli"`).** Spawns the command with piped stdin/stdout/stderr, `.process_group(0)` on Unix, optional `workdir`, environment injection, and template substitution. ANSI escape sequences are stripped by a `vte::Parser` state machine so color codes never reach the API. Output is interpreted as `text`, `json`, or `json_lines`.

**PTY (`kind = "pty"`).** Uses `portable-pty` inside `tokio::task::spawn_blocking`, bridged over bounded channels, for CLIs that require `isatty()` or render a TUI. The same `ProcessGroupGuard` supervision applies.

**Worktree isolation.** With `workspace.enabled = true` and `isolate_workspace = true` on a backend (or `X-Workspace-Mode: worktree` on a request), each session gets a temporary git worktree. It is reused for the session's idle TTL and removed on expiry or graceful shutdown.

**Concurrency and cancellation.** Every driver holds an `Arc<Semaphore>` sized by `max_concurrency`. An owned permit guard is moved into the streaming task, so the slot stays held until the final chunk or client disconnect. All bridges use bounded `mpsc` channels; dropping the consumer cancels the request token and terminates the child process group.

## Resilience and session affinity

- **Ordered fallback:** sticky backend first, then the configured chain. Circuit-open and saturated backends are skipped; failures before streaming begins fail over to the next candidate. `x-vouch-fallback: true` and `vouch_fallbacks_total` record the event.
- **Circuit breaker:** consecutive upstream failures (or an upstream `429` when `open_on_rate_limit = true`) open the circuit for `reset_timeout_secs`. A single `HalfOpen` probe is admitted afterward and closes the circuit on success. States are `Closed`, `HalfOpen`, `Open` and exported as `0`, `1`, `2`.
- **Sticky affinity:** `X-Session-ID`, or a deterministic hash of system + first user turn, keys a `DashMap` with background TTL eviction. This keeps multi-turn orchestrators on the same backend/session.
- **Timeouts:** `request_timeout_secs` wraps routed non-streaming execution; `stream_timeout_secs` caps a single stream; `timeout_secs` is per backend execution; `queue_timeout_ms` bounds the permit wait.
- **Backpressure:** bounded channels, capped stdout/stderr/stream buffers, and `server.max_body_bytes` keep memory bounded under load.

## Security

- **Default listener is loopback only.** Keep it that way unless the host is isolated. The optional Unix socket is created with mode `0600`.
- **Local mode is intentionally permissive.** With no `[[tokens]]`, any (or no) credential is accepted. Never bind a non-loopback address in that mode.
- **Token enforcement** is constant-time over SHA-256 hashes, with sliding-window RPM limits and glob-scoped model access. Prefer long random tokens stored in your orchestrator's secret store, not in source.
- **Do not commit secrets.** Use `api_key_env` for HTTP upstreams and `env` for subprocess backends. Configured secrets are harvested into the log redactor; `Bearer ...`, `sk-...`, common secret JSON fields, `x-api-key`, and configured literals are scrubbed from structured logs and client-visible errors.
- **Ops endpoints are unauthenticated by design.** `/healthz`, `/readyz`, and `/metrics` expose backend ids, health state, and counters — never credentials or prompts. Keep them behind network controls if that metadata is sensitive.
- **CORS is off by default.** Only set `cors_permissive = true` for a trusted browser client.
- **Process isolation.** Children run in their own process group and are killed as a group on timeout, error, or disconnect. Worktree isolation optionally hides your main checkout from code-writing agents.
- **Reporting.** Private vulnerability reports and response expectations are in [SECURITY.md](SECURITY.md). Do not open a public issue for a security flaw.

## Observability

`GET /healthz` — liveness:

```json
{"status":"ok","version":"1.0.0","uptime_secs":42,"models":3,"backends":3,
 "tokens_enforced":true,"workspace_enabled":false}
```

`GET /readyz` — `200` when at least one backend circuit is not open, otherwise `503`. Body includes `ready` and per-backend `id`, `kind`, `max_concurrency`, `circuit`, `consecutive_failures`, `open_remaining_secs`.

`GET /metrics` — Prometheus text exposition (`text/plain; version=0.0.4`):

| Metric | Type | Labels |
| --- | --- | --- |
| `vouch_requests_total` | counter | `protocol`, `endpoint`, `status` |
| `vouch_request_duration_seconds` | histogram | `protocol`, `endpoint` |
| `vouch_queue_wait_seconds` | histogram | `backend` |
| `vouch_active_permits` | gauge | `backend` |
| `vouch_circuit_breaker_state` | gauge | `backend` — `0` closed, `1` half-open, `2` open |
| `vouch_circuit_breaker_transitions_total` | counter | `backend`, `state` |
| `vouch_fallbacks_total` | counter | `from`, `to` |
| `vouch_child_exit_total` | counter | `backend`, `code` (`signal` when signalled) |
| `vouch_token_rate_limited_total` | counter | `token` |
| `vouch_backend_health` | gauge | `backend` — `1` healthy, `0` unhealthy |

`protocol` is `openai`, `anthropic`, `mcp`, or `ops`. Histogram buckets run from 1 ms to 300 s.

`GET /ui` — returns `404` by default. Setting `[server].enable_ui = true` enables a read-only HTML dashboard that polls `/healthz`, `/readyz`, and `/metrics` from the browser with `cache-control: no-store`.

Logs are structured JSON with automatic secret redaction. `RUST_LOG` overrides `server.log_level` (for example `RUST_LOG=vouch=debug`). When running `vouch mcp --transport stdio`, logs go strictly to `stderr` so `stdout` stays pure JSON-RPC.

## CLI reference

| Command | Flags | Behavior |
| --- | --- | --- |
| `vouch serve` | `-c/--config`, `--tcp-listen`, `--unix-socket` | Run the daemon in the foreground. Handles `SIGHUP` reload; drains on `SIGINT`/`SIGTERM`. |
| `vouch routes` | `-c/--config`, `--json` | Validate config and print the model → backend fallback table; `--json` prints the raw route map. |
| `vouch doctor` | `-c/--config`, `--json` | Check configured binaries, auth-file metadata, daemon reachability, and run one live request per backend (`probe_timeout_secs` for CLI/PTY, `5s` for HTTP). Exits `2` when a check fails. |
| `vouch mcp` | `-c/--config`, `--transport`, `--bind` | Expose routes as MCP JSON-RPC tools over `stdio` or HTTP (`POST /mcp`). |
| `vouch top` | `--url`, `--unix-socket`, `--interval-ms` | Live terminal UI polling `/healthz`, `/readyz`, and `/metrics`. |
| `vouch service install` | `-c/--config`, `--binary`, `--no-start` | Write and enable a user systemd unit (Linux) or launchd agent (macOS). |
| `vouch service uninstall` | `--no-stop` | Stop, disable, and remove the user service. |

`doctor` actually executes CLI/PTY backends with a tiny `Reply with OK.` prompt (using `[discovery].probe_timeout_secs`, default `30` seconds) and sends an HTTP GET to HTTP backends; budget for that on a production config. It never prints credential contents.

Exit codes: `0` success, `1` operational error, `2` doctor found at least one problem. Fatal startup errors print `vouch: ...` to stderr.

## Running as a service

```bash
# Install (user-level; no sudo)
vouch service install --config "$HOME/.config/vouch/vouch.toml"

# Verify
systemctl --user status vouch.service
journalctl --user -u vouch.service -f

# Edit config, then reload without dropping requests
$EDITOR "$HOME/.config/vouch/vouch.toml"
systemctl --user reload vouch.service   # SIGHUP triggers atomic reload

# Remove
vouch service uninstall
```

On macOS the agent is `com.ayoubzulfiqar.vouch`, installed at `~/Library/LaunchAgents/com.ayoubzulfiqar.vouch.plist` and managed with `launchctl load`/`unload`. The generated systemd unit uses `Restart=on-failure`, `KillMode=control-group`, `ExecReload=/bin/kill -HUP $MAINPID`, and a `0600` file mode.

`systemctl --user reload vouch.service` sends `SIGHUP` to the daemon, which swaps the configuration atomically without dropping in-flight requests.

## Development

```bash
# Quality gate — all three must pass
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test

# Useful during iteration
cargo check --all-targets --message-format short
cargo test --test integration_test -- --nocapture
```

The integration suite (`tests/integration_test.rs`) is self-contained: it creates mock shell executables and an in-process upstream HTTP server, drives the axum router in-process, and asserts OpenAI Chat Completions & Responses, structured tool calling (`cli` and `http`), MCP JSON-RPC ingress, SSE framing, ANSI stripping, Anthropic events, fallback failover, timeout-triggered `SIGKILL` of an entire process group, `doctor` timeouts, token/RPM enforcement, `/ui` gating, and sticky sessions. No live LLM account or external port is required.

Module map:

```
src/main.rs        CLI, tracing, listeners, signals, graceful shutdown
src/config.rs      typed TOML schema, validation, ArcSwap snapshots
src/error.rs       GatewayError -> OpenAI/Anthropic wire errors
src/usage.rs       tiktoken-rs BPE token counting with char/4 fallback
src/state.rs       shared AppState
src/auth/          discovery + virtual tokens
src/router/        routing, affinity, circuit breaker
src/driver/        trait, HTTP proxy, CLI pipe, PTY, tool bridge, worktree
src/ingress/       OpenAI (/v1/chat/completions, /v1/responses), Anthropic, MCP, ops, /ui
src/cli/           routes, doctor, service, mcp, top
src/telemetry.rs   JSON tracing + redaction
src/metrics.rs     Prometheus instruments
```

See [AGENTS.md](AGENTS.md) for the conventions future agents must follow, and [CONTRIBUTING.md](CONTRIBUTING.md) for the review workflow.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `unknown field ...` at startup | Config key typo | Check the field against [vouch.example.toml](vouch.example.toml); all tables reject unknown keys |
| `401 invalid_api_key` | Tokens configured but credential missing/wrong | Send `Authorization: Bearer *** or `x-api-key` |
| `403 model_not_allowed` | Token's `allowed_models` excludes the model | Widen the glob or use another token |
| `429` immediately | RPM window full, possibly consumed by a `403` | Raise `rpm` or wait for the 60 s window |
| `503 no_healthy_backend` | All circuits open or every backend saturated | Check `doctor`, raise `failure_threshold`/`reset_timeout_secs`, or add a backend |
| `502 upstream error` | CLI exited nonzero or upstream returned an error | Read the redacted stderr/log; run `doctor` |
| `504` on every request | `timeout_secs` too low for the model | Raise the backend or request timeout |
| ANSI codes in output | Binary emits raw bytes | Keep `output_mode = "text"`; the `vte` stripper handles standard sequences |
| Child processes linger after a request | Non-Unix platform or process escaped its group | Verify Unix PGID support and inspect `vouch_child_exit_total` |
| `/readyz` returns `503` with healthy backends | Circuit breaker still in cooldown | Wait `reset_timeout_secs` or inspect `vouch_circuit_breaker_state` |
| Config edits not taking effect | Invalid new config (previous snapshot retained) | Check logs; validate with `vouch routes -c ...` |
| Unix socket bind fails | Stale socket owned by another process | Confirm no other daemon is running; remove the socket path |

## Community, contributing, and security reporting

- [CONTRIBUTING.md](CONTRIBUTING.md) — development setup, quality gates, code conventions, and pull request workflow. It also explains **writing contributing guidelines** and what makes them effective.
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) — explains **what a code of conduct is**, the standards we hold, unacceptable behavior, scope, and how to report a violation.
- [SECURITY.md](SECURITY.md) — explains **setting up a security policy**, supported versions, scope, private reporting, response timelines, and disclosure.
- [Issue templates](.github/ISSUE_TEMPLATE) and [pull request template](.github/PULL_REQUEST_TEMPLATE.md) — structured reports and review checklists.

Please use the issue templates; include a redacted config snippet and logs for bug reports. Never paste tokens, API keys, or auth-file contents into a public issue.

## License

**GNU General Public License v3.0 — SPDX identifier `GPL-3.0-or-later`.**

Copyright (C) 2026 Ayoub Zulfiqar and vouch contributors.

The full license text is in [LICENSE](LICENSE). This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the LICENSE file for details.

For attribution, add a short header to new source files:

```rust
//! SPDX-License-Identifier: GPL-3.0-or-later
```

### Choosing a license

A license is a product decision, not paperwork. The criteria that led to GPL-3.0 here:

| Concern | Why it matters for `vouch` | GPL-3.0 answer |
| --- | --- | --- |
| Local trust boundary | The daemon reads credentials and proxies model output; users should be able to inspect exactly what they run | Source must be available to every user |
| Copyleft | Improvements to the gateway itself should remain free | Derived versions must ship under the same license |
| Network use | Agent gateways are commonly deployed as a service without distributing binaries | GPLv3 closes the "ASP loophole" that GPLv2 left open |
| Patent risk | Authentication/tooling stacks are patent-active | GPLv3 includes an express patent grant and anti-patent-termination clause |
| Compatibility | Many Rust crates are MIT/Apache-2.0 | GPLv3 is compatible with both; those libraries can be linked |
| Downstream freedom | Some users want to embed the gateway in proprietary products | That is the trade-off; if it becomes the priority, dual-licensing or a permissive relicense is a maintainer decision requiring every contributor's consent |

Practical consequences for contributors:

- Contributions are licensed under the same terms; you retain copyright on your work.
- Keep an SPDX identifier in new files and do not copy code with an incompatible license.
- Do not paste third-party code, prompts, or configuration from repositories with unknown licensing.
- Companies that cannot accept GPLv3 should talk to the maintainers before contributing code; they can still file issues, report security problems, and use the tool.

If you are choosing a license for a fork, evaluate permissive (MIT/Apache-2.0), weak copyleft (MPL-2.0, LGPL), and strong copyleft (GPL) against your distribution model and patent posture rather than copying this choice blindly.
