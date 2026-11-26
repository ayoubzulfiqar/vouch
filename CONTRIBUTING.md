# Contributing to vouch

Thanks for helping improve `vouch`. This document covers how to set up a
development environment, the checks every change must pass, the invariants that
must hold in this codebase, and how to propose a change.

## Writing contributing guidelines

Guidelines are useful only when a contributor can act on them without asking.
Three properties make that possible:

- **Concrete.** Name the exact command, file, or invariant. "Run the checks"
  is weak; "`cargo clippy --all-targets -- -D warnings`" is testable and can be
  copy-pasted. Where a rule has a rationale (for example, keeping `rustls` so
  the daemon never depends on system OpenSSL), say so in one line.
- **Testable.** Each rule should have an observable pass/fail signal: a command
  that exits zero, a test that runs, a config example that parses, a header that
  appears in a response. If a reviewer cannot tell whether the rule was
  followed, rewrite the rule.
- **Discoverable.** Keep the rules in one file, link to it from the README, and
  repeat them where the work happens (issue forms, PR template, AGENTS.md). A
  guideline nobody finds before opening a PR is not a guideline.

The sections below follow that pattern: every requirement is tied to a command,
a code invariant, or a named file.

## Development setup

You need a Rust 2024 toolchain. There is no database, service, or live LLM
account required for the test suite.

```bash
cd vouch
cargo build
cp vouch.example.toml vouch.toml   # only if you do not already have one
$EDITOR vouch.toml

# Inspect configuration before running the daemon:
cargo run -- routes -c vouch.toml
cargo run -- doctor -c vouch.toml  # runs live smoke probes against configured backends
cargo run -- serve  -c vouch.toml
```

`--config`/`-c` defaults to `vouch.toml`. `RUST_LOG` overrides
`server.log_level`. Note that `vouch doctor` executes a minimal request through
every configured driver; do not point it at a production config casually.

## The quality gate

Run all four commands from the repository root before opening a pull request.
They must pass with a clean working tree state; CI runs the same checks.

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build
```

Current expectations: 35 unit tests (colocated in `#[cfg(test)]` modules) and 8
integration tests using mock shell scripts in `tests/`. If you add behavior, add
a test that would fail without your change.

## Code invariants

### No `unwrap` or `expect` in runtime paths

Fallible work returns `Result<T>` with `GatewayError`. Convert with `ok_or_else`,
`map_err`, `?`, or `unwrap_or(_default)` for genuinely benign fallbacks.
`unwrap`/`expect` are acceptable only in `#[cfg(test)]` code and for static,
provably infallible initialization such as the `LazyLock` regexes in
`telemetry.rs`. A malformed request, bad upstream, or missing environment
variable must never panic the daemon. Client-visible error strings go through
`telemetry::redact_text`.

### Bound every channel and buffer

- Stream bridges use `mpsc::channel::<Result<StreamChunk>>(32)` — never an
  unbounded channel. A slow SSE consumer applies backpressure through the
  `PermitGuard` held for the stream's lifetime.
- Subprocess stdout is capped at 16 MiB, captured stderr at 8 KiB, collected
  streams at 16 MiB, and request bodies by `server.max_body_bytes`.
- `PermitGuard::acquire` waits at most `queue_timeout_ms`, honors client
  cancellation, releases on `Drop`, and updates the active-permit gauge.

If your change adds a channel, buffer, or collection, state its ceiling and
test it.

### PGID supervision for every child process

Subprocess backends spawn with `process_group(0)` so `PGID == PID`. Move a
`ProcessGroupGuard` into the task that owns the child: call `disarm()` on clean
exit and `terminate()` on timeout, I/O error, cancellation, or dropped stream.
Termination sends `SIGTERM` to the whole group, waits 500 ms, then escalates to
`SIGKILL` if members remain — grandchildren must never outlive a request.
PTY work is blocking (`portable-pty`), so it runs inside `spawn_blocking` and
communicates over bounded channels with the same guard.

### Hot reload via ArcSwap

`ConfigStore`, `Router::table`, `TokenRegistry::policies`, and
`RedactionRegistry::secrets` are `ArcSwap`-backed. A reload is atomic: validate
the file, rebuild token policies, rebuild the router table, update workspace
settings and redaction secrets, then `store_snapshot` last. On any failure,
keep the previous snapshot. Reuse circuit breakers by backend id so cooldown
state survives reload. Changing `server.tcp_listen` or `server.unix_socket`
requires a restart; log that explicitly. Readers keep their `Arc` snapshot for
the duration of a request.

### Keep `reqwest` on `rustls`

`Cargo.toml` pins `reqwest` with `default-features = false` and features
`["json", "stream", "rustls"]`. Do **not** switch to `rustls-tls` or enable
default TLS features. This is a local daemon and must not silently depend on
system OpenSSL.

### Secrets and redaction

`GatewayConfig::secrets()` collects token values, secret-looking `env` values,
secret-looking `extra_headers`, and `api_key_env`, then feeds them to
`telemetry::set_global_secrets` whenever config is installed or reloaded. Never
log a token, prompt body, or upstream key. `/healthz`, `/readyz`, and
`/metrics` expose only ids and circuit state. Never paste real credentials into
an issue, PR, or test fixture.

## Adding a backend driver

1. Add the transport module under `src/driver/` and register it in
   `src/driver/mod.rs`.
2. Implement the `Driver` trait: `id()`, `kind()`, `max_concurrency()`,
   `async fn execute(ExecutionRequest) -> Result<DriverResponse>`, and the
   optional `async fn probe()` used by `vouch doctor`.
3. Acquire a permit with `PermitGuard::acquire` and hold it for the whole
   response, including streaming. Emit a complete `Completion` or a bounded
   `ChunkStream`; respect `request.cancellation` and the configured timeouts.
4. For subprocess backends, use `process_group(0)` and a `ProcessGroupGuard` as
   described above. For HTTP backends, keep `reqwest` on `rustls`.
5. Add a `BackendKind` variant and any fields in `src/config.rs`, with
   `#[serde(deny_unknown_fields)]` retained and a default/default fn. Update
   `BackendConfig::validate` and `GatewayConfig::validate` as needed.
6. Update `vouch.example.toml` and `README.md` in the same PR, and add tests:
   parsing/validation, a success path, a failure/failover path, and a doctor
   probe if applicable.

## Adding a config field

Config is a strict contract. Every new field requires all of these together:

- the struct field plus its `Default`/default fn in `src/config.rs`;
- validation in `GatewayConfig::validate` / `BackendConfig::validate`;
- a documented entry in `vouch.example.toml`;
- a matching description in `README.md`;
- a unit test that parses the field and an invalid-value test where relevant.

Re-validate the example with `GatewayConfig::from_toml_str` after any change.
Keep `deny_unknown_fields` on every schema struct.

## Commit and PR workflow

1. Fork/branch from `main`; keep changes focused and commits small.
2. Run the full quality gate above, including `cargo test` and `cargo build`.
3. Open a PR using `.github/PULL_REQUEST_TEMPLATE.md`. Fill in the checklist:
   fmt/clippy/test, docs and example config, changelog note, security review,
   and linked issue.
4. Be responsive to review. Reviewers should cite the concrete rule or test
   that fails; authors should reply with the command output that fixes it.
5. This project uses a DCO-style sign-off. Sign off every commit:

   ```bash
   git commit -s -m "your commit message"
   ```

   This appends a line in this form, using your real name and email:

   ```
   Signed-off-by: Jane Doe <jane@example.com>
   ```

   By signing off you certify that you wrote the contribution or otherwise
   have the right to submit it under the project's license (GPL-3.0-or-later).
   You may also add `Signed-off-by:` lines for co-authors. Commits without a
   sign-off will be asked to amend before merge.

## Reporting bugs and vulnerabilities

Use the issue forms for bugs and feature requests. For security issues, follow
[SECURITY.md](SECURITY.md) and report privately — never open a public issue or
paste real credentials. Conduct issues are covered by
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
