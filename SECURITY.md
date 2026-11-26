# Security Policy

## Set up a security policy

A security policy is a published promise about how a project handles
vulnerabilities. To be useful it must state: which versions still receive
fixes, what is and is not in scope, how to report a problem privately, how
quickly maintainers will respond, how and when disclosure happens, and what
operators can do to reduce risk today. Without those pieces, reporters either
disclose publicly (exposing users) or give up, and maintainers have no agreed
process to follow under pressure.

Maintainers operationalize this policy by:

- designating an owner for incoming reports and a backup contact;
- enabling GitHub private vulnerability reporting and monitoring the security
  advisory inbox;
- publishing this file and linking it from the README and issue forms;
- triaging reports on the timeline below, keeping a private record of each
  report and its disposition;
- preparing fixes in a private branch, coordinating a CVE/advisory when
  appropriate, and crediting reporters only with their consent;
- rotating contacts and re-reading this policy after each release.

## Supported Versions

| Version | Supported |
| --- | --- |
| 0.1.x (current) | ✅ |
| < 0.1.0 | ❌ |

Fixes are made against the current release line. For older versions, upgrade to
the current release to receive a fix.

## Scope

`vouch` is a local authentication multiplexer and process-supervised gateway.
It handles credentials and spawns subprocesses, so the following are in scope.

**In scope:**

- Token enforcement: SHA-256 constant-time comparison, RPM windows, model
  globs, and the behavior when no `[[tokens]]` are configured.
- Secret handling: `api_key_env`, `env`, `extra_headers`, redaction in logs and
  client-visible errors, and the 0600 Unix socket mode.
- Subprocess sandbox: PGID isolation, SIGTERM → 500 ms → SIGKILL escalation,
  bounded stdout/stderr/streams, and git-worktree isolation.
- Protocol adapters and ingress: request body limits, SSE decoding, header
  parsing (`X-Session-ID`, `X-Workspace-Mode`, `x-request-id`), and the
  unauthenticated `/healthz`, `/readyz`, and `/metrics` endpoints — except for
  data they intentionally expose (ids and circuit state).
- Routing/resilience: circuit breakers, failover, and affinity issues that
  could cause an authorization bypass or unbounded resource consumption.
- `vouch doctor` read-only guarantees: auth files must be reported by metadata
  only, never by contents.

**Out of scope:**

- Vulnerabilities in upstream backends, models, CLIs, or SDKs that `vouch`
  talks to; report those to their maintainers.
- Security of the host, account, or network that runs `vouch`, or of clients
  that leak their own credentials.
- Issues that require an attacker to already have local code execution as the
  `vouch` user (or root), or write access to `vouch.toml`. The config file is
  trusted input; protect it with filesystem permissions.
- Unreproducible automated scanner output with no demonstrated impact or
  reproduction steps.

## Reporting a Vulnerability

**Do not open a public issue for a security vulnerability.** Public issue
templates are for bugs and features only.

Report privately using either channel:

- GitHub private vulnerability reporting: `<repository-url>/security/advisories/new`
  (**placeholder** — replace with the repository's actual security-advisory URL
  once a remote is configured).
- Email: **[SECURITY EMAIL — placeholder: replace with a maintainer-monitored
  address before publishing. Do not substitute a real personal address here
  without its owner's consent.]**

Include:

- affected version/commit and configuration (with all secrets redacted);
- impact and an attacker model (what the attacker can and cannot do);
- step-by-step reproduction, PoC, or minimal config;
- any suggested fix.

**Never paste real tokens, API keys, prompts, or auth-file contents into a
report.** Replace them with obvious placeholders such as `REDACTED`. Reports
containing live credentials will be treated as compromised credentials and
should be rotated immediately.

## Response Timeline

- **Acknowledgment:** within 48 hours of receipt.
- **Triage and initial severity assessment:** within 5 days, including a request
  for missing information if needed.
- **Fix target:** 30 days for high/critical issues, and a best-effort target for
  lower severity issues agreed with the reporter. If a fix needs longer (for
  example, coordinated disclosure with an upstream), maintainers will explain
  why and give a revised date.

If you do not receive an acknowledgment within 48 hours, follow up once via the
alternate channel above before disclosing publicly.

## Disclosure Process

1. Maintainers acknowledge the report, confirm scope, and open a private
   tracking record.
2. The reporter and maintainers agree on severity, affected versions, and a
   target date.
3. The fix is developed and tested in a private branch; the public repository
   sees no details until the release is ready.
4. A GitHub security advisory and, where warranted, a CVE request are prepared.
   Reporter credit is included only with the reporter's consent.
5. A patched release is published. The advisory describes the impact,
   affected versions, and remediation, with a link to the fix.
6. Only after the release is available does the issue become public. If a fix
   cannot be produced, maintainers will say so plainly and discuss a disclosure
   date with the reporter.

## Hardening Guidance for Operators

- Keep the default loopback listener (`127.0.0.1:8080`). Never bind a
  non-loopback address in local mode (no `[[tokens]]`), where any credential is
  accepted.
- Prefer the optional Unix socket; it is created with mode `0600`.
- Configure `[[tokens]]` with strong values (≥ 8 chars), set `rpm`, and scope
  `allowed_models` with globs. Rotate tokens that may have been exposed.
- Keep `cors_permissive = false` unless you have a specific browser use case.
- Store `vouch.toml` and any auth files with restrictive permissions (for
  example, `chmod 600 vouch.toml`). Prefer `api_key_env` and child `env` over
  inline secrets in TOML.
- Run `vouch` as an unprivileged user. Subprocess backends inherit that
  account's privileges and run in their own process group.
- Leave `server.max_body_bytes` at a sensible ceiling and keep per-backend
  `max_concurrency`/`queue_timeout_ms` bounded; these are the resource-exhaustion
  controls.
- Understand workspace isolation: when `[workspace] enabled = true` (or
  `X-Workspace-Mode: worktree`), subprocesses see a git worktree. Disable it if
  they must not see a repository checkout.
- `/healthz`, `/readyz`, and `/metrics` are unauthenticated by design. Keep
  them behind network controls if backend ids and circuit state are sensitive.
- Treat `vouch doctor` as a live probe: it executes a minimal request through
  every configured driver. Do not run it casually against production
  credentials or endpoints.
