//! `vouch doctor` — read-only local discovery plus live backend smoke probes.
//!
//! The command deliberately never writes to the machine. It resolves every
//! configured executable on `PATH`, inspects the credential files left behind
//! by common agent CLIs, TCP-probes local daemons, and then sends one tiny
//! request through each configured driver. A human-readable report is printed
//! by default; `--json` emits the same report for automation.

use std::io::Write;
use std::path::Path;

use serde::Serialize;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::{
    DoctorArgs,
    init::{extra_user_bin_dirs, find_cli_binary, tcp_probe},
};
use crate::{
    auth::discovery::{AuthFileStatus, BinaryStatus, DaemonStatus, discover, format_age},
    config::{BackendKind, GatewayConfig},
    driver::{ChatMessage, DriverResponse, ExecutionRequest, WorkspaceMode, collect_stream},
    error::{GatewayError, Result},
    persistence::write_atomic_0600,
    router::ReadinessReport,
    state::AppState,
};

/// Result of sending one minimal request through a backend.
#[derive(Debug, Clone, Serialize)]
pub struct ProbeResult {
    /// Backend id.
    pub backend: String,
    /// Transport kind.
    pub kind: &'static str,
    /// Whether the probe succeeded.
    pub ok: bool,
    /// Detail on failure, or a short success note.
    pub detail: String,
}

/// Complete doctor report.
#[derive(Debug, Clone, Serialize)]
pub struct DoctorReport {
    /// Absolute path of the configuration file that was inspected.
    pub config: String,
    /// Configured subprocess binaries.
    pub binaries: Vec<BinaryStatus>,
    /// Known credential files and their freshness.
    pub auth_files: Vec<AuthFileStatus>,
    /// Local daemon reachability.
    pub daemons: Vec<DaemonStatus>,
    /// Live smoke-probe results.
    pub probes: Vec<ProbeResult>,
    /// Circuit-breaker readiness at the time of the check.
    readiness: ReadinessReport,
    /// Human-readable problems that should be fixed before serving traffic.
    pub problems: Vec<String>,
    /// True when nothing actionable was found.
    pub healthy: bool,
}

/// Run the doctor command. Returns `true` when the report is healthy.
pub async fn run(args: &DoctorArgs) -> Result<bool> {
    let mut config = GatewayConfig::load(&args.config)?;
    let mut report = inspect(&args.config, &config).await?;

    if args.fix && !report.healthy {
        let fixes = plan_fixes(&config, &report).await;
        if !fixes.is_empty() {
            let applied = apply_planned_fixes(&args.config, &mut config, &fixes, args.yes)?;
            if applied {
                reset_live_circuits(&config, &fixes).await;
                report = inspect(&args.config, &config).await?;
            }
        }
    }

    if args.json {
        let mut stdout = std::io::stdout().lock();
        let payload = serde_json::to_string_pretty(&report)?;
        writeln!(stdout, "{payload}")?;
    } else {
        print_report(&report);
    }
    Ok(report.healthy)
}

/// One proposed configuration repair synthesized by `vouch doctor --fix`.
#[derive(Debug, Clone, Serialize)]
pub struct ProposedFix {
    /// Target backend id.
    pub backend_id: String,
    /// Config field updated (`"command"` or `"base_url"`).
    pub field: &'static str,
    /// Value prior to repair.
    pub old_value: String,
    /// Replacement discovered on this host.
    pub new_value: String,
}

const KNOWN_DAEMON_CANDIDATES: &[(&str, &str, u16)] = &[
    ("hermes", "127.0.0.1", 8645),
    ("opencode", "127.0.0.1", 4096),
    ("ollama", "127.0.0.1", 11434),
];

/// Compute candidate CLI path and local daemon port repairs without modifying disk.
pub async fn plan_fixes(config: &GatewayConfig, report: &DoctorReport) -> Vec<ProposedFix> {
    let mut fixes = Vec::new();
    let extra_dirs = extra_user_bin_dirs();

    for backend in &config.backends {
        match backend.kind {
            BackendKind::Cli | BackendKind::Pty => {
                let Some(cmd) = backend.command.as_deref() else {
                    continue;
                };
                let missing = report
                    .binaries
                    .iter()
                    .find(|b| b.name == cmd)
                    .is_some_and(|b| !b.found);
                if !missing {
                    continue;
                }
                let base_name = Path::new(cmd)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(cmd);
                if let Some(found) = find_cli_binary(base_name, &extra_dirs) {
                    let resolved = found.display().to_string();
                    if resolved != cmd {
                        fixes.push(ProposedFix {
                            backend_id: backend.id.clone(),
                            field: "command",
                            old_value: cmd.to_string(),
                            new_value: resolved,
                        });
                    }
                }
            }
            BackendKind::Http => {
                let Some(cur_url) = backend.base_url.as_deref() else {
                    continue;
                };
                let unreachable = report
                    .daemons
                    .iter()
                    .find(|d| d.backend == backend.id)
                    .is_some_and(|d| !d.reachable);
                if !unreachable {
                    continue;
                }
                let lower_id = backend.id.to_ascii_lowercase();
                for &(hint, host, port) in KNOWN_DAEMON_CANDIDATES {
                    if lower_id.contains(hint) && tcp_probe(host, port).await {
                        let new_url = format!("http://{host}:{port}");
                        if new_url != cur_url.trim_end_matches('/') {
                            fixes.push(ProposedFix {
                                backend_id: backend.id.clone(),
                                field: "base_url",
                                old_value: cur_url.to_string(),
                                new_value: new_url,
                            });
                            break;
                        }
                    }
                }
            }
        }
    }
    fixes
}

/// Print proposed fixes, optionally prompt for confirmation, validate mutated TOML,
/// and atomically update `path` with mode `0600`.
pub fn apply_planned_fixes(
    path: &Path,
    config: &mut GatewayConfig,
    fixes: &[ProposedFix],
    assume_yes: bool,
) -> Result<bool> {
    if fixes.is_empty() {
        return Ok(false);
    }
    {
        let mut stdout = std::io::stdout().lock();
        let _ = writeln!(stdout, "\nProposed config fixes ({}):", path.display());
        for fix in fixes {
            let _ = writeln!(
                stdout,
                "  backend {:?}.{}: {:?} -> {:?}",
                fix.backend_id, fix.field, fix.old_value, fix.new_value
            );
        }
        if !assume_yes {
            let _ = write!(stdout, "Apply {} fix(es)? [y/N]: ", fixes.len());
            let _ = stdout.flush();
        }
    }
    if !assume_yes {
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer)?;
        let normalized = answer.trim().to_ascii_lowercase();
        if normalized != "y" && normalized != "yes" {
            return Ok(false);
        }
    }

    let mut updated = config.clone();
    for fix in fixes {
        if let Some(backend) = updated.backends.iter_mut().find(|b| b.id == fix.backend_id) {
            match fix.field {
                "command" => backend.command = Some(fix.new_value.clone()),
                "base_url" => backend.base_url = Some(fix.new_value.clone()),
                _ => {}
            }
        }
    }
    let toml_text = toml::to_string_pretty(&updated)
        .map_err(|error| GatewayError::Internal(format!("serialize fixed config: {error}")))?;
    GatewayConfig::from_toml_str(&toml_text)?;
    write_atomic_0600(path, toml_text.as_bytes())?;
    *config = updated;
    Ok(true)
}

async fn reset_live_circuits(config: &GatewayConfig, fixes: &[ProposedFix]) {
    if !config.server.enable_admin {
        return;
    }
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
    {
        Ok(c) => c,
        Err(_) => return,
    };
    let admin_token = config
        .tokens
        .iter()
        .find(|t| t.admin)
        .map(|t| t.token.as_str());
    for fix in fixes {
        let url = format!(
            "http://{}/admin/circuits/{}/reset",
            config.server.tcp_listen, fix.backend_id
        );
        let mut req = client.post(url);
        if let Some(tok) = admin_token {
            req = req.bearer_auth(tok);
        }
        let _ = req.send().await;
    }
}

/// Build a report from an already validated configuration.
pub async fn inspect(path: &std::path::Path, config: &GatewayConfig) -> Result<DoctorReport> {
    let mut problems = Vec::new();
    let mut discovery = discover(config).await?;

    for binary in &discovery.binaries {
        if !binary.found {
            problems.push(format!("binary {:?} not found on PATH", binary.name));
        }
    }
    for daemon in &discovery.daemons {
        if !daemon.reachable {
            problems.push(format!(
                "daemon {:?} unreachable at {}: {}",
                daemon.backend,
                daemon.url,
                daemon.detail.as_deref().unwrap_or("no detail")
            ));
        }
    }

    // Pin the canonical path so relative config paths render consistently.
    discovery.auth_files.sort_by(|a, b| a.path.cmp(&b.path));

    let state = AppState::new(path.to_path_buf(), config.clone())?;
    let cli_timeout = std::time::Duration::from_secs(config.discovery.probe_timeout_secs.max(1));
    let mut probes = Vec::new();
    for backend in state.router().backends() {
        let probe = smoke_probe(&backend.config.id, &backend, &state, cli_timeout).await;
        if !probe.ok {
            problems.push(format!("backend {:?}: {}", probe.backend, probe.detail));
        }
        probes.push(probe);
    }

    let readiness = state.router().readiness();
    if !readiness.ready {
        problems.push("every backend circuit is open".into());
    }
    state.shutdown().await;

    let healthy = problems.is_empty();
    Ok(DoctorReport {
        config: path.to_string_lossy().into_owned(),
        binaries: discovery.binaries,
        auth_files: discovery.auth_files,
        daemons: discovery.daemons,
        probes,
        readiness,
        problems,
        healthy,
    })
}

/// Send one minimal request through a backend and drain the response.
async fn smoke_probe(
    backend: &str,
    handle: &crate::router::BackendHandle,
    state: &AppState,
    cli_timeout: std::time::Duration,
) -> ProbeResult {
    let kind = handle.config.kind;
    let timeout = match kind {
        BackendKind::Http => std::time::Duration::from_secs(5),
        BackendKind::Cli | BackendKind::Pty => cli_timeout,
    };
    let result = tokio::time::timeout(timeout, probe_inner(backend, handle, state)).await;

    match result {
        Ok(Ok(note)) => ProbeResult {
            backend: backend.to_string(),
            kind: kind_label(kind),
            ok: true,
            detail: note,
        },
        Ok(Err(error)) => ProbeResult {
            backend: backend.to_string(),
            kind: kind_label(kind),
            ok: false,
            detail: crate::telemetry::redact_text(&error.to_string()),
        },
        Err(_) => ProbeResult {
            backend: backend.to_string(),
            kind: kind_label(kind),
            ok: false,
            detail: format!("smoke probe timed out after {} seconds", timeout.as_secs()),
        },
    }
}

async fn probe_inner(
    backend: &str,
    handle: &crate::router::BackendHandle,
    _state: &AppState,
) -> Result<String> {
    if handle.config.kind == BackendKind::Http {
        return match handle.driver.probe().await {
            Ok(()) => Ok("HTTP probe succeeded".into()),
            Err(detail) => Err(GatewayError::Upstream(detail)),
        };
    }

    let request = ExecutionRequest {
        request_id: format!("doctor-{}", Uuid::new_v4().simple()),
        session_key: format!("doctor-{backend}"),
        model: "doctor/probe".into(),
        system: None,
        messages: vec![ChatMessage::text("user", "Reply with OK.")],
        prompt: "Reply with OK.".into(),
        stream: true,
        temperature: None,
        max_tokens: Some(16),
        workspace_mode: WorkspaceMode::None,
        workdir: None,
        cancel: CancellationToken::new(),
        tools: Vec::new(),
        tool_choice: None,
        response_format: None,
        backend_session_id: None,
        delta_messages: None,
        session_capture: std::sync::Arc::new(parking_lot::Mutex::new(None)),
        trace_context: None,
        profile_id: None,
        profile_env: std::collections::BTreeMap::new(),
        allowed_backends: None,
        extra: serde_json::Map::new(),
    };

    let response = handle.driver.execute(request).await?;
    match response {
        DriverResponse::Complete(completion) => Ok(format!(
            "returned {} characters",
            completion.content.chars().count()
        )),
        DriverResponse::Stream(stream) => {
            let (content, _, _) = collect_stream(stream, 64 * 1024).await?;
            Ok(format!("streamed {} characters", content.chars().count()))
        }
    }
}

fn kind_label(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::Http => "http",
        BackendKind::Cli => "cli",
        BackendKind::Pty => "pty",
    }
}

/// Render the human-readable report.
fn print_report(report: &DoctorReport) {
    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(stdout, "vouch doctor — {}", report.config);
    let _ = writeln!(stdout, "\nBinaries:");
    for binary in &report.binaries {
        let status = if binary.found { "ok" } else { "MISSING" };
        let path = binary
            .path
            .as_ref()
            .map_or_else(|| "-".to_string(), |path| path.display().to_string());
        let _ = writeln!(stdout, "  {:<24} {:<8} {}", binary.name, status, path);
    }
    if report.binaries.is_empty() {
        let _ = writeln!(stdout, "  (no subprocess backends configured)");
    }

    let _ = writeln!(stdout, "\nCredential files:");
    for file in &report.auth_files {
        let state = if file.exists {
            format!("present, modified {}", format_age(file.age_secs))
        } else {
            "not present".to_string()
        };
        let _ = writeln!(stdout, "  {:<48} {}", file.path.display(), state);
    }
    if report.auth_files.is_empty() {
        let _ = writeln!(stdout, "  (discovery disabled or no known files)");
    }

    let _ = writeln!(stdout, "\nDaemons:");
    for daemon in &report.daemons {
        let status = if daemon.reachable {
            "reachable"
        } else {
            "unreachable"
        };
        let _ = writeln!(
            stdout,
            "  {:<20} {:<36} {}",
            daemon.backend, daemon.url, status
        );
    }
    if report.daemons.is_empty() {
        let _ = writeln!(stdout, "  (no HTTP backends configured)");
    }

    let _ = writeln!(stdout, "\nLive probes:");
    for probe in &report.probes {
        let status = if probe.ok { "ok" } else { "FAILED" };
        let _ = writeln!(
            stdout,
            "  {:<20} {:<6} {:<6} {}",
            probe.backend, probe.kind, status, probe.detail
        );
    }
    if report.probes.is_empty() {
        let _ = writeln!(stdout, "  (no backends configured)");
    }

    let _ = writeln!(
        stdout,
        "\nCircuit state: {}",
        if report.readiness.ready {
            "ready"
        } else {
            "NOT READY"
        }
    );
    if report.problems.is_empty() {
        let _ = writeln!(stdout, "Doctor: all checks passed.");
    } else {
        let _ = writeln!(
            stdout,
            "Doctor: {} problem(s) found:",
            report.problems.len()
        );
        for problem in &report.problems {
            let _ = writeln!(stdout, "  - {problem}");
        }
    }
}
