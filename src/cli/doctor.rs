//! `vouch doctor` — read-only local discovery plus live backend smoke probes.
//!
//! The command deliberately never writes to the machine. It resolves every
//! configured executable on `PATH`, inspects the credential files left behind
//! by common agent CLIs, TCP-probes local daemons, and then sends one tiny
//! request through each configured driver. A human-readable report is printed
//! by default; `--json` emits the same report for automation.

use std::io::Write;

use serde::Serialize;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::DoctorArgs;
use crate::{
    auth::discovery::{AuthFileStatus, BinaryStatus, DaemonStatus, discover, format_age},
    config::{BackendKind, GatewayConfig},
    driver::{ChatMessage, DriverResponse, ExecutionRequest, WorkspaceMode, collect_stream},
    error::{GatewayError, Result},
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
    let config = GatewayConfig::load(&args.config)?;
    let report = inspect(&args.config, &config).await?;

    if args.json {
        let mut stdout = std::io::stdout().lock();
        let payload = serde_json::to_string_pretty(&report)?;
        writeln!(stdout, "{payload}")?;
    } else {
        print_report(&report);
    }
    Ok(report.healthy)
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
