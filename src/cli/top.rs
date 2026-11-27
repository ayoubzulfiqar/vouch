//! SPDX-License-Identifier: GPL-3.0-or-later
//! Live terminal dashboard (`vouch top`) polling `/healthz`, `/readyz`, and `/metrics`.

use std::{
    collections::BTreeMap,
    io::{self, Stdout},
    path::Path,
    time::{Duration, Instant},
};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table},
};
use serde::Deserialize;

use crate::{
    cli::TopArgs,
    error::{GatewayError, Result},
};

#[derive(Debug, Default, Deserialize)]
struct HealthzPayload {
    status: String,
    version: String,
    uptime_secs: u64,
    models: usize,
    backends: usize,
    tokens_enforced: bool,
    workspace_enabled: bool,
}

#[derive(Debug, Default, Deserialize)]
struct ReadyzBackend {
    id: String,
    kind: String,
    max_concurrency: usize,
    circuit: String,
    consecutive_failures: u32,
    open_remaining_secs: u64,
    #[serde(default)]
    auth_expired: bool,
}

#[derive(Debug, Default, Deserialize)]
struct ReadyzPayload {
    ready: bool,
    backends: Vec<ReadyzBackend>,
}

/// Parsed subset of Prometheus `/metrics` rendered by `vouch top`.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct MetricsSnapshot {
    /// Active permits by backend id.
    pub active_permits: BTreeMap<String, u64>,
    /// Backend health gauge (`1` = healthy, `0` = unhealthy).
    pub backend_health: BTreeMap<String, bool>,
    /// Exit code `0` counts per backend.
    pub child_exits_ok: BTreeMap<String, u64>,
    /// Non-zero or signalled child exit counts per backend.
    pub child_exits_err: BTreeMap<String, u64>,
    /// Requests grouped by `(protocol, status)`.
    pub requests_by_proto_status: BTreeMap<(String, String), u64>,
    /// Total fallback activations.
    pub fallbacks_total: u64,
    /// Total virtual-token rate limit rejections.
    pub rate_limited_total: u64,
}

struct DashboardData {
    target_label: String,
    health: Option<HealthzPayload>,
    ready: Option<ReadyzPayload>,
    metrics: MetricsSnapshot,
    last_error: Option<String>,
}

struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(stdout, LeaveAlternateScreen);
    }
}

/// Run the live TUI until `q`, `Esc`, or `Ctrl+C`.
pub async fn run(args: &TopArgs) -> Result<()> {
    let (client, base_url, target_label) = build_client(args)?;
    let poll_interval = Duration::from_millis(args.interval_ms.max(100));

    enable_raw_mode().map_err(GatewayError::Io)?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).map_err(GatewayError::Io)?;
    let _guard = TerminalGuard;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).map_err(GatewayError::Io)?;

    let mut data = fetch_snapshot(&client, &base_url, &target_label).await;
    let mut last_poll = Instant::now();

    loop {
        terminal
            .draw(|frame| render_dashboard(frame, &data))
            .map_err(GatewayError::Io)?;

        let remaining = poll_interval.saturating_sub(last_poll.elapsed());
        let wait = remaining.min(Duration::from_millis(100));
        if event::poll(wait).map_err(GatewayError::Io)?
            && let Event::Key(key) = event::read().map_err(GatewayError::Io)?
            && key.kind == KeyEventKind::Press
        {
            let quit = matches!(key.code, KeyCode::Char('q') | KeyCode::Esc)
                || (key.code == KeyCode::Char('c')
                    && key.modifiers.contains(KeyModifiers::CONTROL));
            if quit {
                break;
            }
        }

        if last_poll.elapsed() >= poll_interval {
            data = fetch_snapshot(&client, &base_url, &target_label).await;
            last_poll = Instant::now();
        }
    }

    restore_terminal(&mut terminal)?;
    Ok(())
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode().map_err(GatewayError::Io)?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen).map_err(GatewayError::Io)?;
    terminal.show_cursor().map_err(GatewayError::Io)?;
    Ok(())
}

fn build_client(args: &TopArgs) -> Result<(reqwest::Client, String, String)> {
    let builder = reqwest::Client::builder().timeout(Duration::from_secs(3));
    if let Some(socket) = &args.unix_socket {
        return build_unix_client(builder, socket);
    }
    let base = args.url.trim_end_matches('/').to_string();
    let client = builder
        .build()
        .map_err(|error| GatewayError::Internal(format!("cannot build HTTP client: {error}")))?;
    Ok((client, base.clone(), base))
}

#[cfg(unix)]
fn build_unix_client(
    builder: reqwest::ClientBuilder,
    socket: &Path,
) -> Result<(reqwest::Client, String, String)> {
    let client = builder
        .unix_socket(socket)
        .build()
        .map_err(|error| GatewayError::Internal(format!("cannot build Unix client: {error}")))?;
    Ok((
        client,
        "http://localhost".to_string(),
        format!("unix://{}", socket.display()),
    ))
}

#[cfg(not(unix))]
fn build_unix_client(
    _builder: reqwest::ClientBuilder,
    _socket: &Path,
) -> Result<(reqwest::Client, String, String)> {
    Err(GatewayError::Config(
        "--unix-socket is only supported on Unix hosts".into(),
    ))
}

async fn fetch_snapshot(
    client: &reqwest::Client,
    base_url: &str,
    target_label: &str,
) -> DashboardData {
    let health_url = format!("{base_url}/healthz");
    let ready_url = format!("{base_url}/readyz");
    let metrics_url = format!("{base_url}/metrics");
    let health_fut = fetch_json::<HealthzPayload>(client, &health_url);
    let ready_fut = fetch_json::<ReadyzPayload>(client, &ready_url);
    let metrics_fut = fetch_text(client, &metrics_url);
    let (health_res, ready_res, metrics_res) = tokio::join!(health_fut, ready_fut, metrics_fut);

    let mut errors = Vec::new();
    let health = match health_res {
        Ok(val) => Some(val),
        Err(err) => {
            errors.push(err);
            None
        }
    };
    let ready = match ready_res {
        Ok(val) => Some(val),
        Err(err) => {
            errors.push(err);
            None
        }
    };
    let metrics = match metrics_res {
        Ok(text) => parse_metrics_snapshot(&text),
        Err(err) => {
            errors.push(err);
            MetricsSnapshot::default()
        }
    };

    DashboardData {
        target_label: target_label.to_string(),
        health,
        ready,
        metrics,
        last_error: if errors.is_empty() {
            None
        } else {
            Some(errors.join("; "))
        },
    }
}

async fn fetch_json<T: serde::de::DeserializeOwned>(
    client: &reqwest::Client,
    url: &str,
) -> std::result::Result<T, String> {
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|error| format!("{url}: {error}"))?;
    let text = resp
        .text()
        .await
        .map_err(|error| format!("{url}: {error}"))?;
    serde_json::from_str(&text).map_err(|error| format!("{url}: invalid JSON ({error})"))
}

async fn fetch_text(client: &reqwest::Client, url: &str) -> std::result::Result<String, String> {
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|error| format!("{url}: {error}"))?;
    resp.text().await.map_err(|error| format!("{url}: {error}"))
}

fn render_dashboard(frame: &mut Frame<'_>, data: &DashboardData) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Min(8),
            Constraint::Length(6),
        ])
        .split(frame.area());

    let header_lines = match &data.health {
        Some(health) => {
            let token_badge = if health.tokens_enforced {
                "enforced"
            } else {
                "local-open"
            };
            let ws_badge = if health.workspace_enabled {
                "enabled"
            } else {
                "off"
            };
            let ready_badge = data
                .ready
                .as_ref()
                .map(|r| if r.ready { "READY" } else { "NOT READY" })
                .unwrap_or("UNKNOWN");
            vec![
                Line::from(vec![
                    Span::styled(
                        format!(" vouch v{} ", health.version),
                        Style::default()
                            .fg(Color::Black)
                            .bg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(format!(
                        "  status={}  ready={}  uptime={}s  target={}",
                        health.status, ready_badge, health.uptime_secs, data.target_label
                    )),
                ]),
                Line::from(format!(
                    " routes={}  backends={}  tokens={}  worktree={}   [q / Ctrl+C to quit]",
                    health.models, health.backends, token_badge, ws_badge
                )),
            ]
        }
        None => vec![
            Line::from(vec![
                Span::styled(
                    " vouch UNREACHABLE ",
                    Style::default()
                        .fg(Color::White)
                        .bg(Color::Red)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!("  target={}", data.target_label)),
            ]),
            Line::from(
                data.last_error
                    .clone()
                    .unwrap_or_else(|| "waiting for daemon...".into()),
            ),
        ],
    };

    let header = Paragraph::new(header_lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Gateway Overview "),
    );
    frame.render_widget(header, chunks[0]);

    let rows: Vec<Row<'_>> = data
        .ready
        .as_ref()
        .map(|ready| {
            ready
                .backends
                .iter()
                .map(|backend| {
                    let active = data
                        .metrics
                        .active_permits
                        .get(&backend.id)
                        .copied()
                        .unwrap_or(0);
                    let ok_exits = data
                        .metrics
                        .child_exits_ok
                        .get(&backend.id)
                        .copied()
                        .unwrap_or(0);
                    let err_exits = data
                        .metrics
                        .child_exits_err
                        .get(&backend.id)
                        .copied()
                        .unwrap_or(0);
                    let health = if backend.auth_expired {
                        "AUTH_EXPIRED"
                    } else {
                        match data.metrics.backend_health.get(&backend.id) {
                            Some(true) => "healthy",
                            Some(false) => "unhealthy",
                            None if backend.circuit == "open" => "open",
                            None => "ok",
                        }
                    };
                    let circuit_label = if backend.auth_expired {
                        format!(
                            "AUTH_EXPIRED ({}s, fails={})",
                            backend.open_remaining_secs, backend.consecutive_failures
                        )
                    } else if backend.circuit == "open" {
                        format!(
                            "OPEN ({}s, fails={})",
                            backend.open_remaining_secs, backend.consecutive_failures
                        )
                    } else if backend.circuit == "half_open" {
                        format!("HALF_OPEN (fails={})", backend.consecutive_failures)
                    } else {
                        format!("CLOSED (fails={})", backend.consecutive_failures)
                    };
                    let circuit_style = if backend.auth_expired {
                        Style::default()
                            .fg(Color::Magenta)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        match backend.circuit.as_str() {
                            "open" => Style::default().fg(Color::Red),
                            "half_open" => Style::default().fg(Color::Yellow),
                            _ => Style::default().fg(Color::Green),
                        }
                    };
                    Row::new(vec![
                        Cell::from(backend.id.clone()),
                        Cell::from(backend.kind.clone()),
                        Cell::from(health.to_string()),
                        Cell::from(format!("{active}/{}", backend.max_concurrency)),
                        Cell::from(circuit_label).style(circuit_style),
                        Cell::from(format!("{ok_exits} ok / {err_exits} err")),
                    ])
                })
                .collect()
        })
        .unwrap_or_default();

    let table = Table::new(
        rows,
        [
            Constraint::Min(18),
            Constraint::Length(8),
            Constraint::Length(11),
            Constraint::Length(12),
            Constraint::Min(22),
            Constraint::Min(16),
        ],
    )
    .header(
        Row::new(vec![
            "Backend",
            "Kind",
            "Health",
            "Permits",
            "Circuit Breaker",
            "Child Exits",
        ])
        .style(Style::default().add_modifier(Modifier::BOLD)),
    )
    .block(Block::default().borders(Borders::ALL).title(" Backends "));
    frame.render_widget(table, chunks[1]);

    let req_summary = if data.metrics.requests_by_proto_status.is_empty() {
        "none yet".to_string()
    } else {
        data.metrics
            .requests_by_proto_status
            .iter()
            .map(|((proto, status), count)| format!("{proto}:{status}={count}"))
            .collect::<Vec<_>>()
            .join("  ")
    };
    let footer_lines = vec![
        Line::from(format!(" Requests: {req_summary}")),
        Line::from(format!(
            " Fallbacks: {}   Token rate limits: {}",
            data.metrics.fallbacks_total, data.metrics.rate_limited_total
        )),
    ];
    let footer = Paragraph::new(footer_lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Traffic & Resilience "),
    );
    frame.render_widget(footer, chunks[2]);
}

/// Parse Prometheus exposition text into a [`MetricsSnapshot`].
pub fn parse_metrics_snapshot(text: &str) -> MetricsSnapshot {
    let mut snap = MetricsSnapshot::default();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((metric_part, value_part)) = line.rsplit_once(' ') else {
            continue;
        };
        let Ok(value_f64) = value_part.trim().parse::<f64>() else {
            continue;
        };
        let value = value_f64.max(0.0).round() as u64;
        let (name, labels) = parse_metric_name_and_labels(metric_part);

        match name {
            "vouch_active_permits" => {
                if let Some(backend) = labels.get("backend") {
                    snap.active_permits.insert(backend.clone(), value);
                }
            }
            "vouch_backend_health" => {
                if let Some(backend) = labels.get("backend") {
                    snap.backend_health.insert(backend.clone(), value > 0);
                }
            }
            "vouch_child_exit_total" => {
                if let Some(backend) = labels.get("backend") {
                    if labels.get("code").map(String::as_str) == Some("0") {
                        *snap.child_exits_ok.entry(backend.clone()).or_insert(0) += value;
                    } else {
                        *snap.child_exits_err.entry(backend.clone()).or_insert(0) += value;
                    }
                }
            }
            "vouch_requests_total" => {
                if let (Some(proto), Some(status)) = (labels.get("protocol"), labels.get("status"))
                {
                    *snap
                        .requests_by_proto_status
                        .entry((proto.clone(), status.clone()))
                        .or_insert(0) += value;
                }
            }
            "vouch_fallbacks_total" => {
                snap.fallbacks_total = snap.fallbacks_total.saturating_add(value);
            }
            "vouch_token_rate_limited_total" => {
                snap.rate_limited_total = snap.rate_limited_total.saturating_add(value);
            }
            _ => {}
        }
    }
    snap
}

fn parse_metric_name_and_labels(input: &str) -> (&str, BTreeMap<String, String>) {
    let mut labels = BTreeMap::new();
    let Some(open) = input.find('{') else {
        return (input.trim(), labels);
    };
    let Some(close) = input.rfind('}') else {
        return (input.trim(), labels);
    };
    let name = input[..open].trim();
    let raw_labels = &input[open + 1..close];
    for pair in raw_labels.split(',') {
        let Some((key, val)) = pair.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let val = val.trim().trim_matches('"');
        if !key.is_empty() {
            labels.insert(key.to_string(), val.to_string());
        }
    }
    (name, labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_prometheus_metrics_snapshot() {
        let text = r#"
# TYPE vouch_requests_total counter
vouch_requests_total{protocol="openai",endpoint="/v1/chat/completions",status="200"} 12
vouch_requests_total{protocol="openai",endpoint="/v1/responses",status="200"} 3
vouch_active_permits{backend="hermes-cli"} 1
vouch_child_exit_total{backend="hermes-cli",code="0"} 5
vouch_child_exit_total{backend="hermes-cli",code="1"} 2
vouch_fallbacks_total{from="a",to="b"} 4
vouch_token_rate_limited_total{token="ci"} 1
vouch_backend_health{backend="hermes-cli"} 1
"#;
        let snap = parse_metrics_snapshot(text);
        assert_eq!(snap.active_permits.get("hermes-cli"), Some(&1));
        assert_eq!(snap.child_exits_ok.get("hermes-cli"), Some(&5));
        assert_eq!(snap.child_exits_err.get("hermes-cli"), Some(&2));
        assert_eq!(
            snap.requests_by_proto_status
                .get(&("openai".into(), "200".into())),
            Some(&15)
        );
        assert_eq!(snap.fallbacks_total, 4);
        assert_eq!(snap.rate_limited_total, 1);
        assert_eq!(snap.backend_health.get("hermes-cli"), Some(&true));
    }
}
