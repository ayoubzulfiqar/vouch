//! SPDX-License-Identifier: GPL-3.0-or-later
//! Query and render live per-token and per-backend usage analytics (`vouch stats`).

use std::{io::Write, time::Duration};

use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};

use crate::{
    cli::{StatsArgs, build_daemon_client},
    error::{GatewayError, Result},
    usage::UsageReport,
};

/// Fetch `GET /v1/usage` from TCP or Unix socket and display formatted analytics.
pub async fn run(args: &StatsArgs) -> Result<()> {
    let report = fetch_usage(args).await?;
    let mut stdout = std::io::stdout().lock();
    if args.json {
        let json = serde_json::to_string_pretty(&report)?;
        writeln!(stdout, "{json}")?;
        return Ok(());
    }

    writeln!(
        stdout,
        "vouch usage summary (uptime: {}s)",
        report.uptime_secs
    )?;
    writeln!(
        stdout,
        "totals: requests={} errors={} prompt_tokens={} completion_tokens={} total_tokens={} p50={}ms p95={}ms p99={}ms",
        report.totals.requests,
        report.totals.errors,
        report.totals.prompt_tokens,
        report.totals.completion_tokens,
        report.totals.total_tokens,
        report.totals.p50_ms,
        report.totals.p95_ms,
        report.totals.p99_ms,
    )?;

    if !report.tokens.is_empty() {
        writeln!(stdout, "\ntokens:")?;
        for t in &report.tokens {
            let budget = match t.daily_token_budget {
                Some(cap) => format!("{}/{cap}", t.daily_tokens_used),
                None => format!("{}/unbounded", t.daily_tokens_used),
            };
            writeln!(
                stdout,
                "  {:<18} req={} err={} total_tok={} daily_utc={} p50={}ms p95={}ms p99={}ms",
                t.name,
                t.summary.requests,
                t.summary.errors,
                t.summary.total_tokens,
                budget,
                t.summary.p50_ms,
                t.summary.p95_ms,
                t.summary.p99_ms,
            )?;
        }
    }

    if !report.backends.is_empty() {
        writeln!(stdout, "\nbackends:")?;
        for b in &report.backends {
            writeln!(
                stdout,
                "  {:<18} req={} err={} total_tok={} p50={}ms p95={}ms p99={}ms",
                b.id,
                b.summary.requests,
                b.summary.errors,
                b.summary.total_tokens,
                b.summary.p50_ms,
                b.summary.p95_ms,
                b.summary.p99_ms,
            )?;
        }
    }

    Ok(())
}

/// Perform the HTTP request against `GET /v1/usage`.
pub async fn fetch_usage(args: &StatsArgs) -> Result<UsageReport> {
    let (client, base_url, _) = build_daemon_client(
        &args.url,
        args.unix_socket.as_deref(),
        Duration::from_secs(5),
    )?;
    let url = format!("{base_url}/v1/usage");
    let mut headers = HeaderMap::new();
    let token = args
        .token
        .clone()
        .or_else(|| std::env::var("VOUCH_API_KEY").ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    if let Some(tok) = token
        && let Ok(val) = HeaderValue::from_str(&format!("Bearer {tok}"))
    {
        headers.insert(AUTHORIZATION, val);
    }

    let response = client
        .get(&url)
        .headers(headers)
        .send()
        .await
        .map_err(|error| GatewayError::Upstream(format!("failed querying {url}: {error}")))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| GatewayError::Upstream(format!("failed reading {url}: {error}")))?;
    if !status.is_success() {
        return Err(GatewayError::Upstream(format!(
            "{url} returned HTTP {status}: {body}"
        )));
    }
    serde_json::from_str(&body)
        .map_err(|error| GatewayError::Upstream(format!("invalid usage JSON from {url}: {error}")))
}
