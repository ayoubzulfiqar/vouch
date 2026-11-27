//! SPDX-License-Identifier: GPL-3.0-or-later
//! Built-in load and latency benchmark runner (`vouch bench`).

use std::{
    io::Write,
    sync::Arc,
    time::{Duration, Instant},
};

use futures::StreamExt;
use serde::Serialize;
use tokio::{sync::Semaphore, task::JoinSet};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    cli::BenchArgs,
    config::GatewayConfig,
    driver::{ChatMessage, DriverResponse, ExecutionRequest, WorkspaceMode},
    error::{GatewayError, Result},
    router::derive_session_key,
    state::AppState,
};

#[derive(Debug, Clone)]
struct SampleOutcome {
    ok: bool,
    ttft_ms: f64,
    total_ms: f64,
    fallback_used: bool,
    backend_id: Option<String>,
    error: Option<String>,
}

/// Summary metrics produced by `vouch bench`.
#[derive(Debug, Clone, Serialize)]
pub struct BenchReport {
    /// Target route model.
    pub model: String,
    /// Total requests executed.
    pub requests: usize,
    /// Concurrency cap used.
    pub concurrency: usize,
    /// Streaming mode.
    pub stream: bool,
    /// Successful completions.
    pub succeeded: usize,
    /// Failed requests.
    pub failed: usize,
    /// Requests that activated a fallback backend.
    pub fallbacks: usize,
    /// Wall-clock duration in milliseconds.
    pub wall_ms: f64,
    /// Completed requests per second.
    pub rps: f64,
    /// Minimum time-to-first-token (ms).
    pub ttft_min_ms: f64,
    /// Median time-to-first-token (ms).
    pub ttft_p50_ms: f64,
    /// 95th-percentile time-to-first-token (ms).
    pub ttft_p95_ms: f64,
    /// Minimum total request latency (ms).
    pub total_min_ms: f64,
    /// Median total request latency (ms).
    pub total_p50_ms: f64,
    /// 95th-percentile total request latency (ms).
    pub total_p95_ms: f64,
    /// Maximum total request latency (ms).
    pub total_max_ms: f64,
}

/// Execute `vouch bench`.
pub async fn run(args: &BenchArgs) -> Result<()> {
    if args.requests == 0 {
        return Err(GatewayError::Config(
            "--requests must be greater than zero".into(),
        ));
    }
    if args.concurrency == 0 {
        return Err(GatewayError::Config(
            "--concurrency must be greater than zero".into(),
        ));
    }

    let config = GatewayConfig::load(&args.config)?;
    let state = AppState::new(args.config.clone(), config)?;
    let model = state.resolve_model(args.model.as_deref())?;
    let report = run_benchmark(
        &state,
        &model,
        args.requests,
        args.concurrency,
        &args.prompt,
        args.stream,
    )
    .await?;
    state.shutdown().await;

    let mut stdout = std::io::stdout().lock();
    if args.json {
        writeln!(stdout, "{}", serde_json::to_string_pretty(&report)?)?;
    } else {
        writeln!(
            stdout,
            "vouch bench — model={} requests={} concurrency={} stream={}",
            report.model, report.requests, report.concurrency, report.stream
        )?;
        writeln!(
            stdout,
            "  succeeded={}/{}  failed={}  fallbacks={}  wall={:.1}ms  throughput={:.2} req/s",
            report.succeeded,
            report.requests,
            report.failed,
            report.fallbacks,
            report.wall_ms,
            report.rps
        )?;
        writeln!(
            stdout,
            "  TTFT (ms):   min={:.1}  p50={:.1}  p95={:.1}",
            report.ttft_min_ms, report.ttft_p50_ms, report.ttft_p95_ms
        )?;
        writeln!(
            stdout,
            "  Total (ms):  min={:.1}  p50={:.1}  p95={:.1}  max={:.1}",
            report.total_min_ms, report.total_p50_ms, report.total_p95_ms, report.total_max_ms
        )?;
    }

    if report.failed > 0 && report.succeeded == 0 {
        return Err(GatewayError::Upstream(
            "all benchmark requests failed".into(),
        ));
    }
    Ok(())
}

/// Run benchmark iterations programmatically against an [`AppState`].
pub async fn run_benchmark(
    state: &Arc<AppState>,
    model: &str,
    requests: usize,
    concurrency: usize,
    prompt: &str,
    stream: bool,
) -> Result<BenchReport> {
    let sem = Arc::new(Semaphore::new(concurrency.max(1)));
    let mut join_set = JoinSet::new();
    let wall_start = Instant::now();

    for idx in 0..requests {
        let permit_sem = sem.clone();
        let st = state.clone();
        let m = model.to_string();
        let p = prompt.to_string();
        join_set.spawn(async move {
            let Ok(_permit) = permit_sem.acquire_owned().await else {
                return SampleOutcome {
                    ok: false,
                    ttft_ms: 0.0,
                    total_ms: 0.0,
                    fallback_used: false,
                    backend_id: None,
                    error: Some("semaphore closed".into()),
                };
            };
            run_single_sample(&st, &m, &p, stream, idx).await
        });
    }

    let mut samples = Vec::with_capacity(requests);
    while let Some(res) = join_set.join_next().await {
        match res {
            Ok(outcome) => samples.push(outcome),
            Err(error) => samples.push(SampleOutcome {
                ok: false,
                ttft_ms: 0.0,
                total_ms: 0.0,
                fallback_used: false,
                backend_id: None,
                error: Some(format!("task join error: {error}")),
            }),
        }
    }
    let wall_elapsed = wall_start.elapsed();
    let wall_ms = wall_elapsed.as_secs_f64() * 1000.0;

    let mut succeeded = 0usize;
    let mut failed = 0usize;
    let mut fallbacks = 0usize;
    let mut ttfts = Vec::new();
    let mut totals = Vec::new();

    for sample in samples {
        if sample.fallback_used {
            fallbacks += 1;
        }
        if sample.ok {
            succeeded += 1;
            ttfts.push(sample.ttft_ms);
            totals.push(sample.total_ms);
        } else {
            failed += 1;
            let _ = (&sample.backend_id, &sample.error);
        }
    }

    let rps = if wall_elapsed.as_secs_f64() > 0.0 {
        succeeded as f64 / wall_elapsed.as_secs_f64()
    } else {
        0.0
    };

    Ok(BenchReport {
        model: model.to_string(),
        requests,
        concurrency,
        stream,
        succeeded,
        failed,
        fallbacks,
        wall_ms,
        rps,
        ttft_min_ms: min_val(&ttfts),
        ttft_p50_ms: percentile(&ttfts, 50),
        ttft_p95_ms: percentile(&ttfts, 95),
        total_min_ms: min_val(&totals),
        total_p50_ms: percentile(&totals, 50),
        total_p95_ms: percentile(&totals, 95),
        total_max_ms: max_val(&totals),
    })
}

async fn run_single_sample(
    state: &AppState,
    model: &str,
    prompt: &str,
    stream: bool,
    idx: usize,
) -> SampleOutcome {
    let req_id = format!("bench-{idx}-{}", Uuid::new_v4().simple());
    let session_key = derive_session_key(Some(&req_id), None, prompt);
    let request = ExecutionRequest {
        request_id: req_id,
        session_key,
        model: model.to_string(),
        system: None,
        messages: vec![ChatMessage::text("user", prompt)],
        prompt: format!("User:\n{prompt}"),
        stream,
        temperature: None,
        max_tokens: Some(64),
        workspace_mode: WorkspaceMode::None,
        workdir: None,
        cancel: CancellationToken::new(),
        tools: Vec::new(),
        tool_choice: None,
        response_format: None,
        backend_session_id: None,
        delta_messages: None,
        session_capture: Arc::new(parking_lot::Mutex::new(None)),
        extra: serde_json::Map::new(),
    };

    let started = Instant::now();
    let timeout = Duration::from_secs(state.config().server.request_timeout_secs.max(1));
    let routed = match tokio::time::timeout(
        timeout,
        state
            .router()
            .execute(request, WorkspaceMode::None, state.workspaces().as_ref()),
    )
    .await
    {
        Ok(Ok(routed)) => routed,
        Ok(Err(error)) => {
            return SampleOutcome {
                ok: false,
                ttft_ms: started.elapsed().as_secs_f64() * 1000.0,
                total_ms: started.elapsed().as_secs_f64() * 1000.0,
                fallback_used: false,
                backend_id: None,
                error: Some(error.to_string()),
            };
        }
        Err(_) => {
            return SampleOutcome {
                ok: false,
                ttft_ms: started.elapsed().as_secs_f64() * 1000.0,
                total_ms: started.elapsed().as_secs_f64() * 1000.0,
                fallback_used: false,
                backend_id: None,
                error: Some("request timed out".into()),
            };
        }
    };

    let backend_id = Some(routed.backend_id);
    let fallback_used = routed.fallback_used;
    match routed.response {
        DriverResponse::Complete(_) => {
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            SampleOutcome {
                ok: true,
                ttft_ms: ms,
                total_ms: ms,
                fallback_used,
                backend_id,
                error: None,
            }
        }
        DriverResponse::Stream(mut s) => {
            let mut ttft_ms = None;
            while let Some(item) = s.next().await {
                match item {
                    Ok(chunk) => {
                        if ttft_ms.is_none()
                            && (!chunk.delta.is_empty()
                                || chunk.thinking_delta.is_some()
                                || !chunk.tool_calls.is_empty())
                        {
                            ttft_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
                        }
                    }
                    Err(error) => {
                        let ms = started.elapsed().as_secs_f64() * 1000.0;
                        return SampleOutcome {
                            ok: false,
                            ttft_ms: ttft_ms.unwrap_or(ms),
                            total_ms: ms,
                            fallback_used,
                            backend_id,
                            error: Some(error.to_string()),
                        };
                    }
                }
            }
            let total_ms = started.elapsed().as_secs_f64() * 1000.0;
            SampleOutcome {
                ok: true,
                ttft_ms: ttft_ms.unwrap_or(total_ms),
                total_ms,
                fallback_used,
                backend_id,
                error: None,
            }
        }
    }
}

fn sorted_copy(values: &[f64]) -> Vec<f64> {
    let mut copy = values.to_vec();
    copy.sort_by(f64::total_cmp);
    copy
}

fn min_val(values: &[f64]) -> f64 {
    sorted_copy(values).first().copied().unwrap_or(0.0)
}

fn max_val(values: &[f64]) -> f64 {
    sorted_copy(values).last().copied().unwrap_or(0.0)
}

fn percentile(values: &[f64], pct: usize) -> f64 {
    let sorted = sorted_copy(values);
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() * pct).saturating_add(99) / 100).saturating_sub(1);
    sorted
        .get(idx.min(sorted.len() - 1))
        .copied()
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_percentiles_accurately() {
        let vals = [10.0, 20.0, 30.0, 40.0, 100.0];
        assert_eq!(min_val(&vals), 10.0);
        assert_eq!(percentile(&vals, 50), 30.0);
        assert_eq!(percentile(&vals, 95), 100.0);
        assert_eq!(max_val(&vals), 100.0);
    }
}
