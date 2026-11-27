//! SPDX-License-Identifier: GPL-3.0-or-later
//! BPE token accounting via `tiktoken-rs` plus cumulative per-token/per-backend
//! usage and latency analytics.

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::OnceLock,
    time::Duration,
};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tiktoken_rs::{CoreBPE, cl100k_base, o200k_base};

use crate::{auth::tokens::TokenBudgetStatus, driver::Usage};

const MAX_LATENCY_SAMPLES: usize = 1024;

static CL100K: OnceLock<Option<CoreBPE>> = OnceLock::new();
static O200K: OnceLock<Option<CoreBPE>> = OnceLock::new();

fn encoder_for_model(model: &str) -> Option<&'static CoreBPE> {
    let lower = model.to_ascii_lowercase();
    let prefer_o200k = lower.contains("gpt-4o")
        || lower.contains("gpt-5")
        || lower.contains("o1")
        || lower.contains("o3")
        || lower.contains("o4");
    if prefer_o200k && let Some(bpe) = O200K.get_or_init(|| o200k_base().ok()).as_ref() {
        return Some(bpe);
    }
    CL100K
        .get_or_init(|| cl100k_base().ok())
        .as_ref()
        .or_else(|| O200K.get_or_init(|| o200k_base().ok()).as_ref())
}

/// Count BPE tokens in `text` for `model`, falling back safely if BPE init fails.
pub fn count_tokens(model: &str, text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    match encoder_for_model(model) {
        Some(bpe) => bpe.encode_with_special_tokens(text).len() as u64,
        None => (text.chars().count() as u64).div_ceil(4),
    }
}

/// Estimate prompt and completion token usage for `model`.
pub fn estimate_usage(model: &str, prompt: &str, completion: &str) -> Usage {
    Usage::new(count_tokens(model, prompt), count_tokens(model, completion))
}

/// Aggregate request, token, and latency statistics for a token or backend.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageSummary {
    /// Total completed + failed invocations.
    pub requests: u64,
    /// Invocations that ended in error.
    pub errors: u64,
    /// Cumulative prompt BPE tokens.
    pub prompt_tokens: u64,
    /// Cumulative completion BPE tokens.
    pub completion_tokens: u64,
    /// Cumulative `prompt_tokens + completion_tokens`.
    pub total_tokens: u64,
    /// 50th percentile latency in milliseconds across recent calls.
    pub p50_ms: u64,
    /// 95th percentile latency in milliseconds across recent calls.
    pub p95_ms: u64,
    /// 99th percentile latency in milliseconds across recent calls.
    pub p99_ms: u64,
}

/// Usage metrics for one caller identity (`name`, never exposing raw credentials).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsageReport {
    /// Caller / token display name.
    pub name: String,
    /// Optional daily BPE budget.
    pub daily_token_budget: Option<u64>,
    /// Tokens consumed today (UTC) toward the daily budget.
    pub daily_tokens_used: u64,
    /// Cumulative statistics.
    #[serde(flatten)]
    pub summary: UsageSummary,
}

/// Usage metrics for one backend id.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendUsageReport {
    /// Configured backend id.
    pub id: String,
    /// Cumulative statistics.
    #[serde(flatten)]
    pub summary: UsageSummary,
}

/// Response payload returned by `GET /v1/usage` and printed by `vouch stats`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageReport {
    /// Process uptime in seconds.
    pub uptime_secs: u64,
    /// Gateway-wide totals across all callers.
    pub totals: UsageSummary,
    /// Per-token breakdown (no raw secrets).
    pub tokens: Vec<TokenUsageReport>,
    /// Per-backend breakdown.
    pub backends: Vec<BackendUsageReport>,
}

/// Serializable entity counters persisted to `state_dir`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersistedEntityStats {
    /// Total requests.
    pub requests: u64,
    /// Failed requests.
    pub errors: u64,
    /// Prompt tokens.
    pub prompt_tokens: u64,
    /// Completion tokens.
    pub completion_tokens: u64,
    /// Recent latency samples in milliseconds (bounded).
    #[serde(default)]
    pub latencies_ms: Vec<u64>,
}

/// Serializable snapshot of [`UsageTracker`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersistedUsage {
    /// Stats keyed by token display name.
    #[serde(default)]
    pub by_token: BTreeMap<String, PersistedEntityStats>,
    /// Stats keyed by backend id.
    #[serde(default)]
    pub by_backend: BTreeMap<String, PersistedEntityStats>,
}

#[derive(Debug, Default, Clone)]
struct EntityStats {
    requests: u64,
    errors: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    latencies_ms: VecDeque<u64>,
}

impl EntityStats {
    fn record_latency(&mut self, elapsed: Duration) {
        let ms = elapsed.as_millis().min(u128::from(u64::MAX)) as u64;
        if self.latencies_ms.len() >= MAX_LATENCY_SAMPLES {
            self.latencies_ms.pop_front();
        }
        self.latencies_ms.push_back(ms);
    }

    fn summary(&self) -> UsageSummary {
        let mut sorted: Vec<u64> = self.latencies_ms.iter().copied().collect();
        sorted.sort_unstable();
        UsageSummary {
            requests: self.requests,
            errors: self.errors,
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens: self.prompt_tokens.saturating_add(self.completion_tokens),
            p50_ms: percentile(&sorted, 50),
            p95_ms: percentile(&sorted, 95),
            p99_ms: percentile(&sorted, 99),
        }
    }

    fn to_persisted(&self) -> PersistedEntityStats {
        PersistedEntityStats {
            requests: self.requests,
            errors: self.errors,
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            latencies_ms: self.latencies_ms.iter().copied().collect(),
        }
    }

    fn from_persisted(p: &PersistedEntityStats) -> Self {
        let start = p.latencies_ms.len().saturating_sub(MAX_LATENCY_SAMPLES);
        Self {
            requests: p.requests,
            errors: p.errors,
            prompt_tokens: p.prompt_tokens,
            completion_tokens: p.completion_tokens,
            latencies_ms: p.latencies_ms[start..].iter().copied().collect(),
        }
    }
}

fn percentile(sorted: &[u64], pct: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) * pct).div_ceil(100);
    sorted[idx.min(sorted.len() - 1)]
}

/// Thread-safe cumulative usage & latency analytics store.
#[derive(Debug, Default)]
pub struct UsageTracker {
    by_token: DashMap<String, EntityStats>,
    by_backend: DashMap<String, EntityStats>,
}

impl UsageTracker {
    /// Create an empty analytics tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a completed call (both buffered and streaming).
    pub fn record_success(
        &self,
        token_name: &str,
        backend_id: &str,
        usage: Usage,
        elapsed: Duration,
    ) {
        {
            let mut entry = self.by_token.entry(token_name.to_string()).or_default();
            entry.requests = entry.requests.saturating_add(1);
            entry.prompt_tokens = entry.prompt_tokens.saturating_add(usage.prompt_tokens);
            entry.completion_tokens = entry
                .completion_tokens
                .saturating_add(usage.completion_tokens);
            entry.record_latency(elapsed);
        }
        {
            let mut entry = self.by_backend.entry(backend_id.to_string()).or_default();
            entry.requests = entry.requests.saturating_add(1);
            entry.prompt_tokens = entry.prompt_tokens.saturating_add(usage.prompt_tokens);
            entry.completion_tokens = entry
                .completion_tokens
                .saturating_add(usage.completion_tokens);
            entry.record_latency(elapsed);
        }
    }

    /// Record a failed call against the caller and optional backend.
    pub fn record_error(&self, token_name: &str, backend_id: Option<&str>, elapsed: Duration) {
        {
            let mut entry = self.by_token.entry(token_name.to_string()).or_default();
            entry.requests = entry.requests.saturating_add(1);
            entry.errors = entry.errors.saturating_add(1);
            entry.record_latency(elapsed);
        }
        if let Some(backend_id) = backend_id {
            let mut entry = self.by_backend.entry(backend_id.to_string()).or_default();
            entry.requests = entry.requests.saturating_add(1);
            entry.errors = entry.errors.saturating_add(1);
            entry.record_latency(elapsed);
        }
    }

    /// Build a safe snapshot report merging token budgets with cumulative stats.
    pub fn report(&self, uptime_secs: u64, budgets: &[TokenBudgetStatus]) -> UsageReport {
        let budget_map: HashMap<&str, &TokenBudgetStatus> =
            budgets.iter().map(|b| (b.name.as_str(), b)).collect();

        let mut token_map: BTreeMap<String, TokenUsageReport> = BTreeMap::new();
        for b in budgets {
            token_map.insert(
                b.name.clone(),
                TokenUsageReport {
                    name: b.name.clone(),
                    daily_token_budget: b.daily_token_budget,
                    daily_tokens_used: b.daily_tokens_used,
                    summary: UsageSummary::default(),
                },
            );
        }

        let mut totals = EntityStats::default();
        for entry in &self.by_token {
            totals.requests = totals.requests.saturating_add(entry.requests);
            totals.errors = totals.errors.saturating_add(entry.errors);
            totals.prompt_tokens = totals.prompt_tokens.saturating_add(entry.prompt_tokens);
            totals.completion_tokens = totals
                .completion_tokens
                .saturating_add(entry.completion_tokens);
            for &ms in &entry.latencies_ms {
                if totals.latencies_ms.len() >= MAX_LATENCY_SAMPLES {
                    totals.latencies_ms.pop_front();
                }
                totals.latencies_ms.push_back(ms);
            }
            let summary = entry.summary();
            let budget = budget_map.get(entry.key().as_str());
            token_map.insert(
                entry.key().clone(),
                TokenUsageReport {
                    name: entry.key().clone(),
                    daily_token_budget: budget.and_then(|b| b.daily_token_budget),
                    daily_tokens_used: budget.map(|b| b.daily_tokens_used).unwrap_or(0),
                    summary,
                },
            );
        }

        let mut backends: Vec<BackendUsageReport> = self
            .by_backend
            .iter()
            .map(|entry| BackendUsageReport {
                id: entry.key().clone(),
                summary: entry.summary(),
            })
            .collect();
        backends.sort_by(|a, b| a.id.cmp(&b.id));

        UsageReport {
            uptime_secs,
            totals: totals.summary(),
            tokens: token_map.into_values().collect(),
            backends,
        }
    }

    /// Snapshot current counters for disk persistence.
    pub fn snapshot(&self) -> PersistedUsage {
        let mut by_token = BTreeMap::new();
        for entry in &self.by_token {
            by_token.insert(entry.key().clone(), entry.to_persisted());
        }
        let mut by_backend = BTreeMap::new();
        for entry in &self.by_backend {
            by_backend.insert(entry.key().clone(), entry.to_persisted());
        }
        PersistedUsage {
            by_token,
            by_backend,
        }
    }

    /// Restore previously persisted usage counters on startup.
    pub fn restore(&self, saved: &PersistedUsage) {
        for (k, v) in &saved.by_token {
            self.by_token
                .insert(k.clone(), EntityStats::from_persisted(v));
        }
        for (k, v) in &saved.by_backend {
            self.by_backend
                .insert(k.clone(), EntityStats::from_persisted(v));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_bpe_tokens_for_cl100k_and_o200k() {
        assert_eq!(count_tokens("coding/default", ""), 0);
        let cl = count_tokens("coding/default", "hello world");
        let o200k = count_tokens("gpt-4o", "hello world");
        assert!(cl >= 2);
        assert!(o200k >= 2);
    }

    #[test]
    fn tracks_usage_and_quantiles() {
        let tracker = UsageTracker::new();
        tracker.record_success(
            "dev",
            "hermes",
            Usage::new(10, 20),
            Duration::from_millis(50),
        );
        tracker.record_success(
            "dev",
            "hermes",
            Usage::new(5, 15),
            Duration::from_millis(150),
        );
        tracker.record_error("dev", Some("hermes"), Duration::from_millis(10));

        let report = tracker.report(42, &[]);
        assert_eq!(report.totals.requests, 3);
        assert_eq!(report.totals.errors, 1);
        assert_eq!(report.totals.total_tokens, 50);
        assert!(report.totals.p95_ms >= 50);
    }
}
