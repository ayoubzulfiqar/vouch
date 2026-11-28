//! Virtual ingress token enforcement.
//!
//! When no `[[tokens]]` are configured the gateway runs in zero-friction local
//! mode and accepts any (or no) credential. Once tokens exist, every ingress
//! call is checked with a constant-time hash comparison, an optional sliding
//! one-minute RPM window, and a glob-based model allowlist.

use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, OnceLock},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use arc_swap::ArcSwap;
use axum::http::HeaderMap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{
    config::{ModelAllowlist, TokenConfig, parse_rfc3339_unix_secs},
    error::{GatewayError, Result},
    metrics,
};

const WINDOW: Duration = Duration::from_secs(60);
const SECS_PER_DAY: u64 = 86_400;

/// Rolling UTC-day BPE token counter persisted by token SHA-256 hex digest.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DailyTokenCounter {
    /// UTC day number (`unix_secs / 86400`).
    pub utc_day: u64,
    /// Cumulative prompt + completion BPE tokens consumed on `utc_day`.
    pub tokens: u64,
}

/// Snapshot of one configured token's daily budget status (zero raw secrets).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBudgetStatus {
    /// Safe token display name.
    pub name: String,
    /// Configured daily BPE budget, if any.
    pub daily_token_budget: Option<u64>,
    /// Tokens consumed today (UTC).
    pub daily_tokens_used: u64,
}

fn utc_day_for(now: SystemTime) -> u64 {
    now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() / SECS_PER_DAY
}

fn seconds_until_utc_midnight(now: SystemTime) -> u64 {
    let secs = now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    (SECS_PER_DAY - (secs % SECS_PER_DAY)).max(1)
}

/// RAII permit guard enforcing `[[tokens]].max_concurrent` across buffered and streamed calls.
#[derive(Debug)]
pub struct TokenConcurrencyGuard {
    #[allow(dead_code)]
    permit: Option<OwnedSemaphorePermit>,
}

impl TokenConcurrencyGuard {
    /// No-op guard for tokens without `max_concurrent` limits or anonymous local requests.
    pub fn unlimited() -> Self {
        Self { permit: None }
    }
}

/// Combined mutable quota state guarded by a single lock so pre-execution
/// reservations and post-execution reconciliations are atomic.
#[derive(Debug)]
struct TokenBudgetState {
    daily: DailyTokenCounter,
    tpm_window: VecDeque<(Instant, u64)>,
    reserved_tokens: u64,
}

/// RAII token reservation acquired prior to dispatching an upstream backend call.
///
/// Reconciles reserved prompt BPE tokens against actual total (prompt +
/// completion) tokens via [`TokenBudgetReservation::settle`]. If dropped
/// unsettled (for example when pre-execution validation or setup fails), the
/// reserved tokens are released without debiting consumed usage.
#[derive(Debug)]
pub struct TokenBudgetReservation {
    policy: Option<Arc<TokenPolicy>>,
    reserved: u64,
    settled: bool,
}

impl TokenBudgetReservation {
    /// Empty reservation for callers with no configured token policy.
    pub fn none() -> Self {
        Self {
            policy: None,
            reserved: 0,
            settled: true,
        }
    }

    /// Reserved prompt token count.
    pub fn reserved_tokens(&self) -> u64 {
        self.reserved
    }

    /// Reconcile the reservation against actual consumed total tokens (`prompt + completion`).
    pub fn settle(&mut self, actual_total: u64) {
        if self.settled {
            return;
        }
        self.settled = true;
        if let Some(policy) = &self.policy {
            policy.settle_tokens(self.reserved, actual_total);
        }
    }
}

impl Drop for TokenBudgetReservation {
    fn drop(&mut self) {
        if !self.settled {
            self.settled = true;
            if let Some(policy) = &self.policy {
                policy.release_reservation(self.reserved);
            }
        }
    }
}

/// One configured credential.
#[derive(Debug)]
pub struct TokenPolicy {
    /// Label used in logs and metrics.
    pub name: String,
    /// SHA-256 of the presented secret.
    token_hash: [u8; 32],
    /// Hex digest used purely as an anonymous persistence key.
    hash_hex: String,
    /// Optional requests-per-minute cap.
    pub rpm: Option<u32>,
    /// Optional tokens-per-minute sliding 60-second cap.
    pub tpm: Option<u64>,
    /// Optional rolling UTC day BPE token cap.
    pub daily_token_budget: Option<u64>,
    /// Optional concurrent in-flight request limit.
    pub max_concurrent: Option<usize>,
    /// Optional expiration UNIX epoch seconds compiled from RFC-3339 `expires_at`.
    pub expires_at_unix_secs: Option<u64>,
    /// Optional backend allowlist; `None` permits every backend.
    allowed_backends: Option<Vec<String>>,
    /// Administrative privileges for `/admin/*` endpoints.
    pub admin: bool,
    /// Compiled model allowlist; `None` means every model.
    allowlist: Option<ModelAllowlist>,
    window: Mutex<VecDeque<Instant>>,
    budget_state: Mutex<TokenBudgetState>,
    concurrency: Option<Arc<Semaphore>>,
}

impl TokenPolicy {
    /// Compile a token policy.
    pub fn new(config: &TokenConfig) -> Result<Self> {
        let hash = Sha256::digest(config.token.as_bytes());
        let mut token_hash = [0u8; 32];
        token_hash.copy_from_slice(&hash);
        let hash_hex = hex::encode(token_hash);
        let name = config
            .name
            .clone()
            .unwrap_or_else(|| format!("token-{}", &hash_hex[..8]));
        let allowlist = if config.allowed_models.is_empty() {
            None
        } else {
            Some(ModelAllowlist::compile(&config.allowed_models)?)
        };
        let expires_at_unix_secs = config
            .expires_at
            .as_deref()
            .map(parse_rfc3339_unix_secs)
            .transpose()?;
        let concurrency = config
            .max_concurrent
            .map(|limit| Arc::new(Semaphore::new(limit.max(1))));
        Ok(Self {
            name,
            token_hash,
            hash_hex,
            rpm: config.rpm,
            tpm: config.tpm,
            daily_token_budget: config.daily_token_budget,
            max_concurrent: config.max_concurrent,
            expires_at_unix_secs,
            allowed_backends: config.allowed_backends.clone(),
            admin: config.admin,
            allowlist,
            window: Mutex::new(VecDeque::new()),
            budget_state: Mutex::new(TokenBudgetState {
                daily: DailyTokenCounter {
                    utc_day: utc_day_for(SystemTime::now()),
                    tokens: 0,
                },
                tpm_window: VecDeque::new(),
                reserved_tokens: 0,
            }),
            concurrency,
        })
    }

    /// Whether this token may call `model`.
    pub fn allows_model(&self, model: &str) -> bool {
        self.allowlist
            .as_ref()
            .map(|allowlist| allowlist.matches(model))
            .unwrap_or(true)
    }

    /// Optional backend allowlist for routing enforcement.
    pub fn allowed_backends(&self) -> Option<&[String]> {
        self.allowed_backends.as_deref()
    }

    /// Whether this token is expired as of `now`.
    fn check_expiry(&self, now: SystemTime) -> Result<()> {
        let Some(expires_at) = self.expires_at_unix_secs else {
            return Ok(());
        };
        let now_secs = now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        if now_secs >= expires_at {
            return Err(GatewayError::Unauthorized(format!(
                "token {:?} has expired",
                self.name
            )));
        }
        Ok(())
    }

    /// Acquire a bounded concurrency permit if `max_concurrent` is configured.
    pub fn acquire_concurrency_permit(&self) -> Result<TokenConcurrencyGuard> {
        let Some(sem) = &self.concurrency else {
            return Ok(TokenConcurrencyGuard::unlimited());
        };
        let limit = self.max_concurrent.unwrap_or(1);
        match sem.clone().try_acquire_owned() {
            Ok(permit) => Ok(TokenConcurrencyGuard {
                permit: Some(permit),
            }),
            Err(_) => {
                metrics::record_token_rate_limited(&self.name);
                Err(GatewayError::RateLimited {
                    retry_after_secs: 1,
                    message: format!(
                        "token {:?} exceeded its max_concurrent limit ({limit})",
                        self.name
                    ),
                })
            }
        }
    }

    fn roll_and_prune(state: &mut TokenBudgetState, now_sys: SystemTime, now_inst: Instant) -> u64 {
        let today = utc_day_for(now_sys);
        if state.daily.utc_day != today {
            state.daily.utc_day = today;
            state.daily.tokens = 0;
        }
        while let Some((ts, _)) = state.tpm_window.front() {
            if now_inst.duration_since(*ts) >= WINDOW {
                state.tpm_window.pop_front();
            } else {
                break;
            }
        }
        state.tpm_window.iter().map(|(_, toks)| *toks).sum()
    }

    fn check_quotas_locked(
        &self,
        state: &TokenBudgetState,
        tpm_used: u64,
        reserve_tokens: u64,
        now_sys: SystemTime,
        now_inst: Instant,
    ) -> Result<()> {
        if let Some(budget) = self.daily_token_budget {
            let projected = state.daily.tokens.saturating_add(state.reserved_tokens);
            if projected >= budget
                || (reserve_tokens > 0 && projected.saturating_add(reserve_tokens) > budget)
            {
                let retry_after_secs = seconds_until_utc_midnight(now_sys);
                metrics::record_token_rate_limited(&self.name);
                return Err(GatewayError::DailyTokenBudgetExceeded {
                    retry_after_secs,
                    message: format!(
                        "token {:?} exceeded its daily BPE budget of {budget} tokens ({} used, {} reserved today)",
                        self.name, state.daily.tokens, state.reserved_tokens
                    ),
                });
            }
        }
        if let Some(tpm) = self.tpm {
            let projected_tpm = tpm_used.saturating_add(state.reserved_tokens);
            if projected_tpm >= tpm
                || (reserve_tokens > 0 && projected_tpm.saturating_add(reserve_tokens) > tpm)
            {
                let retry_after_secs = state
                    .tpm_window
                    .front()
                    .map(|(front, _)| {
                        WINDOW
                            .saturating_sub(now_inst.duration_since(*front))
                            .as_secs()
                    })
                    .unwrap_or(1)
                    .max(1);
                metrics::record_token_rate_limited(&self.name);
                return Err(GatewayError::RateLimited {
                    retry_after_secs,
                    message: format!(
                        "token {:?} exceeded its {tpm} tokens-per-minute limit ({tpm_used} used, {} reserved in last 60s)",
                        self.name, state.reserved_tokens
                    ),
                });
            }
        }
        Ok(())
    }

    /// Atomically verify daily BPE budget and sliding 60-second TPM quota and
    /// reserve `estimated_prompt_tokens` while the request executes.
    pub fn check_and_reserve_tokens(
        self: &Arc<Self>,
        estimated_prompt_tokens: u64,
    ) -> Result<TokenBudgetReservation> {
        let now_sys = SystemTime::now();
        let now_inst = Instant::now();
        let mut state = self.budget_state.lock();
        let tpm_used = Self::roll_and_prune(&mut state, now_sys, now_inst);
        self.check_quotas_locked(&state, tpm_used, estimated_prompt_tokens, now_sys, now_inst)?;
        state.reserved_tokens = state
            .reserved_tokens
            .saturating_add(estimated_prompt_tokens);
        Ok(TokenBudgetReservation {
            policy: Some(self.clone()),
            reserved: estimated_prompt_tokens,
            settled: false,
        })
    }

    /// Release an unsettled prompt reservation without crediting usage.
    pub fn release_reservation(&self, reserved: u64) {
        if reserved == 0 {
            return;
        }
        let mut state = self.budget_state.lock();
        state.reserved_tokens = state.reserved_tokens.saturating_sub(reserved);
    }

    /// Release `reserved` in-flight tokens and record `actual_total` consumed BPE tokens.
    pub fn settle_tokens(&self, reserved: u64, actual_total: u64) {
        let now_sys = SystemTime::now();
        let now_inst = Instant::now();
        let mut state = self.budget_state.lock();
        state.reserved_tokens = state.reserved_tokens.saturating_sub(reserved);
        Self::roll_and_prune(&mut state, now_sys, now_inst);
        if actual_total > 0 {
            state.daily.tokens = state.daily.tokens.saturating_add(actual_total);
            if self.tpm.is_some() {
                state.tpm_window.push_back((now_inst, actual_total));
            }
        }
    }

    /// Record consumed prompt + completion BPE tokens against both the daily day bucket and 60s TPM window.
    pub fn record_tokens(&self, tokens: u64) {
        if tokens == 0 {
            return;
        }
        self.settle_tokens(0, tokens);
    }

    /// Snapshot current UTC day token consumption.
    pub fn daily_counter(&self) -> DailyTokenCounter {
        let today = utc_day_for(SystemTime::now());
        let mut state = self.budget_state.lock();
        if state.daily.utc_day != today {
            state.daily.utc_day = today;
            state.daily.tokens = 0;
        }
        state.daily
    }

    fn set_daily_counter_if_today(&self, saved: DailyTokenCounter) {
        let today = utc_day_for(SystemTime::now());
        if saved.utc_day != today {
            return;
        }
        let mut state = self.budget_state.lock();
        if state.daily.utc_day != today {
            state.daily = saved;
        } else {
            state.daily.tokens = state.daily.tokens.max(saved.tokens);
        }
    }

    fn check_rate_limit(&self) -> Result<()> {
        let now_sys = SystemTime::now();
        self.check_expiry(now_sys)?;
        let now = Instant::now();
        {
            let mut state = self.budget_state.lock();
            let tpm_used = Self::roll_and_prune(&mut state, now_sys, now);
            self.check_quotas_locked(&state, tpm_used, 0, now_sys, now)?;
        }
        let Some(rpm) = self.rpm else {
            return Ok(());
        };
        let mut window = self.window.lock();
        while let Some(front) = window.front() {
            if now.duration_since(*front) >= WINDOW {
                window.pop_front();
            } else {
                break;
            }
        }
        if window.len() >= rpm as usize {
            let retry_after_secs = window
                .front()
                .map(|front| WINDOW.saturating_sub(now.duration_since(*front)).as_secs())
                .unwrap_or(1)
                .max(1);
            metrics::record_token_rate_limited(&self.name);
            return Err(GatewayError::RateLimited {
                retry_after_secs,
                message: format!(
                    "token {:?} exceeded its {} requests-per-minute limit",
                    self.name, rpm
                ),
            });
        }
        window.push_back(now);
        Ok(())
    }
}

/// Authenticated caller identity.
#[derive(Debug, Clone)]
pub struct Principal {
    /// Display name.
    pub name: String,
    /// True in zero-friction local mode.
    pub anonymous: bool,
    policy: Option<Arc<TokenPolicy>>,
}

static SHARED_UNLIMITED_PERMIT: OnceLock<Arc<TokenConcurrencyGuard>> = OnceLock::new();

impl Principal {
    /// Anonymous local principal.
    pub fn local() -> Self {
        Self {
            name: "local".into(),
            anonymous: true,
            policy: None,
        }
    }

    /// Shared no-op concurrency permit for anonymous or unlimited callers.
    pub fn unlimited_permit() -> Arc<TokenConcurrencyGuard> {
        SHARED_UNLIMITED_PERMIT
            .get_or_init(|| Arc::new(TokenConcurrencyGuard::unlimited()))
            .clone()
    }

    /// Whether this caller may use `model`.
    pub fn allows_model(&self, model: &str) -> bool {
        self.policy
            .as_ref()
            .map(|policy| policy.allows_model(model))
            .unwrap_or(true)
    }

    /// Whether this caller may use either the requested model/alias or resolved canonical route.
    pub fn allows_model_or_resolved(&self, requested: &str, resolved: &str) -> bool {
        self.allows_model(requested) || self.allows_model(resolved)
    }

    /// Configured backend allowlist for this caller, if any.
    pub fn allowed_backends(&self) -> Option<Vec<String>> {
        self.policy
            .as_ref()
            .and_then(|policy| policy.allowed_backends().map(|items| items.to_vec()))
    }

    /// True if this token grants administrative control access (`/admin/*`).
    pub fn is_admin(&self) -> bool {
        self.policy.as_ref().is_some_and(|policy| policy.admin)
    }

    /// Acquire a bounded concurrency permit if configured on this token.
    pub fn acquire_concurrency_permit(&self) -> Result<Arc<TokenConcurrencyGuard>> {
        match &self.policy {
            Some(policy) => Ok(Arc::new(policy.acquire_concurrency_permit()?)),
            None => Ok(Self::unlimited_permit()),
        }
    }

    /// Atomically reserve estimated prompt BPE tokens against daily and TPM budgets.
    pub fn reserve_tokens(&self, estimated_prompt_tokens: u64) -> Result<TokenBudgetReservation> {
        match &self.policy {
            Some(policy) => policy.check_and_reserve_tokens(estimated_prompt_tokens),
            None => Ok(TokenBudgetReservation::none()),
        }
    }

    /// Record completed prompt + completion BPE tokens against this caller's daily and TPM budgets.
    pub fn record_tokens(&self, tokens: u64) {
        if let Some(policy) = &self.policy {
            policy.record_tokens(tokens);
        }
    }

    /// Whether the caller is authenticated by a configured token.
    pub fn is_authenticated(&self) -> bool {
        self.policy.is_some()
    }
}

/// Token registry with atomic reload support.
#[derive(Debug)]
pub struct TokenRegistry {
    policies: ArcSwap<Vec<Arc<TokenPolicy>>>,
}

impl TokenRegistry {
    /// Build a registry from configuration.
    pub fn new(tokens: &[TokenConfig]) -> Result<Self> {
        let registry = Self {
            policies: ArcSwap::from_pointee(Vec::new()),
        };
        registry.reload(tokens)?;
        Ok(registry)
    }

    /// Replace all policies, preserving today's BPE usage counters for matching token hashes.
    pub fn reload(&self, tokens: &[TokenConfig]) -> Result<()> {
        let prior = self.export_daily_state();
        let mut policies = Vec::with_capacity(tokens.len());
        for token in tokens {
            let policy = TokenPolicy::new(token)?;
            if let Some(saved) = prior.get(&policy.hash_hex) {
                policy.set_daily_counter_if_today(*saved);
            }
            policies.push(Arc::new(policy));
        }
        self.policies.store(Arc::new(policies));
        Ok(())
    }

    /// Snapshot current UTC-day token counters keyed by anonymous SHA-256 hex digest.
    pub fn export_daily_state(&self) -> BTreeMap<String, DailyTokenCounter> {
        let mut map = BTreeMap::new();
        for policy in self.policies.load().iter() {
            let counter = policy.daily_counter();
            if counter.tokens > 0 {
                map.insert(policy.hash_hex.clone(), counter);
            }
        }
        map
    }

    /// Restore previously persisted daily token counters matching today's UTC date.
    pub fn restore_daily_state(&self, saved: &BTreeMap<String, DailyTokenCounter>) {
        for policy in self.policies.load().iter() {
            if let Some(counter) = saved.get(&policy.hash_hex) {
                policy.set_daily_counter_if_today(*counter);
            }
        }
    }

    /// Summary of configured token daily budgets (safe to expose without secrets).
    pub fn budget_statuses(&self) -> Vec<TokenBudgetStatus> {
        self.policies
            .load()
            .iter()
            .map(|policy| TokenBudgetStatus {
                name: policy.name.clone(),
                daily_token_budget: policy.daily_token_budget,
                daily_tokens_used: policy.daily_counter().tokens,
            })
            .collect()
    }

    /// Whether token enforcement is active.
    pub fn enabled(&self) -> bool {
        !self.policies.load().is_empty()
    }

    /// Authenticate without checking a model.
    pub fn authenticate(&self, headers: &HeaderMap) -> Result<Principal> {
        self.authenticate_with_key_hint(headers, None)
    }

    /// Authenticate checking headers first, then an optional query-string `key` override.
    pub fn authenticate_with_key_hint(
        &self,
        headers: &HeaderMap,
        key_hint: Option<&str>,
    ) -> Result<Principal> {
        let policies = self.policies.load_full();
        if policies.is_empty() {
            return Ok(Principal::local());
        }
        let presented = extract_credential(headers)
            .or_else(|| {
                key_hint
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(ToString::to_string)
            })
            .ok_or_else(|| {
                GatewayError::Unauthorized(
                    "missing API key: send Authorization: Bearer <token>, x-api-key, or x-goog-api-key"
                        .into(),
                )
            })?;
        let presented_digest = Sha256::digest(presented.as_bytes());
        let mut presented_hash = [0u8; 32];
        presented_hash.copy_from_slice(&presented_digest);
        let mut matched: Option<Arc<TokenPolicy>> = None;
        // Iterate all policies so timing does not reveal the matching index.
        for policy in policies.iter() {
            if policy.token_hash.ct_eq(&presented_hash).into() {
                matched = Some(policy.clone());
            }
        }
        let policy = matched.ok_or_else(|| GatewayError::Unauthorized("invalid API key".into()))?;
        policy.check_rate_limit()?;
        Ok(Principal {
            name: policy.name.clone(),
            anonymous: false,
            policy: Some(policy),
        })
    }

    /// Authenticate and enforce the token's model allowlist.
    pub fn authorize(&self, headers: &HeaderMap, model: &str) -> Result<Principal> {
        self.authorize_route(headers, model, model)
    }

    /// Authenticate and enforce `allowed_models` against both requested alias/model and resolved route.
    pub fn authorize_route(
        &self,
        headers: &HeaderMap,
        requested: &str,
        resolved: &str,
    ) -> Result<Principal> {
        self.authorize_route_with_key_hint(headers, None, requested, resolved)
    }

    /// Authenticate using headers or query `key` override and enforce model scope.
    pub fn authorize_route_with_key_hint(
        &self,
        headers: &HeaderMap,
        key_hint: Option<&str>,
        requested: &str,
        resolved: &str,
    ) -> Result<Principal> {
        let principal = self.authenticate_with_key_hint(headers, key_hint)?;
        if !principal.allows_model_or_resolved(requested, resolved) {
            return Err(GatewayError::Forbidden(format!(
                "token {:?} is not allowed to use model {requested:?}",
                principal.name
            )));
        }
        Ok(principal)
    }
}

fn extract_credential(headers: &HeaderMap) -> Option<String> {
    if let Some(value) = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    {
        let value = value.trim();
        if let Some(token) = value
            .strip_prefix("Bearer ")
            .or_else(|| value.strip_prefix("bearer "))
        {
            let token = token.trim();
            if !token.is_empty() {
                return Some(token.to_string());
            }
        }
        // Some local clients send the raw token without a scheme.
        if !value.is_empty() && !value.contains(' ') {
            return Some(value.to_string());
        }
    }
    headers
        .get("x-api-key")
        .or_else(|| headers.get("x-goog-api-key"))
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers(token: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}")).expect("header"),
        );
        headers
    }

    #[test]
    fn local_mode_accepts_anything() {
        let registry = TokenRegistry::new(&[]).expect("registry");
        assert!(registry.authenticate(&HeaderMap::new()).is_ok());
        assert!(
            registry
                .authorize(&headers("anything"), "any/model")
                .is_ok()
        );
    }

    fn token_cfg(token: &str) -> TokenConfig {
        TokenConfig {
            token: token.into(),
            name: Some("test".into()),
            rpm: None,
            tpm: None,
            daily_token_budget: None,
            max_concurrent: None,
            expires_at: None,
            allowed_models: Vec::new(),
            allowed_backends: None,
            admin: false,
        }
    }

    #[test]
    fn enforces_token_and_model_scope() {
        let mut cfg = token_cfg("vouch-secret-token");
        cfg.allowed_models = vec!["coding/*".into()];
        let registry = TokenRegistry::new(&[cfg]).expect("registry");
        assert!(registry.authenticate(&HeaderMap::new()).is_err());
        assert!(
            registry
                .authenticate(&headers("wrong-token-value"))
                .is_err()
        );
        assert!(
            registry
                .authorize(&headers("vouch-secret-token"), "coding/default")
                .is_ok()
        );
        assert!(
            registry
                .authorize_route(
                    &headers("vouch-secret-token"),
                    "fast-coder",
                    "coding/default"
                )
                .is_ok()
        );
        assert!(
            registry
                .authorize(&headers("vouch-secret-token"), "hermes/agent")
                .is_err()
        );
    }

    #[test]
    fn enforces_rpm() {
        let mut cfg = token_cfg("vouch-secret-token");
        cfg.rpm = Some(2);
        let registry = TokenRegistry::new(&[cfg]).expect("registry");
        assert!(
            registry
                .authenticate(&headers("vouch-secret-token"))
                .is_ok()
        );
        assert!(
            registry
                .authenticate(&headers("vouch-secret-token"))
                .is_ok()
        );
        let error = registry
            .authenticate(&headers("vouch-secret-token"))
            .expect_err("must rate limit");
        assert!(matches!(error, GatewayError::RateLimited { .. }));
    }

    #[test]
    fn accepts_x_api_key_and_goog_key() {
        let mut cfg = token_cfg("vouch-secret-token");
        cfg.name = None;
        let registry = TokenRegistry::new(&[cfg]).expect("registry");
        let mut h1 = HeaderMap::new();
        h1.insert("x-api-key", HeaderValue::from_static("vouch-secret-token"));
        assert!(registry.authenticate(&h1).is_ok());

        let mut h2 = HeaderMap::new();
        h2.insert(
            "x-goog-api-key",
            HeaderValue::from_static("vouch-secret-token"),
        );
        assert!(registry.authenticate(&h2).is_ok());
        assert!(
            registry
                .authenticate_with_key_hint(&HeaderMap::new(), Some("vouch-secret-token"))
                .is_ok()
        );
    }

    #[test]
    fn enforces_daily_token_budget_and_preserves_across_reload() {
        let mut cfg = token_cfg("vouch-secret-budget");
        cfg.name = Some("budgeted".into());
        cfg.daily_token_budget = Some(100);
        let registry = TokenRegistry::new(std::slice::from_ref(&cfg)).expect("registry");
        let p = registry
            .authenticate(&headers("vouch-secret-budget"))
            .expect("initial auth ok");
        p.record_tokens(120);
        // Reload keeps daily usage counter for unchanged token hash.
        registry.reload(std::slice::from_ref(&cfg)).expect("reload");
        let err = registry
            .authenticate(&headers("vouch-secret-budget"))
            .expect_err("daily budget must block");
        assert!(matches!(
            err,
            GatewayError::DailyTokenBudgetExceeded {
                retry_after_secs,
                ..
            } if (1..=86_400).contains(&retry_after_secs)
        ));
    }

    #[test]
    fn enforces_tpm_expiry_max_concurrent_and_admin() {
        let mut tpm_cfg = token_cfg("vouch-tpm-token");
        tpm_cfg.tpm = Some(50);
        tpm_cfg.max_concurrent = Some(1);
        tpm_cfg.allowed_backends = Some(vec!["fast".into()]);
        tpm_cfg.admin = true;
        let registry = TokenRegistry::new(&[tpm_cfg]).expect("registry");

        let p = registry
            .authenticate(&headers("vouch-tpm-token"))
            .expect("auth");
        assert!(p.is_admin());
        assert_eq!(p.allowed_backends(), Some(vec!["fast".to_string()]));
        let permit = p.acquire_concurrency_permit().expect("first permit");
        assert!(p.acquire_concurrency_permit().is_err());
        drop(permit);
        let _permit2 = p.acquire_concurrency_permit().expect("released permit");

        p.record_tokens(50);
        let tpm_err = registry
            .authenticate(&headers("vouch-tpm-token"))
            .expect_err("tpm exceeded");
        assert!(matches!(tpm_err, GatewayError::RateLimited { .. }));

        let mut exp_cfg = token_cfg("vouch-expired-token");
        exp_cfg.expires_at = Some("2020-01-01T00:00:00Z".into());
        let exp_reg = TokenRegistry::new(&[exp_cfg]).expect("exp registry");
        let err = exp_reg
            .authenticate(&headers("vouch-expired-token"))
            .expect_err("expired");
        assert!(matches!(err, GatewayError::Unauthorized(_)));
    }

    #[test]
    fn atomic_pre_execution_reservation_prevents_concurrent_budget_overshoot() {
        let mut cfg = token_cfg("vouch-reserve-token");
        cfg.daily_token_budget = Some(100);
        cfg.tpm = Some(100);
        let registry = TokenRegistry::new(&[cfg]).expect("registry");

        let p = registry
            .authenticate(&headers("vouch-reserve-token"))
            .expect("auth");
        // First request reserves 70 estimated prompt tokens.
        let mut res1 = p.reserve_tokens(70).expect("reserve 70");
        // Concurrent second request asking for 50 projected tokens must be rejected (70 + 50 > 100).
        let err = p.reserve_tokens(50).expect_err("must prevent overshoot");
        assert!(matches!(err, GatewayError::DailyTokenBudgetExceeded { .. }));

        // Settle res1 with only 60 actual total tokens; freeing 10 reserved headroom.
        res1.settle(60);
        // Now 40 remaining tokens fit cleanly, and dropping unsettled releases reservation.
        let res2 = p.reserve_tokens(40).expect("reserve 40");
        drop(res2);
        // Since res2 dropped unsettled, 40 tokens remain available again.
        let mut res3 = p.reserve_tokens(40).expect("reserve 40 after drop release");
        res3.settle(40);
        assert_eq!(
            p.policy.as_ref().expect("policy").daily_counter().tokens,
            100
        );
    }
}
