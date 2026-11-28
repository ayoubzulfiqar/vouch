//! SPDX-License-Identifier: GPL-3.0-or-later
//! Deterministic SHA-256 response cache (`[cache]`).

use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use arc_swap::ArcSwap;
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use dashmap::DashMap;
use serde_json::json;
use sha2::{Digest, Sha256};
use tokio::{
    sync::{mpsc, watch},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

use crate::{
    config::CacheConfig,
    driver::{
        ChunkStream, Completion, ContentPart, ExecutionRequest, MessageContent, StreamChunk,
        ThinkingBlock, ToolCall, ToolCallDelta, Usage,
    },
    metrics,
    persistence::{PersistedCacheEntry, evict_oldest_if_full, instant_from_unix, unix_now_secs},
};

/// HTTP response header indicating deterministic cache outcome (`HIT`, `MISS`, or `BYPASS`).
pub const CACHE_STATUS_HEADER: HeaderName = HeaderName::from_static("x-vouch-cache");
/// HTTP response header attached when served by in-flight singleflight coalescing.
pub const COALESCED_HEADER: HeaderName = HeaderName::from_static("x-vouch-coalesced");

/// Cached completion plus original backend identifier.
#[derive(Debug, Clone)]
pub struct CachedCompletion {
    /// Completed response body, tool calls, finish reason, and BPE usage.
    pub completion: Completion,
    /// Original backend id surfaced in `x-vouch-backend`.
    pub backend_id: String,
}

/// Cache lookup classification for one routed request.
#[derive(Debug)]
pub enum CacheLookup {
    /// `[cache].enabled = false`; caching is inactive and `X-Vouch-Cache` is omitted.
    Disabled,
    /// Eligible request was served from memory; includes the stored [`CachedCompletion`].
    Hit(CachedCompletion),
    /// Eligible request was not cached yet; holds the deterministic SHA-256 key to populate.
    Miss(String),
    /// Caching skipped (`Cache-Control: no-cache`/`no-store`, high temperature, or worktree isolation).
    Bypass,
}

impl CacheLookup {
    /// Value for `X-Vouch-Cache` when `[cache].enabled` is true.
    pub fn header_value(&self) -> Option<HeaderValue> {
        match self {
            Self::Disabled => None,
            Self::Hit(_) => Some(HeaderValue::from_static("HIT")),
            Self::Miss(_) => Some(HeaderValue::from_static("MISS")),
            Self::Bypass => Some(HeaderValue::from_static("BYPASS")),
        }
    }

    /// SHA-256 key when this lookup was a [`CacheLookup::Miss`].
    pub fn miss_key(&self) -> Option<&str> {
        match self {
            Self::Miss(key) => Some(key.as_str()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct CachedEntry {
    content: String,
    finish_reason: String,
    tool_calls: Vec<ToolCall>,
    thinking: Vec<ThinkingBlock>,
    backend_id: String,
    prompt_tokens: u64,
    completion_tokens: u64,
    inserted_at: Instant,
    updated_unix_secs: u64,
    seq: u64,
}

/// Singleflight leader/waiter slot outcome pushed to concurrent identical requests.
#[derive(Debug, Clone)]
pub enum InflightResult {
    /// Leader finished with a shareable [`CachedCompletion`].
    Ready(CachedCompletion),
    /// Leader failed or dropped before producing a shareable completion.
    Aborted,
}

/// Role returned by [`ResponseCache::begin_or_join_inflight`].
pub enum CoalesceRole {
    /// Coalescing is inactive for this request (execute directly without registering).
    Direct,
    /// Caller won leadership; executes upstream once and notifies waiting peers via [`InflightLeaderGuard`].
    Leader(InflightLeaderGuard),
    /// Caller joined an identical active execution and awaits the leader's [`InflightResult`].
    Waiter(watch::Receiver<Option<InflightResult>>),
}

/// RAII singleflight leader guard that removes the key and unblocks waiters on drop.
pub struct InflightLeaderGuard {
    key: String,
    inflight: Arc<DashMap<String, watch::Sender<Option<InflightResult>>>>,
    tx: watch::Sender<Option<InflightResult>>,
    finished: bool,
}

impl InflightLeaderGuard {
    /// Publish the completed response to all queued identical callers and unregister the slot.
    pub fn complete(mut self, result: CachedCompletion) {
        self.finished = true;
        self.inflight.remove(&self.key);
        let _ = self.tx.send(Some(InflightResult::Ready(result)));
    }
}

impl Drop for InflightLeaderGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.inflight.remove(&self.key);
            let _ = self.tx.send(Some(InflightResult::Aborted));
        }
    }
}

/// Bounded TTL cache for low-temperature deterministic completions with
/// optional in-flight singleflight coalescing.
#[derive(Debug)]
pub struct ResponseCache {
    config: ArcSwap<CacheConfig>,
    entries: DashMap<String, CachedEntry>,
    inflight: Arc<DashMap<String, watch::Sender<Option<InflightResult>>>>,
    next_seq: AtomicU64,
}

impl ResponseCache {
    /// Build an empty cache configured from `[cache]`.
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config: ArcSwap::from_pointee(config),
            entries: DashMap::new(),
            inflight: Arc::new(DashMap::new()),
            next_seq: AtomicU64::new(1),
        }
    }

    /// True when `[cache].enabled` is active.
    pub fn enabled(&self) -> bool {
        self.config.load().enabled
    }

    /// Apply hot-reloaded `[cache]` settings without clearing live entries unless disabled.
    pub fn update_config(&self, next: CacheConfig) {
        if !next.enabled {
            self.entries.clear();
        }
        let max_entries = next.max_entries.max(1);
        self.config.store(Arc::new(next));
        while self.entries.len() > max_entries {
            let oldest = self
                .entries
                .iter()
                .min_by_key(|entry| entry.value().seq)
                .map(|entry| entry.key().clone());
            match oldest {
                Some(key) => {
                    self.entries.remove(&key);
                }
                None => break,
            }
        }
    }

    /// Number of currently held cache entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if zero entries are stored.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove every entry from memory, returning the number purged.
    pub fn purge(&self) -> usize {
        let count = self.entries.len();
        self.entries.clear();
        count
    }

    /// Check cache eligibility and look up any valid unexpired entry for `(route, request)`.
    pub fn lookup(
        &self,
        route: &str,
        request: &ExecutionRequest,
        headers: &HeaderMap,
        workspace_isolated: bool,
    ) -> CacheLookup {
        let cfg = self.config.load();
        let coalesce = cfg.coalesce_enabled();
        if !cfg.enabled && !coalesce {
            return CacheLookup::Disabled;
        }
        let effective_temp = request.temperature;
        let high_temp = effective_temp.is_some_and(|t| t > cfg.max_temperature);
        if workspace_isolated || high_temp || header_bypasses_cache(headers) {
            if cfg.enabled {
                metrics::record_cache_request("bypass");
                return CacheLookup::Bypass;
            }
            return CacheLookup::Disabled;
        }

        let key = compute_cache_key(route, request, effective_temp, cfg.normalize_whitespace);
        if cfg.enabled {
            let ttl = Duration::from_secs(cfg.ttl_secs.max(1));
            if let Some(entry) = self.entries.get(&key) {
                if entry.inserted_at.elapsed() <= ttl {
                    let completion = Completion {
                        content: entry.content.clone(),
                        finish_reason: entry.finish_reason.clone(),
                        usage: Usage::new(entry.prompt_tokens, entry.completion_tokens),
                        raw: None,
                        tool_calls: entry.tool_calls.clone(),
                        thinking: entry.thinking.clone(),
                    };
                    metrics::record_cache_request("hit");
                    return CacheLookup::Hit(CachedCompletion {
                        completion,
                        backend_id: entry.backend_id.clone(),
                    });
                }
                drop(entry);
                self.entries.remove(&key);
            }
            metrics::record_cache_request("miss");
        }

        CacheLookup::Miss(key)
    }

    /// True when singleflight request coalescing is currently active.
    pub fn coalesce_enabled(&self) -> bool {
        self.config.load().coalesce_enabled()
    }

    /// Enter or join an active singleflight execution slot for a deterministic key.
    pub fn begin_or_join_inflight(&self, key: Option<&str>, stream: bool) -> CoalesceRole {
        let cfg = self.config.load();
        let Some(key) = key else {
            return CoalesceRole::Direct;
        };
        if stream || !cfg.coalesce_enabled() {
            return CoalesceRole::Direct;
        }
        match self.inflight.entry(key.to_string()) {
            dashmap::mapref::entry::Entry::Occupied(occupied) => {
                CoalesceRole::Waiter(occupied.get().subscribe())
            }
            dashmap::mapref::entry::Entry::Vacant(vacant) => {
                let (tx, _rx) = watch::channel(None);
                vacant.insert(tx.clone());
                CoalesceRole::Leader(InflightLeaderGuard {
                    key: key.to_string(),
                    inflight: self.inflight.clone(),
                    tx,
                    finished: false,
                })
            }
        }
    }

    /// Store a completed response under `key`, evicting the oldest entry if full.
    pub fn store(&self, key: String, backend_id: &str, completion: &Completion) {
        let cfg = self.config.load();
        if !cfg.enabled {
            return;
        }
        if !self.entries.contains_key(&key) {
            evict_oldest_if_full(&self.entries, cfg.max_entries, |entry| entry.seq);
        }
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        self.entries.insert(
            key,
            CachedEntry {
                content: completion.content.clone(),
                finish_reason: completion.finish_reason.clone(),
                tool_calls: completion.tool_calls.clone(),
                thinking: completion.thinking.clone(),
                backend_id: backend_id.to_string(),
                prompt_tokens: completion.usage.prompt_tokens,
                completion_tokens: completion.usage.completion_tokens,
                inserted_at: Instant::now(),
                updated_unix_secs: unix_now_secs(),
                seq,
            },
        );
    }

    /// Export non-expired entries when `[cache].persist = true`.
    pub fn export_snapshot(&self) -> Vec<PersistedCacheEntry> {
        let cfg = self.config.load();
        if !cfg.enabled || !cfg.persist {
            return Vec::new();
        }
        let ttl = Duration::from_secs(cfg.ttl_secs.max(1));
        self.entries
            .iter()
            .filter(|entry| entry.inserted_at.elapsed() <= ttl)
            .map(|entry| PersistedCacheEntry {
                key: entry.key().clone(),
                content: entry.content.clone(),
                finish_reason: entry.finish_reason.clone(),
                tool_calls: entry.tool_calls.clone(),
                thinking: entry.thinking.clone(),
                backend_id: entry.backend_id.clone(),
                prompt_tokens: entry.prompt_tokens,
                completion_tokens: entry.completion_tokens,
                stored_at_unix_secs: entry.updated_unix_secs,
                seq: entry.seq,
            })
            .collect()
    }

    /// Restore non-expired cache entries loaded from `state_dir`.
    pub fn restore_snapshot(&self, saved: &[PersistedCacheEntry]) {
        let cfg = self.config.load();
        if !cfg.enabled || !cfg.persist || saved.is_empty() {
            return;
        }
        let ttl = Duration::from_secs(cfg.ttl_secs.max(1));
        let now = unix_now_secs();
        let mut max_seq = 0u64;
        for entry in saved {
            if self.entries.len() >= cfg.max_entries.max(1) {
                break;
            }
            let Some(inserted_at) = instant_from_unix(entry.stored_at_unix_secs, now, ttl) else {
                continue;
            };
            max_seq = max_seq.max(entry.seq);
            self.entries.insert(
                entry.key.clone(),
                CachedEntry {
                    content: entry.content.clone(),
                    finish_reason: entry.finish_reason.clone(),
                    tool_calls: entry.tool_calls.clone(),
                    thinking: entry.thinking.clone(),
                    backend_id: entry.backend_id.clone(),
                    prompt_tokens: entry.prompt_tokens,
                    completion_tokens: entry.completion_tokens,
                    inserted_at,
                    updated_unix_secs: entry.stored_at_unix_secs,
                    seq: entry.seq,
                },
            );
        }
        self.next_seq.fetch_max(max_seq + 1, Ordering::Relaxed);
    }

    /// Periodically expire stale cache entries until shutdown.
    pub fn spawn_eviction_task(self: &Arc<Self>, shutdown: CancellationToken) -> JoinHandle<()> {
        let cache = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let ttl = Duration::from_secs(cache.config.load().ttl_secs.max(1));
                        cache.entries.retain(|_, entry| entry.inserted_at.elapsed() <= ttl);
                    }
                    _ = shutdown.cancelled() => break,
                }
            }
        })
    }
}

/// Convert a buffered cached [`Completion`] into a bounded replay [`ChunkStream`].
pub fn synthesize_stream_from_completion(completion: &Completion) -> ChunkStream {
    let (tx, rx) = mpsc::channel(8);
    let content = completion.content.clone();
    let thinking = completion.thinking.clone();
    let tool_calls = completion.tool_calls.clone();
    let finish_reason = completion.finish_reason.clone();
    let usage = completion.usage;
    tokio::spawn(async move {
        for block in thinking {
            match block {
                ThinkingBlock::Thinking {
                    thinking,
                    signature,
                } => {
                    let mut chunk = StreamChunk::thinking(thinking);
                    chunk.signature_delta = signature;
                    if tx.send(Ok(chunk)).await.is_err() {
                        return;
                    }
                }
                ThinkingBlock::RedactedThinking { data } => {
                    let mut chunk = StreamChunk::delta("");
                    chunk.redacted_thinking = Some(data);
                    if tx.send(Ok(chunk)).await.is_err() {
                        return;
                    }
                }
            }
        }
        if !content.is_empty() && tx.send(Ok(StreamChunk::delta(content))).await.is_err() {
            return;
        }
        if !tool_calls.is_empty() {
            let deltas: Vec<ToolCallDelta> = tool_calls
                .into_iter()
                .enumerate()
                .map(|(index, call)| ToolCallDelta {
                    index,
                    id: Some(call.id),
                    name: Some(call.name),
                    arguments_delta: Some(call.arguments),
                })
                .collect();
            if tx.send(Ok(StreamChunk::tool_calls(deltas))).await.is_err() {
                return;
            }
        }
        let _ = tx
            .send(Ok(StreamChunk::finish(finish_reason, Some(usage))))
            .await;
    });
    Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
}

/// True if HTTP `Cache-Control` asks to bypass the gateway response cache.
pub fn header_bypasses_cache(headers: &HeaderMap) -> bool {
    headers
        .get_all(axum::http::header::CACHE_CONTROL)
        .iter()
        .filter_map(|val| val.to_str().ok())
        .flat_map(|val| val.split(','))
        .map(|directive| directive.trim().to_ascii_lowercase())
        .any(|directive| directive == "no-cache" || directive == "no-store")
}

fn normalize_ws(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Compute deterministic SHA-256 hex digest over `(route, messages, tools, response_format, temperature)`.
pub fn compute_cache_key(
    route: &str,
    request: &ExecutionRequest,
    effective_temperature: Option<f64>,
    normalize_whitespace: bool,
) -> String {
    let maybe_norm = |s: &str| {
        if normalize_whitespace {
            normalize_ws(s)
        } else {
            s.to_string()
        }
    };
    let system = request.system.as_deref().map(maybe_norm);
    let messages_json: Vec<serde_json::Value> = request
        .messages
        .iter()
        .map(|m| {
            let content = match &m.content {
                MessageContent::Text(t) => json!(maybe_norm(t)),
                MessageContent::Parts(parts) if normalize_whitespace => {
                    let mapped: Vec<ContentPart> = parts
                        .iter()
                        .map(|p| match p {
                            ContentPart::Text {
                                text,
                                cache_control,
                            } => ContentPart::Text {
                                text: normalize_ws(text),
                                cache_control: cache_control.clone(),
                            },
                            other => other.clone(),
                        })
                        .collect();
                    json!(mapped)
                }
                MessageContent::Parts(parts) => json!(parts),
            };
            json!({
                "role": m.role,
                "name": m.name,
                "content": content,
                "tool_call_id": m.tool_call_id,
                "tool_calls": m.tool_calls,
            })
        })
        .collect();

    let payload = json!({
        "v": if normalize_whitespace { 2 } else { 1 },
        "route": route,
        "system": system,
        "messages": messages_json,
        "tools": request.tools,
        "response_format": request.response_format.as_ref().map(|rf| rf.to_openai_json()),
        "temperature": effective_temperature.map(|t| format!("{t:.6}")),
        "max_tokens": request.max_tokens,
    });
    let canonical = payload.to_string();
    let digest = Sha256::digest(canonical.as_bytes());
    hex::encode(digest)
}
