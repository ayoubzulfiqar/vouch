//! SPDX-License-Identifier: GPL-3.0-or-later
//! Canonical cross-agent session ledger (`[shared_memory]`).
//!
//! Records redacted conversation turns, tool execution summaries, touched
//! repository paths, token totals, and manual checkpoints across agents so
//! handoffs occur deterministically without spending LLM summarization tokens.

use std::{
    collections::VecDeque,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::{
    persistence::{
        PersistedCanonicalTurn, PersistedHandoffEvent, PersistedSharedSession,
        evict_oldest_if_full, instant_from_unix, unix_now_secs,
    },
    telemetry,
};

const MAX_FILES_PER_SESSION: usize = 128;
const MAX_CHECKPOINTS_PER_SESSION: usize = 64;
const MAX_HANDOFF_HISTORY: usize = 64;
const MAX_TURN_TEXT_CHARS: usize = 2000;

/// Why session execution moved from one backend agent to another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandoffReason {
    /// Earlier backend failed, timed out, or had an open circuit / saturation.
    Failover,
    /// Scheduling strategy (`least_latency` or `hedged`) selected another candidate.
    PolicyReorder,
    /// Caller switched model/route within the same `X-Session-ID`.
    ModelSwitch,
}

impl HandoffReason {
    /// Wire/metric label for this handoff trigger.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Failover => "failover",
            Self::PolicyReorder => "policy_reorder",
            Self::ModelSwitch => "model_switch",
        }
    }
}

/// Recorded cross-agent transition within a session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HandoffEvent {
    /// Previous active backend identifier.
    pub from_backend: String,
    /// Receiving backend identifier.
    pub to_backend: String,
    /// Trigger reason (`failover`, `policy_reorder`, `model_switch`).
    pub reason: String,
    /// Event timestamp in seconds since UNIX epoch.
    pub timestamp: u64,
}

/// One normalized, secret-redacted turn in the canonical session ledger.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalTurn {
    /// Turn role (`user`, `assistant`, `tool`).
    pub role: String,
    /// Redacted textual message content.
    pub content: String,
    /// Concise redacted summary of tool calls or execution outcomes, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_summary: Option<String>,
    /// Backend that handled or observed this turn.
    pub backend_id: String,
    /// Timestamp in seconds since UNIX epoch.
    pub timestamp: u64,
}

/// Complete canonical state retained for one `session_id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionLedgerEntry {
    /// Session identifier (`X-Session-ID` or deterministic fallback key).
    pub session_id: String,
    /// Creation UNIX timestamp (seconds).
    pub created_at: u64,
    /// Last update UNIX timestamp (seconds).
    pub updated_at: u64,
    /// Last resolved route/model handled in this session.
    pub last_model: Option<String>,
    /// Currently active backend id.
    pub active_backend: Option<String>,
    /// Currently active credential profile id, if any.
    pub active_profile: Option<String>,
    /// Chronological handoff timeline.
    pub backend_history: Vec<HandoffEvent>,
    /// Rolling window of canonical turns (capped by `max_turns_per_session`).
    pub turns: VecDeque<CanonicalTurn>,
    /// Deduplicated sanitized relative repository paths touched/mentioned.
    pub touched_files: Vec<String>,
    /// Redacted manual/MCP checkpoint notes.
    pub checkpoints: Vec<String>,
    /// Cumulative prompt BPE tokens across agents.
    pub total_prompt_tokens: u64,
    /// Cumulative completion BPE tokens across agents.
    pub total_completion_tokens: u64,
    #[serde(skip)]
    last_used: Option<Instant>,
    #[serde(skip)]
    seq: u64,
}

impl SessionLedgerEntry {
    /// Monotonic last-used timestamp initialized safely on deserialization/restore.
    fn touched_elapsed(&self) -> Duration {
        self.last_used
            .map(|inst| inst.elapsed())
            .unwrap_or(Duration::ZERO)
    }
}

/// Compact summary returned by `GET /v1/sessions` and MCP `vouch_session_status`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    /// Session identifier.
    pub session_id: String,
    /// Currently active backend id.
    pub active_backend: Option<String>,
    /// Currently active credential profile id.
    pub active_profile: Option<String>,
    /// Last resolved model route.
    pub last_model: Option<String>,
    /// Number of retained canonical turns.
    pub turn_count: usize,
    /// Number of cross-agent handoffs in this session.
    pub handoff_count: usize,
    /// Touched/referenced relative file paths.
    pub touched_files: Vec<String>,
    /// Number of recorded checkpoints.
    pub checkpoint_count: usize,
    /// Cumulative prompt tokens.
    pub total_prompt_tokens: u64,
    /// Cumulative completion tokens.
    pub total_completion_tokens: u64,
    /// Creation timestamp (UNIX seconds).
    pub created_at: u64,
    /// Last activity timestamp (UNIX seconds).
    pub updated_at: u64,
}

impl From<&SessionLedgerEntry> for SessionSummary {
    fn from(entry: &SessionLedgerEntry) -> Self {
        Self {
            session_id: entry.session_id.clone(),
            active_backend: entry.active_backend.clone(),
            active_profile: entry.active_profile.clone(),
            last_model: entry.last_model.clone(),
            turn_count: entry.turns.len(),
            handoff_count: entry.backend_history.len(),
            touched_files: entry.touched_files.clone(),
            checkpoint_count: entry.checkpoints.len(),
            total_prompt_tokens: entry.total_prompt_tokens,
            total_completion_tokens: entry.total_completion_tokens,
            created_at: entry.created_at,
            updated_at: entry.updated_at,
        }
    }
}

/// Parameters recorded after a request turn completes.
#[derive(Debug, Clone)]
pub struct RecordTurnInput {
    /// Session key.
    pub session_id: String,
    /// Resolved route name.
    pub model: String,
    /// Backend that fulfilled the turn.
    pub backend_id: String,
    /// Credential profile id used, if any.
    pub profile_id: Option<String>,
    /// Latest user prompt text for this turn.
    pub user_text: String,
    /// Assistant output text produced in this turn.
    pub assistant_text: String,
    /// Concise summary of tool calls or arguments.
    pub tool_summary: Option<String>,
    /// Additional candidate path strings (e.g. from tool call arguments).
    pub raw_path_hints: Vec<String>,
    /// Workspace root used to sanitize paths safely.
    pub workspace_root: Option<std::path::PathBuf>,
    /// Prompt tokens consumed in this turn.
    pub prompt_tokens: u64,
    /// Completion tokens generated in this turn.
    pub completion_tokens: u64,
}

/// Bounded thread-safe cross-agent canonical session ledger.
#[derive(Debug)]
pub struct SessionLedgerStore {
    entries: DashMap<String, SessionLedgerEntry>,
    max_sessions: AtomicUsize,
    max_turns_per_session: AtomicUsize,
    seq: AtomicU64,
    total_handoffs: AtomicU64,
}

impl Default for SessionLedgerStore {
    fn default() -> Self {
        Self::new(10_000, 32)
    }
}

impl SessionLedgerStore {
    /// Create a bounded session ledger store.
    pub fn new(max_sessions: usize, max_turns_per_session: usize) -> Self {
        Self {
            entries: DashMap::new(),
            max_sessions: AtomicUsize::new(max_sessions.max(1)),
            max_turns_per_session: AtomicUsize::new(max_turns_per_session.max(1)),
            seq: AtomicU64::new(1),
            total_handoffs: AtomicU64::new(0),
        }
    }

    /// Update retention ceilings on config reload.
    pub fn set_limits(&self, max_sessions: usize, max_turns_per_session: usize) {
        let cap = max_sessions.max(1);
        let max_turns = max_turns_per_session.max(1);
        self.max_sessions.store(cap, Ordering::Relaxed);
        self.max_turns_per_session
            .store(max_turns, Ordering::Relaxed);
        if self.entries.len() > cap {
            evict_oldest_if_full(&self.entries, cap.saturating_add(1), |e| e.seq);
        }
        for mut item in self.entries.iter_mut() {
            while item.turns.len() > max_turns {
                item.turns.pop_front();
            }
        }
    }

    /// Number of currently stored sessions.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if zero shared sessions exist.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total handoff events recorded across the daemon lifetime (plus restored count).
    pub fn total_handoffs(&self) -> u64 {
        self.total_handoffs.load(Ordering::Relaxed)
    }

    /// Look up one unexpired session ledger entry.
    pub fn get(&self, session_id: &str, ttl: Duration) -> Option<SessionLedgerEntry> {
        let entry = self.entries.get(session_id)?;
        if entry.touched_elapsed() > ttl {
            drop(entry);
            self.entries.remove(session_id);
            None
        } else {
            Some(entry.clone())
        }
    }

    /// List summaries of all unexpired sessions ordered newest-first.
    pub fn list(&self, ttl: Duration) -> Vec<SessionSummary> {
        self.evict_expired(ttl);
        let mut out: Vec<SessionSummary> = self
            .entries
            .iter()
            .map(|e| SessionSummary::from(e.value()))
            .collect();
        out.sort_by(|a, b| {
            b.updated_at
                .cmp(&a.updated_at)
                .then_with(|| a.session_id.cmp(&b.session_id))
        });
        out
    }

    /// Delete one session entry, returning whether it existed.
    pub fn remove(&self, session_id: &str) -> bool {
        self.entries.remove(session_id).is_some()
    }

    fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::Relaxed)
    }

    fn new_entry(&self, session_id: &str, now_unix: u64) -> SessionLedgerEntry {
        SessionLedgerEntry {
            session_id: session_id.to_string(),
            created_at: now_unix,
            updated_at: now_unix,
            last_model: None,
            active_backend: None,
            active_profile: None,
            backend_history: Vec::new(),
            turns: VecDeque::new(),
            touched_files: Vec::new(),
            checkpoints: Vec::new(),
            total_prompt_tokens: 0,
            total_completion_tokens: 0,
            last_used: Some(Instant::now()),
            seq: self.next_seq(),
        }
    }

    /// Record completed request turns, redacted text, sanitized touched paths, and token totals.
    pub fn record_turn(&self, input: RecordTurnInput) {
        let cap = self.max_sessions.load(Ordering::Relaxed).max(1);
        let max_turns = self.max_turns_per_session.load(Ordering::Relaxed).max(1);
        let now_unix = unix_now_secs();
        let ws_root = input.workspace_root.as_deref();

        if !self.entries.contains_key(&input.session_id) {
            evict_oldest_if_full(&self.entries, cap, |e| e.seq);
        }

        let seq = self.next_seq();
        let mut entry = self
            .entries
            .entry(input.session_id.clone())
            .or_insert_with(|| self.new_entry(&input.session_id, now_unix));

        entry.updated_at = now_unix;
        entry.last_used = Some(Instant::now());
        entry.seq = seq;
        entry.last_model = Some(input.model.clone());
        entry.active_backend = Some(input.backend_id.clone());
        entry.active_profile = input.profile_id.clone();
        entry.total_prompt_tokens = entry
            .total_prompt_tokens
            .saturating_add(input.prompt_tokens);
        entry.total_completion_tokens = entry
            .total_completion_tokens
            .saturating_add(input.completion_tokens);

        let clean_user = truncate_clean(&telemetry::redact_text(&input.user_text));
        if !clean_user.is_empty() {
            entry.turns.push_back(CanonicalTurn {
                role: "user".into(),
                content: clean_user,
                tool_summary: None,
                backend_id: input.backend_id.clone(),
                timestamp: now_unix,
            });
        }

        let clean_assistant = truncate_clean(&telemetry::redact_text(&input.assistant_text));
        let clean_tools = input
            .tool_summary
            .as_deref()
            .map(telemetry::redact_text)
            .map(|s| truncate_clean(&s))
            .filter(|s| !s.is_empty());

        if !clean_assistant.is_empty() || clean_tools.is_some() {
            entry.turns.push_back(CanonicalTurn {
                role: "assistant".into(),
                content: clean_assistant,
                tool_summary: clean_tools,
                backend_id: input.backend_id.clone(),
                timestamp: now_unix,
            });
        }

        while entry.turns.len() > max_turns {
            entry.turns.pop_front();
        }

        for hint in &input.raw_path_hints {
            for rel in extract_sanitized_repo_paths(hint, ws_root) {
                push_unique_bounded(&mut entry.touched_files, rel, MAX_FILES_PER_SESSION);
            }
        }
        for source in [
            input.user_text.as_str(),
            input.assistant_text.as_str(),
            input.tool_summary.as_deref().unwrap_or(""),
        ] {
            for rel in extract_sanitized_repo_paths(source, ws_root) {
                push_unique_bounded(&mut entry.touched_files, rel, MAX_FILES_PER_SESSION);
            }
        }
    }

    /// Record a committed cross-agent handoff transition.
    pub fn record_handoff(
        &self,
        session_id: &str,
        from_backend: &str,
        to_backend: &str,
        reason: HandoffReason,
    ) {
        if from_backend == to_backend {
            return;
        }
        self.total_handoffs.fetch_add(1, Ordering::Relaxed);
        let now_unix = unix_now_secs();
        let cap = self.max_sessions.load(Ordering::Relaxed).max(1);
        if !self.entries.contains_key(session_id) {
            evict_oldest_if_full(&self.entries, cap, |e| e.seq);
        }
        let seq = self.next_seq();
        let mut entry = self
            .entries
            .entry(session_id.to_string())
            .or_insert_with(|| self.new_entry(session_id, now_unix));
        entry.updated_at = now_unix;
        entry.last_used = Some(Instant::now());
        entry.seq = seq;
        entry.active_backend = Some(to_backend.to_string());
        entry.backend_history.push(HandoffEvent {
            from_backend: from_backend.to_string(),
            to_backend: to_backend.to_string(),
            reason: reason.as_str().to_string(),
            timestamp: now_unix,
        });
        if entry.backend_history.len() > MAX_HANDOFF_HISTORY {
            let drain_count = entry.backend_history.len() - MAX_HANDOFF_HISTORY;
            entry.backend_history.drain(..drain_count);
        }
    }

    /// Append a redacted checkpoint note and optional sanitized touched files to `session_id`.
    pub fn add_checkpoint(
        &self,
        session_id: &str,
        note: &str,
        files: &[String],
        workspace_root: Option<&Path>,
    ) -> SessionLedgerEntry {
        let cap = self.max_sessions.load(Ordering::Relaxed).max(1);
        let now_unix = unix_now_secs();
        if !self.entries.contains_key(session_id) {
            evict_oldest_if_full(&self.entries, cap, |e| e.seq);
        }
        let seq = self.next_seq();
        let mut entry = self
            .entries
            .entry(session_id.to_string())
            .or_insert_with(|| self.new_entry(session_id, now_unix));
        entry.updated_at = now_unix;
        entry.last_used = Some(Instant::now());
        entry.seq = seq;

        let clean_note = truncate_clean(&telemetry::redact_text(note));
        if !clean_note.is_empty() {
            entry.checkpoints.push(clean_note);
            if entry.checkpoints.len() > MAX_CHECKPOINTS_PER_SESSION {
                let drain_count = entry.checkpoints.len() - MAX_CHECKPOINTS_PER_SESSION;
                entry.checkpoints.drain(..drain_count);
            }
        }
        for file in files {
            if let Some(rel) = sanitize_relative_repo_path(file, workspace_root) {
                push_unique_bounded(&mut entry.touched_files, rel, MAX_FILES_PER_SESSION);
            }
        }
        for rel in extract_sanitized_repo_paths(note, workspace_root) {
            push_unique_bounded(&mut entry.touched_files, rel, MAX_FILES_PER_SESSION);
        }
        entry.clone()
    }

    /// Remove and return expired session entries for optional final checkpoint sync.
    pub fn evict_expired(&self, ttl: Duration) -> Vec<SessionLedgerEntry> {
        let keys: Vec<String> = self
            .entries
            .iter()
            .filter(|e| e.value().touched_elapsed() > ttl)
            .map(|e| e.key().clone())
            .collect();
        let mut evicted = Vec::with_capacity(keys.len());
        for key in keys {
            if let Some((_, entry)) = self.entries.remove(&key) {
                evicted.push(entry);
            }
        }
        evicted
    }

    /// Snapshot live session ledgers for `gateway_state.json` persistence.
    pub fn export_snapshot(&self, ttl: Duration) -> (Vec<PersistedSharedSession>, u64) {
        self.evict_expired(ttl);
        let sessions = self
            .entries
            .iter()
            .map(|e| {
                let v = e.value();
                PersistedSharedSession {
                    session_id: v.session_id.clone(),
                    created_at: v.created_at,
                    updated_at: v.updated_at,
                    last_model: v.last_model.clone(),
                    active_backend: v.active_backend.clone(),
                    active_profile: v.active_profile.clone(),
                    backend_history: v
                        .backend_history
                        .iter()
                        .map(|h| PersistedHandoffEvent {
                            from_backend: h.from_backend.clone(),
                            to_backend: h.to_backend.clone(),
                            reason: h.reason.clone(),
                            timestamp: h.timestamp,
                        })
                        .collect(),
                    turns: v
                        .turns
                        .iter()
                        .map(|t| PersistedCanonicalTurn {
                            role: t.role.clone(),
                            content: t.content.clone(),
                            tool_summary: t.tool_summary.clone(),
                            backend_id: t.backend_id.clone(),
                            timestamp: t.timestamp,
                        })
                        .collect(),
                    touched_files: v.touched_files.clone(),
                    checkpoints: v.checkpoints.clone(),
                    total_prompt_tokens: v.total_prompt_tokens,
                    total_completion_tokens: v.total_completion_tokens,
                }
            })
            .collect();
        (sessions, self.total_handoffs())
    }

    /// Restore non-expired persisted shared sessions on daemon boot.
    pub fn restore_snapshot(
        &self,
        items: &[PersistedSharedSession],
        total_handoffs: u64,
        ttl: Duration,
    ) {
        let now_unix = unix_now_secs();
        let cap = self.max_sessions.load(Ordering::Relaxed).max(1);
        let max_turns = self.max_turns_per_session.load(Ordering::Relaxed).max(1);
        self.total_handoffs.store(total_handoffs, Ordering::Relaxed);

        for item in items {
            let Some(last_used) = instant_from_unix(item.updated_at, now_unix, ttl) else {
                continue;
            };
            if !self.entries.contains_key(&item.session_id) {
                evict_oldest_if_full(&self.entries, cap, |e| e.seq);
            }
            let seq = self.next_seq();
            let start_turn = item.turns.len().saturating_sub(max_turns);
            let turns: VecDeque<CanonicalTurn> = item.turns[start_turn..]
                .iter()
                .map(|t| CanonicalTurn {
                    role: t.role.clone(),
                    content: telemetry::redact_text(&t.content),
                    tool_summary: t.tool_summary.as_deref().map(telemetry::redact_text),
                    backend_id: t.backend_id.clone(),
                    timestamp: t.timestamp,
                })
                .collect();
            let entry = SessionLedgerEntry {
                session_id: item.session_id.clone(),
                created_at: item.created_at,
                updated_at: item.updated_at,
                last_model: item.last_model.clone(),
                active_backend: item.active_backend.clone(),
                active_profile: item.active_profile.clone(),
                backend_history: item
                    .backend_history
                    .iter()
                    .map(|h| HandoffEvent {
                        from_backend: h.from_backend.clone(),
                        to_backend: h.to_backend.clone(),
                        reason: h.reason.clone(),
                        timestamp: h.timestamp,
                    })
                    .collect(),
                turns,
                touched_files: item
                    .touched_files
                    .iter()
                    .filter_map(|p| sanitize_relative_repo_path(p, None))
                    .collect(),
                checkpoints: item
                    .checkpoints
                    .iter()
                    .map(|c| telemetry::redact_text(c))
                    .collect(),
                total_prompt_tokens: item.total_prompt_tokens,
                total_completion_tokens: item.total_completion_tokens,
                last_used: Some(last_used),
                seq,
            };
            self.entries.insert(item.session_id.clone(), entry);
        }
    }

    /// Spawn a background TTL sweep task, passing evicted ledgers to `on_evict`.
    pub fn spawn_eviction_task<F>(
        self: &Arc<Self>,
        ttl_secs: Arc<AtomicU64>,
        shutdown: CancellationToken,
        on_evict: F,
    ) -> tokio::task::JoinHandle<()>
    where
        F: Fn(Vec<SessionLedgerEntry>) + Send + Sync + 'static,
    {
        let store = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                tokio::select! {
                    _ = shutdown.cancelled() => break,
                    _ = interval.tick() => {
                        let ttl = Duration::from_secs(ttl_secs.load(Ordering::Relaxed).max(1));
                        let evicted = store.evict_expired(ttl);
                        if !evicted.is_empty() {
                            on_evict(evicted);
                        }
                    }
                }
            }
        })
    }
}

fn push_unique_bounded(vec: &mut Vec<String>, item: String, max_len: usize) {
    if item.is_empty() || vec.iter().any(|existing| existing == &item) {
        return;
    }
    if vec.len() >= max_len {
        let excess = (vec.len() + 1).saturating_sub(max_len);
        vec.drain(..excess);
    }
    vec.push(item);
}

fn truncate_clean(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= MAX_TURN_TEXT_CHARS {
        return trimmed.to_string();
    }
    let truncated: String = trimmed.chars().take(MAX_TURN_TEXT_CHARS).collect();
    format!("{truncated}...")
}

/// Format a concise `session` summary line suitable for `trace`'s `record_session`.
pub fn format_trace_session_summary(entry: &SessionLedgerEntry) -> String {
    let agent = entry.active_backend.as_deref().unwrap_or("unknown");
    let latest_goal = entry
        .turns
        .iter()
        .rev()
        .find(|t| t.role == "user")
        .map(|t| t.content.as_str())
        .unwrap_or("");
    let latest_reply = entry
        .turns
        .iter()
        .rev()
        .find(|t| t.role == "assistant")
        .map(|t| t.content.as_str())
        .unwrap_or("");
    let mut parts = vec![format!(
        "vouch session {} (agent={}, turns={}, handoffs={})",
        entry.session_id,
        agent,
        entry.turns.len(),
        entry.backend_history.len()
    )];
    if !entry.checkpoints.is_empty() {
        parts.push(format!("checkpoints: {}", entry.checkpoints.join(" | ")));
    }
    if !latest_goal.is_empty() {
        parts.push(format!("goal: {}", truncate_short(latest_goal, 240)));
    }
    if !latest_reply.is_empty() {
        parts.push(format!("outcome: {}", truncate_short(latest_reply, 240)));
    }
    telemetry::redact_text(&parts.join("; "))
}

fn truncate_short(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let prefix: String = text.chars().take(max_chars).collect();
    format!("{prefix}...")
}

/// Sanitize a candidate path into a safe relative repository path without `..` traversal.
pub fn sanitize_relative_repo_path(raw: &str, workspace_root: Option<&Path>) -> Option<String> {
    let mut cleaned = raw.trim().trim_matches(|c: char| {
        matches!(
            c,
            '"' | '\'' | '`' | ',' | ';' | '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>'
        )
    });
    if cleaned.is_empty() || cleaned.len() > 256 || cleaned.contains(['\0', '\\']) {
        return None;
    }
    if cleaned.contains("://") || cleaned.starts_with("data:") || cleaned.starts_with("mailto:") {
        return None;
    }
    // Strip trailing `:line` or `:line:col` suffixes often seen in compiler output.
    if let Some((head, tail)) = cleaned.rsplit_once(':')
        && !head.is_empty()
        && tail.chars().all(|c| c.is_ascii_digit())
    {
        cleaned = head;
        if let Some((head2, tail2)) = cleaned.rsplit_once(':')
            && !head2.is_empty()
            && tail2.chars().all(|c| c.is_ascii_digit())
        {
            cleaned = head2;
        }
    }
    cleaned = cleaned.trim_end_matches('.');
    if cleaned.is_empty() || cleaned.starts_with('-') {
        return None;
    }

    let candidate_path = Path::new(cleaned);
    let relative_buf;
    let rel = if candidate_path.has_root() || candidate_path.is_absolute() {
        let root = workspace_root?;
        if let Ok(stripped) = candidate_path.strip_prefix(root) {
            relative_buf = stripped.to_path_buf();
            relative_buf.as_path()
        } else {
            let canon_root = std::fs::canonicalize(root).ok()?;
            let canon_cand = std::fs::canonicalize(candidate_path).ok()?;
            let stripped = canon_cand.strip_prefix(&canon_root).ok()?;
            relative_buf = stripped.to_path_buf();
            relative_buf.as_path()
        }
    } else {
        candidate_path
    };

    let mut normalized_parts = Vec::new();
    for component in rel.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::Normal(seg) => {
                let s = seg.to_str()?;
                if s.is_empty() || s.contains('\0') {
                    return None;
                }
                normalized_parts.push(s.to_string());
            }
            std::path::Component::ParentDir
            | std::path::Component::RootDir
            | std::path::Component::Prefix(_) => return None,
        }
    }
    if normalized_parts.is_empty() {
        return None;
    }

    let joined = normalized_parts.join("/");
    let looks_like_file = joined.contains('/') || has_known_repo_filename_shape(&joined);
    if !looks_like_file {
        return None;
    }
    Some(joined)
}

fn has_known_repo_filename_shape(name: &str) -> bool {
    if matches!(
        name,
        "Cargo.toml"
            | "Cargo.lock"
            | "Makefile"
            | "Dockerfile"
            | "README.md"
            | "CHANGELOG.md"
            | "AGENTS.md"
            | "package.json"
            | "tsconfig.json"
            | "go.mod"
    ) {
        return true;
    }
    let Some((_, ext)) = name.rsplit_once('.') else {
        return false;
    };
    let lower = ext.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "rs" | "toml"
            | "ts"
            | "tsx"
            | "js"
            | "jsx"
            | "mjs"
            | "cjs"
            | "py"
            | "go"
            | "java"
            | "kt"
            | "c"
            | "cc"
            | "cpp"
            | "h"
            | "hpp"
            | "rb"
            | "php"
            | "swift"
            | "md"
            | "json"
            | "yaml"
            | "yml"
            | "sql"
            | "sh"
            | "html"
            | "css"
            | "scss"
            | "svg"
            | "proto"
    )
}

/// Deterministically extract deduplicated, traversal-safe repository file paths from text or JSON tool arguments.
pub fn extract_sanitized_repo_paths(text: &str, workspace_root: Option<&Path>) -> Vec<String> {
    let mut found = Vec::new();
    // Parse structured JSON if the entire payload happens to be JSON (e.g., tool arguments).
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(text.trim()) {
        collect_paths_from_json(&val, workspace_root, &mut found);
    }
    for token in text.split(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                '"' | '\'' | '`' | ',' | ';' | '(' | ')' | '[' | ']' | '{' | '}' | '=' | '<' | '>'
            )
    }) {
        if token.contains("..") {
            continue;
        }
        if let Some(rel) = sanitize_relative_repo_path(token, workspace_root) {
            push_unique_bounded(&mut found, rel, MAX_FILES_PER_SESSION);
        }
    }
    found
}

fn collect_paths_from_json(
    val: &serde_json::Value,
    workspace_root: Option<&Path>,
    out: &mut Vec<String>,
) {
    match val {
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                if matches!(
                    k.as_str(),
                    "path" | "file" | "file_path" | "target_path" | "target_file" | "files"
                ) {
                    if let Some(s) = v.as_str()
                        && !s.contains("..")
                        && let Some(rel) = sanitize_relative_repo_path(s, workspace_root)
                    {
                        push_unique_bounded(out, rel, MAX_FILES_PER_SESSION);
                    } else if let Some(arr) = v.as_array() {
                        for item in arr {
                            if let Some(s) = item.as_str()
                                && !s.contains("..")
                                && let Some(rel) = sanitize_relative_repo_path(s, workspace_root)
                            {
                                push_unique_bounded(out, rel, MAX_FILES_PER_SESSION);
                            }
                        }
                    }
                } else {
                    collect_paths_from_json(v, workspace_root, out);
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                collect_paths_from_json(item, workspace_root, out);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizes_paths_and_blocks_traversal() {
        let root = Path::new("/repo");
        assert_eq!(
            sanitize_relative_repo_path("./src/auth/tokens.rs:42", Some(root)),
            Some("src/auth/tokens.rs".to_string())
        );
        assert_eq!(
            sanitize_relative_repo_path("/repo/src/lib.rs", Some(root)),
            Some("src/lib.rs".to_string())
        );
        assert_eq!(
            sanitize_relative_repo_path("../etc/passwd", Some(root)),
            None
        );
        assert_eq!(
            sanitize_relative_repo_path("src/../../etc/passwd", Some(root)),
            None
        );
        assert_eq!(sanitize_relative_repo_path("/etc/passwd", Some(root)), None);
        assert_eq!(
            sanitize_relative_repo_path("https://example.com/a/b.rs", Some(root)),
            None
        );
    }

    #[test]
    fn redacts_secrets_and_caps_turns() {
        let store = SessionLedgerStore::new(10, 2);
        store.record_turn(RecordTurnInput {
            session_id: "s1".into(),
            model: "coding/default".into(),
            backend_id: "alpha".into(),
            profile_id: None,
            user_text: "Inspect src/auth.rs with Bearer supersecret12345".into(),
            assistant_text: "Done sk-secretkey9999999999".into(),
            tool_summary: Some(r#"{"path":"src/config.rs","token":"secretval123"}"#.into()),
            raw_path_hints: vec![r#"{"file_path":"src/main.rs"}"#.into()],
            workspace_root: None,
            prompt_tokens: 10,
            completion_tokens: 5,
        });
        let entry = store.get("s1", Duration::from_secs(60)).expect("entry");
        assert_eq!(entry.turns.len(), 2);
        assert!(!entry.turns[0].content.contains("supersecret12345"));
        assert!(!entry.turns[1].content.contains("sk-secretkey9999999999"));
        assert!(entry.touched_files.contains(&"src/auth.rs".to_string()));
        assert!(entry.touched_files.contains(&"src/config.rs".to_string()));
        assert!(entry.touched_files.contains(&"src/main.rs".to_string()));

        store.record_turn(RecordTurnInput {
            session_id: "s1".into(),
            model: "coding/default".into(),
            backend_id: "beta".into(),
            profile_id: None,
            user_text: "turn 2".into(),
            assistant_text: "reply 2".into(),
            tool_summary: None,
            raw_path_hints: Vec::new(),
            workspace_root: None,
            prompt_tokens: 3,
            completion_tokens: 4,
        });
        let entry2 = store.get("s1", Duration::from_secs(60)).expect("entry2");
        assert_eq!(entry2.turns.len(), 2);
        assert_eq!(entry2.turns[0].content, "turn 2");
        assert_eq!(entry2.turns[1].content, "reply 2");
    }
}
