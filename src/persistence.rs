//! SPDX-License-Identifier: GPL-3.0-or-later
//! Atomic `0600` JSON persistence helpers for `defaults.state_dir`.

use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    hash::Hash,
    io::Write,
    path::Path,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    auth::tokens::DailyTokenCounter,
    driver::ChatMessage,
    error::{GatewayError, Result},
    usage::PersistedUsage,
};

/// Single consolidated snapshot file stored inside `defaults.state_dir`.
pub const STATE_FILE_NAME: &str = "gateway_state.json";

const SNAPSHOT_VERSION: u32 = 1;

/// Persisted sticky session mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedAffinityEntry {
    /// Session affinity key.
    pub key: String,
    /// Last healthy backend selected for this session.
    pub backend_id: String,
    /// Last access timestamp in seconds since UNIX epoch.
    pub updated_unix_secs: u64,
}

/// Persisted native backend CLI session state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedBackendSession {
    /// Session affinity key.
    pub session_key: String,
    /// Backend identifier.
    pub backend_id: String,
    /// Optional backend-native session handle.
    pub backend_session_id: Option<String>,
    /// True when turn 1 already committed history on this backend.
    pub has_history: bool,
    /// Last access timestamp in seconds since UNIX epoch.
    pub updated_unix_secs: u64,
}

/// Persisted `/v1/responses` chaining entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedResponse {
    /// Response identifier (`resp_...`).
    pub id: String,
    /// Active system prompt at completion.
    pub system: Option<String>,
    /// Full message history including assistant output and tool calls.
    pub messages: Vec<ChatMessage>,
    /// Storage timestamp in seconds since UNIX epoch.
    pub stored_at_unix_secs: u64,
}

/// Durable snapshot persisted to `defaults.state_dir/gateway_state.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GatewaySnapshot {
    /// Schema format version.
    pub version: u32,
    /// Snapshot creation time in seconds since UNIX epoch.
    pub saved_at_unix_secs: u64,
    /// Sticky session mappings.
    #[serde(default)]
    pub session_affinity: Vec<PersistedAffinityEntry>,
    /// Backend native continuation handles.
    #[serde(default)]
    pub backend_sessions: Vec<PersistedBackendSession>,
    /// Stateful `/v1/responses` entries.
    #[serde(default)]
    pub responses: Vec<PersistedResponse>,
    /// Per-token UTC-day BPE consumption keyed by SHA-256 hex digest (zero secrets).
    #[serde(default)]
    pub daily_token_budgets: BTreeMap<String, DailyTokenCounter>,
    /// Cumulative per-token and per-backend usage and latency analytics.
    #[serde(default)]
    pub usage: PersistedUsage,
}

impl GatewaySnapshot {
    /// Construct a snapshot tagged with the current schema version and timestamp.
    pub fn new(
        session_affinity: Vec<PersistedAffinityEntry>,
        backend_sessions: Vec<PersistedBackendSession>,
        responses: Vec<PersistedResponse>,
        daily_token_budgets: BTreeMap<String, DailyTokenCounter>,
        usage: PersistedUsage,
    ) -> Self {
        Self {
            version: SNAPSHOT_VERSION,
            saved_at_unix_secs: unix_now_secs(),
            session_affinity,
            backend_sessions,
            responses,
            daily_token_budgets,
            usage,
        }
    }
}

/// Reconstruct a monotonic [`Instant`] from a persisted UNIX timestamp,
/// returning `None` when the entry has already exceeded `ttl`.
pub fn instant_from_unix(
    updated_unix_secs: u64,
    now_unix_secs: u64,
    ttl: Duration,
) -> Option<Instant> {
    let elapsed = Duration::from_secs(now_unix_secs.saturating_sub(updated_unix_secs));
    if elapsed > ttl {
        return None;
    }
    Some(
        Instant::now()
            .checked_sub(elapsed)
            .unwrap_or_else(Instant::now),
    )
}

/// Evict the entries with the smallest sequence number from `map` until its length
/// is strictly below `max_entries` so an incoming new key stays within the cap.
pub fn evict_oldest_if_full<K, V, F>(map: &DashMap<K, V>, max_entries: usize, mut seq_fn: F)
where
    K: Eq + Hash + Clone,
    F: FnMut(&V) -> u64,
{
    let cap = max_entries.max(1);
    while map.len() >= cap {
        let oldest = map
            .iter()
            .min_by_key(|entry| seq_fn(entry.value()))
            .map(|entry| entry.key().clone());
        match oldest {
            Some(key) => {
                map.remove(&key);
            }
            None => break,
        }
    }
}

/// Current UNIX timestamp in whole seconds.
pub fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Atomically write `bytes` to `path` with `0600` permissions on Unix.
pub fn write_atomic_0600(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|error| {
            GatewayError::Io(std::io::Error::new(
                error.kind(),
                format!("cannot create directory {}: {error}", parent.display()),
            ))
        })?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700));
        }
    }

    let tmp_name = format!(
        ".{}.tmp.{}",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("state"),
        Uuid::new_v4().simple()
    );
    let tmp_path = path.with_file_name(tmp_name);
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }

    let mut file = options.open(&tmp_path).map_err(|error| {
        GatewayError::Io(std::io::Error::new(
            error.kind(),
            format!(
                "cannot create temp state file {}: {error}",
                tmp_path.display()
            ),
        ))
    })?;
    if let Err(error) = file.write_all(bytes).and_then(|()| file.sync_all()) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(GatewayError::Io(std::io::Error::new(
            error.kind(),
            format!("cannot write state file {}: {error}", tmp_path.display()),
        )));
    }
    drop(file);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&tmp_path, std::fs::Permissions::from_mode(0o600));
    }

    if let Err(error) = std::fs::rename(&tmp_path, path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(GatewayError::Io(std::io::Error::new(
            error.kind(),
            format!(
                "cannot atomically replace state file {}: {error}",
                path.display()
            ),
        )));
    }
    Ok(())
}

/// Load a persisted snapshot from `state_dir`, returning `None` cleanly when absent or unreadable.
pub fn load_snapshot(state_dir: &Path) -> Option<GatewaySnapshot> {
    let path = state_dir.join(STATE_FILE_NAME);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => {
            tracing::warn!(path = %path.display(), error = %error, "cannot read persisted state; starting fresh");
            return None;
        }
    };
    match serde_json::from_slice::<GatewaySnapshot>(&bytes) {
        Ok(snapshot) if snapshot.version == SNAPSHOT_VERSION => Some(snapshot),
        Ok(snapshot) => {
            tracing::warn!(
                path = %path.display(),
                version = snapshot.version,
                "unsupported persisted state version; starting fresh"
            );
            None
        }
        Err(error) => {
            tracing::warn!(path = %path.display(), error = %error, "corrupt persisted state; starting fresh");
            None
        }
    }
}

/// Persist `snapshot` atomically to `state_dir/gateway_state.json` (`0600`).
pub fn save_snapshot(state_dir: &Path, snapshot: &GatewaySnapshot) -> Result<()> {
    let path = state_dir.join(STATE_FILE_NAME);
    let payload = serde_json::to_vec_pretty(snapshot)
        .map_err(|error| GatewayError::Internal(format!("cannot serialize state: {error}")))?;
    write_atomic_0600(&path, &payload)
}
