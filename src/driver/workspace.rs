//! Ephemeral git worktree isolation.
//!
//! A backend may ask for an isolated working tree per session. Worktrees are
//! cached by session key, reused while active, and removed by a background TTL
//! eviction pass. Cleanup is best-effort and never panics: if `git worktree
//! remove` fails the directory is deleted directly.

use std::{
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use arc_swap::ArcSwap;
use dashmap::DashMap;
use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    config::WorkspaceConfig,
    error::{GatewayError, Result},
    telemetry,
};

/// One live worktree.
#[derive(Debug)]
pub struct WorkspaceEntry {
    /// Absolute path of the worktree.
    pub path: PathBuf,
    /// Creation time.
    pub created_at: Instant,
    repo: PathBuf,
    git: String,
    last_used: Mutex<Instant>,
    cleaned: AtomicBool,
}

impl WorkspaceEntry {
    /// Update the idle timestamp.
    pub fn touch(&self) {
        *self.last_used.lock() = Instant::now();
    }

    /// Idle duration.
    pub fn idle_for(&self) -> Duration {
        self.last_used.lock().elapsed()
    }

    /// Remove the worktree. Safe to call more than once.
    pub async fn cleanup(&self) {
        if self.cleaned.swap(true, Ordering::SeqCst) {
            return;
        }
        let mut command = tokio::process::Command::new(&self.git);
        command
            .arg("-C")
            .arg(&self.repo)
            .arg("worktree")
            .arg("remove")
            .arg("--force")
            .arg(&self.path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        let removed = matches!(
            tokio::time::timeout(Duration::from_secs(20), command.status()).await,
            Ok(Ok(status)) if status.success()
        );
        if !removed {
            let _ = tokio::fs::remove_dir_all(&self.path).await;
        }
        let _ = tokio::fs::remove_file(pid_sidecar_path(&self.path)).await;
        let _ = tokio::process::Command::new(&self.git)
            .arg("-C")
            .arg(&self.repo)
            .arg("worktree")
            .arg("prune")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await;
    }
}

impl Drop for WorkspaceEntry {
    fn drop(&mut self) {
        if self.cleaned.swap(true, Ordering::SeqCst) {
            return;
        }
        let path = self.path.clone();
        let repo = self.repo.clone();
        let git = self.git.clone();
        // Drop cannot await. A short-lived thread performs the cleanup; the
        // manager's explicit shutdown path awaits the same operation.
        let _ = std::thread::Builder::new()
            .name("vouch-worktree-cleanup".into())
            .spawn(move || {
                let status = std::process::Command::new(&git)
                    .arg("-C")
                    .arg(&repo)
                    .arg("worktree")
                    .arg("remove")
                    .arg("--force")
                    .arg(&path)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
                if !matches!(status, Ok(status) if status.success()) {
                    let _ = std::fs::remove_dir_all(&path);
                }
                let _ = std::fs::remove_file(pid_sidecar_path(&path));
                let _ = std::process::Command::new(&git)
                    .arg("-C")
                    .arg(&repo)
                    .arg("worktree")
                    .arg("prune")
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
            });
    }
}

/// Owns the active worktree cache.
#[derive(Debug)]
pub struct WorkspaceManager {
    config: ArcSwap<WorkspaceConfig>,
    entries: DashMap<String, Arc<WorkspaceEntry>>,
}

impl WorkspaceManager {
    /// Build a manager from configuration.
    pub fn new(config: WorkspaceConfig) -> Self {
        Self {
            config: ArcSwap::from_pointee(config),
            entries: DashMap::new(),
        }
    }

    /// Replace configuration after a hot reload.
    pub fn update_config(&self, config: WorkspaceConfig) {
        self.config.store(Arc::new(config));
    }

    /// Current configuration snapshot.
    pub fn config(&self) -> Arc<WorkspaceConfig> {
        self.config.load_full()
    }

    /// Whether isolation is enabled.
    pub fn enabled(&self) -> bool {
        self.config.load().enabled
    }

    /// Get or create a worktree for a session.
    pub async fn acquire(&self, session_key: &str) -> Result<Arc<WorkspaceEntry>> {
        let config = self.config.load_full();
        if !config.enabled {
            return Err(GatewayError::BadRequest(
                "workspace isolation is not enabled".into(),
            ));
        }
        let repo = config.repo_path.clone().ok_or_else(|| {
            GatewayError::Config("workspace.repo_path is required when isolation is enabled".into())
        })?;
        let ttl = Duration::from_secs(config.ttl_secs.max(1));

        if let Some(existing) = self.entries.get(session_key) {
            if existing.idle_for() < ttl {
                existing.touch();
                return Ok(existing.clone());
            }
            let key = session_key.to_string();
            drop(existing);
            self.entries.remove(&key);
        }

        let root = config
            .root
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("vouch-worktrees"));
        tokio::fs::create_dir_all(&root).await.map_err(|error| {
            GatewayError::Internal(format!(
                "cannot create worktree root {}: {error}",
                root.display()
            ))
        })?;

        let safe = sanitize_session_key(session_key);
        let suffix = Uuid::new_v4().simple().to_string();
        let path = root.join(format!("{safe}-{}", &suffix[..8.min(suffix.len())]));

        let mut command = tokio::process::Command::new(&config.git_binary);
        command
            .arg("-C")
            .arg(&repo)
            .arg("worktree")
            .arg("add")
            .arg("--detach")
            .arg(&path)
            .arg("HEAD")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let output = tokio::time::timeout(Duration::from_secs(30), command.output())
            .await
            .map_err(|_| GatewayError::Internal("git worktree creation timed out".into()))?
            .map_err(|error| GatewayError::Internal(format!("cannot execute git: {error}")))?;

        if !output.status.success() {
            let stderr = telemetry::redact_text(&String::from_utf8_lossy(&output.stderr));
            let _ = tokio::fs::remove_dir_all(&path).await;
            return Err(GatewayError::Internal(format!(
                "git worktree add failed in {}: {}",
                repo.display(),
                crate::driver::truncate(stderr.trim(), 600)
            )));
        }

        let _ = tokio::fs::write(pid_sidecar_path(&path), std::process::id().to_string()).await;

        let entry = Arc::new(WorkspaceEntry {
            path,
            created_at: Instant::now(),
            repo,
            git: config.git_binary.clone(),
            last_used: Mutex::new(Instant::now()),
            cleaned: AtomicBool::new(false),
        });

        // Resolve a race where two requests created a worktree for the same
        // session concurrently: keep the first inserted entry and clean ours.
        match self.entries.entry(session_key.to_string()) {
            dashmap::mapref::entry::Entry::Occupied(occupied) => {
                let existing = occupied.get().clone();
                drop(occupied);
                entry.cleanup().await;
                existing.touch();
                Ok(existing)
            }
            dashmap::mapref::entry::Entry::Vacant(vacant) => {
                vacant.insert(entry.clone());
                Ok(entry)
            }
        }
    }

    /// Remove worktrees idle for longer than `ttl`.
    pub fn evict_expired(&self, ttl: Duration) {
        let expired: Vec<String> = self
            .entries
            .iter()
            .filter(|entry| entry.idle_for() > ttl)
            .map(|entry| entry.key().clone())
            .collect();
        for key in expired {
            if let Some((_, entry)) = self.entries.remove(&key) {
                drop(entry);
            }
        }
    }

    /// Number of live worktrees.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no worktrees are active.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove every worktree, awaiting git cleanup.
    pub async fn cleanup_all(&self) {
        let entries: Vec<(String, Arc<WorkspaceEntry>)> = self
            .entries
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();
        for (key, _) in &entries {
            self.entries.remove(key);
        }
        for (_, entry) in entries {
            entry.cleanup().await;
        }
    }

    /// Sweep orphaned worktree directories left in `workspace.root` from crashed
    /// daemon processes or older than `max_age`, pruning git worktree metadata.
    pub async fn sweep_orphans(&self, max_age: Duration) -> usize {
        let config = self.config.load_full();
        if !config.enabled {
            return 0;
        }
        let root = config
            .root
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("vouch-worktrees"));
        let Ok(mut entries) = tokio::fs::read_dir(&root).await else {
            return 0;
        };
        let active_paths: std::collections::HashSet<PathBuf> = self
            .entries
            .iter()
            .map(|entry| entry.value().path.clone())
            .collect();
        let current_pid = std::process::id();
        let mut removed = 0usize;

        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".vouch.pid"))
            {
                let base_str = path.to_string_lossy();
                let target_dir = PathBuf::from(base_str.trim_end_matches(".vouch.pid"));
                if !target_dir.exists() {
                    let _ = tokio::fs::remove_file(&path).await;
                }
                continue;
            }
            let Ok(ft) = entry.file_type().await else {
                continue;
            };
            if !ft.is_dir() || active_paths.contains(&path) {
                continue;
            }
            let sidecar = pid_sidecar_path(&path);
            let sidecar_pid = tokio::fs::read_to_string(&sidecar)
                .await
                .ok()
                .and_then(|raw| raw.trim().parse::<u32>().ok());
            let age_exceeded = tokio::fs::metadata(&path)
                .await
                .and_then(|meta| meta.modified())
                .ok()
                .and_then(|mtime| mtime.elapsed().ok())
                .is_some_and(|age| age > max_age);

            let is_orphan = match sidecar_pid {
                Some(pid) if pid == current_pid => false,
                Some(pid) => !crate::driver::attachments::is_process_alive(pid) || age_exceeded,
                None => age_exceeded,
            };
            if !is_orphan {
                continue;
            }

            let removed_via_git = if let Some(repo) = &config.repo_path {
                let mut cmd = tokio::process::Command::new(&config.git_binary);
                cmd.arg("-C")
                    .arg(repo)
                    .arg("worktree")
                    .arg("remove")
                    .arg("--force")
                    .arg(&path)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .kill_on_drop(true);
                matches!(
                    tokio::time::timeout(Duration::from_secs(20), cmd.status()).await,
                    Ok(Ok(status)) if status.success()
                )
            } else {
                false
            };
            if !removed_via_git && tokio::fs::remove_dir_all(&path).await.is_err() {
                continue;
            }
            let _ = tokio::fs::remove_file(&sidecar).await;
            removed += 1;
        }

        if removed > 0
            && let Some(repo) = &config.repo_path
        {
            let _ = tokio::process::Command::new(&config.git_binary)
                .arg("-C")
                .arg(repo)
                .arg("worktree")
                .arg("prune")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .await;
        }
        removed
    }

    /// Spawn a periodic TTL eviction task.
    pub fn spawn_eviction_task(
        self: &Arc<Self>,
        shutdown: CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
        let manager = self.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(30));
            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        let ttl = Duration::from_secs(manager.config.load().ttl_secs.max(1));
                        manager.evict_expired(ttl);
                    }
                    _ = shutdown.cancelled() => break,
                }
            }
        })
    }
}

fn pid_sidecar_path(worktree_path: &Path) -> PathBuf {
    let mut raw = worktree_path.as_os_str().to_os_string();
    raw.push(".vouch.pid");
    PathBuf::from(raw)
}

fn sanitize_session_key(key: &str) -> String {
    let mut output: String = key
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .take(32)
        .collect();
    if output.is_empty() {
        output.push_str("session");
    }
    output
}

/// Validate that a repository can host worktrees.
pub fn validate_repo(git: &str, repo: &Path) -> std::result::Result<(), String> {
    let output = std::process::Command::new(git)
        .arg("-C")
        .arg(repo)
        .arg("rev-parse")
        .arg("--is-inside-work-tree")
        .stdin(Stdio::null())
        .output()
        .map_err(|error| format!("cannot execute {git}: {error}"))?;
    if output.status.success() && String::from_utf8_lossy(&output.stdout).trim() == "true" {
        Ok(())
    } else {
        Err(format!(
            "{} is not a git work tree: {}",
            repo.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn git_available() -> bool {
        std::process::Command::new("git")
            .arg("--version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn creates_and_removes_worktree() {
        if !git_available() {
            return;
        }
        let temp = tempfile::tempdir().expect("temp dir");
        let repo = temp.path().join("repo");
        std::fs::create_dir_all(&repo).expect("repo dir");
        let run = |args: &[&str]| {
            let status = std::process::Command::new("git")
                .arg("-C")
                .arg(&repo)
                .args(args)
                .env("GIT_AUTHOR_NAME", "vouch")
                .env("GIT_AUTHOR_EMAIL", "vouch@example.invalid")
                .env("GIT_COMMITTER_NAME", "vouch")
                .env("GIT_COMMITTER_EMAIL", "vouch@example.invalid")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .expect("git");
            assert!(status.success(), "git {args:?}");
        };
        run(&["init", "-q"]);
        std::fs::write(repo.join("README.md"), "hello").expect("write");
        run(&["add", "."]);
        run(&["commit", "-q", "-m", "init"]);

        let manager = WorkspaceManager::new(WorkspaceConfig {
            enabled: true,
            repo_path: Some(repo.clone()),
            root: Some(temp.path().join("worktrees")),
            ttl_secs: 60,
            git_binary: "git".into(),
            cleanup_on_shutdown: true,
        });
        let entry = manager.acquire("session-1").await.expect("worktree");
        assert!(entry.path.exists());
        assert!(pid_sidecar_path(&entry.path).exists());

        // Create a synthetic crashed-daemon orphan worktree directory with a dead PID.
        let orphan_dir = temp.path().join("worktrees").join("orphan-deadbeef");
        std::fs::create_dir_all(&orphan_dir).expect("orphan dir");
        std::fs::write(pid_sidecar_path(&orphan_dir), "99999999").expect("orphan pid");
        let swept = manager.sweep_orphans(Duration::from_secs(3600)).await;
        assert_eq!(swept, 1);
        assert!(!orphan_dir.exists());
        assert!(!pid_sidecar_path(&orphan_dir).exists());

        let path = entry.path.clone();
        manager.cleanup_all().await;
        assert!(!path.exists());
        assert!(!pid_sidecar_path(&path).exists());
    }
}
