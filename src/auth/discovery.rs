//! Local CLI and credential discovery used by `vouch doctor`.
//!
//! Discovery is intentionally read-only. It resolves configured binaries on
//! `PATH`, reports on well-known auth files for popular agent CLIs, and
//! TCP-probes local daemon endpoints.

use std::{
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};

use serde::Serialize;
use tokio::net::TcpStream;

use crate::{config::GatewayConfig, error::Result};

/// Binary resolution result.
#[derive(Debug, Clone, Serialize)]
pub struct BinaryStatus {
    /// Configured command.
    pub name: String,
    /// Whether the executable was found.
    pub found: bool,
    /// Resolved absolute path.
    pub path: Option<PathBuf>,
}

/// Credential-file freshness.
#[derive(Debug, Clone, Serialize)]
pub struct AuthFileStatus {
    /// File or directory path.
    pub path: PathBuf,
    /// Whether it exists.
    pub exists: bool,
    /// Age in seconds since last modification.
    pub age_secs: Option<u64>,
    /// Size in bytes for regular files.
    pub size_bytes: Option<u64>,
}

/// Local daemon reachability.
#[derive(Debug, Clone, Serialize)]
pub struct DaemonStatus {
    /// Backend id.
    pub backend: String,
    /// Probed URL.
    pub url: String,
    /// Whether the TCP connection succeeded.
    pub reachable: bool,
    /// Failure detail.
    pub detail: Option<String>,
}

/// Combined discovery report.
#[derive(Debug, Clone, Serialize)]
pub struct DiscoveryReport {
    /// Configured subprocess binaries.
    pub binaries: Vec<BinaryStatus>,
    /// Known and extra credential files.
    pub auth_files: Vec<AuthFileStatus>,
    /// Configured HTTP daemons.
    pub daemons: Vec<DaemonStatus>,
}

/// Run discovery against the active configuration.
pub async fn discover(config: &GatewayConfig) -> Result<DiscoveryReport> {
    let mut binaries = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for backend in &config.backends {
        if let Some(command) = &backend.command
            && seen.insert(command.clone())
        {
            let path = resolve_binary(command);
            binaries.push(BinaryStatus {
                name: command.clone(),
                found: path.is_some(),
                path,
            });
        }
    }

    let mut auth_files = Vec::new();
    if config.discovery.enabled {
        let mut paths = known_auth_paths();
        paths.extend(config.discovery.extra_auth_paths.iter().cloned());
        for path in paths {
            auth_files.push(inspect_auth_path(&path));
        }
    }

    let mut daemons = Vec::new();
    if config.discovery.probe_ports {
        for backend in &config.backends {
            let Some(base) = backend.base_url.as_deref() else {
                continue;
            };
            let (reachable, detail) = probe_url(base).await;
            daemons.push(DaemonStatus {
                backend: backend.id.clone(),
                url: base.to_string(),
                reachable,
                detail,
            });
        }
    }

    Ok(DiscoveryReport {
        binaries,
        auth_files,
        daemons,
    })
}

/// Resolve a command on `PATH`, or verify an explicit path.
pub fn resolve_binary(command: &str) -> Option<PathBuf> {
    let command = command.trim();
    if command.is_empty() {
        return None;
    }
    if command.contains(std::path::MAIN_SEPARATOR) || command.contains('/') {
        let path = PathBuf::from(command);
        return if is_executable(&path) {
            Some(path)
        } else {
            None
        };
    }
    let path_var = std::env::var_os("PATH")?;
    for directory in std::env::split_paths(&path_var) {
        let candidate = directory.join(command);
        if is_executable(&candidate) {
            return Some(candidate);
        }
        #[cfg(windows)]
        {
            for extension in ["exe", "cmd", "bat"] {
                let candidate = directory.join(format!("{command}.{extension}"));
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

fn is_executable(path: &Path) -> bool {
    let Ok(metadata) = std::fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Well-known auth files for supported and adjacent agent CLIs.
pub fn known_auth_paths() -> Vec<PathBuf> {
    let Some(home) = home_dir() else {
        return Vec::new();
    };
    let candidates = [
        ".hermes/auth.json",
        ".hermes/config.toml",
        ".config/opencode/auth.json",
        ".config/opencode/config.json",
        ".claude/.credentials.json",
        ".claude.json",
        ".codex/auth.json",
        ".config/anthropic",
        ".config/github-copilot/hosts.json",
        ".config/artex/auth.json",
    ];
    candidates
        .iter()
        .map(|relative| home.join(relative))
        .collect()
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
}

fn inspect_auth_path(path: &Path) -> AuthFileStatus {
    match std::fs::metadata(path) {
        Ok(metadata) => {
            let age_secs = metadata
                .modified()
                .ok()
                .and_then(|modified| SystemTime::now().duration_since(modified).ok())
                .map(|duration| duration.as_secs());
            AuthFileStatus {
                path: path.to_path_buf(),
                exists: true,
                age_secs,
                size_bytes: if metadata.is_file() {
                    Some(metadata.len())
                } else {
                    None
                },
            }
        }
        Err(_) => AuthFileStatus {
            path: path.to_path_buf(),
            exists: false,
            age_secs: None,
            size_bytes: None,
        },
    }
}

async fn probe_url(base: &str) -> (bool, Option<String>) {
    let Ok(url) = reqwest::Url::parse(base) else {
        return (false, Some("invalid URL".into()));
    };
    let Some(host) = url.host_str() else {
        return (false, Some("URL has no host".into()));
    };
    let port = url.port_or_known_default().unwrap_or(80);
    match tokio::time::timeout(Duration::from_millis(500), TcpStream::connect((host, port))).await {
        Ok(Ok(_)) => (true, None),
        Ok(Err(error)) => (false, Some(error.to_string())),
        Err(_) => (false, Some("connection timed out".into())),
    }
}

/// Format an age for human-readable output.
pub fn format_age(age_secs: Option<u64>) -> String {
    match age_secs {
        Some(age) if age < 60 => format!("{age}s ago"),
        Some(age) if age < 3600 => format!("{}m ago", age / 60),
        Some(age) if age < 86_400 => format!("{}h ago", age / 3600),
        Some(age) => format!("{}d ago", age / 86_400),
        None => "unknown".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_known_binary() {
        #[cfg(windows)]
        assert!(resolve_binary("cmd").is_some());
        #[cfg(not(windows))]
        assert!(resolve_binary("sh").is_some());
    }

    #[test]
    fn rejects_missing_binary() {
        assert!(resolve_binary("vouch-definitely-not-installed-xyz").is_none());
    }

    #[test]
    fn reports_missing_auth_file() {
        let status = inspect_auth_path(Path::new("/nonexistent/vouch/auth.json"));
        assert!(!status.exists);
    }
}
