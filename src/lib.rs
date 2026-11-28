//! `vouch` — a local authentication multiplexer and process-supervised
//! gateway daemon for AI orchestrators.
//!
//! The crate is split into small subsystems so that the ingress adapters, the
//! routing policy, and the egress drivers can be tested and reasoned about
//! independently:
//!
//! * [`config`] — typed TOML configuration and hot-reload storage.
//! * [`error`] — the single error type used across the daemon.
//! * [`auth`] — virtual ingress tokens and local CLI/auth discovery.
//! * [`router`] — model routing, fallback chains, and session affinity.
//! * [`driver`] — HTTP, CLI, and PTY egress implementations.
//! * [`ingress`] — OpenAI, Anthropic, and operational HTTP endpoints.
//! * [`cli`] — operator-facing subcommands.
//! * [`telemetry`] — structured logging and secret redaction.
//! * [`metrics`] — Prometheus instrumentation.
//! * [`state`] — the shared, thread-safe application state.

pub mod auth;
pub mod cache;
pub mod cli;
pub mod config;
pub mod driver;
pub mod error;
pub mod ingress;
pub mod memory;
pub mod metrics;
pub mod persistence;
pub mod recorder;
pub mod router;
pub mod security;
pub mod state;
pub mod telemetry;
pub mod usage;

/// Crate version, sourced from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Validate `unix_socket` length (`< 100` bytes for macOS/BSD `sockaddr_un`),
/// create and harden parent directory (`0700` on Unix when user-owned and non-sticky),
/// probe any pre-existing socket so a live daemon is never unlinked, and remove stale
/// dead socket files before binding.
pub fn prepare_unix_socket_path(path: &std::path::Path) -> error::Result<()> {
    let len = path.to_string_lossy().len();
    if len == 0 || len >= 100 {
        return Err(error::GatewayError::Config(format!(
            "server.unix_socket {:?} must be non-empty and under 100 bytes for macOS/BSD sockaddr_un compatibility (got {len} bytes)",
            path.display()
        )));
    }
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|error| {
            error::GatewayError::Io(std::io::Error::new(
                error.kind(),
                format!("cannot create {}: {error}", parent.display()),
            ))
        })?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(meta) = std::fs::metadata(parent) {
                let sticky = (meta.permissions().mode() & 0o1000) != 0;
                let system_root = matches!(
                    parent.to_str(),
                    Some("/" | "/tmp" | "/var/tmp" | "/run" | "/var/run")
                );
                if !sticky && !system_root {
                    std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700))
                        .map_err(|error| {
                            error::GatewayError::Io(std::io::Error::new(
                                error.kind(),
                                format!(
                                    "cannot set 0700 permissions on {}: {error}",
                                    parent.display()
                                ),
                            ))
                        })?;
                }
            }
        }
    }
    if std::fs::symlink_metadata(path).is_ok() {
        #[cfg(unix)]
        {
            match std::os::unix::net::UnixStream::connect(path) {
                Ok(_live) => {
                    return Err(error::GatewayError::Io(std::io::Error::new(
                        std::io::ErrorKind::AddrInUse,
                        format!(
                            "Unix socket {} already has an active listener",
                            path.display()
                        ),
                    )));
                }
                Err(err)
                    if matches!(
                        err.kind(),
                        std::io::ErrorKind::ConnectionRefused | std::io::ErrorKind::NotFound
                    ) =>
                {
                    std::fs::remove_file(path).map_err(|error| {
                        error::GatewayError::Io(std::io::Error::new(
                            error.kind(),
                            format!("cannot remove stale socket {}: {error}", path.display()),
                        ))
                    })?;
                }
                Err(err) => {
                    return Err(error::GatewayError::Io(std::io::Error::new(
                        err.kind(),
                        format!(
                            "cannot probe existing Unix socket {}: {err}",
                            path.display()
                        ),
                    )));
                }
            }
        }
        #[cfg(not(unix))]
        {
            std::fs::remove_file(path).map_err(|error| {
                error::GatewayError::Io(std::io::Error::new(
                    error.kind(),
                    format!("cannot remove stale socket {}: {error}", path.display()),
                ))
            })?;
        }
    }
    Ok(())
}
