//! User-level service installation for `vouch`.
//!
//! Linux hosts get a systemd user unit; macOS hosts get a launchd agent. Both
//! are written under the invoking user's home directory and never require
//! elevated privileges. The generator is pure so that the exact rendered
//! unit can be unit-tested without touching the filesystem.

use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

use super::{ServiceArgs, ServiceCommand, ServiceInstallArgs, ServiceUninstallArgs};
use crate::error::{GatewayError, Result};

const SERVICE_NAME: &str = "vouch";
const LAUNCHD_LABEL: &str = "io.github.ayoubzulfiqar.vouch";

/// Dispatch a service subcommand.
pub fn run(args: &ServiceArgs) -> Result<()> {
    match &args.command {
        ServiceCommand::Install(install) => install_service(install),
        ServiceCommand::Uninstall(uninstall) => uninstall_service(uninstall),
    }
}

/// Render the platform-appropriate unit file for the given paths.
pub fn render_unit(binary: &Path, config: &Path) -> Result<String> {
    let binary = canonical_display(binary)?;
    let config = canonical_display(config)?;
    if std::env::consts::OS == "macos" {
        Ok(render_launchd(&binary, &config))
    } else if cfg!(target_os = "linux") {
        Ok(render_systemd(&binary, &config))
    } else {
        Err(GatewayError::Config(format!(
            "service installation is unsupported on {}",
            std::env::consts::OS
        )))
    }
}

/// Path of the unit file managed by this command.
pub fn unit_path() -> Result<PathBuf> {
    let home =
        std::env::var_os("HOME").ok_or_else(|| GatewayError::Config("HOME is not set".into()))?;
    let mut path = PathBuf::from(home);
    if std::env::consts::OS == "macos" {
        path.push("Library");
        path.push("LaunchAgents");
        path.push(format!("{LAUNCHD_LABEL}.plist"));
    } else if cfg!(target_os = "linux") {
        path.push(".config");
        path.push("systemd");
        path.push("user");
        path.push(format!("{SERVICE_NAME}.service"));
    } else {
        return Err(GatewayError::Config(format!(
            "service installation is unsupported on {}",
            std::env::consts::OS
        )));
    }
    Ok(path)
}

fn canonical_display(path: &Path) -> Result<String> {
    let resolved = fs::canonicalize(path).map_err(|error| {
        GatewayError::Config(format!("cannot resolve {}: {error}", path.display()))
    })?;
    Ok(resolved.display().to_string())
}

/// Deterministic systemd user unit.
fn render_systemd(binary: &str, config: &str) -> String {
    format!(
        "[Unit]\n\
         Description=vouch local authentication multiplexer\n\
         After=network-online.target\n\
         Wants=network-online.target\n\
         \n\
         [Service]\n\
         Type=simple\n\
         ExecStart={binary} serve --config {config}\n\
         ExecReload=/bin/kill -HUP $MAINPID\n\
         Restart=on-failure\n\
         RestartSec=3\n\
         KillMode=control-group\n\
         TimeoutStopSec=10\n\
         NoNewPrivileges=true\n\
         PrivateTmp=true\n\
         \n\
         [Install]\n\
         WantedBy=default.target\n"
    )
}

/// Deterministic launchd agent plist.
fn render_launchd(binary: &str, config: &str) -> String {
    let logs = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
         <plist version=\"1.0\">\n\
         <dict>\n\
           <key>Label</key>\n\
           <string>{LAUNCHD_LABEL}</string>\n\
           <key>ProgramArguments</key>\n\
           <array>\n\
             <string>{binary}</string>\n\
             <string>serve</string>\n\
             <string>--config</string>\n\
             <string>{config}</string>\n\
           </array>\n\
           <key>RunAtLoad</key>\n\
           <true/>\n\
           <key>KeepAlive</key>\n\
           <true/>\n\
           <key>StandardOutPath</key>\n\
           <string>{logs}/Library/Logs/{LAUNCHD_LABEL}.out.log</string>\n\
           <key>StandardErrorPath</key>\n\
           <string>{logs}/Library/Logs/{LAUNCHD_LABEL}.err.log</string>\n\
         </dict>\n\
         </plist>\n"
    )
}

fn install_service(args: &ServiceInstallArgs) -> Result<()> {
    let config = &args.config;
    if !config.is_file() {
        return Err(GatewayError::Config(format!(
            "configuration file {} does not exist",
            config.display()
        )));
    }
    let binary = match &args.binary {
        Some(path) => path.clone(),
        None => std::env::current_exe().map_err(|error| {
            GatewayError::Config(format!("cannot find current executable: {error}"))
        })?,
    };

    let unit = render_unit(&binary, config)?;
    let path = unit_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            GatewayError::Io(std::io::Error::new(
                error.kind(),
                format!("cannot create {}: {error}", parent.display()),
            ))
        })?;
    }
    write_private(&path, &unit)?;
    println!("wrote {}", path.display());

    if std::env::consts::OS == "macos" {
        if !args.no_start {
            run_command("launchctl", &["load", "-w", &path.to_string_lossy()])?;
        }
        println!("start with: launchctl load -w {}", path.display());
    } else if cfg!(target_os = "linux") {
        run_command("systemctl", &["--user", "daemon-reload"])?;
        if !args.no_start {
            run_command(
                "systemctl",
                &[
                    "--user",
                    "enable",
                    "--now",
                    &format!("{SERVICE_NAME}.service"),
                ],
            )?;
        }
        println!("status with: systemctl --user status {SERVICE_NAME}.service");
    }
    Ok(())
}

fn uninstall_service(args: &ServiceUninstallArgs) -> Result<()> {
    let path = unit_path()?;
    if std::env::consts::OS == "macos" {
        if !args.no_stop && path.exists() {
            let _ = run_command("launchctl", &["unload", &path.to_string_lossy()]);
        }
    } else if cfg!(target_os = "linux") && !args.no_stop {
        let _ = run_command(
            "systemctl",
            &[
                "--user",
                "disable",
                "--now",
                &format!("{SERVICE_NAME}.service"),
            ],
        );
    }

    if path.exists() {
        fs::remove_file(&path).map_err(|error| {
            GatewayError::Io(std::io::Error::new(
                error.kind(),
                format!("cannot remove {}: {error}", path.display()),
            ))
        })?;
        println!("removed {}", path.display());
    } else {
        println!("no unit file at {}", path.display());
    }

    if cfg!(target_os = "linux") && std::env::consts::OS != "macos" {
        let _ = run_command("systemctl", &["--user", "daemon-reload"]);
    }
    Ok(())
}

/// Write a unit file readable only by its owner.
fn write_private(path: &Path, contents: &str) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .map_err(|error| {
            GatewayError::Io(std::io::Error::new(
                error.kind(),
                format!("cannot open {}: {error}", path.display()),
            ))
        })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(0o600))
            .map_err(|error| {
                GatewayError::Io(std::io::Error::new(
                    error.kind(),
                    format!("cannot set permissions on {}: {error}", path.display()),
                ))
            })?;
    }
    file.write_all(contents.as_bytes())?;
    file.flush()?;
    Ok(())
}

/// Run a supervisor command, returning its combined output on failure.
fn run_command(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).args(args).output().map_err(|error| {
        GatewayError::Io(std::io::Error::new(
            error.kind(),
            format!("cannot run {program}: {error}"),
        ))
    })?;
    if output.status.success() {
        return Ok(String::from_utf8_lossy(&output.stdout).into_owned());
    }
    Err(GatewayError::Internal(format!(
        "{program} {} failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr).trim()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_systemd_unit_with_absolute_paths() {
        let unit = render_systemd("/usr/local/bin/vouch", "/etc/vouch/vouch.toml");
        assert!(
            unit.contains("ExecStart=/usr/local/bin/vouch serve --config /etc/vouch/vouch.toml")
        );
        assert!(unit.contains("WantedBy=default.target"));
        assert!(unit.contains("KillMode=control-group"));
        assert!(unit.contains("ExecReload=/bin/kill -HUP $MAINPID"));
    }

    #[test]
    fn launchd_label_matches_project_namespace() {
        assert_eq!(LAUNCHD_LABEL, "io.github.ayoubzulfiqar.vouch");
        let plist = render_launchd("/usr/local/bin/vouch", "/etc/vouch/vouch.toml");
        assert!(plist.contains(&format!("<string>{LAUNCHD_LABEL}</string>")));
    }
}
