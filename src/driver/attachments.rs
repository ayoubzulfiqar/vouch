//! SPDX-License-Identifier: GPL-3.0-or-later
//! Temporary file lifecycle for CLI/PTY large-prompt spillover and multimodal
//! attachments.
//!
//! Ensures files written to `std::env::temp_dir()` use `0600` permissions on
//! Unix and are deterministically unlinked when their RAII guard drops.

use std::{
    fs::OpenOptions,
    io::Write,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::{Path, PathBuf},
    time::Duration,
};

use base64::Engine as _;
use uuid::Uuid;

use crate::{
    config::AttachmentsConfig,
    driver::{ChatMessage, ContentPart, ExecutionRequest, MessageContent, tool_bridge},
    error::{GatewayError, Result},
};

const MAX_IMAGE_REDIRECTS: usize = 3;
const REMOTE_IMAGE_TIMEOUT: Duration = Duration::from_secs(15);

/// RAII temporary file used when `stdin = false` and an inline CLI/PTY prompt
/// exceeds `max_inline_prompt_bytes` (`ARG_MAX` / `E2BIG` guard).
#[derive(Debug)]
pub struct PromptSpillFile {
    path: PathBuf,
}

impl PromptSpillFile {
    /// Persist `contents` into a private `0600` file under `std::env::temp_dir()`.
    pub fn create(contents: &str) -> Result<Self> {
        let filename = pid_temp_filename("vouch_prompt", "txt");
        let path = write_private_temp_file(&filename, contents.as_bytes())?;
        Ok(Self { path })
    }

    /// Instruction substituted for `{prompt}` when spilling to disk.
    pub fn prompt_reference(&self) -> String {
        format!(
            "Read and execute the full prompt instructions located in file: {}",
            self.path.display()
        )
    }

    /// Underlying file path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for PromptSpillFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// RAII cleanup guard owning decoded multimodal attachment files for a single
/// CLI/PTY backend execution.
#[derive(Debug, Default)]
pub struct TempAttachmentGuard {
    paths: Vec<PathBuf>,
}

impl TempAttachmentGuard {
    /// Empty guard.
    pub fn new() -> Self {
        Self { paths: Vec::new() }
    }

    /// Paths currently owned by this guard.
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    fn push(&mut self, path: PathBuf) {
        self.paths.push(path);
    }
}

impl Drop for TempAttachmentGuard {
    fn drop(&mut self) {
        for path in &self.paths {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Parse an RFC 2397 base64 data URI (`data:<media_type>;base64,<payload>`).
pub fn parse_data_uri(url: &str) -> Option<(String, String)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, payload) = rest.split_once(";base64,")?;
    let media_type = meta.trim();
    let media_type = if media_type.is_empty() {
        "application/octet-stream"
    } else {
        media_type
    };
    Some((media_type.to_string(), payload.trim().to_string()))
}

/// Decode a base64 attachment into a private `0600` temp file and return its path.
pub fn write_base64_attachment(media_type: &str, data: &str) -> Result<PathBuf> {
    let cleaned: String = data.chars().filter(|c| !c.is_whitespace()).collect();
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(cleaned.as_bytes())
        .map_err(|error| GatewayError::BadRequest(format!("invalid base64 attachment: {error}")))?;
    let ext = extension_for_media_type(media_type);
    let filename = pid_temp_filename("vouch_att", ext);
    write_private_temp_file(&filename, &decoded)
}

/// Render an execution request for a CLI/PTY backend, materializing any base64
/// image attachments to private temporary files and appending tool / structured
/// output instructions.
pub fn materialize_prompt(request: &ExecutionRequest) -> Result<(String, TempAttachmentGuard)> {
    let mut guard = TempAttachmentGuard::new();
    let mut prompt = init_system_prompt(request);
    for message in request.active_messages() {
        render_message_with_attachments(message, &mut prompt, &mut guard)?;
    }
    Ok((finalize_prompt_suffix(request, prompt), guard))
}

/// Async version of [`materialize_prompt`] that honors `[attachments]` settings,
/// including SSRF-safe downloading of remote `image_url` payloads when
/// `attachments.fetch_remote_urls` is enabled.
pub async fn materialize_prompt_with_config(
    request: &ExecutionRequest,
    config: &AttachmentsConfig,
) -> Result<(String, TempAttachmentGuard)> {
    let mut guard = TempAttachmentGuard::new();
    let mut prompt = init_system_prompt(request);
    for message in request.active_messages() {
        render_message_with_attachments_async(message, &mut prompt, &mut guard, config).await?;
    }
    Ok((finalize_prompt_suffix(request, prompt), guard))
}

fn init_system_prompt(request: &ExecutionRequest) -> String {
    let mut prompt = String::new();
    if !request.is_resuming()
        && let Some(system) = request.system.as_deref()
        && !system.trim().is_empty()
    {
        prompt.push_str("System:\n");
        prompt.push_str(system.trim());
        prompt.push_str("\n\n");
    }
    prompt
}

fn finalize_prompt_suffix(request: &ExecutionRequest, prompt: String) -> String {
    let mut prompt = prompt.trim_end().to_string();
    if !request.tools.is_empty() {
        tool_bridge::append_tool_instructions(
            &mut prompt,
            &request.tools,
            request.tool_choice.as_ref(),
        );
    }
    if let Some(format) = &request.response_format {
        tool_bridge::append_structured_output_instructions(&mut prompt, format);
    }
    prompt
}

async fn render_message_with_attachments_async(
    message: &ChatMessage,
    output: &mut String,
    guard: &mut TempAttachmentGuard,
    config: &AttachmentsConfig,
) -> Result<()> {
    match &message.content {
        MessageContent::Text(_) => {
            tool_bridge::format_message(message, output);
        }
        MessageContent::Parts(parts) => {
            let mut flattened = String::new();
            for part in parts {
                match part {
                    ContentPart::Text { text, .. } => {
                        append_flattened_line(&mut flattened, text.trim());
                    }
                    ContentPart::ImageBase64 { media_type, data } => {
                        let path = write_base64_attachment(media_type, data)?;
                        guard.push(path.clone());
                        append_flattened_line(
                            &mut flattened,
                            &format!("[Attached image file: {}]", path.display()),
                        );
                    }
                    ContentPart::ImageUrl { url } => {
                        if let Some((media_type, data)) = parse_data_uri(url) {
                            let path = write_base64_attachment(&media_type, &data)?;
                            guard.push(path.clone());
                            append_flattened_line(
                                &mut flattened,
                                &format!("[Attached image file: {}]", path.display()),
                            );
                        } else if config.fetch_remote_urls {
                            let (bytes, media_type) = fetch_remote_image(url, config).await?;
                            let filename = pid_temp_filename(
                                "vouch_att",
                                extension_for_media_type(media_type),
                            );
                            let path = write_private_temp_file(&filename, &bytes)?;
                            guard.push(path.clone());
                            append_flattened_line(
                                &mut flattened,
                                &format!("[Attached image file: {}]", path.display()),
                            );
                        } else {
                            append_flattened_line(
                                &mut flattened,
                                &format!("[Attached image URL: {url}]"),
                            );
                        }
                    }
                }
            }
            let rendered_message = ChatMessage {
                role: message.role.clone(),
                content: MessageContent::Text(flattened),
                tool_calls: message.tool_calls.clone(),
                tool_call_id: message.tool_call_id.clone(),
                name: message.name.clone(),
            };
            tool_bridge::format_message(&rendered_message, output);
        }
    }
    Ok(())
}

fn append_flattened_line(flattened: &mut String, piece: &str) {
    if piece.is_empty() {
        return;
    }
    if !flattened.is_empty() {
        flattened.push('\n');
    }
    flattened.push_str(piece);
}

fn render_message_with_attachments(
    message: &ChatMessage,
    output: &mut String,
    guard: &mut TempAttachmentGuard,
) -> Result<()> {
    match &message.content {
        MessageContent::Text(_) => {
            tool_bridge::format_message(message, output);
        }
        MessageContent::Parts(parts) => {
            let mut flattened = String::new();
            for part in parts {
                match part {
                    ContentPart::Text { text, .. } => {
                        if !text.trim().is_empty() {
                            if !flattened.is_empty() {
                                flattened.push('\n');
                            }
                            flattened.push_str(text.trim());
                        }
                    }
                    ContentPart::ImageBase64 { media_type, data } => {
                        let path = write_base64_attachment(media_type, data)?;
                        let marker = format!("[Attached image file: {}]", path.display());
                        guard.push(path);
                        if !flattened.is_empty() {
                            flattened.push('\n');
                        }
                        flattened.push_str(&marker);
                    }
                    ContentPart::ImageUrl { url } => {
                        if let Some((media_type, data)) = parse_data_uri(url) {
                            let path = write_base64_attachment(&media_type, &data)?;
                            let marker = format!("[Attached image file: {}]", path.display());
                            guard.push(path);
                            if !flattened.is_empty() {
                                flattened.push('\n');
                            }
                            flattened.push_str(&marker);
                        } else {
                            if !flattened.is_empty() {
                                flattened.push('\n');
                            }
                            flattened.push_str(&format!("[Attached image URL: {url}]"));
                        }
                    }
                }
            }
            let rendered_message = ChatMessage {
                role: message.role.clone(),
                content: MessageContent::Text(flattened),
                tool_calls: message.tool_calls.clone(),
                tool_call_id: message.tool_call_id.clone(),
                name: message.name.clone(),
            };
            tool_bridge::format_message(&rendered_message, output);
        }
    }
    Ok(())
}

fn extension_for_media_type(media_type: &str) -> &'static str {
    match media_type
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "image/png" => "png",
        "image/jpeg" | "image/jpg" => "jpg",
        "image/webp" => "webp",
        "image/gif" => "gif",
        "image/svg+xml" => "svg",
        "application/pdf" => "pdf",
        "text/plain" => "txt",
        _ => "bin",
    }
}

/// Check whether an IP address belongs to loopback, private, link-local, or
/// otherwise non-public ranges that must be blocked for remote image fetches.
pub fn is_forbidden_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_forbidden_v4(v4),
        IpAddr::V6(v6) => {
            if let Some(mapped) = v6.to_ipv4_mapped().or_else(|| {
                (!v6.is_unspecified() && !v6.is_loopback())
                    .then(|| v6.to_ipv4())
                    .flatten()
            }) {
                return is_forbidden_v4(mapped);
            }
            let is_site_local = (v6.segments()[0] & 0xffc0) == 0xfec0;
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || v6.is_unique_local()
                || v6.is_unicast_link_local()
                || is_site_local
        }
    }
}

fn is_forbidden_v4(v4: Ipv4Addr) -> bool {
    let octets = v4.octets();
    let is_this_network = octets[0] == 0;
    let is_cgnat = octets[0] == 100 && (octets[1] & 0b1100_0000) == 64;
    let is_benchmark = octets[0] == 198 && matches!(octets[1], 18 | 19);
    is_this_network
        || v4.is_loopback()
        || v4.is_private()
        || v4.is_link_local()
        || v4.is_unspecified()
        || v4.is_broadcast()
        || v4.is_multicast()
        || v4.is_documentation()
        || is_cgnat
        || is_benchmark
}

async fn resolve_and_validate_addrs(
    url: &reqwest::Url,
    allow_private_ips: bool,
) -> Result<(Option<String>, Vec<SocketAddr>)> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err(GatewayError::BadRequest(format!(
            "unsupported remote image URL scheme {:?}",
            url.scheme()
        )));
    }
    let Some(raw_host) = url.host_str() else {
        return Err(GatewayError::BadRequest(
            "remote image URL is missing a host".into(),
        ));
    };
    let bare_host = raw_host.trim_start_matches('[').trim_end_matches(']');
    let port = url.port_or_known_default().unwrap_or(80);
    let (domain_for_pinning, addrs) = if let Ok(ip) = bare_host.parse::<IpAddr>() {
        (None, vec![SocketAddr::new(ip, port)])
    } else {
        let resolved: Vec<SocketAddr> = tokio::net::lookup_host((bare_host, port))
            .await
            .map_err(|error| {
                GatewayError::BadRequest(format!(
                    "cannot resolve remote image host {bare_host:?}: {error}"
                ))
            })?
            .collect();
        if resolved.is_empty() {
            return Err(GatewayError::BadRequest(format!(
                "remote image host {bare_host:?} resolved to no addresses"
            )));
        }
        (Some(bare_host.to_string()), resolved)
    };
    if !allow_private_ips {
        for addr in &addrs {
            if is_forbidden_ip(addr.ip()) {
                return Err(GatewayError::BadRequest(format!(
                    "refusing remote image fetch from blocked non-public IP {}",
                    addr.ip()
                )));
            }
        }
    }
    Ok((domain_for_pinning, addrs))
}

/// Download a remote `image_url` enforcing SSRF checks, redirect validation,
/// bounded byte streaming, `Content-Type` header checks, and image magic bytes.
pub async fn fetch_remote_image(
    url_str: &str,
    config: &AttachmentsConfig,
) -> Result<(Vec<u8>, &'static str)> {
    let mut current = reqwest::Url::parse(url_str).map_err(|error| {
        GatewayError::BadRequest(format!("invalid remote image URL {url_str:?}: {error}"))
    })?;

    for hop in 0..=MAX_IMAGE_REDIRECTS {
        let (domain, addrs) =
            resolve_and_validate_addrs(&current, config.allow_private_ips).await?;
        let mut builder = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(REMOTE_IMAGE_TIMEOUT);
        if let Some(domain) = domain.as_deref() {
            builder = builder.resolve_to_addrs(domain, &addrs);
        }
        let client = builder.build().map_err(|error| {
            GatewayError::Internal(format!("cannot build image HTTP client: {error}"))
        })?;

        let mut response = client.get(current.clone()).send().await.map_err(|error| {
            GatewayError::Upstream(format!("failed to fetch remote image: {error}"))
        })?;

        if response.status().is_redirection() {
            if hop == MAX_IMAGE_REDIRECTS {
                return Err(GatewayError::BadRequest(
                    "remote image URL exceeded maximum redirect count".into(),
                ));
            }
            let location = response
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|value| value.to_str().ok())
                .ok_or_else(|| {
                    GatewayError::Upstream("remote image redirect missing Location header".into())
                })?;
            current = current.join(location).map_err(|error| {
                GatewayError::BadRequest(format!("invalid redirect Location {location:?}: {error}"))
            })?;
            continue;
        }

        if !response.status().is_success() {
            return Err(GatewayError::Upstream(format!(
                "remote image fetch returned HTTP {}",
                response.status()
            )));
        }

        if let Some(len) = response.content_length()
            && len as usize > config.max_image_bytes
        {
            return Err(GatewayError::BadRequest(format!(
                "remote image content-length ({len} bytes) exceeds max_image_bytes ({})",
                config.max_image_bytes
            )));
        }

        if let Some(ct) = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
        {
            let mime = ct
                .split(';')
                .next()
                .unwrap_or("")
                .trim()
                .to_ascii_lowercase();
            if !mime.is_empty() && !mime.starts_with("image/") {
                return Err(GatewayError::BadRequest(format!(
                    "remote image returned non-image Content-Type {mime:?}"
                )));
            }
        }

        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|error| {
            GatewayError::Upstream(format!("error reading remote image body: {error}"))
        })? {
            if bytes.len() + chunk.len() > config.max_image_bytes {
                return Err(GatewayError::BadRequest(format!(
                    "remote image payload exceeds max_image_bytes ({})",
                    config.max_image_bytes
                )));
            }
            bytes.extend_from_slice(&chunk);
        }

        let detected = sniff_image_media_type(&bytes).ok_or_else(|| {
            GatewayError::BadRequest(
                "remote image payload did not match supported image magic bytes (png/jpeg/gif/webp)"
                    .into(),
            )
        })?;
        return Ok((bytes, detected));
    }

    Err(GatewayError::BadRequest(
        "remote image fetch failed after redirects".into(),
    ))
}

fn sniff_image_media_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(b"\xFF\xD8\xFF") {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

fn pid_temp_filename(prefix: &str, ext: &str) -> String {
    format!(
        "{prefix}_{}_{}.{ext}",
        std::process::id(),
        Uuid::new_v4().simple()
    )
}

fn parse_temp_pid(name: &str) -> Option<u32> {
    let rest = name
        .strip_prefix("vouch_att_")
        .or_else(|| name.strip_prefix("vouch_prompt_"))?;
    let (pid_part, _) = rest.split_once('_')?;
    pid_part.parse::<u32>().ok()
}

/// Check whether a process ID is currently alive on this host.
#[cfg(unix)]
pub fn is_process_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    let Ok(raw) = i32::try_from(pid) else {
        return false;
    };
    !matches!(
        nix::sys::signal::kill(nix::unistd::Pid::from_raw(raw), None),
        Err(nix::errno::Errno::ESRCH)
    )
}

#[cfg(not(unix))]
pub fn is_process_alive(_pid: u32) -> bool {
    false
}

/// Sweep orphaned `vouch_att_*` and `vouch_prompt_*` temp files left behind by
/// crashed daemon instances or aged past `max_age`.
pub fn sweep_orphaned_temp_files(max_age: Duration) -> usize {
    sweep_orphaned_temp_files_in(&std::env::temp_dir(), max_age)
}

/// Sweep orphaned `vouch_att_*` and `vouch_prompt_*` temp files inside a target directory.
pub fn sweep_orphaned_temp_files_in(dir: &Path, max_age: Duration) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut removed = 0usize;
    let current_pid = std::process::id();
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_file() {
            continue;
        }
        let name_os = entry.file_name();
        let Some(name) = name_os.to_str() else {
            continue;
        };
        if !name.starts_with("vouch_att_") && !name.starts_with("vouch_prompt_") {
            continue;
        }
        let age_exceeded = entry
            .metadata()
            .and_then(|meta| meta.modified())
            .ok()
            .and_then(|mtime| mtime.elapsed().ok())
            .is_some_and(|age| age > max_age);

        let should_remove = match parse_temp_pid(name) {
            Some(pid) if pid == current_pid => false,
            Some(pid) => !is_process_alive(pid) || age_exceeded,
            None => age_exceeded,
        };
        if should_remove && std::fs::remove_file(entry.path()).is_ok() {
            removed += 1;
        }
    }
    removed
}

fn write_private_temp_file(filename: &str, bytes: &[u8]) -> Result<PathBuf> {
    let path = std::env::temp_dir().join(filename);
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&path).map_err(|error| {
        GatewayError::Io(std::io::Error::new(
            error.kind(),
            format!("cannot create temp file {}: {error}", path.display()),
        ))
    })?;
    if let Err(error) = file.write_all(bytes).and_then(|()| file.flush()) {
        let _ = std::fs::remove_file(&path);
        return Err(GatewayError::Io(std::io::Error::new(
            error.kind(),
            format!("cannot write temp file {}: {error}", path.display()),
        )));
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_spill_file_writes_0600_and_cleans_on_drop() {
        let spill = PromptSpillFile::create("giant prompt payload").expect("create spill file");
        let path = spill.path().to_path_buf();
        assert!(path.exists());
        assert_eq!(
            std::fs::read_to_string(&path).expect("read spill"),
            "giant prompt payload"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).expect("meta").permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
        assert!(
            spill
                .prompt_reference()
                .contains(&path.display().to_string())
        );
        drop(spill);
        assert!(!path.exists());
    }

    #[test]
    fn parses_data_uri_and_writes_attachment_with_drop_cleanup() {
        // 1x1 transparent PNG base64
        let b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
        let uri = format!("data:image/png;base64,{b64}");
        let (mime, raw) = parse_data_uri(&uri).expect("valid data uri");
        assert_eq!(mime, "image/png");
        assert_eq!(raw, b64);

        let path = write_base64_attachment(&mime, &raw).expect("write attachment");
        assert!(path.exists());
        assert_eq!(path.extension().and_then(|s| s.to_str()), Some("png"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn blocks_loopback_private_link_local_and_mapped_v6_ips() {
        assert!(is_forbidden_ip("127.0.0.1".parse().expect("ip")));
        assert!(is_forbidden_ip("10.0.0.5".parse().expect("ip")));
        assert!(is_forbidden_ip("172.16.0.1".parse().expect("ip")));
        assert!(is_forbidden_ip("192.168.1.10".parse().expect("ip")));
        assert!(is_forbidden_ip("169.254.169.254".parse().expect("ip")));
        assert!(is_forbidden_ip("100.64.0.1".parse().expect("ip")));
        assert!(is_forbidden_ip("::1".parse().expect("ip")));
        assert!(is_forbidden_ip("::ffff:127.0.0.1".parse().expect("ip")));
        assert!(is_forbidden_ip("fe80::1".parse().expect("ip")));
        assert!(!is_forbidden_ip("8.8.8.8".parse().expect("ip")));
        assert!(!is_forbidden_ip(
            "2606:4700:4700::1111".parse().expect("ip")
        ));
    }

    #[test]
    fn sweeps_dead_pid_orphans_while_preserving_live_pid_files() {
        let dir = tempfile::tempdir().expect("temp");
        let dead_prompt = dir.path().join("vouch_prompt_99999999_deadbeef.txt");
        let dead_att = dir.path().join("vouch_att_99999999_deadbeef.png");
        let live_prompt = dir
            .path()
            .join(format!("vouch_prompt_{}_alive123.txt", std::process::id()));
        std::fs::write(&dead_prompt, "orphan").expect("write dead prompt");
        std::fs::write(&dead_att, "orphan").expect("write dead att");
        std::fs::write(&live_prompt, "active").expect("write live prompt");

        let removed = sweep_orphaned_temp_files_in(dir.path(), Duration::from_secs(3600));
        assert_eq!(removed, 2);
        assert!(!dead_prompt.exists());
        assert!(!dead_att.exists());
        assert!(live_prompt.exists());
    }
}
