//! SPDX-License-Identifier: GPL-3.0-or-later
//! Temporary file lifecycle for CLI/PTY large-prompt spillover and multimodal
//! attachments.
//!
//! Ensures files written to `std::env::temp_dir()` use `0600` permissions on
//! Unix and are deterministically unlinked when their RAII guard drops.

use std::{
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
};

use base64::Engine as _;
use uuid::Uuid;

use crate::{
    driver::{ChatMessage, ContentPart, ExecutionRequest, MessageContent, tool_bridge},
    error::{GatewayError, Result},
};

/// RAII temporary file used when `stdin = false` and an inline CLI/PTY prompt
/// exceeds `max_inline_prompt_bytes` (`ARG_MAX` / `E2BIG` guard).
#[derive(Debug)]
pub struct PromptSpillFile {
    path: PathBuf,
}

impl PromptSpillFile {
    /// Persist `contents` into a private `0600` file under `std::env::temp_dir()`.
    pub fn create(contents: &str) -> Result<Self> {
        let filename = format!("vouch_prompt_{}.txt", Uuid::new_v4().simple());
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
    let filename = format!("vouch_att_{}.{ext}", Uuid::new_v4().simple());
    write_private_temp_file(&filename, &decoded)
}

/// Render an execution request for a CLI/PTY backend, materializing any base64
/// image attachments to private temporary files and appending tool / structured
/// output instructions.
pub fn materialize_prompt(request: &ExecutionRequest) -> Result<(String, TempAttachmentGuard)> {
    let mut guard = TempAttachmentGuard::new();
    let mut prompt = String::new();

    if !request.is_resuming()
        && let Some(system) = request.system.as_deref()
        && !system.trim().is_empty()
    {
        prompt.push_str("System:\n");
        prompt.push_str(system.trim());
        prompt.push_str("\n\n");
    }

    let messages = request.active_messages();
    for message in messages {
        render_message_with_attachments(message, &mut prompt, &mut guard)?;
    }

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

    Ok((prompt, guard))
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
}
