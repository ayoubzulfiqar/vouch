//! Pseudo-terminal driver for CLIs that refuse to flush output when
//! `isatty() == false` or that render a TUI before printing a result.
//!
//! `portable-pty` is blocking by design, so it runs inside
//! `tokio::task::spawn_blocking` and communicates through bounded channels.
//! The same process-group guard used by the headless driver supervises the
//! child, and dropping the returned stream kills the group.

use std::{
    io::{Read, Write},
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use portable_pty::{CommandBuilder, PtySize, native_pty_system};
use tokio::sync::{Semaphore, mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    config::{
        AttachmentsConfig, AutoRespondRule, BackendConfig, BackendKind, DefaultsConfig, OutputMode,
    },
    driver::{
        AnsiStripper, ChunkStream, Driver, DriverResponse, ExecutionRequest, OutputFilterSet,
        PermitGuard, ProcessGroupGuard, ResponseFormat, StreamChunk, StreamOutputFilterAdapter,
        Utf8ChunkDecoder,
        attachments::{PromptSpillFile, TempAttachmentGuard, materialize_prompt_with_config},
        capture_session_id, classify_cli_failure, dispatch_stream_chunks, emit_subprocess_tail,
        finalize_subprocess_completion, finish_subprocess_text_stream, io_error, strip_ansi,
        tool_bridge::SubprocessStreamFilter,
    },
    error::{GatewayError, Result},
    metrics,
};

const MAX_OUTPUT_BYTES: usize = 16 * 1024 * 1024;
const AUTO_RESPOND_WINDOW_BYTES: usize = 4096;

#[derive(Debug)]
struct PtyExit {
    success: bool,
    code: i32,
    signal: Option<String>,
}

impl PtyExit {
    fn label(&self) -> String {
        match self.signal.as_deref() {
            Some(signal) => format!("signal {signal}"),
            None => format!("exit code {}", self.code),
        }
    }

    fn metric_code(&self) -> String {
        self.signal.clone().unwrap_or_else(|| self.code.to_string())
    }
}

struct CompiledRespondRule {
    pattern: String,
    regex: Option<regex::Regex>,
    response: String,
}

impl CompiledRespondRule {
    fn compile(rules: &[AutoRespondRule]) -> Vec<Self> {
        rules
            .iter()
            .map(|rule| Self {
                pattern: rule.pattern.clone(),
                regex: regex::Regex::new(&rule.pattern).ok(),
                response: rule.response.clone(),
            })
            .collect()
    }

    fn matches(&self, window: &str) -> bool {
        window.contains(&self.pattern)
            || self
                .regex
                .as_ref()
                .is_some_and(|regex| regex.is_match(window))
    }
}

/// True when an I/O read error on a PTY primary fd represents normal child-close EOF (`EIO` on Linux/macOS/BSD).
pub fn is_pty_eof_error(error: &std::io::Error) -> bool {
    #[cfg(unix)]
    {
        error.raw_os_error() == Some(nix::errno::Errno::EIO as i32)
    }
    #[cfg(not(unix))]
    {
        let _ = error;
        false
    }
}

struct SpawnedPty {
    data: mpsc::Receiver<Result<Vec<u8>>>,
    exit: oneshot::Receiver<PtyExit>,
    guard: ProcessGroupGuard,
    full_prompt: String,
    attachments: TempAttachmentGuard,
    spill: Option<PromptSpillFile>,
}

/// PTY-backed driver.
#[derive(Debug)]
pub struct PtyRunnerDriver {
    config: BackendConfig,
    attachments: AttachmentsConfig,
    output_filters: OutputFilterSet,
    semaphore: Arc<Semaphore>,
    queue_timeout: Duration,
}

impl PtyRunnerDriver {
    /// Build a driver from backend configuration.
    pub fn new(config: BackendConfig, defaults: &DefaultsConfig) -> Result<Self> {
        let queue_timeout = Duration::from_millis(
            config
                .queue_timeout_ms
                .unwrap_or(defaults.queue_timeout_ms)
                .max(1),
        );
        let output_filters = OutputFilterSet::compile(&config.output_filters)?;
        Ok(Self {
            semaphore: Arc::new(Semaphore::new(config.max_concurrency.max(1))),
            config,
            attachments: AttachmentsConfig::default(),
            output_filters,
            queue_timeout,
        })
    }

    /// Configure attachment fetching behavior (`[attachments]`).
    pub fn with_attachments(mut self, attachments: AttachmentsConfig) -> Self {
        self.attachments = attachments;
        self
    }

    /// Spawn the PTY session and return the raw byte channel plus lifecycle
    /// handles.
    async fn spawn_session(&self, request: &ExecutionRequest) -> Result<SpawnedPty> {
        let command = self.config.command.clone().ok_or_else(|| {
            GatewayError::Config(format!("backend {} has no command", self.config.id))
        })?;
        let (full_prompt, attachments) =
            materialize_prompt_with_config(request, &self.attachments).await?;
        let spill = if crate::driver::cli_pipe::should_spill_prompt_to_file(
            self.config.stdin,
            full_prompt.len(),
            self.config.max_inline_prompt_bytes,
            cfg!(windows),
            &command,
        ) {
            Some(PromptSpillFile::create(&full_prompt)?)
        } else {
            None
        };
        let arg_prompt = match &spill {
            Some(file) => file.prompt_reference(),
            None => full_prompt.clone(),
        };
        let backend_session_id = request
            .backend_session_id
            .as_deref()
            .unwrap_or(&request.session_key);
        let arg_templates = if request.is_resuming() && !self.config.resume_args.is_empty() {
            &self.config.resume_args
        } else {
            &self.config.args
        };
        let args: Vec<String> = arg_templates
            .iter()
            .map(|arg| {
                arg.replace("{prompt}", &arg_prompt)
                    .replace("{model}", &request.model)
                    .replace("{session_id}", &request.session_key)
                    .replace("{request_id}", &request.request_id)
                    .replace("{backend_session_id}", backend_session_id)
                    .replace("{profile_id}", request.profile_id.as_deref().unwrap_or(""))
            })
            .collect();
        let env = self.config.env.clone();
        let profile_env = request.profile_env.clone();
        let workdir = request
            .workdir
            .clone()
            .or_else(|| self.config.workdir.clone());
        let stdin = self.config.stdin;
        let send_eof = self.config.pty_send_eof;
        let backend = self.config.id.clone();
        let request_id = request.request_id.clone();
        let session_key = request.session_key.clone();
        let model = request.model.clone();
        let explicit_backend_sid = request.backend_session_id.clone();
        let profile_id = request.profile_id.clone();
        let rows = self.config.pty_rows.max(1);
        let cols = self.config.pty_cols.max(1);
        let auto_respond = CompiledRespondRule::compile(&self.config.pty_auto_respond);

        let (data_sender, data_receiver) = mpsc::channel::<Result<Vec<u8>>>(32);
        let (exit_sender, exit_receiver) = oneshot::channel::<PtyExit>();
        let (setup_sender, setup_receiver) =
            oneshot::channel::<std::result::Result<ProcessGroupGuard, String>>();

        let prompt_for_stdin = full_prompt.clone();
        tokio::task::spawn_blocking(move || {
            let pty_system = native_pty_system();
            let pair = match pty_system.openpty(PtySize {
                rows,
                cols,
                pixel_width: 0,
                pixel_height: 0,
            }) {
                Ok(pair) => pair,
                Err(error) => {
                    let _ = setup_sender.send(Err(format!("cannot open pty: {error}")));
                    return;
                }
            };

            let mut builder = CommandBuilder::new(command);
            builder.args(args);
            for (key, value) in env {
                builder.env(key, value);
            }
            for (key, value) in profile_env {
                builder.env(key, value);
            }
            builder.env("VOUCH_REQUEST_ID", &request_id);
            builder.env("VOUCH_SESSION_ID", &session_key);
            builder.env("VOUCH_MODEL", &model);
            if let Some(sid) = explicit_backend_sid {
                builder.env("VOUCH_BACKEND_SESSION_ID", sid);
            }
            if let Some(pid) = profile_id {
                builder.env("VOUCH_PROFILE_ID", pid);
            }
            if let Some(workdir) = workdir {
                builder.cwd(workdir);
            }

            let mut child = match pair.slave.spawn_command(builder) {
                Ok(child) => child,
                Err(error) => {
                    let _ = setup_sender.send(Err(format!("cannot spawn pty child: {error}")));
                    return;
                }
            };
            drop(pair.slave);

            let Some(pid) = child.process_id() else {
                let _ = setup_sender.send(Err("pty child did not report a pid".into()));
                let _ = child.kill();
                return;
            };
            let guard = ProcessGroupGuard::new(pid, backend.clone());
            if setup_sender.send(Ok(guard.clone_guard())).is_err() {
                guard.terminate();
                let _ = child.kill();
                return;
            }

            let mut reader = match pair.master.try_clone_reader() {
                Ok(reader) => reader,
                Err(error) => {
                    let _ = data_sender.blocking_send(Err(GatewayError::Internal(format!(
                        "cannot read pty: {error}"
                    ))));
                    guard.terminate();
                    let _ = child.wait();
                    return;
                }
            };

            let mut writer_opt = if stdin || !auto_respond.is_empty() {
                match pair.master.take_writer() {
                    Ok(writer) => Some(writer),
                    Err(error) => {
                        let _ = data_sender.blocking_send(Err(GatewayError::Internal(format!(
                            "cannot write pty: {error}"
                        ))));
                        None
                    }
                }
            } else {
                None
            };

            if stdin && let Some(writer) = writer_opt.as_mut() {
                let _ = writer.write_all(prompt_for_stdin.as_bytes());
                if !prompt_for_stdin.ends_with('\n') {
                    let _ = writer.write_all(b"\n");
                }
                if send_eof && auto_respond.is_empty() {
                    // EOT in canonical mode asks the terminal line
                    // discipline for EOF, which many REPLs honor.
                    let _ = writer.write_all(b"\x04");
                }
                let _ = writer.flush();
                if auto_respond.is_empty() {
                    writer_opt = None;
                }
            }

            let mut respond_stripper = AnsiStripper::new();
            let mut respond_window = String::new();
            let mut buffer = [0u8; 8192];
            loop {
                match reader.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(read) => {
                        let slice = &buffer[..read];
                        if !auto_respond.is_empty()
                            && let Some(writer) = writer_opt.as_mut()
                        {
                            let clean = respond_stripper.feed(slice);
                            if !clean.is_empty() {
                                respond_window.push_str(&clean);
                                trim_rolling_window(&mut respond_window, AUTO_RESPOND_WINDOW_BYTES);
                                if let Some(rule) =
                                    auto_respond.iter().find(|r| r.matches(&respond_window))
                                {
                                    let _ = writer.write_all(rule.response.as_bytes());
                                    if !rule.response.ends_with(['\n', '\r']) {
                                        let _ = writer.write_all(b"\n");
                                    }
                                    let _ = writer.flush();
                                    respond_window.clear();
                                }
                            }
                        }
                        if data_sender.blocking_send(Ok(slice.to_vec())).is_err() {
                            guard.terminate();
                            break;
                        }
                    }
                    Err(error) => {
                        if is_pty_eof_error(&error) {
                            break;
                        }
                        let _ = data_sender
                            .blocking_send(Err(io_error("cannot read pty output", error)));
                        guard.terminate();
                        break;
                    }
                }
            }
            drop(writer_opt);

            match child.wait() {
                Ok(status) => {
                    if status.success() {
                        guard.disarm();
                    }
                    let _ = exit_sender.send(PtyExit {
                        success: status.success(),
                        code: status.exit_code() as i32,
                        signal: status.signal().map(ToString::to_string),
                    });
                }
                Err(error) => {
                    guard.terminate();
                    let _ = exit_sender.send(PtyExit {
                        success: false,
                        code: -1,
                        signal: Some(format!("wait failed: {error}")),
                    });
                }
            }
        });

        let guard = tokio::time::timeout(Duration::from_secs(10), setup_receiver)
            .await
            .map_err(|_| GatewayError::Upstream("pty setup timed out".into()))?
            .map_err(|_| GatewayError::Upstream("pty setup task failed".into()))?
            .map_err(GatewayError::Upstream)?;
        Ok(SpawnedPty {
            data: data_receiver,
            exit: exit_receiver,
            guard,
            full_prompt,
            attachments,
            spill,
        })
    }

    async fn complete(
        &self,
        request: &ExecutionRequest,
        permit: PermitGuard,
    ) -> Result<DriverResponse> {
        let _permit = permit;
        let timeout = Duration::from_secs(self.config.timeout_secs.max(1));
        let idle_timeout = self
            .config
            .idle_timeout_secs
            .map(|secs| Duration::from_secs(secs.max(1)));
        let SpawnedPty {
            mut data,
            exit,
            guard,
            full_prompt,
            attachments: _attachments,
            spill: _spill,
        } = self.spawn_session(request).await?;

        tokio::pin!(exit);
        let collect: Result<(Vec<u8>, PtyExit)> = async {
            let mut raw = Vec::new();
            let total_deadline = tokio::time::Instant::now() + timeout;
            let mut last_activity = tokio::time::Instant::now();

            loop {
                let effective_deadline = match idle_timeout {
                    Some(idle) => total_deadline.min(last_activity + idle),
                    None => total_deadline,
                };
                let item = tokio::select! {
                    _ = request.cancel.cancelled() => return Err(GatewayError::ClientDisconnected),
                    _ = tokio::time::sleep_until(effective_deadline) => {
                        let now = tokio::time::Instant::now();
                        let fired = if now >= total_deadline {
                            timeout
                        } else {
                            idle_timeout.unwrap_or(timeout)
                        };
                        return Err(GatewayError::Timeout(fired));
                    }
                    item = data.recv() => item,
                };
                let Some(item) = item else {
                    break;
                };
                let bytes = item?;
                if !bytes.is_empty() {
                    last_activity = tokio::time::Instant::now();
                }
                if raw.len() + bytes.len() > MAX_OUTPUT_BYTES {
                    return Err(GatewayError::Upstream(format!(
                        "backend {} exceeded the {} byte output limit",
                        self.config.id, MAX_OUTPUT_BYTES
                    )));
                }
                raw.extend_from_slice(&bytes);
            }

            let exit = tokio::select! {
                _ = request.cancel.cancelled() => return Err(GatewayError::ClientDisconnected),
                _ = tokio::time::sleep_until(total_deadline) => return Err(GatewayError::Timeout(timeout)),
                res = &mut exit => res.map_err(|_| GatewayError::Upstream("pty exit channel closed".into()))?,
            };
            Ok((raw, exit))
        }
        .await;

        match collect {
            Ok((raw, exit)) => {
                guard.disarm();
                metrics::record_child_exit(&self.config.id, &exit.metric_code());
                let text = self.output_filters.apply(&strip_ansi(&raw));
                if !exit.success {
                    return Err(classify_cli_failure(
                        &self.config.id,
                        &format!("exited with {}", exit.label()),
                        &text,
                    ));
                }
                capture_session_id(
                    self.config.session_id_regex.as_deref(),
                    "",
                    &text,
                    &request.session_capture,
                );
                Ok(DriverResponse::Complete(finalize_subprocess_completion(
                    &request.model,
                    &full_prompt,
                    &text,
                    self.config.output_mode,
                    !request.tools.is_empty(),
                    request.response_format.as_ref(),
                )?))
            }
            Err(error) => {
                guard.terminate();
                drop(data);
                let _ = tokio::time::timeout(Duration::from_secs(2), &mut exit).await;
                metrics::record_child_exit(&self.config.id, "killed");
                Err(error)
            }
        }
    }

    async fn stream(
        &self,
        request: ExecutionRequest,
        permit: PermitGuard,
    ) -> Result<DriverResponse> {
        let timeout = Duration::from_secs(self.config.timeout_secs.max(1));
        let idle_timeout = self
            .config
            .idle_timeout_secs
            .map(|secs| Duration::from_secs(secs.max(1)));
        let SpawnedPty {
            mut data,
            exit,
            guard,
            full_prompt,
            attachments: _attachments,
            spill: _spill,
        } = self.spawn_session(&request).await?;

        let backend = self.config.id.clone();
        let output_mode = self.config.output_mode;
        let session_id_regex = self.config.session_id_regex.clone();
        let output_filters = self.output_filters.clone();
        let parse_tools = !request.tools.is_empty();
        let response_format: Option<ResponseFormat> = request.response_format.clone();
        let model = request.model.clone();
        let cancel = request.cancel.clone();
        let session_capture = request.session_capture.clone();
        let (sender, receiver) = mpsc::channel::<Result<StreamChunk>>(32);

        tokio::spawn(async move {
            let _permit = permit;
            let _attachments = _attachments;
            let _spill = _spill;

            let total_deadline = tokio::time::Instant::now() + timeout;
            let mut last_activity = tokio::time::Instant::now();
            let mut utf8_decoder = Utf8ChunkDecoder::new();
            let mut stripper = AnsiStripper::new();
            let mut filter = SubprocessStreamFilter::new(parse_tools, response_format.clone());
            let mut out_filter = StreamOutputFilterAdapter::new(output_filters);
            let mut json_buffer = String::new();
            let mut line_buffer = String::new();
            let mut produced = String::new();
            tokio::pin!(exit);

            loop {
                let effective_deadline = match idle_timeout {
                    Some(idle) => total_deadline.min(last_activity + idle),
                    None => total_deadline,
                };
                let item = tokio::select! {
                    _ = cancel.cancelled() => {
                        guard.terminate();
                        drop(data);
                        let _ = tokio::time::timeout(Duration::from_secs(2), &mut exit).await;
                        let _ = sender.send(Err(GatewayError::ClientDisconnected)).await;
                        return;
                    }
                    _ = tokio::time::sleep_until(effective_deadline) => {
                        let fired = if tokio::time::Instant::now() >= total_deadline {
                            timeout
                        } else {
                            idle_timeout.unwrap_or(timeout)
                        };
                        guard.terminate();
                        drop(data);
                        let _ = tokio::time::timeout(Duration::from_secs(2), &mut exit).await;
                        metrics::record_child_exit(&backend, "killed");
                        let _ = sender.send(Err(GatewayError::Timeout(fired))).await;
                        return;
                    }
                    item = data.recv() => item,
                };
                let Some(item) = item else {
                    break;
                };
                let bytes = match item {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        guard.terminate();
                        drop(data);
                        let _ = tokio::time::timeout(Duration::from_secs(2), &mut exit).await;
                        let _ = sender.send(Err(error)).await;
                        return;
                    }
                };
                if !bytes.is_empty() {
                    last_activity = tokio::time::Instant::now();
                }
                let utf8 = utf8_decoder.feed(&bytes);
                let stripped = stripper.feed(utf8.as_bytes());
                if stripped.is_empty() {
                    continue;
                }
                if produced.len() + stripped.len() > MAX_OUTPUT_BYTES {
                    guard.terminate();
                    drop(data);
                    let _ = tokio::time::timeout(Duration::from_secs(2), &mut exit).await;
                    let _ = sender
                        .send(Err(GatewayError::Upstream(format!(
                            "backend {backend} exceeded the {MAX_OUTPUT_BYTES} byte output limit"
                        ))))
                        .await;
                    return;
                }
                produced.push_str(&stripped);
                match output_mode {
                    OutputMode::Text => {
                        let filtered = out_filter.push(&stripped);
                        if !filtered.is_empty() {
                            let chunks = match filter.push(&filtered) {
                                Ok(chunks) => chunks,
                                Err(error) => {
                                    guard.terminate();
                                    drop(data);
                                    let _ = tokio::time::timeout(Duration::from_secs(2), &mut exit)
                                        .await;
                                    let _ = sender.send(Err(error)).await;
                                    return;
                                }
                            };
                            if !dispatch_stream_chunks(&sender, chunks).await {
                                guard.terminate();
                                drop(data);
                                let _ =
                                    tokio::time::timeout(Duration::from_secs(2), &mut exit).await;
                                return;
                            }
                        }
                    }
                    OutputMode::Json => json_buffer.push_str(&stripped),
                    OutputMode::JsonLines => {
                        line_buffer.push_str(&stripped);
                        let max_line = crate::driver::cli_pipe::MAX_LINE_BUFFER_BYTES;
                        if line_buffer.split('\n').any(|seg| seg.len() > max_line) {
                            guard.terminate();
                            drop(data);
                            let _ = tokio::time::timeout(Duration::from_secs(2), &mut exit).await;
                            let _ = sender
                                .send(Err(GatewayError::ProcessFailure(format!(
                                    "backend {backend} exceeded {max_line} byte single-line limit"
                                ))))
                                .await;
                            return;
                        }
                        while let Some(position) = line_buffer.find('\n') {
                            let line: String = line_buffer.drain(..=position).collect();
                            let line = line.trim_end_matches(['\n', '\r']);
                            if line.trim().is_empty() {
                                continue;
                            }
                            let delta = out_filter
                                .apply_raw(&crate::driver::cli_pipe::jsonl_line_to_delta(line));
                            if !delta.is_empty()
                                && sender.send(Ok(StreamChunk::delta(delta))).await.is_err()
                            {
                                guard.terminate();
                                drop(data);
                                let _ =
                                    tokio::time::timeout(Duration::from_secs(2), &mut exit).await;
                                return;
                            }
                        }
                    }
                }
            }

            let tail_utf8 = utf8_decoder.finish();
            let mut tail_combined = String::new();
            if !tail_utf8.is_empty() {
                let stripped_tail = stripper.feed(tail_utf8.as_bytes());
                if !stripped_tail.is_empty() {
                    produced.push_str(&stripped_tail);
                    match output_mode {
                        OutputMode::Text => {
                            tail_combined.push_str(&out_filter.push(&stripped_tail));
                        }
                        OutputMode::Json => json_buffer.push_str(&stripped_tail),
                        OutputMode::JsonLines => line_buffer.push_str(&stripped_tail),
                    }
                }
            }
            if output_mode == OutputMode::Text {
                tail_combined.push_str(&out_filter.finish());
                if !tail_combined.is_empty() {
                    let chunks = match filter.push(&tail_combined) {
                        Ok(chunks) => chunks,
                        Err(error) => {
                            guard.terminate();
                            drop(data);
                            let _ = tokio::time::timeout(Duration::from_secs(2), &mut exit).await;
                            let _ = sender.send(Err(error)).await;
                            return;
                        }
                    };
                    if !dispatch_stream_chunks(&sender, chunks).await {
                        guard.terminate();
                        drop(data);
                        let _ = tokio::time::timeout(Duration::from_secs(2), &mut exit).await;
                        return;
                    }
                }
            }

            let exit = tokio::select! {
                _ = cancel.cancelled() => {
                    guard.terminate();
                    drop(data);
                    let _ = tokio::time::timeout(Duration::from_secs(2), &mut exit).await;
                    let _ = sender.send(Err(GatewayError::ClientDisconnected)).await;
                    return;
                }
                _ = tokio::time::sleep_until(total_deadline) => {
                    guard.terminate();
                    drop(data);
                    let _ = tokio::time::timeout(Duration::from_secs(2), &mut exit).await;
                    metrics::record_child_exit(&backend, "killed");
                    let _ = sender.send(Err(GatewayError::Timeout(timeout))).await;
                    return;
                }
                res = &mut exit => res.map_err(|_| GatewayError::Upstream("pty exit channel closed".into())),
            };
            let exit = match exit {
                Ok(exit) => exit,
                Err(error) => {
                    guard.terminate();
                    let _ = sender.send(Err(error)).await;
                    return;
                }
            };

            metrics::record_child_exit(&backend, &exit.metric_code());
            if exit.success {
                guard.disarm();
                capture_session_id(session_id_regex.as_deref(), "", &produced, &session_capture);

                if output_mode == OutputMode::Json {
                    match crate::driver::cli_pipe::parse_json_content(&json_buffer) {
                        Ok(content) => {
                            let content = if let Some(format) = response_format.as_ref() {
                                match crate::driver::tool_bridge::enforce_response_format(
                                    &content, format,
                                ) {
                                    Ok(valid) => valid,
                                    Err(error) => {
                                        let _ = sender.send(Err(error)).await;
                                        return;
                                    }
                                }
                            } else {
                                content
                            };
                            if !content.is_empty()
                                && sender.send(Ok(StreamChunk::delta(content))).await.is_err()
                            {
                                return;
                            }
                        }
                        Err(error) => {
                            let _ = sender.send(Err(error)).await;
                            return;
                        }
                    }
                } else if output_mode == OutputMode::JsonLines && !line_buffer.trim().is_empty() {
                    let delta = crate::driver::cli_pipe::jsonl_line_to_delta(line_buffer.trim());
                    if !delta.is_empty()
                        && sender.send(Ok(StreamChunk::delta(delta))).await.is_err()
                    {
                        return;
                    }
                }

                if output_mode == OutputMode::Text {
                    let _ = finish_subprocess_text_stream(
                        &sender,
                        &mut filter,
                        &model,
                        &full_prompt,
                        &produced,
                    )
                    .await;
                } else {
                    let _ = emit_subprocess_tail(
                        &sender,
                        &model,
                        &full_prompt,
                        &produced,
                        output_mode,
                        parse_tools,
                        response_format.as_ref(),
                    )
                    .await;
                }
            } else {
                guard.disarm();
                let _ = sender
                    .send(Err(classify_cli_failure(
                        &backend,
                        &format!("exited with {}", exit.label()),
                        &produced,
                    )))
                    .await;
            }
        });

        let stream: ChunkStream = Box::pin(ReceiverStream::new(receiver));
        Ok(DriverResponse::Stream(stream))
    }
}

fn trim_rolling_window(window: &mut String, max_bytes: usize) {
    if window.len() <= max_bytes {
        return;
    }
    let mut start = window.len().saturating_sub(max_bytes);
    while start < window.len() && !window.is_char_boundary(start) {
        start += 1;
    }
    window.drain(..start);
}

#[async_trait]
impl Driver for PtyRunnerDriver {
    fn id(&self) -> &str {
        &self.config.id
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Pty
    }

    fn max_concurrency(&self) -> usize {
        self.config.max_concurrency
    }

    async fn execute(&self, request: ExecutionRequest) -> Result<DriverResponse> {
        let permit = PermitGuard::acquire(
            self.semaphore.clone(),
            &self.config.id,
            self.queue_timeout,
            &request.cancel,
        )
        .await?;
        if request.stream {
            self.stream(request, permit).await
        } else {
            self.complete(&request, permit).await
        }
    }
}
