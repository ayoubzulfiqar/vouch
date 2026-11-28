//! Headless stdin/stdout subprocess driver.
//!
//! The child is always placed in its own process group. A
//! [`ProcessGroupGuard`] is moved into the task that owns the child, so a
//! timeout, I/O error, or dropped SSE receiver tears down the whole group
//! (including grandchildren) instead of leaking processes.

use std::{
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use serde_json::Value;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::{Child, Command},
    sync::{Notify, Semaphore, mpsc},
};
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    config::{AttachmentsConfig, BackendConfig, BackendKind, DefaultsConfig, OutputMode, ReadMode},
    driver::{
        AnsiStripper, ChatMessage, ChunkStream, Driver, DriverResponse, EmbeddingRequest,
        EmbeddingResult, ExecutionRequest, OutputFilterSet, PermitGuard, ProcessGroupGuard,
        ResponseFormat, StreamChunk, StreamOutputFilterAdapter, Usage, Utf8ChunkDecoder,
        WorkspaceMode,
        attachments::{PromptSpillFile, TempAttachmentGuard, materialize_prompt_with_config},
        capture_session_id, classify_cli_failure, dispatch_stream_chunks, emit_subprocess_tail,
        extract_content, finalize_subprocess_completion, finish_subprocess_text_stream, io_error,
        strip_ansi,
        tool_bridge::SubprocessStreamFilter,
        truncate,
    },
    error::{GatewayError, Result},
    metrics, telemetry,
};

/// Maximum buffered stdout for non-streaming executions.
const MAX_OUTPUT_BYTES: usize = 16 * 1024 * 1024;
/// Maximum captured stderr used in diagnostics.
const MAX_STDERR_BYTES: usize = 8 * 1024;

/// Tracks the timestamp of the last byte observed on child stdout or stderr so
/// the idle-stall watchdog resets on any subprocess activity.
#[derive(Debug)]
struct ActivityTracker {
    base: tokio::time::Instant,
    last_offset_ms: AtomicU64,
    notify: Notify,
}

impl ActivityTracker {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            base: tokio::time::Instant::now(),
            last_offset_ms: AtomicU64::new(0),
            notify: Notify::new(),
        })
    }

    fn touch(&self) {
        let elapsed_ms = tokio::time::Instant::now()
            .saturating_duration_since(self.base)
            .as_millis() as u64;
        self.last_offset_ms.store(elapsed_ms, Ordering::Relaxed);
        self.notify.notify_waiters();
    }

    fn deadline(&self, idle: Duration) -> tokio::time::Instant {
        let offset = Duration::from_millis(self.last_offset_ms.load(Ordering::Relaxed));
        self.base + offset + idle
    }
}

struct SpawnedCli {
    child: Child,
    guard: ProcessGroupGuard,
    full_prompt: String,
    _attachments: TempAttachmentGuard,
    _spill: Option<PromptSpillFile>,
}

/// Headless CLI pipe driver.
#[derive(Debug)]
pub struct CliPipeDriver {
    config: BackendConfig,
    attachments: AttachmentsConfig,
    output_filters: OutputFilterSet,
    semaphore: Arc<Semaphore>,
    queue_timeout: Duration,
}

impl CliPipeDriver {
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

    async fn spawn(&self, request: &ExecutionRequest) -> Result<SpawnedCli> {
        let command_path = self.config.command.as_deref().ok_or_else(|| {
            GatewayError::Config(format!("backend {} has no command", self.config.id))
        })?;

        let (full_prompt, attachments) =
            materialize_prompt_with_config(request, &self.attachments).await?;
        let spill = if !self.config.stdin
            && full_prompt.len() > self.config.max_inline_prompt_bytes.max(1)
        {
            Some(PromptSpillFile::create(&full_prompt)?)
        } else {
            None
        };
        let arg_prompt = match &spill {
            Some(file) => file.prompt_reference(),
            None => full_prompt.clone(),
        };

        let arg_templates = if request.is_resuming() && !self.config.resume_args.is_empty() {
            &self.config.resume_args
        } else {
            &self.config.args
        };

        let mut command = Command::new(command_path);
        for arg in arg_templates {
            command.arg(render_arg(arg, &arg_prompt, request));
        }
        command.envs(&self.config.env);
        command.envs(&request.profile_env);
        command.env("VOUCH_REQUEST_ID", &request.request_id);
        command.env("VOUCH_SESSION_ID", &request.session_key);
        command.env("VOUCH_MODEL", &request.model);
        if let Some(backend_session_id) = &request.backend_session_id {
            command.env("VOUCH_BACKEND_SESSION_ID", backend_session_id);
        }
        if let Some(profile_id) = &request.profile_id {
            command.env("VOUCH_PROFILE_ID", profile_id);
        }
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        #[cfg(unix)]
        {
            command.process_group(0);
        }
        if let Some(workdir) = request.workdir.as_ref().or(self.config.workdir.as_ref()) {
            command.current_dir(workdir);
        }
        let child = command.spawn().map_err(|error| {
            GatewayError::Upstream(format!(
                "cannot spawn backend {} ({}): {error}",
                self.config.id,
                command_name(&self.config)
            ))
        })?;
        let pid = child.id().ok_or_else(|| {
            GatewayError::Internal(format!("backend {} did not report a pid", self.config.id))
        })?;
        let guard = ProcessGroupGuard::new(pid, self.config.id.clone());
        Ok(SpawnedCli {
            child,
            guard,
            full_prompt,
            _attachments: attachments,
            _spill: spill,
        })
    }

    fn write_stdin(&self, child: &mut Child, prompt: String) -> tokio::task::JoinHandle<()> {
        let stdin = child.stdin.take();
        if !self.config.stdin {
            drop(stdin);
            return tokio::spawn(async {});
        }
        tokio::spawn(async move {
            let Some(mut stdin) = stdin else {
                return;
            };
            if stdin.write_all(prompt.as_bytes()).await.is_err() {
                return;
            }
            if !prompt.ends_with('\n') && stdin.write_all(b"\n").await.is_err() {
                return;
            }
            let _ = stdin.shutdown().await;
        })
    }

    fn drain_stderr(
        &self,
        child: &mut Child,
        tracker: Arc<ActivityTracker>,
    ) -> tokio::task::JoinHandle<String> {
        let stderr = child.stderr.take();
        tokio::spawn(async move {
            let Some(mut stderr) = stderr else {
                return String::new();
            };
            let mut captured = Vec::new();
            let mut buffer = [0u8; 4096];
            loop {
                match stderr.read(&mut buffer).await {
                    Ok(0) => break,
                    Ok(read) => {
                        tracker.touch();
                        if captured.len() < MAX_STDERR_BYTES {
                            let remaining = MAX_STDERR_BYTES - captured.len();
                            captured.extend_from_slice(&buffer[..read.min(remaining)]);
                        }
                    }
                    Err(_) => break,
                }
            }
            String::from_utf8_lossy(&captured).into_owned()
        })
    }

    async fn execute_complete(
        &self,
        request: ExecutionRequest,
        permit: PermitGuard,
    ) -> Result<DriverResponse> {
        let _permit = permit;
        let timeout = Duration::from_secs(self.config.timeout_secs.max(1));
        let idle_timeout = self
            .config
            .idle_timeout_secs
            .map(|secs| Duration::from_secs(secs.max(1)));
        let SpawnedCli {
            mut child,
            guard,
            full_prompt,
            _attachments,
            _spill,
        } = self.spawn(&request).await?;

        let tracker = ActivityTracker::new();
        let stdin_task = self.write_stdin(&mut child, full_prompt.clone());
        let stderr_task = self.drain_stderr(&mut child, tracker.clone());

        let outcome: Result<(Vec<u8>, std::process::ExitStatus)> = async {
            let mut stdout = child.stdout.take().ok_or_else(|| {
                GatewayError::Internal(format!("backend {} has no stdout", self.config.id))
            })?;
            let mut buffer = Vec::new();
            let mut chunk = [0u8; 8192];
            let total_deadline = tokio::time::Instant::now() + timeout;

            loop {
                let effective_deadline = effective_deadline(total_deadline, idle_timeout, &tracker);
                let woke = tokio::select! {
                    _ = request.cancel.cancelled() => return Err(GatewayError::ClientDisconnected),
                    _ = tokio::time::sleep_until(effective_deadline) => {
                        if let Some(fired) = expired_timeout(total_deadline, timeout, idle_timeout, &tracker) {
                            return Err(GatewayError::Timeout(fired));
                        }
                        continue;
                    }
                    _ = tracker.notify.notified(), if idle_timeout.is_some() => continue,
                    res = stdout.read(&mut chunk) => res?,
                };
                if woke == 0 {
                    break;
                }
                tracker.touch();
                if buffer.len() + woke > MAX_OUTPUT_BYTES {
                    return Err(GatewayError::Upstream(format!(
                        "backend {} exceeded the {} byte output limit",
                        self.config.id, MAX_OUTPUT_BYTES
                    )));
                }
                buffer.extend_from_slice(&chunk[..woke]);
            }

            let status = loop {
                let effective_deadline = effective_deadline(total_deadline, idle_timeout, &tracker);
                tokio::select! {
                    _ = request.cancel.cancelled() => return Err(GatewayError::ClientDisconnected),
                    _ = tokio::time::sleep_until(effective_deadline) => {
                        if let Some(fired) = expired_timeout(total_deadline, timeout, idle_timeout, &tracker) {
                            return Err(GatewayError::Timeout(fired));
                        }
                    }
                    _ = tracker.notify.notified(), if idle_timeout.is_some() => {}
                    status = child.wait() => break status?,
                }
            };
            Ok((buffer, status))
        }
        .await;

        match outcome {
            Ok((buffer, status)) => {
                guard.disarm();
                let _ = tokio::time::timeout(Duration::from_secs(1), stdin_task).await;
                let stderr = tokio::time::timeout(Duration::from_secs(1), stderr_task)
                    .await
                    .ok()
                    .and_then(std::result::Result::ok)
                    .unwrap_or_default();
                record_exit(&self.config.id, &status);
                if !status.success() {
                    return Err(classify_cli_failure(
                        &self.config.id,
                        &format!("exited with {}", status_label(&status)),
                        &stderr,
                    ));
                }
                let raw = self.output_filters.apply(&strip_ansi(&buffer));
                capture_session_id(
                    self.config.session_id_regex.as_deref(),
                    &stderr,
                    &raw,
                    &request.session_capture,
                );
                Ok(DriverResponse::Complete(finalize_subprocess_completion(
                    &request.model,
                    &full_prompt,
                    &raw,
                    self.config.output_mode,
                    !request.tools.is_empty(),
                    request.response_format.as_ref(),
                )?))
            }
            Err(error) => {
                guard.terminate();
                let _ = tokio::time::timeout(Duration::from_secs(2), child.wait()).await;
                let _ = stdin_task.await;
                if matches!(error, GatewayError::Timeout(_)) {
                    metrics::record_child_exit(&self.config.id, "killed");
                } else {
                    metrics::record_child_exit(&self.config.id, "error");
                }
                Err(error)
            }
        }
    }

    async fn execute_stream(
        &self,
        request: ExecutionRequest,
        permit: PermitGuard,
    ) -> Result<DriverResponse> {
        let timeout = Duration::from_secs(self.config.timeout_secs.max(1));
        let idle_timeout = self
            .config
            .idle_timeout_secs
            .map(|secs| Duration::from_secs(secs.max(1)));
        let SpawnedCli {
            mut child,
            guard,
            full_prompt,
            _attachments,
            _spill,
        } = self.spawn(&request).await?;

        let tracker = ActivityTracker::new();
        let stdin_task = self.write_stdin(&mut child, full_prompt.clone());
        let stderr_task = self.drain_stderr(&mut child, tracker.clone());
        let backend = self.config.id.clone();
        let output_mode = self.config.output_mode;
        let read_mode = self.config.read_mode;
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

            let mut stdout = match child.stdout.take() {
                Some(stdout) => stdout,
                None => {
                    let _ = sender
                        .send(Err(GatewayError::Internal(format!(
                            "backend {backend} has no stdout"
                        ))))
                        .await;
                    guard.terminate();
                    let _ = child.wait().await;
                    return;
                }
            };
            let total_deadline = tokio::time::Instant::now() + timeout;
            let mut utf8_decoder = Utf8ChunkDecoder::new();
            let mut stripper = AnsiStripper::new();
            let mut raw_line_carry: Vec<u8> = Vec::new();
            let mut filter = SubprocessStreamFilter::new(parse_tools, response_format.clone());
            let mut out_filter = StreamOutputFilterAdapter::new(output_filters);
            let mut json_buffer = String::new();
            let mut line_buffer = String::new();
            let mut produced = String::new();
            let mut buffer = [0u8; 8192];

            loop {
                let effective_deadline = effective_deadline(total_deadline, idle_timeout, &tracker);
                let read = tokio::select! {
                    _ = cancel.cancelled() => {
                        guard.terminate();
                        let _ = child.wait().await;
                        let _ = sender.send(Err(GatewayError::ClientDisconnected)).await;
                        return;
                    }
                    _ = tokio::time::sleep_until(effective_deadline) => {
                        let Some(fired) = expired_timeout(total_deadline, timeout, idle_timeout, &tracker) else {
                            continue;
                        };
                        guard.terminate();
                        let _ = child.wait().await;
                        metrics::record_child_exit(&backend, "killed");
                        let _ = sender.send(Err(GatewayError::Timeout(fired))).await;
                        return;
                    }
                    _ = tracker.notify.notified(), if idle_timeout.is_some() => continue,
                    result = stdout.read(&mut buffer) => result,
                };
                match read {
                    Ok(0) => break,
                    Ok(read) => {
                        tracker.touch();
                        let utf8 = match read_mode {
                            ReadMode::Chunk => utf8_decoder.feed(&buffer[..read]),
                            ReadMode::Line => {
                                raw_line_carry.extend_from_slice(&buffer[..read]);
                                if let Some(last_nl) =
                                    raw_line_carry.iter().rposition(|&b| b == b'\n')
                                {
                                    let ready: Vec<u8> = raw_line_carry.drain(..=last_nl).collect();
                                    utf8_decoder.feed(&ready)
                                } else {
                                    continue;
                                }
                            }
                        };
                        let stripped = stripper.feed(utf8.as_bytes());
                        if stripped.is_empty() {
                            continue;
                        }
                        if produced.len() + stripped.len() > MAX_OUTPUT_BYTES {
                            guard.terminate();
                            let _ = child.wait().await;
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
                                if filtered.is_empty() {
                                    continue;
                                }
                                let chunks = filter.push(&filtered);
                                if !dispatch_stream_chunks(&sender, chunks).await {
                                    guard.terminate();
                                    let _ = child.wait().await;
                                    return;
                                }
                            }
                            OutputMode::Json => json_buffer.push_str(&stripped),
                            OutputMode::JsonLines => {
                                line_buffer.push_str(&stripped);
                                while let Some(position) = line_buffer.find('\n') {
                                    let line: String = line_buffer.drain(..=position).collect();
                                    let line = line.trim_end_matches(['\n', '\r']);
                                    if line.trim().is_empty() {
                                        continue;
                                    }
                                    let delta = out_filter.apply_raw(&jsonl_line_to_delta(line));
                                    if !delta.is_empty()
                                        && sender.send(Ok(StreamChunk::delta(delta))).await.is_err()
                                    {
                                        guard.terminate();
                                        let _ = child.wait().await;
                                        return;
                                    }
                                }
                            }
                        }
                    }
                    Err(error) => {
                        guard.terminate();
                        let _ = child.wait().await;
                        metrics::record_child_exit(&backend, "error");
                        let _ = sender
                            .send(Err(io_error("cannot read backend stdout", error)))
                            .await;
                        return;
                    }
                }
            }

            // Flush any buffered bytes held for ReadMode::Line or split UTF-8 sequences at EOF.
            let mut tail_utf8 = String::new();
            if !raw_line_carry.is_empty() {
                tail_utf8.push_str(&utf8_decoder.feed(&raw_line_carry));
            }
            tail_utf8.push_str(&utf8_decoder.finish());
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
                    let chunks = filter.push(&tail_combined);
                    if !dispatch_stream_chunks(&sender, chunks).await {
                        guard.terminate();
                        let _ = child.wait().await;
                        return;
                    }
                }
            }

            let status = loop {
                let effective_deadline = effective_deadline(total_deadline, idle_timeout, &tracker);
                tokio::select! {
                    _ = cancel.cancelled() => {
                        guard.terminate();
                        let _ = child.wait().await;
                        let _ = sender.send(Err(GatewayError::ClientDisconnected)).await;
                        return;
                    }
                    _ = tokio::time::sleep_until(effective_deadline) => {
                        if let Some(fired) = expired_timeout(total_deadline, timeout, idle_timeout, &tracker) {
                            guard.terminate();
                            let _ = child.wait().await;
                            metrics::record_child_exit(&backend, "killed");
                            let _ = sender.send(Err(GatewayError::Timeout(fired))).await;
                            return;
                        }
                    }
                    _ = tracker.notify.notified(), if idle_timeout.is_some() => {}
                    status = child.wait() => break status,
                }
            };
            let stderr = stderr_task.await.unwrap_or_default();
            let _ = tokio::time::timeout(Duration::from_secs(1), stdin_task).await;

            match status {
                Ok(status) if status.success() => {
                    guard.disarm();
                    record_exit(&backend, &status);
                    capture_session_id(
                        session_id_regex.as_deref(),
                        &stderr,
                        &produced,
                        &session_capture,
                    );

                    if output_mode == OutputMode::Json {
                        match parse_json_content(&json_buffer) {
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
                    } else if output_mode == OutputMode::JsonLines && !line_buffer.trim().is_empty()
                    {
                        let delta = jsonl_line_to_delta(line_buffer.trim());
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
                }
                Ok(status) => {
                    record_exit(&backend, &status);
                    guard.disarm();
                    let _ = sender
                        .send(Err(classify_cli_failure(
                            &backend,
                            &format!("exited with {}", status_label(&status)),
                            &stderr,
                        )))
                        .await;
                }
                Err(error) => {
                    let _ = sender
                        .send(Err(io_error("cannot wait for backend", error)))
                        .await;
                }
            }
        });

        let stream: ChunkStream = Box::pin(ReceiverStream::new(receiver));
        Ok(DriverResponse::Stream(stream))
    }
}

fn effective_deadline(
    total_deadline: tokio::time::Instant,
    idle_timeout: Option<Duration>,
    tracker: &ActivityTracker,
) -> tokio::time::Instant {
    match idle_timeout {
        Some(idle) => total_deadline.min(tracker.deadline(idle)),
        None => total_deadline,
    }
}

fn expired_timeout(
    total_deadline: tokio::time::Instant,
    total_timeout: Duration,
    idle_timeout: Option<Duration>,
    tracker: &ActivityTracker,
) -> Option<Duration> {
    let now = tokio::time::Instant::now();
    if now >= total_deadline {
        return Some(total_timeout);
    }
    if let Some(idle) = idle_timeout
        && now >= tracker.deadline(idle)
    {
        return Some(idle);
    }
    None
}

#[async_trait]
impl Driver for CliPipeDriver {
    fn id(&self) -> &str {
        &self.config.id
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Cli
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
            self.execute_stream(request, permit).await
        } else {
            self.execute_complete(request, permit).await
        }
    }

    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResult> {
        let prompt = if request.inputs.len() == 1 {
            request.inputs[0].clone()
        } else {
            serde_json::to_string(&request.inputs).unwrap_or_default()
        };
        let exec_req = ExecutionRequest {
            request_id: request.request_id,
            session_key: "embeddings".into(),
            model: request.model.clone(),
            system: None,
            messages: vec![ChatMessage::text("user", prompt.clone())],
            prompt: prompt.clone(),
            stream: false,
            temperature: None,
            max_tokens: None,
            workspace_mode: WorkspaceMode::None,
            workdir: None,
            cancel: request.cancel,
            tools: Vec::new(),
            tool_choice: None,
            response_format: None,
            backend_session_id: None,
            delta_messages: None,
            session_capture: Arc::new(parking_lot::Mutex::new(None)),
            trace_context: request.trace_context,
            profile_id: None,
            profile_env: std::collections::BTreeMap::new(),
            allowed_backends: None,
            extra: Default::default(),
        };
        let permit = PermitGuard::acquire(
            self.semaphore.clone(),
            &self.config.id,
            self.queue_timeout,
            &exec_req.cancel,
        )
        .await?;
        let DriverResponse::Complete(completion) = self.execute_complete(exec_req, permit).await?
        else {
            return Err(GatewayError::Internal(
                "cli_pipe complete unexpectedly returned stream".into(),
            ));
        };
        let parsed: Value = serde_json::from_str(completion.content.trim()).map_err(|error| {
            GatewayError::Upstream(format!(
                "backend {} embeddings output is not valid JSON vector(s): {error}",
                self.config.id
            ))
        })?;
        let embeddings = parse_cli_embeddings(&self.config.id, &parsed, request.inputs.len())?;
        Ok(EmbeddingResult {
            embeddings,
            usage: Usage::new(crate::usage::count_tokens(&request.model, &prompt), 0),
            model: request.model,
        })
    }
}

fn parse_cli_embeddings(
    backend_id: &str,
    value: &Value,
    expected_len: usize,
) -> Result<Vec<Vec<f32>>> {
    if let Some(arr) = value.as_array() {
        if arr.is_empty() {
            return Ok(Vec::new());
        }
        if arr[0].is_number() {
            let single: Vec<f32> = arr
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();
            return Ok(vec![single]);
        }
        if arr[0].is_array() {
            let mut out = Vec::with_capacity(arr.len());
            for row in arr {
                let Some(items) = row.as_array() else {
                    return Err(GatewayError::Upstream(format!(
                        "backend {backend_id} returned mixed embedding vector array"
                    )));
                };
                out.push(
                    items
                        .iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect(),
                );
            }
            return Ok(out);
        }
    }
    if let Some(data) = value.get("data").and_then(Value::as_array) {
        let out: Vec<Vec<f32>> = data
            .iter()
            .filter_map(|item| item.get("embedding").and_then(Value::as_array))
            .map(|vec| {
                vec.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .collect();
        if !out.is_empty() || expected_len == 0 {
            return Ok(out);
        }
    }
    if let Some(vecs) = value.get("embeddings").and_then(Value::as_array) {
        let out: Vec<Vec<f32>> = vecs
            .iter()
            .filter_map(Value::as_array)
            .map(|vec| {
                vec.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .collect();
        if !out.is_empty() || expected_len == 0 {
            return Ok(out);
        }
    }
    Err(GatewayError::Upstream(format!(
        "backend {backend_id} embeddings output must be a JSON vector or array of vectors"
    )))
}

fn render_arg(template: &str, prompt: &str, request: &ExecutionRequest) -> String {
    let backend_session_id = request
        .backend_session_id
        .as_deref()
        .unwrap_or(&request.session_key);
    let profile_id = request.profile_id.as_deref().unwrap_or("");
    template
        .replace("{prompt}", prompt)
        .replace("{model}", &request.model)
        .replace("{session_id}", &request.session_key)
        .replace("{request_id}", &request.request_id)
        .replace("{backend_session_id}", backend_session_id)
        .replace("{profile_id}", profile_id)
}

fn command_name(config: &BackendConfig) -> &str {
    config.command.as_deref().unwrap_or("<none>")
}

fn record_exit(backend: &str, status: &std::process::ExitStatus) {
    let code = status
        .code()
        .map(|code| code.to_string())
        .unwrap_or_else(|| "signal".to_string());
    metrics::record_child_exit(backend, &code);
}

fn status_label(status: &std::process::ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("exit code {code}"),
        None => "a signal".to_string(),
    }
}

pub(crate) fn parse_json_content(raw: &str) -> Result<String> {
    let value: Value = serde_json::from_str(raw.trim()).map_err(|error| {
        GatewayError::Upstream(format!("backend emitted invalid JSON: {error}"))
    })?;
    if let Some(error) = crate::driver::extract_upstream_error(&value) {
        return Err(GatewayError::Upstream(format!(
            "backend reported an error: {}",
            truncate(&telemetry::redact_text(&error), 500)
        )));
    }
    extract_content(&value).ok_or_else(|| {
        GatewayError::Upstream("backend JSON did not contain assistant content".into())
    })
}

pub(crate) fn parse_json_lines(raw: &str) -> String {
    let mut output = String::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        output.push_str(&jsonl_line_to_delta(line));
    }
    output
}

pub(crate) fn jsonl_line_to_delta(line: &str) -> String {
    match serde_json::from_str::<Value>(line.trim()) {
        Ok(value) => {
            let (delta, _, _) = crate::driver::extract_delta(&value);
            delta
                .or_else(|| extract_content(&value))
                .unwrap_or_default()
        }
        Err(_) => line.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_argument_templates() {
        let request = ExecutionRequest {
            request_id: "req".into(),
            session_key: "sess".into(),
            model: "coding/default".into(),
            system: None,
            messages: Vec::new(),
            prompt: "hello".into(),
            stream: false,
            temperature: None,
            max_tokens: None,
            workspace_mode: crate::driver::WorkspaceMode::Auto,
            workdir: None,
            cancel: tokio_util::sync::CancellationToken::new(),
            tools: Vec::new(),
            tool_choice: None,
            response_format: None,
            backend_session_id: Some("bsess_42".into()),
            delta_messages: None,
            session_capture: Arc::new(parking_lot::Mutex::new(None)),
            trace_context: None,
            profile_id: None,
            profile_env: std::collections::BTreeMap::new(),
            allowed_backends: None,
            extra: serde_json::Map::new(),
        };
        assert_eq!(render_arg("--p={prompt}", "hello", &request), "--p=hello");
        assert_eq!(render_arg("{model}", "hello", &request), "coding/default");
        assert_eq!(
            render_arg("--resume={backend_session_id}", "hello", &request),
            "--resume=bsess_42"
        );
    }

    #[test]
    fn parses_json_lines() {
        let raw = "{\"delta\":\"a\"}\n{\"delta\":\"b\"}\n";
        assert_eq!(parse_json_lines(raw), "ab");
    }

    #[test]
    fn rejects_invalid_json_output() {
        assert!(parse_json_content("not json").is_err());
    }
}
