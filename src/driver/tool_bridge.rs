//! SPDX-License-Identifier: GPL-3.0-or-later
//! Structured tool-calling bridge for CLI and PTY backends.
//!
//! Normalizes tool definitions across OpenAI Chat Completions, OpenAI
//! Responses, and Anthropic Messages schemas, renders deterministic tool
//! invocation instructions for subprocess backends, and extracts
//! `<vouch_tool_call>...</vouch_tool_call>` payloads into structured
//! [`ToolCall`] values.

use serde_json::{Value, json};
use uuid::Uuid;

use crate::{
    driver::{ChatMessage, ResponseFormat, StreamChunk, ToolCall, ToolCallDelta},
    error::{GatewayError, Result},
};

const OPEN_TAG: &str = "<vouch_tool_call>";
const CLOSE_TAG: &str = "</vouch_tool_call>";

/// Normalize an OpenAI, OpenAI Responses, or Anthropic tool definition into
/// OpenAI Chat Completions function-tool format.
pub fn normalize_tool_definition(tool: &Value) -> Value {
    if tool.get("function").is_some() {
        return tool.clone();
    }
    if let Some(name) = tool.get("name").and_then(Value::as_str) {
        let description = tool
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("");
        let parameters = tool
            .get("parameters")
            .or_else(|| tool.get("input_schema"))
            .cloned()
            .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
        return json!({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        });
    }
    tool.clone()
}

/// Normalize a slice of raw tool definitions.
pub fn normalize_tools(tools: Vec<Value>) -> Vec<Value> {
    tools
        .into_iter()
        .map(|tool| normalize_tool_definition(&tool))
        .collect()
}

/// Format conversation turns including structured tool calls and tool results.
pub fn format_message(message: &ChatMessage, output: &mut String) {
    let text = message.content.as_plain_text();
    if message.role == "tool" {
        let call_id = message.tool_call_id.as_deref().unwrap_or("unknown");
        if let Some(name) = message.name.as_deref() {
            output.push_str(&format!("ToolResult ({name}, id={call_id}):\n"));
        } else {
            output.push_str(&format!("ToolResult (id={call_id}):\n"));
        }
        output.push_str(text.trim());
        output.push_str("\n\n");
        return;
    }

    output.push_str(&capitalize_role(&message.role));
    output.push_str(":\n");
    if !text.trim().is_empty() {
        output.push_str(text.trim());
        output.push('\n');
    }
    for call in &message.tool_calls {
        let args: Value =
            serde_json::from_str(&call.arguments).unwrap_or_else(|_| json!(call.arguments));
        let payload = json!({
            "id": call.id,
            "name": call.name,
            "arguments": args,
        });
        output.push_str(OPEN_TAG);
        output.push_str(&payload.to_string());
        output.push_str(CLOSE_TAG);
        output.push('\n');
    }
    output.push('\n');
}

/// Append tool-calling instructions and JSON schemas to a rendered CLI prompt.
pub fn append_tool_instructions(prompt: &mut String, tools: &[Value], tool_choice: Option<&Value>) {
    if tools.is_empty() {
        return;
    }
    if !prompt.is_empty() && !prompt.ends_with("\n\n") {
        if prompt.ends_with('\n') {
            prompt.push('\n');
        } else {
            prompt.push_str("\n\n");
        }
    }

    prompt.push_str(
        "Available tools:\n\
         When you decide to invoke a tool, respond with one or more blocks using this exact XML tag format:\n\
         <vouch_tool_call>{\"name\":\"TOOL_NAME\",\"arguments\":{...}}</vouch_tool_call>\n\
         Use valid JSON inside the tags. If no tool call is needed, answer normally without tags.\n\n",
    );

    if let Some(choice) = tool_choice {
        match choice {
            Value::String(mode) => {
                prompt.push_str(&format!("Tool choice constraint: {mode}\n\n"));
            }
            Value::Object(_) => {
                if let Some(name) = choice
                    .pointer("/function/name")
                    .or_else(|| choice.get("name"))
                    .and_then(Value::as_str)
                {
                    prompt.push_str(&format!("Required tool: {name}\n\n"));
                }
            }
            _ => {}
        }
    }

    for tool in tools {
        let normalized = normalize_tool_definition(tool);
        let function = normalized.get("function").unwrap_or(&normalized);
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let description = function
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("");
        let parameters = function
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object"}));
        if description.is_empty() {
            prompt.push_str(&format!("- {name}: parameters={parameters}\n"));
        } else {
            prompt.push_str(&format!(
                "- {name}: {description} | parameters={parameters}\n"
            ));
        }
    }
}

/// Extract `<vouch_tool_call>...</vouch_tool_call>` blocks from model text.
///
/// Malformed blocks are preserved in the returned text so no information is
/// silently dropped.
pub fn parse_tool_calls(raw: &str) -> (String, Vec<ToolCall>) {
    let mut remaining = String::with_capacity(raw.len());
    let mut calls = Vec::new();
    let mut cursor = 0;

    while let Some(rel_start) = raw[cursor..].find(OPEN_TAG) {
        let start = cursor + rel_start;
        remaining.push_str(&raw[cursor..start]);
        let body_start = start + OPEN_TAG.len();
        let Some(rel_end) = raw[body_start..].find(CLOSE_TAG) else {
            remaining.push_str(&raw[start..]);
            cursor = raw.len();
            break;
        };
        let body_end = body_start + rel_end;
        let inner = raw[body_start..body_end].trim();
        if let Some(call) = parse_single_tool_call(inner) {
            calls.push(call);
        } else {
            remaining.push_str(&raw[start..body_end + CLOSE_TAG.len()]);
        }
        cursor = body_end + CLOSE_TAG.len();
    }
    if cursor < raw.len() {
        remaining.push_str(&raw[cursor..]);
    }

    (remaining.trim().to_string(), calls)
}

/// Append strict JSON / JSON-Schema output instructions to a CLI/PTY prompt.
pub fn append_structured_output_instructions(prompt: &mut String, format: &ResponseFormat) {
    if !prompt.is_empty() && !prompt.ends_with("\n\n") {
        if prompt.ends_with('\n') {
            prompt.push('\n');
        } else {
            prompt.push_str("\n\n");
        }
    }
    match format {
        ResponseFormat::JsonObject => {
            prompt.push_str(
                "Output format requirement:\n\
                 Respond ONLY with a single valid JSON value. Do NOT wrap the output in markdown code fences or add explanatory text.",
            );
        }
        ResponseFormat::JsonSchema { name, schema, .. } => {
            let label = name.as_deref().unwrap_or("response");
            prompt.push_str(&format!(
                "Output format requirement:\n\
                 Respond ONLY with raw valid JSON matching the schema `{label}` below. \
                 Do NOT wrap the JSON in markdown code fences (` ``` `) or add explanatory prose.\n\
                 JSON Schema:\n{schema}",
            ));
        }
    }
}

/// Strip leading and trailing Markdown code fences (such as ```` ```json ... ``` ````).
pub fn strip_markdown_fences(raw: &str) -> String {
    let trimmed = raw.trim();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return trimmed.to_string();
    };
    let after_lang = match rest.find('\n') {
        Some(idx) => {
            let header = rest[..idx].trim();
            if header.is_empty()
                || header
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))
            {
                &rest[idx + 1..]
            } else {
                rest
            }
        }
        None => rest,
    };
    let body = after_lang
        .trim_end()
        .strip_suffix("```")
        .unwrap_or(after_lang);
    body.trim().to_string()
}

/// Strip accidental Markdown fences and validate structured JSON/Schema output.
pub fn enforce_response_format(raw: &str, format: &ResponseFormat) -> Result<String> {
    let cleaned = strip_markdown_fences(raw);
    let parsed: Value = serde_json::from_str(&cleaned).map_err(|error| {
        GatewayError::Upstream(format!(
            "CLI/PTY backend failed structured output JSON validation: {error}"
        ))
    })?;

    if let ResponseFormat::JsonSchema { schema, .. } = format {
        validate_json_schema_value(&parsed, schema, "$")?;
    }
    Ok(cleaned)
}

fn validate_json_schema_value(value: &Value, schema: &Value, path: &str) -> Result<()> {
    if let Some(expected_type) = schema.get("type").and_then(Value::as_str) {
        let ok = match expected_type {
            "object" => value.is_object(),
            "array" => value.is_array(),
            "string" => value.is_string(),
            "boolean" => value.is_boolean(),
            "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
            "number" => value.is_number(),
            "null" => value.is_null(),
            _ => true,
        };
        if !ok {
            return Err(GatewayError::Upstream(format!(
                "structured output at {path} expected JSON type {expected_type:?}"
            )));
        }
    }

    if let Some(obj) = value.as_object() {
        if let Some(required) = schema.get("required").and_then(Value::as_array) {
            for req in required.iter().filter_map(Value::as_str) {
                if !obj.contains_key(req) {
                    return Err(GatewayError::Upstream(format!(
                        "structured output at {path} missing required field {req:?}"
                    )));
                }
            }
        }
        if let Some(props) = schema.get("properties").and_then(Value::as_object) {
            for (key, prop_schema) in props {
                if let Some(child) = obj.get(key) {
                    validate_json_schema_value(child, prop_schema, &format!("{path}.{key}"))?;
                }
            }
        }
    } else if let Some(items) = value.as_array()
        && let Some(item_schema) = schema.get("items")
    {
        for (idx, item) in items.iter().enumerate() {
            validate_json_schema_value(item, item_schema, &format!("{path}[{idx}]"))?;
        }
    }
    Ok(())
}

fn parse_single_tool_call(inner: &str) -> Option<ToolCall> {
    let value: Value = serde_json::from_str(inner).ok()?;
    let name = value
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())?
        .to_string();
    let arguments = match value.get("arguments") {
        Some(Value::String(raw)) => raw.clone(),
        Some(other) => other.to_string(),
        None => "{}".to_string(),
    };
    Some(ToolCall {
        id: format!("call_{}", Uuid::new_v4().simple()),
        name,
        arguments,
    })
}

/// Incremental event emitted by [`StreamingToolParser`].
#[derive(Debug, Clone, PartialEq)]
pub enum StreamParseEvent {
    /// Confirmed assistant text that is safe to emit immediately.
    Text(String),
    /// Completed structured tool-call delta parsed from `<vouch_tool_call>...</vouch_tool_call>`.
    ToolCall(ToolCallDelta),
}

/// Incremental parser for `<vouch_tool_call>...</vouch_tool_call>` blocks split across chunks.
///
/// Buffers potential tag prefixes at chunk boundaries so partial `<vouch_tool_call>` tags
/// are never leaked to streaming clients while normal assistant prose streams immediately.
#[derive(Debug, Default)]
pub struct StreamingToolParser {
    in_tag: bool,
    buffer: String,
    next_index: usize,
    saw_tool_calls: bool,
}

impl StreamingToolParser {
    /// Create a new streaming tool parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// True if at least one complete `<vouch_tool_call>` block has been parsed.
    pub fn saw_tool_calls(&self) -> bool {
        self.saw_tool_calls
    }

    /// Feed an incoming chunk and return confirmed text and/or completed tool calls.
    pub fn push(&mut self, chunk: &str) -> Vec<StreamParseEvent> {
        if !chunk.is_empty() {
            self.buffer.push_str(chunk);
        }
        let mut events = Vec::new();
        loop {
            if !self.in_tag {
                if let Some(start) = self.buffer.find(OPEN_TAG) {
                    if start > 0 {
                        events.push(StreamParseEvent::Text(self.buffer[..start].to_string()));
                    }
                    self.buffer.drain(..start + OPEN_TAG.len());
                    self.in_tag = true;
                } else {
                    let hold = trailing_open_tag_prefix_len(&self.buffer);
                    let emit_len = self.buffer.len().saturating_sub(hold);
                    if emit_len > 0 {
                        events.push(StreamParseEvent::Text(self.buffer[..emit_len].to_string()));
                        self.buffer.drain(..emit_len);
                    }
                    break;
                }
            } else if let Some(end) = self.buffer.find(CLOSE_TAG) {
                let inner = self.buffer[..end].trim();
                if let Some(call) = parse_single_tool_call(inner) {
                    self.saw_tool_calls = true;
                    events.push(StreamParseEvent::ToolCall(ToolCallDelta {
                        index: self.next_index,
                        id: Some(call.id),
                        name: Some(call.name),
                        arguments_delta: Some(call.arguments),
                    }));
                    self.next_index += 1;
                } else {
                    events.push(StreamParseEvent::Text(format!(
                        "{OPEN_TAG}{}{CLOSE_TAG}",
                        &self.buffer[..end]
                    )));
                }
                self.buffer.drain(..end + CLOSE_TAG.len());
                self.in_tag = false;
            } else {
                break;
            }
        }
        events
    }

    /// Flush any buffered tail at EOF.
    pub fn finish(&mut self) -> Vec<StreamParseEvent> {
        if self.buffer.is_empty() {
            return Vec::new();
        }
        let mut events = Vec::new();
        if self.in_tag {
            events.push(StreamParseEvent::Text(format!(
                "{OPEN_TAG}{}",
                std::mem::take(&mut self.buffer)
            )));
            self.in_tag = false;
        } else {
            events.push(StreamParseEvent::Text(std::mem::take(&mut self.buffer)));
        }
        events
    }
}

fn trailing_open_tag_prefix_len(text: &str) -> usize {
    let max_len = (OPEN_TAG.len() - 1).min(text.len());
    for len in (1..=max_len).rev() {
        if text.is_char_boundary(text.len() - len) && text.ends_with(&OPEN_TAG[..len]) {
            return len;
        }
    }
    0
}

#[derive(Debug)]
enum FencePhase {
    Leading(String),
    Body { backtick_hold: String },
}

/// Incremental Markdown code-fence stripper and JSON/JSON-Schema validator for streams.
#[derive(Debug)]
pub struct StreamingJsonFenceStripper {
    format: ResponseFormat,
    phase: FencePhase,
    accumulated: String,
}

impl StreamingJsonFenceStripper {
    /// Create a streaming fence stripper & validator for `format`.
    pub fn new(format: ResponseFormat) -> Self {
        Self {
            format,
            phase: FencePhase::Leading(String::new()),
            accumulated: String::new(),
        }
    }

    /// Feed confirmed assistant text and return fence-stripped text ready for emission.
    pub fn push(&mut self, chunk: &str) -> Option<String> {
        if chunk.is_empty() {
            return None;
        }
        let mut out = String::new();
        match &mut self.phase {
            FencePhase::Leading(buf) => {
                buf.push_str(chunk);
                let trimmed = buf.trim_start();
                if trimmed.is_empty() {
                    if buf.len() > 64 {
                        let ready = std::mem::take(buf);
                        self.phase = FencePhase::Body {
                            backtick_hold: String::new(),
                        };
                        self.push_body(&ready, &mut out);
                    }
                } else if !trimmed.starts_with('`') {
                    let ready = std::mem::take(buf);
                    self.phase = FencePhase::Body {
                        backtick_hold: String::new(),
                    };
                    self.push_body(&ready, &mut out);
                } else if trimmed.len() < 3 {
                    if !"```".starts_with(trimmed) {
                        let ready = std::mem::take(buf);
                        self.phase = FencePhase::Body {
                            backtick_hold: String::new(),
                        };
                        self.push_body(&ready, &mut out);
                    }
                } else if let Some(after) = trimmed.strip_prefix("```") {
                    if let Some(nl) = after.find('\n') {
                        let header = after[..nl].trim();
                        if is_fence_lang_header(header) {
                            let rest = after[nl + 1..].to_string();
                            self.phase = FencePhase::Body {
                                backtick_hold: String::new(),
                            };
                            self.push_body(&rest, &mut out);
                        } else {
                            let rest = after.to_string();
                            self.phase = FencePhase::Body {
                                backtick_hold: String::new(),
                            };
                            self.push_body(&rest, &mut out);
                        }
                    } else if after.chars().any(|ch| !is_fence_lang_char(ch)) {
                        let rest = after.to_string();
                        self.phase = FencePhase::Body {
                            backtick_hold: String::new(),
                        };
                        self.push_body(&rest, &mut out);
                    } else if after.len() > 32 {
                        let ready = std::mem::take(buf);
                        self.phase = FencePhase::Body {
                            backtick_hold: String::new(),
                        };
                        self.push_body(&ready, &mut out);
                    }
                } else {
                    let ready = std::mem::take(buf);
                    self.phase = FencePhase::Body {
                        backtick_hold: String::new(),
                    };
                    self.push_body(&ready, &mut out);
                }
            }
            FencePhase::Body { .. } => {
                self.push_body(chunk, &mut out);
            }
        }
        if out.is_empty() { None } else { Some(out) }
    }

    fn push_body(&mut self, text: &str, out: &mut String) {
        let FencePhase::Body { backtick_hold } = &mut self.phase else {
            return;
        };
        let mut combined = std::mem::take(backtick_hold);
        combined.push_str(text);

        // Hold back a trailing suffix if it begins with backticks followed only
        // by backticks/whitespace (potential closing ``` fence split across reads).
        let hold_start = find_trailing_closing_fence_start(&combined);
        if hold_start < combined.len() {
            *backtick_hold = combined[hold_start..].to_string();
            let emit = &combined[..hold_start];
            self.accumulated.push_str(emit);
            out.push_str(emit);
        } else {
            self.accumulated.push_str(&combined);
            out.push_str(&combined);
        }
    }

    /// Flush buffered tail and validate accumulated JSON when `validate` is true.
    pub fn finish(&mut self, validate: bool) -> Result<Option<String>> {
        let mut tail = None;
        let phase = std::mem::replace(
            &mut self.phase,
            FencePhase::Body {
                backtick_hold: String::new(),
            },
        );
        match phase {
            FencePhase::Leading(buf) => {
                let cleaned = strip_markdown_fences(&buf);
                self.accumulated.push_str(&cleaned);
                if !cleaned.is_empty() {
                    tail = Some(cleaned);
                }
            }
            FencePhase::Body { backtick_hold } => {
                let trimmed = backtick_hold.trim();
                if !(trimmed.starts_with("```") && trimmed.chars().all(|c| c == '`'))
                    && !backtick_hold.is_empty()
                {
                    self.accumulated.push_str(&backtick_hold);
                    tail = Some(backtick_hold);
                }
            }
        }
        if validate {
            enforce_response_format(&self.accumulated, &self.format)?;
        }
        Ok(tail)
    }
}

fn is_fence_lang_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_')
}

fn is_fence_lang_header(header: &str) -> bool {
    header.is_empty() || header.chars().all(is_fence_lang_char)
}

fn find_trailing_closing_fence_start(text: &str) -> usize {
    let trimmed_end = text.trim_end();
    if trimmed_end.is_empty() || !trimmed_end.ends_with('`') {
        return text.len();
    }
    let backtick_start = trimmed_end
        .rfind(|ch| ch != '`')
        .map(|idx| idx + text[idx..].chars().next().map_or(1, char::len_utf8))
        .unwrap_or(0);
    let backtick_count = trimmed_end.len().saturating_sub(backtick_start);
    if (1..=3).contains(&backtick_count) {
        backtick_start
    } else {
        text.len()
    }
}

/// Combined streaming post-processor for CLI/PTY text streams: parses split
/// `<vouch_tool_call>` blocks and strips/validates `response_format` JSON.
#[derive(Debug)]
pub struct SubprocessStreamFilter {
    tool_parser: Option<StreamingToolParser>,
    fence_stripper: Option<StreamingJsonFenceStripper>,
}

impl SubprocessStreamFilter {
    /// Create a filter for `OutputMode::Text` subprocess streams.
    pub fn new(parse_tools: bool, response_format: Option<ResponseFormat>) -> Self {
        Self {
            tool_parser: parse_tools.then(StreamingToolParser::new),
            fence_stripper: response_format.map(StreamingJsonFenceStripper::new),
        }
    }

    /// True when at least one tool call has been emitted.
    pub fn saw_tool_calls(&self) -> bool {
        self.tool_parser
            .as_ref()
            .is_some_and(StreamingToolParser::saw_tool_calls)
    }

    /// Process a clean text chunk into stream chunks ready to forward immediately.
    pub fn push(&mut self, clean_text: &str) -> Vec<StreamChunk> {
        let mut chunks = Vec::new();
        if let Some(parser) = &mut self.tool_parser {
            for event in parser.push(clean_text) {
                self.push_event(event, &mut chunks);
            }
        } else {
            self.push_text_event(clean_text.to_string(), &mut chunks);
        }
        chunks
    }

    fn push_event(&mut self, event: StreamParseEvent, out: &mut Vec<StreamChunk>) {
        match event {
            StreamParseEvent::Text(text) => self.push_text_event(text, out),
            StreamParseEvent::ToolCall(delta) => out.push(StreamChunk::tool_calls(vec![delta])),
        }
    }

    fn push_text_event(&mut self, text: String, out: &mut Vec<StreamChunk>) {
        if text.is_empty() {
            return;
        }
        if let Some(stripper) = &mut self.fence_stripper {
            if let Some(stripped) = stripper.push(&text)
                && !stripped.is_empty()
            {
                out.push(StreamChunk::delta(stripped));
            }
        } else {
            out.push(StreamChunk::delta(text));
        }
    }

    /// Flush buffered tails and validate structured JSON output when applicable.
    ///
    /// Returns trailing chunks plus the terminal finish reason (`"tool_calls"` or `"stop"`).
    pub fn finish(&mut self) -> Result<(Vec<StreamChunk>, &'static str)> {
        let mut chunks = Vec::new();
        if let Some(parser) = &mut self.tool_parser {
            let pending = parser.finish();
            for event in pending {
                self.push_event(event, &mut chunks);
            }
        }
        let saw_tools = self.saw_tool_calls();
        if let Some(stripper) = &mut self.fence_stripper
            && let Some(tail) = stripper.finish(!saw_tools)?
            && !tail.is_empty()
        {
            chunks.push(StreamChunk::delta(tail));
        }
        let reason = if saw_tools { "tool_calls" } else { "stop" };
        Ok((chunks, reason))
    }
}

fn capitalize_role(role: &str) -> String {
    let mut chars = role.chars();
    match chars.next() {
        Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
        None => "User".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_anthropic_and_responses_tools() {
        let anthropic = json!({
            "name": "read_file",
            "description": "Read a path",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}
        });
        let normalized = normalize_tool_definition(&anthropic);
        assert_eq!(normalized["type"], "function");
        assert_eq!(normalized["function"]["name"], "read_file");
        assert_eq!(
            normalized["function"]["parameters"]["properties"]["path"]["type"],
            "string"
        );
    }

    #[test]
    fn parses_multiple_tool_calls_and_preserves_text() {
        let raw = "Checking now.\n<vouch_tool_call>{\"name\":\"ping\",\"arguments\":{\"host\":\"example.com\"}}</vouch_tool_call>\n<vouch_tool_call>{\"name\":\"echo\",\"arguments\":\"{\\\"msg\\\":\\\"hi\\\"}\"}</vouch_tool_call>";
        let (text, calls) = parse_tool_calls(raw);
        assert_eq!(text, "Checking now.");
        assert_eq!(calls.len(), 2);
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].name, "ping");
        assert_eq!(calls[0].arguments, "{\"host\":\"example.com\"}");
        assert_eq!(calls[1].name, "echo");
        assert_eq!(calls[1].arguments, "{\"msg\":\"hi\"}");
    }

    #[test]
    fn preserves_malformed_tool_call_as_text() {
        let raw = "oops <vouch_tool_call>{bad json</vouch_tool_call>";
        let (text, calls) = parse_tool_calls(raw);
        assert!(calls.is_empty());
        assert_eq!(text, raw);
    }

    #[test]
    fn strips_fences_and_validates_structured_json_schema() {
        let schema = json!({
            "type": "object",
            "required": ["city", "temp"],
            "properties": {
                "city": {"type": "string"},
                "temp": {"type": "number"}
            }
        });
        let format = ResponseFormat::JsonSchema {
            name: Some("weather".into()),
            schema: schema.clone(),
            strict: true,
            raw: json!({"type": "json_schema", "json_schema": {"name": "weather", "schema": schema}}),
        };
        let valid = "```json\n{\"city\":\"Paris\",\"temp\":18}\n```";
        let out = enforce_response_format(valid, &format).expect("valid JSON passes");
        assert_eq!(out, "{\"city\":\"Paris\",\"temp\":18}");

        let missing_req = "{\"city\":\"Paris\"}";
        assert!(enforce_response_format(missing_req, &format).is_err());

        let invalid_json = "not json at all";
        assert!(enforce_response_format(invalid_json, &format).is_err());
    }

    #[test]
    fn streaming_tool_parser_buffers_split_tags_without_leaking() {
        let mut parser = StreamingToolParser::new();
        let e1 = parser.push("Let me check <vouch_tool");
        assert_eq!(e1, vec![StreamParseEvent::Text("Let me check ".into())]);

        let e2 = parser.push("_call>{\"name\":\"search\",");
        assert!(e2.is_empty());

        let e3 = parser.push("\"arguments\":{\"q\":\"rust\"}}</vouch_tool_call> done");
        assert_eq!(e3.len(), 2);
        match &e3[0] {
            StreamParseEvent::ToolCall(delta) => {
                assert_eq!(delta.index, 0);
                assert_eq!(delta.name.as_deref(), Some("search"));
                assert_eq!(delta.arguments_delta.as_deref(), Some("{\"q\":\"rust\"}"));
            }
            other => panic!("expected tool call, got {other:?}"),
        }
        assert_eq!(e3[1], StreamParseEvent::Text(" done".into()));
        assert!(parser.finish().is_empty());
        assert!(parser.saw_tool_calls());
    }

    #[test]
    fn streaming_fence_stripper_handles_split_fences_and_validates() {
        let format = ResponseFormat::JsonObject;
        let mut stripper = StreamingJsonFenceStripper::new(format.clone());
        assert_eq!(stripper.push("```js"), None);
        assert_eq!(
            stripper.push("on\n{\"ok\":true}"),
            Some("{\"ok\":true}".into())
        );
        assert_eq!(stripper.push("\n``"), Some("\n".into()));
        assert_eq!(stripper.push("`\n"), None);
        assert_eq!(stripper.finish(true).expect("valid json"), None);

        let mut bad = StreamingJsonFenceStripper::new(format);
        let _ = bad.push("not-json");
        assert!(bad.finish(true).is_err());
    }
}
