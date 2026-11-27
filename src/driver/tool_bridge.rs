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
    driver::{ChatMessage, ResponseFormat, ToolCall},
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
}
