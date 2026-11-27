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

use crate::driver::{ChatMessage, ToolCall};

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
    if message.role == "tool" {
        let call_id = message.tool_call_id.as_deref().unwrap_or("unknown");
        if let Some(name) = message.name.as_deref() {
            output.push_str(&format!("ToolResult ({name}, id={call_id}):\n"));
        } else {
            output.push_str(&format!("ToolResult (id={call_id}):\n"));
        }
        output.push_str(message.content.trim());
        output.push_str("\n\n");
        return;
    }

    output.push_str(&capitalize_role(&message.role));
    output.push_str(":\n");
    if !message.content.trim().is_empty() {
        output.push_str(message.content.trim());
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
}
