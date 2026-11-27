//! End-to-end tests for the ingress adapters, routing policy, egress drivers,
//! and process supervision. Everything runs against mock shell scripts and an
//! in-process axum router, so `cargo test` needs no live LLM credentials.

use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Method, Request, StatusCode},
    response::Response,
};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::time::sleep;
use tower::ServiceExt;
use vouch::{config::GatewayConfig, ingress::build_router, state::AppState};

/// Owns a temp directory with the config file and all mock executables.
struct Harness {
    _dir: TempDir,
    app: Router,
    state: std::sync::Arc<AppState>,
}

impl Harness {
    /// Build a harness from a complete TOML document.
    async fn new(config: String) -> Self {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("vouch.toml");
        fs::write(&path, &config).expect("write config");
        let config = GatewayConfig::from_toml_str(&config).expect("valid test config");
        let state = AppState::new(path, config).expect("state");
        state.start_background_tasks();
        let app = build_router(state.clone());
        Self {
            _dir: dir,
            app,
            state,
        }
    }

    /// Send a JSON request through the full ingress stack.
    async fn post_json(&self, uri: &str, headers: &[(&str, &str)], body: Value) -> Response {
        let mut builder = Request::builder()
            .method(Method::POST)
            .uri(uri)
            .header("content-type", "application/json");
        for (name, value) in headers {
            builder = builder.header(*name, *value);
        }
        let request = builder
            .body(Body::from(serde_json::to_vec(&body).expect("serialize")))
            .expect("request");
        self.app.clone().oneshot(request).await.expect("router")
    }

    /// Send a GET request through the full ingress stack.
    async fn get(&self, uri: &str, headers: &[(&str, &str)]) -> Response {
        let mut builder = Request::builder().method(Method::GET).uri(uri);
        for (name, value) in headers {
            builder = builder.header(*name, *value);
        }
        let request = builder.body(Body::empty()).expect("request");
        self.app.clone().oneshot(request).await.expect("router")
    }
}

/// Read a response body as UTF-8.
async fn body_text(response: Response) -> String {
    let bytes = to_bytes(response.into_body(), 8 * 1024 * 1024)
        .await
        .expect("body");
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Write an executable shell script into the given directory.
fn script(dir: &Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, body).expect("write script");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).expect("chmod");
    }
    path
}

/// Render a CLI backend block.
fn cli_backend(id: &str, command: &Path, timeout_secs: u64, env: &str) -> String {
    format!(
        "[[backends]]\n\
         id = \"{id}\"\n\
         kind = \"cli\"\n\
         command = \"{}\"\n\
         stdin = true\n\
         output_mode = \"text\"\n\
         timeout_secs = {timeout_secs}\n\
         max_concurrency = 1\n\
         {env}\n",
        command.display()
    )
}

/// Render a complete configuration document.
fn config(backends: &str, routes: &str, tokens: &str, request_timeout_secs: u64) -> String {
    format!(
        "[server]\n\
         tcp_listen = \"127.0.0.1:0\"\n\
         request_timeout_secs = {request_timeout_secs}\n\
         stream_timeout_secs = 10\n\
         keep_alive_interval_secs = 0\n\
         log_level = \"off\"\n\
         \n\
         [defaults]\n\
         model = \"coding/default\"\n\
         queue_timeout_ms = 500\n\
         \n\
         [resilience]\n\
         failure_threshold = 100\n\
         reset_timeout_secs = 1\n\
         \n\
         {backends}\n\
         [routes]\n\
         {routes}\n\
         {tokens}"
    )
}

const PLAIN_SCRIPT: &str = "#!/bin/sh\nread line\nprintf 'hello from mock\\n'\n";

fn chat_body(model: &str, stream: bool) -> Value {
    json!({
        "model": model,
        "stream": stream,
        "messages": [{"role": "user", "content": "say hello"}]
    })
}

fn auth_header(token: &str) -> (&'static str, String) {
    ("authorization", format!("Bearer {token}"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn models_ops_and_non_streaming_completion() {
    let dir = TempDir::new().expect("tempdir");
    let mock = script(dir.path(), "plain.sh", PLAIN_SCRIPT);
    let harness = Harness::new(config(
        &cli_backend("mock", &mock, 5, ""),
        "\"coding/default\" = [\"mock\"]\n",
        "",
        10,
    ))
    .await;

    let models = harness.get("/v1/models", &[]).await;
    assert_eq!(models.status(), StatusCode::OK);
    let payload: Value = serde_json::from_str(&body_text(models).await).expect("models JSON");
    assert!(!payload["data"].as_array().expect("data").is_empty());

    assert_eq!(harness.get("/healthz", &[]).await.status(), StatusCode::OK);
    assert_eq!(harness.get("/readyz", &[]).await.status(), StatusCode::OK);

    let metrics = harness.get("/metrics", &[]).await;
    assert_eq!(metrics.status(), StatusCode::OK);
    let metrics = body_text(metrics).await;
    assert!(metrics.contains("vouch_requests_total"));

    let response = harness
        .post_json(
            "/v1/chat/completions",
            &[],
            chat_body("coding/default", false),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-vouch-backend"], "mock");
    assert_eq!(response.headers()["x-vouch-fallback"], "false");
    let payload: Value = serde_json::from_str(&body_text(response).await).expect("chat JSON");
    assert_eq!(
        payload["choices"][0]["message"]["content"],
        "hello from mock"
    );
    assert_eq!(payload["choices"][0]["finish_reason"], "stop");

    harness.state.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn openai_sse_stream_and_ansi_stripping() {
    let dir = TempDir::new().expect("tempdir");
    let streamer = script(
        dir.path(),
        "stream.sh",
        "#!/bin/sh\nread line\nprintf 'alpha\\n'\nsleep 0.05\nprintf '\\033[31mbeta\\033[0m\\n'\n",
    );
    let harness = Harness::new(config(
        &cli_backend("streamer", &streamer, 5, ""),
        "\"coding/default\" = [\"streamer\"]\n",
        "",
        10,
    ))
    .await;

    let response = harness
        .post_json(
            "/v1/chat/completions",
            &[],
            chat_body("coding/default", true),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()["content-type"].to_str().expect("header"),
        "text/event-stream"
    );
    let text = body_text(response).await;
    assert!(text.contains("data: "), "missing SSE data frames: {text}");
    assert!(text.contains("alpha"), "missing first delta: {text}");
    assert!(text.contains("beta"), "missing second delta: {text}");
    assert!(
        !text.contains('\u{1b}'),
        "ANSI escape leaked into SSE: {text}"
    );
    assert!(
        text.contains("data: [DONE]"),
        "missing terminal frame: {text}"
    );

    harness.state.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn anthropic_messages_json_and_sse() {
    let dir = TempDir::new().expect("tempdir");
    let mock = script(dir.path(), "plain.sh", PLAIN_SCRIPT);
    let harness = Harness::new(config(
        &cli_backend("mock", &mock, 5, ""),
        "\"coding/default\" = [\"mock\"]\n",
        "",
        10,
    ))
    .await;

    let payload = json!({
        "model": "coding/default",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hello"}]
    });
    let response = harness
        .post_json("/v1/messages", &[], payload.clone())
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_str(&body_text(response).await).expect("messages JSON");
    assert_eq!(body["content"][0]["text"], "hello from mock");
    assert_eq!(body["stop_reason"], "end_turn");

    let mut stream = payload;
    stream["stream"] = json!(true);
    let response = harness.post_json("/v1/messages", &[], stream).await;
    assert_eq!(response.status(), StatusCode::OK);
    let text = body_text(response).await;
    assert!(text.contains("message_start"), "{text}");
    assert!(text.contains("content_block_delta"), "{text}");
    assert!(text.contains("message_stop"), "{text}");

    harness.state.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn falls_back_to_second_backend() {
    let dir = TempDir::new().expect("tempdir");
    let broken = script(
        dir.path(),
        "broken.sh",
        "#!/bin/sh\nread line\necho 'backend down' >&2\nexit 7\n",
    );
    let good = script(dir.path(), "good.sh", PLAIN_SCRIPT);
    let harness = Harness::new(config(
        &format!(
            "{}\n{}",
            cli_backend("primary", &broken, 5, ""),
            cli_backend("secondary", &good, 5, "")
        ),
        "\"coding/default\" = [\"primary\", \"secondary\"]\n",
        "",
        10,
    ))
    .await;

    let response = harness
        .post_json(
            "/v1/chat/completions",
            &[],
            chat_body("coding/default", false),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-vouch-backend"], "secondary");
    assert_eq!(response.headers()["x-vouch-fallback"], "true");
    let payload: Value = serde_json::from_str(&body_text(response).await).expect("chat JSON");
    assert_eq!(
        payload["choices"][0]["message"]["content"],
        "hello from mock"
    );

    harness.state.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn timeout_kills_entire_process_group() {
    let dir = TempDir::new().expect("tempdir");
    let parent_pid = dir.path().join("parent.pid");
    let child_pid = dir.path().join("child.pid");
    let hanging = script(
        dir.path(),
        "hang.sh",
        "#!/bin/sh\necho $$ > \"$VOUCH_TEST_PARENT_PID\"\nsleep 60 &\necho $! > \"$VOUCH_TEST_CHILD_PID\"\nwait\n",
    );
    let env = format!(
        "env = {{ VOUCH_TEST_PARENT_PID = \"{}\", VOUCH_TEST_CHILD_PID = \"{}\" }}\n",
        parent_pid.display(),
        child_pid.display()
    );
    let harness = Harness::new(config(
        &cli_backend("hanging", &hanging, 1, &env),
        "\"coding/default\" = [\"hanging\"]\n",
        "",
        10,
    ))
    .await;

    let response = harness
        .post_json(
            "/v1/chat/completions",
            &[],
            chat_body("coding/default", false),
        )
        .await;
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    let text = body_text(response).await;
    assert!(text.contains("timed out"), "{text}");

    // Allow the guard's SIGTERM -> 500ms grace -> SIGKILL escalation to finish.
    sleep(Duration::from_secs(2)).await;

    let parent = fs::read_to_string(&parent_pid).expect("parent pid file");
    let child = fs::read_to_string(&child_pid).expect("child pid file");
    for (role, raw) in [("parent", parent), ("child", child)] {
        let pid: i32 = raw.trim().parse().expect("pid");
        let alive = std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false);
        assert!(!alive, "{role} pid {pid} survived process-group cleanup");
    }

    harness.state.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn virtual_tokens_enforce_auth_scope_and_rpm() {
    let dir = TempDir::new().expect("tempdir");
    let mock = script(dir.path(), "plain.sh", PLAIN_SCRIPT);
    let tokens = "[[tokens]]\ntoken = \"test-token-123\"\nname = \"ci\"\nrpm = 2\nallowed_models = [\"coding/*\"]\n";
    let harness = Harness::new(config(
        &cli_backend("mock", &mock, 5, ""),
        "\"coding/default\" = [\"mock\"]\n\"general\" = [\"mock\"]\n",
        tokens,
        10,
    ))
    .await;

    let (name, token) = auth_header("test-token-123");
    let accepted = [("authorization", token.as_str())];
    let _ = name;

    let response = harness
        .post_json(
            "/v1/chat/completions",
            &[],
            chat_body("coding/default", false),
        )
        .await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        harness.get("/v1/models", &[]).await.status(),
        StatusCode::UNAUTHORIZED
    );

    let wrong = auth_header("wrong-token-999");
    let response = harness
        .post_json(
            "/v1/chat/completions",
            &[("authorization", wrong.1.as_str())],
            chat_body("coding/default", false),
        )
        .await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let response = harness
        .post_json(
            "/v1/chat/completions",
            &accepted,
            chat_body("general", false),
        )
        .await;
    assert_eq!(response.status(), StatusCode::FORBIDDEN);

    // The forbidden call still consumed one slot of the two-request window;
    // this is the second and final permitted request.
    let response = harness
        .post_json(
            "/v1/chat/completions",
            &accepted,
            chat_body("coding/default", false),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);

    let response = harness
        .post_json(
            "/v1/chat/completions",
            &accepted,
            chat_body("coding/default", false),
        )
        .await;
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);

    harness.state.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sticky_session_affinity_is_stable() {
    let dir = TempDir::new().expect("tempdir");
    let mock = script(
        dir.path(),
        "session.sh",
        "#!/bin/sh\nread line\nprintf 'session=%s\\n' \"$VOUCH_SESSION_ID\"\n",
    );
    let harness = Harness::new(config(
        &cli_backend("sessioned", &mock, 5, ""),
        "\"coding/default\" = [\"sessioned\"]\n",
        "",
        10,
    ))
    .await;

    let response = harness
        .post_json(
            "/v1/chat/completions",
            &[("x-session-id", "my-session")],
            chat_body("coding/default", false),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let payload: Value = serde_json::from_str(&body_text(response).await).expect("chat JSON");
    assert_eq!(
        payload["choices"][0]["message"]["content"],
        "session=my-session"
    );

    harness.state.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_model_is_reported_as_openai_error() {
    let dir = TempDir::new().expect("tempdir");
    let mock = script(dir.path(), "plain.sh", PLAIN_SCRIPT);
    let harness = Harness::new(config(
        &cli_backend("mock", &mock, 5, ""),
        "\"coding/default\" = [\"mock\"]\n",
        "",
        10,
    ))
    .await;

    let response = harness
        .post_json(
            "/v1/chat/completions",
            &[],
            chat_body("does/not/exist", false),
        )
        .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let payload: Value = serde_json::from_str(&body_text(response).await).expect("error JSON");
    assert_eq!(payload["error"]["type"], "not_found_error");
    assert_eq!(payload["error"]["code"], "model_not_found");

    harness.state.shutdown().await;
}

mod v1_tests {
    use super::*;
    use axum::{Json, Router as AxumRouter, response::IntoResponse, routing::post};
    use tokio::net::TcpListener;
    use vouch::{
        auth::Principal,
        cli::{doctor, mcp},
    };

    const TOOL_SCRIPT: &str = r#"#!/bin/sh
PROMPT=$(cat)
if printf '%s' "$PROMPT" | grep -q "ToolResult"; then
  printf 'Final answer: 18C in Paris.\n'
else
  printf '<vouch_tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</vouch_tool_call>\n'
fi
"#;

    const SLEEPY_SCRIPT: &str = r#"#!/bin/sh
sleep 4
printf 'awake\n'
"#;

    fn tool_cli_toml(command: &Path) -> String {
        format!(
            r#"
[defaults]
model = "coding/default"

[[backends]]
id = "tool_cli"
kind = "cli"
command = {:?}
stdin = true
output_mode = "text"
timeout_secs = 10

[routes]
"coding/default" = ["tool_cli"]
"#,
            command.display().to_string()
        )
    }

    #[test]
    fn vouch_example_toml_parses_and_validates() {
        let parsed = GatewayConfig::from_toml_str(include_str!("../vouch.example.toml"))
            .expect("vouch.example.toml should parse and validate");
        assert!(!parsed.routes.is_empty());
        assert_eq!(parsed.discovery.probe_timeout_secs, 30);
        assert!(!parsed.server.enable_ui);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn doctor_honors_configurable_probe_timeout() {
        let dir = TempDir::new().expect("tempdir");
        let fast = script(dir.path(), "fast.sh", PLAIN_SCRIPT);
        let slow = script(dir.path(), "slow.sh", SLEEPY_SCRIPT);

        let fast_toml = format!(
            r#"
[discovery]
enabled = false
probe_ports = false
probe_timeout_secs = 5

{}
[routes]
"coding/default" = ["fast"]
"#,
            cli_backend("fast", &fast, 10, "")
        );
        let fast_cfg_path = dir.path().join("fast.toml");
        fs::write(&fast_cfg_path, &fast_toml).expect("write fast config");
        let fast_cfg = GatewayConfig::from_toml_str(&fast_toml).expect("valid fast config");
        let fast_report = doctor::inspect(&fast_cfg_path, &fast_cfg)
            .await
            .expect("doctor inspect succeeds");
        assert!(fast_report.healthy, "expected healthy fast probe report");
        assert_eq!(fast_report.probes.len(), 1);
        assert!(fast_report.probes[0].ok);

        let slow_toml = format!(
            r#"
[discovery]
enabled = false
probe_ports = false
probe_timeout_secs = 1

{}
[routes]
"coding/default" = ["slow"]
"#,
            cli_backend("slow", &slow, 10, "")
        );
        let slow_cfg_path = dir.path().join("slow.toml");
        fs::write(&slow_cfg_path, &slow_toml).expect("write slow config");
        let slow_cfg = GatewayConfig::from_toml_str(&slow_toml).expect("valid slow config");
        let slow_report = doctor::inspect(&slow_cfg_path, &slow_cfg)
            .await
            .expect("doctor inspect completes on timeout");
        assert!(!slow_report.healthy);
        assert_eq!(slow_report.probes.len(), 1);
        assert!(!slow_report.probes[0].ok);
        assert!(
            slow_report.probes[0].detail.contains("timed out"),
            "unexpected detail: {}",
            slow_report.probes[0].detail
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cli_tool_calling_round_trip_openai_and_anthropic() {
        let dir = TempDir::new().expect("tempdir");
        let tool_bin = script(dir.path(), "tool.sh", TOOL_SCRIPT);
        let harness = Harness::new(tool_cli_toml(&tool_bin)).await;

        let openai_tools = json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Fetch weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }]);

        // OpenAI turn 1 (non-streaming) produces structured tool_calls + tiktoken usage.
        let r1 = harness
            .post_json(
                "/v1/chat/completions",
                &[],
                json!({
                    "model": "coding/default",
                    "stream": false,
                    "messages": [{"role": "user", "content": "Weather in Paris?"}],
                    "tools": openai_tools
                }),
            )
            .await;
        assert_eq!(r1.status(), StatusCode::OK);
        let body1: Value = serde_json::from_str(&body_text(r1).await).expect("turn 1 JSON");
        assert_eq!(body1["choices"][0]["finish_reason"], "tool_calls");
        let call = &body1["choices"][0]["message"]["tool_calls"][0];
        let call_id = call["id"].as_str().expect("generated call id").to_string();
        assert_eq!(call["function"]["name"], "get_weather");
        let args: Value =
            serde_json::from_str(call["function"]["arguments"].as_str().expect("args string"))
                .expect("valid arguments JSON");
        assert_eq!(args["city"], "Paris");
        let prompt_tokens = body1["usage"]["prompt_tokens"]
            .as_u64()
            .expect("prompt_tokens");
        let completion_tokens = body1["usage"]["completion_tokens"]
            .as_u64()
            .expect("completion_tokens");
        assert!(prompt_tokens > 0);
        assert!(completion_tokens > 0);
        assert_eq!(
            body1["usage"]["total_tokens"],
            prompt_tokens + completion_tokens
        );

        // OpenAI turn 1 (streaming) emits tool_calls delta and finish_reason=tool_calls.
        let r_stream = harness
            .post_json(
                "/v1/chat/completions",
                &[],
                json!({
                    "model": "coding/default",
                    "stream": true,
                    "messages": [{"role": "user", "content": "Weather in Paris?"}],
                    "tools": openai_tools
                }),
            )
            .await;
        assert_eq!(r_stream.status(), StatusCode::OK);
        let sse = body_text(r_stream).await;
        assert!(
            sse.contains("get_weather"),
            "missing tool name in SSE: {sse}"
        );
        assert!(
            sse.contains("\"finish_reason\":\"tool_calls\""),
            "missing tool_calls finish_reason in SSE: {sse}"
        );
        assert!(sse.contains("data: [DONE]"));

        // OpenAI turn 2 with tool result completes cleanly.
        let r2 = harness
            .post_json(
                "/v1/chat/completions",
                &[],
                json!({
                    "model": "coding/default",
                    "stream": false,
                    "messages": [
                        {"role": "user", "content": "Weather in Paris?"},
                        {
                            "role": "assistant",
                            "content": null,
                            "tool_calls": [{
                                "id": call_id,
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                            }]
                        },
                        {"role": "tool", "tool_call_id": call_id, "name": "get_weather", "content": "18C"}
                    ],
                    "tools": openai_tools
                }),
            )
            .await;
        assert_eq!(r2.status(), StatusCode::OK);
        let body2: Value = serde_json::from_str(&body_text(r2).await).expect("turn 2 JSON");
        assert_eq!(body2["choices"][0]["finish_reason"], "stop");
        assert!(
            body2["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .contains("18C")
        );

        // Anthropic /v1/messages non-streaming + streaming tool_use.
        let anth_tools = json!([{
            "name": "get_weather",
            "description": "Fetch weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}}
            }
        }]);
        let a1 = harness
            .post_json(
                "/v1/messages",
                &[],
                json!({
                    "model": "coding/default",
                    "max_tokens": 128,
                    "messages": [{"role": "user", "content": "Weather in Paris?"}],
                    "tools": anth_tools
                }),
            )
            .await;
        assert_eq!(a1.status(), StatusCode::OK);
        let anth_body: Value = serde_json::from_str(&body_text(a1).await).expect("anth JSON");
        assert_eq!(anth_body["stop_reason"], "tool_use");
        let blocks = anth_body["content"].as_array().expect("content blocks");
        let tool_block = blocks
            .iter()
            .find(|b| b["type"] == "tool_use")
            .expect("tool_use block");
        assert_eq!(tool_block["name"], "get_weather");
        assert_eq!(tool_block["input"]["city"], "Paris");

        let a_stream = harness
            .post_json(
                "/v1/messages",
                &[],
                json!({
                    "model": "coding/default",
                    "max_tokens": 128,
                    "stream": true,
                    "messages": [{"role": "user", "content": "Weather in Paris?"}],
                    "tools": anth_tools
                }),
            )
            .await;
        assert_eq!(a_stream.status(), StatusCode::OK);
        let anth_sse = body_text(a_stream).await;
        assert!(
            anth_sse.contains("\"type\":\"tool_use\""),
            "sse: {anth_sse}"
        );
        assert!(anth_sse.contains("input_json_delta"), "sse: {anth_sse}");
        assert!(
            anth_sse.contains("\"stop_reason\":\"tool_use\""),
            "sse: {anth_sse}"
        );

        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn http_proxy_tool_calling_and_usage_passthrough() {
        async fn mock_upstream(Json(payload): Json<Value>) -> Response {
            let has_tools = payload
                .get("tools")
                .and_then(Value::as_array)
                .is_some_and(|arr| !arr.is_empty());
            assert!(has_tools, "upstream should receive forwarded tools");

            if payload.get("stream").and_then(Value::as_bool) == Some(true) {
                let sse = concat!(
                    "data: {\"id\":\"up-s1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_http_stream_1\",\"type\":\"function\",\"function\":{\"name\":\"lookup_ticket\",\"arguments\":\"{\\\"id\\\":\\\"T\"}}]}}]}\n\n",
                    "data: {\"id\":\"up-s1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"-42\\\"}\"}}]}}]}\n\n",
                    "data: {\"id\":\"up-s1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7,\"total_tokens\":18}}\n\n",
                    "data: [DONE]\n\n"
                );
                return Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "text/event-stream")
                    .body(Body::from(sse))
                    .expect("valid SSE response");
            }

            Json(json!({
                "id": "up-1",
                "object": "chat.completion",
                "created": 1,
                "model": "upstream-model",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_http_1",
                            "type": "function",
                            "function": {
                                "name": "lookup_ticket",
                                "arguments": "{\"id\":\"T-42\"}"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
            }))
            .into_response()
        }

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral upstream port");
        let addr = listener.local_addr().expect("upstream local addr");
        let upstream_app = AxumRouter::new().route("/v1/chat/completions", post(mock_upstream));
        let server_task = tokio::spawn(async move {
            let _ = axum::serve(listener, upstream_app).await;
        });

        let toml = format!(
            r#"
[defaults]
model = "http/tools"

[[backends]]
id = "mock_http"
kind = "http"
base_url = "http://{addr}"

[routes]
"http/tools" = ["mock_http"]
"#
        );
        let harness = Harness::new(toml).await;
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "lookup_ticket",
                "parameters": {"type": "object", "properties": {"id": {"type": "string"}}}
            }
        }]);

        let non_stream = harness
            .post_json(
                "/v1/chat/completions",
                &[],
                json!({
                    "model": "http/tools",
                    "stream": false,
                    "messages": [{"role": "user", "content": "Check ticket T-42"}],
                    "tools": tools
                }),
            )
            .await;
        assert_eq!(non_stream.status(), StatusCode::OK);
        let ns_body: Value = serde_json::from_str(&body_text(non_stream).await).expect("http JSON");
        assert_eq!(ns_body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            ns_body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "lookup_ticket"
        );
        assert_eq!(ns_body["usage"]["prompt_tokens"], 11);
        assert_eq!(ns_body["usage"]["completion_tokens"], 7);
        assert_eq!(ns_body["usage"]["total_tokens"], 18);

        let streamed = harness
            .post_json(
                "/v1/chat/completions",
                &[],
                json!({
                    "model": "http/tools",
                    "stream": true,
                    "messages": [{"role": "user", "content": "Check ticket T-42"}],
                    "tools": tools
                }),
            )
            .await;
        assert_eq!(streamed.status(), StatusCode::OK);
        let sse = body_text(streamed).await;
        assert!(sse.contains("lookup_ticket"), "sse: {sse}");
        assert!(sse.contains("call_http_stream_1"), "sse: {sse}");
        assert!(
            sse.contains("\"finish_reason\":\"tool_calls\""),
            "sse: {sse}"
        );

        harness.state.shutdown().await;
        server_task.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn responses_endpoint_json_streaming_and_function_calls() {
        let dir = TempDir::new().expect("tempdir");
        let plain = script(dir.path(), "plain.sh", PLAIN_SCRIPT);
        let tool_bin = script(dir.path(), "tool.sh", TOOL_SCRIPT);
        let toml = format!(
            r#"
[defaults]
model = "plain/default"

{}
[[backends]]
id = "tool_cli"
kind = "cli"
command = {:?}
stdin = true
output_mode = "text"
timeout_secs = 10

[routes]
"plain/default" = ["plain"]
"tools/default" = ["tool_cli"]
"#,
            cli_backend("plain", &plain, 5, ""),
            tool_bin.display().to_string()
        );
        let harness = Harness::new(toml).await;

        let resp_json = harness
            .post_json(
                "/v1/responses",
                &[],
                json!({
                    "model": "plain/default",
                    "instructions": "Be concise",
                    "input": "Say hi from responses"
                }),
            )
            .await;
        assert_eq!(resp_json.status(), StatusCode::OK);
        let payload: Value =
            serde_json::from_str(&body_text(resp_json).await).expect("responses JSON");
        assert_eq!(payload["object"], "response");
        assert_eq!(payload["status"], "completed");
        assert_eq!(payload["output"][0]["type"], "message");
        assert!(
            payload["output"][0]["content"][0]["text"]
                .as_str()
                .unwrap_or("")
                .contains("hello")
        );
        assert!(payload["usage"]["input_tokens"].as_u64().unwrap_or(0) > 0);
        assert!(payload["usage"]["output_tokens"].as_u64().unwrap_or(0) > 0);

        let resp_sse = harness
            .post_json(
                "/v1/responses",
                &[],
                json!({
                    "model": "plain/default",
                    "stream": true,
                    "input": [{"role": "user", "content": "Stream responses"}]
                }),
            )
            .await;
        assert_eq!(resp_sse.status(), StatusCode::OK);
        let sse_text = body_text(resp_sse).await;
        assert!(
            sse_text.contains("event: response.created"),
            "sse: {sse_text}"
        );
        assert!(
            sse_text.contains("event: response.output_text.delta"),
            "sse: {sse_text}"
        );
        assert!(
            sse_text.contains("event: response.output_text.done"),
            "sse: {sse_text}"
        );
        assert!(
            sse_text.contains("event: response.completed"),
            "sse: {sse_text}"
        );

        let resp_tool = harness
            .post_json(
                "/v1/responses",
                &[],
                json!({
                    "model": "tools/default",
                    "input": "Weather in Paris?",
                    "tools": [{
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                    }]
                }),
            )
            .await;
        assert_eq!(resp_tool.status(), StatusCode::OK);
        let tool_payload: Value =
            serde_json::from_str(&body_text(resp_tool).await).expect("responses tool JSON");
        let fc = tool_payload["output"]
            .as_array()
            .expect("output array")
            .iter()
            .find(|item| item["type"] == "function_call")
            .expect("function_call output item");
        assert_eq!(fc["name"], "get_weather");
        assert!(fc["arguments"].as_str().unwrap_or("").contains("Paris"));

        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mcp_json_rpc_and_http_endpoint_route_correctly() {
        let dir = TempDir::new().expect("tempdir");
        let plain = script(dir.path(), "plain.sh", PLAIN_SCRIPT);
        let tokens = r#"
[[tokens]]
name = "mcp-client"
token = "mcp-secret-token-1"
allowed_models = ["coding/*"]
"#;
        let harness = Harness::new(config(
            &cli_backend("plain", &plain, 5, ""),
            "\"coding/default\" = [\"plain\"]\n",
            tokens,
            10,
        ))
        .await;

        let local = Principal::local();
        let init = mcp::handle_rpc(
            &harness.state,
            &local,
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05"}
            }),
        )
        .await
        .expect("initialize returns JSON-RPC response");
        assert_eq!(init["result"]["serverInfo"]["name"], "vouch");
        assert_eq!(init["result"]["serverInfo"]["version"], vouch::VERSION);

        let notif = mcp::handle_rpc(
            &harness.state,
            &local,
            json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        )
        .await;
        assert!(notif.is_none());

        let listed = mcp::handle_rpc(
            &harness.state,
            &local,
            json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        )
        .await
        .expect("tools/list response");
        let tool_names: Vec<&str> = listed["result"]["tools"]
            .as_array()
            .expect("tools array")
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();
        assert!(tool_names.contains(&"vouch_ask"));
        assert!(tool_names.contains(&"route_coding_default"));

        let called = mcp::handle_rpc(
            &harness.state,
            &local,
            json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "route_coding_default",
                    "arguments": {"prompt": "ping via mcp"}
                }
            }),
        )
        .await
        .expect("tools/call response");
        assert_eq!(called["result"]["isError"], false);
        assert!(
            called["result"]["content"][0]["text"]
                .as_str()
                .unwrap_or("")
                .contains("hello")
        );

        // HTTP /mcp enforces bearer token authentication and returns 202 for notifications.
        let unauth = harness
            .post_json(
                "/mcp",
                &[],
                json!({"jsonrpc": "2.0", "id": 4, "method": "ping"}),
            )
            .await;
        assert_eq!(unauth.status(), StatusCode::UNAUTHORIZED);

        let authed_ping = harness
            .post_json(
                "/mcp",
                &[("authorization", "Bearer mcp-secret-token-1")],
                json!({"jsonrpc": "2.0", "id": 5, "method": "ping"}),
            )
            .await;
        assert_eq!(authed_ping.status(), StatusCode::OK);

        let authed_notif = harness
            .post_json(
                "/mcp",
                &[("authorization", "Bearer mcp-secret-token-1")],
                json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            )
            .await;
        assert_eq!(authed_notif.status(), StatusCode::ACCEPTED);

        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ui_dashboard_respects_enable_ui_flag() {
        let dir = TempDir::new().expect("tempdir");
        let plain = script(dir.path(), "plain.sh", PLAIN_SCRIPT);

        let disabled = Harness::new(config(
            &cli_backend("plain", &plain, 5, ""),
            "\"coding/default\" = [\"plain\"]\n",
            "",
            10,
        ))
        .await;
        let r_disabled = disabled.get("/ui", &[]).await;
        assert_eq!(r_disabled.status(), StatusCode::NOT_FOUND);
        disabled.state.shutdown().await;

        let enabled_toml = format!(
            r#"
[server]
enable_ui = true

{}
[routes]
"coding/default" = ["plain"]
"#,
            cli_backend("plain", &plain, 5, "")
        );
        let enabled = Harness::new(enabled_toml).await;
        let r_enabled = enabled.get("/ui", &[]).await;
        assert_eq!(r_enabled.status(), StatusCode::OK);
        assert_eq!(
            r_enabled
                .headers()
                .get("cache-control")
                .and_then(|v| v.to_str().ok()),
            Some("no-store")
        );
        let html = body_text(r_enabled).await;
        assert!(html.contains("/healthz"));
        assert!(html.contains("/metrics"));
        enabled.state.shutdown().await;
    }
}
