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

fn cli_backend_v3(id: &str, command: &Path, read_mode: &str) -> String {
    format!(
        r#"
[[backends]]
id = {id:?}
kind = "cli"
command = {:?}
stdin = true
read_mode = {read_mode:?}
max_concurrency = 2
queue_timeout_ms = 2000
timeout_secs = 10
"#,
        command.display().to_string()
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

mod v2_tests {
    use super::*;
    use axum::{Json, Router as AxumRouter, response::IntoResponse, routing::post};
    use tokio::net::TcpListener;
    use vouch::cli::bench;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn prompt_spill_file_avoids_arg_max_and_cleans_up() {
        let dir = TempDir::new().expect("tempdir");
        let spill_script = script(
            dir.path(),
            "spill_reader.sh",
            r#"#!/bin/sh
ARG="$1"
FILE_PATH=$(printf '%s' "$ARG" | sed -n 's/.*located in file: //p')
if [ -z "$FILE_PATH" ] || [ ! -f "$FILE_PATH" ]; then
  echo "missing_spill_file arg=$ARG"
  exit 1
fi
BYTES=$(wc -c < "$FILE_PATH" | tr -d ' ')
echo "SPILLED_PATH=$FILE_PATH BYTES=$BYTES"
"#,
        );
        let toml = format!(
            r#"
[defaults]
model = "coding/default"

[[backends]]
id = "spill_cli"
kind = "cli"
command = {:?}
args = ["{{prompt}}"]
stdin = false
max_inline_prompt_bytes = 64
timeout_secs = 10

[routes]
"coding/default" = ["spill_cli"]
"#,
            spill_script.display().to_string()
        );
        let harness = Harness::new(toml).await;
        let long_msg = "A".repeat(256);
        let resp = harness
            .post_json(
                "/v1/chat/completions",
                &[],
                json!({
                    "model": "coding/default",
                    "stream": false,
                    "messages": [{"role": "user", "content": long_msg}]
                }),
            )
            .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let payload: Value = serde_json::from_str(&body_text(resp).await).expect("JSON");
        let text = payload["choices"][0]["message"]["content"]
            .as_str()
            .expect("content");
        assert!(text.contains("SPILLED_PATH="), "content: {text}");
        let spilled_path = text
            .split_whitespace()
            .find_map(|tok| tok.strip_prefix("SPILLED_PATH="))
            .expect("path token");
        assert!(
            !Path::new(spilled_path).exists(),
            "spill file {spilled_path} should be deleted after drop"
        );
        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pre_content_whitespace_buffering_and_auth_expired_failover() {
        let dir = TempDir::new().expect("tempdir");
        let expired = script(
            dir.path(),
            "expired.sh",
            r#"#!/bin/sh
printf '\033[33m   \n\t\033[0m'
printf 'Authentication error: please sign in to continue.\n' >&2
exit 1
"#,
        );
        let healthy = script(
            dir.path(),
            "healthy.sh",
            r#"#!/bin/sh
printf 'recovered answer\n'
"#,
        );
        let toml = format!(
            r#"
[defaults]
model = "coding/default"

{}
{}
[routes]
"coding/default" = ["expired", "healthy"]
"#,
            cli_backend("expired", &expired, 10, ""),
            cli_backend("healthy", &healthy, 10, "")
        );
        let harness = Harness::new(toml).await;

        let streamed = harness
            .post_json(
                "/v1/chat/completions",
                &[],
                chat_body("coding/default", true),
            )
            .await;
        assert_eq!(streamed.status(), StatusCode::OK);
        assert_eq!(
            streamed
                .headers()
                .get("x-vouch-backend")
                .and_then(|v| v.to_str().ok()),
            Some("healthy")
        );
        assert_eq!(
            streamed
                .headers()
                .get("x-vouch-fallback")
                .and_then(|v| v.to_str().ok()),
            Some("true")
        );
        let sse = body_text(streamed).await;
        assert!(sse.contains("recovered answer"), "sse: {sse}");
        assert!(!sse.contains("please sign in"), "sse leaked stderr: {sse}");

        let ready = harness.get("/readyz", &[]).await;
        assert_eq!(ready.status(), StatusCode::OK);
        let ready_json: Value = serde_json::from_str(&body_text(ready).await).expect("readyz JSON");
        assert_eq!(ready_json["ready"], true);
        let expired_status = ready_json["backends"]
            .as_array()
            .expect("backends")
            .iter()
            .find(|b| b["id"] == "expired")
            .expect("expired entry");
        assert_eq!(expired_status["circuit"], "open");
        assert_eq!(expired_status["auth_expired"], true);

        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn http_429_retry_after_header_overrides_reset_timeout() {
        async fn mock_429() -> Response {
            (
                StatusCode::TOO_MANY_REQUESTS,
                [("retry-after", "4")],
                "rate limit exceeded",
            )
                .into_response()
        }

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let upstream = AxumRouter::new().route("/v1/chat/completions", post(mock_429));
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, upstream).await;
        });

        let toml = format!(
            r#"
[defaults]
model = "http/limited"

[resilience]
reset_timeout_secs = 1
open_on_rate_limit = true

[[backends]]
id = "limited"
kind = "http"
base_url = "http://{addr}"

[routes]
"http/limited" = ["limited"]
"#
        );
        let harness = Harness::new(toml).await;
        let resp = harness
            .post_json(
                "/v1/chat/completions",
                &[],
                chat_body("http/limited", false),
            )
            .await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            resp.headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok()),
            Some("4")
        );

        let ready = harness.get("/readyz", &[]).await;
        assert_eq!(ready.status(), StatusCode::SERVICE_UNAVAILABLE);
        let ready_json: Value = serde_json::from_str(&body_text(ready).await).expect("readyz");
        assert!(
            ready_json["backends"][0]["open_remaining_secs"]
                .as_u64()
                .unwrap_or(0)
                >= 2
        );

        harness.state.shutdown().await;
        task.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn idle_stall_watchdog_terminates_hung_cli_and_resets_on_activity() {
        let dir = TempDir::new().expect("tempdir");
        let stalled = script(
            dir.path(),
            "stall.sh",
            r#"#!/bin/sh
sleep 15
echo "never reached"
"#,
        );
        let active = script(
            dir.path(),
            "active.sh",
            r#"#!/bin/sh
printf 'a'
sleep 0.4
printf 'b'
sleep 0.4
printf 'c\n'
"#,
        );
        let toml = format!(
            r#"
[[backends]]
id = "stalled"
kind = "cli"
command = {:?}
stdin = true
timeout_secs = 30
idle_timeout_secs = 1

[[backends]]
id = "active"
kind = "cli"
command = {:?}
stdin = true
timeout_secs = 30
idle_timeout_secs = 1

[routes]
"test/stall" = ["stalled"]
"test/active" = ["active"]
"#,
            stalled.display().to_string(),
            active.display().to_string(),
        );
        let harness = Harness::new(toml).await;

        let t0 = std::time::Instant::now();
        let resp = harness
            .post_json("/v1/chat/completions", &[], chat_body("test/stall", false))
            .await;
        let elapsed = t0.elapsed();
        assert_eq!(resp.status(), StatusCode::GATEWAY_TIMEOUT);
        assert!(
            elapsed < Duration::from_secs(6),
            "idle watchdog should abort quickly, took {elapsed:?}"
        );

        let ok_resp = harness
            .post_json("/v1/chat/completions", &[], chat_body("test/active", false))
            .await;
        assert_eq!(ok_resp.status(), StatusCode::OK);
        let payload: Value = serde_json::from_str(&body_text(ok_resp).await).expect("JSON");
        assert_eq!(payload["choices"][0]["message"]["content"], "abc");

        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pty_honors_configured_dimensions_and_auto_respond() {
        let dir = TempDir::new().expect("tempdir");
        let pty_bin = script(
            dir.path(),
            "pty_prompt.sh",
            r#"#!/bin/sh
SIZE=$(stty size 2>/dev/null || echo "unknown")
printf 'Continue? [y/N] '
IFS= read -r ANS
printf 'SIZE=%s ANS=%s\n' "$SIZE" "$ANS"
"#,
        );
        let toml = format!(
            r#"
[defaults]
model = "pty/default"

[[backends]]
id = "pty_interactive"
kind = "pty"
command = {:?}
args = ["{{prompt}}"]
stdin = false
pty_cols = 177
pty_rows = 55
timeout_secs = 10

[[backends.pty_auto_respond]]
pattern = "Continue? [y/N]"
response = "yes"

[routes]
"pty/default" = ["pty_interactive"]
"#,
            pty_bin.display().to_string()
        );
        let harness = Harness::new(toml).await;

        let resp = harness
            .post_json("/v1/chat/completions", &[], chat_body("pty/default", false))
            .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let payload: Value = serde_json::from_str(&body_text(resp).await).expect("JSON");
        let content = payload["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert!(content.contains("SIZE=55 177"), "content: {content}");
        assert!(content.contains("ANS=yes"), "content: {content}");

        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn multimodal_base64_image_extraction_and_cleanup() {
        let dir = TempDir::new().expect("tempdir");
        let vision_cli = script(
            dir.path(),
            "vision.sh",
            r#"#!/bin/sh
BODY=$(cat)
IMG=$(printf '%s' "$BODY" | sed -n 's/.*\[Attached image file: \([^]]*\)\].*/\1/p' | head -n1)
if [ -z "$IMG" ] || [ ! -s "$IMG" ]; then
  echo "NO_IMAGE body=$BODY"
  exit 1
fi
BYTES=$(wc -c < "$IMG" | tr -d ' ')
echo "IMAGE_PATH=$IMG IMAGE_BYTES=$BYTES"
"#,
        );
        let toml = format!(
            r#"
[defaults]
model = "vision/cli"

{}
[routes]
"vision/cli" = ["vision"]
"#,
            cli_backend("vision", &vision_cli, 10, "")
        );
        let harness = Harness::new(toml).await;

        let b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
        let resp = harness
            .post_json(
                "/v1/chat/completions",
                &[],
                json!({
                    "model": "vision/cli",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe image"},
                            {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{b64}")}}
                        ]
                    }]
                }),
            )
            .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let payload: Value = serde_json::from_str(&body_text(resp).await).expect("JSON");
        let content = payload["choices"][0]["message"]["content"]
            .as_str()
            .expect("content");
        assert!(content.contains("IMAGE_PATH="), "content: {content}");
        assert!(content.contains("IMAGE_BYTES=68"), "content: {content}");
        let img_path = content
            .split_whitespace()
            .find_map(|tok| tok.strip_prefix("IMAGE_PATH="))
            .expect("path token");
        assert!(
            !Path::new(img_path).exists(),
            "temp image {img_path} should be deleted on drop"
        );

        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn structured_outputs_validate_schema_and_strip_fences() {
        let dir = TempDir::new().expect("tempdir");
        let fenced_ok = script(
            dir.path(),
            "fenced_ok.sh",
            "#!/bin/sh\nprintf '```json\\n{\"city\":\"Tokyo\",\"temp\":21}\\n```\\n'\n",
        );
        let invalid_out = script(
            dir.path(),
            "invalid_out.sh",
            "#!/bin/sh\nprintf 'Sorry, I cannot comply in JSON.'\n",
        );
        let toml = format!(
            r#"
{}
{}
[routes]
"json/good" = ["fenced_ok"]
"json/bad" = ["invalid_out"]
"#,
            cli_backend("fenced_ok", &fenced_ok, 10, ""),
            cli_backend("invalid_out", &invalid_out, 10, "")
        );
        let harness = Harness::new(toml).await;

        let schema = json!({
            "type": "object",
            "required": ["city", "temp"],
            "properties": {
                "city": {"type": "string"},
                "temp": {"type": "number"}
            }
        });
        let good = harness
            .post_json(
                "/v1/chat/completions",
                &[],
                json!({
                    "model": "json/good",
                    "messages": [{"role": "user", "content": "Weather"}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "weather",
                            "strict": true,
                            "schema": schema
                        }
                    }
                }),
            )
            .await;
        assert_eq!(good.status(), StatusCode::OK);
        let good_body: Value = serde_json::from_str(&body_text(good).await).expect("JSON");
        let parsed: Value = serde_json::from_str(
            good_body["choices"][0]["message"]["content"]
                .as_str()
                .expect("content"),
        )
        .expect("clean JSON content");
        assert_eq!(parsed["city"], "Tokyo");
        assert_eq!(parsed["temp"], 21);

        let bad = harness
            .post_json(
                "/v1/responses",
                &[],
                json!({
                    "model": "json/bad",
                    "input": "Weather",
                    "text": {
                        "format": {
                            "type": "json_object"
                        }
                    }
                }),
            )
            .await;
        assert_eq!(bad.status(), StatusCode::BAD_GATEWAY);
        let bad_body: Value = serde_json::from_str(&body_text(bad).await).expect("err JSON");
        assert_eq!(bad_body["error"]["code"], "upstream_error");

        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn native_cli_session_resumes_with_delta_prompt() {
        let dir = TempDir::new().expect("tempdir");
        let session_cli = script(
            dir.path(),
            "session_cli.sh",
            r#"#!/bin/sh
if [ "$1" = "--new" ]; then
  printf 'SESSION_ID=native-777\n' >&2
  printf 'NEW:%s\n' "$2"
elif [ "$1" = "--resume" ]; then
  printf 'RESUMED(%s):%s\n' "$2" "$3"
else
  exit 2
fi
"#,
        );
        let toml = format!(
            r#"
[defaults]
model = "session/cli"

[[backends]]
id = "resumable"
kind = "cli"
command = {:?}
args = ["--new", "{{prompt}}"]
resume_args = ["--resume", "{{backend_session_id}}", "{{prompt}}"]
session_id_regex = "SESSION_ID=([a-z0-9-]+)"
stdin = false
timeout_secs = 10

[routes]
"session/cli" = ["resumable"]
"#,
            session_cli.display().to_string()
        );
        let harness = Harness::new(toml).await;

        let turn1 = harness
            .post_json(
                "/v1/chat/completions",
                &[("x-session-id", "delta-conv-1")],
                json!({
                    "model": "session/cli",
                    "messages": [{"role": "user", "content": "TURN1_SECRET"}]
                }),
            )
            .await;
        assert_eq!(turn1.status(), StatusCode::OK);
        let body1: Value = serde_json::from_str(&body_text(turn1).await).expect("JSON");
        let text1 = body1["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert!(text1.contains("NEW:"), "text1: {text1}");
        assert!(text1.contains("TURN1_SECRET"), "text1: {text1}");

        let turn2 = harness
            .post_json(
                "/v1/chat/completions",
                &[("x-session-id", "delta-conv-1")],
                json!({
                    "model": "session/cli",
                    "messages": [
                        {"role": "user", "content": "TURN1_SECRET"},
                        {"role": "assistant", "content": "ok"},
                        {"role": "user", "content": "TURN2_DELTA"}
                    ]
                }),
            )
            .await;
        assert_eq!(turn2.status(), StatusCode::OK);
        let body2: Value = serde_json::from_str(&body_text(turn2).await).expect("JSON");
        let text2 = body2["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert!(text2.contains("RESUMED(native-777):"), "text2: {text2}");
        assert!(text2.contains("TURN2_DELTA"), "text2: {text2}");
        assert!(
            !text2.contains("TURN1_SECRET"),
            "turn 2 must only send delta after assistant turn: {text2}"
        );

        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn responses_stateful_previous_response_id_and_zdr() {
        let dir = TempDir::new().expect("tempdir");
        let echo_stdin = script(
            dir.path(),
            "echo_all.sh",
            "#!/bin/sh\ntr '\\n' ' '\necho ''\n",
        );
        let toml = config(
            &cli_backend("echo", &echo_stdin, 10, ""),
            "\"coding/default\" = [\"echo\"]\n",
            "",
            10,
        );
        let harness = Harness::new(toml).await;

        // Turn 1 stored by default
        let r1 = harness
            .post_json(
                "/v1/responses",
                &[],
                json!({
                    "model": "coding/default",
                    "input": "Remember BLUEBERRY"
                }),
            )
            .await;
        assert_eq!(r1.status(), StatusCode::OK);
        let body1: Value = serde_json::from_str(&body_text(r1).await).expect("r1 JSON");
        let id1 = body1["id"].as_str().expect("id1").to_string();

        // Turn 2 references previous_response_id
        let r2 = harness
            .post_json(
                "/v1/responses",
                &[],
                json!({
                    "model": "coding/default",
                    "previous_response_id": id1,
                    "input": "What fruit?"
                }),
            )
            .await;
        assert_eq!(r2.status(), StatusCode::OK);
        let body2: Value = serde_json::from_str(&body_text(r2).await).expect("r2 JSON");
        let out2 = body2["output"][0]["content"][0]["text"]
            .as_str()
            .unwrap_or("");
        assert!(out2.contains("BLUEBERRY"), "expected history: {out2}");
        assert!(out2.contains("What fruit?"), "expected new input: {out2}");

        // ZDR (store = false) must not be persisted
        let zdr = harness
            .post_json(
                "/v1/responses",
                &[],
                json!({
                    "model": "coding/default",
                    "store": false,
                    "input": "ephemeral secret"
                }),
            )
            .await;
        assert_eq!(zdr.status(), StatusCode::OK);
        let zdr_body: Value = serde_json::from_str(&body_text(zdr).await).expect("zdr JSON");
        assert_eq!(zdr_body["store"], false);
        let zdr_id = zdr_body["id"].as_str().expect("zdr id");

        let miss = harness
            .post_json(
                "/v1/responses",
                &[],
                json!({
                    "model": "coding/default",
                    "previous_response_id": zdr_id,
                    "input": "check"
                }),
            )
            .await;
        assert_eq!(miss.status(), StatusCode::NOT_FOUND);

        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mcp_2026_stateless_discover_headers_and_async_tasks() {
        let dir = TempDir::new().expect("tempdir");
        let plain = script(dir.path(), "plain.sh", PLAIN_SCRIPT);
        let slow = script(dir.path(), "slow.sh", "#!/bin/sh\nsleep 10\necho awake\n");
        let toml = format!(
            r#"
[defaults]
model = "coding/default"

{}
{}
[routes]
"coding/default" = ["plain"]
"coding/slow" = ["slow"]
"#,
            cli_backend("plain", &plain, 10, ""),
            cli_backend("slow", &slow, 20, "")
        );
        let harness = Harness::new(toml).await;

        // Stateless server/discover works with zero prior initialize handshake.
        let disc = harness
            .post_json(
                "/mcp",
                &[("mcp-method", "server/discover")],
                json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "server/discover",
                    "params": {
                        "_meta": {
                            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
                        }
                    }
                }),
            )
            .await;
        assert_eq!(disc.status(), StatusCode::OK);
        let disc_json: Value = serde_json::from_str(&body_text(disc).await).expect("discover");
        assert_eq!(disc_json["result"]["protocolVersion"], "2026-07-28");
        assert_eq!(disc_json["result"]["stateless"], true);

        // Mismatched Mcp-Method header fails with -32600.
        let bad_hdr = harness
            .post_json(
                "/mcp",
                &[("mcp-method", "tools/call")],
                json!({"jsonrpc": "2.0", "id": 2, "method": "ping"}),
            )
            .await;
        assert_eq!(bad_hdr.status(), StatusCode::BAD_REQUEST);
        let bad_hdr_json: Value = serde_json::from_str(&body_text(bad_hdr).await).expect("err");
        assert_eq!(bad_hdr_json["error"]["code"], -32600);

        // Mismatched Mcp-Name header fails with -32600.
        let bad_name = harness
            .post_json(
                "/mcp",
                &[("mcp-method", "tools/call"), ("mcp-name", "wrong_tool")],
                json!({
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "vouch_ask", "arguments": {"prompt": "hi"}}
                }),
            )
            .await;
        assert_eq!(bad_name.status(), StatusCode::BAD_REQUEST);

        // tools/list includes 2026-07-28 cache hints.
        let list = harness
            .post_json(
                "/mcp",
                &[],
                json!({"jsonrpc": "2.0", "id": 4, "method": "tools/list"}),
            )
            .await;
        assert_eq!(list.status(), StatusCode::OK);
        let list_json: Value = serde_json::from_str(&body_text(list).await).expect("list");
        assert_eq!(list_json["result"]["ttlMs"], 60_000);
        assert_eq!(list_json["result"]["cacheScope"], "private");

        // tasks/create -> tasks/result (waiting) completes cleanly.
        let created = harness
            .post_json(
                "/mcp",
                &[("mcp-method", "tasks/create")],
                json!({
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tasks/create",
                    "params": {
                        "name": "route_coding_default",
                        "arguments": {"prompt": "async task ping"}
                    }
                }),
            )
            .await;
        assert_eq!(created.status(), StatusCode::OK);
        let created_json: Value = serde_json::from_str(&body_text(created).await).expect("task");
        assert_eq!(created_json["result"]["status"], "working");
        let task_id = created_json["result"]["taskId"].as_str().expect("taskId");

        let res = harness
            .post_json(
                "/mcp",
                &[],
                json!({
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": "tasks/result",
                    "params": {"taskId": task_id, "waitMs": 5000}
                }),
            )
            .await;
        assert_eq!(res.status(), StatusCode::OK);
        let res_json: Value = serde_json::from_str(&body_text(res).await).expect("result");
        assert_eq!(res_json["result"]["status"], "completed");
        assert_eq!(res_json["result"]["isError"], false);

        // tasks/cancel cancels a long-running task.
        let created_slow = harness
            .post_json(
                "/mcp",
                &[],
                json!({
                    "jsonrpc": "2.0",
                    "id": 7,
                    "method": "tasks/create",
                    "params": {
                        "name": "route_coding_slow",
                        "arguments": {"prompt": "slow"}
                    }
                }),
            )
            .await;
        let slow_json: Value = serde_json::from_str(&body_text(created_slow).await).expect("slow");
        let slow_id = slow_json["result"]["taskId"].as_str().expect("slow taskId");

        let cancelled = harness
            .post_json(
                "/mcp",
                &[],
                json!({
                    "jsonrpc": "2.0",
                    "id": 8,
                    "method": "tasks/cancel",
                    "params": {"taskId": slow_id}
                }),
            )
            .await;
        assert_eq!(cancelled.status(), StatusCode::OK);
        let cancelled_json: Value =
            serde_json::from_str(&body_text(cancelled).await).expect("cancelled");
        assert_eq!(cancelled_json["result"]["status"], "cancelled");

        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn anthropic_extended_thinking_and_cache_control_passthrough() {
        async fn mock_thinking_upstream(Json(payload): Json<Value>) -> Response {
            assert_eq!(payload["thinking"]["type"], "enabled");
            assert_eq!(payload["thinking"]["budget_tokens"], 1024);
            let has_cache = payload["messages"].as_array().is_some_and(|msgs| {
                msgs.iter().any(|m| {
                    m["content"]
                        .as_array()
                        .is_some_and(|parts| parts.iter().any(|p| p.get("cache_control").is_some()))
                })
            });
            assert!(
                has_cache,
                "cache_control should be preserved upstream: {payload}"
            );

            if payload.get("stream").and_then(Value::as_bool) == Some(true) {
                let sse = concat!(
                    "event: content_block_delta\n",
                    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Let me reason.\"}}\n\n",
                    "event: content_block_delta\n",
                    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_abc\"}}\n\n",
                    "event: content_block_delta\n",
                    "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Final answer\"}}\n\n",
                    "event: message_delta\n",
                    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":9}}\n\n",
                    "event: message_stop\n",
                    "data: {\"type\":\"message_stop\"}\n\n"
                );
                return Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "text/event-stream")
                    .body(Body::from(sse))
                    .expect("SSE response");
            }

            Json(json!({
                "id": "msg_up_1",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Step by step", "signature": "sig_xyz"},
                    {"type": "redacted_thinking", "data": "opaque_blob"},
                    {"type": "text", "text": "42"}
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 20, "output_tokens": 12}
            }))
            .into_response()
        }

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let app = AxumRouter::new().route("/v1/messages", post(mock_thinking_upstream));
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let toml = format!(
            r#"
[defaults]
model = "claude/thinking"

[[backends]]
id = "anth_up"
kind = "http"
base_url = "http://{addr}"
upstream_path = "/v1/messages"

[routes]
"claude/thinking" = ["anth_up"]
"#
        );
        let harness = Harness::new(toml).await;

        let req_body = |stream: bool| {
            json!({
                "model": "claude/thinking",
                "max_tokens": 2048,
                "stream": stream,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "Compute answer",
                        "cache_control": {"type": "ephemeral"}
                    }]
                }]
            })
        };

        let ns = harness
            .post_json("/v1/messages", &[], req_body(false))
            .await;
        assert_eq!(ns.status(), StatusCode::OK);
        let ns_json: Value = serde_json::from_str(&body_text(ns).await).expect("JSON");
        let blocks = ns_json["content"].as_array().expect("content array");
        assert_eq!(blocks[0]["type"], "thinking");
        assert_eq!(blocks[0]["thinking"], "Step by step");
        assert_eq!(blocks[0]["signature"], "sig_xyz");
        assert_eq!(blocks[1]["type"], "redacted_thinking");
        assert_eq!(blocks[2]["type"], "text");
        assert_eq!(blocks[2]["text"], "42");

        let s = harness.post_json("/v1/messages", &[], req_body(true)).await;
        assert_eq!(s.status(), StatusCode::OK);
        let sse = body_text(s).await;
        assert!(sse.contains("thinking_delta"), "sse: {sse}");
        assert!(sse.contains("Let me reason."), "sse: {sse}");
        assert!(sse.contains("signature_delta"), "sse: {sse}");
        assert!(sse.contains("Final answer"), "sse: {sse}");

        harness.state.shutdown().await;
        task.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bench_runner_collects_latency_and_ttft() {
        let dir = TempDir::new().expect("tempdir");
        let plain = script(dir.path(), "plain.sh", PLAIN_SCRIPT);
        let toml = format!(
            r#"
[defaults]
model = "coding/default"

[[backends]]
id = "plain"
kind = "cli"
command = {:?}
stdin = true
max_concurrency = 2
queue_timeout_ms = 2000
timeout_secs = 10

[routes]
"coding/default" = ["plain"]
"#,
            plain.display().to_string()
        );
        let harness = Harness::new(toml).await;

        let report = bench::run_benchmark(&harness.state, "coding/default", 4, 2, "ping", true)
            .await
            .expect("bench report");
        assert_eq!(report.succeeded, 4);
        assert_eq!(report.failed, 0);
        assert!(report.total_p95_ms >= report.ttft_min_ms);
        assert!(report.rps > 0.0);

        harness.state.shutdown().await;
    }
}

mod v3_tests {
    use super::*;
    use axum::{
        Router as AxumRouter,
        extract::State as AxumState,
        http::header,
        response::IntoResponse,
        routing::{get, post},
    };
    use futures::{SinkExt, StreamExt};
    use tokio::net::TcpListener;
    use tokio_tungstenite::tungstenite::{Message as WsMessage, client::IntoClientRequest};
    use vouch::cli::{InitArgs, StatsArgs, init, stats};

    #[tokio::test]
    async fn example_config_parses_cleanly() {
        let example = include_str!("../vouch.example.toml");
        let parsed = GatewayConfig::from_toml_str(example).expect("vouch.example.toml must parse");
        assert!(!parsed.aliases.is_empty());
        assert!(!parsed.routing_rules.is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn chunk_mode_streams_before_newline_and_preserves_split_utf8() {
        let dir = TempDir::new().expect("tempdir");
        // Emits "hello " + first 2 bytes of 4-byte '🦀' WITHOUT newline, sleeps 250ms,
        // then emits trailing 2 bytes + " world\n".
        let chunk_script = script(
            dir.path(),
            "split_utf8.sh",
            r#"#!/bin/sh
cat >/dev/null
printf 'hello \360\237'
sleep 0.25
printf '\246\200 world\n'
"#,
        );
        let toml = format!(
            r#"
[defaults]
model = "chunk/default"

{}
{}
[routes]
"chunk/default" = ["chunked"]
"line/default" = ["lined"]
"#,
            cli_backend_v3("chunked", &chunk_script, "chunk"),
            cli_backend_v3("lined", &chunk_script, "line"),
        );
        let harness = Harness::new(toml).await;

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let app = harness.app.clone();
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let client = reqwest::Client::new();
        let started = tokio::time::Instant::now();
        let mut resp = client
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&json!({
                "model": "chunk/default",
                "stream": true,
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .expect("send chunk stream");
        assert_eq!(resp.status(), StatusCode::OK);

        // Read SSE chunks until we observe the first non-empty content delta ("hello ").
        let mut first_content_elapsed = None;
        let mut accumulated_sse = String::new();
        while let Some(chunk) = resp.chunk().await.expect("chunk") {
            let text = String::from_utf8_lossy(&chunk);
            accumulated_sse.push_str(&text);
            if first_content_elapsed.is_none() && accumulated_sse.contains("\"content\":\"hello \"")
            {
                first_content_elapsed = Some(started.elapsed());
            }
        }
        let first_elapsed = first_content_elapsed.expect("must stream hello delta");
        assert!(
            first_elapsed < Duration::from_millis(220),
            "chunk mode should stream before child finishes sleeping (elapsed={first_elapsed:?})"
        );
        assert!(
            accumulated_sse.contains("🦀 world"),
            "split 4-byte UTF-8 sequence must reconstruct intact: {accumulated_sse}"
        );

        // Contrast: read_mode = "line" holds bytes until newline/EOF arrives (~250ms).
        let started_line = tokio::time::Instant::now();
        let mut line_resp = client
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&json!({
                "model": "line/default",
                "stream": true,
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .expect("send line stream");
        let mut first_line_content = None;
        let mut line_sse = String::new();
        while let Some(chunk) = line_resp.chunk().await.expect("chunk") {
            line_sse.push_str(&String::from_utf8_lossy(&chunk));
            if first_line_content.is_none() && line_sse.contains("\"content\":\"hello ") {
                first_line_content = Some(started_line.elapsed());
            }
        }
        assert!(
            first_line_content.expect("line content") >= Duration::from_millis(200),
            "line mode must wait until newline arrives"
        );
        assert!(line_sse.contains("🦀 world"), "line sse: {line_sse}");

        server.abort();
        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn streams_split_tool_call_tags_without_leaking_xml() {
        let dir = TempDir::new().expect("tempdir");
        let tool_script = script(
            dir.path(),
            "split_tool.sh",
            r#"#!/bin/sh
cat >/dev/null
printf 'Checking files now. <vouch_to'
sleep 0.05
printf 'ol_call>{"name":"list_dir",'
sleep 0.05
printf '"arguments":{"path":"."}}</vouch_tool_call>\n'
"#,
        );
        let toml = format!(
            r#"
[defaults]
model = "tools/stream"

{}
[routes]
"tools/stream" = ["splitter"]
"#,
            cli_backend_v3("splitter", &tool_script, "chunk")
        );
        let harness = Harness::new(toml).await;

        let resp = harness
            .post_json(
                "/v1/chat/completions",
                &[],
                json!({
                    "model": "tools/stream",
                    "stream": true,
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "list_dir",
                            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
                        }
                    }],
                    "messages": [{"role": "user", "content": "list workspace"}]
                }),
            )
            .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let sse = body_text(resp).await;
        assert!(
            sse.contains("Checking files now. "),
            "prose should stream: {sse}"
        );
        assert!(
            !sse.contains("<vouch_tool_call") && !sse.contains("</vouch_tool_call"),
            "XML tag leaked in SSE stream: {sse}"
        );
        assert!(
            sse.contains("\"name\":\"list_dir\""),
            "missing tool call: {sse}"
        );
        assert!(
            sse.contains("\"finish_reason\":\"tool_calls\""),
            "missing tool_calls finish_reason: {sse}"
        );

        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn streaming_response_format_strips_split_fences_and_validates_schema() {
        let dir = TempDir::new().expect("tempdir");
        let ok_json = script(
            dir.path(),
            "ok_fenced.sh",
            r#"#!/bin/sh
cat >/dev/null
printf '```js'
printf 'on\n{"city":'
printf '"Osaka"}\n``'
printf '`\n'
"#,
        );
        let bad_json = script(
            dir.path(),
            "bad_json.sh",
            r#"#!/bin/sh
cat >/dev/null
printf '```json\n{"city": 123}\n```\n'
"#,
        );
        let toml = format!(
            r#"
[defaults]
model = "json/ok"

{}
{}
[routes]
"json/ok" = ["ok"]
"json/bad" = ["bad"]
"#,
            cli_backend_v3("ok", &ok_json, "chunk"),
            cli_backend_v3("bad", &bad_json, "chunk"),
        );
        let harness = Harness::new(toml).await;
        let schema = json!({
            "type": "object",
            "required": ["city"],
            "properties": {"city": {"type": "string"}}
        });

        let ok_resp = harness
            .post_json(
                "/v1/chat/completions",
                &[],
                json!({
                    "model": "json/ok",
                    "stream": true,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "loc", "schema": schema}
                    },
                    "messages": [{"role": "user", "content": "city"}]
                }),
            )
            .await;
        assert_eq!(ok_resp.status(), StatusCode::OK);
        let ok_sse = body_text(ok_resp).await;
        assert!(!ok_sse.contains("```"), "markdown fence leaked: {ok_sse}");
        assert!(ok_sse.contains("Osaka"), "JSON body missing: {ok_sse}");
        assert!(
            ok_sse.contains("\"finish_reason\":\"stop\""),
            "expected stop: {ok_sse}"
        );

        let bad_resp = harness
            .post_json(
                "/v1/chat/completions",
                &[],
                json!({
                    "model": "json/bad",
                    "stream": true,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "loc", "schema": schema}
                    },
                    "messages": [{"role": "user", "content": "city"}]
                }),
            )
            .await;
        let bad_sse = body_text(bad_resp).await;
        assert!(
            bad_sse.contains("\"code\":\"stream_error\""),
            "invalid schema should emit structured stream_error: {bad_sse}"
        );
        assert!(
            !bad_sse.contains("\"finish_reason\":\"stop\""),
            "invalid schema must not emit stop: {bad_sse}"
        );

        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bounded_stores_and_state_dir_persistence_survive_restart() {
        let dir = TempDir::new().expect("tempdir");
        let state_dir = dir.path().join("state");
        let echo_script = script(
            dir.path(),
            "echo_ok.sh",
            "#!/bin/sh\ncat >/dev/null\nprintf 'persisted-ok'\n",
        );
        let toml = format!(
            r#"
[defaults]
model = "coding/default"
max_session_entries = 2
state_dir = {:?}

{}
[routes]
"coding/default" = ["echo"]
"#,
            state_dir.display().to_string(),
            cli_backend_v3("echo", &echo_script, "chunk")
        );

        // First boot: insert 3 responses; oldest must be evicted (cap = 2).
        let h1 = Harness::new(toml.clone()).await;
        let mut ids = Vec::new();
        for idx in 0..3 {
            let r = h1
                .post_json(
                    "/v1/responses",
                    &[("x-session-id", &format!("sess-{idx}"))],
                    json!({"model": "coding/default", "input": format!("turn {idx}"), "store": true}),
                )
                .await;
            assert_eq!(r.status(), StatusCode::OK);
            let body: Value = serde_json::from_str(&body_text(r).await).expect("JSON");
            ids.push(body["id"].as_str().expect("id").to_string());
        }
        assert_eq!(h1.state.responses().len(), 2);
        assert_eq!(h1.state.router().sessions().len(), 2);

        // Oldest response was evicted
        let oldest = h1
            .post_json(
                "/v1/responses",
                &[],
                json!({"model": "coding/default", "previous_response_id": ids[0], "input": "follow"}),
            )
            .await;
        assert_eq!(oldest.status(), StatusCode::NOT_FOUND);

        h1.state.shutdown().await;
        let state_file = state_dir.join("gateway_state.json");
        assert!(state_file.exists());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&state_file)
                .expect("meta")
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600);
        }

        // Second boot on same state_dir reloads responses, sessions, and usage counters.
        let h2 = Harness::new(toml).await;
        let resumed = h2
            .post_json(
                "/v1/responses",
                &[],
                json!({"model": "coding/default", "previous_response_id": ids[2], "input": "continued"}),
            )
            .await;
        assert_eq!(resumed.status(), StatusCode::OK);

        let usage = h2.get("/v1/usage", &[]).await;
        assert_eq!(usage.status(), StatusCode::OK);
        let usage_json: Value = serde_json::from_str(&body_text(usage).await).expect("usage JSON");
        assert!(
            usage_json["totals"]["requests"].as_u64().unwrap_or(0) >= 4,
            "usage totals should include previous boot: {usage_json}"
        );
        h2.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn remote_image_url_ssrf_blocked_and_allowed_fetch_cleans_up() {
        let dir = TempDir::new().expect("tempdir");
        let inspect = script(
            dir.path(),
            "inspect_prompt.sh",
            r#"#!/bin/sh
prompt="$(cat)"
printf 'SAW:%s\n' "$prompt"
"#,
        );

        async fn png_handler() -> Response {
            // Minimal valid 1x1 PNG header bytes
            let png: &[u8] = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR";
            ([(header::CONTENT_TYPE, "image/png")], png).into_response()
        }
        async fn html_handler() -> Response {
            ([(header::CONTENT_TYPE, "text/html")], "<html>secret</html>").into_response()
        }
        async fn redirect_to_png() -> Response {
            (
                StatusCode::FOUND,
                [(header::LOCATION, "/img.png")],
                "redirecting",
            )
                .into_response()
        }

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let mock_app = AxumRouter::new()
            .route("/img.png", get(png_handler))
            .route("/redir", get(redirect_to_png))
            .route("/bad.html", get(html_handler));
        let mock_srv = tokio::spawn(async move {
            let _ = axum::serve(listener, mock_app).await;
        });

        // 1) SSRF guard: fetch_remote_urls = true, allow_private_ips = false -> blocks 127.0.0.1
        let strict_toml = format!(
            r#"
[defaults]
model = "vision/default"

[attachments]
fetch_remote_urls = true
allow_private_ips = false
max_image_bytes = 1024

{}
[routes]
"vision/default" = ["inspect"]
"#,
            cli_backend_v3("inspect", &inspect, "chunk")
        );
        let h_strict = Harness::new(strict_toml).await;
        let ssrf_resp = h_strict
            .post_json(
                "/v1/chat/completions",
                &[],
                json!({
                    "model": "vision/default",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {"type": "image_url", "image_url": {"url": format!("http://{addr}/redir")}}
                        ]
                    }]
                }),
            )
            .await;
        assert_eq!(ssrf_resp.status(), StatusCode::BAD_REQUEST);
        assert!(body_text(ssrf_resp).await.contains("blocked non-public IP"));
        h_strict.state.shutdown().await;

        // 2) Explicitly allowed local test fetch: follows redirect, downloads PNG to 0600 temp file, cleans up on drop!
        let allow_toml = format!(
            r#"
[defaults]
model = "vision/default"

[attachments]
fetch_remote_urls = true
allow_private_ips = true
max_image_bytes = 1024

{}
[routes]
"vision/default" = ["inspect"]
"#,
            cli_backend_v3("inspect", &inspect, "chunk")
        );
        let h_allow = Harness::new(allow_toml).await;

        let bad_mime = h_allow
            .post_json(
                "/v1/chat/completions",
                &[],
                json!({
                    "model": "vision/default",
                    "messages": [{
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": format!("http://{addr}/bad.html")}}]
                    }]
                }),
            )
            .await;
        assert_eq!(bad_mime.status(), StatusCode::BAD_REQUEST);

        let ok_resp = h_allow
            .post_json(
                "/v1/chat/completions",
                &[],
                json!({
                    "model": "vision/default",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {"type": "image_url", "image_url": {"url": format!("http://{addr}/redir")}}
                        ]
                    }]
                }),
            )
            .await;
        assert_eq!(ok_resp.status(), StatusCode::OK);
        let ok_json: Value = serde_json::from_str(&body_text(ok_resp).await).expect("JSON");
        let text = ok_json["choices"][0]["message"]["content"]
            .as_str()
            .expect("content");
        assert!(text.contains("[Attached image file: "), "text: {text}");
        let start =
            text.find("[Attached image file: ").expect("marker") + "[Attached image file: ".len();
        let end = text[start..].find(']').expect("end bracket") + start;
        let att_path = Path::new(&text[start..end]);
        assert!(
            !att_path.exists(),
            "downloaded remote attachment must be unlinked on TempAttachmentGuard drop"
        );

        h_allow.state.shutdown().await;
        mock_srv.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_responses_streams_and_cancels_child_pgid() {
        let dir = TempDir::new().expect("tempdir");
        let pid_file = dir.path().join("slow.pid");
        let fast = script(
            dir.path(),
            "ws_fast.sh",
            "#!/bin/sh\ncat >/dev/null\nprintf 'hello from '\nsleep 0.03\nprintf 'websocket'\n",
        );
        let slow = script(
            dir.path(),
            "ws_slow.sh",
            &format!(
                "#!/bin/sh\ncat >/dev/null\necho $$ > {:?}\nprintf 'starting... '\nsleep 30\n",
                pid_file.display().to_string()
            ),
        );
        let toml = format!(
            r#"
[defaults]
model = "ws/fast"

{}
{}
[routes]
"ws/fast" = ["fast"]
"ws/slow" = ["slow"]

[[tokens]]
token = "vouch-ws-secret-token"
name = "ws-client"
"#,
            cli_backend_v3("fast", &fast, "chunk"),
            cli_backend_v3("slow", &slow, "chunk")
        );
        let harness = Harness::new(toml).await;
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let app = harness.app.clone();
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        // Unauthenticated WS upgrade is rejected
        let unauth_req = format!("ws://{addr}/v1/responses")
            .into_client_request()
            .expect("req");
        assert!(tokio_tungstenite::connect_async(unauth_req).await.is_err());

        // Authenticated WS upgrade
        let mut req = format!("ws://{addr}/v1/responses")
            .into_client_request()
            .expect("req");
        req.headers_mut().insert(
            "authorization",
            "Bearer vouch-ws-secret-token".parse().expect("hdr"),
        );
        let (mut ws, _) = tokio_tungstenite::connect_async(req)
            .await
            .expect("ws connect");

        ws.send(WsMessage::Text(
            json!({
                "type": "response.create",
                "model": "ws/fast",
                "input": "hi ws"
            })
            .to_string()
            .into(),
        ))
        .await
        .expect("send create");

        let mut saw_created = false;
        let mut saw_delta = String::new();
        let mut completed_id = None;
        while let Some(Ok(frame)) = ws.next().await {
            if let WsMessage::Text(txt) = frame {
                let v: Value = serde_json::from_str(&txt).expect("json frame");
                match v["type"].as_str() {
                    Some("response.created") => saw_created = true,
                    Some("response.output_text.delta") => {
                        saw_delta.push_str(v["delta"].as_str().unwrap_or(""));
                    }
                    Some("response.completed") => {
                        completed_id = v["response"]["id"].as_str().map(ToString::to_string);
                        break;
                    }
                    other => panic!("unexpected frame: {other:?} {v}"),
                }
            }
        }
        assert!(saw_created);
        assert_eq!(saw_delta, "hello from websocket");
        assert!(completed_id.is_some());

        // Start slow request, wait for slow child PID & response.created, then send response.cancel
        ws.send(WsMessage::Text(
            json!({
                "type": "response.create",
                "model": "ws/slow",
                "input": "slow"
            })
            .to_string()
            .into(),
        ))
        .await
        .expect("send slow");

        let mut seen_slow_created = false;
        while !seen_slow_created || !pid_file.exists() {
            tokio::select! {
                _ = sleep(Duration::from_millis(20)) => {}
                frame = ws.next() => {
                    if let Some(Ok(WsMessage::Text(txt))) = frame {
                        let v: Value = serde_json::from_str(&txt).expect("json");
                        if v["type"] == "response.created" {
                            seen_slow_created = true;
                        }
                    }
                }
            }
        }
        let pid_str = fs::read_to_string(&pid_file).expect("pid file");
        let pid: u32 = pid_str.trim().parse().expect("numeric pid");

        ws.send(WsMessage::Text(
            json!({"type": "response.cancel"}).to_string().into(),
        ))
        .await
        .expect("send cancel");

        // Poll up to 2s for PGID guard to terminate slow subprocess
        let mut reaped = false;
        for _ in 0..40 {
            if !Path::new(&format!("/proc/{pid}")).exists() {
                reaped = true;
                break;
            }
            sleep(Duration::from_millis(50)).await;
        }
        assert!(
            reaped,
            "slow child pid {pid} should be reaped after response.cancel"
        );

        let _ = ws.close(None).await;
        server.abort();
        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn aliases_routing_rules_daily_budget_and_usage_analytics() {
        let dir = TempDir::new().expect("tempdir");
        let mk_tag_script = |name: &str| {
            script(
                dir.path(),
                name,
                &format!("#!/bin/sh\ncat >/dev/null\nprintf 'TAG:{name}:1234567890'\n"),
            )
        };
        let b_def = mk_tag_script("def.sh");
        let b_tools = mk_tag_script("tools.sh");
        let b_img = mk_tag_script("img.sh");
        let b_long = mk_tag_script("long.sh");
        let b_vip = mk_tag_script("vip.sh");

        let toml = format!(
            r#"
[defaults]
model = "gpt-4o"

{}
{}
{}
{}
{}

[routes]
"coding/default" = ["def"]
"coding/tools" = ["tools"]
"coding/vision" = ["img"]
"coding/long" = ["long"]
"coding/vip" = ["vip"]

[aliases]
"gpt-4o" = "coding/default"

[[routing_rules]]
name = "header-vip"
model_pattern = "*"
header_equals = {{ "x-tier" = "vip" }}
target_route = "coding/vip"

[[routing_rules]]
name = "tools-route"
model_pattern = "*"
has_tools = true
target_route = "coding/tools"

[[routing_rules]]
name = "image-route"
model_pattern = "*"
has_images = true
target_route = "coding/vision"

[[routing_rules]]
name = "long-route"
model_pattern = "*"
min_prompt_tokens = 25
target_route = "coding/long"

[[tokens]]
token = "vouch-alias-token"
name = "alias-only"
allowed_models = ["gpt-4o"]

[[tokens]]
token = "vouch-route-token"
name = "route-and-budget"
daily_token_budget = 25
allowed_models = ["coding/*"]
"#,
            cli_backend_v3("def", &b_def, "chunk"),
            cli_backend_v3("tools", &b_tools, "chunk"),
            cli_backend_v3("img", &b_img, "chunk"),
            cli_backend_v3("long", &b_long, "chunk"),
            cli_backend_v3("vip", &b_vip, "chunk"),
        );
        let harness = Harness::new(toml).await;

        // GET /v1/models exposes aliases alongside canonical routes
        let models_resp = harness
            .get(
                "/v1/models",
                &[("authorization", "Bearer vouch-alias-token")],
            )
            .await;
        assert_eq!(models_resp.status(), StatusCode::OK);
        let models_json: Value = serde_json::from_str(&body_text(models_resp).await).expect("JSON");
        let ids: Vec<&str> = models_json["data"]
            .as_array()
            .expect("data")
            .iter()
            .filter_map(|m| m["id"].as_str())
            .collect();
        assert!(
            ids.contains(&"gpt-4o"),
            "alias missing from /v1/models: {ids:?}"
        );
        assert!(ids.contains(&"coding/default"));

        // Token allowing ["gpt-4o"] works on "gpt-4o", while ["coding/*"] also works on "gpt-4o"!
        let r_alias = harness
            .post_json(
                "/v1/chat/completions",
                &[("authorization", "Bearer vouch-alias-token")],
                chat_body("gpt-4o", false),
            )
            .await;
        assert_eq!(r_alias.status(), StatusCode::OK);
        assert_eq!(
            r_alias
                .headers()
                .get("x-vouch-backend")
                .and_then(|v| v.to_str().ok()),
            Some("def")
        );

        let r_vip = harness
            .post_json(
                "/v1/chat/completions",
                &[
                    ("authorization", "Bearer vouch-alias-token"),
                    ("x-tier", "vip"),
                ],
                chat_body("gpt-4o", false),
            )
            .await;
        assert_eq!(
            r_vip
                .headers()
                .get("x-vouch-backend")
                .and_then(|v| v.to_str().ok()),
            Some("vip")
        );

        let r_tools = harness
            .post_json(
                "/v1/chat/completions",
                &[("authorization", "Bearer vouch-alias-token")],
                json!({
                    "model": "gpt-4o",
                    "tools": [{"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}],
                    "messages": [{"role": "user", "content": "hi"}]
                }),
            )
            .await;
        assert_eq!(
            r_tools
                .headers()
                .get("x-vouch-backend")
                .and_then(|v| v.to_str().ok()),
            Some("tools")
        );

        let r_img = harness
            .post_json(
                "/v1/chat/completions",
                &[("authorization", "Bearer vouch-alias-token")],
                json!({
                    "model": "gpt-4o",
                    "messages": [{
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]
                    }]
                }),
            )
            .await;
        assert_eq!(
            r_img
                .headers()
                .get("x-vouch-backend")
                .and_then(|v| v.to_str().ok()),
            Some("img")
        );

        let long_text = "word ".repeat(35);
        let r_long = harness
            .post_json(
                "/v1/chat/completions",
                &[("authorization", "Bearer vouch-alias-token")],
                json!({
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": long_text}]
                }),
            )
            .await;
        assert_eq!(
            r_long
                .headers()
                .get("x-vouch-backend")
                .and_then(|v| v.to_str().ok()),
            Some("long")
        );

        // Daily BPE token budget enforcement on vouch-route-token (allowed_models=["coding/*"])
        // First request resolves alias "gpt-4o" -> "coding/default" and succeeds, exceeding 25 BPE tokens.
        let r_budget_1 = harness
            .post_json(
                "/v1/chat/completions",
                &[("authorization", "Bearer vouch-route-token")],
                json!({
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Please count thirty BPE tokens across this request so the daily limit triggers cleanly!"}]
                }),
            )
            .await;
        assert_eq!(r_budget_1.status(), StatusCode::OK);

        // Second request must be blocked with 429 daily_token_budget_exceeded + Retry-After header
        let r_budget_2 = harness
            .post_json(
                "/v1/chat/completions",
                &[("authorization", "Bearer vouch-route-token")],
                chat_body("coding/default", false),
            )
            .await;
        assert_eq!(r_budget_2.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(r_budget_2.headers().get("retry-after").is_some());
        let err_json: Value = serde_json::from_str(&body_text(r_budget_2).await).expect("err json");
        assert_eq!(err_json["error"]["type"], "rate_limit_error");
        assert_eq!(err_json["error"]["code"], "daily_token_budget_exceeded");

        // GET /v1/usage & vouch::cli::stats::fetch_usage return redacted analytics
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let app = harness.app.clone();
        let srv = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let report = stats::fetch_usage(&StatsArgs {
            url: format!("http://{addr}"),
            unix_socket: None,
            token: Some("vouch-alias-token".into()),
            json: true,
        })
        .await
        .expect("fetch_usage");
        let report_json = serde_json::to_string(&report).expect("json");
        assert!(!report_json.contains("vouch-alias-token"));
        assert!(!report_json.contains("vouch-route-token"));
        assert!(report.totals.requests >= 6);
        assert!(report.totals.total_tokens > 0);
        assert!(!report.backends.is_empty());

        srv.abort();
        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn otlp_trace_exporter_propagates_traceparent_and_redacts() {
        let dir = TempDir::new().expect("tempdir");
        let echo = script(
            dir.path(),
            "otel_echo.sh",
            "#!/bin/sh\ncat >/dev/null\nprintf 'traced response'\n",
        );

        #[derive(Clone, Default)]
        struct OtlpSink {
            batches: std::sync::Arc<parking_lot::Mutex<Vec<Value>>>,
        }
        async fn ingest_otlp(
            AxumState(sink): AxumState<OtlpSink>,
            body: axum::body::Bytes,
        ) -> Response {
            if let Ok(val) = serde_json::from_slice::<Value>(&body) {
                sink.batches.lock().push(val);
            }
            StatusCode::OK.into_response()
        }

        let sink = OtlpSink::default();
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let otel_srv_app = AxumRouter::new()
            .route("/v1/traces", post(ingest_otlp))
            .with_state(sink.clone());
        let otel_srv = tokio::spawn(async move {
            let _ = axum::serve(listener, otel_srv_app).await;
        });

        let toml = format!(
            r#"
[defaults]
model = "coding/default"

[telemetry]
otlp_endpoint = "http://{addr}/v1/traces"
service_name = "vouch-otlp-test"
otlp_headers = {{ Authorization = "Bearer otlp-top-secret-999" }}

{}
[routes]
"coding/default" = ["echo"]
"#,
            cli_backend_v3("echo", &echo, "chunk")
        );
        let harness = Harness::new(toml).await;

        let tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        let resp = harness
            .post_json(
                "/v1/chat/completions",
                &[("traceparent", tp), ("tracestate", "v=1")],
                chat_body("coding/default", false),
            )
            .await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Graceful shutdown flushes queued OTLP batch cleanly
        harness.state.shutdown().await;
        otel_srv.abort();

        let batches = sink.batches.lock();
        assert!(!batches.is_empty(), "OTLP sink received no batches");
        let mut span_names = Vec::new();
        for batch in batches.iter() {
            let serialized = batch.to_string();
            assert!(!serialized.contains("otlp-top-secret-999"));
            let spans = batch["resourceSpans"][0]["scopeSpans"][0]["spans"]
                .as_array()
                .expect("spans");
            for span in spans {
                assert_eq!(span["traceId"], "4bf92f3577b34da6a3ce929d0e0e4736");
                span_names.push(span["name"].as_str().unwrap_or("").to_string());
            }
        }
        assert!(span_names.iter().any(|n| n == "ingress"), "{span_names:?}");
        assert!(
            span_names.iter().any(|n| n == "router.dispatch"),
            "{span_names:?}"
        );
        assert!(
            span_names.iter().any(|n| n == "driver.exec"),
            "{span_names:?}"
        );
    }

    #[tokio::test]
    async fn vouch_init_writes_0600_validated_config_and_guards_overwrite() {
        let dir = TempDir::new().expect("tempdir");
        let target = dir.path().join("vouch.toml");
        init::run(&InitArgs {
            output: target.clone(),
            docker_bridge: false,
            force: false,
        })
        .await
        .expect("vouch init first run");
        assert!(target.exists());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&target).expect("meta").permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
        GatewayConfig::load(&target).expect("generated vouch.toml must pass validation");

        let err = init::run(&InitArgs {
            output: target.clone(),
            docker_bridge: false,
            force: false,
        })
        .await
        .expect_err("must refuse overwrite without --force");
        assert!(err.to_string().contains("--force"));

        init::run(&InitArgs {
            output: target.clone(),
            docker_bridge: true,
            force: true,
        })
        .await
        .expect("vouch init --force should overwrite");
        GatewayConfig::load(&target).expect("docker bridge config must pass validation");
    }
}

mod v4_tests {
    use super::*;
    use futures::StreamExt;
    use vouch::usage::UsageReport;

    fn cli_echo_script(dir: &Path, name: &str) -> PathBuf {
        script(dir, name, "#!/bin/sh\ncat >/dev/null\nprintf 'v4-ok\\n'\n")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_drop_accounts_emitted_tokens_tpm_and_cost() {
        let dir = TempDir::new().expect("tempdir");
        let slow_stream = script(
            dir.path(),
            "slow_stream.sh",
            "#!/bin/sh\ncat >/dev/null\nprintf 'alpha beta gamma delta epsilon zeta eta theta\\n'\nsleep 10\n",
        );
        let toml = format!(
            r#"
[defaults]
model = "coding/default"

[pricing."coding/default"]
input_per_1m_usd = 2.0
output_per_1m_usd = 10.0

{}
[routes]
"coding/default" = ["streamer"]

[[tokens]]
token = "vouch-stream-drop-tok"
name = "drop-tester"
tpm = 500
daily_token_budget = 1000
max_concurrent = 1
"#,
            cli_backend_v3("streamer", &slow_stream, "chunk")
        );
        let harness = Harness::new(toml).await;

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("authorization", "Bearer vouch-stream-drop-tok")
            .body(Body::from(
                serde_json::to_vec(&json!({
                    "model": "coding/default",
                    "stream": true,
                    "messages": [{"role": "user", "content": "emit and hang"}]
                }))
                .expect("serialize"),
            ))
            .expect("req");
        let resp = harness.app.clone().oneshot(req).await.expect("router");
        assert_eq!(resp.status(), StatusCode::OK);
        let mut body_stream = resp.into_body().into_data_stream();

        let mut received = String::new();
        while let Some(Ok(bytes)) = body_stream.next().await {
            received.push_str(&String::from_utf8_lossy(&bytes));
            if received.contains("theta") {
                break;
            }
        }
        assert!(received.contains("alpha"));
        // Drop the response stream before [DONE], simulating client disconnect.
        drop(body_stream);
        sleep(Duration::from_millis(80)).await;

        let usage_resp = harness
            .get(
                "/v1/usage",
                &[("authorization", "Bearer vouch-stream-drop-tok")],
            )
            .await;
        assert_eq!(usage_resp.status(), StatusCode::OK);
        let report: UsageReport =
            serde_json::from_str(&body_text(usage_resp).await).expect("usage JSON");
        assert_eq!(report.tokens.len(), 1);
        assert!(
            report.tokens[0].summary.completion_tokens > 0,
            "stream drop must debit emitted completion tokens: {:?}",
            report.tokens[0]
        );
        assert!(report.tokens[0].daily_tokens_used > 0);
        assert!(report.tokens[0].summary.total_cost_usd > 0.0);

        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn parallel_multi_tool_stream_and_schema_repair_retries() {
        let dir = TempDir::new().expect("tempdir");
        let state_file = dir.path().join("attempt_count");
        let multi_tool = script(
            dir.path(),
            "multi_tool.sh",
            "#!/bin/sh\ncat >/dev/null\nprintf '<vouch_tool_call>{\"name\":\"ping\",\"arguments\":{\"h\":\"a\"}}</vouch_tool_call>'\nprintf '<vouch_tool_call>{\"name\":\"echo\",\"arguments\":{\"m\":\"b\"}}</vouch_tool_call>\\n'\n",
        );
        let repair_cli = script(
            dir.path(),
            "repair.sh",
            &format!(
                "#!/bin/sh\nprompt=$(cat)\ncount=0\nif [ -f '{}' ]; then count=$(cat '{}'); fi\ncount=$((count + 1))\necho \"$count\" > '{}'\nif [ \"$count\" -eq 1 ]; then\n  printf '{{\"city\":\"Paris\"}}\\n'\nelse\n  echo \"$prompt\" | grep -q \"missing required field\" && printf '{{\"city\":\"Paris\",\"temp\":21}}\\n' || printf 'missing-feedback\\n'\nfi\n",
                state_file.display(),
                state_file.display(),
                state_file.display(),
            ),
        );

        let toml = format!(
            r#"
[defaults]
model = "tools/multi"
schema_repair_retries = 1

{}
{}
[routes]
"tools/multi" = ["mt"]
"json/repair" = ["rep"]
"#,
            cli_backend_v3("mt", &multi_tool, "chunk"),
            cli_backend_v3("rep", &repair_cli, "chunk"),
        );
        let harness = Harness::new(toml).await;

        let sse_resp = harness
            .post_json(
                "/v1/chat/completions",
                &[],
                json!({
                    "model": "tools/multi",
                    "stream": true,
                    "tools": [
                        {"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}},
                        {"type": "function", "function": {"name": "echo", "parameters": {"type": "object"}}}
                    ],
                    "messages": [{"role": "user", "content": "run both"}]
                }),
            )
            .await;
        assert_eq!(sse_resp.status(), StatusCode::OK);
        let sse_text = body_text(sse_resp).await;
        assert!(sse_text.contains("\"name\":\"ping\""));
        assert!(sse_text.contains("\"name\":\"echo\""));
        assert!(sse_text.contains("\"index\":0"));
        assert!(sse_text.contains("\"index\":1"));
        assert!(sse_text.contains("\"finish_reason\":\"tool_calls\""));

        // Schema repair retry feeds missing-field error back into attempt #2 cleanly
        let json_resp = harness
            .post_json(
                "/v1/chat/completions",
                &[],
                json!({
                    "model": "json/repair",
                    "stream": false,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "weather",
                            "strict": true,
                            "schema": {
                                "type": "object",
                                "required": ["city", "temp"],
                                "additionalProperties": false,
                                "properties": {
                                    "city": {"type": "string"},
                                    "temp": {"type": "integer"}
                                }
                            }
                        }
                    },
                    "messages": [{"role": "user", "content": "weather in paris"}]
                }),
            )
            .await;
        assert_eq!(json_resp.status(), StatusCode::OK);
        let body: Value = serde_json::from_str(&body_text(json_resp).await).expect("json");
        assert_eq!(
            body["choices"][0]["message"]["content"],
            "{\"city\":\"Paris\",\"temp\":21}"
        );
        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn hedged_and_least_latency_routing_strategies() {
        let dir = TempDir::new().expect("tempdir");
        let slow_pid = dir.path().join("slow_hedged.pid");
        let slow_script = script(
            dir.path(),
            "slow_hedge.sh",
            &format!(
                "#!/bin/sh\necho $$ > '{}'\ncat >/dev/null\nsleep 5\nprintf 'too-late\\n'\n",
                slow_pid.display(),
            ),
        );
        let fast_script = script(
            dir.path(),
            "fast_hedge.sh",
            "#!/bin/sh\ncat >/dev/null\nprintf 'fast-winner\\n'\n",
        );
        let laggy_script = script(
            dir.path(),
            "laggy.sh",
            "#!/bin/sh\ncat >/dev/null\nsleep 0.12\nprintf 'from-slower\\n'\n",
        );
        let quick_script = script(
            dir.path(),
            "quick.sh",
            "#!/bin/sh\ncat >/dev/null\nprintf 'from-quicker\\n'\n",
        );

        let toml = format!(
            r#"
[defaults]
model = "hedged/route"

{}
{}
{}
{}
[routes]
"hedged/route" = ["slow", "fast"]
"ewma/route" = ["slower", "quicker"]

[route_policies."hedged/route"]
strategy = "hedged"
hedge_delay_ms = 60

[route_policies."ewma/route"]
strategy = "least_latency"
"#,
            cli_backend_v3("slow", &slow_script, "chunk"),
            cli_backend_v3("fast", &fast_script, "chunk"),
            cli_backend_v3("slower", &laggy_script, "chunk"),
            cli_backend_v3("quicker", &quick_script, "chunk"),
        );
        let harness = Harness::new(toml).await;

        let resp = harness
            .post_json(
                "/v1/chat/completions",
                &[],
                chat_body("hedged/route", false),
            )
            .await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers()["x-vouch-backend"], "fast");
        assert_eq!(resp.headers()["x-vouch-fallback"], "true");
        let payload: Value = serde_json::from_str(&body_text(resp).await).expect("json");
        assert_eq!(payload["choices"][0]["message"]["content"], "fast-winner");

        // Slow loser must be terminated cleanly and circuit must stay closed
        if let Ok(pid_str) = fs::read_to_string(&slow_pid)
            && let Ok(pid) = pid_str.trim().parse::<u32>()
        {
            let mut reaped = false;
            for _ in 0..40 {
                if !Path::new(&format!("/proc/{pid}")).exists() {
                    reaped = true;
                    break;
                }
                sleep(Duration::from_millis(50)).await;
            }
            assert!(reaped, "hedged loser pid {pid} should be reaped");
        }
        let ready: Value =
            serde_json::from_str(&body_text(harness.get("/readyz", &[]).await).await)
                .expect("readyz");
        let slow_entry = ready["backends"]
            .as_array()
            .expect("backends")
            .iter()
            .find(|b| b["id"] == "slow")
            .expect("slow entry");
        assert_eq!(slow_entry["circuit"], "closed");

        let metrics_text = body_text(harness.get("/metrics", &[]).await).await;
        assert!(metrics_text.contains("vouch_hedged_wins_total"));

        // Least latency: req1 probes slower (120ms), req2 probes unmeasured quicker (~10ms), req3 stays on quicker
        let r1 = harness
            .post_json("/v1/chat/completions", &[], chat_body("ewma/route", false))
            .await;
        assert_eq!(r1.headers()["x-vouch-backend"], "slower");
        let r2 = harness
            .post_json("/v1/chat/completions", &[], chat_body("ewma/route", false))
            .await;
        assert_eq!(r2.headers()["x-vouch-backend"], "quicker");
        let r3 = harness
            .post_json("/v1/chat/completions", &[], chat_body("ewma/route", false))
            .await;
        assert_eq!(r3.headers()["x-vouch-backend"], "quicker");

        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deterministic_cache_hit_miss_bypass_stream_and_persistence() {
        let dir = TempDir::new().expect("tempdir");
        let state_dir = dir.path().join("state");
        let exec_counter = dir.path().join("exec_count");
        let count_script = script(
            dir.path(),
            "count.sh",
            &format!(
                "#!/bin/sh\ncat >/dev/null\ncount=0\nif [ -f '{}' ]; then count=$(cat '{}'); fi\ncount=$((count + 1))\necho \"$count\" > '{}'\nprintf 'cached-reply-%s\\n' \"$count\"\n",
                exec_counter.display(),
                exec_counter.display(),
                exec_counter.display(),
            ),
        );
        let toml = format!(
            r#"
[defaults]
model = "coding/default"
state_dir = "{}"

[cache]
enabled = true
max_entries = 4
ttl_secs = 3600
max_temperature = 0.2
persist = true

{}
[routes]
"coding/default" = ["counter"]
"#,
            state_dir.display(),
            cli_backend_v3("counter", &count_script, "chunk")
        );

        let harness = Harness::new(toml.clone()).await;
        let det_body = json!({
            "model": "coding/default",
            "stream": false,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": "same prompt"}]
        });
        let r1 = harness
            .post_json("/v1/chat/completions", &[], det_body.clone())
            .await;
        assert_eq!(r1.headers()["x-vouch-cache"], "MISS");
        assert_eq!(fs::read_to_string(&exec_counter).expect("cnt").trim(), "1");

        let r2 = harness
            .post_json("/v1/chat/completions", &[], det_body.clone())
            .await;
        assert_eq!(r2.headers()["x-vouch-cache"], "HIT");
        assert_eq!(fs::read_to_string(&exec_counter).expect("cnt").trim(), "1");

        // Streaming replay of cached response should also HIT without re-running backend
        let mut stream_body = det_body.clone();
        stream_body["stream"] = json!(true);
        let r3 = harness
            .post_json("/v1/chat/completions", &[], stream_body)
            .await;
        assert_eq!(r3.headers()["x-vouch-cache"], "HIT");
        let sse = body_text(r3).await;
        assert!(sse.contains("cached-reply-1"));
        assert!(sse.contains("[DONE]"));
        assert_eq!(fs::read_to_string(&exec_counter).expect("cnt").trim(), "1");

        // Cache-Control: no-store and high temperature must BYPASS
        let r4 = harness
            .post_json(
                "/v1/chat/completions",
                &[("cache-control", "no-store")],
                det_body.clone(),
            )
            .await;
        assert_eq!(r4.headers()["x-vouch-cache"], "BYPASS");
        let mut hot = det_body.clone();
        hot["temperature"] = json!(0.8);
        let r5 = harness.post_json("/v1/chat/completions", &[], hot).await;
        assert_eq!(r5.headers()["x-vouch-cache"], "BYPASS");

        harness.state.shutdown().await;
        assert!(state_dir.join("gateway_state.lock").exists());
        assert!(state_dir.join("gateway_state.json").exists());

        // Cold boot restores persisted cache entry and hits without backend execution
        let before_restart = fs::read_to_string(&exec_counter)
            .expect("cnt")
            .trim()
            .to_string();
        let harness2 = Harness::new(toml).await;
        let r6 = harness2
            .post_json("/v1/chat/completions", &[], det_body)
            .await;
        assert_eq!(r6.headers()["x-vouch-cache"], "HIT");
        assert_eq!(
            fs::read_to_string(&exec_counter).expect("cnt").trim(),
            before_restart
        );
        let metrics = body_text(harness2.get("/metrics", &[]).await).await;
        assert!(metrics.contains("vouch_cache_requests_total"));
        harness2.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn gemini_and_ollama_ingress_compatibility() {
        let dir = TempDir::new().expect("tempdir");
        let echo = cli_echo_script(dir.path(), "compat.sh");
        let toml = format!(
            r#"
[defaults]
model = "coding/default"

{}
[routes]
"coding/default" = ["echo"]

[[tokens]]
token = "vouch-compat-key-123"
name = "compat-user"
"#,
            cli_backend_v3("echo", &echo, "chunk")
        );
        let harness = Harness::new(toml).await;

        // Gemini generateContent via x-goog-api-key
        let gem = harness
            .post_json(
                "/v1beta/models/coding/default:generateContent",
                &[("x-goog-api-key", "vouch-compat-key-123")],
                json!({
                    "systemInstruction": {"parts": [{"text": "Be concise"}]},
                    "contents": [{"role": "user", "parts": [{"text": "hi gemini"}]}],
                    "generationConfig": {"temperature": 0.1, "maxOutputTokens": 64}
                }),
            )
            .await;
        assert_eq!(gem.status(), StatusCode::OK);
        let gem_json: Value = serde_json::from_str(&body_text(gem).await).expect("gemini");
        assert_eq!(
            gem_json["candidates"][0]["content"]["parts"][0]["text"],
            "v4-ok"
        );
        assert_eq!(gem_json["candidates"][0]["finishReason"], "STOP");

        // Gemini streamGenerateContent?alt=sse&key=
        let gem_sse = harness
            .post_json(
                "/v1beta/models/coding/default:streamGenerateContent?alt=sse&key=vouch-compat-key-123",
                &[],
                json!({"contents": [{"parts": [{"text": "stream gemini"}]}]}),
            )
            .await;
        assert_eq!(gem_sse.status(), StatusCode::OK);
        let gem_sse_body = body_text(gem_sse).await;
        assert!(gem_sse_body.contains("v4-ok"));

        // Ollama /api/tags
        let tags = harness
            .get(
                "/api/tags",
                &[("authorization", "Bearer vouch-compat-key-123")],
            )
            .await;
        assert_eq!(tags.status(), StatusCode::OK);
        let tags_json: Value = serde_json::from_str(&body_text(tags).await).expect("tags");
        assert_eq!(tags_json["models"][0]["name"], "coding/default");

        // Ollama /api/chat buffered
        let ochat = harness
            .post_json(
                "/api/chat",
                &[("authorization", "Bearer vouch-compat-key-123")],
                json!({
                    "model": "coding/default",
                    "stream": false,
                    "messages": [{"role": "user", "content": "hi ollama"}]
                }),
            )
            .await;
        assert_eq!(ochat.status(), StatusCode::OK);
        let ochat_json: Value = serde_json::from_str(&body_text(ochat).await).expect("ochat");
        assert_eq!(ochat_json["message"]["content"], "v4-ok");
        assert_eq!(ochat_json["done"], true);

        // Ollama /api/generate default streaming NDJSON
        let ogen = harness
            .post_json(
                "/api/generate",
                &[("authorization", "Bearer vouch-compat-key-123")],
                json!({"model": "coding/default", "prompt": "hi gen"}),
            )
            .await;
        assert_eq!(ogen.status(), StatusCode::OK);
        let ndjson = body_text(ogen).await;
        let lines: Vec<Value> = ndjson
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("ndjson line"))
            .collect();
        let joined: String = lines
            .iter()
            .filter_map(|l| l["response"].as_str())
            .collect();
        assert_eq!(joined.trim(), "v4-ok", "ndjson lines={lines:?}");
        assert_eq!(lines.last().expect("last")["done"], true);

        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn multi_profile_rotation_and_failover() {
        let dir = TempDir::new().expect("tempdir");
        let prof_script = script(
            dir.path(),
            "profile_worker.sh",
            "#!/bin/sh\ncat >/dev/null\nif [ \"$VOUCH_PROFILE_ID\" = \"acct-1\" ]; then\n  echo \"HTTP 429 rate limit exceeded Retry-After: 30\" >&2\n  exit 1\nfi\nprintf 'served-by-%s\\n' \"$VOUCH_PROFILE_ID\"\n",
        );
        let toml = format!(
            r#"
[defaults]
model = "coding/default"

[[backends]]
id = "multi-prof"
kind = "cli"
command = "{}"
stdin = true

[[backends.profiles]]
id = "acct-1"
env = {{ ACCOUNT_KEY = "p1-key" }}
weight = 2

[[backends.profiles]]
id = "acct-2"
env = {{ ACCOUNT_KEY = "p2-key" }}
weight = 1

[routes]
"coding/default" = ["multi-prof"]
"#,
            prof_script.display()
        );
        let harness = Harness::new(toml).await;
        let r1 = harness
            .post_json(
                "/v1/chat/completions",
                &[],
                chat_body("coding/default", false),
            )
            .await;
        assert_eq!(r1.status(), StatusCode::OK);
        assert_eq!(r1.headers()["x-vouch-backend"], "multi-prof");
        let payload: Value = serde_json::from_str(&body_text(r1).await).expect("json");
        assert_eq!(
            payload["choices"][0]["message"]["content"],
            "served-by-acct-2"
        );

        // Parent backend circuit remained closed when acct-2 recovered inside backend pool
        let ready: Value =
            serde_json::from_str(&body_text(harness.get("/readyz", &[]).await).await)
                .expect("readyz");
        assert_eq!(ready["backends"][0]["circuit"], "closed");
        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn token_policies_and_admin_endpoints() {
        let dir = TempDir::new().expect("tempdir");
        let trip_script = script(dir.path(), "trip.sh", "#!/bin/sh\ncat >/dev/null\nexit 1\n");
        let ok_script = cli_echo_script(dir.path(), "ok.sh");
        let toml = format!(
            r#"
[server]
enable_admin = true

[resilience]
failure_threshold = 1
reset_timeout_secs = 60

[cache]
enabled = true

{}
{}
[routes]
"coding/default" = ["bad", "good"]

[[tokens]]
token = "vouch-admin-tok"
name = "admin"
admin = true

[[tokens]]
token = "vouch-user-tok"
name = "limited"
allowed_backends = ["good"]
admin = false

[[tokens]]
token = "vouch-no-backend-tok"
name = "blocked"
allowed_backends = ["nonexistent"]

[[tokens]]
token = "vouch-expired-tok"
name = "expired"
expires_at = "2020-01-01T00:00:00Z"
"#,
            cli_backend_v3("bad", &trip_script, "chunk"),
            cli_backend_v3("good", &ok_script, "chunk"),
        );
        let harness = Harness::new(toml).await;

        // Expired token rejected with 401
        let exp = harness
            .post_json(
                "/v1/chat/completions",
                &[("authorization", "Bearer vouch-expired-tok")],
                chat_body("coding/default", false),
            )
            .await;
        assert_eq!(exp.status(), StatusCode::UNAUTHORIZED);

        // allowed_backends with zero intersection -> 403 Forbidden
        let blocked = harness
            .post_json(
                "/v1/chat/completions",
                &[("authorization", "Bearer vouch-no-backend-tok")],
                chat_body("coding/default", false),
            )
            .await;
        assert_eq!(blocked.status(), StatusCode::FORBIDDEN);

        // allowed_backends filtering skips `bad` completely and hits `good` cleanly
        let scoped = harness
            .post_json(
                "/v1/chat/completions",
                &[("authorization", "Bearer vouch-user-tok")],
                chat_body("coding/default", false),
            )
            .await;
        assert_eq!(scoped.status(), StatusCode::OK);
        assert_eq!(scoped.headers()["x-vouch-backend"], "good");

        // Admin call trips `bad` first via admin token then verifies non-admin is rejected 403
        let admin_call = harness
            .post_json(
                "/v1/chat/completions",
                &[("authorization", "Bearer vouch-admin-tok")],
                chat_body("coding/default", false),
            )
            .await;
        assert_eq!(admin_call.status(), StatusCode::OK);
        assert_eq!(admin_call.headers()["x-vouch-backend"], "good");

        // Non-admin user cannot reset circuits or purge cache
        let forbidden_reset = harness
            .post_json(
                "/admin/circuits/bad/reset",
                &[("authorization", "Bearer vouch-user-tok")],
                json!({}),
            )
            .await;
        assert_eq!(forbidden_reset.status(), StatusCode::FORBIDDEN);

        // Admin resets `bad` circuit and unknown backend returns 404
        let reset_ok = harness
            .post_json(
                "/admin/circuits/bad/reset",
                &[("authorization", "Bearer vouch-admin-tok")],
                json!({}),
            )
            .await;
        assert_eq!(reset_ok.status(), StatusCode::OK);
        let reset_404 = harness
            .post_json(
                "/admin/circuits/missing/reset",
                &[("authorization", "Bearer vouch-admin-tok")],
                json!({}),
            )
            .await;
        assert_eq!(reset_404.status(), StatusCode::NOT_FOUND);

        // DELETE /admin/cache purges cache
        let purge_req = Request::builder()
            .method(Method::DELETE)
            .uri("/admin/cache")
            .header("authorization", "Bearer vouch-admin-tok")
            .body(Body::empty())
            .expect("req");
        let purge_resp = harness
            .app
            .clone()
            .oneshot(purge_req)
            .await
            .expect("router");
        assert_eq!(purge_resp.status(), StatusCode::OK);
        let purge_json: Value = serde_json::from_str(&body_text(purge_resp).await).expect("json");
        assert!(purge_json["evicted"].as_u64().expect("evicted") >= 1);

        // POST /admin/reload reloads cleanly
        let reload_resp = harness
            .post_json(
                "/admin/reload",
                &[("authorization", "Bearer vouch-admin-tok")],
                json!({}),
            )
            .await;
        assert_eq!(reload_resp.status(), StatusCode::OK);

        harness.state.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mcp_resources_prompts_and_upstream_aggregation() {
        let dir = TempDir::new().expect("tempdir");
        let echo = cli_echo_script(dir.path(), "echo.sh");
        let mock_upstream = script(
            dir.path(),
            "mock_mcp.sh",
            r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *"tools/list"*)
      printf '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"ping_host","description":"Ping remote","inputSchema":{"type":"object","properties":{"host":{"type":"string"}}}}]}}\n'
      ;;
    *"tools/call"*)
      printf '{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"pong:example.com"}],"isError":false}}\n'
      ;;
  esac
done
"#,
        );
        let toml = format!(
            r#"
[defaults]
model = "coding/default"

{}
[routes]
"coding/default" = ["echo"]

[[mcp_prompts]]
name = "diff-review"
description = "Review repository diff"
template = "Inspect changes in {{{{repo}}}}"
arguments = [{{ name = "repo", description = "Target repo", required = true }}]

[[mcp_upstreams]]
id = "net"
transport = "stdio"
command = "{}"
timeout_secs = 5
"#,
            cli_backend_v3("echo", &echo, "chunk"),
            mock_upstream.display()
        );
        let harness = Harness::new(toml).await;

        let init = harness
            .post_json(
                "/mcp",
                &[],
                json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
            )
            .await;
        let init_json: Value = serde_json::from_str(&body_text(init).await).expect("init");
        assert!(init_json["result"]["capabilities"]["resources"].is_object());
        assert!(init_json["result"]["capabilities"]["prompts"].is_object());

        let rlist = harness
            .post_json(
                "/mcp",
                &[],
                json!({"jsonrpc": "2.0", "id": 2, "method": "resources/list", "params": {}}),
            )
            .await;
        let rlist_json: Value = serde_json::from_str(&body_text(rlist).await).expect("rlist");
        assert_eq!(
            rlist_json["result"]["resources"]
                .as_array()
                .expect("arr")
                .len(),
            3
        );

        for uri in ["vouch://health", "vouch://routes", "vouch://usage"] {
            let rread = harness
                .post_json(
                    "/mcp",
                    &[],
                    json!({"jsonrpc": "2.0", "id": 3, "method": "resources/read", "params": {"uri": uri}}),
                )
                .await;
            let rread_json: Value = serde_json::from_str(&body_text(rread).await).expect("rread");
            assert_eq!(rread_json["result"]["contents"][0]["uri"], uri);
        }

        let plist = harness
            .post_json(
                "/mcp",
                &[],
                json!({"jsonrpc": "2.0", "id": 4, "method": "prompts/list", "params": {}}),
            )
            .await;
        let plist_json: Value = serde_json::from_str(&body_text(plist).await).expect("plist");
        assert_eq!(plist_json["result"]["prompts"][0]["name"], "diff-review");

        let pget = harness
            .post_json(
                "/mcp",
                &[],
                json!({
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "prompts/get",
                    "params": {"name": "diff-review", "arguments": {"repo": "vouch"}}
                }),
            )
            .await;
        let pget_json: Value = serde_json::from_str(&body_text(pget).await).expect("pget");
        assert_eq!(
            pget_json["result"]["messages"][0]["content"]["text"],
            "Inspect changes in vouch"
        );

        let tlist = harness
            .post_json(
                "/mcp",
                &[],
                json!({"jsonrpc": "2.0", "id": 6, "method": "tools/list", "params": {}}),
            )
            .await;
        let tlist_json: Value = serde_json::from_str(&body_text(tlist).await).expect("tlist");
        assert!(
            tlist_json["result"]["tools"]
                .as_array()
                .expect("tools")
                .iter()
                .any(|t| t["name"] == "net__ping_host")
        );

        let tcall = harness
            .post_json(
                "/mcp",
                &[],
                json!({
                    "jsonrpc": "2.0",
                    "id": 7,
                    "method": "tools/call",
                    "params": {"name": "net__ping_host", "arguments": {"host": "example.com"}}
                }),
            )
            .await;
        let tcall_json: Value = serde_json::from_str(&body_text(tcall).await).expect("tcall");
        assert_eq!(
            tcall_json["result"]["content"][0]["text"],
            "pong:example.com"
        );

        harness.state.shutdown().await;
    }
}

mod v5_tests {
    use super::*;
    use base64::Engine as _;

    #[tokio::test]
    async fn hedged_win_attributes_winner_and_cancels_loser_without_500ms_penalty() {
        let dir = TempDir::new().expect("temp");
        // Primary sleeps 3 seconds (ignores fast cancel gracefully or traps SIGTERM), backup responds fast.
        let slow = script(
            dir.path(),
            "slow.sh",
            "#!/bin/sh\nsleep 3\necho 'from-slow'\n",
        );
        let fast = script(dir.path(), "fast.sh", "#!/bin/sh\necho 'from-fast'\n");

        let config = format!(
            r#"
[server]
tcp_listen = "127.0.0.1:0"

[[backends]]
id = "slow"
kind = "cli"
command = {:?}
stdin = true

[[backends]]
id = "fast"
kind = "cli"
command = {:?}
stdin = true

[routes]
"coding/hedged" = ["slow", "fast"]

[route_policies."coding/hedged"]
strategy = "hedged"
hedge_delay_ms = 50
"#,
            slow, fast
        );
        let h = Harness::new(config).await;
        let started = std::time::Instant::now();
        let resp = h
            .post_json(
                "/v1/chat/completions",
                &[],
                json!({
                    "model": "coding/hedged",
                    "messages": [{"role": "user", "content": "hi"}]
                }),
            )
            .await;
        let elapsed = started.elapsed();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers().get("x-vouch-backend").unwrap(), "fast");
        assert_eq!(resp.headers().get("x-vouch-fallback").unwrap(), "true");
        let body: Value = serde_json::from_str(&body_text(resp).await).expect("json");
        assert_eq!(body["choices"][0]["message"]["content"], "from-fast");
        assert!(
            elapsed < Duration::from_millis(900),
            "winner waited on slow teardown: {elapsed:?}"
        );

        // Usage attribution must credit winning backend `fast`, not `slow`.
        let usage = h.get("/v1/usage", &[]).await;
        let usage_json: Value = serde_json::from_str(&body_text(usage).await).expect("usage");
        let backends = usage_json["backends"].as_array().expect("backends");
        assert!(
            backends
                .iter()
                .any(|b| b["id"] == "fast" && b["requests"] == 1)
        );

        h.state.shutdown().await;
    }

    #[tokio::test]
    async fn singleflight_coalescing_deduplicates_concurrent_identical_requests() {
        let dir = TempDir::new().expect("temp");
        let counter_file = dir.path().join("exec_count.txt");
        let responder = script(
            dir.path(),
            "coalesce.sh",
            &format!(
                "#!/bin/sh\necho x >> {:?}\nsleep 0.12\ncat >/dev/null\necho 'coalesced-ok'\n",
                counter_file
            ),
        );

        let config = format!(
            r#"
[server]
tcp_listen = "127.0.0.1:0"

[cache]
enabled = true
coalesce_inflight = true
normalize_whitespace = true

[[backends]]
id = "mock"
kind = "cli"
command = {:?}
stdin = true
max_concurrency = 4

[routes]
"coding/default" = ["mock"]
"#,
            responder
        );
        let h = Harness::new(config).await;

        let req_a = h.post_json(
            "/v1/chat/completions",
            &[],
            json!({"model": "coding/default", "temperature": 0.0, "messages": [{"role": "user", "content": "same   prompt"}]}),
        );
        let req_b = h.post_json(
            "/v1/chat/completions",
            &[],
            json!({"model": "coding/default", "temperature": 0.0, "messages": [{"role": "user", "content": "same prompt"}]}),
        );
        let req_c = h.post_json(
            "/v1/chat/completions",
            &[],
            json!({"model": "coding/default", "temperature": 0.0, "messages": [{"role": "user", "content": "  same\tprompt  "}]}),
        );
        let (r1, r2, r3) = tokio::join!(req_a, req_b, req_c);

        let coalesced_count = [&r1, &r2, &r3]
            .iter()
            .filter(|r| {
                r.headers()
                    .get("x-vouch-coalesced")
                    .is_some_and(|v| v == "true")
            })
            .count();
        assert_eq!(coalesced_count, 2, "two waiters should be coalesced");

        for r in [r1, r2, r3] {
            assert_eq!(r.status(), StatusCode::OK);
            let body: Value = serde_json::from_str(&body_text(r).await).expect("json");
            assert_eq!(body["choices"][0]["message"]["content"], "coalesced-ok");
        }

        let exec_lines = fs::read_to_string(&counter_file)
            .expect("count")
            .lines()
            .count();
        assert_eq!(exec_lines, 1, "backend must execute only once");

        let metrics_text = body_text(h.get("/metrics", &[]).await).await;
        assert!(metrics_text.contains("vouch_coalesced_requests_total"));
        h.state.shutdown().await;
    }

    #[tokio::test]
    async fn pre_execution_budget_reservation_blocks_concurrent_burst_overshoot() {
        let dir = TempDir::new().expect("temp");
        let responder = script(
            dir.path(),
            "slow_budget.sh",
            "#!/bin/sh\nsleep 0.15\necho 'ok'\n",
        );

        let config = format!(
            r#"
[server]
tcp_listen = "127.0.0.1:0"

[[backends]]
id = "mock"
kind = "cli"
command = {:?}
stdin = true
max_concurrency = 4

[routes]
"coding/default" = ["mock"]

[[tokens]]
token = "vouch-budget-secret"
name = "tight"
daily_token_budget = 25
"#,
            responder
        );
        let h = Harness::new(config).await;
        let auth = ("authorization", "Bearer vouch-budget-secret");
        let auth_hdr = [auth];
        // Prompt has > 14 BPE tokens so two concurrent identical requests (~16+16 > 25) cannot both reserve.
        let payload = json!({
            "model": "coding/default",
            "messages": [{"role": "user", "content": "one two three four five six seven eight nine ten eleven twelve"}]
        });
        let r1 = h.post_json("/v1/chat/completions", &auth_hdr, payload.clone());
        let r2 = h.post_json("/v1/chat/completions", &auth_hdr, payload);
        let (a, b) = tokio::join!(r1, r2);
        let statuses = [a.status(), b.status()];
        assert!(statuses.contains(&StatusCode::OK));
        assert!(statuses.contains(&StatusCode::TOO_MANY_REQUESTS));
        h.state.shutdown().await;
    }

    #[tokio::test]
    async fn egress_guard_blocks_and_redacts_and_output_filters_strip_noise() {
        let dir = TempDir::new().expect("temp");
        let spawn_marker = dir.path().join("spawned.txt");
        let echo_script = script(
            dir.path(),
            "echo_stdin.sh",
            &format!(
                "#!/bin/sh\necho 'sp' >> {:?}\necho 'BANNER_NOISE: ignore'\ncat\n",
                spawn_marker
            ),
        );

        // 1. Block mode must return 403 before spawning backend.
        let block_cfg = format!(
            r#"
[server]
tcp_listen = "127.0.0.1:0"

[security.egress_guard]
mode = "block"
block_private_keys = true

[[backends]]
id = "mock"
kind = "cli"
command = {:?}
stdin = true

[routes]
"coding/default" = ["mock"]
"#,
            echo_script
        );
        let h_block = Harness::new(block_cfg).await;
        let blocked = h_block
            .post_json(
                "/v1/chat/completions",
                &[],
                json!({
                    "model": "coding/default",
                    "messages": [{"role": "user", "content": "leak AKIAIOSFODNN7EXAMPLE here"}]
                }),
            )
            .await;
        assert_eq!(blocked.status(), StatusCode::FORBIDDEN);
        let err_json: Value = serde_json::from_str(&body_text(blocked).await).expect("err");
        assert_eq!(err_json["error"]["code"], "egress_policy_blocked");
        assert!(
            !spawn_marker.exists(),
            "blocked request must not spawn backend"
        );
        h_block.state.shutdown().await;

        // 2. Redact mode scrubs secrets outbound & output_filters strip banner lines inbound.
        let redact_cfg = format!(
            r#"
[server]
tcp_listen = "127.0.0.1:0"

[security.egress_guard]
mode = "redact"
block_private_keys = true
extra_patterns = ["CORP_SECRET_[0-9]+"]

[[backends]]
id = "mock"
kind = "cli"
command = {:?}
stdin = true
output_filters = [{{ pattern = "(?m)^BANNER_NOISE:.*\\n?", replace = "" }}]

[routes]
"coding/default" = ["mock"]
"#,
            echo_script
        );
        let h_redact = Harness::new(redact_cfg).await;
        let ok = h_redact
            .post_json(
                "/v1/chat/completions",
                &[],
                json!({
                    "model": "coding/default",
                    "messages": [{"role": "user", "content": "check AKIAIOSFODNN7EXAMPLE and CORP_SECRET_99"}]
                }),
            )
            .await;
        assert_eq!(ok.status(), StatusCode::OK);
        let ok_json: Value = serde_json::from_str(&body_text(ok).await).expect("ok");
        let reply = ok_json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap();
        assert!(reply.contains("[REDACTED_EGRESS_SECRET]"));
        assert!(!reply.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(!reply.contains("CORP_SECRET_99"));
        assert!(!reply.contains("BANNER_NOISE"));
        h_redact.state.shutdown().await;
    }

    #[tokio::test]
    async fn embeddings_route_supports_float_and_base64_formats() {
        let dir = TempDir::new().expect("temp");
        let embed_sh = script(
            dir.path(),
            "embed.sh",
            "#!/bin/sh\ncat >/dev/null\necho '[[0.25, -0.5, 1.0]]'\n",
        );
        let config = format!(
            r#"
[server]
tcp_listen = "127.0.0.1:0"

[[backends]]
id = "emb"
kind = "cli"
command = {:?}
stdin = true

[routes]
"text-embedding-local" = ["emb"]
"#,
            embed_sh
        );
        let h = Harness::new(config).await;
        let resp = h
            .post_json(
                "/v1/embeddings",
                &[],
                json!({
                    "model": "text-embedding-local",
                    "input": "hello world",
                    "encoding_format": "base64"
                }),
            )
            .await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers().get("x-vouch-backend").unwrap(), "emb");
        let json: Value = serde_json::from_str(&body_text(resp).await).expect("json");
        assert_eq!(json["object"], "list");
        let b64 = json["data"][0]["embedding"].as_str().expect("base64");
        let raw = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .expect("decode");
        assert_eq!(raw.len(), 12);
        let first = f32::from_le_bytes(raw[0..4].try_into().unwrap());
        assert!((first - 0.25).abs() < 1e-6);
        h.state.shutdown().await;
    }

    #[tokio::test]
    async fn flight_recorder_traces_ring_buffer_and_replay_work_end_to_end() {
        let dir = TempDir::new().expect("temp");
        let flight_path = dir.path().join("flight.jsonl");
        let echo_sh = script(dir.path(), "echo_ok.sh", "#!/bin/sh\necho 'OK'\n");
        let cfg_text = format!(
            r#"
[server]
tcp_listen = "127.0.0.1:0"
enable_ui = true

[flight_recorder]
enabled = true
max_records = 100
path = {:?}

[[backends]]
id = "mock"
kind = "cli"
command = {:?}
stdin = true

[routes]
"coding/default" = ["mock"]
"#,
            flight_path, echo_sh
        );
        let h = Harness::new(cfg_text).await;
        let res = h
            .post_json(
                "/v1/chat/completions",
                &[],
                json!({
                    "model": "coding/default",
                    "messages": [{"role": "user", "content": "ping"}]
                }),
            )
            .await;
        assert_eq!(res.status(), StatusCode::OK);

        // Traces ring buffer endpoint
        let traces_res = h.get("/v1/traces", &[]).await;
        assert_eq!(traces_res.status(), StatusCode::OK);
        let traces_json: Value =
            serde_json::from_str(&body_text(traces_res).await).expect("traces");
        assert_eq!(traces_json["traces"].as_array().unwrap().len(), 1);
        assert_eq!(traces_json["traces"][0]["backend"], "mock");

        // Dashboard HTML v2 contains Playground & Trace Waterfall tabs
        let ui_html = body_text(h.get("/ui", &[]).await).await;
        assert!(ui_html.contains("Route Playground"));
        assert!(ui_html.contains("Trace Waterfall"));
        assert!(ui_html.contains("Reset Circuit"));

        h.state.flight_recorder().flush().await;
        assert!(flight_path.exists());

        let replay_args = vouch::cli::ReplayArgs {
            config: h._dir.path().join("vouch.toml"),
            file: Some(flight_path),
            route: None,
            limit: Some(5),
            json: true,
        };
        vouch::cli::replay::run(&replay_args)
            .await
            .expect("replay ok");

        h.state.shutdown().await;
    }

    #[tokio::test]
    async fn cross_protocol_gemini_thought_and_ollama_tools_fidelity() {
        let dir = TempDir::new().expect("temp");
        let cli = script(
            dir.path(),
            "tool_reply.sh",
            "#!/bin/sh\ncat >/dev/null\necho '<vouch_tool_call>{\"name\":\"lookup\",\"arguments\":{\"k\":1}}</vouch_tool_call>'\n",
        );
        let cfg = format!(
            r#"
[server]
tcp_listen = "127.0.0.1:0"

[[backends]]
id = "mock"
kind = "cli"
command = {:?}
stdin = true

[routes]
"coding/default" = ["mock"]
"#,
            cli
        );
        let h = Harness::new(cfg).await;

        // Ollama /api/chat with inbound tool_calls + tool result history
        let ollama_resp = h
            .post_json(
                "/api/chat",
                &[],
                json!({
                    "model": "coding/default",
                    "stream": false,
                    "messages": [
                        {"role": "user", "content": "find k"},
                        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "lookup", "arguments": {"k": 0}}}]},
                        {"role": "tool", "name": "lookup", "tool_call_id": "c1", "content": "val=0"}
                    ],
                    "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
                }),
            )
            .await;
        assert_eq!(ollama_resp.status(), StatusCode::OK);
        let ojson: Value = serde_json::from_str(&body_text(ollama_resp).await).expect("ojson");
        assert_eq!(
            ojson["message"]["tool_calls"][0]["function"]["name"],
            "lookup"
        );

        // Gemini streaming with functionCall deltas
        let gem_resp = h
            .post_json(
                "/v1beta/models/coding%2Fdefault:streamGenerateContent?alt=sse",
                &[],
                json!({
                    "contents": [{"role": "user", "parts": [{"text": "run lookup"}]}],
                    "tools": [{"functionDeclarations": [{"name": "lookup", "parameters": {"type": "object"}}]}]
                }),
            )
            .await;
        assert_eq!(gem_resp.status(), StatusCode::OK);
        let gem_sse = body_text(gem_resp).await;
        assert!(gem_sse.contains("\"functionCall\""));
        assert!(gem_sse.contains("lookup"));
        h.state.shutdown().await;
    }

    #[test]
    fn vouch_example_toml_parses_cleanly_for_v5() {
        let example = include_str!("../vouch.example.toml");
        GatewayConfig::from_toml_str(example).expect("vouch.example.toml must validate");
    }

    #[tokio::test]
    async fn doctor_fix_applies_planned_repairs_atomically_and_revalidates() {
        let dir = TempDir::new().expect("temp");
        let cfg_path = dir.path().join("vouch.toml");
        let echo_sh = script(dir.path(), "real-tool.sh", "#!/bin/sh\necho 'OK'\n");
        let broken = r#"
[server]
tcp_listen = "127.0.0.1:0"

[[backends]]
id = "broken"
kind = "cli"
command = "/nonexistent/path/real-tool.sh"
stdin = true

[routes]
"coding/default" = ["broken"]
"#;
        fs::write(&cfg_path, broken).expect("write");
        let mut cfg = GatewayConfig::from_toml_str(broken).expect("cfg");
        let fixes = vec![vouch::cli::doctor::ProposedFix {
            backend_id: "broken".into(),
            field: "command",
            old_value: "/nonexistent/path/real-tool.sh".into(),
            new_value: echo_sh.display().to_string(),
        }];
        let applied = vouch::cli::doctor::apply_planned_fixes(&cfg_path, &mut cfg, &fixes, true)
            .expect("apply");
        assert!(applied);
        let reloaded = GatewayConfig::load(&cfg_path).expect("reload");
        assert_eq!(
            reloaded.backends[0].command.as_deref(),
            Some(echo_sh.to_string_lossy().as_ref())
        );
    }
}
