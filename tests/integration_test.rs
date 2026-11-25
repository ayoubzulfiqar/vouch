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
