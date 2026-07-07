//! MCP task support (SEP-1686, spec 2025-11-25): the deferred backend behind
//! [`McpTool`](super::McpTool)'s [`ToolDyn::dispatch_structured`] path.
//!
//! A task-augmented `tools/call` returns a `CreateTaskResult` instead of the
//! tool's output; the caller then owns the lifecycle: poll `tasks/get`,
//! retrieve the final payload via `tasks/result`, stop it via `tasks/cancel`.
//! [`McpTaskHandle`] wraps that lifecycle as a rig
//! [`ToolTaskHandle`](crate::tool::ToolTaskHandle), and
//! [`McpTaskNotifications`] routes `notifications/tasks/status` into waiting
//! handles so they wake before their next poll tick. Notifications are an
//! optimization only — per spec a requestor MUST NOT rely on them, so polling
//! remains the source of truth.
//!
//! rmcp 2.1.0's client peer has no task helpers, so [`ServerSinkTaskExt`]
//! drives the `tasks/*` requests through `Peer::send_request` directly.
//!
//! # Related-task metadata
//!
//! The only client-initiated task requests issued here are `tasks/get`,
//! `tasks/result`, and `tasks/cancel`, which identify the task in their params
//! — per spec these SHOULD NOT carry `io.modelcontextprotocol/related-task`
//! `_meta`, so none is attached. Future task-scoped request types must attach
//! [`rmcp::model::RelatedTaskMetadata`] under its `META_KEY`.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use futures::future::{Either, select};

use rmcp::model::{
    CallToolRequestParams, CallToolResult, CancelTaskParams, CancelTaskRequest, ClientRequest,
    CreateTaskResult, ErrorCode, GetTaskParams, GetTaskPayloadParams, GetTaskPayloadRequest,
    GetTaskRequest, ServerResult, TasksCapability,
};

use super::{McpToolError, call_tool_result_to_text};
use crate::tool::task::{TaskResumer, ToolTaskDescriptor, ToolTaskHandle, ToolTaskStatus};
use crate::tool::{ToolError, ToolExecutionResult, ToolFailure, ToolFailureKind};
use crate::wasm_compat::WasmBoxedFuture;

/// `_meta` key on `CreateTaskResult` carrying the server's suggested
/// model-facing immediate response (MCP tasks, 2025-11-25).
pub const MODEL_IMMEDIATE_RESPONSE_META_KEY: &str =
    "io.modelcontextprotocol/model-immediate-response";

/// Fallback polling cadence when the server supplies no `pollInterval`.
pub const DEFAULT_MCP_TASK_POLL_INTERVAL: Duration = Duration::from_secs(2);

/// Floor applied to server-suggested poll intervals so a misbehaving server
/// cannot induce a busy-poll.
const MIN_MCP_TASK_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// When an [`McpTool`](super::McpTool) dispatches a call as an MCP task
/// (SEP-1686).
///
/// Consulted only on the
/// [`ToolDyn::dispatch_structured`](crate::tool::ToolDyn::dispatch_structured)
/// path; the plain `call`/`call_structured` paths never create tasks. Spec
/// rules always apply: a server without the `tasks` capability, or a tool
/// whose `taskSupport` is `forbidden`/absent, is never invoked as a task
/// regardless of policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum McpTaskPolicy {
    /// Never dispatch as a task. A tool whose `taskSupport` is `required`
    /// fails fast with a classified error (the server would reject a plain
    /// call with `-32601` anyway).
    Never,
    /// Task only when the tool *requires* it (`taskSupport: required`); plain
    /// call otherwise. The spec-compliant minimum, and the default for a bare
    /// [`McpTool`](super::McpTool).
    #[default]
    Required,
    /// Task whenever permitted (`taskSupport: optional` or `required`); plain
    /// call when forbidden/absent. The default for tools registered through
    /// [`McpClientHandler`](super::McpClientHandler), fully exploiting a
    /// task-aware agent loop.
    Preferred,
}

/// Task metadata attached to the final [`ToolExecutionResult`] of a deferred
/// MCP tool call, via
/// [`ToolResultExtensions`](crate::tool::ToolResultExtensions). Never sent to
/// the model.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct McpTaskInfo {
    /// The server-assigned task id.
    pub task_id: String,
    /// The last observed lifecycle status when the result was produced.
    pub final_status: ToolTaskStatus,
    /// The server's human-readable status message, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status_message: Option<String>,
    /// The server-reported ISO-8601 creation timestamp, verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    /// `"{name}@{version}"` identity of the server that ran the task.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_key: Option<String>,
}

/// Map an rmcp [`ServiceError`](rmcp::ServiceError) from a task RPC into a
/// classified [`McpToolError`].
///
/// `-32602` (Invalid params) on `tasks/get|result|cancel` means the task id is
/// unknown or expired → [`NotFound`](ToolFailureKind::NotFound). `-32601`
/// means the server does not implement the method →
/// [`Provider`](ToolFailureKind::Provider). Transport losses →
/// [`Network`](ToolFailureKind::Network); request timeout →
/// [`Timeout`](ToolFailureKind::Timeout); cancellation →
/// [`Cancelled`](ToolFailureKind::Cancelled).
fn map_task_service_error(op: &str, task_id: &str, err: rmcp::ServiceError) -> McpToolError {
    match err {
        rmcp::ServiceError::McpError(e) if e.code == ErrorCode::INVALID_PARAMS => {
            McpToolError::new(
                ToolFailureKind::NotFound,
                format!(
                    "MCP task '{task_id}' not found or expired during {op}: {}",
                    e.message
                ),
            )
        }
        rmcp::ServiceError::McpError(e) if e.code == ErrorCode::METHOD_NOT_FOUND => {
            McpToolError::new(
                ToolFailureKind::Provider,
                format!("MCP server does not support {op}: {}", e.message),
            )
        }
        rmcp::ServiceError::McpError(e) => McpToolError::new(
            ToolFailureKind::Provider,
            format!("{op} for MCP task '{task_id}' failed: {e}"),
        ),
        rmcp::ServiceError::Timeout { timeout } => McpToolError::new(
            ToolFailureKind::Timeout,
            format!("{op} for MCP task '{task_id}' timed out after {timeout:?}"),
        ),
        rmcp::ServiceError::Cancelled { reason } => McpToolError::new(
            ToolFailureKind::Cancelled,
            format!(
                "{op} for MCP task '{task_id}' was cancelled: {}",
                reason.unwrap_or_else(|| "<unknown>".to_string())
            ),
        ),
        rmcp::ServiceError::TransportSend(_) | rmcp::ServiceError::TransportClosed => {
            McpToolError::new(
                ToolFailureKind::Network,
                format!("{op} for MCP task '{task_id}' hit a transport failure: {err}"),
            )
        }
        rmcp::ServiceError::UnexpectedResponse => McpToolError::new(
            ToolFailureKind::Provider,
            format!("{op} for MCP task '{task_id}' returned an unexpected response type"),
        ),
        // `ServiceError` is #[non_exhaustive]; future variants classify as Other.
        other => McpToolError::new(
            ToolFailureKind::Other,
            format!("{op} for MCP task '{task_id}' failed: {other}"),
        ),
    }
}

/// Convert rmcp's `#[non_exhaustive]` task status into rig's
/// [`ToolTaskStatus`]. An unknown future variant maps to `Working` with a
/// warning — the safest interpretation (keep polling; the agent loop's
/// deadline bounds it).
fn convert_status(status: &rmcp::model::TaskStatus) -> ToolTaskStatus {
    match status {
        rmcp::model::TaskStatus::Working => ToolTaskStatus::Working,
        rmcp::model::TaskStatus::InputRequired => ToolTaskStatus::InputRequired,
        rmcp::model::TaskStatus::Completed => ToolTaskStatus::Completed,
        rmcp::model::TaskStatus::Failed => ToolTaskStatus::Failed,
        rmcp::model::TaskStatus::Cancelled => ToolTaskStatus::Cancelled,
        other => {
            tracing::warn!(?other, "unknown MCP task status; treating as working");
            ToolTaskStatus::Working
        }
    }
}

/// A status snapshot pushed by `notifications/tasks/status`.
#[derive(Debug, Clone)]
pub(crate) struct TaskStatusUpdate {
    pub(crate) status: rmcp::model::TaskStatus,
    pub(crate) status_message: Option<String>,
    pub(crate) poll_interval: Option<u64>,
}

impl TaskStatusUpdate {
    fn from_task(task: &rmcp::model::Task) -> Self {
        Self {
            status: task.status.clone(),
            status_message: task.status_message.clone(),
            poll_interval: task.poll_interval,
        }
    }
}

/// Shared registry routing `notifications/tasks/status` to waiting task
/// handles.
///
/// The [`McpClientHandler`](super::McpClientHandler) owns an `Arc` and
/// publishes from its `on_task_status` callback; every
/// [`McpTool`](super::McpTool) it builds gets a clone, and each
/// [`McpTaskHandle`] subscribes by task id. Publish/subscribe hold the lock
/// only briefly and never across an `.await`. An entry is dropped when a
/// terminal update arrives with no live subscribers, and when a handle
/// releases its slot, so unwatched tasks cannot grow the map unboundedly.
#[derive(Debug, Default)]
pub struct McpTaskNotifications {
    channels:
        std::sync::Mutex<HashMap<String, tokio::sync::watch::Sender<Option<TaskStatusUpdate>>>>,
}

impl McpTaskNotifications {
    fn lock(
        &self,
    ) -> std::sync::MutexGuard<
        '_,
        HashMap<String, tokio::sync::watch::Sender<Option<TaskStatusUpdate>>>,
    > {
        self.channels
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Subscribe to status updates for `task_id`, creating the slot if the
    /// notification has not arrived yet.
    pub(crate) fn subscribe(
        &self,
        task_id: &str,
    ) -> tokio::sync::watch::Receiver<Option<TaskStatusUpdate>> {
        let mut channels = self.lock();
        channels
            .entry(task_id.to_string())
            .or_insert_with(|| tokio::sync::watch::channel(None).0)
            .subscribe()
    }

    /// Publish a status notification into the matching slot.
    ///
    /// A notification may race ahead of `subscribe`, so a non-terminal update
    /// for an unknown task id creates its slot and buffers the update; a
    /// terminal update with no live subscribers removes the slot instead.
    pub(crate) fn publish(&self, task: &rmcp::model::Task) {
        let update = TaskStatusUpdate::from_task(task);
        let terminal = convert_status(&update.status).is_terminal();
        let mut channels = self.lock();
        match channels.entry(task.task_id.clone()) {
            std::collections::hash_map::Entry::Occupied(entry) => {
                entry.get().send_replace(Some(update));
                if terminal && entry.get().receiver_count() == 0 {
                    entry.remove();
                }
            }
            std::collections::hash_map::Entry::Vacant(slot) => {
                if !terminal {
                    let (sender, _) = tokio::sync::watch::channel(Some(update));
                    slot.insert(sender);
                }
            }
        }
    }

    /// Release the slot for `task_id` (called when a handle is dropped).
    pub(crate) fn release(&self, task_id: &str) {
        self.lock().remove(task_id);
    }
}

/// Client-side task helpers for [`rmcp::service::ServerSink`] (SEP-1686),
/// absent from rmcp 2.1.0's `Peer<RoleClient>` API.
///
/// Every method drives the raw request through `Peer::send_request` and
/// hand-matches the [`ServerResult`] union; errors are classified via
/// [`McpToolError`].
pub trait ServerSinkTaskExt {
    /// `tools/call` augmented with task metadata — expects a
    /// `CreateTaskResult`. The `task` field must already be set on `params`.
    fn call_tool_as_task(
        &self,
        params: CallToolRequestParams,
    ) -> WasmBoxedFuture<'_, Result<CreateTaskResult, McpToolError>>;

    /// `tasks/get` — the task's current status snapshot.
    fn get_task<'a>(
        &'a self,
        task_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<rmcp::model::Task, McpToolError>>;

    /// `tasks/result` — blocks server-side until the task is terminal, then
    /// yields the original `CallToolResult`.
    fn get_task_payload<'a>(
        &'a self,
        task_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<CallToolResult, McpToolError>>;

    /// `tasks/cancel` — returns the post-cancel task snapshot.
    fn cancel_task<'a>(
        &'a self,
        task_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<rmcp::model::Task, McpToolError>>;

    /// The negotiated server-level tasks capability, if the handshake
    /// completed and the server declared one.
    fn tasks_capability(&self) -> Option<TasksCapability>;

    /// `"{name}@{version}"` identity key of the connected server, for
    /// [`ToolTaskDescriptor::server_key`].
    fn server_key(&self) -> Option<String>;
}

impl ServerSinkTaskExt for rmcp::service::ServerSink {
    fn call_tool_as_task(
        &self,
        params: CallToolRequestParams,
    ) -> WasmBoxedFuture<'_, Result<CreateTaskResult, McpToolError>> {
        Box::pin(async move {
            let request = ClientRequest::CallToolRequest(rmcp::model::Request::new(params));
            match self.send_request(request).await {
                Ok(ServerResult::CreateTaskResult(created)) => Ok(created),
                // Strict: a server that ignores task augmentation and answers
                // with a plain result is misbehaving; degrading silently would
                // hide the bug (the caller asked for a task lifecycle).
                Ok(ServerResult::CallToolResult(_)) => Err(McpToolError::new(
                    ToolFailureKind::Provider,
                    "MCP server ignored task augmentation and returned a plain tool result"
                        .to_string(),
                )),
                Ok(other) => Err(McpToolError::new(
                    ToolFailureKind::Provider,
                    format!("task-augmented tools/call returned an unexpected result: {other:?}"),
                )),
                Err(err) => Err(map_task_service_error("tools/call (task)", "<new>", err)),
            }
        })
    }

    fn get_task<'a>(
        &'a self,
        task_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<rmcp::model::Task, McpToolError>> {
        Box::pin(async move {
            let request = ClientRequest::GetTaskRequest(GetTaskRequest::new(GetTaskParams::new(
                task_id.to_string(),
            )));
            match self.send_request(request).await {
                Ok(ServerResult::GetTaskResult(result)) => Ok(result.task),
                Ok(other) => Err(McpToolError::new(
                    ToolFailureKind::Provider,
                    format!("tasks/get returned an unexpected result: {other:?}"),
                )),
                Err(err) => Err(map_task_service_error("tasks/get", task_id, err)),
            }
        })
    }

    fn get_task_payload<'a>(
        &'a self,
        task_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<CallToolResult, McpToolError>> {
        Box::pin(async move {
            let request = ClientRequest::GetTaskPayloadRequest(GetTaskPayloadRequest::new(
                GetTaskPayloadParams::new(task_id.to_string()),
            ));
            match self.send_request(request).await {
                // The untagged `ServerResult` decode may already produce a
                // typed `CallToolResult` for a tool payload...
                Ok(ServerResult::CallToolResult(result)) => Ok(result),
                // ...but `GetTaskPayloadResult` deliberately fails to
                // deserialize, so payloads that don't shape as a
                // `CallToolResult` arrive as the raw JSON catch-all and are
                // re-parsed here.
                Ok(ServerResult::CustomResult(custom)) => {
                    serde_json::from_value::<CallToolResult>(custom.0).map_err(|err| {
                        McpToolError::new(
                            ToolFailureKind::Provider,
                            format!(
                                "tasks/result payload for MCP task '{task_id}' was not a \
                                 CallToolResult: {err}"
                            ),
                        )
                    })
                }
                Ok(other) => Err(McpToolError::new(
                    ToolFailureKind::Provider,
                    format!("tasks/result returned an unexpected result: {other:?}"),
                )),
                Err(err) => Err(map_task_service_error("tasks/result", task_id, err)),
            }
        })
    }

    fn cancel_task<'a>(
        &'a self,
        task_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<rmcp::model::Task, McpToolError>> {
        Box::pin(async move {
            let request = ClientRequest::CancelTaskRequest(CancelTaskRequest::new(
                CancelTaskParams::new(task_id.to_string()),
            ));
            match self.send_request(request).await {
                Ok(ServerResult::CancelTaskResult(result)) => Ok(result.task),
                // `CancelTaskResult` and `GetTaskResult` share a wire shape
                // (`_meta` + flattened Task), and rmcp's untagged
                // `ServerResult` decode tries `GetTaskResult` first — accept
                // it as the equivalent task snapshot.
                Ok(ServerResult::GetTaskResult(result)) => Ok(result.task),
                Ok(other) => Err(McpToolError::new(
                    ToolFailureKind::Provider,
                    format!("tasks/cancel returned an unexpected result: {other:?}"),
                )),
                Err(err) => Err(map_task_service_error("tasks/cancel", task_id, err)),
            }
        })
    }

    fn tasks_capability(&self) -> Option<TasksCapability> {
        self.peer_info()
            .and_then(|info| info.capabilities.tasks.clone())
    }

    fn server_key(&self) -> Option<String> {
        self.peer_info()
            .map(|info| format!("{}@{}", info.server_info.name, info.server_info.version))
    }
}

/// A live [`ToolTaskHandle`] over an MCP task (SEP-1686).
///
/// Obtained from a task-augmented dispatch
/// ([`McpTool`](super::McpTool)'s `dispatch_structured`) or rehydrated from a
/// persisted [`ToolTaskDescriptor`] via [`McpTaskHandle::resume`]. Every RPC
/// this handle issues is bounded by its per-request timeout, so no single
/// await can wedge (the same discipline as
/// [`McpTool::with_timeout`](super::McpTool::with_timeout), issue #1914); the
/// overall wait is unbounded — deadlines belong to the agent loop.
pub struct McpTaskHandle {
    sink: rmcp::service::ServerSink,
    tool_name: String,
    task_id: String,
    created_at: Option<String>,
    ttl_ms: Option<u64>,
    /// Server-suggested poll cadence, clamped to
    /// [`MIN_MCP_TASK_POLL_INTERVAL`]; `None` when the server sent none.
    poll_interval: Option<Duration>,
    immediate_response: Option<String>,
    /// Per-request bound applied to every task RPC.
    request_timeout: Option<Duration>,
    notifications: Option<Arc<McpTaskNotifications>>,
    watch: Option<tokio::sync::watch::Receiver<Option<TaskStatusUpdate>>>,
    server_key: Option<String>,
}

/// Clamp a server-provided poll interval (ms) against busy-polling.
fn clamp_poll_interval(poll_interval_ms: u64) -> Duration {
    Duration::from_millis(poll_interval_ms).max(MIN_MCP_TASK_POLL_INTERVAL)
}

impl McpTaskHandle {
    /// Build a handle from a fresh `CreateTaskResult` (the deferred-dispatch
    /// path).
    pub(crate) fn from_create_result(
        sink: rmcp::service::ServerSink,
        tool_name: String,
        created: CreateTaskResult,
        request_timeout: Option<Duration>,
        notifications: Option<Arc<McpTaskNotifications>>,
    ) -> Self {
        let immediate_response = created
            .meta
            .as_ref()
            .and_then(|meta| meta.0.get(MODEL_IMMEDIATE_RESPONSE_META_KEY))
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        let task = created.task;
        let watch = notifications
            .as_ref()
            .map(|registry| registry.subscribe(&task.task_id));
        let server_key = sink.server_key();
        Self {
            sink,
            tool_name,
            task_id: task.task_id,
            created_at: Some(task.created_at),
            ttl_ms: task.ttl,
            poll_interval: task.poll_interval.map(clamp_poll_interval),
            immediate_response,
            request_timeout,
            notifications,
            watch,
            server_key,
        }
    }

    /// Rehydrate a handle from a persisted descriptor (durable resume). The
    /// caller supplies the live sink for `descriptor.server_key`.
    ///
    /// # Errors
    /// Returns an [`InvalidArgs`](ToolFailureKind::InvalidArgs) failure when
    /// the descriptor belongs to a different backend. A `server_key` mismatch
    /// only warns: keys are advisory (a server may legitimately report a new
    /// version after a redeploy) and the task id decides.
    pub fn resume(
        sink: rmcp::service::ServerSink,
        descriptor: ToolTaskDescriptor,
        request_timeout: Option<Duration>,
        notifications: Option<Arc<McpTaskNotifications>>,
    ) -> Result<Self, ToolFailure> {
        if descriptor.backend != ToolTaskDescriptor::BACKEND_MCP {
            return Err(ToolFailure::invalid_args(format!(
                "descriptor for backend '{}' cannot be resumed as an MCP task",
                descriptor.backend
            )));
        }
        let live_key = sink.server_key();
        if let (Some(persisted), Some(live)) = (&descriptor.server_key, &live_key)
            && persisted != live
        {
            tracing::warn!(
                persisted_server_key = %persisted,
                live_server_key = %live,
                task_id = %descriptor.task_id,
                "resuming MCP task against a server with a different identity key"
            );
        }
        let watch = notifications
            .as_ref()
            .map(|registry| registry.subscribe(&descriptor.task_id));
        Ok(Self {
            sink,
            tool_name: descriptor.tool_name,
            task_id: descriptor.task_id,
            created_at: descriptor.created_at,
            ttl_ms: descriptor.ttl_ms,
            poll_interval: descriptor.poll_interval_ms.map(clamp_poll_interval),
            immediate_response: descriptor.immediate_response,
            request_timeout,
            notifications,
            watch,
            server_key: live_key,
        })
    }

    /// Bound a task RPC by the per-request timeout, classifying an elapse as
    /// a [`Timeout`](ToolFailureKind::Timeout) error.
    async fn bounded<T>(
        &self,
        op: &str,
        fut: impl Future<Output = Result<T, McpToolError>>,
    ) -> Result<T, McpToolError> {
        match self.request_timeout {
            Some(timeout) => crate::wasm_compat::timeout(timeout, fut)
                .await
                .map_err(|_| {
                    McpToolError::new(
                        ToolFailureKind::Timeout,
                        format!(
                            "{op} for MCP task '{}' timed out after {timeout:?}",
                            self.task_id
                        ),
                    )
                })?,
            None => fut.await,
        }
    }

    /// Attach this handle's task metadata to a final result so hooks and
    /// telemetry can observe it (never sent to the model).
    fn attach_info(
        &self,
        result: ToolExecutionResult,
        final_status: ToolTaskStatus,
        status_message: Option<String>,
    ) -> ToolExecutionResult {
        result.with_extension(McpTaskInfo {
            task_id: self.task_id.clone(),
            final_status,
            status_message,
            created_at: self.created_at.clone(),
            server_key: self.server_key.clone(),
        })
    }

    /// A terminal-status snapshot that is not `Completed`, folded into the
    /// classified failure result [`wait`](ToolTaskHandle::wait) returns.
    fn terminal_failure(
        &self,
        status: ToolTaskStatus,
        status_message: Option<String>,
    ) -> ToolExecutionResult {
        let (message, failure) = match status {
            ToolTaskStatus::InputRequired => {
                let message = format!(
                    "MCP task '{}' for tool '{}' requires interactive input, which rig does not \
                     support yet",
                    self.task_id, self.tool_name
                );
                (
                    message.clone(),
                    ToolFailure::other(message)
                        .with_code("mcp_task_input_required")
                        .with_retryable(false),
                )
            }
            ToolTaskStatus::Cancelled => {
                let message = status_message
                    .clone()
                    .unwrap_or_else(|| format!("MCP task '{}' was cancelled", self.task_id));
                (message.clone(), ToolFailure::cancelled(message))
            }
            // Failed, or a defensive fold for a non-terminal status.
            _ => {
                let message = status_message
                    .clone()
                    .unwrap_or_else(|| format!("MCP task '{}' failed", self.task_id));
                (message.clone(), ToolFailure::other(message))
            }
        };
        self.attach_info(
            ToolExecutionResult::failed(message, failure),
            status,
            status_message,
        )
    }

    /// The current effective poll interval.
    fn effective_poll_interval(&self) -> Duration {
        self.poll_interval.unwrap_or(DEFAULT_MCP_TASK_POLL_INTERVAL)
    }
}

impl Drop for McpTaskHandle {
    fn drop(&mut self) {
        if let Some(notifications) = &self.notifications {
            notifications.release(&self.task_id);
        }
    }
}

/// What one `wait` iteration observed.
enum WaitWake {
    /// `tasks/result` resolved with the payload.
    Payload(Result<CallToolResult, McpToolError>),
    /// A notification or the poll cadence fired; snapshot the status.
    CheckStatus,
}

impl ToolTaskHandle for McpTaskHandle {
    fn task_id(&self) -> &str {
        &self.task_id
    }

    fn status(&self) -> WasmBoxedFuture<'_, Result<ToolTaskStatus, ToolFailure>> {
        Box::pin(async move {
            self.bounded("tasks/get", self.sink.get_task(&self.task_id))
                .await
                .map(|task| convert_status(&task.status))
                .map_err(McpToolError::into_failure)
        })
    }

    fn wait(self: Box<Self>) -> WasmBoxedFuture<'static, ToolExecutionResult> {
        Box::pin(async move {
            let mut this = self;
            // Hold the notification receiver outside the handle for the whole
            // wait: the racing arms below borrow `this` immutably (RPCs) and
            // the receiver mutably, which must be disjoint places.
            let mut watch = this.watch.take();
            loop {
                // One iteration = one attempt to finish: race the blocking
                // `tasks/result` long-poll (bounded per-request) against a
                // notification wakeup and the poll-cadence fallback. Losing
                // arms are dropped — abandoning an in-flight `tasks/result`
                // locally is safe (the server keeps the task).
                let wake = {
                    let interval = this.effective_poll_interval();
                    let payload = std::pin::pin!(
                        this.bounded("tasks/result", this.sink.get_task_payload(&this.task_id),)
                    );
                    let cadence = std::pin::pin!(async {
                        // A pending future bounded by the poll interval is the
                        // portable timer (no tokio::time in library code).
                        let _ = crate::wasm_compat::timeout(interval, std::future::pending::<()>())
                            .await;
                    });
                    let wakeup = std::pin::pin!(async {
                        match watch.as_mut() {
                            Some(receiver) => {
                                // A closed channel means the registry slot was
                                // dropped; fall back to pure cadence polling.
                                if receiver.changed().await.is_err() {
                                    std::future::pending::<()>().await;
                                }
                            }
                            None => std::future::pending::<()>().await,
                        }
                    });
                    match select(payload, select(wakeup, cadence)).await {
                        Either::Left((payload, _)) => match payload {
                            // A per-request timeout on the long-poll is the
                            // expected slow-task outcome, not a failure: fall
                            // through to a status check and keep waiting.
                            Err(err) if err.kind() == ToolFailureKind::Timeout => {
                                WaitWake::CheckStatus
                            }
                            other => WaitWake::Payload(other),
                        },
                        Either::Right(_) => WaitWake::CheckStatus,
                    }
                };

                match wake {
                    WaitWake::Payload(Ok(result)) => {
                        return match call_tool_result_to_text(result) {
                            Ok(text) => this.attach_info(
                                ToolExecutionResult::success(text),
                                ToolTaskStatus::Completed,
                                None,
                            ),
                            // The task finished but the tool reported an error
                            // result (`is_error: true`) or unsupported content.
                            Err(err) => {
                                let message = err.to_string();
                                let failure = err.into_failure();
                                this.attach_info(
                                    ToolExecutionResult::failed(message.clone(), failure),
                                    ToolTaskStatus::Failed,
                                    Some(message),
                                )
                            }
                        };
                    }
                    WaitWake::Payload(Err(err)) => {
                        let message = err.to_string();
                        let failure = err.into_failure();
                        return this.attach_info(
                            ToolExecutionResult::failed(message.clone(), failure),
                            ToolTaskStatus::Working,
                            Some(message),
                        );
                    }
                    WaitWake::CheckStatus => {
                        // Prefer the freshly-buffered notification snapshot;
                        // fall back to a bounded tasks/get.
                        let snapshot = watch
                            .as_mut()
                            .and_then(|receiver| receiver.borrow_and_update().clone());
                        let (status, status_message, poll_interval) = match snapshot {
                            Some(update) => (
                                convert_status(&update.status),
                                update.status_message,
                                update.poll_interval,
                            ),
                            None => {
                                match this
                                    .bounded("tasks/get", this.sink.get_task(&this.task_id))
                                    .await
                                {
                                    Ok(task) => (
                                        convert_status(&task.status),
                                        task.status_message,
                                        task.poll_interval,
                                    ),
                                    // A transient status-poll failure must not
                                    // abort the wait; NotFound (expired) must.
                                    Err(err) if err.kind() == ToolFailureKind::NotFound => {
                                        let message = err.to_string();
                                        let failure = err.into_failure();
                                        return this.attach_info(
                                            ToolExecutionResult::failed(message.clone(), failure),
                                            ToolTaskStatus::Working,
                                            Some(message),
                                        );
                                    }
                                    Err(err) => {
                                        tracing::warn!(
                                            task_id = %this.task_id,
                                            error = %err,
                                            "transient MCP task status poll failure"
                                        );
                                        (ToolTaskStatus::Working, None, None)
                                    }
                                }
                            }
                        };
                        if let Some(interval) = poll_interval {
                            this.poll_interval = Some(clamp_poll_interval(interval));
                        }
                        match status {
                            // Completed: loop — the next tasks/result returns
                            // promptly with the payload.
                            ToolTaskStatus::Working | ToolTaskStatus::Completed => {}
                            terminal_or_input => {
                                return this.terminal_failure(terminal_or_input, status_message);
                            }
                        }
                    }
                }
            }
        })
    }

    fn cancel(&self) -> WasmBoxedFuture<'_, Result<(), ToolFailure>> {
        Box::pin(async move {
            self.bounded("tasks/cancel", self.sink.cancel_task(&self.task_id))
                .await
                .map(|_| ())
                .map_err(McpToolError::into_failure)
        })
    }

    fn poll_hint(&self) -> Option<Duration> {
        self.poll_interval
    }

    fn descriptor(&self) -> ToolTaskDescriptor {
        ToolTaskDescriptor {
            server_key: self.server_key.clone(),
            created_at: self.created_at.clone(),
            ttl_ms: self.ttl_ms,
            poll_interval_ms: self
                .poll_interval
                .map(|interval| u64::try_from(interval.as_millis()).unwrap_or(u64::MAX)),
            immediate_response: self.immediate_response.clone(),
            ..ToolTaskDescriptor::new(
                ToolTaskDescriptor::BACKEND_MCP,
                self.task_id.clone(),
                self.tool_name.clone(),
            )
        }
    }

    fn immediate_response(&self) -> Option<&str> {
        self.immediate_response.as_deref()
    }
}

/// A [`TaskResumer`] that rehydrates MCP task handles against one connected
/// server.
///
/// Obtain one per MCP connection via
/// [`McpClientHandler::task_resumer`](super::McpClientHandler::task_resumer)
/// and register it on the agent runner. Descriptors for other backends, or
/// for a different `server_key`, yield `Ok(None)` so the next registered
/// resumer is consulted.
pub struct McpTaskResumer {
    sink: rmcp::service::ServerSink,
    request_timeout: Option<Duration>,
    notifications: Option<Arc<McpTaskNotifications>>,
}

impl McpTaskResumer {
    /// Create a resumer over a live server sink.
    pub fn new(
        sink: rmcp::service::ServerSink,
        request_timeout: Option<Duration>,
        notifications: Option<Arc<McpTaskNotifications>>,
    ) -> Self {
        Self {
            sink,
            request_timeout,
            notifications,
        }
    }
}

impl TaskResumer for McpTaskResumer {
    fn resume<'a>(
        &'a self,
        descriptor: &'a ToolTaskDescriptor,
    ) -> WasmBoxedFuture<'a, Result<Option<Box<dyn ToolTaskHandle>>, ToolError>> {
        Box::pin(async move {
            if descriptor.backend != ToolTaskDescriptor::BACKEND_MCP {
                return Ok(None);
            }
            // A persisted key that names a different server is "not mine":
            // let a resumer holding the right connection claim it.
            if let (Some(persisted), Some(live)) = (&descriptor.server_key, self.sink.server_key())
                && *persisted != live
            {
                return Ok(None);
            }
            let handle = McpTaskHandle::resume(
                self.sink.clone(),
                descriptor.clone(),
                self.request_timeout,
                self.notifications.clone(),
            )
            .map_err(|failure| {
                ToolError::ToolCallError(
                    format!("failed to resume MCP task: {}", failure.message).into(),
                )
            })?;
            Ok(Some(Box::new(handle) as Box<dyn ToolTaskHandle>))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use rmcp::model::{
        ClientInfo, ContentBlock, ErrorData, Implementation, ListToolsResult,
        PaginatedRequestParams, ProtocolVersion, ServerCapabilities, ServerInfo,
        ServerNotification, Task, TaskStatus, TaskStatusNotification, TaskStatusNotificationParam,
        TaskSupport, Tool, ToolExecution,
    };
    use rmcp::service::{RequestContext, RoleServer};
    use rmcp::{ServerHandler, ServiceExt};
    use tokio::sync::{Notify, RwLock};

    use super::*;
    use crate::tool::rmcp::{McpClientHandler, McpTool};
    use crate::tool::server::ToolServer;
    use crate::tool::{ToolCallExtensions, ToolDispatch, ToolDyn, ToolOutcome};

    type TaskEntry = (Task, Option<CallToolResult>);

    /// A deterministic task-capable MCP server: task lifecycle transitions are
    /// driven by the test (`complete`/`fail`/`set_status`/`forget`), never by
    /// timers, and `tasks/result` long-polls on a `Notify` (or hangs forever in
    /// `hang_results` mode, isolating the status/notification wake paths).
    #[derive(Clone)]
    struct ControlledTaskServer {
        tools: Vec<Tool>,
        declare_capability: bool,
        immediate_response: Option<String>,
        poll_interval_ms: Option<u64>,
        hang_results: bool,
        state: Arc<RwLock<std::collections::HashMap<String, TaskEntry>>>,
        terminal: Arc<Notify>,
        recorded: Arc<RwLock<Vec<&'static str>>>,
        next_id: Arc<AtomicUsize>,
    }

    impl ControlledTaskServer {
        fn new(task_support: Option<TaskSupport>) -> Self {
            let mut tool = Tool::new(
                "work".to_string(),
                "does deferred work".to_string(),
                Arc::new(serde_json::Map::new()),
            );
            if let Some(support) = task_support {
                tool = tool.with_execution(ToolExecution::from_raw(Some(support)));
            }
            Self {
                tools: vec![tool],
                declare_capability: true,
                immediate_response: None,
                poll_interval_ms: Some(50),
                hang_results: false,
                state: Arc::default(),
                terminal: Arc::default(),
                recorded: Arc::default(),
                next_id: Arc::default(),
            }
        }

        fn without_capability(mut self) -> Self {
            self.declare_capability = false;
            self
        }

        fn with_immediate_response(mut self, text: &str) -> Self {
            self.immediate_response = Some(text.to_string());
            self
        }

        fn with_poll_interval_ms(mut self, ms: u64) -> Self {
            self.poll_interval_ms = Some(ms);
            self
        }

        fn with_hanging_results(mut self) -> Self {
            self.hang_results = true;
            self
        }

        async fn complete(&self, task_id: &str, text: &str) {
            let mut state = self.state.write().await;
            if let Some((task, payload)) = state.get_mut(task_id) {
                task.status = TaskStatus::Completed;
                *payload = Some(CallToolResult::success(vec![ContentBlock::text(text)]));
            }
            drop(state);
            self.terminal.notify_waiters();
        }

        async fn fail(&self, task_id: &str, message: &str) {
            let mut state = self.state.write().await;
            if let Some((task, _)) = state.get_mut(task_id) {
                task.status = TaskStatus::Failed;
                task.status_message = Some(message.to_string());
            }
            drop(state);
            self.terminal.notify_waiters();
        }

        async fn set_status(&self, task_id: &str, status: TaskStatus) {
            let mut state = self.state.write().await;
            if let Some((task, _)) = state.get_mut(task_id) {
                task.status = status;
            }
            drop(state);
            self.terminal.notify_waiters();
        }

        async fn forget(&self, task_id: &str) {
            self.state.write().await.remove(task_id);
            self.terminal.notify_waiters();
        }

        async fn snapshot(&self, task_id: &str) -> Option<Task> {
            self.state
                .read()
                .await
                .get(task_id)
                .map(|(task, _)| task.clone())
        }

        async fn recorded_calls(&self) -> Vec<&'static str> {
            self.recorded.read().await.clone()
        }
    }

    impl ServerHandler for ControlledTaskServer {
        fn get_info(&self) -> ServerInfo {
            let mut capabilities = ServerCapabilities::builder().enable_tools().build();
            if self.declare_capability {
                capabilities.tasks = Some(TasksCapability::server_default());
            }
            ServerInfo::new(capabilities)
                .with_protocol_version(ProtocolVersion::LATEST)
                .with_server_info(Implementation::new("test-task-server", "0.1.0"))
        }

        async fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            Ok(ListToolsResult::with_all_items(self.tools.clone()))
        }

        // rmcp's `handle_request` consults this to validate taskSupport; the
        // default returns `None`, which would skip validation entirely.
        fn get_tool(&self, name: &str) -> Option<Tool> {
            self.tools.iter().find(|tool| tool.name == name).cloned()
        }

        async fn call_tool(
            &self,
            request: CallToolRequestParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<CallToolResult, ErrorData> {
            self.recorded.write().await.push("call_tool");
            Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "plain:{}",
                request.name
            ))]))
        }

        async fn enqueue_task(
            &self,
            request: CallToolRequestParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<CreateTaskResult, ErrorData> {
            self.recorded.write().await.push("enqueue_task");
            let id = format!("task-{}", self.next_id.fetch_add(1, Ordering::SeqCst));
            let mut task = Task::new(
                id.clone(),
                TaskStatus::Working,
                "2026-01-01T00:00:00Z".to_string(),
                "2026-01-01T00:00:00Z".to_string(),
            );
            if let Some(ms) = self.poll_interval_ms {
                task = task.with_poll_interval(ms);
            }
            if let Some(ttl) = request.task.as_ref().and_then(|t| t.ttl) {
                task = task.with_ttl(ttl);
            }
            self.state.write().await.insert(id, (task.clone(), None));
            let mut created = CreateTaskResult::new(task);
            if let Some(text) = &self.immediate_response {
                let mut meta = rmcp::model::Meta::new();
                meta.0.insert(
                    MODEL_IMMEDIATE_RESPONSE_META_KEY.to_string(),
                    serde_json::json!(text),
                );
                created = created.with_meta(meta);
            }
            Ok(created)
        }

        async fn get_task_info(
            &self,
            request: GetTaskParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<rmcp::model::GetTaskResult, ErrorData> {
            match self.state.read().await.get(&request.task_id) {
                Some((task, _)) => Ok(rmcp::model::GetTaskResult::new(task.clone())),
                None => Err(ErrorData::invalid_params("task not found", None)),
            }
        }

        async fn get_task_result(
            &self,
            request: GetTaskPayloadParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<rmcp::model::GetTaskPayloadResult, ErrorData> {
            if self.hang_results {
                std::future::pending::<()>().await;
            }
            loop {
                let notified = self.terminal.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                {
                    let state = self.state.read().await;
                    match state.get(&request.task_id) {
                        Some((task, Some(payload)))
                            if matches!(
                                task.status,
                                TaskStatus::Completed | TaskStatus::Failed | TaskStatus::Cancelled
                            ) =>
                        {
                            let value = serde_json::to_value(payload)
                                .map_err(|err| ErrorData::internal_error(err.to_string(), None))?;
                            return Ok(rmcp::model::GetTaskPayloadResult::new(value));
                        }
                        Some((task, None))
                            if matches!(
                                task.status,
                                TaskStatus::Completed | TaskStatus::Failed | TaskStatus::Cancelled
                            ) =>
                        {
                            return Err(ErrorData::invalid_params("task result unavailable", None));
                        }
                        Some(_) => {}
                        None => {
                            return Err(ErrorData::invalid_params("task not found", None));
                        }
                    }
                }
                notified.await;
            }
        }

        async fn cancel_task(
            &self,
            request: CancelTaskParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<rmcp::model::CancelTaskResult, ErrorData> {
            let mut state = self.state.write().await;
            match state.get_mut(&request.task_id) {
                Some((task, _)) => {
                    task.status = TaskStatus::Cancelled;
                    task.status_message = Some("cancelled by request".to_string());
                    Ok(rmcp::model::CancelTaskResult::new(task.clone()))
                }
                None => Err(ErrorData::invalid_params("task not found", None)),
            }
        }
    }

    /// Serve `server` over an in-process duplex transport and return a bare
    /// connected client whose peer is the `ServerSink` rig tools wrap.
    async fn connect_bare(
        server: ControlledTaskServer,
    ) -> rmcp::service::RunningService<rmcp::service::RoleClient, ClientInfo> {
        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);
        tokio::spawn(async move {
            let running = server
                .serve((server_from_client, server_to_client))
                .await
                .expect("server failed to start");
            running.waiting().await.ok();
        });
        ClientInfo::default()
            .serve((client_from_server, client_to_server))
            .await
            .expect("client connect failed")
    }

    /// List the server's single tool and wrap it as an `McpTool` with `policy`.
    async fn bare_tool(
        client: &rmcp::service::RunningService<rmcp::service::RoleClient, ClientInfo>,
        policy: McpTaskPolicy,
    ) -> McpTool {
        let tools = client.peer().list_all_tools().await.expect("list_tools");
        McpTool::from_mcp_server(tools[0].clone(), client.peer().clone()).with_task_policy(policy)
    }

    fn expect_completed(dispatch: ToolDispatch) -> ToolExecutionResult {
        match dispatch {
            ToolDispatch::Completed(result) => result,
            ToolDispatch::Deferred(handle) => {
                panic!(
                    "expected a completed dispatch, got task '{}'",
                    handle.task_id()
                )
            }
        }
    }

    fn expect_deferred(dispatch: ToolDispatch) -> Box<dyn ToolTaskHandle> {
        match dispatch {
            ToolDispatch::Deferred(handle) => handle,
            ToolDispatch::Completed(result) => panic!(
                "expected a deferred dispatch, got completed result: {:?}",
                result.model_output()
            ),
        }
    }

    fn expect_failure(result: &ToolExecutionResult) -> &ToolFailure {
        match result.outcome() {
            ToolOutcome::Error(failure) => failure,
            other => panic!("expected an error outcome, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn forbidden_tool_never_dispatches_as_task() {
        let server = ControlledTaskServer::new(None);
        let client = connect_bare(server.clone()).await;
        let tool = bare_tool(&client, McpTaskPolicy::Preferred).await;

        let dispatch = tool
            .dispatch_structured("{}".to_string(), &ToolCallExtensions::EMPTY)
            .await;
        let result = expect_completed(dispatch);
        assert_eq!(result.model_output(), "plain:work");
        assert_eq!(server.recorded_calls().await, vec!["call_tool"]);
    }

    #[tokio::test]
    async fn required_tool_with_policy_never_fails_fast() {
        let server = ControlledTaskServer::new(Some(TaskSupport::Required));
        let client = connect_bare(server.clone()).await;
        let tool = bare_tool(&client, McpTaskPolicy::Never).await;

        let dispatch = tool
            .dispatch_structured("{}".to_string(), &ToolCallExtensions::EMPTY)
            .await;
        let result = expect_completed(dispatch);
        let failure = expect_failure(&result);
        assert_eq!(failure.kind, ToolFailureKind::Other);
        assert_eq!(failure.code.as_deref(), Some("mcp_task_required"));
        assert_eq!(failure.retryable, Some(false));
        // Fail-fast: the server never saw the call.
        assert!(server.recorded_calls().await.is_empty());
    }

    #[tokio::test]
    async fn required_tool_default_policy_dispatches_deferred() {
        let server = ControlledTaskServer::new(Some(TaskSupport::Required));
        let client = connect_bare(server.clone()).await;
        // Bare-tool default policy is Required: task the call.
        let tool = bare_tool(&client, McpTaskPolicy::default()).await;

        let dispatch = tool
            .dispatch_structured("{}".to_string(), &ToolCallExtensions::EMPTY)
            .await;
        let handle = expect_deferred(dispatch);
        assert!(handle.task_id().starts_with("task-"));
        assert_eq!(handle.poll_hint(), Some(Duration::from_millis(100)));
        assert_eq!(server.recorded_calls().await, vec!["enqueue_task"]);
    }

    #[tokio::test]
    async fn optional_tool_policy_matrix() {
        let server = ControlledTaskServer::new(Some(TaskSupport::Optional));
        let client = connect_bare(server.clone()).await;

        // Required (default): optional tools stay plain calls.
        let tool = bare_tool(&client, McpTaskPolicy::Required).await;
        let result = expect_completed(
            tool.dispatch_structured("{}".to_string(), &ToolCallExtensions::EMPTY)
                .await,
        );
        assert_eq!(result.model_output(), "plain:work");

        // Preferred: optional tools become tasks.
        let tool = bare_tool(&client, McpTaskPolicy::Preferred).await;
        let handle = expect_deferred(
            tool.dispatch_structured("{}".to_string(), &ToolCallExtensions::EMPTY)
                .await,
        );
        assert!(handle.task_id().starts_with("task-"));
        assert_eq!(
            server.recorded_calls().await,
            vec!["call_tool", "enqueue_task"]
        );
    }

    #[tokio::test]
    async fn no_server_task_capability_forces_plain_call() {
        // Optional tool + Preferred policy: without the server capability the
        // call must stay plain.
        let server = ControlledTaskServer::new(Some(TaskSupport::Optional)).without_capability();
        let client = connect_bare(server.clone()).await;
        let tool = bare_tool(&client, McpTaskPolicy::Preferred).await;
        let result = expect_completed(
            tool.dispatch_structured("{}".to_string(), &ToolCallExtensions::EMPTY)
                .await,
        );
        assert_eq!(result.model_output(), "plain:work");
        assert_eq!(server.recorded_calls().await, vec!["call_tool"]);

        // Required tool on an inconsistent (no-capability) server: rig
        // attempts a plain call and surfaces whatever the server decides —
        // never a client-side task.
        let server = ControlledTaskServer::new(Some(TaskSupport::Required)).without_capability();
        let client = connect_bare(server.clone()).await;
        let tool = bare_tool(&client, McpTaskPolicy::Preferred).await;
        let dispatch = tool
            .dispatch_structured("{}".to_string(), &ToolCallExtensions::EMPTY)
            .await;
        let result = expect_completed(dispatch);
        // rmcp's server router rejects a plain call to a required-task tool.
        let failure = expect_failure(&result);
        assert_eq!(failure.kind, ToolFailureKind::Provider);
    }

    #[tokio::test]
    async fn deferred_wait_happy_path() {
        let server = ControlledTaskServer::new(Some(TaskSupport::Required));
        let client = connect_bare(server.clone()).await;
        let tool = bare_tool(&client, McpTaskPolicy::Required).await;

        let handle = expect_deferred(
            tool.dispatch_structured("{}".to_string(), &ToolCallExtensions::EMPTY)
                .await,
        );
        let task_id = handle.task_id().to_string();

        let descriptor = handle.descriptor();
        assert_eq!(descriptor.backend, ToolTaskDescriptor::BACKEND_MCP);
        assert_eq!(descriptor.task_id, task_id);
        assert_eq!(descriptor.tool_name, "work");
        assert_eq!(
            descriptor.server_key.as_deref(),
            Some("test-task-server@0.1.0")
        );
        let round_tripped: ToolTaskDescriptor = serde_json::from_value(
            serde_json::to_value(&descriptor).expect("descriptor serializes"),
        )
        .expect("descriptor deserializes");
        assert_eq!(round_tripped, descriptor);

        let completer = {
            let server = server.clone();
            let task_id = task_id.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(100)).await;
                server.complete(&task_id, "task output").await;
            })
        };

        let result = tokio::time::timeout(Duration::from_secs(5), handle.wait())
            .await
            .expect("wait must resolve once the task completes");
        completer.await.expect("completer");

        assert_eq!(result.model_output(), "task output");
        assert!(matches!(result.outcome(), ToolOutcome::Success));
        let info = result
            .extensions()
            .get::<McpTaskInfo>()
            .expect("final result carries McpTaskInfo");
        assert_eq!(info.task_id, task_id);
        assert_eq!(info.final_status, ToolTaskStatus::Completed);
        assert_eq!(info.server_key.as_deref(), Some("test-task-server@0.1.0"));
    }

    #[tokio::test]
    async fn immediate_response_hint_is_extracted() {
        let server = ControlledTaskServer::new(Some(TaskSupport::Required))
            .with_immediate_response("working on it");
        let client = connect_bare(server.clone()).await;
        let tool = bare_tool(&client, McpTaskPolicy::Required).await;

        let handle = expect_deferred(
            tool.dispatch_structured("{}".to_string(), &ToolCallExtensions::EMPTY)
                .await,
        );
        assert_eq!(handle.immediate_response(), Some("working on it"));
        assert_eq!(
            handle.descriptor().immediate_response.as_deref(),
            Some("working on it")
        );
    }

    #[tokio::test]
    async fn cancel_maps_to_cancelled() {
        // Hanging results isolate the status-poll path, so `wait` learns of
        // the cancellation deterministically via `tasks/get`.
        let server = ControlledTaskServer::new(Some(TaskSupport::Required)).with_hanging_results();
        let client = connect_bare(server.clone()).await;
        let tool = bare_tool(&client, McpTaskPolicy::Required).await;

        let handle = expect_deferred(
            tool.dispatch_structured("{}".to_string(), &ToolCallExtensions::EMPTY)
                .await,
        );
        let task_id = handle.task_id().to_string();

        handle.cancel().await.expect("cancel succeeds");
        let snapshot = server.snapshot(&task_id).await.expect("task exists");
        assert_eq!(snapshot.status, TaskStatus::Cancelled);
        assert_eq!(
            handle.status().await.expect("status"),
            ToolTaskStatus::Cancelled
        );

        let result = tokio::time::timeout(Duration::from_secs(5), handle.wait())
            .await
            .expect("wait resolves via the status poll");
        let failure = expect_failure(&result);
        assert_eq!(failure.kind, ToolFailureKind::Cancelled);
        let info = result
            .extensions()
            .get::<McpTaskInfo>()
            .expect("cancelled result carries McpTaskInfo");
        assert_eq!(info.final_status, ToolTaskStatus::Cancelled);
    }

    #[tokio::test]
    async fn failed_status_carries_status_message() {
        let server = ControlledTaskServer::new(Some(TaskSupport::Required)).with_hanging_results();
        let client = connect_bare(server.clone()).await;
        let tool = bare_tool(&client, McpTaskPolicy::Required).await;

        let handle = expect_deferred(
            tool.dispatch_structured("{}".to_string(), &ToolCallExtensions::EMPTY)
                .await,
        );
        let task_id = handle.task_id().to_string();
        server.fail(&task_id, "disk full").await;

        let result = tokio::time::timeout(Duration::from_secs(5), handle.wait())
            .await
            .expect("wait resolves via the status poll");
        let failure = expect_failure(&result);
        assert_eq!(failure.kind, ToolFailureKind::Other);
        assert!(
            result.model_output().contains("disk full"),
            "status message must reach the model output, got {:?}",
            result.model_output()
        );
        let info = result
            .extensions()
            .get::<McpTaskInfo>()
            .expect("failed result carries McpTaskInfo");
        assert_eq!(info.final_status, ToolTaskStatus::Failed);
        assert_eq!(info.status_message.as_deref(), Some("disk full"));
    }

    #[tokio::test]
    async fn notification_wakes_wait_before_poll_interval() {
        // A 60s poll interval and hanging `tasks/result` mean only the
        // `notifications/tasks/status` wakeup can resolve the wait quickly.
        let server = ControlledTaskServer::new(Some(TaskSupport::Required))
            .with_poll_interval_ms(60_000)
            .with_hanging_results();

        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);

        let server_clone = server.clone();
        let server_service_handle = tokio::spawn(async move {
            server_clone
                .serve((server_from_client, server_to_client))
                .await
                .expect("server failed to start")
        });

        let tool_server_handle = ToolServer::new().run();
        let handler = McpClientHandler::new(ClientInfo::default(), tool_server_handle.clone());
        let _service = handler
            .connect((client_from_server, client_to_server))
            .await
            .expect("connect failed");
        let server_service = server_service_handle.await.expect("server service");

        // Handler default policy is Preferred; the required-task tool defers.
        let handle = expect_deferred(
            tool_server_handle
                .dispatch_tool_structured("work", "{}", &ToolCallExtensions::EMPTY)
                .await,
        );
        let task_id = handle.task_id().to_string();

        server.fail(&task_id, "boom").await;
        let failed_task = server.snapshot(&task_id).await.expect("task exists");
        server_service
            .peer()
            .send_notification(ServerNotification::TaskStatusNotification(
                TaskStatusNotification::new(TaskStatusNotificationParam::new(failed_task)),
            ))
            .await
            .expect("notification sent");

        let result = tokio::time::timeout(Duration::from_secs(5), handle.wait())
            .await
            .expect("the status notification must wake the wait well before the 60s poll tick");
        let failure = expect_failure(&result);
        assert_eq!(failure.kind, ToolFailureKind::Other);
        assert!(result.model_output().contains("boom"));
    }

    #[tokio::test]
    async fn expired_task_maps_to_not_found() {
        let server = ControlledTaskServer::new(Some(TaskSupport::Required));
        let client = connect_bare(server.clone()).await;
        let tool = bare_tool(&client, McpTaskPolicy::Required).await;

        let handle = expect_deferred(
            tool.dispatch_structured("{}".to_string(), &ToolCallExtensions::EMPTY)
                .await,
        );
        let task_id = handle.task_id().to_string();
        server.forget(&task_id).await;

        let status = handle
            .status()
            .await
            .expect_err("status of an expired task errs");
        assert_eq!(status.kind, ToolFailureKind::NotFound);

        let result = tokio::time::timeout(Duration::from_secs(5), handle.wait())
            .await
            .expect("wait resolves fast on an expired task");
        let failure = expect_failure(&result);
        assert_eq!(failure.kind, ToolFailureKind::NotFound);
    }

    #[tokio::test]
    async fn input_required_surfaces_as_classified_failure() {
        let server = ControlledTaskServer::new(Some(TaskSupport::Required)).with_hanging_results();
        let client = connect_bare(server.clone()).await;
        let tool = bare_tool(&client, McpTaskPolicy::Required).await;

        let handle = expect_deferred(
            tool.dispatch_structured("{}".to_string(), &ToolCallExtensions::EMPTY)
                .await,
        );
        let task_id = handle.task_id().to_string();
        server.set_status(&task_id, TaskStatus::InputRequired).await;

        assert_eq!(
            handle.status().await.expect("status"),
            ToolTaskStatus::InputRequired
        );

        let result = tokio::time::timeout(Duration::from_secs(5), handle.wait())
            .await
            .expect("wait resolves via the status poll");
        let failure = expect_failure(&result);
        assert_eq!(failure.code.as_deref(), Some("mcp_task_input_required"));
        assert_eq!(failure.retryable, Some(false));
        let info = result
            .extensions()
            .get::<McpTaskInfo>()
            .expect("input_required result carries McpTaskInfo");
        assert_eq!(info.final_status, ToolTaskStatus::InputRequired);
    }

    #[test]
    fn map_task_service_error_classifies_kinds() {
        let cases = [
            (
                rmcp::ServiceError::McpError(ErrorData::invalid_params("gone", None)),
                ToolFailureKind::NotFound,
            ),
            (
                rmcp::ServiceError::McpError(ErrorData::method_not_found::<
                    rmcp::model::GetTaskMethod,
                >()),
                ToolFailureKind::Provider,
            ),
            (
                rmcp::ServiceError::McpError(ErrorData::internal_error("boom", None)),
                ToolFailureKind::Provider,
            ),
            (
                rmcp::ServiceError::Timeout {
                    timeout: Duration::from_secs(1),
                },
                ToolFailureKind::Timeout,
            ),
            (
                rmcp::ServiceError::Cancelled { reason: None },
                ToolFailureKind::Cancelled,
            ),
            (
                rmcp::ServiceError::TransportClosed,
                ToolFailureKind::Network,
            ),
            (
                rmcp::ServiceError::UnexpectedResponse,
                ToolFailureKind::Provider,
            ),
        ];
        for (err, expected) in cases {
            let mapped = map_task_service_error("tasks/get", "task-1", err);
            assert_eq!(mapped.kind(), expected);
        }
    }

    #[test]
    fn convert_status_covers_all_known_variants() {
        let cases = [
            (TaskStatus::Working, ToolTaskStatus::Working),
            (TaskStatus::InputRequired, ToolTaskStatus::InputRequired),
            (TaskStatus::Completed, ToolTaskStatus::Completed),
            (TaskStatus::Failed, ToolTaskStatus::Failed),
            (TaskStatus::Cancelled, ToolTaskStatus::Cancelled),
        ];
        for (rmcp_status, expected) in cases {
            assert_eq!(convert_status(&rmcp_status), expected);
        }
    }

    #[test]
    fn notification_registry_buffers_and_cleans_up() {
        let registry = McpTaskNotifications::default();
        let working = Task::new(
            "task-1".to_string(),
            TaskStatus::Working,
            "2026-01-01T00:00:00Z".to_string(),
            "2026-01-01T00:00:00Z".to_string(),
        );

        // A notification racing ahead of subscribe is buffered...
        registry.publish(&working);
        let receiver = registry.subscribe("task-1");
        let buffered = receiver.borrow().clone().expect("buffered update");
        assert_eq!(buffered.status, TaskStatus::Working);

        // ...a terminal update with a live subscriber keeps the slot...
        let mut done = working.clone();
        done.status = TaskStatus::Completed;
        registry.publish(&done);
        assert_eq!(registry.lock().len(), 1);

        // ...and a terminal update with no subscribers removes it.
        drop(receiver);
        registry.publish(&done);
        assert_eq!(registry.lock().len(), 0);

        // A terminal update for an unknown task never creates a slot.
        let mut other = done.clone();
        other.task_id = "task-2".to_string();
        registry.publish(&other);
        assert_eq!(registry.lock().len(), 0);

        registry.release("task-1");
    }
}
