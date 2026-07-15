//! MCP task support (SEP-1686, spec 2025-11-25): the deferred backend behind
//! tools registered through [`McpClientHandler`](super::McpClientHandler).
//!
//! A task-augmented `tools/call` returns a `CreateTaskResult` instead of the
//! tool's output; the caller then owns the lifecycle: poll `tasks/get`,
//! retrieve the final payload via `tasks/result`, stop it via `tasks/cancel`.
//! [`McpTaskHandle`] wraps that lifecycle as a Rig
//! [`ToolTaskHandle`]. The MCP client handler
//! routes `notifications/tasks/status` into waiting
//! handles so they wake before their next poll tick. Notifications are an
//! optimization only — per spec a requestor MUST NOT rely on them, so polling
//! remains the source of truth.
//!
//! rmcp's client peer has no task helpers, so an internal extension drives
//! `tasks/*` requests through Rig's cancellable, deadline-aware MCP request
//! path.
//!
//! # Related-task metadata
//!
//! The only client-initiated task requests issued here are `tasks/get`,
//! `tasks/result`, and `tasks/cancel`, which identify the task in their params
//! — per spec these SHOULD NOT carry `io.modelcontextprotocol/related-task`
//! `_meta`, so none is attached. Future task-scoped request types must attach
//! [`rmcp::model::RelatedTaskMetadata`] under its `META_KEY`.

use std::collections::{HashMap, hash_map::Entry};
use std::future::pending;
use std::pin::pin;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use futures::future::{Either, select};

use rmcp::model::{
    CallToolRequestParams, CallToolResult, CancelTaskParams, CancelTaskRequest, ClientRequest,
    CreateTaskResult, ErrorCode, GetTaskParams, GetTaskPayloadParams, GetTaskPayloadRequest,
    GetTaskRequest, ServerResult, TasksCapability,
};
use tokio::time::Instant;

use super::{mcp_result_output, preserve_mcp_result, send_mcp_request};
use crate::tool::task::{TaskResumer, ToolTaskDescriptor, ToolTaskHandle, ToolTaskStatus};
use crate::tool::{ToolContext, ToolErrorKind, ToolExecutionError, ToolTaskResult};
use crate::wasm_compat::{WasmBoxedFuture, timeout};

type McpToolError = ToolExecutionError;

#[derive(Clone, Copy)]
enum McpTaskOperation {
    Launch,
    Get,
    Result,
    Cancel,
}

impl McpTaskOperation {
    const fn name(self) -> &'static str {
        match self {
            Self::Launch => "tools/call (task)",
            Self::Get => "tasks/get",
            Self::Result => "tasks/result",
            Self::Cancel => "tasks/cancel",
        }
    }
}

/// `_meta` key on `CreateTaskResult` carrying the server's suggested
/// model-facing immediate response (MCP tasks, 2025-11-25).
pub const MODEL_IMMEDIATE_RESPONSE_META_KEY: &str =
    "io.modelcontextprotocol/model-immediate-response";

/// Fallback polling cadence when the server supplies no `pollInterval`.
pub const DEFAULT_MCP_TASK_POLL_INTERVAL: Duration = Duration::from_secs(2);

/// Floor applied to server-suggested poll intervals so a misbehaving server
/// cannot induce a busy-poll.
const MIN_MCP_TASK_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Keep portable timer construction within a range supported by every target.
/// A longer server suggestion is still respected by chaining slices.
const MAX_MCP_TASK_TIMER_SLICE: Duration = Duration::from_secs(24 * 60 * 60);

async fn sleep_poll_interval(mut remaining: Duration) {
    loop {
        let slice = remaining.min(MAX_MCP_TASK_TIMER_SLICE);
        let _ = timeout(slice, pending::<()>()).await;
        remaining = remaining.saturating_sub(slice);
        if remaining.is_zero() {
            return;
        }
    }
}

/// Controls when tools registered through
/// [`McpClientHandler`](super::McpClientHandler) dispatch as MCP tasks
/// (SEP-1686).
///
/// Consulted only on the task-aware dispatch path. Spec rules always apply: a
/// server without the `tasks` capability, or a tool
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
    /// call otherwise. The spec-compliant minimum and the default.
    #[default]
    Required,
    /// Task whenever permitted (`taskSupport: optional` or `required`); plain
    /// call when forbidden/absent. This is an explicit opt-in for optional
    /// task support.
    Preferred,
}

/// Task metadata attached to the final [`ToolTaskResult`]'s typed context.
/// Never sent to the model.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct McpTaskInfo {
    /// The server-assigned task id.
    pub task_id: String,
    /// The last observed lifecycle status when the result was produced.
    pub status: ToolTaskStatus,
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
/// `-32602` (Invalid params) on `tasks/get|result` means the task id is unknown
/// or expired → [`NotFound`](ToolErrorKind::NotFound). The same code on launch
/// or cancellation is a provider rejection because no lookup-only meaning is
/// safe to infer. `-32601` means the server does not implement the method →
/// [`Provider`](ToolErrorKind::Provider). Transport losses →
/// [`Network`](ToolErrorKind::Network); request timeout →
/// [`Timeout`](ToolErrorKind::Timeout); cancellation →
/// [`Cancelled`](ToolErrorKind::Cancelled).
fn map_task_service_error(
    operation: McpTaskOperation,
    task_id: &str,
    err: rmcp::ServiceError,
) -> McpToolError {
    let op = operation.name();
    match err {
        rmcp::ServiceError::McpError(e)
            if e.code == ErrorCode::INVALID_PARAMS
                && matches!(operation, McpTaskOperation::Get | McpTaskOperation::Result) =>
        {
            McpToolError::new(
                ToolErrorKind::NotFound,
                format!(
                    "MCP task '{task_id}' not found or expired during {op}: {}",
                    e.message
                ),
            )
        }
        rmcp::ServiceError::McpError(e) if e.code == ErrorCode::INVALID_PARAMS => {
            McpToolError::new(
                ToolErrorKind::Provider,
                format!(
                    "MCP server rejected {op} for task '{task_id}': {}",
                    e.message
                ),
            )
        }
        rmcp::ServiceError::McpError(e) if e.code == ErrorCode::METHOD_NOT_FOUND => {
            McpToolError::new(
                ToolErrorKind::Provider,
                format!("MCP server does not support {op}: {}", e.message),
            )
        }
        rmcp::ServiceError::McpError(e) => McpToolError::new(
            ToolErrorKind::Provider,
            format!("{op} for MCP task '{task_id}' failed: {e}"),
        ),
        rmcp::ServiceError::Timeout { timeout } => McpToolError::new(
            ToolErrorKind::Timeout,
            format!("{op} for MCP task '{task_id}' timed out after {timeout:?}"),
        ),
        rmcp::ServiceError::Cancelled { reason } => McpToolError::new(
            ToolErrorKind::Cancelled,
            format!(
                "{op} for MCP task '{task_id}' was cancelled: {}",
                reason.unwrap_or_else(|| "<unknown>".to_string())
            ),
        ),
        rmcp::ServiceError::TransportSend(_) | rmcp::ServiceError::TransportClosed => {
            McpToolError::new(
                ToolErrorKind::Network,
                format!("{op} for MCP task '{task_id}' hit a transport failure: {err}"),
            )
        }
        rmcp::ServiceError::UnexpectedResponse => McpToolError::new(
            ToolErrorKind::Provider,
            format!("{op} for MCP task '{task_id}' returned an unexpected response type"),
        ),
        // `ServiceError` is #[non_exhaustive]; future variants classify as Other.
        other => McpToolError::new(
            ToolErrorKind::Other,
            format!("{op} for MCP task '{task_id}' failed: {other}"),
        ),
    }
}

/// Convert rmcp's `#[non_exhaustive]` task status into Rig's
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
/// publishes from its `on_task_status` callback; every registered tool gets a
/// clone, and each
/// [`McpTaskHandle`] subscribes by task id. Publish/subscribe hold the lock
/// only briefly and never across an `.await`. Notifications for unknown task
/// ids are ignored, and a handle releases its slot when dropped, so the map is
/// bounded by live handles. A notification racing ahead of subscription may be
/// missed; polling remains authoritative as the MCP specification requires.
#[derive(Debug, Default)]
pub(crate) struct McpTaskNotifications {
    channels: Mutex<HashMap<String, tokio::sync::watch::Sender<Option<TaskStatusUpdate>>>>,
}

impl McpTaskNotifications {
    fn lock(
        &self,
    ) -> MutexGuard<'_, HashMap<String, tokio::sync::watch::Sender<Option<TaskStatusUpdate>>>> {
        self.channels.lock().unwrap_or_else(PoisonError::into_inner)
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
    /// Unknown task ids are ignored. Notifications are an optimization, and
    /// retaining unclaimed updates would allow the registry to grow without a
    /// live handle.
    pub(crate) fn publish(&self, task: &rmcp::model::Task) {
        let update = TaskStatusUpdate::from_task(task);
        let terminal = convert_status(&update.status).is_terminal();
        let mut channels = self.lock();
        match channels.entry(task.task_id.clone()) {
            Entry::Occupied(entry) => {
                entry.get().send_replace(Some(update));
                if terminal && entry.get().receiver_count() == 0 {
                    entry.remove();
                }
            }
            Entry::Vacant(_) => {}
        }
    }

    /// Release the slot for `task_id` (called when a handle is dropped).
    pub(crate) fn release(&self, task_id: &str) {
        self.lock().remove(task_id);
    }
}

/// Client-side task helpers for [`rmcp::service::ServerSink`] (SEP-1686),
/// absent from rmcp 2.2.0's `Peer<RoleClient>` API.
///
/// Every method drives the raw request through Rig's cancellable request
/// helper and hand-matches the [`ServerResult`] union; errors are classified
/// via [`McpToolError`].
pub(crate) trait ServerSinkTaskExt {
    /// `tools/call` augmented with task metadata — expects a
    /// `CreateTaskResult`. The `task` field must already be set on `params`.
    fn call_tool_as_task(
        &self,
        params: CallToolRequestParams,
        timeout: Option<Duration>,
    ) -> WasmBoxedFuture<'_, Result<CreateTaskResult, McpToolError>>;

    /// `tasks/get` — the task's current status snapshot.
    fn get_task<'a>(
        &'a self,
        task_id: &'a str,
        timeout: Option<Duration>,
    ) -> WasmBoxedFuture<'a, Result<rmcp::model::Task, McpToolError>>;

    /// `tasks/result` — blocks server-side until the task is terminal, then
    /// yields the original `CallToolResult`.
    fn get_task_payload<'a>(
        &'a self,
        task_id: &'a str,
        timeout: Option<Duration>,
    ) -> WasmBoxedFuture<'a, Result<CallToolResult, McpToolError>>;

    /// `tasks/cancel` — returns the post-cancel task snapshot.
    fn cancel_task<'a>(
        &'a self,
        task_id: &'a str,
        timeout: Option<Duration>,
    ) -> WasmBoxedFuture<'a, Result<rmcp::model::Task, McpToolError>>;

    /// The negotiated server-level tasks capability, if the handshake
    /// completed and the server declared one.
    fn tasks_capability(&self) -> Option<TasksCapability>;

    /// `"{name}@{version}"` identity key of the connected server, for
    /// [`ToolTaskDescriptor::server_key`].
    fn server_key(&self) -> Option<String>;
}

async fn send_task_request(
    sink: &rmcp::service::ServerSink,
    request: ClientRequest,
    timeout: Option<Duration>,
) -> Result<ServerResult, rmcp::ServiceError> {
    let deadline = timeout.and_then(|duration| {
        let deadline = Instant::now().checked_add(duration);
        if deadline.is_none() {
            tracing::warn!(
                ?duration,
                "MCP task request timeout exceeds the platform timer range; treating it as unbounded"
            );
        }
        deadline.map(|deadline| (deadline, duration))
    });
    send_mcp_request(sink, request, deadline).await
}

impl ServerSinkTaskExt for rmcp::service::ServerSink {
    fn call_tool_as_task(
        &self,
        params: CallToolRequestParams,
        timeout: Option<Duration>,
    ) -> WasmBoxedFuture<'_, Result<CreateTaskResult, McpToolError>> {
        Box::pin(async move {
            let request = ClientRequest::CallToolRequest(rmcp::model::Request::new(params));
            match send_task_request(self, request, timeout).await {
                Ok(ServerResult::CreateTaskResult(created)) => Ok(created),
                // Strict: a server that ignores task augmentation and answers
                // with a plain result is misbehaving; degrading silently would
                // hide the bug (the caller asked for a task lifecycle).
                Ok(ServerResult::CallToolResult(_)) => Err(McpToolError::new(
                    ToolErrorKind::Provider,
                    "MCP server ignored task augmentation and returned a plain tool result"
                        .to_string(),
                )),
                Ok(other) => Err(McpToolError::new(
                    ToolErrorKind::Provider,
                    format!("task-augmented tools/call returned an unexpected result: {other:?}"),
                )),
                Err(err) => Err(map_task_service_error(
                    McpTaskOperation::Launch,
                    "<new>",
                    err,
                )),
            }
        })
    }

    fn get_task<'a>(
        &'a self,
        task_id: &'a str,
        timeout: Option<Duration>,
    ) -> WasmBoxedFuture<'a, Result<rmcp::model::Task, McpToolError>> {
        Box::pin(async move {
            let request = ClientRequest::GetTaskRequest(GetTaskRequest::new(GetTaskParams::new(
                task_id.to_string(),
            )));
            match send_task_request(self, request, timeout).await {
                Ok(ServerResult::GetTaskResult(result)) => Ok(result.task),
                Ok(other) => Err(McpToolError::new(
                    ToolErrorKind::Provider,
                    format!("tasks/get returned an unexpected result: {other:?}"),
                )),
                Err(err) => Err(map_task_service_error(McpTaskOperation::Get, task_id, err)),
            }
        })
    }

    fn get_task_payload<'a>(
        &'a self,
        task_id: &'a str,
        timeout: Option<Duration>,
    ) -> WasmBoxedFuture<'a, Result<CallToolResult, McpToolError>> {
        Box::pin(async move {
            let request = ClientRequest::GetTaskPayloadRequest(GetTaskPayloadRequest::new(
                GetTaskPayloadParams::new(task_id.to_string()),
            ));
            match send_task_request(self, request, timeout).await {
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
                            ToolErrorKind::Provider,
                            format!(
                                "tasks/result payload for MCP task '{task_id}' was not a \
                                 CallToolResult: {err}"
                            ),
                        )
                    })
                }
                Ok(other) => Err(McpToolError::new(
                    ToolErrorKind::Provider,
                    format!("tasks/result returned an unexpected result: {other:?}"),
                )),
                Err(err) => Err(map_task_service_error(
                    McpTaskOperation::Result,
                    task_id,
                    err,
                )),
            }
        })
    }

    fn cancel_task<'a>(
        &'a self,
        task_id: &'a str,
        timeout: Option<Duration>,
    ) -> WasmBoxedFuture<'a, Result<rmcp::model::Task, McpToolError>> {
        Box::pin(async move {
            let request = ClientRequest::CancelTaskRequest(CancelTaskRequest::new(
                CancelTaskParams::new(task_id.to_string()),
            ));
            match send_task_request(self, request, timeout).await {
                Ok(ServerResult::CancelTaskResult(result)) => Ok(result.task),
                // `CancelTaskResult` and `GetTaskResult` share a wire shape
                // (`_meta` + flattened Task), and rmcp's untagged
                // `ServerResult` decode tries `GetTaskResult` first — accept
                // it as the equivalent task snapshot.
                Ok(ServerResult::GetTaskResult(result)) => Ok(result.task),
                Ok(other) => Err(McpToolError::new(
                    ToolErrorKind::Provider,
                    format!("tasks/cancel returned an unexpected result: {other:?}"),
                )),
                Err(err) => Err(map_task_service_error(
                    McpTaskOperation::Cancel,
                    task_id,
                    err,
                )),
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
/// Obtained when an MCP tool's canonical dispatch returns
/// [`ToolDispatch::Deferred`](crate::tool::ToolDispatch::Deferred), or
/// rehydrated from a persisted [`ToolTaskDescriptor`] via
/// [`McpTaskHandle::resume`]. Every RPC this handle issues is bounded by its
/// per-request timeout, so no single await can wedge (the same discipline as
/// [`McpClientHandler::with_timeout`](super::McpClientHandler::with_timeout),
/// issue #1914); the overall wait is unbounded — deadlines belong to the agent
/// loop.
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
    /// Whether the connection answers elicitations: `input_required` then
    /// means "waiting on the elicitation round-trip" rather than a dead end,
    /// and [`wait`](ToolTaskHandle::wait) keeps waiting instead of failing.
    elicitation_available: bool,
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
        elicitation_available: bool,
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
            elicitation_available,
        }
    }

    /// Rehydrate a handle from a persisted descriptor (durable resume). The
    /// caller supplies the live sink for `descriptor.server_key`.
    ///
    /// # Errors
    /// Returns an [`InvalidArgs`](ToolErrorKind::InvalidArgs) failure when
    /// the descriptor belongs to a different backend. A `server_key` mismatch
    /// only warns: keys are advisory (a server may legitimately report a new
    /// version after a redeploy) and the task id decides.
    pub fn resume(
        sink: rmcp::service::ServerSink,
        descriptor: ToolTaskDescriptor,
        request_timeout: Option<Duration>,
    ) -> Result<Self, ToolExecutionError> {
        Self::resume_with_options(sink, descriptor, request_timeout, None, false)
    }

    fn resume_with_options(
        sink: rmcp::service::ServerSink,
        descriptor: ToolTaskDescriptor,
        request_timeout: Option<Duration>,
        notifications: Option<Arc<McpTaskNotifications>>,
        elicitation_available: bool,
    ) -> Result<Self, ToolExecutionError> {
        if descriptor.backend != ToolTaskDescriptor::BACKEND_MCP {
            return Err(ToolExecutionError::invalid_args(format!(
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
            elicitation_available,
        })
    }

    /// Attach this handle's task metadata to a final result so hooks and
    /// telemetry can observe it (never sent to the model).
    fn attach_info(
        &self,
        mut result: ToolTaskResult,
        final_status: ToolTaskStatus,
        status_message: Option<String>,
    ) -> ToolTaskResult {
        result.context_mut().insert_result(McpTaskInfo {
            task_id: self.task_id.clone(),
            status: final_status,
            status_message,
            created_at: self.created_at.clone(),
            server_key: self.server_key.clone(),
        });
        result
    }

    /// A terminal-status snapshot that is not `Completed`, folded into the
    /// classified failure result [`wait`](ToolTaskHandle::wait) returns.
    fn terminal_failure(
        &self,
        status: ToolTaskStatus,
        status_message: Option<String>,
    ) -> ToolTaskResult {
        let result = match status {
            ToolTaskStatus::InputRequired => {
                let message = format!(
                    "MCP task '{}' for tool '{}' requires interactive input, but no elicitation \
                     handler is registered on this connection (see \
                     `McpClientHandler::with_elicitation_handler`)",
                    self.task_id, self.tool_name
                );
                ToolTaskResult::failed(
                    ToolExecutionError::other(message)
                        .with_code("mcp_task_input_required")
                        .with_retryable(false),
                )
            }
            ToolTaskStatus::Cancelled => {
                let message = status_message
                    .clone()
                    .unwrap_or_else(|| format!("MCP task '{}' was cancelled", self.task_id));
                ToolTaskResult::cancelled(ToolExecutionError::cancelled(message))
            }
            // Failed, or a defensive fold for a non-terminal status.
            _ => {
                let message = status_message
                    .clone()
                    .unwrap_or_else(|| format!("MCP task '{}' failed", self.task_id));
                ToolTaskResult::failed(ToolExecutionError::other(message))
            }
        };
        self.attach_info(result, status, status_message)
    }

    fn finish_payload(
        &self,
        result: CallToolResult,
        observed_status: Option<ToolTaskStatus>,
        status_message: Option<String>,
    ) -> ToolTaskResult {
        let is_error = result.is_error == Some(true);
        // MCP defines a task-backed tool call with `isError: true` as Failed,
        // not Completed. Infer from the canonical payload when no terminal
        // status snapshot won the race.
        let inferred_status = if is_error {
            ToolTaskStatus::Failed
        } else {
            ToolTaskStatus::Completed
        };
        let final_status = observed_status
            .filter(|status| status.is_terminal())
            .unwrap_or(inferred_status);
        let mut context = ToolContext::new();
        preserve_mcp_result(&mut context, &result);
        let completion = match mcp_result_output(&result) {
            Ok(output) if final_status == ToolTaskStatus::Cancelled => {
                let message = status_message
                    .clone()
                    .unwrap_or_else(|| format!("MCP task '{}' was cancelled", self.task_id));
                ToolTaskResult::cancelled(
                    ToolExecutionError::cancelled(message).with_model_output(output),
                )
            }
            Ok(output) if final_status == ToolTaskStatus::Failed && !is_error => {
                let message = status_message
                    .clone()
                    .unwrap_or_else(|| "the server reported Failed with a success payload".into());
                ToolTaskResult::failed(
                    ToolExecutionError::provider(format!(
                        "MCP task '{}' returned a status/payload mismatch: {message}",
                        self.task_id
                    ))
                    .with_code("mcp_task_status_mismatch")
                    .with_retryable(false)
                    .with_model_output(output),
                )
            }
            Ok(output) if final_status == ToolTaskStatus::Completed && is_error => {
                ToolTaskResult::failed(
                    ToolExecutionError::provider(format!(
                        "MCP task '{}' reported Completed with an error payload",
                        self.task_id
                    ))
                    .with_code("mcp_task_status_mismatch")
                    .with_retryable(false)
                    .with_model_output(output),
                )
            }
            Ok(output) if is_error => ToolTaskResult::failed(
                ToolExecutionError::other(format!(
                    "MCP task '{}' reported an execution error",
                    self.task_id
                ))
                .with_model_output(output),
            ),
            Ok(output) => ToolTaskResult::success(output),
            Err(error) if final_status == ToolTaskStatus::Cancelled => {
                let message = status_message
                    .clone()
                    .unwrap_or_else(|| format!("MCP task '{}' was cancelled", self.task_id));
                tracing::debug!(
                    task_id = %self.task_id,
                    %error,
                    "ignored malformed payload for a cancelled MCP task"
                );
                ToolTaskResult::cancelled(ToolExecutionError::cancelled(message))
            }
            Err(error) => ToolTaskResult::failed(error),
        }
        .with_context(context);
        self.attach_info(completion, final_status, status_message)
    }

    fn finish_payload_error(
        &self,
        error: McpToolError,
        observed_status: Option<ToolTaskStatus>,
        status_message: Option<String>,
    ) -> ToolTaskResult {
        let final_status = observed_status
            .filter(|status| status.is_terminal())
            .unwrap_or(ToolTaskStatus::Failed);
        let message = status_message.or_else(|| Some(error.to_string()));
        let result = match final_status {
            ToolTaskStatus::Cancelled => ToolTaskResult::cancelled(error),
            _ => ToolTaskResult::failed(error),
        };
        self.attach_info(result, final_status, message)
    }

    async fn refresh_terminal_observation(
        &self,
        observed_status: Option<ToolTaskStatus>,
        status_message: Option<String>,
    ) -> (Option<ToolTaskStatus>, Option<String>) {
        if observed_status.is_some_and(ToolTaskStatus::is_terminal) {
            return (observed_status, status_message);
        }
        match self
            .sink
            .get_task(&self.task_id, self.request_timeout)
            .await
        {
            Ok(task) => {
                let status = convert_status(&task.status);
                let message = task.status_message.or(status_message);
                (Some(status), message)
            }
            Err(error) => {
                tracing::debug!(
                    task_id = %self.task_id,
                    error = %error,
                    "could not refresh MCP task status after receiving its result"
                );
                (observed_status, status_message)
            }
        }
    }
}

impl Drop for McpTaskHandle {
    fn drop(&mut self) {
        if let Some(notifications) = &self.notifications {
            notifications.release(&self.task_id);
        }
    }
}

impl ToolTaskHandle for McpTaskHandle {
    fn task_id(&self) -> &str {
        &self.task_id
    }

    fn status(&self) -> WasmBoxedFuture<'_, Result<ToolTaskStatus, ToolExecutionError>> {
        Box::pin(async move {
            self.sink
                .get_task(&self.task_id, self.request_timeout)
                .await
                .map(|task| convert_status(&task.status))
        })
    }

    fn wait(&self) -> WasmBoxedFuture<'_, ToolTaskResult> {
        Box::pin(async move {
            let mut watch = self.watch.clone();
            let mut pending_update = watch
                .as_mut()
                .and_then(|receiver| receiver.borrow_and_update().clone());
            let mut poll_interval = self.poll_interval;
            let mut observed_status = None;
            let mut status_message = None;
            // Keep exactly one blocking tasks/result request alive between
            // notification and poll ticks. Dropping and recreating it on every
            // tick abandons requests at the transport and can accumulate
            // server-side long polls.
            let mut payload = self
                .sink
                .get_task_payload(&self.task_id, self.request_timeout);

            loop {
                let snapshot = if let Some(update) = pending_update.take() {
                    Some(update)
                } else {
                    let interval = poll_interval.unwrap_or(DEFAULT_MCP_TASK_POLL_INTERVAL);
                    let cadence = pin!(async {
                        sleep_poll_interval(interval).await;
                    });
                    let wakeup = pin!(async {
                        match watch.as_mut() {
                            Some(receiver) => loop {
                                // A closed channel means the registry slot was
                                // dropped; fall back to pure cadence polling.
                                if receiver.changed().await.is_err() {
                                    return pending::<TaskStatusUpdate>().await;
                                }
                                if let Some(update) = receiver.borrow_and_update().clone() {
                                    break update;
                                }
                            },
                            None => pending::<TaskStatusUpdate>().await,
                        }
                    });
                    match select(payload, select(wakeup, cadence)).await {
                        Either::Left((result, _)) => match result {
                            Ok(result) => {
                                let (observed_status, status_message) = self
                                    .refresh_terminal_observation(observed_status, status_message)
                                    .await;
                                return self.finish_payload(
                                    result,
                                    observed_status,
                                    status_message,
                                );
                            }
                            // A per-request timeout on the long-poll is the
                            // expected slow-task outcome. The cancellable
                            // request helper already sent JSON-RPC
                            // cancellation; reopen one long poll and refresh
                            // status.
                            Err(err) if err.kind() == ToolErrorKind::Timeout => {
                                payload = self
                                    .sink
                                    .get_task_payload(&self.task_id, self.request_timeout);
                                None
                            }
                            Err(error) => {
                                return self.finish_payload_error(
                                    error,
                                    observed_status,
                                    status_message,
                                );
                            }
                        },
                        Either::Right((Either::Left((update, _)), pending_payload)) => {
                            payload = pending_payload;
                            Some(update)
                        }
                        Either::Right((Either::Right(_), pending_payload)) => {
                            payload = pending_payload;
                            None
                        }
                    }
                };

                let (status, message, suggested_interval) = match snapshot {
                    Some(update) => (
                        convert_status(&update.status),
                        update.status_message,
                        update.poll_interval,
                    ),
                    None => {
                        match self
                            .sink
                            .get_task(&self.task_id, self.request_timeout)
                            .await
                        {
                            Ok(task) => (
                                convert_status(&task.status),
                                task.status_message,
                                task.poll_interval,
                            ),
                            // A transient status-poll failure must not abort
                            // the wait; NotFound (expired) must.
                            Err(error) if error.kind() == ToolErrorKind::NotFound => {
                                return self.finish_payload_error(
                                    error,
                                    observed_status,
                                    status_message,
                                );
                            }
                            Err(error) => {
                                tracing::warn!(
                                    task_id = %self.task_id,
                                    error = %error,
                                    "transient MCP task status poll failure"
                                );
                                (ToolTaskStatus::Working, None, None)
                            }
                        }
                    }
                };
                observed_status = Some(status);
                if message.is_some() {
                    status_message = message;
                }
                if let Some(interval) = suggested_interval {
                    poll_interval = Some(clamp_poll_interval(interval));
                }
                match status {
                    ToolTaskStatus::Working
                    | ToolTaskStatus::Completed
                    | ToolTaskStatus::Failed
                    | ToolTaskStatus::Cancelled => {}
                    // With an elicitation handler on the connection,
                    // input_required means the interaction can progress over
                    // the still-open tasks/result request. Without one, local
                    // execution cannot continue and fails explicitly.
                    ToolTaskStatus::InputRequired if self.elicitation_available => {}
                    ToolTaskStatus::InputRequired => {
                        return self.terminal_failure(status, status_message);
                    }
                }
            }
        })
    }

    fn cancel(&self) -> WasmBoxedFuture<'_, Result<(), ToolExecutionError>> {
        Box::pin(async move {
            self.sink
                .cancel_task(&self.task_id, self.request_timeout)
                .await
                .map(|_| ())
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
/// and register it on the agent runner. Descriptors for other backends, for a
/// different `server_key`, or for a task id this connection does not own yield
/// `Ok(None)` so the next registered resumer is consulted. The task-id check
/// disambiguates separate servers that advertise the same `name@version`.
pub struct McpTaskResumer {
    sink: rmcp::service::ServerSink,
    request_timeout: Option<Duration>,
    notifications: Option<Arc<McpTaskNotifications>>,
    /// Whether the connection answers elicitations; resumed handles inherit it
    /// so `input_required` keeps waiting (see [`McpTaskHandle`]).
    elicitation_available: bool,
}

impl McpTaskResumer {
    /// Create a resumer over a live server sink.
    pub fn new(sink: rmcp::service::ServerSink, request_timeout: Option<Duration>) -> Self {
        Self {
            sink,
            request_timeout,
            notifications: None,
            elicitation_available: false,
        }
    }

    pub(crate) fn with_notifications(mut self, notifications: Arc<McpTaskNotifications>) -> Self {
        self.notifications = Some(notifications);
        self
    }

    /// Mark the connection as elicitation-capable, so resumed handles keep
    /// waiting through `input_required` instead of failing.
    pub fn with_elicitation_available(mut self, available: bool) -> Self {
        self.elicitation_available = available;
        self
    }
}

impl TaskResumer for McpTaskResumer {
    fn resume<'a>(
        &'a self,
        descriptor: &'a ToolTaskDescriptor,
    ) -> WasmBoxedFuture<'a, Result<Option<Box<dyn ToolTaskHandle>>, ToolExecutionError>> {
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
            let handle = McpTaskHandle::resume_with_options(
                self.sink.clone(),
                descriptor.clone(),
                self.request_timeout,
                self.notifications.clone(),
                self.elicitation_available,
            )?;
            match handle.status().await {
                Ok(_) => Ok(Some(Box::new(handle) as Box<dyn ToolTaskHandle>)),
                Err(error) if error.kind() == ToolErrorKind::NotFound => Ok(None),
                Err(error) => Err(error),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
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
    use crate::tool::{ErasedTool, ToolContext, ToolDispatch, ToolResult};

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
        state: Arc<RwLock<HashMap<String, TaskEntry>>>,
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

        async fn complete_with_tool_error(&self, task_id: &str, message: &str) {
            let mut state = self.state.write().await;
            if let Some((task, payload)) = state.get_mut(task_id) {
                task.status = TaskStatus::Completed;
                *payload = Some(CallToolResult::error(vec![ContentBlock::text(message)]));
            }
            drop(state);
            self.terminal.notify_waiters();
        }

        async fn fail(&self, task_id: &str, message: &str) {
            let mut state = self.state.write().await;
            if let Some((task, payload)) = state.get_mut(task_id) {
                task.status = TaskStatus::Failed;
                task.status_message = Some(message.to_string());
                *payload = Some(CallToolResult::error(vec![ContentBlock::text(message)]));
            }
            drop(state);
            self.terminal.notify_waiters();
        }

        async fn fail_with_success_payload(&self, task_id: &str, message: &str) {
            let mut state = self.state.write().await;
            if let Some((task, payload)) = state.get_mut(task_id) {
                task.status = TaskStatus::Failed;
                task.status_message = Some(message.to_string());
                *payload = Some(CallToolResult::success(vec![ContentBlock::text(message)]));
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
            self.recorded.write().await.push("get_task_result");
            if self.hang_results {
                pending::<()>().await;
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
            let task = match state.get_mut(&request.task_id) {
                Some((task, payload)) => {
                    task.status = TaskStatus::Cancelled;
                    task.status_message = Some("cancelled by request".to_string());
                    *payload = Some(CallToolResult::error(vec![ContentBlock::text(
                        "cancelled by request",
                    )]));
                    task.clone()
                }
                None => return Err(ErrorData::invalid_params("task not found", None)),
            };
            drop(state);
            self.terminal.notify_waiters();
            Ok(rmcp::model::CancelTaskResult::new(task))
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

    async fn dispatch(tool: &McpTool) -> ToolDispatch {
        ErasedTool::dispatch(tool, "{}".to_string(), &mut ToolContext::new()).await
    }

    fn expect_completed(dispatch: ToolDispatch) -> ToolResult {
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
                result.output()
            ),
        }
    }

    fn expect_failure(result: &ToolResult) -> &ToolExecutionError {
        result
            .error()
            .unwrap_or_else(|| panic!("expected an error result, got {result:?}"))
    }

    fn expect_task_failure(result: &ToolTaskResult) -> &ToolExecutionError {
        expect_failure(result.result())
    }

    #[tokio::test]
    async fn forbidden_tool_never_dispatches_as_task() {
        let server = ControlledTaskServer::new(None);
        let client = connect_bare(server.clone()).await;
        let tool = bare_tool(&client, McpTaskPolicy::Preferred).await;

        let dispatch = dispatch(&tool).await;
        let result = expect_completed(dispatch);
        assert_eq!(result.output().render(), "plain:work");
        assert_eq!(server.recorded_calls().await, vec!["call_tool"]);
    }

    #[tokio::test]
    async fn required_tool_with_policy_never_fails_fast() {
        let server = ControlledTaskServer::new(Some(TaskSupport::Required));
        let client = connect_bare(server.clone()).await;
        let tool = bare_tool(&client, McpTaskPolicy::Never).await;

        let dispatch = dispatch(&tool).await;
        let result = expect_completed(dispatch);
        let failure = expect_failure(&result);
        assert_eq!(failure.kind(), ToolErrorKind::Other);
        assert_eq!(failure.code(), Some("mcp_task_required"));
        assert_eq!(failure.retryable(), Some(false));
        // Fail-fast: the server never saw the call.
        assert!(server.recorded_calls().await.is_empty());
    }

    #[tokio::test]
    async fn required_tool_default_policy_dispatches_deferred() {
        let server = ControlledTaskServer::new(Some(TaskSupport::Required));
        let client = connect_bare(server.clone()).await;
        // Bare-tool default policy is Required: task the call.
        let tool = bare_tool(&client, McpTaskPolicy::default()).await;

        let dispatch = dispatch(&tool).await;
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
        let result = expect_completed(dispatch(&tool).await);
        assert_eq!(result.output().render(), "plain:work");

        // Preferred: optional tools become tasks.
        let tool = bare_tool(&client, McpTaskPolicy::Preferred).await;
        let handle = expect_deferred(dispatch(&tool).await);
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
        let result = expect_completed(dispatch(&tool).await);
        assert_eq!(result.output().render(), "plain:work");
        assert_eq!(server.recorded_calls().await, vec!["call_tool"]);

        // Required tool on an inconsistent (no-capability) server fails before
        // dispatch instead of circumventing taskSupport with a plain call.
        let server = ControlledTaskServer::new(Some(TaskSupport::Required)).without_capability();
        let client = connect_bare(server.clone()).await;
        let tool = bare_tool(&client, McpTaskPolicy::Preferred).await;
        let dispatch = dispatch(&tool).await;
        let result = expect_completed(dispatch);
        let failure = expect_failure(&result);
        assert_eq!(failure.kind(), ToolErrorKind::Provider);
        assert_eq!(failure.code(), Some("mcp_task_capability_missing"));
        assert!(server.recorded_calls().await.is_empty());
    }

    #[tokio::test]
    async fn deferred_wait_happy_path() {
        let server = ControlledTaskServer::new(Some(TaskSupport::Required));
        let client = connect_bare(server.clone()).await;
        let tool = bare_tool(&client, McpTaskPolicy::Required).await;

        let handle = expect_deferred(dispatch(&tool).await);
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

        assert_eq!(result.result().output().render(), "task output");
        assert!(result.result().is_success());
        let info = result
            .context()
            .result::<McpTaskInfo>()
            .expect("final result carries McpTaskInfo");
        assert_eq!(info.task_id, task_id);
        assert_eq!(info.status, ToolTaskStatus::Completed);
        assert_eq!(info.server_key.as_deref(), Some("test-task-server@0.1.0"));
    }

    #[tokio::test]
    async fn immediate_response_hint_is_extracted() {
        let server = ControlledTaskServer::new(Some(TaskSupport::Required))
            .with_immediate_response("working on it");
        let client = connect_bare(server.clone()).await;
        let tool = bare_tool(&client, McpTaskPolicy::Required).await;

        let handle = expect_deferred(dispatch(&tool).await);
        assert_eq!(handle.immediate_response(), Some("working on it"));
        assert_eq!(
            handle.descriptor().immediate_response.as_deref(),
            Some("working on it")
        );
    }

    #[tokio::test]
    async fn cancel_maps_to_cancelled() {
        let server = ControlledTaskServer::new(Some(TaskSupport::Required));
        let client = connect_bare(server.clone()).await;
        let tool = bare_tool(&client, McpTaskPolicy::Required).await;

        let handle = expect_deferred(dispatch(&tool).await);
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
            .expect("wait resolves with the cancelled task result");
        let failure = expect_task_failure(&result);
        assert_eq!(failure.kind(), ToolErrorKind::Cancelled);
        let info = result
            .context()
            .result::<McpTaskInfo>()
            .expect("cancelled result carries McpTaskInfo");
        assert_eq!(info.status, ToolTaskStatus::Cancelled);
    }

    #[tokio::test]
    async fn failed_status_carries_status_message() {
        let server = ControlledTaskServer::new(Some(TaskSupport::Required));
        let client = connect_bare(server.clone()).await;
        let tool = bare_tool(&client, McpTaskPolicy::Required).await;

        let handle = expect_deferred(dispatch(&tool).await);
        let task_id = handle.task_id().to_string();
        server.fail(&task_id, "disk full").await;

        let result = tokio::time::timeout(Duration::from_secs(5), handle.wait())
            .await
            .expect("wait resolves with the failed task result");
        let failure = expect_task_failure(&result);
        assert_eq!(failure.kind(), ToolErrorKind::Other);
        assert!(
            result.result().output().render().contains("disk full"),
            "status message must reach the model output, got {:?}",
            result.result().output()
        );
        let info = result
            .context()
            .result::<McpTaskInfo>()
            .expect("failed result carries McpTaskInfo");
        assert_eq!(info.status, ToolTaskStatus::Failed);
        assert_eq!(info.status_message.as_deref(), Some("disk full"));
    }

    #[tokio::test]
    async fn completed_status_with_an_error_payload_is_a_provider_failure() {
        let server = ControlledTaskServer::new(Some(TaskSupport::Required));
        let client = connect_bare(server.clone()).await;
        let tool = bare_tool(&client, McpTaskPolicy::Required).await;

        let handle = expect_deferred(dispatch(&tool).await);
        let task_id = handle.task_id().to_string();
        server
            .complete_with_tool_error(&task_id, "tool rejected the input")
            .await;

        let result = handle.wait().await;
        assert_eq!(result.status(), ToolTaskStatus::Failed);
        let failure = expect_task_failure(&result);
        assert_eq!(failure.kind(), ToolErrorKind::Provider);
        assert_eq!(failure.code(), Some("mcp_task_status_mismatch"));
        assert!(
            result
                .result()
                .output()
                .render()
                .contains("tool rejected the input")
        );
        let info = result
            .context()
            .result::<McpTaskInfo>()
            .expect("result carries McpTaskInfo");
        assert_eq!(info.status, ToolTaskStatus::Completed);
    }

    #[tokio::test]
    async fn failed_status_with_a_success_payload_is_a_provider_failure() {
        let server = ControlledTaskServer::new(Some(TaskSupport::Required));
        let client = connect_bare(server.clone()).await;
        let tool = bare_tool(&client, McpTaskPolicy::Required).await;

        let handle = expect_deferred(dispatch(&tool).await);
        let task_id = handle.task_id().to_string();
        server
            .fail_with_success_payload(&task_id, "backend failed")
            .await;

        let result = handle.wait().await;
        assert_eq!(result.status(), ToolTaskStatus::Failed);
        let failure = expect_task_failure(&result);
        assert_eq!(failure.kind(), ToolErrorKind::Provider);
        assert_eq!(failure.code(), Some("mcp_task_status_mismatch"));
        assert!(result.result().output().render().contains("backend failed"));
        let info = result
            .context()
            .result::<McpTaskInfo>()
            .expect("result carries McpTaskInfo");
        assert_eq!(info.status, ToolTaskStatus::Failed);
    }

    #[tokio::test]
    async fn notification_does_not_abandon_the_result_request() {
        // A notification may wake status processing, but the one canonical
        // tasks/result long poll must remain alive until its payload exists.
        let server =
            ControlledTaskServer::new(Some(TaskSupport::Required)).with_poll_interval_ms(60_000);

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

        // Handler default policy is Required; the required-task tool defers.
        let dispatched = tool_server_handle
            .dispatch("work", "{}", &ToolContext::new())
            .await;
        let handle = expect_deferred(dispatched.outcome);
        let task_id = handle.task_id().to_string();

        let wait = handle.wait();
        tokio::pin!(wait);
        tokio::time::timeout(Duration::from_millis(50), wait.as_mut())
            .await
            .expect_err("the result must still be pending");

        server.set_status(&task_id, TaskStatus::Failed).await;
        let failed_task = server.snapshot(&task_id).await.expect("task exists");
        server_service
            .peer()
            .send_notification(ServerNotification::TaskStatusNotification(
                TaskStatusNotification::new(TaskStatusNotificationParam::new(failed_task)),
            ))
            .await
            .expect("notification sent");

        tokio::time::timeout(Duration::from_millis(50), wait.as_mut())
            .await
            .expect_err("a status notification is not a substitute for tasks/result");
        assert_eq!(
            server
                .recorded_calls()
                .await
                .iter()
                .filter(|operation| **operation == "get_task_result")
                .count(),
            1,
            "status wakeups must not abandon and recreate the long poll"
        );

        server.fail(&task_id, "boom").await;
        let result = tokio::time::timeout(Duration::from_secs(5), wait.as_mut())
            .await
            .expect("the original result request resolves once the payload exists");
        let failure = expect_task_failure(&result);
        assert_eq!(failure.kind(), ToolErrorKind::Other);
        assert!(result.result().output().render().contains("boom"));
    }

    #[tokio::test]
    async fn expired_task_maps_to_not_found() {
        let server = ControlledTaskServer::new(Some(TaskSupport::Required));
        let client = connect_bare(server.clone()).await;
        let tool = bare_tool(&client, McpTaskPolicy::Required).await;

        let handle = expect_deferred(dispatch(&tool).await);
        let task_id = handle.task_id().to_string();
        server.forget(&task_id).await;

        let status = handle
            .status()
            .await
            .expect_err("status of an expired task errs");
        assert_eq!(status.kind(), ToolErrorKind::NotFound);

        let result = tokio::time::timeout(Duration::from_secs(5), handle.wait())
            .await
            .expect("wait resolves fast on an expired task");
        let failure = expect_task_failure(&result);
        assert_eq!(failure.kind(), ToolErrorKind::NotFound);
    }

    #[tokio::test]
    async fn resumer_disambiguates_servers_with_the_same_identity_key() {
        let first_server = ControlledTaskServer::new(Some(TaskSupport::Required));
        let first_client = connect_bare(first_server).await;
        let second_server = ControlledTaskServer::new(Some(TaskSupport::Required));
        let second_client = connect_bare(second_server).await;
        let tool = bare_tool(&second_client, McpTaskPolicy::Required).await;
        let handle = expect_deferred(dispatch(&tool).await);
        let descriptor = handle.descriptor();
        drop(handle);

        let wrong = McpTaskResumer::new(first_client.peer().clone(), None)
            .resume(&descriptor)
            .await
            .expect("a missing task is not a resumer error");
        assert!(
            wrong.is_none(),
            "the wrong same-key server must not claim it"
        );

        let right = McpTaskResumer::new(second_client.peer().clone(), None)
            .resume(&descriptor)
            .await
            .expect("the owning server responds")
            .expect("the owning server claims the task");
        assert_eq!(right.task_id(), descriptor.task_id);
    }

    #[tokio::test]
    async fn input_required_surfaces_as_classified_failure() {
        let server = ControlledTaskServer::new(Some(TaskSupport::Required)).with_hanging_results();
        let client = connect_bare(server.clone()).await;
        let tool = bare_tool(&client, McpTaskPolicy::Required).await;

        let handle = expect_deferred(dispatch(&tool).await);
        let task_id = handle.task_id().to_string();
        server.set_status(&task_id, TaskStatus::InputRequired).await;

        assert_eq!(
            handle.status().await.expect("status"),
            ToolTaskStatus::InputRequired
        );

        let result = tokio::time::timeout(Duration::from_secs(5), handle.wait())
            .await
            .expect("wait resolves via the status poll");
        let failure = expect_task_failure(&result);
        assert_eq!(failure.code(), Some("mcp_task_input_required"));
        assert_eq!(failure.retryable(), Some(false));
        let info = result
            .context()
            .result::<McpTaskInfo>()
            .expect("input_required result carries McpTaskInfo");
        assert_eq!(info.status, ToolTaskStatus::InputRequired);
    }

    #[test]
    fn map_task_service_error_classifies_kinds() {
        let cases = [
            (
                rmcp::ServiceError::McpError(ErrorData::invalid_params("gone", None)),
                ToolErrorKind::NotFound,
            ),
            (
                rmcp::ServiceError::McpError(ErrorData::method_not_found::<
                    rmcp::model::GetTaskMethod,
                >()),
                ToolErrorKind::Provider,
            ),
            (
                rmcp::ServiceError::McpError(ErrorData::internal_error("boom", None)),
                ToolErrorKind::Provider,
            ),
            (
                rmcp::ServiceError::Timeout {
                    timeout: Duration::from_secs(1),
                },
                ToolErrorKind::Timeout,
            ),
            (
                rmcp::ServiceError::Cancelled { reason: None },
                ToolErrorKind::Cancelled,
            ),
            (rmcp::ServiceError::TransportClosed, ToolErrorKind::Network),
            (
                rmcp::ServiceError::UnexpectedResponse,
                ToolErrorKind::Provider,
            ),
        ];
        for (err, expected) in cases {
            let mapped = map_task_service_error(McpTaskOperation::Get, "task-1", err);
            assert_eq!(mapped.kind(), expected);
        }

        let already_terminal = map_task_service_error(
            McpTaskOperation::Cancel,
            "task-1",
            rmcp::ServiceError::McpError(ErrorData::invalid_params("already terminal", None)),
        );
        assert_eq!(already_terminal.kind(), ToolErrorKind::Provider);
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
    fn notification_registry_tracks_only_live_handles() {
        let registry = McpTaskNotifications::default();
        let working = Task::new(
            "task-1".to_string(),
            TaskStatus::Working,
            "2026-01-01T00:00:00Z".to_string(),
            "2026-01-01T00:00:00Z".to_string(),
        );

        // Unknown notifications are disposable; polling is authoritative.
        registry.publish(&working);
        assert!(registry.lock().is_empty());

        // Once a handle subscribes, updates are routed to it.
        let receiver = registry.subscribe("task-1");
        registry.publish(&working);
        let routed = receiver.borrow().clone().expect("routed update");
        assert_eq!(routed.status, TaskStatus::Working);

        // A terminal update with a live subscriber keeps the slot...
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

#[cfg(test)]
mod elicitation_tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Mutex, PoisonError};
    use std::time::Duration;

    use rmcp::handler::client::ClientHandler;
    use rmcp::model::{
        CallToolRequestParams, CallToolResult, ClientInfo, ContentBlock, ElicitRequestParams,
        ElicitResult, ElicitationAction, ErrorData, GetTaskParams, GetTaskPayloadParams,
        Implementation, ListToolsResult, Meta, PaginatedRequestParams, ProtocolVersion,
        RelatedTaskMetadata, ServerCapabilities, ServerInfo, Task, TaskStatus, TaskSupport,
        TasksCapability, Tool, ToolExecution,
    };
    use rmcp::service::{RequestContext, RoleServer};
    use rmcp::{ServerHandler, ServiceExt};
    use tokio::sync::{Notify, RwLock};

    use super::*;
    use crate::agent::run::TaskCompletionPolicy;
    use crate::tool::rmcp::McpClientHandler;
    use crate::tool::rmcp::elicitation::{McpElicitationHandler, related_task_id};
    use crate::tool::server::{ToolServer, ToolServerHandle};
    use crate::tool::{ToolContext, ToolDispatch, ToolTaskDescriptor};
    use crate::wasm_compat::WasmBoxedFuture;

    type TaskEntry = (Task, Option<CallToolResult>);

    /// A task server whose tasks pause in `input_required` and elicit an
    /// `answer` from the client; an `Accept` completes the task with a payload
    /// derived from the elicited content, a `Decline`/`Cancel` fails it.
    #[derive(Clone)]
    struct ElicitingTaskServer {
        state: Arc<RwLock<HashMap<String, TaskEntry>>>,
        terminal: Arc<Notify>,
        next_id: Arc<AtomicUsize>,
    }

    impl ElicitingTaskServer {
        fn new() -> Self {
            Self {
                state: Arc::default(),
                terminal: Arc::default(),
                next_id: Arc::default(),
            }
        }

        fn tool() -> Tool {
            Tool::new(
                "work".to_string(),
                "needs human input".to_string(),
                Arc::new(serde_json::Map::new()),
            )
            .with_execution(ToolExecution::from_raw(Some(TaskSupport::Required)))
        }

        async fn settle(&self, task_id: &str, status: TaskStatus, payload: Option<CallToolResult>) {
            let mut state = self.state.write().await;
            if let Some((task, slot)) = state.get_mut(task_id) {
                task.status = status;
                *slot = payload;
            }
            drop(state);
            self.terminal.notify_waiters();
        }
    }

    impl ServerHandler for ElicitingTaskServer {
        fn get_info(&self) -> ServerInfo {
            let mut capabilities = ServerCapabilities::builder().enable_tools().build();
            capabilities.tasks = Some(TasksCapability::server_default());
            ServerInfo::new(capabilities)
                .with_protocol_version(ProtocolVersion::LATEST)
                .with_server_info(Implementation::new("eliciting-server", "0.1.0"))
        }

        fn get_tool(&self, name: &str) -> Option<Tool> {
            (name == "work").then(Self::tool)
        }

        async fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            Ok(ListToolsResult::with_all_items(vec![Self::tool()]))
        }

        async fn enqueue_task(
            &self,
            _request: CallToolRequestParams,
            context: RequestContext<RoleServer>,
        ) -> Result<rmcp::model::CreateTaskResult, ErrorData> {
            let id = format!(
                "elicit-task-{}",
                self.next_id.fetch_add(1, Ordering::SeqCst)
            );
            let now = "2026-01-01T00:00:00Z".to_string();
            let task = Task::new(id.clone(), TaskStatus::InputRequired, now.clone(), now)
                .with_poll_interval(25);
            self.state
                .write()
                .await
                .insert(id.clone(), (task.clone(), None));

            // Elicit from the client out-of-band (never inside the request
            // handler, which must answer first).
            let server = self.clone();
            let peer = context.peer.clone();
            tokio::spawn(async move {
                let mut meta = Meta::new();
                meta.0.insert(
                    RelatedTaskMetadata::META_KEY.to_string(),
                    serde_json::json!({ "taskId": id }),
                );
                let schema = rmcp::model::ElicitationSchema::builder()
                    .required_string("answer")
                    .build()
                    .expect("schema builds");
                let params = ElicitRequestParams::FormElicitationParams {
                    meta: Some(meta),
                    message: "What is the answer?".to_string(),
                    requested_schema: schema,
                };
                match peer.create_elicitation(params).await {
                    Ok(result) if result.action == ElicitationAction::Accept => {
                        let answer = result
                            .content
                            .as_ref()
                            .and_then(|content| content.get("answer"))
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("<missing>")
                            .to_string();
                        server
                            .settle(
                                &id,
                                TaskStatus::Completed,
                                Some(CallToolResult::success(vec![ContentBlock::text(format!(
                                    "elicited:{answer}"
                                ))])),
                            )
                            .await;
                    }
                    Ok(_) => {
                        let mut state = server.state.write().await;
                        if let Some((task, payload)) = state.get_mut(&id) {
                            task.status = TaskStatus::Failed;
                            task.status_message = Some("input declined".to_string());
                            *payload = Some(CallToolResult::error(vec![ContentBlock::text(
                                "input declined",
                            )]));
                        }
                        drop(state);
                        server.terminal.notify_waiters();
                    }
                    Err(err) => {
                        tracing::warn!("elicitation request failed: {err}");
                    }
                }
            });

            Ok(rmcp::model::CreateTaskResult::new(task))
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
    }

    /// A recorded elicitation observation: the parsed related-task id and the
    /// request message.
    type SeenElicitation = (Option<String>, String);

    /// Answers every form elicitation with `Accept { answer }`, recording the
    /// message and the parsed related-task id.
    #[derive(Clone)]
    struct RecordingElicitationHandler {
        answer: String,
        seen: Arc<Mutex<Vec<SeenElicitation>>>,
    }

    impl RecordingElicitationHandler {
        fn new(answer: &str) -> Self {
            Self {
                answer: answer.to_string(),
                seen: Arc::default(),
            }
        }

        fn seen(&self) -> Vec<SeenElicitation> {
            self.seen
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
        }
    }

    impl McpElicitationHandler for RecordingElicitationHandler {
        fn elicit(
            &self,
            request: ElicitRequestParams,
        ) -> WasmBoxedFuture<'_, Result<ElicitResult, rmcp::ErrorData>> {
            Box::pin(async move {
                use rmcp::model::RequestParamsMeta;
                let related = related_task_id(request.meta());
                let message = match &request {
                    ElicitRequestParams::FormElicitationParams { message, .. } => message.clone(),
                    ElicitRequestParams::UrlElicitationParams { message, .. } => message.clone(),
                    _ => String::new(),
                };
                self.seen
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push((related, message));
                Ok(ElicitResult::new(ElicitationAction::Accept)
                    .with_content(serde_json::json!({ "answer": self.answer })))
            })
        }
    }

    /// Connect an `ElicitingTaskServer` through an `McpClientHandler` (with or
    /// without an elicitation handler) and return the shared tool server handle.
    async fn connect_with(
        server: ElicitingTaskServer,
        handler: Option<RecordingElicitationHandler>,
    ) -> (
        ToolServerHandle,
        rmcp::service::RunningService<rmcp::service::RoleClient, McpClientHandler>,
    ) {
        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);
        tokio::spawn(async move {
            let running = server
                .serve((server_from_client, server_to_client))
                .await
                .expect("server failed to start");
            running.waiting().await.ok();
        });
        let tool_server_handle = ToolServer::new().run();
        let mut client = McpClientHandler::new(ClientInfo::default(), tool_server_handle.clone());
        if let Some(handler) = handler {
            client = client.with_elicitation_handler(handler);
        }
        let service = client
            .connect((client_from_server, client_to_server))
            .await
            .expect("connect failed");
        (tool_server_handle, service)
    }

    async fn dispatch(handle: &ToolServerHandle) -> ToolDispatch {
        handle
            .dispatch("work", "{}", &ToolContext::new())
            .await
            .outcome
    }

    #[test]
    fn get_info_declares_elicitation_capability() {
        let tool_server_handle = ToolServer::new().run();
        let bare = McpClientHandler::new(ClientInfo::default(), tool_server_handle.clone());
        assert!(
            bare.get_info().capabilities.elicitation.is_none(),
            "no handler ⇒ no capability"
        );

        let with_handler = McpClientHandler::new(ClientInfo::default(), tool_server_handle)
            .with_elicitation_handler(RecordingElicitationHandler::new("42"));
        let capability = with_handler
            .get_info()
            .capabilities
            .elicitation
            .expect("registered handler must be advertised");
        assert!(capability.form.is_some(), "default capability is form-mode");
        assert!(capability.url.is_none());
    }

    #[tokio::test]
    async fn elicitation_completes_an_input_required_task() {
        let server = ElicitingTaskServer::new();
        let handler = RecordingElicitationHandler::new("blue");
        let (tool_server_handle, _service) = connect_with(server, Some(handler.clone())).await;

        let dispatch = dispatch(&tool_server_handle).await;
        let ToolDispatch::Deferred(handle) = dispatch else {
            panic!("required-task tool must defer");
        };
        let task_id = handle.task_id().to_string();

        let result = tokio::time::timeout(Duration::from_secs(5), handle.wait())
            .await
            .expect("wait must survive input_required and resolve after the elicitation");

        assert!(result.result().is_success());
        assert_eq!(result.result().output().render(), "elicited:blue");
        let info = result
            .context()
            .result::<McpTaskInfo>()
            .expect("final result carries McpTaskInfo");
        assert_eq!(info.status, ToolTaskStatus::Completed);

        // The handler observed exactly one request, correlated to the task.
        let seen = handler.seen();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].0.as_deref(), Some(task_id.as_str()));
        assert_eq!(seen[0].1, "What is the answer?");
    }

    #[tokio::test]
    async fn no_handler_declines_and_input_required_stays_a_failure() {
        let server = ElicitingTaskServer::new();
        let (tool_server_handle, _service) = connect_with(server, None).await;

        let dispatch = dispatch(&tool_server_handle).await;
        let ToolDispatch::Deferred(handle) = dispatch else {
            panic!("required-task tool must defer");
        };

        // The server's elicitation is auto-declined (today's behavior), so the
        // task fails server-side; regardless, the handle without the flag
        // fails fast the moment it observes input_required.
        let result = tokio::time::timeout(Duration::from_secs(5), handle.wait())
            .await
            .expect("wait resolves fast without a handler");
        let failure = result
            .result()
            .error()
            .unwrap_or_else(|| panic!("expected a failure result, got {result:?}"));
        assert!(
            failure.code() == Some("mcp_task_input_required")
                || result.result().output().render().contains("input declined"),
            "either the fail-fast or the server-side decline must surface: {failure:?}"
        );
    }

    #[tokio::test]
    async fn agent_loop_end_to_end_elicitation() {
        use crate::agent::AgentBuilder;
        use crate::test_utils::{MockCompletionModel, MockTurn};

        let server = ElicitingTaskServer::new();
        let handler = RecordingElicitationHandler::new("the moon");
        let (tool_server_handle, _service) = connect_with(server, Some(handler)).await;

        let model = MockCompletionModel::from_turns([
            MockTurn::tool_call("tc1", "work", serde_json::json!({})),
            MockTurn::text("done with human help"),
        ]);
        let response = AgentBuilder::new(model)
            .tool_server_handle(tool_server_handle)
            .build()
            .runner("ask the human")
            .max_turns(4)
            .task_completion_policy(TaskCompletionPolicy::JoinTurn)
            .run()
            .await
            .expect("run should succeed");

        assert_eq!(response.output, "done with human help");
        let history = serde_json::to_string(&response.messages).expect("history serializes");
        assert!(
            history.contains("elicited:the moon"),
            "the elicited value must reach the model: {history}"
        );
    }

    #[tokio::test]
    async fn resumer_from_elicitation_handler_survives_input_required() {
        use crate::tool::TaskResumer;

        let server = ElicitingTaskServer::new();
        let handler = RecordingElicitationHandler::new("resumed answer");

        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);
        tokio::spawn(async move {
            let running = server
                .serve((server_from_client, server_to_client))
                .await
                .expect("server failed to start");
            running.waiting().await.ok();
        });
        let tool_server_handle = ToolServer::new().run();
        let client = McpClientHandler::new(ClientInfo::default(), tool_server_handle.clone())
            .with_elicitation_handler(handler);
        let service = client
            .connect((client_from_server, client_to_server))
            .await
            .expect("connect failed");

        // Launch a task (goes input_required + elicits), then resume it by
        // descriptor through the handler-derived resumer: the resumed handle
        // must keep waiting through input_required and resolve.
        let dispatch = dispatch(&tool_server_handle).await;
        let ToolDispatch::Deferred(original) = dispatch else {
            panic!("required-task tool must defer");
        };
        let descriptor: ToolTaskDescriptor = original.descriptor();
        drop(original);

        let resumer =
            McpTaskResumer::new(service.peer().clone(), None).with_elicitation_available(true);
        let handle = resumer
            .resume(&descriptor)
            .await
            .expect("resume must not error")
            .expect("the mcp resumer must claim an mcp descriptor");

        let result = tokio::time::timeout(Duration::from_secs(5), handle.wait())
            .await
            .expect("resumed wait must survive input_required");
        assert!(result.result().is_success());
        assert_eq!(result.result().output().render(), "elicited:resumed answer");
    }
}
