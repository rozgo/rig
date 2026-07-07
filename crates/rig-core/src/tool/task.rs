//! Deferred tool execution: the task boundary.
//!
//! A tool dispatched through [`ToolDyn::dispatch_structured`](super::ToolDyn::dispatch_structured)
//! resolves to a [`ToolDispatch`]: either an already-final
//! [`ToolExecutionResult`] (every ordinary tool), or a
//! [`ToolTaskHandle`] — a live handle to work the backend accepted but has not
//! finished (e.g. an MCP task, SEP-1686). The caller — typically the agent
//! loop — owns the deferred lifecycle: it may poll [`ToolTaskHandle::status`]
//! at the [`ToolTaskHandle::poll_hint`] cadence, drive the task to completion
//! with [`ToolTaskHandle::wait`], or stop it with [`ToolTaskHandle::cancel`].
//!
//! # Local abandonment vs. cancellation
//!
//! Dropping a [`ToolTaskHandle`] (or the future returned by
//! [`wait`](ToolTaskHandle::wait)) abandons the task **locally only** — the
//! backend keeps running it. [`cancel`](ToolTaskHandle::cancel) is the real,
//! best-effort cancellation.
//!
//! # Durability
//!
//! [`ToolTaskHandle::descriptor`] returns a serializable
//! [`ToolTaskDescriptor`] that outlives the handle: a suspended agent run
//! persists descriptors and rehydrates live handles from them on resume (see
//! [`TaskResumer`](crate::agent::TaskResumer)).

use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::result::{ToolExecutionResult, ToolFailure};
use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};

/// Lifecycle status of a deferred tool task (rig-native mirror of the MCP
/// 2025-11-25 `TaskStatus`, SEP-1686).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolTaskStatus {
    /// The backend accepted the call and is working on it.
    Working,
    /// The backend requires additional input before work can continue.
    InputRequired,
    /// The task completed; the result payload is retrievable.
    Completed,
    /// The task failed and will not continue.
    Failed,
    /// The task was cancelled and will not continue.
    Cancelled,
}

impl ToolTaskStatus {
    /// Stable identifier for tracing and metrics (`"working"`,
    /// `"input_required"`, `"completed"`, `"failed"`, `"cancelled"`).
    pub const fn as_str(self) -> &'static str {
        match self {
            ToolTaskStatus::Working => "working",
            ToolTaskStatus::InputRequired => "input_required",
            ToolTaskStatus::Completed => "completed",
            ToolTaskStatus::Failed => "failed",
            ToolTaskStatus::Cancelled => "cancelled",
        }
    }

    /// Whether this status is terminal: the task will never change state again.
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            ToolTaskStatus::Completed | ToolTaskStatus::Failed | ToolTaskStatus::Cancelled
        )
    }
}

impl std::fmt::Display for ToolTaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Serializable descriptor of a deferred tool task, for durable agent runs.
///
/// Persist this (serde) when a run is suspended; on resume, route it to the
/// backend named by [`backend`](Self::backend) (e.g.
/// [`BACKEND_MCP`](Self::BACKEND_MCP)) to rehydrate a live
/// [`ToolTaskHandle`]. Timestamps are the backend's own ISO-8601 strings,
/// carried verbatim — a resume layer computes deadlines from its own clock,
/// captured when the dispatch happened.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolTaskDescriptor {
    /// Backend discriminator for resume routing (e.g. [`Self::BACKEND_MCP`]).
    pub backend: String,
    /// Backend-assigned task id (`Task.taskId` for MCP).
    pub task_id: String,
    /// The rig tool name the call was dispatched to.
    pub tool_name: String,
    /// Opaque identity of the backing server (for MCP: `"{name}@{version}"`
    /// from the negotiated server info), used to find the right connection on
    /// resume.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_key: Option<String>,
    /// Backend-reported ISO-8601 creation timestamp, verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    /// Retention window in milliseconds the backend agreed to honor (`None` =
    /// unlimited). After it elapses the task id resolves as not found.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_ms: Option<u64>,
    /// Backend-suggested polling interval in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub poll_interval_ms: Option<u64>,
    /// The backend's model-facing immediate-response hint, if any, so a
    /// resumed run can replay it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub immediate_response: Option<String>,
}

impl ToolTaskDescriptor {
    /// Backend discriminator used by MCP task handles.
    pub const BACKEND_MCP: &'static str = "mcp";

    /// Create a descriptor with the required routing fields; the optional
    /// fields default to `None` and are set by struct update or the backend.
    pub fn new(
        backend: impl Into<String>,
        task_id: impl Into<String>,
        tool_name: impl Into<String>,
    ) -> Self {
        Self {
            backend: backend.into(),
            task_id: task_id.into(),
            tool_name: tool_name.into(),
            server_key: None,
            created_at: None,
            ttl_ms: None,
            poll_interval_ms: None,
            immediate_response: None,
        }
    }
}

/// A live handle to a deferred tool execution.
///
/// # Contract
///
/// - [`wait`](Self::wait) consumes the handle and always resolves to a
///   [`ToolExecutionResult`]: failures are classified into the result's
///   [`ToolOutcome`](crate::tool::ToolOutcome), never returned as a bare
///   error — the same contract as
///   [`ToolDyn::call_structured`](super::ToolDyn::call_structured).
/// - Dropping the handle, or the future returned by [`wait`](Self::wait),
///   abandons the task **locally**; the backend keeps running it. Call
///   [`cancel`](Self::cancel) to stop it.
/// - [`status`](Self::status)/[`cancel`](Self::cancel) errors are
///   transport/protocol problems, classified as [`ToolFailure`] so policies
///   can act on the kind.
pub trait ToolTaskHandle: WasmCompatSend + WasmCompatSync + 'static {
    /// The backend-assigned task id.
    fn task_id(&self) -> &str;

    /// Fetch the task's current lifecycle status from the backend.
    fn status(&self) -> WasmBoxedFuture<'_, Result<ToolTaskStatus, ToolFailure>>;

    /// Drive the task to a terminal state and produce the final structured
    /// result. See the trait docs for the never-`Err` contract.
    fn wait(self: Box<Self>) -> WasmBoxedFuture<'static, ToolExecutionResult>;

    /// Request cancellation. Best-effort: a task already in a terminal state
    /// stays terminal.
    fn cancel(&self) -> WasmBoxedFuture<'_, Result<(), ToolFailure>>;

    /// Suggested delay between [`status`](Self::status) polls, if the backend
    /// provided one.
    fn poll_hint(&self) -> Option<Duration>;

    /// Serializable descriptor for durable runs (see the module docs).
    fn descriptor(&self) -> ToolTaskDescriptor;

    /// The backend's model-facing immediate-response hint (MCP `_meta` key
    /// `io.modelcontextprotocol/model-immediate-response`), if any.
    fn immediate_response(&self) -> Option<&str>;
}

/// Rehydrates a live [`ToolTaskHandle`] for a suspended deferred task when a
/// serialized agent run is resumed.
///
/// Register implementations on the agent runner; each is consulted in
/// registration order for every persisted [`ToolTaskDescriptor`] that has no
/// live handle. The MCP backend provides
/// [`McpTaskResumer`](crate::tool::rmcp::McpTaskResumer).
pub trait TaskResumer: WasmCompatSend + WasmCompatSync {
    /// Attempt to resume `descriptor`. `Ok(None)` means "not mine" (e.g. a
    /// different backend or server key) and the next registered resumer is
    /// consulted; `Err` is a definitive failure for this descriptor.
    fn resume<'a>(
        &'a self,
        descriptor: &'a ToolTaskDescriptor,
    ) -> WasmBoxedFuture<'a, Result<Option<Box<dyn ToolTaskHandle>>, super::ToolError>>;
}

/// The outcome of dispatching a tool call: already complete, or deferred
/// behind a task handle the caller must drive.
#[non_exhaustive]
pub enum ToolDispatch {
    /// The tool ran synchronously; the result is final.
    Completed(ToolExecutionResult),
    /// The backend accepted the call as a task; drive it via the handle.
    Deferred(Box<dyn ToolTaskHandle>),
}

impl ToolDispatch {
    /// Resolve to a final result: `Completed` passes through, `Deferred` is
    /// awaited via [`ToolTaskHandle::wait`]. Convenience for callers that do
    /// not implement a task-aware loop.
    pub async fn resolve(self) -> ToolExecutionResult {
        match self {
            ToolDispatch::Completed(result) => result,
            ToolDispatch::Deferred(handle) => handle.wait().await,
        }
    }
}

impl std::fmt::Debug for ToolDispatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolDispatch::Completed(result) => f.debug_tuple("Completed").field(result).finish(),
            ToolDispatch::Deferred(handle) => f
                .debug_struct("Deferred")
                .field("task_id", &handle.task_id())
                .finish_non_exhaustive(),
        }
    }
}

// A dispatch (and the handle inside it) crosses `.await` points in the agent
// loop and is the output of the `WasmBoxedFuture` returned by
// `ToolDyn::dispatch_structured`, so on native targets it must stay
// `Send + Sync`. This fails to compile if a future change drops the property.
#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ToolDispatch>();
    assert_send_sync::<ToolTaskDescriptor>();
    assert_send_sync::<ToolTaskStatus>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_terminality_and_names() {
        let table = [
            (ToolTaskStatus::Working, "working", false),
            (ToolTaskStatus::InputRequired, "input_required", false),
            (ToolTaskStatus::Completed, "completed", true),
            (ToolTaskStatus::Failed, "failed", true),
            (ToolTaskStatus::Cancelled, "cancelled", true),
        ];
        for (status, name, terminal) in table {
            assert_eq!(status.as_str(), name);
            assert_eq!(status.to_string(), name);
            assert_eq!(status.is_terminal(), terminal, "terminality of {name}");
        }
    }

    #[test]
    fn descriptor_serde_round_trip_omits_absent_optionals() {
        let descriptor = ToolTaskDescriptor::new(ToolTaskDescriptor::BACKEND_MCP, "t-1", "search");
        let json = serde_json::to_value(&descriptor).expect("descriptor serializes");
        assert_eq!(
            json,
            serde_json::json!({"backend": "mcp", "task_id": "t-1", "tool_name": "search"})
        );

        let full = ToolTaskDescriptor {
            server_key: Some("srv@1.0.0".into()),
            created_at: Some("2026-07-07T00:00:00Z".into()),
            ttl_ms: Some(60_000),
            poll_interval_ms: Some(2_000),
            immediate_response: Some("started".into()),
            ..descriptor
        };
        let round_tripped: ToolTaskDescriptor =
            serde_json::from_value(serde_json::to_value(&full).expect("serializes"))
                .expect("deserializes");
        assert_eq!(round_tripped, full);
    }

    #[test]
    fn status_serde_uses_snake_case() {
        let json = serde_json::to_value(ToolTaskStatus::InputRequired).expect("serializes");
        assert_eq!(json, serde_json::json!("input_required"));
        let status: ToolTaskStatus =
            serde_json::from_value(serde_json::json!("cancelled")).expect("deserializes");
        assert_eq!(status, ToolTaskStatus::Cancelled);
    }

    struct StubHandle {
        output: String,
    }

    impl ToolTaskHandle for StubHandle {
        fn task_id(&self) -> &str {
            "stub-task"
        }

        fn status(&self) -> WasmBoxedFuture<'_, Result<ToolTaskStatus, ToolFailure>> {
            Box::pin(async { Ok(ToolTaskStatus::Completed) })
        }

        fn wait(self: Box<Self>) -> WasmBoxedFuture<'static, ToolExecutionResult> {
            Box::pin(async move { ToolExecutionResult::success(self.output) })
        }

        fn cancel(&self) -> WasmBoxedFuture<'_, Result<(), ToolFailure>> {
            Box::pin(async { Ok(()) })
        }

        fn poll_hint(&self) -> Option<Duration> {
            Some(Duration::from_millis(50))
        }

        fn descriptor(&self) -> ToolTaskDescriptor {
            ToolTaskDescriptor::new("stub", "stub-task", "stub_tool")
        }

        fn immediate_response(&self) -> Option<&str> {
            None
        }
    }

    #[tokio::test]
    async fn dispatch_resolve_awaits_deferred_handles() {
        let completed = ToolDispatch::Completed(ToolExecutionResult::success("done"));
        assert_eq!(completed.resolve().await.model_output(), "done");

        let deferred = ToolDispatch::Deferred(Box::new(StubHandle {
            output: "task output".into(),
        }));
        assert!(format!("{deferred:?}").contains("stub-task"));
        assert_eq!(deferred.resolve().await.model_output(), "task output");
    }
}
