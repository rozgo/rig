//! Driver-side engine for deferred tool tasks.
//!
//! [`TaskTable`] is the run-scoped working set of live
//! [`ToolTaskHandle`](crate::tool::ToolTaskHandle)s the shared engine
//! (`drive_agent`) owns: launched handles enter through
//! [`TaskTable::launch`], each wrapped in a [`drive_task`] stream that polls
//! the backend at its hint cadence, enforces the runner's per-task deadline,
//! honors cooperative cancellation, and emits [`TaskDriveEvent`]s. Everything
//! is caller-driven futures — no background spawns — so it compiles on the
//! wasm targets rig supports.

use std::collections::HashMap;
use std::time::Duration;

use futures::StreamExt;
use futures::stream::SelectAll;

use crate::agent::run::{PendingTask, TaskCompletionPolicy};
use crate::tool::{ToolExecutionResult, ToolFailure, ToolTaskHandle, ToolTaskStatus};
use crate::wasm_compat::WasmCompatSend;

/// Fallback status-poll cadence when neither the handle nor the runner
/// supplies one.
pub(crate) const DEFAULT_TASK_POLL_INTERVAL: Duration = Duration::from_secs(2);

/// Bound on draining the wait set during a graceful cancel, so a backend that
/// ignores cancellation cannot wedge the run's exit path.
pub(crate) const TASK_CANCEL_GRACE: Duration = Duration::from_secs(5);

/// A future that resolves after `duration` (or never, when `None`) — the
/// portable timer used by this engine (no `tokio::time` in library code).
async fn sleep_or_forever(duration: Option<Duration>) {
    match duration {
        Some(duration) => {
            let _ = crate::wasm_compat::timeout(duration, std::future::pending::<()>()).await;
        }
        None => std::future::pending::<()>().await,
    }
}

/// One event surfaced by a task being driven. Carries the launching call's
/// identifiers so the engine can fire hooks and feed the machine without a
/// lookup.
pub(crate) enum TaskDriveEvent {
    /// The task's status changed (emitted on change only).
    Status {
        internal_call_id: String,
        tool_name: String,
        /// The provider-supplied call id, when available.
        call_id: Option<String>,
        task_id: String,
        status: ToolTaskStatus,
    },
    /// Terminal: the raw execution result (the task-result hook has NOT run).
    Resolved {
        internal_call_id: String,
        tool_name: String,
        /// The launching tool_use id (the tool call's `id`).
        tool_use_id: String,
        /// The provider-supplied call id, when available.
        call_id: Option<String>,
        task_id: String,
        policy: TaskCompletionPolicy,
        /// The raw structured result.
        result: ToolExecutionResult,
        /// The terminal status the task ended in.
        status: ToolTaskStatus,
        /// Whether the model-visible output is synthetic (cancellation reason
        /// or deadline text) rather than real tool output.
        synthetic: bool,
    },
}

/// What one `drive_task` iteration observed.
enum TaskWake {
    Status(Result<ToolTaskStatus, ToolFailure>),
    Cancelled(String),
    Deadline,
}

#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
type TaskEventStream = std::pin::Pin<Box<dyn futures::Stream<Item = TaskDriveEvent> + Send>>;
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
type TaskEventStream = std::pin::Pin<Box<dyn futures::Stream<Item = TaskDriveEvent>>>;

/// Drive one deferred task to a terminal [`TaskDriveEvent::Resolved`].
///
/// The loop polls [`status`](ToolTaskHandle::status) at the handle's
/// [`poll_hint`](ToolTaskHandle::poll_hint) (falling back to
/// `poll_fallback`, then [`DEFAULT_TASK_POLL_INTERVAL`]), emitting a
/// [`Status`](TaskDriveEvent::Status) on change; on a terminal status it
/// consumes the handle via [`wait`](ToolTaskHandle::wait) (which resolves
/// promptly for a terminal task) to fetch the result. A cancel signal or the
/// deadline elapsing best-effort cancels the backend task and resolves with a
/// classified synthetic failure. Transient status-poll errors are retried at
/// the poll cadence; a not-found error (expired task) resolves immediately.
fn drive_task(
    handle: Box<dyn ToolTaskHandle>,
    pending: &PendingTask,
    poll_fallback: Option<Duration>,
    deadline: Option<Duration>,
    mut cancel_rx: tokio::sync::watch::Receiver<Option<String>>,
) -> impl futures::Stream<Item = TaskDriveEvent> + WasmCompatSend + 'static {
    let internal_call_id = pending.internal_call_id.clone();
    let tool_name = pending.tool_call.function.name.clone();
    let tool_use_id = pending.tool_call.id.clone();
    let call_id = pending.tool_call.call_id.clone();
    let task_id = pending.descriptor.task_id.clone();
    let policy = pending.policy;
    let mut last_status = pending.last_status;

    async_stream::stream! {
        use futures::FutureExt;

        let poll_interval = handle
            .poll_hint()
            .or(poll_fallback)
            .unwrap_or(DEFAULT_TASK_POLL_INTERVAL);
        // Armed once for the task's whole lifetime and fused so it stays
        // safely pollable after firing; a `deadline_hit` flag carries the
        // elapse across race sites to the classifying check at the loop top.
        let deadline_fut = sleep_or_forever(deadline).fuse();
        futures::pin_mut!(deadline_fut);
        let mut deadline_hit = false;
        let mut handle_slot = Some(handle);

        // Convenience for the terminal paths below.
        macro_rules! resolved {
            ($result:expr, $status:expr, $synthetic:expr) => {
                TaskDriveEvent::Resolved {
                    internal_call_id: internal_call_id.clone(),
                    tool_name: tool_name.clone(),
                    tool_use_id: tool_use_id.clone(),
                    call_id: call_id.clone(),
                    task_id: task_id.clone(),
                    policy,
                    result: $result,
                    status: $status,
                    synthetic: $synthetic,
                }
            };
        }

        loop {
            let Some(handle_ref) = handle_slot.as_deref() else {
                // Unreachable in practice: every terminal branch returns.
                return;
            };

            // Classify signals observed during the previous poll-interval wait
            // before issuing another status RPC.
            let wake = if deadline_hit {
                TaskWake::Deadline
            } else if let Some(reason) = cancel_rx.borrow_and_update().clone() {
                TaskWake::Cancelled(reason)
            } else {
                // Race the status poll against cancellation and the deadline.
                use futures::future::{Either, select};
                let status_fut = handle_ref.status();
                futures::pin_mut!(status_fut);
                let cancel_fut = async {
                    loop {
                        if let Some(reason) = cancel_rx.borrow_and_update().clone() {
                            return reason;
                        }
                        if cancel_rx.changed().await.is_err() {
                            // Sender dropped: cancellation can never arrive.
                            std::future::pending::<()>().await;
                        }
                    }
                };
                futures::pin_mut!(cancel_fut);
                match select(status_fut, select(cancel_fut, deadline_fut.as_mut())).await {
                    Either::Left((status, _)) => TaskWake::Status(status),
                    Either::Right((Either::Left((reason, _)), _)) => TaskWake::Cancelled(reason),
                    Either::Right((Either::Right(_), _)) => TaskWake::Deadline,
                }
            };

            match wake {
                TaskWake::Status(Ok(status)) if status.is_terminal() => {
                    let Some(handle) = handle_slot.take() else { return };
                    let result = handle.wait().await;
                    yield resolved!(result, status, false);
                    return;
                }
                TaskWake::Status(Ok(status)) => {
                    if last_status != Some(status) {
                        last_status = Some(status);
                        yield TaskDriveEvent::Status {
                            internal_call_id: internal_call_id.clone(),
                            tool_name: tool_name.clone(),
                            call_id: call_id.clone(),
                            task_id: task_id.clone(),
                            status,
                        };
                    }
                }
                TaskWake::Status(Err(failure))
                    if failure.kind == crate::tool::ToolFailureKind::NotFound =>
                {
                    // The task expired or was purged: unrecoverable.
                    let message = failure.message.clone();
                    yield resolved!(
                        ToolExecutionResult::failed(message, failure),
                        ToolTaskStatus::Failed,
                        false
                    );
                    return;
                }
                TaskWake::Status(Err(failure)) => {
                    // Transient (network/timeout/...): keep polling; the
                    // deadline bounds a permanently failing poll.
                    tracing::warn!(
                        task_id = %task_id,
                        error = %failure.message,
                        "transient deferred-task status poll failure"
                    );
                }
                TaskWake::Cancelled(reason) => {
                    if let Some(handle) = handle_slot.take()
                        && let Err(failure) = handle.cancel().await
                    {
                        tracing::warn!(
                            task_id = %task_id,
                            error = %failure.message,
                            "failed to cancel a deferred task"
                        );
                    }
                    yield resolved!(
                        ToolExecutionResult::failed(
                            reason.clone(),
                            ToolFailure::cancelled(reason)
                        ),
                        ToolTaskStatus::Cancelled,
                        true
                    );
                    return;
                }
                TaskWake::Deadline => {
                    let message = format!(
                        "deferred task `{task_id}` for tool `{tool_name}` exceeded its deadline \
                         and was cancelled"
                    );
                    if let Some(handle) = handle_slot.take()
                        && let Err(failure) = handle.cancel().await
                    {
                        tracing::warn!(
                            task_id = %task_id,
                            error = %failure.message,
                            "failed to cancel a deferred task at its deadline"
                        );
                    }
                    yield resolved!(
                        ToolExecutionResult::failed(
                            message.clone(),
                            ToolFailure::timeout(message)
                        ),
                        ToolTaskStatus::Cancelled,
                        true
                    );
                    return;
                }
            }

            // Wait out the poll interval, still responsive to cancel/deadline
            // (either observed here is classified at the top of the loop).
            {
                use futures::future::{Either, select};
                let tick = sleep_or_forever(Some(poll_interval));
                futures::pin_mut!(tick);
                let cancel_changed = cancel_rx.changed();
                futures::pin_mut!(cancel_changed);
                if let Either::Right((Either::Right(_), _)) =
                    select(tick, select(cancel_changed, deadline_fut.as_mut())).await
                {
                    deadline_hit = true;
                }
            }
        }
    }
}

/// Per-task launch configuration resolved from the runner.
#[derive(Clone, Copy, Default)]
pub(crate) struct TaskDriveConfig {
    /// Fallback status-poll cadence.
    pub poll_fallback: Option<Duration>,
    /// Per-task wall-clock deadline from launch.
    pub deadline: Option<Duration>,
}

/// The run-scoped working set of live deferred tasks, owned by the shared
/// engine. Never serialized: on resume it is rebuilt from the run's persisted
/// descriptors via the registered
/// [`TaskResumer`](crate::tool::TaskResumer)s.
pub(crate) struct TaskTable {
    /// The live task streams, polled together.
    wait_set: SelectAll<TaskEventStream>,
    /// Cooperative-cancel triggers keyed by `internal_call_id`. An entry also
    /// marks the task as having a live driver (the resume check).
    cancels: HashMap<String, tokio::sync::watch::Sender<Option<String>>>,
    /// Per-task `tool_task` telemetry spans, recorded at resolution.
    spans: HashMap<String, tracing::Span>,
}

impl TaskTable {
    pub(crate) fn new() -> Self {
        Self {
            wait_set: SelectAll::new(),
            cancels: HashMap::new(),
            spans: HashMap::new(),
        }
    }

    /// Whether any task is still being driven.
    pub(crate) fn is_empty(&self) -> bool {
        self.wait_set.is_empty()
    }

    /// Whether a live driver exists for `internal_call_id`.
    pub(crate) fn is_live(&self, internal_call_id: &str) -> bool {
        self.cancels.contains_key(internal_call_id)
    }

    /// Start driving a deferred task: wire its cancel trigger, build its
    /// telemetry span (linked after the launching `execute_tool` span), and
    /// add its [`drive_task`] stream to the wait set.
    pub(crate) fn launch(
        &mut self,
        handle: Box<dyn ToolTaskHandle>,
        pending: &PendingTask,
        config: TaskDriveConfig,
        launching_span: &tracing::Span,
    ) {
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(None);
        let span = new_tool_task_span(pending);
        span.follows_from(launching_span.id());
        let stream = drive_task(
            handle,
            pending,
            config.poll_fallback,
            config.deadline,
            cancel_rx,
        );
        let stream: TaskEventStream = Box::pin(tracing_futures::Instrument::instrument(
            stream,
            span.clone(),
        ));
        self.wait_set.push(stream);
        self.cancels
            .insert(pending.internal_call_id.clone(), cancel_tx);
        self.spans.insert(pending.internal_call_id.clone(), span);
    }

    /// The next event from any live task; `None` when no task is live.
    pub(crate) async fn next_event(&mut self) -> Option<TaskDriveEvent> {
        if self.wait_set.is_empty() {
            return None;
        }
        self.wait_set.next().await
    }

    /// Request cancellation of one task; its driver resolves it as a
    /// cancelled result carrying `reason`.
    pub(crate) fn cancel_one(&self, internal_call_id: &str, reason: impl Into<String>) {
        if let Some(sender) = self.cancels.get(internal_call_id) {
            sender.send_replace(Some(reason.into()));
        }
    }

    /// Release a task's driver bookkeeping once it resolved; returns its
    /// telemetry span for post-hook result recording.
    pub(crate) fn finish(&mut self, internal_call_id: &str) -> Option<tracing::Span> {
        self.cancels.remove(internal_call_id);
        self.spans.remove(internal_call_id)
    }

    /// Detach every live task: drop the drivers (abandoning the tasks
    /// locally; the backends keep running them) without requesting
    /// cancellation. Used by
    /// [`TaskDrainPolicy::Detach`](crate::agent::run::TaskDrainPolicy::Detach).
    pub(crate) fn detach(&mut self) {
        self.wait_set = SelectAll::new();
        self.cancels.clear();
        self.spans.clear();
    }

    /// Best-effort cancel every live task and drain the wait set, bounded by
    /// [`TASK_CANCEL_GRACE`] so an unresponsive backend cannot wedge the exit
    /// path. Dropped (undrained) drivers abandon their tasks locally.
    pub(crate) async fn graceful_cancel(&mut self, reason: &str) {
        if self.wait_set.is_empty() {
            return;
        }
        for sender in self.cancels.values() {
            sender.send_replace(Some(reason.to_string()));
        }
        let drain = async { while self.wait_set.next().await.is_some() {} };
        if crate::wasm_compat::timeout(TASK_CANCEL_GRACE, drain)
            .await
            .is_err()
        {
            tracing::warn!(
                "deferred tasks did not finish cancelling within {TASK_CANCEL_GRACE:?}; \
                 abandoning them locally"
            );
        }
        self.cancels.clear();
        self.spans.clear();
    }
}

/// Build the per-task `tool_task` telemetry span, linked (`follows_from`) to
/// the launching `execute_tool` span. The terminal result/outcome are recorded
/// only after the `ToolTaskResult` hook runs — the same redaction discipline
/// as the inline tool path.
fn new_tool_task_span(pending: &PendingTask) -> tracing::Span {
    tracing::info_span!(
        "tool_task",
        gen_ai.operation.name = "tool_task",
        gen_ai.tool.type = "function",
        gen_ai.tool.name = %pending.tool_call.function.name,
        gen_ai.tool.call.id = %pending.tool_call.id,
        gen_ai.tool.task.id = %pending.descriptor.task_id,
        gen_ai.tool.task.status = tracing::field::Empty,
        gen_ai.tool.call.result = tracing::field::Empty,
        gen_ai.tool.call.outcome = tracing::field::Empty,
        gen_ai.tool.error.type = tracing::field::Empty,
    )
}
