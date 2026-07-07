//! First-class MCP tasks (SEP-1686) in the rig agent loop.
//!
//! An MCP server advertises a slow, task-capable tool (`taskSupport:
//! optional`). Rig's `McpClientHandler` registers it with the default
//! `Preferred` task policy, so the agent dispatches it task-augmented; under
//! the default `ContinueTurns` completion policy the model immediately
//! receives the server's `model-immediate-response` placeholder, keeps
//! working, and the real result is injected into a later turn as a labeled
//! "Background task update" notice the final answer incorporates.
//!
//! The transport is an in-process duplex pipe so the example is
//! self-contained; swap in a `StreamableHttpClientTransport` for a networked
//! server (see `examples/rmcp`). Requires `OPENAI_API_KEY`.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use rig::agent::{Flow, HookContext, StepEvent};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::CompletionModel;
use rig::providers::openai;
use rig::tool::rmcp::McpClientHandler;
use rig::tool::server::ToolServer;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, CancelTaskParams, CancelTaskResult, ContentBlock,
    CreateTaskResult, ErrorData, GetTaskParams, GetTaskPayloadParams, GetTaskPayloadResult,
    GetTaskResult, Implementation, ListToolsResult, Meta, PaginatedRequestParams, ProtocolVersion,
    ServerCapabilities, ServerInfo, ServerNotification, Task, TaskStatus, TaskStatusNotification,
    TaskStatusNotificationParam, TaskSupport, TasksCapability, Tool, ToolExecution,
};
use rmcp::service::{RequestContext, RoleServer};
use rmcp::{ServerHandler, ServiceExt};
use tokio::sync::{Notify, RwLock};

type TaskEntry = (Task, Option<CallToolResult>);

/// A deliberately slow "research" server: every task takes a few seconds and
/// pushes `notifications/tasks/status` so waiting clients wake early.
#[derive(Clone)]
struct DeepThoughtServer {
    state: Arc<RwLock<HashMap<String, TaskEntry>>>,
    terminal: Arc<Notify>,
    next_id: Arc<AtomicUsize>,
}

impl DeepThoughtServer {
    fn new() -> Self {
        Self {
            state: Arc::default(),
            terminal: Arc::default(),
            next_id: Arc::default(),
        }
    }

    fn tool() -> Tool {
        Tool::new(
            "deep_analysis".to_string(),
            "Runs a deep (slow) analysis of the given topic and returns the finding.".to_string(),
            Arc::new(
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "topic": { "type": "string", "description": "What to analyze" }
                    }
                })
                .as_object()
                .cloned()
                .unwrap_or_default(),
            ),
        )
        .with_execution(ToolExecution::from_raw(Some(TaskSupport::Optional)))
    }
}

impl ServerHandler for DeepThoughtServer {
    fn get_info(&self) -> ServerInfo {
        let mut capabilities = ServerCapabilities::builder().enable_tools().build();
        capabilities.tasks = Some(TasksCapability::server_default());
        ServerInfo::new(capabilities)
            .with_protocol_version(ProtocolVersion::LATEST)
            .with_server_info(Implementation::new("deep-thought", "1.0.0"))
            .with_instructions(
                "Provides deep_analysis, a slow tool best invoked as a background task.",
            )
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        (name == "deep_analysis").then(Self::tool)
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items(vec![Self::tool()]))
    }

    async fn call_tool(
        &self,
        _request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        // The plain (non-task) path, for clients that do not use tasks.
        Ok(CallToolResult::success(vec![ContentBlock::text(
            "Shallow analysis (no task): the answer is probably 42.",
        )]))
    }

    async fn enqueue_task(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CreateTaskResult, ErrorData> {
        let id = format!("analysis-{}", self.next_id.fetch_add(1, Ordering::SeqCst));
        let topic = request
            .arguments
            .as_ref()
            .and_then(|arguments| arguments.get("topic"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("everything")
            .to_string();
        let now = "2026-01-01T00:00:00Z".to_string();
        let mut task = Task::new(id.clone(), TaskStatus::Working, now.clone(), now)
            .with_status_message(format!("analyzing {topic}"))
            .with_poll_interval(500);
        if let Some(ttl) = request.task.as_ref().and_then(|t| t.ttl) {
            task = task.with_ttl(ttl);
        }
        self.state
            .write()
            .await
            .insert(id.clone(), (task.clone(), None));

        // The actual "work": a few seconds later the task completes, waking
        // both the long-polling tasks/result and — via the status
        // notification — any client poller.
        let state = self.state.clone();
        let terminal = self.terminal.clone();
        let peer = context.peer.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(3)).await;
            let finished = {
                let mut state = state.write().await;
                match state.get_mut(&id) {
                    Some((task, payload)) => {
                        task.status = TaskStatus::Completed;
                        *payload =
                            Some(CallToolResult::success(vec![ContentBlock::text(format!(
                                "Deep analysis of {topic} complete: the answer is 42 \
                                 (confidence 0.97)."
                            ))]));
                        Some(task.clone())
                    }
                    None => None,
                }
            };
            terminal.notify_waiters();
            if let Some(task) = finished {
                let notification = ServerNotification::TaskStatusNotification(
                    TaskStatusNotification::new(TaskStatusNotificationParam::new(task)),
                );
                if let Err(err) = peer.send_notification(notification).await {
                    tracing::warn!("failed to push a task status notification: {err}");
                }
            }
        });

        let mut meta = Meta::new();
        meta.0.insert(
            rig::tool::rmcp::MODEL_IMMEDIATE_RESPONSE_META_KEY.to_string(),
            serde_json::json!(
                "Deep analysis started in the background; keep working — the finding will be \
                 delivered when ready."
            ),
        );
        Ok(CreateTaskResult::new(task).with_meta(meta))
    }

    async fn get_task_info(
        &self,
        request: GetTaskParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<GetTaskResult, ErrorData> {
        match self.state.read().await.get(&request.task_id) {
            Some((task, _)) => Ok(GetTaskResult::new(task.clone())),
            None => Err(ErrorData::invalid_params("task not found", None)),
        }
    }

    async fn get_task_result(
        &self,
        request: GetTaskPayloadParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<GetTaskPayloadResult, ErrorData> {
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
                        return Ok(GetTaskPayloadResult::new(value));
                    }
                    Some(_) => {}
                    None => return Err(ErrorData::invalid_params("task not found", None)),
                }
            }
            notified.await;
        }
    }

    async fn cancel_task(
        &self,
        request: CancelTaskParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CancelTaskResult, ErrorData> {
        let mut state = self.state.write().await;
        match state.get_mut(&request.task_id) {
            Some((task, _)) => {
                task.status = TaskStatus::Cancelled;
                task.status_message = Some("cancelled by request".to_string());
                Ok(CancelTaskResult::new(task.clone()))
            }
            None => Err(ErrorData::invalid_params("task not found", None)),
        }
    }
}

/// Prints the task lifecycle as the loop observes it.
#[derive(Clone)]
struct TaskTracer;

impl<M: CompletionModel> rig::agent::AgentHook<M> for TaskTracer {
    async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
        match event {
            StepEvent::ToolTaskStarted {
                tool_name, task_id, ..
            } => println!("⏳ task {task_id} started for tool `{tool_name}`"),
            StepEvent::ToolTaskStatus {
                task_id, status, ..
            } => println!("   task {task_id} is {status}"),
            StepEvent::ToolTaskResult {
                task_id, result, ..
            } => println!("✅ task {task_id} resolved: {result}"),
            _ => {}
        }
        Flow::cont()
    }

    fn observes(&self, kind: rig::agent::StepEventKind) -> bool {
        matches!(
            kind,
            rig::agent::StepEventKind::ToolTaskStarted
                | rig::agent::StepEventKind::ToolTaskStatus
                | rig::agent::StepEventKind::ToolTaskResult
        )
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    // Serve the MCP server over an in-process duplex transport.
    let (client_to_server, server_from_client) = tokio::io::duplex(8192);
    let (server_to_client, client_from_server) = tokio::io::duplex(8192);
    tokio::spawn(async move {
        match DeepThoughtServer::new()
            .serve((server_from_client, server_to_client))
            .await
        {
            Ok(running) => {
                running.waiting().await.ok();
            }
            Err(err) => tracing::error!("MCP server failed to start: {err}"),
        }
    });

    // Register the server's tools; the handler's default `Preferred` task
    // policy makes optional-task tools dispatch task-augmented, and its
    // notification registry wakes waiting handles on status pushes.
    let tool_server_handle = ToolServer::new().run();
    let handler = McpClientHandler::new(
        rmcp::model::ClientInfo::default(),
        tool_server_handle.clone(),
    )
    .with_task_ttl(Duration::from_secs(120));
    let _mcp_service = handler
        .connect((client_from_server, client_to_server))
        .await?;

    let openai_client = openai::Client::from_env()?;
    let response = openai_client
        .agent(openai::GPT_4O)
        .preamble(
            "You are a research assistant. deep_analysis runs in the background: when it is \
             started you receive a placeholder result — keep reasoning about what else the \
             user needs, and incorporate the background finding when it is delivered.",
        )
        .tool_server_handle(tool_server_handle)
        .build()
        .runner("Run a deep analysis of the meaning of life, then summarize the finding in one sentence.")
        .max_turns(6)
        .task_deadline(Duration::from_secs(30))
        .add_hook(TaskTracer)
        .run()
        .await?;

    println!("\n🤖 {}", response.output);
    Ok(())
}
