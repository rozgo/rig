//! MCP elicitation (SEP-1686): a background task that asks the human.
//!
//! The MCP server's `plan_trip` tool runs as a background task that pauses in
//! `input_required` and elicits the traveler's budget from the client. The
//! Rig side registers a stdin [`McpElicitationHandler`] — **fail-closed**:
//! EOF or an empty line declines, mirroring
//! `examples/agent_with_human_in_the_loop` — so the agent's answer
//! incorporates a value only a human could provide, collected mid-task.
//!
//! Requires `OPENAI_API_KEY`. Type a budget (e.g. `2000 usd`) when prompted.

use std::{
    collections::HashMap,
    io::{Write, stdin, stdout},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use rig::client::{CompletionClient, ProviderClient};
use rig::providers::openai;
use rig::tool::rmcp::{McpClientHandler, McpElicitationHandler, related_task_id};
use rig::tool::server::ToolServer;
use rig::wasm_compat::WasmBoxedFuture;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ClientInfo, ContentBlock, CreateTaskResult,
    ElicitRequestParams, ElicitResult, ElicitationAction, ElicitationSchema, ErrorData,
    GetTaskParams, GetTaskPayloadParams, GetTaskPayloadResult, GetTaskResult, Implementation,
    ListToolsResult, Meta, PaginatedRequestParams, ProtocolVersion, RelatedTaskMetadata,
    RequestParamsMeta, ServerCapabilities, ServerInfo, Task, TaskStatus, TaskSupport,
    TasksCapability, Tool, ToolExecution,
};
use rmcp::service::{RequestContext, RoleServer};
use rmcp::{ServerHandler, ServiceExt};
use tokio::sync::{Notify, RwLock};

type TaskEntry = (Task, Option<CallToolResult>);

/// A travel-planning server whose task needs the traveler's budget: the task
/// starts in `input_required`, elicits the budget (correlated to the task via
/// `io.modelcontextprotocol/related-task`), and completes once answered.
#[derive(Clone)]
struct TravelPlannerServer {
    state: Arc<RwLock<HashMap<String, TaskEntry>>>,
    terminal: Arc<Notify>,
    next_id: Arc<AtomicUsize>,
}

impl TravelPlannerServer {
    fn new() -> Self {
        Self {
            state: Arc::default(),
            terminal: Arc::default(),
            next_id: Arc::default(),
        }
    }

    fn tool() -> Tool {
        Tool::new(
            "plan_trip".to_string(),
            "Plans a trip to the given destination; asks the traveler for their budget."
                .to_string(),
            Arc::new(
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "destination": { "type": "string", "description": "Where to go" }
                    }
                })
                .as_object()
                .cloned()
                .unwrap_or_default(),
            ),
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

impl ServerHandler for TravelPlannerServer {
    fn get_info(&self) -> ServerInfo {
        let mut capabilities = ServerCapabilities::builder().enable_tools().build();
        capabilities.tasks = Some(TasksCapability::server_default());
        ServerInfo::new(capabilities)
            .with_protocol_version(ProtocolVersion::LATEST)
            .with_server_info(Implementation::new("travel-planner", "1.0.0"))
            .with_instructions("plan_trip runs as a background task and elicits the budget.")
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        (name == "plan_trip").then(Self::tool)
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
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CreateTaskResult, ErrorData> {
        let id = format!("trip-{}", self.next_id.fetch_add(1, Ordering::SeqCst));
        let destination = request
            .arguments
            .as_ref()
            .and_then(|arguments| arguments.get("destination"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("somewhere nice")
            .to_string();
        let now = "2026-01-01T00:00:00Z".to_string();
        let task = Task::new(id.clone(), TaskStatus::InputRequired, now.clone(), now)
            .with_status_message(format!("waiting for the {destination} budget"))
            .with_poll_interval(250);
        self.state
            .write()
            .await
            .insert(id.clone(), (task.clone(), None));

        // Elicit out of band (the enqueue response must go out first).
        let server = self.clone();
        let peer = context.peer.clone();
        tokio::spawn(async move {
            let mut meta = Meta::new();
            meta.0.insert(
                RelatedTaskMetadata::META_KEY.to_string(),
                serde_json::json!({ "taskId": id }),
            );
            let schema = match ElicitationSchema::builder()
                .required_string("budget")
                .build()
            {
                Ok(schema) => schema,
                Err(err) => {
                    tracing::error!("elicitation schema failed to build: {err}");
                    return;
                }
            };
            let params = ElicitRequestParams::FormElicitationParams {
                meta: Some(meta),
                message: format!("What is your budget for the {destination} trip?"),
                requested_schema: schema,
            };
            match peer.create_elicitation(params).await {
                Ok(result) if result.action == ElicitationAction::Accept => {
                    let budget = result
                        .content
                        .as_ref()
                        .and_then(|content| content.get("budget"))
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("<unspecified>")
                        .to_string();
                    server
                        .settle(
                            &id,
                            TaskStatus::Completed,
                            Some(CallToolResult::success(vec![ContentBlock::text(format!(
                                "Trip to {destination} planned within a {budget} budget: \
                                 3 nights, off-season flights, one splurge dinner."
                            ))])),
                        )
                        .await;
                }
                Ok(_) => {
                    server
                        .settle(
                            &id,
                            TaskStatus::Failed,
                            Some(CallToolResult::error(vec![ContentBlock::text(
                                "the traveler declined to provide a budget",
                            )])),
                        )
                        .await;
                }
                Err(err) => {
                    tracing::error!("elicitation request failed: {err}");
                }
            }
        });

        Ok(CreateTaskResult::new(task))
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
}

/// A fail-closed stdin elicitation handler: prints the server's question,
/// reads one line off the async runtime, and declines on EOF or empty input
/// (mirroring `examples/agent_with_human_in_the_loop` — no input never means
/// approval).
struct StdinElicitationHandler;

impl McpElicitationHandler for StdinElicitationHandler {
    fn elicit(
        &self,
        request: ElicitRequestParams,
    ) -> WasmBoxedFuture<'_, Result<ElicitResult, ErrorData>> {
        Box::pin(async move {
            let ElicitRequestParams::FormElicitationParams { message, .. } = &request else {
                // URL-mode is not declared in our capability; decline defensively.
                return Ok(ElicitResult::new(ElicitationAction::Decline));
            };
            if let Some(task_id) = related_task_id(request.meta()) {
                println!("\n🛎  task {task_id} needs input");
            }
            println!("❓ {message}");
            print!("> ");
            let _ = stdout().flush();

            let line = tokio::task::spawn_blocking(|| {
                let mut line = String::new();
                match stdin().read_line(&mut line) {
                    Ok(0) | Err(_) => None,
                    Ok(_) => Some(line.trim().to_string()),
                }
            })
            .await
            .ok()
            .flatten();

            match line {
                Some(budget) if !budget.is_empty() => {
                    Ok(ElicitResult::new(ElicitationAction::Accept)
                        .with_content(serde_json::json!({ "budget": budget })))
                }
                // Fail-closed: no input is a decline, never an answer.
                _ => Ok(ElicitResult::new(ElicitationAction::Decline)),
            }
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();

    // Serve the MCP server over an in-process duplex transport.
    let (client_to_server, server_from_client) = tokio::io::duplex(8192);
    let (server_to_client, client_from_server) = tokio::io::duplex(8192);
    tokio::spawn(async move {
        match TravelPlannerServer::new()
            .serve((server_from_client, server_to_client))
            .await
        {
            Ok(running) => {
                running.waiting().await.ok();
            }
            Err(err) => tracing::error!("MCP server failed to start: {err}"),
        }
    });

    // The registered elicitation handler is advertised in the handshake and
    // lets the background task's input_required phase resolve interactively.
    let tool_server_handle = ToolServer::new().run();
    let handler = McpClientHandler::new(ClientInfo::default(), tool_server_handle.clone())
        .with_elicitation_handler(StdinElicitationHandler);
    let _mcp_service = handler
        .connect((client_from_server, client_to_server))
        .await?;

    let openai_client = openai::Client::from_env()?;
    let response = openai_client
        .agent(openai::GPT_4O)
        .preamble(
            "You are a travel assistant. plan_trip runs in the background and may ask the \
             traveler questions directly; incorporate its final plan into your answer.",
        )
        .tool_server_handle(tool_server_handle)
        .build()
        .runner("Plan a trip to Kyoto and summarize the plan in two sentences.")
        .max_turns(6)
        .task_deadline(Duration::from_secs(120))
        .run()
        .await?;

    println!("\n🤖 {}", response.output);
    Ok(())
}
