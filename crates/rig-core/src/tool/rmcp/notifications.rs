//! Passthrough for the MCP notifications Rig has no native concept for.
//!
//! [`McpClientHandler`](super::McpClientHandler) occupies the connection's one
//! rmcp `ClientHandler` slot, so every notification it does not consume would
//! otherwise hit rmcp's no-op defaults and be unrecoverable — an embedder can
//! *send* `resources/subscribe` through the connection's peer but would never
//! receive the resulting `resources/updated`. Registering an
//! [`McpNotificationDelegate`] via
//! [`McpClientHandler::with_notification_delegate`](super::McpClientHandler::with_notification_delegate)
//! returns those events to the embedder.
//!
//! This is plumbing, not a feature: Rig takes no opinion about what the
//! events mean. Rig forwards **after** its own handling — the tool-list
//! refresh and the task-status wakeup happen first — so a delegate observes
//! state Rig has already applied. Tool-list notifications retain the handler's
//! existing coalescing behavior, so a burst may produce one delegate callback.
//! Every method defaults to a no-op; implement only what you consume.
//!
//! `notifications/message` (server logging) is deliberately absent: its
//! parameter types are deprecated by SEP-2577, and Rig does not build on
//! deprecated MCP surfaces. rmcp's no-op default stands for it.

use std::future::ready;

use rmcp::model::{
    CancelledNotificationParam, CustomNotification, ProgressNotificationParam,
    ResourceUpdatedNotificationParam, TaskStatusNotificationParam,
};

use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};

/// A ready no-op future, the default body of every delegate method.
fn noop() -> WasmBoxedFuture<'static, ()> {
    Box::pin(ready(()))
}

/// Receives the MCP notifications Rig has no native concept for (see the
/// [module docs](self)). Every method defaults to a no-op.
pub trait McpNotificationDelegate: WasmCompatSend + WasmCompatSync {
    /// `notifications/progress` — progress for an in-flight request. The
    /// `progress_token` correlates to the originating request and, for a
    /// task-augmented call, stays valid for the task's whole lifetime.
    fn on_progress(&self, params: ProgressNotificationParam) -> WasmBoxedFuture<'_, ()> {
        let _ = params;
        noop()
    }

    /// `notifications/resources/updated` — a resource the client subscribed
    /// to (via the connection's peer) changed.
    fn on_resource_updated(
        &self,
        params: ResourceUpdatedNotificationParam,
    ) -> WasmBoxedFuture<'_, ()> {
        let _ = params;
        noop()
    }

    /// `notifications/resources/list_changed`.
    fn on_resource_list_changed(&self) -> WasmBoxedFuture<'_, ()> {
        noop()
    }

    /// `notifications/prompts/list_changed`.
    fn on_prompt_list_changed(&self) -> WasmBoxedFuture<'_, ()> {
        noop()
    }

    /// `notifications/cancelled` — the server cancelled an in-flight request.
    fn on_cancelled(&self, params: CancelledNotificationParam) -> WasmBoxedFuture<'_, ()> {
        let _ = params;
        noop()
    }

    /// A notification outside the MCP spec (`CustomNotification` carries the
    /// raw method name and params).
    fn on_custom_notification(&self, params: CustomNotification) -> WasmBoxedFuture<'_, ()> {
        let _ = params;
        noop()
    }

    /// `notifications/tools/list_changed`, observed **after** Rig re-fetched
    /// and re-registered the server's tools.
    fn on_tool_list_changed(&self) -> WasmBoxedFuture<'_, ()> {
        noop()
    }

    /// `notifications/tasks/status`, observed **after** Rig published the
    /// update to its task wakeup registry.
    fn on_task_status(&self, params: TaskStatusNotificationParam) -> WasmBoxedFuture<'_, ()> {
        let _ = params;
        noop()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex, PoisonError};
    use std::time::Duration;

    use rmcp::model::{
        ClientInfo, ErrorData, Implementation, ListToolsResult, Notification,
        PaginatedRequestParams, ProgressNotificationParam, ProgressToken, ProtocolVersion,
        ServerCapabilities, ServerInfo, ServerNotification, Task, TaskStatus,
        TaskStatusNotification, TaskStatusNotificationParam, Tool,
    };
    use rmcp::service::{RequestContext, RoleServer};
    use rmcp::{ServerHandler, ServiceExt};
    use tokio::sync::Notify;
    use tokio::time::timeout;

    use super::*;
    use crate::tool::rmcp::McpClientHandler;
    use crate::tool::server::{ToolServer, ToolServerHandle};

    /// A server whose tool list can be swapped, for the ordering test.
    #[derive(Clone)]
    struct SwappableToolServer {
        tools: Arc<tokio::sync::RwLock<Vec<Tool>>>,
    }

    impl SwappableToolServer {
        fn new(tools: Vec<Tool>) -> Self {
            Self {
                tools: Arc::new(tokio::sync::RwLock::new(tools)),
            }
        }
    }

    impl ServerHandler for SwappableToolServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
                .with_protocol_version(ProtocolVersion::LATEST)
                .with_server_info(Implementation::new("swappable-server", "0.1.0"))
        }

        async fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            Ok(ListToolsResult::with_all_items(
                self.tools.read().await.clone(),
            ))
        }
    }

    fn make_tool(name: &str) -> Tool {
        Tool::new(
            name.to_string(),
            "test tool".to_string(),
            Arc::new(serde_json::Map::new()),
        )
    }

    /// Records everything it observes; on tool-list changes it snapshots the
    /// tool server's definition count (proving rig refreshed first).
    #[derive(Clone)]
    struct RecordingDelegate {
        events: Arc<Mutex<Vec<String>>>,
        tool_server: ToolServerHandle,
        tool_counts: Arc<Mutex<Vec<usize>>>,
        changed: Arc<Notify>,
    }

    impl RecordingDelegate {
        fn new(tool_server: ToolServerHandle) -> Self {
            Self {
                events: Arc::default(),
                tool_server,
                tool_counts: Arc::default(),
                changed: Arc::default(),
            }
        }

        fn record(&self, event: impl Into<String>) {
            self.events
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(event.into());
            self.changed.notify_one();
        }

        fn events(&self) -> Vec<String> {
            self.events
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
        }

        async fn wait_for_event_count(&self, expected: usize) {
            timeout(Duration::from_secs(2), async {
                loop {
                    let changed = self.changed.notified();
                    if self
                        .events
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .len()
                        >= expected
                    {
                        return;
                    }
                    changed.await;
                }
            })
            .await
            .expect("notification delegate did not receive enough events");
        }

        async fn wait_for_tool_count(&self) {
            timeout(Duration::from_secs(2), async {
                loop {
                    let changed = self.changed.notified();
                    if !self
                        .tool_counts
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .is_empty()
                    {
                        return;
                    }
                    changed.await;
                }
            })
            .await
            .expect("notification delegate did not observe the tool refresh");
        }
    }

    impl McpNotificationDelegate for RecordingDelegate {
        fn on_progress(&self, params: ProgressNotificationParam) -> WasmBoxedFuture<'_, ()> {
            self.record(format!("progress:{}", params.progress));
            Box::pin(ready(()))
        }

        fn on_resource_updated(
            &self,
            params: ResourceUpdatedNotificationParam,
        ) -> WasmBoxedFuture<'_, ()> {
            self.record(format!("resource_updated:{}", params.uri));
            Box::pin(ready(()))
        }

        fn on_custom_notification(&self, params: CustomNotification) -> WasmBoxedFuture<'_, ()> {
            self.record(format!("custom:{}", params.method));
            Box::pin(ready(()))
        }

        fn on_tool_list_changed(&self) -> WasmBoxedFuture<'_, ()> {
            Box::pin(async move {
                // Rig refreshes the tool server BEFORE forwarding: the
                // snapshot must reflect the new list.
                let defs = self
                    .tool_server
                    .get_tool_defs(None)
                    .await
                    .unwrap_or_default();
                self.tool_counts
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(defs.len());
                self.record("tool_list_changed");
            })
        }

        fn on_task_status(&self, params: TaskStatusNotificationParam) -> WasmBoxedFuture<'_, ()> {
            self.record(format!("task_status:{}", params.task.task_id));
            Box::pin(ready(()))
        }
    }

    async fn connect(
        server: SwappableToolServer,
        delegate: Option<RecordingDelegate>,
        tool_server_handle: ToolServerHandle,
    ) -> (
        rmcp::service::RunningService<rmcp::service::RoleClient, McpClientHandler>,
        rmcp::service::RunningService<RoleServer, SwappableToolServer>,
    ) {
        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);
        let server_task = tokio::spawn(async move {
            server
                .serve((server_from_client, server_to_client))
                .await
                .expect("server failed to start")
        });
        let mut client = McpClientHandler::new(ClientInfo::default(), tool_server_handle);
        if let Some(delegate) = delegate {
            client = client.with_notification_delegate(delegate);
        }
        let client_service = client
            .connect((client_from_server, client_to_server))
            .await
            .expect("connect failed");
        let server_service = server_task.await.expect("server service");
        (client_service, server_service)
    }

    #[tokio::test]
    async fn delegate_receives_forwarded_notifications() {
        let server = SwappableToolServer::new(vec![make_tool("alpha")]);
        let tool_server_handle = ToolServer::new().run();
        let delegate = RecordingDelegate::new(tool_server_handle.clone());
        let (_client, server_service) =
            connect(server, Some(delegate.clone()), tool_server_handle).await;

        server_service
            .peer()
            .send_notification(ServerNotification::ProgressNotification(Notification::new(
                ProgressNotificationParam::new(
                    ProgressToken(rmcp::model::NumberOrString::Number(7)),
                    0.5,
                )
                .with_total(1.0),
            )))
            .await
            .expect("progress sent");
        server_service
            .peer()
            .send_notification(ServerNotification::ResourceUpdatedNotification(
                Notification::new(ResourceUpdatedNotificationParam::new("memo://insights")),
            ))
            .await
            .expect("resource update sent");
        server_service
            .peer()
            .send_notification(ServerNotification::CustomNotification(
                CustomNotification::new("notifications/x-heartbeat".to_string(), None),
            ))
            .await
            .expect("custom sent");

        delegate.wait_for_event_count(3).await;
        let events = delegate.events();
        assert!(events.contains(&"progress:0.5".to_string()), "{events:?}");
        assert!(
            events.contains(&"resource_updated:memo://insights".to_string()),
            "{events:?}"
        );
        assert!(
            events.contains(&"custom:notifications/x-heartbeat".to_string()),
            "{events:?}"
        );
    }

    #[tokio::test]
    async fn delegate_forwarding_happens_after_rig_handling() {
        let server = SwappableToolServer::new(vec![make_tool("alpha")]);
        let tool_server_handle = ToolServer::new().run();
        let delegate = RecordingDelegate::new(tool_server_handle.clone());
        let (client_service, server_service) =
            connect(server.clone(), Some(delegate.clone()), tool_server_handle).await;

        // Swap the server's tool list, notify: the delegate's snapshot must
        // already see both tools (rig refreshed before forwarding).
        *server.tools.write().await = vec![make_tool("beta"), make_tool("gamma")];
        server_service
            .peer()
            .notify_tool_list_changed()
            .await
            .expect("tool list change sent");
        delegate.wait_for_tool_count().await;
        let counts = delegate
            .tool_counts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        assert_eq!(counts, vec![2], "delegate must observe the refreshed list");

        // Task-status forwards after the registry publish: a subscriber
        // created beforehand must already hold the update when the delegate
        // runs (checked here after the fact via the recorded event + registry).
        let registry = client_service.service().task_notifications();
        let receiver = registry.subscribe("task-42");
        let task = Task::new(
            "task-42".to_string(),
            TaskStatus::Working,
            "2026-01-01T00:00:00Z".to_string(),
            "2026-01-01T00:00:00Z".to_string(),
        );
        server_service
            .peer()
            .send_notification(ServerNotification::TaskStatusNotification(
                TaskStatusNotification::new(TaskStatusNotificationParam::new(task)),
            ))
            .await
            .expect("task status sent");
        delegate.wait_for_event_count(2).await;
        assert!(
            delegate
                .events()
                .contains(&"task_status:task-42".to_string()),
            "delegate must observe the task status"
        );
        assert!(
            receiver.borrow().is_some(),
            "the wakeup registry must have been published to"
        );
    }
}
