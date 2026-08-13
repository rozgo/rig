//! MCP (Model Context Protocol) integration via the `rmcp` crate.
//!
//! This module provides an explicit 2026-07-28 Discover client. Its
//! [`McpClientGuard`] owns the transport, cache, subscriptions, registrations,
//! Tasks, MRTR reconstruction, and shutdown lifetime. Individual MCP tools keep
//! only a lightweight [`McpRequestHandle`].
//!
//! # Example
//!
//! ```rust,ignore
//! use rig_agent::tool::rmcp::{McpClientConfig, McpClientHandler};
//! use rig_agent::tool::server::ToolServer;
//! use rmcp::ServiceExt;
//!
//! // 1. Create a ToolServer and get a handle
//! let tool_server_handle = ToolServer::new().run();
//!
//! // 2. Configure the exact modern lifecycle and client identity.
//! let config = McpClientConfig::new(client_identity);
//! let handler = McpClientHandler::new(config, tool_server_handle.clone());
//!
//! // 3. Connect to the MCP server and register initial tools
//! let mcp_guard = handler.connect(transport).await?;
//!
//! // 4. Build an agent using the shared tool server handle
//! let agent = openai_client
//!     .agent(openai::GPT_5_2)
//!     .preamble("You are a helpful assistant.")
//!     .tool_server_handle(tool_server_handle)
//!     .build();
//! ```
//!
//! # Per-call metadata
//!
//! Rig's MCP adapter forwards an [`rmcp::model::RequestMetaObject`] (re-exported
//! here as [`RequestMetaObject`]) placed in a [`ToolContext`] as the MCP
//! request's `_meta`
//! (SEP-1319) — the idiomatic channel for per-call values such as auth tokens,
//! application correlation identifiers, or A2A `context_id`/`task_id`, which
//! the model never sees:
//!
//! ```rust,ignore
//! use rig_agent::tool::rmcp::RequestMetaObject;
//! use rig_agent::tool::ToolContext;
//!
//! let mut meta = RequestMetaObject::new();
//! meta.0.insert("authorization".into(), serde_json::json!("Bearer …"));
//! let mut context = ToolContext::new();
//! context.insert(meta);
//! let answer = agent.prompt("…").tool_context(context).await?;
//! ```
//!
//! # Response metadata
//!
//! MCP responses retain their protocol data in the per-dispatch
//! [`ToolContext`]. Result hooks can inspect the untouched
//! [`rmcp::model::CallToolResult`], its `structuredContent` as a
//! [`serde_json::Value`], and response [`MetaObject`] with
//! `event.tool_context.result::<T>()`. These values are host-only; only the
//! response's ordered presentation content is sent to the model.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroUsize;
use std::sync::{
    Arc, Mutex as StdMutex, Weak,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::Duration;

use rmcp::model::{
    CallToolRequest, CallToolResponse, CallToolResult, CancelTaskParams, ClientCapabilities,
    ClientInfo, ClientRequest, ContentBlock, DiscoverResult, ExtensionCapabilities, GetTaskParams,
    GetTaskResult, Implementation, InputRequiredResult, PaginatedRequestParams, ProtocolVersion,
    ResourceContents, ServerNotification, ServerResult, SubscriptionFilter, TaskPayload,
    UpdateTaskParams,
};
use rmcp::service::PeerRequestOptions;
use rmcp::{ClientCacheConfig, ClientLifecycleMode, ClientServiceExt};
use tokio::sync::{Mutex, Notify, RwLock, watch};

use crate::tool::ErasedTool;
use crate::tool::server::{ManagedToolToken, ToolServerHandle};
use crate::tool::{
    DeferredToolDescriptor, DeferredToolDriver, DeferredToolHandle, DeferredToolResolver,
    DeferredToolResolverRegistry, DeferredToolState, InputRequest as RigInputRequest,
    InputRequests as RigInputRequests, InputResponse as RigInputResponse,
    InputResponses as RigInputResponses, ToolContext, ToolExecution, ToolExecutionError,
    ToolOutput, ToolResult,
};
use rig_core::message::{ImageMediaType, MimeType, ToolResultContent};
use rig_core::wasm_compat::WasmBoxedFuture;

/// General result metadata returned by an MCP server.
pub use rmcp::model::MetaObject;
/// Request metadata forwarded from a [`ToolContext`] to MCP tool calls.
pub use rmcp::model::RequestMetaObject;

/// Default per-call timeout applied to MCP tools (see issue #1914).
///
/// MCP tool calls await a response that can be lost when a transport closes
/// with an in-flight request, which would otherwise hang the agent forever. A
/// generous default bounds that without disrupting normal, long-running tools.
/// The agent and tool-server `rmcp_tool_with_timeout` builders can override or
/// disable it.
pub const DEFAULT_MCP_TOOL_TIMEOUT: Duration = Duration::from_secs(300);

/// Default deadline for fetching an MCP server's complete tool list.
///
/// Refreshes are versioned as well as bounded: a slow older fetch may finish,
/// but it can never roll the registry back after a newer snapshot commits.
pub const DEFAULT_MCP_REFRESH_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum time spent delivering a best-effort cancellation after a request
/// has already exceeded its caller-visible deadline.
const MCP_CANCELLATION_GRACE_PERIOD: Duration = Duration::from_secs(1);

/// Timeout settings applied to requests and shutdown owned by one MCP client.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct McpRequestTimeoutPolicy {
    /// Maximum time for one tool call. `None` allows an unbounded call.
    pub tool_call: Option<Duration>,
    /// Maximum time for one complete paginated catalog refresh.
    pub catalog: Duration,
    /// Maximum time spent waiting for graceful client shutdown.
    pub shutdown: Duration,
}

impl Default for McpRequestTimeoutPolicy {
    fn default() -> Self {
        Self {
            tool_call: Some(DEFAULT_MCP_TOOL_TIMEOUT),
            catalog: DEFAULT_MCP_REFRESH_TIMEOUT,
            shutdown: Duration::from_secs(5),
        }
    }
}

/// Policy for the guard-owned `subscriptions/listen` worker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct McpSubscriptionPolicy {
    /// Whether Rig should listen for supported MCP notifications.
    pub enabled: bool,
    /// Delay before opening a replacement listen request after an abrupt end.
    pub reconnect_delay: Duration,
    /// Number of buffered subscription notifications.
    pub channel_capacity: NonZeroUsize,
}

impl Default for McpSubscriptionPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            reconnect_delay: Duration::from_millis(250),
            channel_capacity: NonZeroUsize::new(64).unwrap_or(NonZeroUsize::MIN),
        }
    }
}

/// MCP server-to-client input categories backed by an application handler.
///
/// The default advertises none. Enable a category only when the
/// [`DeferredInputHandler`](crate::tool::DeferredInputHandler) installed on
/// the agent runner can produce the corresponding MCP result value.
#[allow(deprecated)]
#[derive(Clone, Debug, Default)]
pub struct McpInputCapabilities {
    roots: Option<rmcp::model::RootsCapabilities>,
    sampling: Option<rmcp::model::SamplingCapability>,
    elicitation: Option<rmcp::model::ElicitationCapability>,
}

#[allow(deprecated)]
impl McpInputCapabilities {
    /// Declare support for roots requests.
    pub fn with_roots(mut self, capability: rmcp::model::RootsCapabilities) -> Self {
        self.roots = Some(capability);
        self
    }

    /// Declare support for sampling requests.
    pub fn with_sampling(mut self, capability: rmcp::model::SamplingCapability) -> Self {
        self.sampling = Some(capability);
        self
    }

    /// Declare support for elicitation requests.
    pub fn with_elicitation(mut self, capability: rmcp::model::ElicitationCapability) -> Self {
        self.elicitation = Some(capability);
        self
    }
}

/// Explicit configuration for a modern, stateless MCP client.
///
/// Rig always negotiates [`ProtocolVersion::V_2026_07_28`]. There is no
/// `LATEST`-based mode and no legacy initialization fallback.
#[derive(Clone, Debug)]
pub struct McpClientConfig {
    client_identity: Implementation,
    client_capabilities: ClientCapabilities,
    enabled_extensions: ExtensionCapabilities,
    request_timeouts: McpRequestTimeoutPolicy,
    subscription: McpSubscriptionPolicy,
    cache: ClientCacheConfig,
    input_capabilities: McpInputCapabilities,
    deferred_backend_id: String,
    mrtr_max_rounds: usize,
}

impl McpClientConfig {
    /// Create a configuration with Tasks enabled and no unsupported input
    /// capabilities advertised.
    pub fn new(client_identity: Implementation) -> Self {
        let mut enabled_extensions = ExtensionCapabilities::new();
        enabled_extensions.insert(
            rmcp::model::TASKS_EXTENSION_ID.to_owned(),
            rmcp::model::JsonObject::new(),
        );
        let deferred_backend_id = format!("mcp:{}", client_identity.name);
        Self {
            client_identity,
            client_capabilities: ClientCapabilities::default(),
            enabled_extensions,
            request_timeouts: McpRequestTimeoutPolicy::default(),
            subscription: McpSubscriptionPolicy::default(),
            cache: ClientCacheConfig::default(),
            input_capabilities: McpInputCapabilities::default(),
            deferred_backend_id,
            mrtr_max_rounds: rmcp::model::DEFAULT_MRTR_MAX_ROUNDS,
        }
    }

    /// The only protocol version Rig's modern MCP adapter implements.
    pub fn protocol_version(&self) -> ProtocolVersion {
        ProtocolVersion::V_2026_07_28
    }

    /// Replace the non-extension client capabilities.
    pub fn with_client_capabilities(mut self, capabilities: ClientCapabilities) -> Self {
        self.client_capabilities = capabilities;
        // These categories are reserved for `with_input_capabilities`, which
        // makes the required application-handler contract explicit.
        self.client_capabilities.roots = None;
        self.client_capabilities.sampling = None;
        self.client_capabilities.elicitation = None;
        self
    }

    /// Advertise only input categories the application can actually fulfil.
    pub fn with_input_capabilities(mut self, capabilities: McpInputCapabilities) -> Self {
        self.input_capabilities = capabilities;
        self
    }

    /// Replace the advertised extension settings. Tasks is added back because
    /// the Rig adapter implements and always advertises that extension.
    pub fn with_enabled_extensions(mut self, extensions: ExtensionCapabilities) -> Self {
        self.enabled_extensions = extensions;
        self.enabled_extensions
            .entry(rmcp::model::TASKS_EXTENSION_ID.to_owned())
            .or_default();
        self
    }

    /// Configure request deadlines.
    pub fn with_request_timeouts(mut self, policy: McpRequestTimeoutPolicy) -> Self {
        self.request_timeouts = policy;
        self
    }

    /// Configure the notification listener.
    pub fn with_subscription_policy(mut self, policy: McpSubscriptionPolicy) -> Self {
        self.subscription = policy;
        self
    }

    /// Configure cache TTL, scope partitioning, and stale-response behavior.
    pub fn with_cache_policy(mut self, policy: ClientCacheConfig) -> Self {
        self.cache = policy;
        self
    }

    /// Set the stable resolver key stored in serialized MCP deferred
    /// descriptors. Reuse the same value when reconstructing after restart.
    pub fn with_deferred_backend_id(mut self, backend_id: impl Into<String>) -> Self {
        self.deferred_backend_id = backend_id.into();
        self
    }

    /// Bound the number of direct MRTR input rounds for one tool call.
    pub fn with_mrtr_max_rounds(mut self, max_rounds: usize) -> Self {
        self.mrtr_max_rounds = max_rounds;
        self
    }

    /// Client identity sent by Discover and every later request.
    pub fn client_identity(&self) -> &Implementation {
        &self.client_identity
    }

    /// Effective capabilities, including all configured extensions.
    pub fn client_capabilities(&self) -> ClientCapabilities {
        let mut capabilities = self.client_capabilities.clone();
        #[allow(deprecated)]
        {
            capabilities.roots = self.input_capabilities.roots.clone();
            capabilities.sampling = self.input_capabilities.sampling.clone();
        }
        capabilities.elicitation = self.input_capabilities.elicitation.clone();
        let extensions = capabilities.extensions.get_or_insert_default();
        extensions.extend(self.enabled_extensions.clone());
        capabilities
    }

    /// Request timeout policy.
    pub fn request_timeouts(&self) -> &McpRequestTimeoutPolicy {
        &self.request_timeouts
    }

    /// Subscription policy.
    pub fn subscription_policy(&self) -> &McpSubscriptionPolicy {
        &self.subscription
    }

    /// Cache policy.
    pub fn cache_policy(&self) -> &ClientCacheConfig {
        &self.cache
    }

    /// Stable deferred resolver key used by this authenticated client context.
    pub fn deferred_backend_id(&self) -> &str {
        &self.deferred_backend_id
    }

    /// Maximum direct MRTR input rounds.
    pub fn mrtr_max_rounds(&self) -> usize {
        self.mrtr_max_rounds
    }

    fn client_info(&self) -> ClientInfo {
        ClientInfo::new(self.client_capabilities(), self.client_identity.clone())
            .with_protocol_version(ProtocolVersion::V_2026_07_28)
    }
}

struct TaskNotificationState {
    task_counts: StdMutex<BTreeMap<String, usize>>,
    waiters: StdMutex<HashMap<String, Arc<Notify>>>,
    revision: watch::Sender<u64>,
}

impl Default for TaskNotificationState {
    fn default() -> Self {
        let (revision, _) = watch::channel(0);
        Self {
            task_counts: StdMutex::new(BTreeMap::new()),
            waiters: StdMutex::new(HashMap::new()),
            revision,
        }
    }
}

impl TaskNotificationState {
    fn register(self: &Arc<Self>, task_id: String) -> TaskNotificationRegistration {
        {
            let mut counts = self
                .task_counts
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            *counts.entry(task_id.clone()).or_default() += 1;
        }
        self.revision.send_modify(|revision| *revision += 1);
        TaskNotificationRegistration {
            state: self.clone(),
            task_id,
        }
    }

    fn unregister(&self, task_id: &str) {
        {
            let mut counts = self
                .task_counts
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if let Some(count) = counts.get_mut(task_id) {
                *count -= 1;
                if *count == 0 {
                    counts.remove(task_id);
                }
            }
        }
        self.revision.send_modify(|revision| *revision += 1);
    }

    fn task_ids(&self) -> Vec<String> {
        self.task_counts
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .keys()
            .cloned()
            .collect()
    }

    fn waiter(&self, task_id: &str) -> Arc<Notify> {
        self.waiters
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .entry(task_id.to_owned())
            .or_insert_with(|| Arc::new(Notify::new()))
            .clone()
    }

    fn wake(&self, task_id: &str) {
        let waiter = self
            .waiters
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(task_id)
            .cloned();
        if let Some(waiter) = waiter {
            waiter.notify_one();
        }
    }

    fn subscribe(&self) -> watch::Receiver<u64> {
        self.revision.subscribe()
    }
}

struct TaskNotificationRegistration {
    state: Arc<TaskNotificationState>,
    task_id: String,
}

impl Drop for TaskNotificationRegistration {
    fn drop(&mut self) {
        self.state.unregister(&self.task_id);
    }
}

pub(crate) struct McpClientState {
    peer: rmcp::service::ServerSink,
    available: AtomicBool,
    deferred_backend_id: String,
    mrtr_max_rounds: usize,
    task_notifications: Arc<TaskNotificationState>,
}

/// Lightweight cloneable access to one guard-owned MCP client.
///
/// The handle does not keep the transport alive. Once its [`McpClientGuard`]
/// is closed or dropped, requests fail as unavailable.
#[derive(Clone)]
pub struct McpRequestHandle {
    state: Weak<McpClientState>,
}

impl std::fmt::Debug for McpRequestHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("McpRequestHandle")
            .field("available", &self.is_available())
            .finish()
    }
}

impl McpRequestHandle {
    fn from_state(state: &Arc<McpClientState>) -> Self {
        Self {
            state: Arc::downgrade(state),
        }
    }

    fn peer(&self) -> Result<rmcp::service::ServerSink, McpClientError> {
        let state = self.state.upgrade().ok_or(McpClientError::Unavailable)?;
        if !state.available.load(Ordering::Acquire) || state.peer.is_transport_closed() {
            return Err(McpClientError::Unavailable);
        }
        Ok(state.peer.clone())
    }

    fn state(&self) -> Result<Arc<McpClientState>, McpClientError> {
        let state = self.state.upgrade().ok_or(McpClientError::Unavailable)?;
        if !state.available.load(Ordering::Acquire) || state.peer.is_transport_closed() {
            return Err(McpClientError::Unavailable);
        }
        Ok(state)
    }

    fn deferred_backend_id(&self) -> Result<String, McpClientError> {
        Ok(self.state()?.deferred_backend_id.clone())
    }

    /// Whether the owning guard and transport can still accept requests.
    pub fn is_available(&self) -> bool {
        self.state.upgrade().is_some_and(|state| {
            state.available.load(Ordering::Acquire) && !state.peer.is_transport_closed()
        })
    }

    #[cfg(test)]
    pub(crate) fn for_test(peer: rmcp::service::ServerSink) -> (Self, Arc<McpClientState>) {
        let state = Arc::new(McpClientState {
            peer,
            available: AtomicBool::new(true),
            deferred_backend_id: "mcp:test".to_owned(),
            mrtr_max_rounds: rmcp::model::DEFAULT_MRTR_MAX_ROUNDS,
            task_notifications: Arc::new(TaskNotificationState::default()),
        });
        (Self::from_state(&state), state)
    }
}

/// Crate-private adapter used by Rig's public MCP registration methods.
#[derive(Clone)]
pub(crate) struct McpTool {
    definition: rmcp::model::Tool,
    client: McpRequestHandle,
    /// Per-call timeout. When `Some`, an MCP `call_tool` that does not complete
    /// within this duration resolves to a [`ToolExecutionError`] instead of blocking
    /// forever (see issue #1914). When `None`, the call is unbounded.
    ///
    /// On elapse RMCP sends a cancellation notification so both peers can
    /// release request-scoped resources.
    timeout: Option<Duration>,
}

impl McpTool {
    /// Create an adapter from an MCP tool definition and server sink.
    ///
    /// Applies [`DEFAULT_MCP_TOOL_TIMEOUT`] so a lost/never-answered response
    /// cannot hang the agent forever (issue #1914).
    pub(crate) fn from_mcp_server(definition: rmcp::model::Tool, client: McpRequestHandle) -> Self {
        Self {
            definition,
            client,
            timeout: Some(DEFAULT_MCP_TOOL_TIMEOUT),
        }
    }

    /// Set (or clear) the per-call timeout, consuming and returning the tool.
    ///
    /// Pass a [`Duration`] to bound calls, or `None` to make them unbounded.
    /// On timeout the call resolves to a [`ToolExecutionError`] (which the agent loop
    /// surfaces to the model as a tool result, so the agent can recover rather
    /// than hang). RMCP sends a cancellation notification when the deadline
    /// elapses.
    pub(crate) fn with_timeout(mut self, timeout: impl Into<Option<Duration>>) -> Self {
        self.timeout = timeout.into();
        self
    }

    /// The per-call timeout, if any.
    #[cfg(test)]
    pub(crate) fn timeout(&self) -> Option<Duration> {
        self.timeout
    }
}

/// Parse the JSON `args` string into MCP call arguments.
///
/// Argument decoding failure at the MCP object boundary.
#[derive(Debug, thiserror::Error)]
enum McpArgumentError {
    /// Malformed JSON.
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// Valid JSON that cannot be represented by MCP's object-valued arguments.
    #[error("expected a JSON object or null, got {0}")]
    NonObject(&'static str),
}

fn json_value_kind(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Returns no argument map for empty input or explicit JSON `null`, and an MCP
/// argument map for a JSON object. Other valid JSON shapes are rejected: silently
/// turning an array or scalar into a no-argument request can execute a different
/// operation than the model requested.
fn parse_mcp_arguments(args: &str) -> Result<Option<rmcp::model::JsonObject>, McpArgumentError> {
    let trimmed = args.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let value: serde_json::Value = serde_json::from_str(trimmed)?;
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Object(_) => Ok(Some(serde_json::from_value(value)?)),
        value => Err(McpArgumentError::NonObject(json_value_kind(&value))),
    }
}

async fn call_mcp_tool(
    peer: &rmcp::service::ServerSink,
    params: rmcp::model::CallToolRequestParams,
    timeout: Option<Duration>,
) -> Result<CallToolResponse, rmcp::ServiceError> {
    let deadline = timeout.map(|timeout| (tokio::time::Instant::now() + timeout, timeout));
    let response = send_mcp_request(
        peer,
        ClientRequest::CallToolRequest(CallToolRequest::new(params)),
        deadline,
    )
    .await?;

    match response {
        ServerResult::CallToolResult(result) => Ok(CallToolResponse::Complete(result)),
        ServerResult::InputRequiredResult(result) => Ok(CallToolResponse::InputRequired(result)),
        ServerResult::CreateTaskResult(result) => Ok(CallToolResponse::Task(result)),
        _ => Err(rmcp::ServiceError::UnexpectedResponse),
    }
}

async fn send_mcp_request(
    peer: &rmcp::service::ServerSink,
    request: ClientRequest,
    deadline: Option<(tokio::time::Instant, Duration)>,
) -> Result<ServerResult, rmcp::ServiceError> {
    let handle = match deadline {
        Some((deadline, timeout)) => {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return Err(rmcp::ServiceError::Timeout { timeout });
            }
            rig_core::wasm_compat::timeout(
                remaining,
                peer.send_cancellable_request(request, PeerRequestOptions::no_options()),
            )
            .await
            .map_err(|_| rmcp::ServiceError::Timeout { timeout })??
        }
        None => {
            peer.send_cancellable_request(request, PeerRequestOptions::no_options())
                .await?
        }
    };

    let Some((deadline, timeout)) = deadline else {
        return handle.await_response().await;
    };
    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
    let mut handle = handle;
    match rig_core::wasm_compat::timeout(remaining, &mut handle.rx).await {
        Ok(response) => response.map_err(|_| rmcp::ServiceError::TransportClosed)?,
        Err(_) => {
            cancel_timed_out_request(handle);
            Err(rmcp::ServiceError::Timeout { timeout })
        }
    }
}

/// Keep cancellation delivery out of the caller's deadline. RMCP's cancellation
/// notification uses the same bounded outbound queue as requests, so awaiting it
/// inline could exceed the timeout precisely when that queue is saturated. The
/// detached delivery is itself bounded so a stalled transport cannot retain one
/// task and request handle for every timed-out call indefinitely.
fn cancel_timed_out_request(handle: rmcp::service::RequestHandle<rmcp::service::RoleClient>) {
    let cancellation = async move {
        bounded_best_effort_cancellation(
            handle.cancel(Some(
                rmcp::service::RequestHandle::<rmcp::service::RoleClient>::REQUEST_TIMEOUT_REASON
                    .to_owned(),
            )),
            MCP_CANCELLATION_GRACE_PERIOD,
        )
        .await;
    };

    // This module is native-only (see the `compile_error!` in `tool/mod.rs`), so
    // there is no `spawn_local` branch to pick: `tokio::spawn` is always right
    // here.
    tokio::spawn(cancellation);
}

async fn bounded_best_effort_cancellation(
    cancellation: impl std::future::Future<Output = Result<(), rmcp::ServiceError>>,
    grace_period: Duration,
) {
    let _ = rig_core::wasm_compat::timeout(grace_period, cancellation).await;
}

const RESERVED_CLIENT_META_KEYS: [&str; 3] = [
    "io.modelcontextprotocol/protocolVersion",
    "io.modelcontextprotocol/clientInfo",
    "io.modelcontextprotocol/clientCapabilities",
];

fn without_reserved_client_meta(mut meta: Option<RequestMetaObject>) -> Option<RequestMetaObject> {
    if let Some(meta) = meta.as_mut() {
        for key in RESERVED_CLIENT_META_KEYS {
            meta.remove(key);
        }
    }
    meta
}

const W3C_TRACE_META_KEYS: [&str; 2] = ["traceparent", "tracestate"];

fn retained_trace_meta(meta: Option<&RequestMetaObject>) -> BTreeMap<String, serde_json::Value> {
    let Some(meta) = meta else {
        return BTreeMap::new();
    };
    W3C_TRACE_META_KEYS
        .into_iter()
        .filter_map(|key| meta.get(key).cloned().map(|value| (key.to_owned(), value)))
        .collect()
}

fn request_meta_from_trace(
    trace_meta: &BTreeMap<String, serde_json::Value>,
) -> Option<RequestMetaObject> {
    if trace_meta.is_empty() {
        return None;
    }
    let mut meta = RequestMetaObject::new();
    for (key, value) in trace_meta {
        meta.insert(key.clone(), value.clone());
    }
    Some(meta)
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
enum McpDeferredPayload {
    Task {
        task_id: String,
        expires_at_unix_ms: Option<i64>,
        poll_interval_ms: Option<u64>,
        request_timeout_ms: Option<u64>,
        #[serde(default)]
        trace_meta: BTreeMap<String, serde_json::Value>,
    },
    Mrtr {
        tool_name: String,
        arguments: Option<rmcp::model::JsonObject>,
        request_state: Option<String>,
        input_requests: rmcp::model::InputRequests,
        round: usize,
        max_rounds: usize,
        request_timeout_ms: Option<u64>,
        #[serde(default)]
        trace_meta: BTreeMap<String, serde_json::Value>,
    },
}

fn duration_to_millis(duration: Option<Duration>) -> Option<u64> {
    duration.map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
}

fn duration_from_millis(duration: Option<u64>) -> Option<Duration> {
    duration.map(Duration::from_millis)
}

fn task_expiry_unix_ms(task: &rmcp::model::Task) -> Result<Option<i64>, ToolExecutionError> {
    let Some(ttl_ms) = task.ttl_ms else {
        return Ok(None);
    };
    let created_at = chrono::DateTime::parse_from_rfc3339(&task.created_at).map_err(|error| {
        ToolExecutionError::provider(format!(
            "MCP Task '{}' has an invalid createdAt timestamp: {error}",
            task.task_id
        ))
        .with_source(error)
    })?;
    let ttl_ms = i64::try_from(ttl_ms).map_err(|error| {
        ToolExecutionError::provider(format!(
            "MCP Task '{}' TTL is too large to enforce",
            task.task_id
        ))
        .with_source(error)
    })?;
    created_at
        .timestamp_millis()
        .checked_add(ttl_ms)
        .map(Some)
        .ok_or_else(|| {
            ToolExecutionError::provider(format!(
                "MCP Task '{}' expiry timestamp overflowed",
                task.task_id
            ))
        })
}

fn task_descriptor(
    client: &McpRequestHandle,
    task: &rmcp::model::Task,
    request_timeout: Option<Duration>,
    request_meta: Option<&RequestMetaObject>,
) -> Result<DeferredToolDescriptor, ToolExecutionError> {
    let payload = McpDeferredPayload::Task {
        task_id: task.task_id.clone(),
        expires_at_unix_ms: task_expiry_unix_ms(task)?,
        poll_interval_ms: task.poll_interval_ms,
        request_timeout_ms: duration_to_millis(request_timeout),
        trace_meta: retained_trace_meta(request_meta),
    };
    let payload = serde_json::to_value(payload).map_err(|error| {
        ToolExecutionError::provider(format!(
            "failed to serialize MCP Task '{}': {error}",
            task.task_id
        ))
        .with_source(error)
    })?;
    let backend = client.deferred_backend_id().map_err(|error| {
        ToolExecutionError::provider(format!("MCP client is unavailable: {error}"))
            .with_source(error)
    })?;
    Ok(DeferredToolDescriptor::new(
        backend,
        task.task_id.clone(),
        payload,
    ))
}

fn mrtr_descriptor(
    client: &McpRequestHandle,
    request: &rmcp::model::CallToolRequestParams,
    input_required: &InputRequiredResult,
    round: usize,
    request_timeout: Option<Duration>,
) -> Result<DeferredToolDescriptor, ToolExecutionError> {
    let state = client.state().map_err(|error| {
        ToolExecutionError::provider(format!("MCP client is unavailable: {error}"))
            .with_source(error)
    })?;
    let payload = McpDeferredPayload::Mrtr {
        tool_name: request.name.to_string(),
        arguments: request.arguments.clone(),
        request_state: input_required.request_state.clone(),
        input_requests: input_required.input_requests.clone().unwrap_or_default(),
        round,
        max_rounds: state.mrtr_max_rounds,
        request_timeout_ms: duration_to_millis(request_timeout),
        trace_meta: retained_trace_meta(request.meta.as_ref()),
    };
    let payload = serde_json::to_value(payload).map_err(|error| {
        ToolExecutionError::provider(format!("failed to serialize MCP MRTR state: {error}"))
            .with_source(error)
    })?;
    Ok(DeferredToolDescriptor::new(
        state.deferred_backend_id.clone(),
        rig_core::id::generate(),
        payload,
    ))
}

#[allow(deprecated)]
fn rig_input_requests(requests: &rmcp::model::InputRequests) -> RigInputRequests {
    RigInputRequests(
        requests
            .iter()
            .map(|(id, request)| {
                let kind = match request {
                    rmcp::model::InputRequest::CreateMessage(_) => "sampling",
                    rmcp::model::InputRequest::Elicitation(_) => "elicitation",
                    rmcp::model::InputRequest::ListRoots(_) => "roots",
                    _ => "unsupported",
                };
                RigInputRequest {
                    id: id.clone(),
                    kind: kind.to_owned(),
                    prompt: None,
                    schema: serde_json::to_value(request).ok(),
                    metadata: serde_json::Map::new(),
                }
            })
            .collect(),
    )
}

fn rmcp_input_responses(responses: RigInputResponses) -> rmcp::model::InputResponses {
    responses
        .0
        .into_iter()
        .map(|response: RigInputResponse| (response.request_id, response.value))
        .collect()
}

fn mcp_call_result(
    result: &CallToolResult,
    tool_name: Option<&str>,
) -> Result<ToolResult, ToolExecutionError> {
    let output = mcp_result_output(result)?;
    if result.is_error == Some(true) {
        let message = tool_name.map_or_else(
            || "MCP tool reported an execution error".to_owned(),
            |name| format!("MCP tool '{name}' reported an execution error"),
        );
        Ok(ToolResult::failed(
            ToolExecutionError::other(message).with_model_output(output),
        ))
    } else {
        Ok(ToolResult::success(output))
    }
}

#[derive(Clone, Default)]
struct McpPublishedContext {
    call_result: Option<CallToolResult>,
    task_result: Option<GetTaskResult>,
    input_required: Option<InputRequiredResult>,
}

fn publish_mcp_deferred_context(
    published: &StdMutex<McpPublishedContext>,
    context: &mut ToolContext,
) {
    let published = published
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clone();
    if let Some(result) = published.call_result {
        preserve_mcp_result(context, &result);
    }
    if let Some(result) = published.task_result {
        if let Some(meta) = result.meta.clone() {
            context.insert_result(meta);
        }
        context.insert_result(result);
    }
    if let Some(result) = published.input_required {
        if let Some(meta) = result.meta.clone() {
            context.insert_result(meta);
        }
        context.insert_result(result);
    }
}

fn request_deadline(timeout: Option<Duration>) -> Option<(tokio::time::Instant, Duration)> {
    timeout.map(|timeout| (tokio::time::Instant::now() + timeout, timeout))
}

async fn get_mcp_task(
    peer: &rmcp::service::ServerSink,
    task_id: &str,
    meta: Option<RequestMetaObject>,
    timeout: Option<Duration>,
) -> Result<GetTaskResult, rmcp::ServiceError> {
    let mut params = GetTaskParams::new(task_id);
    params.meta = meta;
    match send_mcp_request(
        peer,
        ClientRequest::GetTaskRequest(rmcp::model::GetTaskRequest::new(params)),
        request_deadline(timeout),
    )
    .await?
    {
        ServerResult::GetTaskResult(result) => Ok(result),
        _ => Err(rmcp::ServiceError::UnexpectedResponse),
    }
}

async fn update_mcp_task(
    peer: &rmcp::service::ServerSink,
    task_id: &str,
    responses: rmcp::model::InputResponses,
    meta: Option<RequestMetaObject>,
    timeout: Option<Duration>,
) -> Result<(), rmcp::ServiceError> {
    let mut params = UpdateTaskParams::new(task_id, responses);
    params.meta = meta;
    match send_mcp_request(
        peer,
        ClientRequest::UpdateTaskRequest(rmcp::model::UpdateTaskRequest::new(params)),
        request_deadline(timeout),
    )
    .await?
    {
        ServerResult::TaskAckResult(_) | ServerResult::EmptyResult(_) => Ok(()),
        _ => Err(rmcp::ServiceError::UnexpectedResponse),
    }
}

async fn cancel_mcp_task(
    peer: &rmcp::service::ServerSink,
    task_id: &str,
    meta: Option<RequestMetaObject>,
    timeout: Option<Duration>,
) -> Result<(), rmcp::ServiceError> {
    let mut params = CancelTaskParams::new(task_id);
    params.meta = meta;
    match send_mcp_request(
        peer,
        ClientRequest::CancelTaskRequest(rmcp::model::CancelTaskRequest::new(params)),
        request_deadline(timeout),
    )
    .await?
    {
        ServerResult::TaskAckResult(_) | ServerResult::EmptyResult(_) => Ok(()),
        _ => Err(rmcp::ServiceError::UnexpectedResponse),
    }
}

fn mcp_service_error(operation: &str, error: rmcp::ServiceError) -> ToolExecutionError {
    match error {
        timeout @ rmcp::ServiceError::Timeout { timeout: duration } => {
            ToolExecutionError::timeout(format!("MCP {operation} timed out after {duration:?}"))
                .with_source(timeout)
        }
        error => ToolExecutionError::network(format!("MCP {operation} failed: {error}"))
            .with_source(error)
            .with_retryable(false),
    }
}

struct McpTaskRuntime {
    terminal: Option<DeferredToolState>,
    polled_once: bool,
    next_poll: tokio::time::Instant,
    poll_interval: Duration,
    expires_at_unix_ms: Option<i64>,
}

struct McpTaskDriver {
    client: McpRequestHandle,
    task_id: String,
    request_timeout: Option<Duration>,
    trace_meta: BTreeMap<String, serde_json::Value>,
    waiter: Arc<Notify>,
    _registration: TaskNotificationRegistration,
    runtime: Mutex<McpTaskRuntime>,
    published: StdMutex<McpPublishedContext>,
}

impl McpTaskDriver {
    fn new(
        client: McpRequestHandle,
        task_id: String,
        expires_at_unix_ms: Option<i64>,
        poll_interval_ms: Option<u64>,
        request_timeout: Option<Duration>,
        trace_meta: BTreeMap<String, serde_json::Value>,
    ) -> Result<Self, ToolExecutionError> {
        let state = client.state().map_err(|error| {
            ToolExecutionError::provider(format!("MCP client is unavailable: {error}"))
                .with_source(error)
        })?;
        let waiter = state.task_notifications.waiter(&task_id);
        let registration = state.task_notifications.register(task_id.clone());
        Ok(Self {
            client,
            task_id,
            request_timeout,
            trace_meta,
            waiter,
            _registration: registration,
            runtime: Mutex::new(McpTaskRuntime {
                terminal: None,
                polled_once: false,
                next_poll: tokio::time::Instant::now(),
                poll_interval: Duration::from_millis(poll_interval_ms.unwrap_or(1_000)),
                expires_at_unix_ms,
            }),
            published: StdMutex::new(McpPublishedContext::default()),
        })
    }

    fn ttl_error(&self) -> ToolExecutionError {
        ToolExecutionError::timeout(format!(
            "MCP Task '{}' expired before reaching a terminal state",
            self.task_id
        ))
        .with_code("mcp_task_ttl_expired")
        .with_retryable(false)
    }

    fn expiry_remaining(expires_at_unix_ms: Option<i64>) -> Option<Duration> {
        let expires_at = expires_at_unix_ms?;
        let remaining = expires_at.saturating_sub(chrono::Utc::now().timestamp_millis());
        Some(Duration::from_millis(u64::try_from(remaining).unwrap_or(0)))
    }

    async fn poll(&self, wait_for_schedule: bool) -> Result<DeferredToolState, ToolExecutionError> {
        let (terminal, polled_once, next_poll, expires_at) = {
            let runtime = self.runtime.lock().await;
            (
                runtime.terminal.clone(),
                runtime.polled_once,
                runtime.next_poll,
                runtime.expires_at_unix_ms,
            )
        };
        if let Some(terminal) = terminal {
            return Ok(terminal);
        }
        if Self::expiry_remaining(expires_at).is_some_and(|remaining| remaining.is_zero()) {
            let terminal = DeferredToolState::Failed(self.ttl_error());
            self.runtime.lock().await.terminal = Some(terminal.clone());
            return Ok(terminal);
        }

        if wait_for_schedule && polled_once {
            let poll_sleep = tokio::time::sleep_until(next_poll);
            tokio::pin!(poll_sleep);
            if let Some(ttl_remaining) = Self::expiry_remaining(expires_at) {
                let ttl_sleep = tokio::time::sleep(ttl_remaining);
                tokio::pin!(ttl_sleep);
                tokio::select! {
                    _ = &mut poll_sleep => {}
                    _ = self.waiter.notified() => {}
                    _ = &mut ttl_sleep => {
                        let terminal = DeferredToolState::Failed(self.ttl_error());
                        self.runtime.lock().await.terminal = Some(terminal.clone());
                        return Ok(terminal);
                    }
                }
            } else {
                tokio::select! {
                    _ = &mut poll_sleep => {}
                    _ = self.waiter.notified() => {}
                }
            }
        }

        // Notifications only shorten the wait. The authoritative state always
        // comes from tasks/get so lost notifications cannot affect correctness.
        let peer = self.client.peer().map_err(|error| {
            ToolExecutionError::network(format!("MCP client is unavailable: {error}"))
                .with_source(error)
                .with_retryable(false)
        })?;
        let result = get_mcp_task(
            &peer,
            &self.task_id,
            request_meta_from_trace(&self.trace_meta),
            self.request_timeout,
        )
        .await
        .map_err(|error| mcp_service_error("tasks/get", error))?;
        if result.task.task.task_id != self.task_id {
            return Err(ToolExecutionError::provider(format!(
                "MCP tasks/get returned task '{}' while '{}' was requested",
                result.task.task.task_id, self.task_id
            ))
            .with_retryable(false));
        }

        let expires_at = task_expiry_unix_ms(&result.task.task)?;
        let poll_interval =
            Duration::from_millis(result.task.task.poll_interval_ms.unwrap_or(1_000));
        let state = match &result.task.payload {
            TaskPayload::Working => DeferredToolState::Working,
            TaskPayload::InputRequired { input_requests } => {
                DeferredToolState::InputRequired(rig_input_requests(input_requests))
            }
            TaskPayload::Completed { result: value } => {
                match serde_json::from_value::<CallToolResult>(serde_json::Value::Object(
                    value.clone(),
                )) {
                    Ok(call_result) => {
                        let tool_result = mcp_call_result(&call_result, None)?;
                        self.published
                            .lock()
                            .unwrap_or_else(|error| error.into_inner())
                            .call_result = Some(call_result);
                        DeferredToolState::Completed(tool_result)
                    }
                    Err(error) => DeferredToolState::Failed(
                        ToolExecutionError::provider(format!(
                            "MCP Task '{}' completed with an invalid tools/call result: {error}",
                            self.task_id
                        ))
                        .with_source(error)
                        .with_retryable(false),
                    ),
                }
            }
            TaskPayload::Failed { error } => DeferredToolState::Failed(
                ToolExecutionError::provider(format!(
                    "MCP Task '{}' failed with JSON-RPC error: {}",
                    self.task_id,
                    serde_json::Value::Object(error.clone())
                ))
                .with_code("mcp_task_failed")
                .with_retryable(false),
            ),
            TaskPayload::Cancelled => DeferredToolState::Cancelled,
            _ => DeferredToolState::Failed(
                ToolExecutionError::provider(format!(
                    "MCP Task '{}' returned an unsupported status payload",
                    self.task_id
                ))
                .with_retryable(false),
            ),
        };
        self.published
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .task_result = Some(result);

        let mut runtime = self.runtime.lock().await;
        if let Some(terminal) = runtime.terminal.clone() {
            return Ok(terminal);
        }
        runtime.polled_once = true;
        runtime.poll_interval = poll_interval;
        runtime.next_poll = tokio::time::Instant::now() + poll_interval;
        runtime.expires_at_unix_ms = expires_at;
        if matches!(
            state,
            DeferredToolState::Completed(_)
                | DeferredToolState::Failed(_)
                | DeferredToolState::Cancelled
        ) {
            runtime.terminal = Some(state.clone());
        }
        Ok(state)
    }
}

impl DeferredToolDriver for McpTaskDriver {
    fn state(&self) -> WasmBoxedFuture<'_, Result<DeferredToolState, ToolExecutionError>> {
        Box::pin(self.poll(true))
    }

    fn submit_input(
        &self,
        responses: RigInputResponses,
    ) -> WasmBoxedFuture<'_, Result<DeferredToolState, ToolExecutionError>> {
        Box::pin(async move {
            if let Some(terminal) = self.runtime.lock().await.terminal.clone() {
                return Ok(terminal);
            }
            let peer = self.client.peer().map_err(|error| {
                ToolExecutionError::network(format!("MCP client is unavailable: {error}"))
                    .with_source(error)
                    .with_retryable(false)
            })?;
            update_mcp_task(
                &peer,
                &self.task_id,
                rmcp_input_responses(responses),
                request_meta_from_trace(&self.trace_meta),
                self.request_timeout,
            )
            .await
            .map_err(|error| mcp_service_error("tasks/update", error))?;
            self.poll(false).await
        })
    }

    fn cancel(&self) -> WasmBoxedFuture<'_, Result<DeferredToolState, ToolExecutionError>> {
        Box::pin(async move {
            if let Some(terminal) = self.runtime.lock().await.terminal.clone() {
                return Ok(terminal);
            }
            let peer = self.client.peer().map_err(|error| {
                ToolExecutionError::network(format!("MCP client is unavailable: {error}"))
                    .with_source(error)
                    .with_retryable(false)
            })?;
            cancel_mcp_task(
                &peer,
                &self.task_id,
                request_meta_from_trace(&self.trace_meta),
                self.request_timeout,
            )
            .await
            .map_err(|error| mcp_service_error("tasks/cancel", error))?;
            self.poll(false).await
        })
    }

    fn publish_result_context(&self, context: &mut ToolContext) {
        publish_mcp_deferred_context(&self.published, context);
    }
}

struct McpMrtrRuntime {
    current: InputRequiredResult,
    round: usize,
    terminal: Option<DeferredToolState>,
}

struct McpMrtrDriver {
    client: McpRequestHandle,
    tool_name: String,
    arguments: Option<rmcp::model::JsonObject>,
    max_rounds: usize,
    request_timeout: Option<Duration>,
    trace_meta: BTreeMap<String, serde_json::Value>,
    runtime: Mutex<McpMrtrRuntime>,
    task_driver: StdMutex<Option<Arc<McpTaskDriver>>>,
    published: StdMutex<McpPublishedContext>,
}

impl McpMrtrDriver {
    async fn retry(
        &self,
        responses: rmcp::model::InputResponses,
    ) -> Result<DeferredToolState, ToolExecutionError> {
        let (request_state, next_round) = {
            let runtime = self.runtime.lock().await;
            if let Some(terminal) = runtime.terminal.clone() {
                return Ok(terminal);
            }
            if runtime.round >= self.max_rounds {
                drop(runtime);
                let terminal = DeferredToolState::Failed(
                    ToolExecutionError::provider(format!(
                        "MCP tool '{}' exceeded the configured MRTR limit of {} rounds",
                        self.tool_name, self.max_rounds
                    ))
                    .with_code("mcp_mrtr_rounds_exceeded")
                    .with_retryable(false),
                );
                self.runtime.lock().await.terminal = Some(terminal.clone());
                return Ok(terminal);
            }
            (runtime.current.request_state.clone(), runtime.round + 1)
        };

        let mut params = rmcp::model::CallToolRequestParams::new(self.tool_name.clone());
        params.arguments = self.arguments.clone();
        params.input_responses = Some(responses);
        params.request_state = request_state;
        params.meta = request_meta_from_trace(&self.trace_meta);
        let peer = self.client.peer().map_err(|error| {
            ToolExecutionError::network(format!("MCP client is unavailable: {error}"))
                .with_source(error)
                .with_retryable(false)
        })?;
        let response = call_mcp_tool(&peer, params, self.request_timeout)
            .await
            .map_err(|error| mcp_service_error("tools/call MRTR retry", error))?;
        match response {
            CallToolResponse::Complete(result) => {
                let tool_result = mcp_call_result(&result, Some(&self.tool_name))?;
                self.published
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .call_result = Some(result);
                let terminal = DeferredToolState::Completed(tool_result);
                self.runtime.lock().await.terminal = Some(terminal.clone());
                Ok(terminal)
            }
            CallToolResponse::InputRequired(result) => {
                self.published
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .input_required = Some(result.clone());
                let state = result
                    .input_requests
                    .as_ref()
                    .map(rig_input_requests)
                    .map(DeferredToolState::InputRequired)
                    .unwrap_or(DeferredToolState::Working);
                let mut runtime = self.runtime.lock().await;
                runtime.current = result;
                runtime.round = next_round;
                Ok(state)
            }
            CallToolResponse::Task(task) => {
                let driver = Arc::new(McpTaskDriver::new(
                    self.client.clone(),
                    task.task.task_id.clone(),
                    task_expiry_unix_ms(&task.task)?,
                    task.task.poll_interval_ms,
                    self.request_timeout,
                    self.trace_meta.clone(),
                )?);
                *self
                    .task_driver
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()) = Some(driver.clone());
                driver.poll(false).await
            }
            _ => Err(ToolExecutionError::provider(
                "MCP tools/call returned an unsupported result type",
            )
            .with_retryable(false)),
        }
    }

    fn current_task_driver(&self) -> Option<Arc<McpTaskDriver>> {
        self.task_driver
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }
}

impl DeferredToolDriver for McpMrtrDriver {
    fn state(&self) -> WasmBoxedFuture<'_, Result<DeferredToolState, ToolExecutionError>> {
        Box::pin(async move {
            if let Some(driver) = self.current_task_driver() {
                return driver.state().await;
            }
            loop {
                let (terminal, requests) = {
                    let runtime = self.runtime.lock().await;
                    (
                        runtime.terminal.clone(),
                        runtime.current.input_requests.clone().unwrap_or_default(),
                    )
                };
                if let Some(terminal) = terminal {
                    return Ok(terminal);
                }
                if !requests.is_empty() {
                    return Ok(DeferredToolState::InputRequired(rig_input_requests(
                        &requests,
                    )));
                }
                let state = self.retry(BTreeMap::new()).await?;
                if !matches!(state, DeferredToolState::Working) {
                    return Ok(state);
                }
            }
        })
    }

    fn submit_input(
        &self,
        responses: RigInputResponses,
    ) -> WasmBoxedFuture<'_, Result<DeferredToolState, ToolExecutionError>> {
        Box::pin(async move {
            if let Some(driver) = self.current_task_driver() {
                return driver.submit_input(responses).await;
            }
            let expected = self
                .runtime
                .lock()
                .await
                .current
                .input_requests
                .clone()
                .unwrap_or_default();
            let responses = rmcp_input_responses(responses);
            if expected.len() != responses.len()
                || !expected.keys().all(|key| responses.contains_key(key))
            {
                return Err(ToolExecutionError::invalid_args(
                    "MRTR responses must exactly match the outstanding input request identifiers",
                ));
            }
            self.retry(responses).await
        })
    }

    fn cancel(&self) -> WasmBoxedFuture<'_, Result<DeferredToolState, ToolExecutionError>> {
        Box::pin(async move {
            if let Some(driver) = self.current_task_driver() {
                return driver.cancel().await;
            }
            let mut runtime = self.runtime.lock().await;
            if let Some(terminal) = runtime.terminal.clone() {
                return Ok(terminal);
            }
            runtime.terminal = Some(DeferredToolState::Cancelled);
            Ok(DeferredToolState::Cancelled)
        })
    }

    fn publish_result_context(&self, context: &mut ToolContext) {
        publish_mcp_deferred_context(&self.published, context);
        if let Some(driver) = self.current_task_driver() {
            driver.publish_result_context(context);
        }
    }
}

/// Reconstructs serialized MCP Task and MRTR executions for one live client.
#[derive(Clone)]
pub struct McpDeferredResolver {
    backend_id: String,
    client: McpRequestHandle,
}

impl std::fmt::Debug for McpDeferredResolver {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("McpDeferredResolver")
            .field("backend_id", &self.backend_id)
            .field("available", &self.client.is_available())
            .finish()
    }
}

impl DeferredToolResolver for McpDeferredResolver {
    fn backend_type(&self) -> &str {
        &self.backend_id
    }

    fn resolve<'a>(
        &'a self,
        descriptor: &'a DeferredToolDescriptor,
    ) -> WasmBoxedFuture<'a, Result<DeferredToolHandle, ToolExecutionError>> {
        Box::pin(async move {
            if descriptor.backend_type() != self.backend_id {
                return Err(ToolExecutionError::invalid_args(format!(
                    "MCP deferred descriptor targets '{}' but this resolver owns '{}'",
                    descriptor.backend_type(),
                    self.backend_id
                )));
            }
            let payload: McpDeferredPayload = serde_json::from_value(descriptor.payload().clone())
                .map_err(|error| {
                    ToolExecutionError::invalid_args(format!(
                        "invalid MCP deferred descriptor payload: {error}"
                    ))
                    .with_source(error)
                })?;
            match payload {
                McpDeferredPayload::Task {
                    task_id,
                    expires_at_unix_ms,
                    poll_interval_ms,
                    request_timeout_ms,
                    trace_meta,
                } => {
                    if descriptor.execution_id() != task_id {
                        return Err(ToolExecutionError::invalid_args(
                            "MCP Task descriptor execution ID does not match its opaque task ID",
                        ));
                    }
                    let driver = McpTaskDriver::new(
                        self.client.clone(),
                        task_id,
                        expires_at_unix_ms,
                        poll_interval_ms,
                        duration_from_millis(request_timeout_ms),
                        trace_meta,
                    )?;
                    Ok(DeferredToolHandle::new(descriptor.clone(), driver))
                }
                McpDeferredPayload::Mrtr {
                    tool_name,
                    arguments,
                    request_state,
                    input_requests,
                    round,
                    max_rounds,
                    request_timeout_ms,
                    trace_meta,
                } => {
                    let current = InputRequiredResult::new(Some(input_requests), request_state);
                    let driver = McpMrtrDriver {
                        client: self.client.clone(),
                        tool_name,
                        arguments,
                        max_rounds,
                        request_timeout: duration_from_millis(request_timeout_ms),
                        trace_meta,
                        runtime: Mutex::new(McpMrtrRuntime {
                            current,
                            round,
                            terminal: None,
                        }),
                        task_driver: StdMutex::new(None),
                        published: StdMutex::new(McpPublishedContext::default()),
                    };
                    Ok(DeferredToolHandle::new(descriptor.clone(), driver))
                }
            }
        })
    }
}

impl McpTool {
    /// Execute one MCP request.
    ///
    /// `meta`, when present, is attached as the MCP request's `_meta`
    /// (SEP-1319) — the idiomatic channel for per-call metadata such as auth
    /// tokens, application correlation identifiers, or A2A
    /// `context_id`/`task_id`. It is supplied by a caller that places an
    /// [`rmcp::model::RequestMetaObject`] into the [`ToolContext`]; otherwise
    /// the call behaves exactly as before.
    fn execute_mcp(
        &self,
        args: String,
        meta: Option<RequestMetaObject>,
    ) -> WasmBoxedFuture<
        '_,
        Result<(CallToolResponse, rmcp::model::CallToolRequestParams), ToolExecutionError>,
    > {
        let name = self.definition.name.clone();

        Box::pin(async move {
            // Validate the JSON arguments before contacting the server: malformed
            // JSON must surface as an InvalidArgs failure, not a silent no-arg call.
            let arguments = parse_mcp_arguments(&args).map_err(|error| {
                ToolExecutionError::invalid_args(format!(
                    "MCP tool '{name}' received invalid arguments: {error}"
                ))
                .with_source(error)
            })?;
            let mut request = arguments
                .map(|arguments| {
                    rmcp::model::CallToolRequestParams::new(name.clone()).with_arguments(arguments)
                })
                .unwrap_or_else(|| rmcp::model::CallToolRequestParams::new(name));
            // The Discover lifecycle injects the guard's immutable protocol
            // version, identity, and capabilities. Per-dispatch metadata may
            // carry trace and application extension fields, but cannot replace
            // those client-owned values.
            request.meta = without_reserved_client_meta(meta);

            let client = self.client.peer().map_err(|error| {
                ToolExecutionError::provider(format!(
                    "MCP tool '{}' is unavailable: {error}",
                    self.definition.name
                ))
                .with_source(error)
            })?;
            match call_mcp_tool(&client, request.clone(), self.timeout).await {
                Ok(result) => Ok((result, request)),
                Err(
                    error @ rmcp::ServiceError::Timeout {
                        timeout: elapsed_timeout,
                    },
                ) => {
                    let timeout = self.timeout.unwrap_or(elapsed_timeout);
                    Err(ToolExecutionError::timeout(format!(
                        "MCP tool '{}' timed out after {timeout:?}",
                        self.definition.name
                    ))
                    .with_source(error))
                }
                // A transport/service error before the tool produced a result.
                Err(error) => Err(ToolExecutionError::provider(format!(
                    "MCP tool '{}' request failed: {error}",
                    self.definition.name
                ))
                .with_source(error)),
            }
        })
    }
}

fn mcp_content_block_as_json(
    content: &ContentBlock,
) -> Result<ToolResultContent, ToolExecutionError> {
    serde_json::to_value(content)
        .map(ToolResultContent::json)
        .map_err(|error| {
            ToolExecutionError::provider(format!(
                "failed to preserve an MCP content block as JSON: {error}"
            ))
            .with_source(error)
        })
}

fn mcp_content_block_to_tool_content(
    content: &ContentBlock,
) -> Result<ToolResultContent, ToolExecutionError> {
    match content {
        ContentBlock::Text(text) => Ok(ToolResultContent::text(text.text.clone())),
        ContentBlock::Image(image) => match ImageMediaType::from_mime_type(&image.mime_type) {
            Some(media_type) => Ok(ToolResultContent::image_base64(
                image.data.clone(),
                Some(media_type),
                None,
            )),
            None => mcp_content_block_as_json(content),
        },
        ContentBlock::Resource(resource) => match &resource.resource {
            // Rig has no resource-content variant. Serializing the complete MCP
            // block keeps its URI, MIME type, metadata, annotations, and body
            // together instead of presenting only the body to the model.
            ResourceContents::TextResourceContents { .. } => mcp_content_block_as_json(content),
            ResourceContents::BlobResourceContents {
                mime_type, blob, ..
            } => match mime_type
                .as_deref()
                .and_then(ImageMediaType::from_mime_type)
            {
                Some(media_type) => Ok(ToolResultContent::image_base64(
                    blob.clone(),
                    Some(media_type),
                    None,
                )),
                _ => mcp_content_block_as_json(content),
            },
            _ => mcp_content_block_as_json(content),
        },
        ContentBlock::ResourceLink(_) | ContentBlock::Audio(_) => {
            mcp_content_block_as_json(content)
        }
        // ContentBlock is non-exhaustive. Preserve future protocol variants in
        // full rather than replacing them with a lossy placeholder.
        _ => mcp_content_block_as_json(content),
    }
}

/// Build the model presentation without flattening or reparsing MCP blocks.
fn mcp_result_output(result: &CallToolResult) -> Result<ToolOutput, ToolExecutionError> {
    let structured = result.structured_content.as_ref();
    let canonical_fallback = structured.map(serde_json::Value::to_string);
    let mut replaced_fallback = false;
    let mut mapped = Vec::with_capacity(result.content.len());

    for block in &result.content {
        let fallback_structured = if !replaced_fallback {
            match (block, canonical_fallback.as_deref(), structured) {
                (ContentBlock::Text(text), Some(fallback), Some(structured))
                    if text.text == fallback =>
                {
                    Some(structured)
                }
                _ => None,
            }
        } else {
            None
        };
        if let Some(structured) = fallback_structured {
            // rmcp's `structured`/`structured_error` constructors include this
            // text block solely for older clients. Replace it in place with the
            // typed value; do not duplicate it as model-visible text.
            mapped.push(ToolResultContent::json(structured.clone()));
            replaced_fallback = true;
        } else {
            mapped.push(mcp_content_block_to_tool_content(block)?);
        }
    }

    if let Some(structured) = structured
        && !replaced_fallback
    {
        // A server may provide genuine text/rich content in addition to its
        // structured result. Keep every real block and place the typed value
        // first deterministically; only the canonical compatibility text is
        // replaced rather than duplicated.
        mapped.insert(0, ToolResultContent::json(structured.clone()));
    }

    if !mapped.is_empty() {
        return ToolOutput::content(mapped);
    }

    // A content-less MCP result normalizes to one empty text block. This is
    // deliberately *not* what the native path does — a native tool returning an
    // empty `Vec<ToolResultContent>` gets an eager `ToolExecutionError`,
    // because that shape is the tool author's own type choice and fixable in
    // one read. An empty MCP result is protocol-legal and outside the caller's
    // control, so erroring here would fail tools the author cannot fix; the
    // empty block keeps the result sendable without inventing text.
    if result.is_error == Some(true) {
        Ok(ToolOutput::text("the MCP tool reported an error"))
    } else {
        Ok(ToolOutput::text(""))
    }
}

fn preserve_mcp_result(context: &mut ToolContext, result: &CallToolResult) {
    if let Some(structured) = result.structured_content.clone() {
        context.insert_result(structured);
    }
    if let Some(meta) = result.meta.clone() {
        context.insert_result(meta);
    }
    context.insert_result(result.clone());
}

impl ErasedTool for McpTool {
    fn name(&self) -> String {
        self.definition.name.to_string()
    }

    fn description(&self) -> String {
        self.definition
            .description
            .clone()
            .unwrap_or(Cow::from(""))
            .to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        self.definition.schema_as_json_value()
    }

    fn is_live(&self) -> bool {
        self.client.is_available()
    }

    fn execute<'a>(
        &'a self,
        args: String,
        context: &'a mut ToolContext,
    ) -> WasmBoxedFuture<'a, crate::tool::ToolExecution> {
        let meta = context.get::<RequestMetaObject>().cloned();
        Box::pin(async move {
            match self.execute_mcp(args, meta).await {
                Ok((CallToolResponse::Complete(result), _request)) => {
                    preserve_mcp_result(context, &result);
                    ToolExecution::complete(
                        match mcp_call_result(&result, Some(self.definition.name.as_ref())) {
                            Ok(result) => result,
                            Err(error) => ToolResult::failed(error),
                        },
                    )
                }
                Ok((CallToolResponse::Task(result), request)) => {
                    if let Some(meta) = result.meta.clone() {
                        context.insert_result(meta);
                    }
                    context.insert_result(result.clone());
                    match task_descriptor(
                        &self.client,
                        &result.task,
                        self.timeout,
                        request.meta.as_ref(),
                    ) {
                        Ok(descriptor) => ToolExecution::Deferred(descriptor),
                        Err(error) => ToolExecution::complete(ToolResult::failed(error)),
                    }
                }
                Ok((CallToolResponse::InputRequired(result), request)) => {
                    if let Some(meta) = result.meta.clone() {
                        context.insert_result(meta);
                    }
                    context.insert_result(result.clone());
                    match mrtr_descriptor(&self.client, &request, &result, 1, self.timeout) {
                        Ok(descriptor) => ToolExecution::Deferred(descriptor),
                        Err(error) => ToolExecution::complete(ToolResult::failed(error)),
                    }
                }
                Err(error) => ToolExecution::complete(ToolResult::failed(error)),
                Ok((_unsupported, _request)) => ToolExecution::complete(ToolResult::failed(
                    ToolExecutionError::provider(
                        "MCP tools/call returned an unsupported result type",
                    )
                    .with_retryable(false),
                )),
            }
        })
    }
}

/// Error type for [`McpClientHandler`] operations.
#[derive(Debug, thiserror::Error)]
pub enum McpClientError {
    /// Failed to establish the MCP connection or complete the handshake.
    #[error("MCP connection error: {0}")]
    ConnectionError(String),

    /// Failed to fetch the tool list from the MCP server.
    #[error("Failed to fetch MCP tool list: {0}")]
    ToolFetchError(#[from] rmcp::ServiceError),

    /// A post-connect Discover request failed.
    #[error("MCP server discovery failed: {0}")]
    DiscoveryError(String),

    /// The server did not finish returning its tool list before the deadline.
    #[error("Timed out fetching MCP tool list after {0:?}")]
    ToolFetchTimeout(Duration),

    /// A final-protocol tool-list page omitted its mandatory cache hints.
    #[error("MCP 2026-07-28 tools/list result omitted required ttlMs or cacheScope")]
    MissingToolListCacheHints,

    /// A final-protocol tool-list page did not identify itself as complete.
    #[error("MCP 2026-07-28 tools/list result did not use resultType 'complete'")]
    InvalidToolListResultType,

    /// The server did not negotiate Rig's required protocol version.
    #[error("MCP server does not support protocol version 2026-07-28")]
    UnsupportedProtocolVersion,

    /// Discover did not advertise the tools capability.
    #[error("MCP server Discover response does not advertise tool support")]
    ToolsNotSupported,

    /// The guard that owned this request handle has closed or been dropped.
    #[error("MCP client is unavailable")]
    Unavailable,

    /// Graceful shutdown failed.
    #[error("MCP client shutdown failed: {0}")]
    Shutdown(String),
}

#[derive(Default)]
struct ManagedToolsState {
    registrations: HashMap<String, ManagedToolToken>,
    committed_refresh: u64,
}

#[derive(Clone)]
struct ManagedRegistrations {
    state: Arc<RwLock<ManagedToolsState>>,
    tool_server: ToolServerHandle,
}

impl ManagedRegistrations {
    async fn remove_owned(&self) {
        let registrations = {
            let mut managed = self.state.write().await;
            std::mem::take(&mut managed.registrations)
        };
        self.tool_server
            .remove_managed_erased_tools(registrations)
            .await;
    }
}

/// Owns one active `subscriptions/listen` worker.
pub struct SubscriptionGuard {
    task: Option<tokio::task::JoinHandle<()>>,
}

impl SubscriptionGuard {
    fn new(task: tokio::task::JoinHandle<()>) -> Self {
        Self { task: Some(task) }
    }

    async fn cancel(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
            let _ = task.await;
        }
    }
}

impl Drop for SubscriptionGuard {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

/// Owns the complete lifetime of one modern MCP client connection.
///
/// Cloned [`McpRequestHandle`] values are deliberately non-owning. Closing or
/// dropping this guard makes them unavailable and removes only this guard's
/// managed tool registrations.
pub struct McpClientGuard {
    request_handle: McpRequestHandle,
    discovery: DiscoverResult,
    listener: Option<SubscriptionGuard>,
    registrations: ManagedRegistrations,
    running: Option<rmcp::service::RunningService<rmcp::service::RoleClient, McpClientHandler>>,
    state: Arc<McpClientState>,
    shutdown_timeout: Duration,
}

impl McpClientGuard {
    /// Obtain a lightweight handle for tools and deferred resolvers.
    pub fn request_handle(&self) -> McpRequestHandle {
        self.request_handle.clone()
    }

    /// The complete response retained from `server/discover`.
    pub fn discovery(&self) -> &DiscoverResult {
        &self.discovery
    }

    /// Build the resolver that reconstructs this client's serialized MCP Task
    /// and MRTR descriptors while the guard remains alive.
    pub fn deferred_resolver(&self) -> McpDeferredResolver {
        McpDeferredResolver {
            backend_id: self.state.deferred_backend_id.clone(),
            client: self.request_handle.clone(),
        }
    }

    /// Register this guard's deferred resolver in a runner registry.
    pub fn register_deferred_resolver(
        &self,
        registry: &DeferredToolResolverRegistry,
    ) -> Result<(), crate::tool::DeferredResolverError> {
        registry.register(self.deferred_resolver())
    }

    /// Gracefully stop listeners and the rmcp service, then remove the exact
    /// registrations still owned by this guard.
    pub async fn close(mut self) -> Result<(), McpClientError> {
        self.state.available.store(false, Ordering::Release);
        if let Some(mut listener) = self.listener.take() {
            listener.cancel().await;
        }
        self.registrations.remove_owned().await;
        if let Some(mut running) = self.running.take() {
            running
                .close_with_timeout(self.shutdown_timeout)
                .await
                .map_err(|error| McpClientError::Shutdown(error.to_string()))?;
        }
        Ok(())
    }

    /// Alias for [`Self::close`] for callers migrating from rmcp's running
    /// service ownership API.
    pub async fn cancel(self) -> Result<(), McpClientError> {
        self.close().await
    }
}

impl Drop for McpClientGuard {
    fn drop(&mut self) {
        self.state.available.store(false, Ordering::Release);
        self.listener.take();
        if let Some(running) = self.running.as_ref() {
            running.cancellation_token().cancel();
        }

        let registrations = self.registrations.clone();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                registrations.remove_owned().await;
            });
        }
    }
}

#[derive(Default)]
struct RefreshActivity {
    active: usize,
    dirty: bool,
}

const MAX_CONCURRENT_REFRESHES: usize = 2;

/// MCP client service used by [`McpClientGuard`].
///
/// Tool-list notifications are consumed by the guard-owned
/// `subscriptions/listen` worker rather than legacy handler callbacks.
///
/// # Usage
///
/// Use [`McpClientHandler::connect`] for a streamlined setup that handles
/// connection, initial tool fetch, and registration in one call:
///
/// ```rust,ignore
/// let tool_server_handle = ToolServer::new().run();
/// let config = McpClientConfig::new(client_identity);
/// let handler = McpClientHandler::new(config, tool_server_handle.clone());
/// let mcp_client = handler.connect(transport).await?;
/// ```
///
/// The returned guard owns the connection, listeners, and registrations.
#[derive(Clone)]
pub struct McpClientHandler {
    config: McpClientConfig,
    tool_server_handle: ToolServerHandle,
    /// Tracks the exact registry generation installed for each tool. Refreshes
    /// only mutate a name while this generation remains current, so a newer
    /// local or peer-handler registration cannot be deleted or overwritten.
    managed_tools: Arc<RwLock<ManagedToolsState>>,
    /// Bounds notification-driven list fetches and coalesces excess signals.
    refresh_activity: Arc<Mutex<RefreshActivity>>,
    /// Monotonic identity assigned when each tool-list fetch begins.
    next_refresh: Arc<AtomicU64>,
}

impl McpClientHandler {
    /// Create a modern MCP client service with explicit configuration.
    ///
    /// The `tool_server_handle` should be a clone of the handle used by the agent,
    /// so that tool updates are reflected in agent requests. Registered tools get
    /// [`DEFAULT_MCP_TOOL_TIMEOUT`]; change it through the configuration or
    /// [`McpClientHandler::with_timeout`].
    pub fn new(config: McpClientConfig, tool_server_handle: ToolServerHandle) -> Self {
        Self {
            config,
            tool_server_handle,
            managed_tools: Arc::new(RwLock::new(ManagedToolsState::default())),
            refresh_activity: Arc::new(Mutex::new(RefreshActivity::default())),
            next_refresh: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Set (or clear) the per-call timeout applied to every MCP tool this handler
    /// registers. Pass a [`Duration`] to bound calls, or `None` to disable.
    ///
    /// This applies the same setting to every tool managed by the handler.
    pub fn with_timeout(mut self, timeout: impl Into<Option<Duration>>) -> Self {
        self.config.request_timeouts.tool_call = timeout.into();
        self
    }

    /// Set the deadline for initial and list-changed tool-list fetches.
    pub fn with_refresh_timeout(mut self, timeout: Duration) -> Self {
        self.config.request_timeouts.catalog = timeout;
        self
    }

    /// Build the internal MCP adapter with this handler's configured timeout.
    fn build_tool(&self, tool: rmcp::model::Tool, client: McpRequestHandle) -> McpTool {
        McpTool::from_mcp_server(tool, client).with_timeout(self.config.request_timeouts.tool_call)
    }

    fn begin_refresh(&self) -> u64 {
        self.next_refresh.fetch_add(1, Ordering::SeqCst) + 1
    }

    async fn fetch_tools(
        &self,
        peer: &rmcp::service::ServerSink,
        request_handle: &McpRequestHandle,
    ) -> Result<Vec<Arc<dyn ErasedTool>>, McpClientError> {
        let refresh_timeout = self.config.request_timeouts.catalog;
        let deadline = tokio::time::Instant::now() + refresh_timeout;
        let mut tools = Vec::new();
        let mut cursor = None;
        let mut restarted_after_cursor_error = false;

        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return Err(McpClientError::ToolFetchTimeout(refresh_timeout));
            }
            let mut params = PaginatedRequestParams::default();
            params.cursor = cursor.clone();
            let page = match rig_core::wasm_compat::timeout(
                remaining,
                peer.list_tools(Some(params)),
            )
            .await
            {
                Ok(Ok(page)) => page,
                Ok(Err(_error)) if cursor.is_some() && !restarted_after_cursor_error => {
                    // rmcp invalidates every cached page after a cursor request
                    // fails. Restart once from the beginning so no page from
                    // the invalid cursor chain survives into this snapshot.
                    restarted_after_cursor_error = true;
                    cursor = None;
                    tools.clear();
                    continue;
                }
                Ok(Err(error)) => return Err(McpClientError::ToolFetchError(error)),
                Err(_) => {
                    return Err(McpClientError::ToolFetchTimeout(refresh_timeout));
                }
            };
            if page.ttl_ms.is_none() || page.cache_scope.is_none() {
                return Err(McpClientError::MissingToolListCacheHints);
            }
            if !page
                .result_type
                .as_ref()
                .is_some_and(rmcp::model::ResultType::is_complete)
            {
                return Err(McpClientError::InvalidToolListResultType);
            }
            tools.extend(page.tools);
            cursor = page.next_cursor;
            if cursor.is_none() {
                break;
            }
        }

        Ok(tools
            .into_iter()
            .map(|tool| {
                Arc::new(self.build_tool(tool, request_handle.clone())) as Arc<dyn ErasedTool>
            })
            .collect())
    }

    async fn try_start_refresh(&self) -> bool {
        let mut activity = self.refresh_activity.lock().await;
        if activity.active >= MAX_CONCURRENT_REFRESHES {
            activity.dirty = true;
            false
        } else {
            activity.active += 1;
            true
        }
    }

    async fn finish_or_restart_refresh(&self) -> bool {
        let mut activity = self.refresh_activity.lock().await;
        if activity.dirty {
            activity.dirty = false;
            true
        } else {
            activity.active -= 1;
            false
        }
    }

    async fn commit_initial(&self, refresh: u64, tools: Vec<Arc<dyn ErasedTool>>) {
        let mut managed = self.managed_tools.write().await;
        if refresh <= managed.committed_refresh {
            tracing::debug!(refresh, "discarding stale initial MCP tool list");
            return;
        }
        managed.registrations = self
            .tool_server_handle
            .add_managed_erased_tools(tools)
            .await;
        managed.committed_refresh = refresh;
    }

    async fn commit_refresh(&self, refresh: u64, tools: Vec<Arc<dyn ErasedTool>>) -> bool {
        let mut managed = self.managed_tools.write().await;
        if refresh <= managed.committed_refresh {
            tracing::debug!(refresh, "discarding stale MCP tool-list response");
            return false;
        }
        let expected = managed.registrations.clone();
        managed.registrations = self
            .tool_server_handle
            .reconcile_managed_erased_tools(expected, tools)
            .await;
        managed.committed_refresh = refresh;
        true
    }

    async fn refresh_from_subscription(
        &self,
        peer: &rmcp::service::ServerSink,
        request_handle: &McpRequestHandle,
    ) {
        if !self.try_start_refresh().await {
            return;
        }

        loop {
            let refresh = self.begin_refresh();
            match self.fetch_tools(peer, request_handle).await {
                Ok(tools) => {
                    if self.commit_refresh(refresh, tools).await {
                        let tool_count = self.managed_tools.read().await.registrations.len();
                        tracing::info!(tool_count, "MCP tool list refreshed successfully");
                    }
                }
                Err(error) => tracing::error!(%error, "failed to refresh MCP tool list"),
            }

            if !self.finish_or_restart_refresh().await {
                break;
            }
        }
    }

    /// Connect to an MCP server, fetch the initial tool list, and register
    /// all tools with the tool server.
    ///
    /// Returns a guard that owns the rmcp service, subscription worker, cache,
    /// and exact managed registrations.
    ///
    /// # Errors
    ///
    /// Returns [`McpClientError`] if the connection or initial tool fetch fails.
    pub async fn connect<T, E, A>(self, transport: T) -> Result<McpClientGuard, McpClientError>
    where
        T: rmcp::transport::IntoTransport<rmcp::service::RoleClient, E, A>,
        E: std::error::Error + Send + Sync + 'static,
    {
        let client_info = self.config.client_info();
        let service = self
            .serve_with_lifecycle(
                transport,
                ClientLifecycleMode::Discover {
                    preferred_versions: vec![ProtocolVersion::V_2026_07_28],
                },
            )
            .await
            .map_err(|e| McpClientError::ConnectionError(e.to_string()))?;

        let handler = service.service().clone();
        let Some(peer_info) = service.peer_info() else {
            return Err(McpClientError::UnsupportedProtocolVersion);
        };
        if peer_info.protocol_version != ProtocolVersion::V_2026_07_28 {
            return Err(McpClientError::UnsupportedProtocolVersion);
        }

        service
            .peer()
            .set_response_cache_config(handler.config.cache.clone())
            .await;
        let discovery_meta = RequestMetaObject::with_client_context(
            ProtocolVersion::V_2026_07_28,
            client_info.client_info,
            client_info.capabilities,
        );
        let discovery = service
            .peer()
            .discover(discovery_meta)
            .await
            .map_err(|error| McpClientError::DiscoveryError(error.to_string()))?;
        if !discovery
            .supported_versions
            .contains(&ProtocolVersion::V_2026_07_28)
        {
            return Err(McpClientError::UnsupportedProtocolVersion);
        }
        if discovery.capabilities.tools.is_none() {
            return Err(McpClientError::ToolsNotSupported);
        }

        let state = Arc::new(McpClientState {
            peer: service.peer().clone(),
            available: AtomicBool::new(true),
            deferred_backend_id: handler.config.deferred_backend_id.clone(),
            mrtr_max_rounds: handler.config.mrtr_max_rounds,
            task_notifications: Arc::new(TaskNotificationState::default()),
        });
        let request_handle = McpRequestHandle::from_state(&state);
        let refresh = handler.begin_refresh();
        let tools = handler.fetch_tools(service.peer(), &request_handle).await?;
        handler.commit_initial(refresh, tools).await;

        let listener = spawn_subscription_listener(
            service.peer().clone(),
            handler.clone(),
            request_handle.clone(),
            &discovery,
        );
        let registrations = ManagedRegistrations {
            state: handler.managed_tools.clone(),
            tool_server: handler.tool_server_handle.clone(),
        };
        let shutdown_timeout = handler.config.request_timeouts.shutdown;

        Ok(McpClientGuard {
            request_handle,
            discovery,
            listener,
            registrations,
            running: Some(service),
            state,
            shutdown_timeout,
        })
    }
}

fn spawn_subscription_listener(
    peer: rmcp::service::ServerSink,
    handler: McpClientHandler,
    request_handle: McpRequestHandle,
    discovery: &DiscoverResult,
) -> Option<SubscriptionGuard> {
    let policy = handler.config.subscription.clone();
    let supports_tool_changes = discovery
        .capabilities
        .tools
        .as_ref()
        .is_some_and(|tools| tools.list_changed == Some(true));
    let supports_tasks = discovery.capabilities.supports_tasks();
    if !policy.enabled || (!supports_tool_changes && !supports_tasks) {
        return None;
    }
    let task_notifications = request_handle
        .state()
        .ok()
        .map(|state| state.task_notifications.clone())?;

    let task = tokio::spawn(async move {
        let mut interest_changes = task_notifications.subscribe();
        loop {
            if peer.is_transport_closed() || !request_handle.is_available() {
                break;
            }

            let task_ids = if supports_tasks {
                task_notifications.task_ids()
            } else {
                Vec::new()
            };
            let mut filter = SubscriptionFilter::new();
            if supports_tool_changes {
                filter.tools_list_changed = Some(true);
            }
            if !task_ids.is_empty() {
                filter.task_ids = Some(task_ids);
            }
            if filter.tools_list_changed != Some(true) && filter.task_ids.is_none() {
                if interest_changes.changed().await.is_err() {
                    break;
                }
                continue;
            }

            let mut subscription = match peer
                .listen_with_capacity(filter, policy.channel_capacity)
                .await
            {
                Ok(subscription) => subscription,
                Err(error) => {
                    if peer.is_transport_closed() {
                        tracing::debug!(%error, "MCP subscription transport closed");
                        break;
                    }
                    tracing::warn!(%error, "MCP subscription listen failed; opening a new request");
                    tokio::time::sleep(policy.reconnect_delay).await;
                    continue;
                }
            };

            if subscription.acknowledged().tools_list_changed != Some(true)
                && subscription.acknowledged().task_ids.is_none()
            {
                tracing::info!(
                    "MCP server accepted none of Rig's notification filters; correctness remains polling and TTL-on-use"
                );
                if interest_changes.changed().await.is_err() {
                    break;
                }
                continue;
            }

            let mut filter_changed = false;
            loop {
                let next = tokio::select! {
                    changed = interest_changes.changed() => {
                        if changed.is_err() {
                            break;
                        }
                        filter_changed = true;
                        let _ = subscription
                            .cancel_with_reason(Some("subscription filter changed".to_owned()))
                            .await;
                        break;
                    }
                    next = subscription.next() => next,
                };
                match next {
                    Ok(Some(ServerNotification::ToolListChangedNotification(_))) => {
                        // The notification invalidates all cached tool pages.
                        // Clearing the peer cache is conservative because rmcp's
                        // targeted invalidator is intentionally crate-private.
                        peer.clear_response_cache().await;
                        let refresh_handler = handler.clone();
                        let refresh_peer = peer.clone();
                        let refresh_handle = request_handle.clone();
                        tokio::spawn(async move {
                            refresh_handler
                                .refresh_from_subscription(&refresh_peer, &refresh_handle)
                                .await;
                        });
                    }
                    Ok(Some(ServerNotification::TaskStatusNotification(notification))) => {
                        // This is only a wake-up hint. The task driver still
                        // performs tasks/get before exposing any new state.
                        task_notifications.wake(&notification.params.task.task.task_id);
                    }
                    Ok(Some(_)) => {
                        // rmcp already verifies the accepted subset and request
                        // ID; no other category was requested by this worker.
                    }
                    Ok(None) => break,
                    Err(error) => {
                        tracing::warn!(%error, "MCP subscription ended with a protocol error");
                        break;
                    }
                }
            }

            if filter_changed {
                continue;
            }

            match subscription.end() {
                Some(rmcp::service::SubscriptionEnd::Graceful(_)) => {
                    tracing::debug!("MCP subscription completed gracefully");
                    break;
                }
                Some(rmcp::service::SubscriptionEnd::Cancelled) => break,
                Some(rmcp::service::SubscriptionEnd::Abrupt)
                | Some(rmcp::service::SubscriptionEnd::Lagged { .. })
                | None => {
                    if peer.is_transport_closed() {
                        tracing::debug!("MCP subscription lost with its transport");
                        break;
                    }
                    tokio::time::sleep(policy.reconnect_delay).await;
                }
                Some(_) => {
                    tracing::warn!("MCP subscription ended in an unknown terminal state");
                    break;
                }
            }
        }
    });
    Some(SubscriptionGuard::new(task))
}

impl rmcp::handler::client::ClientHandler for McpClientHandler {
    fn get_info(&self) -> rmcp::model::ClientInfo {
        self.config.client_info()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        future::pending,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        time::Duration,
    };

    use rmcp::model::*;
    use rmcp::service::{RequestContext, SubscriptionContext};
    use rmcp::{RoleServer, ServerHandler, ServiceExt};
    use serde_json::json;
    use tokio::{
        sync::{Notify, RwLock},
        task::JoinHandle,
    };

    use super::*;
    use crate::tool::{
        ToolErrorKind,
        server::{ToolServer, ToolServerHandle},
    };
    use rig_core::message::ToolResultContent as RigToolResultContent;

    #[derive(Clone)]
    enum Scenario {
        Success,
        StructuredSuccess,
        StructuredOnly,
        Hang,
        ServiceError,
        ToolReportedError,
        ImageToolReportedError,
    }

    #[derive(Clone)]
    struct ScenarioServer {
        scenario: Scenario,
        seen: Arc<RwLock<Option<RequestMetaObject>>>,
        cancelled: Arc<Notify>,
    }

    impl ServerHandler for ScenarioServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
                .with_protocol_version(ProtocolVersion::V_2026_07_28)
                .with_server_info(Implementation::new("rig-mcp-test", "0.1.0"))
        }

        async fn call_tool(
            &self,
            _request: CallToolRequestParams,
            context: RequestContext<RoleServer>,
        ) -> Result<CallToolResponse, ErrorData> {
            *self.seen.write().await = Some(context.meta.clone());
            match self.scenario {
                Scenario::Success => {
                    Ok(CallToolResult::success(vec![ContentBlock::text("ok")]).into())
                }
                Scenario::StructuredSuccess => {
                    let mut response = CallToolResult::success(vec![
                        ContentBlock::text("before"),
                        ContentBlock::image("aGVsbG8=", "image/png"),
                        ContentBlock::text("after"),
                    ]);
                    response.structured_content = Some(json!({
                        "answer": 42,
                        "source": "fixture"
                    }));
                    let mut meta = MetaObject::new();
                    meta.0.insert("response-id".into(), json!("response-123"));
                    response.meta = Some(meta);
                    Ok(response.into())
                }
                Scenario::StructuredOnly => {
                    let mut response = CallToolResult::structured(json!({"answer": 42}));
                    response.content.clear();
                    Ok(response.into())
                }
                Scenario::Hang => {
                    context.ct.cancelled().await;
                    self.cancelled.notify_one();
                    Err(ErrorData::internal_error("fixture request cancelled", None))
                }
                Scenario::ServiceError => {
                    Err(ErrorData::internal_error("fixture service failed", None))
                }
                Scenario::ToolReportedError => Ok(CallToolResult::error(vec![ContentBlock::text(
                    "tool reported exact failure",
                )])
                .into()),
                Scenario::ImageToolReportedError => {
                    Ok(CallToolResult::error(vec![ContentBlock::image(
                        "ZXJyb3ItaW1hZ2U=",
                        "image/png",
                    )])
                    .into())
                }
            }
        }
    }

    struct Fixture {
        handle: ToolServerHandle,
        seen: Arc<RwLock<Option<RequestMetaObject>>>,
        cancelled: Arc<Notify>,
        _client: rmcp::service::RunningService<rmcp::service::RoleClient, ClientInfo>,
        _client_state: Arc<McpClientState>,
        server_task: JoinHandle<()>,
    }

    async fn fixture(scenario: Scenario, timeout: Option<Duration>) -> Fixture {
        let seen = Arc::new(RwLock::new(None));
        let cancelled = Arc::new(Notify::new());
        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);
        let server = ScenarioServer {
            scenario,
            seen: seen.clone(),
            cancelled: cancelled.clone(),
        };
        let server_task = tokio::spawn(async move {
            let running = server
                .serve((server_from_client, server_to_client))
                .await
                .expect("server start");
            running.waiting().await.expect("server error");
        });
        let client = ClientInfo::default()
            .serve((client_from_server, client_to_server))
            .await
            .expect("client connect");
        let (request_handle, client_state) = McpRequestHandle::for_test(client.peer().clone());
        let definition = Tool::new(
            "fixture_tool".to_string(),
            "fixture".to_string(),
            Arc::new(serde_json::Map::new()),
        );
        let handle = ToolServer::new()
            .rmcp_tool_with_timeout(definition, request_handle, timeout)
            .run();
        Fixture {
            handle,
            seen,
            cancelled,
            _client: client,
            _client_state: client_state,
            server_task,
        }
    }

    async fn execute(fixture: &Fixture, args: &str, context: &mut ToolContext) -> ToolResult {
        tokio::time::timeout(
            Duration::from_secs(5),
            fixture.handle.execute("fixture_tool", args, context),
        )
        .await
        .expect("MCP dispatch exceeded the outer safety timeout")
    }

    #[derive(Clone, Copy)]
    enum DeferredScenario {
        Task,
        TaskInput,
        TaskNotification,
        TaskCancel,
        TaskExpired,
        Mrtr,
    }

    #[derive(Clone)]
    struct DeferredScenarioServer {
        scenario: DeferredScenario,
        created_at: String,
        polls: Arc<std::sync::atomic::AtomicUsize>,
        input_updated: Arc<AtomicBool>,
        cancelled: Arc<AtomicBool>,
    }

    impl DeferredScenarioServer {
        fn new(scenario: DeferredScenario) -> Self {
            Self {
                scenario,
                created_at: match scenario {
                    DeferredScenario::TaskExpired => {
                        (chrono::Utc::now() - chrono::TimeDelta::seconds(10)).to_rfc3339()
                    }
                    _ => chrono::Utc::now().to_rfc3339(),
                },
                polls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                input_updated: Arc::new(AtomicBool::new(false)),
                cancelled: Arc::new(AtomicBool::new(false)),
            }
        }

        fn task(&self, status: TaskStatus) -> Task {
            Task::new(
                "opaque-task-id",
                status,
                self.created_at.clone(),
                chrono::Utc::now().to_rfc3339(),
            )
            .with_ttl_ms(match self.scenario {
                DeferredScenario::TaskExpired => 1,
                _ => 60_000,
            })
            .with_poll_interval_ms(match self.scenario {
                DeferredScenario::TaskNotification => 30_000,
                _ => 1,
            })
        }

        fn completed_task(&self) -> DetailedTask {
            let result = serde_json::to_value(CallToolResult::success(vec![ContentBlock::text(
                "deferred complete",
            )]))
            .expect("serialize call result")
            .as_object()
            .expect("call result object")
            .clone();
            DetailedTask::new(
                self.task(TaskStatus::Completed),
                TaskPayload::Completed { result },
            )
        }

        #[allow(deprecated)]
        fn input_requests() -> InputRequests {
            let request = InputRequest::Elicitation(ElicitRequest::new(
                ElicitRequestParams::FormElicitationParams {
                    meta: None,
                    message: "approve?".to_owned(),
                    requested_schema: serde_json::from_value(json!({
                        "type": "object",
                        "properties": { "approved": { "type": "boolean" } },
                        "required": ["approved"]
                    }))
                    .expect("elicitation schema"),
                },
            ));
            BTreeMap::from([("approval".to_owned(), request)])
        }
    }

    impl ServerHandler for DeferredScenarioServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(
                ServerCapabilities::builder()
                    .enable_tools()
                    .enable_tasks()
                    .build(),
            )
            .with_protocol_version(ProtocolVersion::V_2026_07_28)
            .with_server_info(Implementation::new("deferred-fixture", "0.1.0"))
        }

        async fn list_tools(
            &self,
            _: Option<PaginatedRequestParams>,
            _: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            Ok(ListToolsResult::with_all_items(vec![Tool::new(
                "fixture_tool".to_owned(),
                "fixture".to_owned(),
                Arc::new(serde_json::Map::new()),
            )])
            .with_ttl_ms(0)
            .with_cache_scope(CacheScope::Private))
        }

        async fn call_tool(
            &self,
            request: CallToolRequestParams,
            _: RequestContext<RoleServer>,
        ) -> Result<CallToolResponse, ErrorData> {
            match self.scenario {
                DeferredScenario::Mrtr if request.request_state.is_none() => {
                    Ok(InputRequiredResult::new(
                        Some(Self::input_requests()),
                        Some("opaque.request/state".to_owned()),
                    )
                    .into())
                }
                DeferredScenario::Mrtr => {
                    assert_eq!(
                        request.request_state.as_deref(),
                        Some("opaque.request/state")
                    );
                    assert!(
                        request
                            .input_responses
                            .as_ref()
                            .is_some_and(|responses| responses.contains_key("approval"))
                    );
                    Ok(CallToolResult::success(vec![ContentBlock::text("mrtr complete")]).into())
                }
                _ => Ok(CreateTaskResult::new(self.task(TaskStatus::Working)).into()),
            }
        }

        async fn get_task(
            &self,
            request: GetTaskParams,
            _: RequestContext<RoleServer>,
        ) -> Result<GetTaskResult, ErrorData> {
            assert_eq!(request.task_id, "opaque-task-id");
            let poll = self.polls.fetch_add(1, Ordering::SeqCst);
            let task = match self.scenario {
                DeferredScenario::TaskInput if !self.input_updated.load(Ordering::SeqCst) => {
                    DetailedTask::new(
                        self.task(TaskStatus::InputRequired),
                        TaskPayload::InputRequired {
                            input_requests: Self::input_requests(),
                        },
                    )
                }
                DeferredScenario::TaskCancel if self.cancelled.load(Ordering::SeqCst) => {
                    DetailedTask::new(self.task(TaskStatus::Cancelled), TaskPayload::Cancelled)
                }
                DeferredScenario::TaskCancel => {
                    DetailedTask::new(self.task(TaskStatus::Working), TaskPayload::Working)
                }
                DeferredScenario::TaskNotification | DeferredScenario::Task if poll == 0 => {
                    DetailedTask::new(self.task(TaskStatus::Working), TaskPayload::Working)
                }
                _ => self.completed_task(),
            };
            Ok(GetTaskResult::new(task))
        }

        async fn update_task(
            &self,
            request: UpdateTaskParams,
            _: RequestContext<RoleServer>,
        ) -> Result<(), ErrorData> {
            assert_eq!(request.task_id, "opaque-task-id");
            assert!(request.input_responses.contains_key("approval"));
            self.input_updated.store(true, Ordering::SeqCst);
            Ok(())
        }

        async fn cancel_task(
            &self,
            request: CancelTaskParams,
            _: RequestContext<RoleServer>,
        ) -> Result<(), ErrorData> {
            assert_eq!(request.task_id, "opaque-task-id");
            self.cancelled.store(true, Ordering::SeqCst);
            Ok(())
        }

        fn accepted_subscription_filter(
            &self,
            requested: &SubscriptionFilter,
        ) -> Option<SubscriptionFilter> {
            Some(requested.supported_by(&self.get_info().capabilities))
        }

        async fn listen(&self, context: SubscriptionContext) -> Result<(), ErrorData> {
            if matches!(self.scenario, DeferredScenario::TaskNotification) {
                context
                    .sink()
                    .notify_task_status(self.completed_task())
                    .await
                    .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
            }
            context.cancelled().await;
            Ok(())
        }
    }

    async fn deferred_fixture(
        scenario: DeferredScenario,
    ) -> (
        DeferredScenarioServer,
        ToolServerHandle,
        McpClientGuard,
        JoinHandle<()>,
    ) {
        let server = DeferredScenarioServer::new(scenario);
        let observable = server.clone();
        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);
        let server_task = tokio::spawn(async move {
            let running = server
                .serve((server_from_client, server_to_client))
                .await
                .expect("server start");
            let _ = running.waiting().await;
        });
        let tool_server = ToolServer::new().run();
        let mut config = McpClientConfig::new(Implementation::new("rig-test", "0.1.0"))
            .with_deferred_backend_id("mcp:test-deferred");
        if matches!(
            scenario,
            DeferredScenario::Mrtr | DeferredScenario::TaskInput
        ) {
            config = config.with_input_capabilities(
                McpInputCapabilities::default().with_elicitation(ElicitationCapability::default()),
            );
        }
        let guard = McpClientHandler::new(config, tool_server.clone())
            .connect((client_from_server, client_to_server))
            .await
            .expect("client connect");
        (observable, tool_server, guard, server_task)
    }

    async fn deferred_outcome(
        tool_server: &ToolServerHandle,
        guard: &McpClientGuard,
    ) -> (DeferredToolDescriptor, DeferredToolHandle, ToolContext) {
        let mut context = ToolContext::new();
        let outcome = tool_server
            .execute_with_outcome("fixture_tool", "{}", &mut context)
            .await;
        let ToolExecution::Deferred(descriptor) = outcome else {
            panic!("expected deferred execution");
        };
        let serialized = serde_json::to_string(&descriptor).expect("serialize descriptor");
        let restored: DeferredToolDescriptor =
            serde_json::from_str(&serialized).expect("restore descriptor");
        let registry = DeferredToolResolverRegistry::new();
        guard
            .register_deferred_resolver(&registry)
            .expect("register resolver");
        let handle = registry.resolve(&restored).await.expect("resolve deferred");
        (restored, handle, context)
    }

    #[tokio::test]
    async fn task_descriptor_round_trip_polls_to_completed_result() {
        let (server, tool_server, guard, server_task) =
            deferred_fixture(DeferredScenario::Task).await;
        let (_descriptor, handle, mut context) = deferred_outcome(&tool_server, &guard).await;

        assert!(matches!(
            handle.state().await.expect("first task state"),
            DeferredToolState::Working
        ));
        let DeferredToolState::Completed(result) =
            handle.state().await.expect("completed task state")
        else {
            panic!("expected completed task");
        };
        assert_eq!(result.output().as_text(), Some("deferred complete"));
        assert_eq!(server.polls.load(Ordering::SeqCst), 2);
        handle.publish_result_context(&mut context);
        assert!(context.result::<CallToolResult>().is_some());
        assert!(context.result::<GetTaskResult>().is_some());

        guard.close().await.expect("close guard");
        server_task.abort();
    }

    #[tokio::test]
    async fn task_input_uses_tasks_update_and_terminal_state_is_immutable() {
        let (server, tool_server, guard, server_task) =
            deferred_fixture(DeferredScenario::TaskInput).await;
        let (_descriptor, handle, _context) = deferred_outcome(&tool_server, &guard).await;

        let DeferredToolState::InputRequired(requests) =
            handle.state().await.expect("input-required task state")
        else {
            panic!("expected task input");
        };
        assert_eq!(requests.0[0].id, "approval");
        let state = handle
            .submit_input(RigInputResponses(vec![RigInputResponse {
                request_id: "approval".to_owned(),
                value: json!({"action": "accept", "content": {"approved": true}}),
            }]))
            .await
            .expect("submit task input");
        let DeferredToolState::Completed(result) = state else {
            panic!("expected completed task after input");
        };
        assert_eq!(result.output().as_text(), Some("deferred complete"));
        assert!(server.input_updated.load(Ordering::SeqCst));
        assert!(matches!(
            handle.cancel().await.expect("terminal cancellation read"),
            DeferredToolState::Completed(_)
        ));

        guard.close().await.expect("close guard");
        server_task.abort();
    }

    #[tokio::test]
    async fn task_cancellation_is_cooperative_and_confirmed_by_tasks_get() {
        let (server, tool_server, guard, server_task) =
            deferred_fixture(DeferredScenario::TaskCancel).await;
        let (_descriptor, handle, _context) = deferred_outcome(&tool_server, &guard).await;

        assert!(matches!(
            handle.state().await.expect("working task"),
            DeferredToolState::Working
        ));
        assert!(matches!(
            handle.cancel().await.expect("cancel task"),
            DeferredToolState::Cancelled
        ));
        assert!(server.cancelled.load(Ordering::SeqCst));
        assert!(matches!(
            handle.state().await.expect("immutable cancelled state"),
            DeferredToolState::Cancelled
        ));

        guard.close().await.expect("close guard");
        server_task.abort();
    }

    #[tokio::test]
    async fn task_ttl_expiry_stops_without_polling_discarded_state() {
        let (server, tool_server, guard, server_task) =
            deferred_fixture(DeferredScenario::TaskExpired).await;
        let (_descriptor, handle, _context) = deferred_outcome(&tool_server, &guard).await;

        let DeferredToolState::Failed(error) = handle.state().await.expect("expired task state")
        else {
            panic!("expected TTL failure");
        };
        assert_eq!(error.code(), Some("mcp_task_ttl_expired"));
        assert_eq!(server.polls.load(Ordering::SeqCst), 0);
        let DeferredToolState::Failed(again) = handle.state().await.expect("terminal TTL state")
        else {
            panic!("expected immutable TTL failure");
        };
        assert_eq!(again.code(), Some("mcp_task_ttl_expired"));

        guard.close().await.expect("close guard");
        server_task.abort();
    }

    #[tokio::test]
    async fn task_notification_wakes_polling_but_tasks_get_remains_authoritative() {
        let (server, tool_server, guard, server_task) =
            deferred_fixture(DeferredScenario::TaskNotification).await;
        let (_descriptor, handle, _context) = deferred_outcome(&tool_server, &guard).await;

        assert!(matches!(
            handle.state().await.expect("working task"),
            DeferredToolState::Working
        ));
        let state = tokio::time::timeout(Duration::from_secs(2), handle.state())
            .await
            .expect("task notification did not wake long poll")
            .expect("notified task state");
        assert!(matches!(state, DeferredToolState::Completed(_)));
        assert_eq!(
            server.polls.load(Ordering::SeqCst),
            2,
            "notification must wake a correctness-path tasks/get"
        );

        guard.close().await.expect("close guard");
        server_task.abort();
    }

    #[tokio::test]
    async fn direct_mrtr_preserves_request_state_and_uses_a_new_tool_request() {
        let (_server, tool_server, guard, server_task) =
            deferred_fixture(DeferredScenario::Mrtr).await;
        let (_descriptor, handle, mut context) = deferred_outcome(&tool_server, &guard).await;

        let DeferredToolState::InputRequired(requests) =
            handle.state().await.expect("MRTR input state")
        else {
            panic!("expected MRTR input");
        };
        assert_eq!(requests.0[0].kind, "elicitation");
        let state = handle
            .submit_input(RigInputResponses(vec![RigInputResponse {
                request_id: "approval".to_owned(),
                value: json!({"action": "accept", "content": {"approved": true}}),
            }]))
            .await
            .expect("submit MRTR input");
        let DeferredToolState::Completed(result) = state else {
            panic!("expected completed MRTR result");
        };
        assert_eq!(result.output().as_text(), Some("mrtr complete"));
        handle.publish_result_context(&mut context);
        assert!(context.result::<CallToolResult>().is_some());

        guard.close().await.expect("close guard");
        server_task.abort();
    }

    #[tokio::test]
    async fn best_effort_cancellation_drops_stalled_delivery_after_grace_period() {
        struct DropProbe(Arc<AtomicBool>);

        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let dropped = Arc::new(AtomicBool::new(false));
        let drop_probe = DropProbe(dropped.clone());
        let stalled = async move {
            let _drop_probe = drop_probe;
            pending::<Result<(), rmcp::ServiceError>>().await
        };

        tokio::time::timeout(
            Duration::from_secs(1),
            bounded_best_effort_cancellation(stalled, Duration::from_millis(10)),
        )
        .await
        .expect("best-effort cancellation exceeded its grace period");

        assert!(dropped.load(Ordering::SeqCst));
    }

    #[test]
    fn model_presentation_preserves_unrepresentable_mcp_blocks_as_json() {
        let blocks = vec![
            ContentBlock::resource(ResourceContents::TextResourceContents {
                uri: "file:///reports/summary.txt".to_string(),
                mime_type: Some("text/plain".to_string()),
                text: "full report".to_string(),
                meta: None,
            }),
            ContentBlock::resource(ResourceContents::BlobResourceContents {
                uri: "file:///reports/raw.bin".to_string(),
                mime_type: Some("application/octet-stream".to_string()),
                blob: "AAEC".to_string(),
                meta: None,
            }),
            ContentBlock::audio("UklGRg==", "audio/wav"),
            ContentBlock::resource_link(
                Resource::new("file:///reports/linked.txt", "linked.txt")
                    .with_mime_type("text/plain"),
            ),
            ContentBlock::image("YXZpZg==", "image/avif"),
            ContentBlock::resource(ResourceContents::BlobResourceContents {
                uri: "file:///images/chart.avif".to_string(),
                mime_type: Some("image/avif".to_string()),
                blob: "YmxvYi1hdmlm".to_string(),
                meta: None,
            }),
        ];
        let expected = blocks
            .iter()
            .map(|block| {
                RigToolResultContent::json(
                    serde_json::to_value(block).expect("MCP block is JSON serializable"),
                )
            })
            .collect::<Vec<_>>();

        let result = CallToolResult::success(blocks);
        let content = mcp_result_output(&result)
            .expect("MCP content mapping")
            .into_content()
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(content, expected);
        assert!(matches!(
            &content[0],
            RigToolResultContent::Json { value }
                if value["resource"]["uri"] == "file:///reports/summary.txt"
                    && value["resource"]["mimeType"] == "text/plain"
                    && value["resource"]["text"] == "full report"
        ));
        assert!(matches!(
            &content[1],
            RigToolResultContent::Json { value }
                if value["resource"]["uri"] == "file:///reports/raw.bin"
                    && value["resource"]["mimeType"] == "application/octet-stream"
                    && value["resource"]["blob"] == "AAEC"
        ));
        assert!(matches!(
            &content[2],
            RigToolResultContent::Json { value }
                if value["mimeType"] == "audio/wav" && value["data"] == "UklGRg=="
        ));
        assert!(matches!(
            &content[4],
            RigToolResultContent::Json { value }
                if value["mimeType"] == "image/avif" && value["data"] == "YXZpZg=="
        ));
        assert!(matches!(
            &content[5],
            RigToolResultContent::Json { value }
                if value["resource"]["uri"] == "file:///images/chart.avif"
                    && value["resource"]["mimeType"] == "image/avif"
                    && value["resource"]["blob"] == "YmxvYi1hdmlm"
        ));
    }

    #[test]
    fn image_resource_blob_maps_to_an_image_block() {
        let result = CallToolResult::success(vec![ContentBlock::resource(
            ResourceContents::BlobResourceContents {
                uri: "file:///images/chart.png".to_string(),
                mime_type: Some("image/png".to_string()),
                blob: "aW1hZ2U=".to_string(),
                meta: None,
            },
        )]);

        assert_eq!(
            mcp_result_output(&result).expect("MCP content mapping"),
            ToolOutput::one(RigToolResultContent::image_base64(
                "aW1hZ2U=",
                Some(ImageMediaType::PNG),
                None,
            ))
        );
    }

    #[test]
    fn string_valued_structured_content_remains_json() {
        let mut result = CallToolResult::structured(json!("forty-two"));
        result.content.clear();

        assert_eq!(
            mcp_result_output(&result).expect("MCP content mapping"),
            ToolOutput::json(json!("forty-two"))
        );
    }

    #[test]
    fn structured_constructors_replace_their_canonical_text_fallback() {
        let value = json!({"answer": 42});
        for result in [
            CallToolResult::structured(value.clone()),
            CallToolResult::structured_error(value.clone()),
        ] {
            assert_eq!(
                mcp_result_output(&result).expect("MCP structured output"),
                ToolOutput::json(value.clone())
            );
        }
    }

    #[test]
    fn structured_content_is_kept_alongside_real_rich_blocks() {
        let value = json!({"answer": 42});
        let mut result = CallToolResult::structured(value.clone());
        result
            .content
            .push(ContentBlock::image("aW1hZ2U=", "image/png"));
        result
            .content
            .push(ContentBlock::text("human-readable note"));

        let mut expected = vec![RigToolResultContent::json(value)];
        expected.push(RigToolResultContent::image_base64(
            "aW1hZ2U=",
            Some(ImageMediaType::PNG),
            None,
        ));
        expected.push(RigToolResultContent::text("human-readable note"));
        assert_eq!(
            mcp_result_output(&result).expect("MCP structured rich output"),
            ToolOutput::content(expected).expect("fixture content is non-empty")
        );
    }

    #[tokio::test]
    async fn canonical_dispatch_forwards_context_meta() {
        let fixture = fixture(Scenario::Success, Some(Duration::from_secs(1))).await;
        let mut meta = RequestMetaObject::new();
        meta.0.insert("authorization".into(), json!("Bearer test"));
        let mut context = ToolContext::new();
        context.insert(meta);

        let result = execute(&fixture, "{}", &mut context).await;
        assert!(result.is_success());
        assert_eq!(
            fixture
                .seen
                .read()
                .await
                .as_ref()
                .expect("server observed metadata")
                .0
                .get("authorization"),
            Some(&json!("Bearer test"))
        );
        fixture.server_task.abort();
    }

    #[test]
    fn per_call_meta_cannot_override_discover_owned_client_context() {
        let mut meta = RequestMetaObject::new();
        meta.insert(
            "io.modelcontextprotocol/protocolVersion".into(),
            json!("attacker-version"),
        );
        meta.insert(
            "io.modelcontextprotocol/clientInfo".into(),
            json!({"name": "attacker", "version": "0"}),
        );
        meta.insert(
            "io.modelcontextprotocol/clientCapabilities".into(),
            json!({"sampling": {}}),
        );
        meta.set_traceparent("00-0af7651916cd43dd8448eb211c80319c-00f067aa0ba902b7-01");

        let sanitized = without_reserved_client_meta(Some(meta)).expect("metadata remains");
        for key in RESERVED_CLIENT_META_KEYS {
            assert!(!sanitized.contains_key(key));
        }
        assert_eq!(
            sanitized.get_traceparent(),
            Some("00-0af7651916cd43dd8448eb211c80319c-00f067aa0ba902b7-01")
        );
    }

    #[tokio::test]
    async fn canonical_dispatch_classifies_timeout() {
        let fixture = fixture(Scenario::Hang, Some(Duration::from_millis(25))).await;
        let result = execute(&fixture, "{}", &mut ToolContext::new()).await;
        assert!(result.is_error_kind(ToolErrorKind::Timeout));
        assert_eq!(
            result.output().as_text(),
            Some("MCP tool 'fixture_tool' timed out after 25ms")
        );
        tokio::time::timeout(Duration::from_secs(1), fixture.cancelled.notified())
            .await
            .expect("the timed-out MCP request should be cancelled at the peer");
        fixture.server_task.abort();
    }

    #[tokio::test]
    async fn canonical_dispatch_classifies_service_error_and_preserves_source() {
        let fixture = fixture(Scenario::ServiceError, Some(Duration::from_secs(1))).await;
        let result = execute(&fixture, "{}", &mut ToolContext::new()).await;
        let error = result.error().expect("structured MCP service error");
        assert_eq!(error.kind(), ToolErrorKind::Provider);
        assert!(error.is::<rmcp::ServiceError>());
        assert!(error.message().contains("fixture service failed"));
        let output = result.output().render();
        assert!(output.contains("MCP tool 'fixture_tool' request failed"));
        assert!(output.contains("fixture service failed"));
        fixture.server_task.abort();
    }

    #[tokio::test]
    async fn canonical_dispatch_preserves_tool_reported_error_message() {
        let fixture = fixture(Scenario::ToolReportedError, Some(Duration::from_secs(1))).await;
        let result = execute(&fixture, "{}", &mut ToolContext::new()).await;
        assert!(result.is_error_kind(ToolErrorKind::Other));
        assert_eq!(
            result.output(),
            &ToolOutput::one(RigToolResultContent::text("tool reported exact failure"))
        );
        assert_eq!(
            result.error().map(ToolExecutionError::message),
            Some("MCP tool 'fixture_tool' reported an execution error")
        );
        fixture.server_task.abort();
    }

    #[tokio::test]
    async fn canonical_dispatch_preserves_non_text_tool_error_content() {
        let fixture = fixture(
            Scenario::ImageToolReportedError,
            Some(Duration::from_secs(1)),
        )
        .await;
        let mut context = ToolContext::new();
        let result = execute(&fixture, "{}", &mut context).await;

        assert!(result.is_error_kind(ToolErrorKind::Other));
        assert_eq!(
            result.output(),
            &ToolOutput::one(RigToolResultContent::image_base64(
                "ZXJyb3ItaW1hZ2U=",
                Some(ImageMediaType::PNG),
                None,
            ))
        );
        let raw = context
            .result::<CallToolResult>()
            .expect("raw MCP error result metadata");
        assert_eq!(raw.is_error, Some(true));
        assert!(matches!(raw.content.as_slice(), [ContentBlock::Image(_)]));
        fixture.server_task.abort();
    }

    #[tokio::test]
    async fn canonical_dispatch_preserves_ordered_content_and_response_metadata() {
        let fixture = fixture(Scenario::StructuredSuccess, Some(Duration::from_secs(1))).await;
        let mut context = ToolContext::new();
        let result = execute(&fixture, "{}", &mut context).await;

        let mut expected_content = vec![RigToolResultContent::json(json!({
            "answer": 42,
            "source": "fixture"
        }))];
        expected_content.push(RigToolResultContent::text("before"));
        expected_content.push(RigToolResultContent::image_base64(
            "aGVsbG8=",
            Some(ImageMediaType::PNG),
            None,
        ));
        expected_content.push(RigToolResultContent::text("after"));
        assert_eq!(
            result.output(),
            &ToolOutput::content(expected_content).expect("fixture content is non-empty")
        );

        let raw = context
            .result::<CallToolResult>()
            .expect("raw MCP result metadata");
        assert_eq!(raw.content.len(), 3);
        assert_eq!(
            raw.structured_content,
            Some(json!({"answer": 42, "source": "fixture"}))
        );
        assert_eq!(
            context.result::<serde_json::Value>(),
            Some(&json!({"answer": 42, "source": "fixture"}))
        );
        assert_eq!(
            context
                .result::<MetaObject>()
                .and_then(|meta| meta.0.get("response-id")),
            Some(&json!("response-123"))
        );
        fixture.server_task.abort();
    }

    #[tokio::test]
    async fn canonical_dispatch_uses_structured_content_when_blocks_are_empty() {
        let fixture = fixture(Scenario::StructuredOnly, Some(Duration::from_secs(1))).await;
        let mut context = ToolContext::new();
        let result = execute(&fixture, "{}", &mut context).await;

        assert_eq!(result.output(), &ToolOutput::json(json!({"answer": 42})));
        assert_eq!(
            context.result::<serde_json::Value>(),
            Some(&json!({"answer": 42}))
        );
        fixture.server_task.abort();
    }

    #[tokio::test]
    async fn canonical_dispatch_classifies_invalid_json_and_preserves_source() {
        let fixture = fixture(Scenario::Success, Some(Duration::from_secs(1))).await;
        let result = execute(&fixture, "{", &mut ToolContext::new()).await;
        let error = result.error().expect("structured argument error");
        assert_eq!(error.kind(), ToolErrorKind::InvalidArgs);
        assert!(matches!(
            error.downcast_ref::<McpArgumentError>(),
            Some(McpArgumentError::Json(_))
        ));
        let output = result.output().render();
        assert!(output.contains("MCP tool 'fixture_tool' received invalid arguments"));
        assert!(output.contains("invalid JSON"));
        fixture.server_task.abort();
    }

    #[tokio::test]
    async fn canonical_dispatch_rejects_non_object_arguments() {
        let fixture = fixture(Scenario::Success, Some(Duration::from_secs(1))).await;
        for args in [r#"[1,2]"#, r#""text""#, "7", "true"] {
            let result = execute(&fixture, args, &mut ToolContext::new()).await;
            assert!(
                result.is_error_kind(ToolErrorKind::InvalidArgs),
                "{args} must not be coerced into an argument-less MCP call"
            );
        }

        // Empty input and explicit null remain the documented no-argument forms.
        for args in ["", "null"] {
            let result = execute(&fixture, args, &mut ToolContext::new()).await;
            assert!(
                result.is_success(),
                "{args:?} should remain a no-argument call"
            );
        }
        fixture.server_task.abort();
    }
}

#[cfg(test)]
mod migrated_tests {
    use super::{
        MAX_CONCURRENT_REFRESHES, McpClientConfig, McpClientError, McpClientGuard,
        McpClientHandler, McpRequestHandle,
    };
    use crate::tool::{DynamicTool, ToolOutput, server::ToolServer};
    use rmcp::{
        RoleServer, ServerHandler, ServiceExt,
        handler::client::ClientHandler,
        model::*,
        service::{RequestContext, SubscriptionContext},
    };
    use std::{
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };
    use tokio::sync::{Notify, RwLock};

    fn test_config() -> McpClientConfig {
        McpClientConfig::new(Implementation::new("rig-mcp-test-client", "0.1.0"))
    }

    #[derive(Clone)]
    struct DynamicToolServer {
        tools: Arc<RwLock<Vec<Tool>>>,
        changed: Arc<Notify>,
    }
    impl DynamicToolServer {
        fn new(tools: Vec<Tool>) -> Self {
            Self {
                tools: Arc::new(RwLock::new(tools)),
                changed: Arc::new(Notify::new()),
            }
        }
        async fn set_tools(&self, tools: Vec<Tool>) {
            *self.tools.write().await = tools;
            self.changed.notify_one();
        }
    }
    impl ServerHandler for DynamicToolServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(
                ServerCapabilities::builder()
                    .enable_tools()
                    .enable_tool_list_changed()
                    .build(),
            )
            .with_protocol_version(ProtocolVersion::V_2026_07_28)
            .with_server_info(Implementation::new("test-dynamic-server", "0.1.0"))
        }
        async fn list_tools(
            &self,
            _: Option<PaginatedRequestParams>,
            _: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            Ok(
                ListToolsResult::with_all_items(self.tools.read().await.clone())
                    .with_ttl_ms(0)
                    .with_cache_scope(CacheScope::Private),
            )
        }
        async fn call_tool(
            &self,
            request: CallToolRequestParams,
            _: RequestContext<RoleServer>,
        ) -> Result<CallToolResponse, ErrorData> {
            Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "called {}",
                request.name
            ))])
            .into())
        }

        fn accepted_subscription_filter(
            &self,
            requested: &SubscriptionFilter,
        ) -> Option<SubscriptionFilter> {
            Some(requested.supported_by(&self.get_info().capabilities))
        }

        async fn listen(&self, context: SubscriptionContext) -> Result<(), ErrorData> {
            loop {
                tokio::select! {
                    _ = context.cancelled() => return Ok(()),
                    _ = self.changed.notified() => {
                        context
                            .sink()
                            .notify_tool_list_changed()
                            .await
                            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
                    }
                }
            }
        }
    }

    #[derive(Clone)]
    struct OrderedRefreshServer {
        tools: Arc<RwLock<Vec<Tool>>>,
        list_calls: Arc<AtomicUsize>,
        first_refresh_started: Arc<Notify>,
        release_first_refresh: Arc<Notify>,
        first_refresh_returned: Arc<Notify>,
        changed: Arc<Notify>,
    }

    impl OrderedRefreshServer {
        fn new(tools: Vec<Tool>) -> Self {
            Self {
                tools: Arc::new(RwLock::new(tools)),
                list_calls: Arc::new(AtomicUsize::new(0)),
                first_refresh_started: Arc::new(Notify::new()),
                release_first_refresh: Arc::new(Notify::new()),
                first_refresh_returned: Arc::new(Notify::new()),
                changed: Arc::new(Notify::new()),
            }
        }

        async fn set_tools(&self, tools: Vec<Tool>) {
            *self.tools.write().await = tools;
            self.changed.notify_one();
        }

        fn notify_changed(&self) {
            self.changed.notify_one();
        }
    }

    impl ServerHandler for OrderedRefreshServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(
                ServerCapabilities::builder()
                    .enable_tools()
                    .enable_tool_list_changed()
                    .build(),
            )
            .with_protocol_version(ProtocolVersion::V_2026_07_28)
            .with_server_info(Implementation::new("test-ordered-refresh-server", "0.1.0"))
        }

        async fn list_tools(
            &self,
            _: Option<PaginatedRequestParams>,
            _: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            let call = self.list_calls.fetch_add(1, Ordering::SeqCst);
            let tools = self.tools.read().await.clone();

            // Call zero is connect's initial fetch. Hold the first notification's
            // stale snapshot so a second notification is concurrent with it.
            if call == 1 {
                self.first_refresh_started.notify_one();
                self.release_first_refresh.notified().await;
                self.first_refresh_returned.notify_one();
            }

            Ok(ListToolsResult::with_all_items(tools)
                .with_ttl_ms(0)
                .with_cache_scope(CacheScope::Private))
        }

        fn accepted_subscription_filter(
            &self,
            requested: &SubscriptionFilter,
        ) -> Option<SubscriptionFilter> {
            Some(requested.supported_by(&self.get_info().capabilities))
        }

        async fn listen(&self, context: SubscriptionContext) -> Result<(), ErrorData> {
            loop {
                tokio::select! {
                    _ = context.cancelled() => return Ok(()),
                    _ = self.changed.notified() => {
                        context
                            .sink()
                            .notify_tool_list_changed()
                            .await
                            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
                    }
                }
            }
        }
    }

    #[derive(Clone)]
    struct HangingListServer;

    impl ServerHandler for HangingListServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
                .with_protocol_version(ProtocolVersion::V_2026_07_28)
                .with_server_info(Implementation::new("test-hanging-list-server", "0.1.0"))
        }

        async fn list_tools(
            &self,
            _: Option<PaginatedRequestParams>,
            _: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            std::future::pending().await
        }
    }

    #[derive(Clone)]
    struct MissingCacheHintsServer;

    impl ServerHandler for MissingCacheHintsServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
                .with_protocol_version(ProtocolVersion::V_2026_07_28)
                .with_server_info(Implementation::new(
                    "test-missing-cache-hints-server",
                    "0.1.0",
                ))
        }

        async fn list_tools(
            &self,
            _: Option<PaginatedRequestParams>,
            _: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            Ok(ListToolsResult::with_all_items(vec![make_tool(
                "uncached",
                "Missing mandatory hints",
            )]))
        }
    }

    #[derive(Clone, Default)]
    struct PaginatedCatalogServer {
        seen_cursors: Arc<RwLock<Vec<Option<String>>>>,
    }

    impl ServerHandler for PaginatedCatalogServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
                .with_protocol_version(ProtocolVersion::V_2026_07_28)
                .with_server_info(Implementation::new(
                    "test-paginated-catalog-server",
                    "0.1.0",
                ))
        }

        async fn list_tools(
            &self,
            request: Option<PaginatedRequestParams>,
            _: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            let cursor = request.and_then(|request| request.cursor);
            self.seen_cursors.write().await.push(cursor.clone());
            let mut result = match cursor.as_deref() {
                None => ListToolsResult::with_all_items(vec![
                    make_tool("tool_b", "First server entry"),
                    make_tool("tool_a", "Second server entry"),
                ]),
                Some("page-2") => {
                    ListToolsResult::with_all_items(vec![make_tool("tool_c", "Third server entry")])
                }
                Some(other) => {
                    return Err(ErrorData::invalid_params(
                        format!("invalid cursor: {other}"),
                        None,
                    ));
                }
            };
            if cursor.is_none() {
                result.next_cursor = Some("page-2".to_owned());
            }
            Ok(result
                .with_ttl_ms(30_000)
                .with_cache_scope(CacheScope::Private))
        }
    }

    #[derive(Clone, Default)]
    struct InvalidatingCursorServer {
        list_calls: Arc<AtomicUsize>,
        seen_cursors: Arc<RwLock<Vec<Option<String>>>>,
    }

    impl ServerHandler for InvalidatingCursorServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
                .with_protocol_version(ProtocolVersion::V_2026_07_28)
                .with_server_info(Implementation::new(
                    "test-invalidating-cursor-server",
                    "0.1.0",
                ))
        }

        async fn list_tools(
            &self,
            request: Option<PaginatedRequestParams>,
            _: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            let cursor = request.and_then(|request| request.cursor);
            self.seen_cursors.write().await.push(cursor.clone());
            let call = self.list_calls.fetch_add(1, Ordering::SeqCst);
            let mut result = match (call, cursor.as_deref()) {
                (0, None) => {
                    ListToolsResult::with_all_items(vec![make_tool("stale", "Stale first page")])
                }
                (1, Some("expired")) => {
                    return Err(ErrorData::invalid_params("expired cursor", None));
                }
                (2, None) => {
                    ListToolsResult::with_all_items(vec![make_tool("fresh_a", "Fresh first page")])
                }
                (3, Some("fresh-page-2")) => {
                    ListToolsResult::with_all_items(vec![make_tool("fresh_b", "Fresh second page")])
                }
                _ => return Err(ErrorData::invalid_params("unexpected cursor chain", None)),
            };
            match call {
                0 => result.next_cursor = Some("expired".to_owned()),
                2 => result.next_cursor = Some("fresh-page-2".to_owned()),
                _ => {}
            }
            Ok(result
                .with_ttl_ms(30_000)
                .with_cache_scope(CacheScope::Private))
        }
    }

    fn make_tool(name: &str, description: &str) -> Tool {
        Tool::new(
            name.to_string(),
            description.to_string(),
            Arc::new(serde_json::Map::new()),
        )
    }

    fn make_dynamic_tool(name: &str, description: &str) -> DynamicTool {
        DynamicTool::new(
            name,
            description,
            serde_json::json!({"type": "object", "properties": {}}),
            |_context, _args| Box::pin(async { Ok(ToolOutput::text("local")) }),
        )
    }

    async fn connect<S>(
        server: S,
        handle: crate::tool::server::ToolServerHandle,
    ) -> (
        McpClientGuard,
        tokio::task::JoinHandle<rmcp::service::RunningService<rmcp::RoleServer, S>>,
    )
    where
        S: ServerHandler,
    {
        let (c2s, sfc) = tokio::io::duplex(8192);
        let (s2c, cfs) = tokio::io::duplex(8192);
        let server_task =
            tokio::spawn(async move { server.serve((sfc, s2c)).await.expect("server start") });
        let service = McpClientHandler::new(test_config(), handle)
            .connect((cfs, c2s))
            .await
            .expect("connect");
        (service, server_task)
    }

    #[tokio::test]
    async fn client_handler_registers_initial_tools() {
        let server = DynamicToolServer::new(vec![
            make_tool("tool_a", "First"),
            make_tool("tool_b", "Second"),
        ]);
        let handle = ToolServer::new().run();
        let (client, task) = connect(server, handle.clone()).await;
        let defs = handle.get_tool_defs(None).await.unwrap();
        assert_eq!(
            defs.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(),
            vec!["tool_a", "tool_b"]
        );
        client.cancel().await.unwrap();
        task.abort();
    }

    #[tokio::test]
    async fn final_catalog_requires_cache_hints_on_every_page() {
        let (c2s, sfc) = tokio::io::duplex(8192);
        let (s2c, cfs) = tokio::io::duplex(8192);
        let server_task = tokio::spawn(async move {
            MissingCacheHintsServer
                .serve((sfc, s2c))
                .await
                .expect("server start")
        });

        let result = McpClientHandler::new(test_config(), ToolServer::new().run())
            .connect((cfs, c2s))
            .await;
        assert!(matches!(
            result,
            Err(McpClientError::MissingToolListCacheHints)
        ));
        server_task.abort();
    }

    #[tokio::test]
    async fn paginated_catalog_preserves_server_order_and_cursor_chain() {
        let server = PaginatedCatalogServer::default();
        let seen_cursors = server.seen_cursors.clone();
        let handle = ToolServer::new().run();
        let (client, server_task) = connect(server, handle.clone()).await;

        let definitions = handle.get_tool_defs(None).await.expect("tool definitions");
        assert_eq!(
            definitions
                .iter()
                .map(|definition| definition.name.as_str())
                .collect::<Vec<_>>(),
            vec!["tool_b", "tool_a", "tool_c"]
        );
        assert_eq!(
            *seen_cursors.read().await,
            vec![None, Some("page-2".to_owned())]
        );

        client.close().await.expect("client close");
        server_task.abort();
    }

    #[tokio::test]
    async fn invalid_cursor_discards_cached_pages_and_restarts_from_the_beginning() {
        let server = InvalidatingCursorServer::default();
        let seen_cursors = server.seen_cursors.clone();
        let handle = ToolServer::new().run();
        let (client, server_task) = connect(server, handle.clone()).await;

        let definitions = handle.get_tool_defs(None).await.expect("tool definitions");
        assert_eq!(
            definitions
                .iter()
                .map(|definition| definition.name.as_str())
                .collect::<Vec<_>>(),
            vec!["fresh_a", "fresh_b"]
        );
        assert_eq!(
            *seen_cursors.read().await,
            vec![
                None,
                Some("expired".to_owned()),
                None,
                Some("fresh-page-2".to_owned()),
            ]
        );

        client.close().await.expect("client close");
        server_task.abort();
    }

    #[tokio::test]
    async fn modern_guard_retains_discovery_and_owns_tool_lifetime() {
        let server = DynamicToolServer::new(vec![make_tool("tool_a", "First")]);
        let handle = ToolServer::new().run();
        let (client, server_task) = connect(server, handle.clone()).await;

        assert!(
            client
                .discovery()
                .supported_versions
                .contains(&ProtocolVersion::V_2026_07_28)
        );
        assert!(client.discovery().capabilities.tools.is_some());
        assert_eq!(
            client
                .discovery()
                .server_info()
                .expect("server identity")
                .name,
            "test-dynamic-server"
        );
        let request_handle = client.request_handle();
        assert!(request_handle.is_available());

        client.close().await.expect("guard shutdown");
        assert!(!request_handle.is_available());
        assert!(handle.get_tool_defs(None).await.unwrap().is_empty());
        server_task.abort();
    }

    #[test]
    fn modern_config_is_exact_and_advertises_tasks_only_by_default() {
        let config = test_config();
        assert_eq!(config.protocol_version(), ProtocolVersion::V_2026_07_28);
        let capabilities = config.client_capabilities();
        assert!(capabilities.supports_tasks());
        assert!(capabilities.elicitation.is_none());
        assert!(capabilities.sampling.is_none());
        assert!(capabilities.roots.is_none());
    }

    #[tokio::test]
    async fn disconnected_handler_tools_are_retired_on_snapshot() {
        let server = DynamicToolServer::new(vec![make_tool("tool_a", "First")]);
        let handle = ToolServer::new().run();
        let (client, task) = connect(server, handle.clone()).await;
        assert_eq!(handle.get_tool_defs(None).await.unwrap().len(), 1);

        client.cancel().await.unwrap();

        let defs = handle.get_tool_defs(None).await.unwrap();
        assert!(
            defs.is_empty(),
            "a disconnected sole owner must not remain provider-visible"
        );
        task.abort();
    }

    #[tokio::test]
    async fn disconnected_handler_tools_are_retired_on_direct_dispatch() {
        let server = DynamicToolServer::new(vec![make_tool("tool_a", "First")]);
        let handle = ToolServer::new().run();
        let (client, task) = connect(server, handle.clone()).await;
        assert_eq!(handle.get_tool_defs(None).await.unwrap().len(), 1);

        client.cancel().await.unwrap();

        let result = handle
            .execute("tool_a", "{}", &mut crate::tool::ToolContext::new())
            .await;
        assert_eq!(
            result.error().expect("disconnected tool must fail").kind(),
            crate::tool::ToolErrorKind::NotFound
        );
        task.abort();
    }

    #[tokio::test]
    async fn initial_tool_fetch_is_bounded_by_the_refresh_timeout() {
        let (c2s, sfc) = tokio::io::duplex(8192);
        let (s2c, cfs) = tokio::io::duplex(8192);
        let server_task = tokio::spawn(async move {
            HangingListServer
                .serve((sfc, s2c))
                .await
                .expect("server start")
        });
        let refresh_timeout = Duration::from_millis(25);
        let result = McpClientHandler::new(test_config(), ToolServer::new().run())
            .with_refresh_timeout(refresh_timeout)
            .connect((cfs, c2s))
            .await;

        assert!(matches!(
            result,
            Err(McpClientError::ToolFetchTimeout(timeout)) if timeout == refresh_timeout
        ));
        server_task.abort();
    }

    #[tokio::test]
    async fn refresh_activity_is_bounded_and_coalesces_excess_notifications() {
        let handler = McpClientHandler::new(test_config(), ToolServer::new().run());

        assert!(handler.try_start_refresh().await);
        assert!(handler.try_start_refresh().await);
        assert!(!handler.try_start_refresh().await);
        {
            let activity = handler.refresh_activity.lock().await;
            assert_eq!(activity.active, MAX_CONCURRENT_REFRESHES);
            assert!(activity.dirty);
        }

        assert!(handler.finish_or_restart_refresh().await);
        assert!(!handler.finish_or_restart_refresh().await);
        assert!(!handler.finish_or_restart_refresh().await);
        let activity = handler.refresh_activity.lock().await;
        assert_eq!(activity.active, 0);
        assert!(!activity.dirty);
    }

    #[tokio::test]
    async fn client_handler_refreshes_on_tool_list_changed() {
        let server = DynamicToolServer::new(vec![make_tool("alpha", "Alpha")]);
        let handle = ToolServer::new().run();
        let (c2s, sfc) = tokio::io::duplex(8192);
        let (s2c, cfs) = tokio::io::duplex(8192);
        let copy = server.clone();
        let task = tokio::spawn(async move { copy.serve((sfc, s2c)).await.expect("server start") });
        let client = McpClientHandler::new(test_config(), handle.clone())
            .connect((cfs, c2s))
            .await
            .unwrap();
        assert_eq!(handle.get_tool_defs(None).await.unwrap()[0].name, "alpha");
        server
            .set_tools(vec![make_tool("beta", "Beta"), make_tool("gamma", "Gamma")])
            .await;
        let _running = task.await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let defs = handle.get_tool_defs(None).await.unwrap();
                if defs.len() == 2 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("refresh");
        let names = handle
            .get_tool_defs(None)
            .await
            .unwrap()
            .into_iter()
            .map(|d| d.name)
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["beta", "gamma"]);
        client.cancel().await.unwrap();
    }

    #[tokio::test]
    async fn concurrent_refreshes_cannot_roll_back_a_newer_tool_list() {
        let server = OrderedRefreshServer::new(vec![make_tool("stale", "Stale snapshot")]);
        let server_control = server.clone();
        let handle = ToolServer::new().run();
        let (client, server_task) = connect(server, handle.clone()).await;
        let _running_server = server_task.await.unwrap();

        server_control.notify_changed();
        tokio::time::timeout(
            Duration::from_secs(2),
            server_control.first_refresh_started.notified(),
        )
        .await
        .expect("first refresh fetch started");

        assert!(
            client.registrations.state.try_write().is_ok(),
            "a hung network fetch must not hold the managed-registry lock"
        );

        server_control
            .set_tools(vec![make_tool("newest", "Newest snapshot")])
            .await;

        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let defs = handle.get_tool_defs(None).await.unwrap();
                if defs.len() == 1 && defs[0].name == "newest" {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("newest refresh committed while the older fetch remained hung");

        // Let the stale response arrive after the newer snapshot committed. Its
        // lower refresh version must be discarded rather than rolling back.
        server_control.release_first_refresh.notify_one();
        tokio::time::timeout(
            Duration::from_secs(2),
            server_control.first_refresh_returned.notified(),
        )
        .await
        .expect("delayed refresh response returned");
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        let defs = handle.get_tool_defs(None).await.unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "newest");

        assert_eq!(server_control.list_calls.load(Ordering::SeqCst), 3);
        client.cancel().await.unwrap();
    }

    #[tokio::test]
    async fn refresh_rebuilds_owned_tools_in_latest_server_order() {
        let server =
            DynamicToolServer::new(vec![make_tool("alpha", "Alpha"), make_tool("beta", "Beta")]);
        let server_control = server.clone();
        let handle = ToolServer::new().run();
        let (client, server_task) = connect(server, handle.clone()).await;
        server_control
            .set_tools(vec![
                make_tool("beta", "Beta refreshed"),
                make_tool("gamma", "Gamma"),
                make_tool("alpha", "Alpha refreshed"),
            ])
            .await;
        let _running_server = server_task.await.unwrap();

        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let defs = handle.get_tool_defs(None).await.unwrap();
                let names = defs
                    .iter()
                    .map(|definition| definition.name.as_str())
                    .collect::<Vec<_>>();
                if names == ["beta", "gamma", "alpha"] && defs[0].description == "Beta refreshed" {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("latest MCP order committed");
        client.cancel().await.unwrap();
    }

    #[tokio::test]
    async fn one_refresh_reclaims_a_name_after_a_peer_owner_disappears() {
        let handle = ToolServer::new().run();
        let first_server = DynamicToolServer::new(vec![make_tool("shared", "First owner")]);
        let first_control = first_server.clone();
        let (first_client, first_server_task) = connect(first_server, handle.clone()).await;
        let _first_running_server = first_server_task.await.unwrap();

        let second_server = DynamicToolServer::new(vec![make_tool("shared", "Second owner")]);
        let second_control = second_server.clone();
        let (second_client, second_server_task) = connect(second_server, handle.clone()).await;
        let _second_running_server = second_server_task.await.unwrap();
        assert_eq!(
            handle.get_tool_defs(None).await.unwrap()[0].description,
            "Second owner"
        );

        second_control.set_tools(Vec::new()).await;
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if handle.get_tool_defs(None).await.unwrap().is_empty() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("second owner removed its registration");

        // The first handler still has a stale generation token for `shared`.
        // One full-list refresh must reclaim the now-empty slot rather than
        // requiring a second notification to converge.
        first_control
            .set_tools(vec![make_tool("shared", "First owner refreshed")])
            .await;
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let defs = handle.get_tool_defs(None).await.unwrap();
                if defs.len() == 1 && defs[0].description == "First owner refreshed" {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("one refresh reclaimed the empty slot");

        second_client.cancel().await.unwrap();
        first_client.cancel().await.unwrap();
    }

    #[tokio::test]
    async fn refresh_does_not_replace_a_newer_local_registration() {
        let server = DynamicToolServer::new(vec![make_tool("alpha", "MCP alpha")]);
        let server_control = server.clone();
        let handle = ToolServer::new().run();
        let (client, server_task) = connect(server, handle.clone()).await;

        handle
            .add_dynamic_tool(make_dynamic_tool("alpha", "Local alpha"))
            .await;
        server_control
            .set_tools(vec![make_tool("refresh_complete", "Refresh sentinel")])
            .await;
        let _running_server = server_task.await.unwrap();

        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let defs = handle.get_tool_defs(None).await.unwrap();
                if defs
                    .iter()
                    .any(|definition| definition.name == "refresh_complete")
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("MCP refresh completed");

        let defs = handle.get_tool_defs(None).await.unwrap();
        let alpha = defs
            .iter()
            .find(|definition| definition.name == "alpha")
            .expect("alpha remains registered");
        assert_eq!(alpha.description, "Local alpha");

        let result = handle
            .execute("alpha", "{}", &mut crate::tool::ToolContext::new())
            .await;
        assert_eq!(result.output(), &ToolOutput::text("local"));
        client.cancel().await.unwrap();
    }

    #[tokio::test]
    async fn one_handler_refresh_protects_live_peer_and_reclaims_after_disconnect() {
        let server_a = DynamicToolServer::new(vec![make_tool("alpha", "Handler A")]);
        let server_a_control = server_a.clone();
        let server_b = DynamicToolServer::new(vec![make_tool("alpha", "Handler B")]);
        let handle = ToolServer::new().run();

        let (client_a, server_task_a) = connect(server_a, handle.clone()).await;
        let (client_b, server_task_b) = connect(server_b, handle.clone()).await;

        server_a_control
            .set_tools(vec![
                make_tool("alpha", "Refreshed handler A"),
                make_tool("a_refresh_complete", "Refresh sentinel"),
            ])
            .await;
        let _running_server_a = server_task_a.await.unwrap();
        let _running_server_b = server_task_b.await.unwrap();

        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let defs = handle.get_tool_defs(None).await.unwrap();
                if defs
                    .iter()
                    .any(|definition| definition.name == "a_refresh_complete")
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("handler A refresh completed");

        let defs = handle.get_tool_defs(None).await.unwrap();
        let alpha = defs
            .iter()
            .find(|definition| definition.name == "alpha")
            .expect("alpha remains registered");
        assert_eq!(alpha.description, "Handler B");

        // Once B disconnects, its generation must no longer shield the dead
        // registration from A. Otherwise the registry keeps advertising B and
        // execution fails with `Transport closed` indefinitely.
        client_b.cancel().await.unwrap();
        server_a_control
            .set_tools(vec![make_tool("alpha", "Reclaimed handler A")])
            .await;

        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let defs = handle.get_tool_defs(None).await.unwrap();
                if defs
                    .iter()
                    .any(|definition| definition.description == "Reclaimed handler A")
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("handler A reclaimed the disconnected peer's registration");

        let result = handle
            .execute("alpha", "{}", &mut crate::tool::ToolContext::new())
            .await;
        assert!(
            result.is_success(),
            "reclaimed tool should execute: {result:?}"
        );

        client_a.cancel().await.unwrap();
    }

    #[test]
    fn client_handler_get_info_delegates() {
        let info = ClientInfo::new(
            ClientCapabilities::default(),
            Implementation::new("test-client", "1.0.0"),
        );
        let handler = McpClientHandler::new(
            McpClientConfig::new(info.client_info.clone())
                .with_client_capabilities(info.capabilities.clone()),
            ToolServer::new().run(),
        );
        let returned = handler.get_info();
        assert_eq!(returned.client_info.name, "test-client");
        assert_eq!(returned.client_info.version, "1.0.0");
    }

    #[tokio::test]
    async fn mcp_tool_preserves_provider_definition() {
        let tool = make_tool("search_docs", "Search the docs");
        let server = DynamicToolServer::new(vec![tool.clone()]);
        let (c2s, sfc) = tokio::io::duplex(8192);
        let (s2c, cfs) = tokio::io::duplex(8192);
        let task = tokio::spawn(async move {
            let running = server.serve((sfc, s2c)).await.unwrap();
            running.waiting().await.unwrap();
        });
        let client = ClientInfo::default().serve((cfs, c2s)).await.unwrap();
        let (request_handle, _client_state) = McpRequestHandle::for_test(client.peer().clone());
        let handle = ToolServer::new().rmcp_tool(tool, request_handle).run();
        let defs = handle.get_tool_defs(None).await.unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "search_docs");
        assert_eq!(defs[0].description, "Search the docs");
        client.cancel().await.unwrap();

        let defs = handle.get_tool_defs(None).await.unwrap();
        assert!(
            defs.is_empty(),
            "a disconnected directly registered MCP tool must not remain provider-visible"
        );
        task.abort();
    }

    #[tokio::test]
    async fn disconnected_directly_registered_mcp_tool_is_retired_on_dispatch() {
        let tool = make_tool("search_docs", "Search the docs");
        let server = DynamicToolServer::new(vec![tool.clone()]);
        let (c2s, sfc) = tokio::io::duplex(8192);
        let (s2c, cfs) = tokio::io::duplex(8192);
        let task = tokio::spawn(async move {
            let running = server.serve((sfc, s2c)).await.unwrap();
            running.waiting().await.unwrap();
        });
        let client = ClientInfo::default().serve((cfs, c2s)).await.unwrap();
        let (request_handle, _client_state) = McpRequestHandle::for_test(client.peer().clone());
        let handle = ToolServer::new().rmcp_tool(tool, request_handle).run();

        client.cancel().await.unwrap();

        let result = handle
            .execute("search_docs", "{}", &mut crate::tool::ToolContext::new())
            .await;
        assert_eq!(
            result.error().expect("disconnected tool must fail").kind(),
            crate::tool::ToolErrorKind::NotFound
        );
        task.abort();
    }
}
