//! Protocol-neutral deferred tool execution and reconstruction.

use std::{collections::HashMap, sync::Arc, time::Duration};

use rig_core::{
    tool::{ToolExecutionError, ToolResult},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};
use serde::{Deserialize, Serialize};

use super::ToolContext;

/// Current serialized descriptor format.
pub const DEFERRED_TOOL_DESCRIPTOR_VERSION: u16 = 1;

/// Serializable information needed to reconstruct one deferred execution.
///
/// Descriptors are deliberately data-only. Backends must never place
/// credentials, live transports, bearer tokens, or process-local handles in
/// `payload`; those belong in a registered resolver.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct DeferredToolDescriptor {
    version: u16,
    backend_type: String,
    execution_id: String,
    #[serde(default)]
    payload: serde_json::Value,
}

impl DeferredToolDescriptor {
    /// Create a descriptor using Rig's current format version.
    pub fn new(
        backend_type: impl Into<String>,
        execution_id: impl Into<String>,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            version: DEFERRED_TOOL_DESCRIPTOR_VERSION,
            backend_type: backend_type.into(),
            execution_id: execution_id.into(),
            payload,
        }
    }

    /// Serialized format version.
    pub fn version(&self) -> u16 {
        self.version
    }

    /// Resolver key for this backend.
    pub fn backend_type(&self) -> &str {
        &self.backend_type
    }

    /// Backend-opaque execution identifier.
    pub fn execution_id(&self) -> &str {
        &self.execution_id
    }

    /// Backend-specific, credential-free reconstruction data.
    pub fn payload(&self) -> &serde_json::Value {
        &self.payload
    }

    fn validate(&self) -> Result<(), DeferredResolverError> {
        if self.version != DEFERRED_TOOL_DESCRIPTOR_VERSION {
            return Err(DeferredResolverError::UnsupportedVersion(self.version));
        }
        if self.backend_type.trim().is_empty() {
            return Err(DeferredResolverError::MissingBackendType);
        }
        if self.execution_id.trim().is_empty() {
            return Err(DeferredResolverError::MissingExecutionId);
        }
        Ok(())
    }
}

/// One protocol-neutral request for additional application input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct InputRequest {
    /// Identifier matched by [`InputResponse::request_id`].
    pub id: String,
    /// Backend-neutral category such as `elicitation`, `sampling`, or `roots`.
    pub kind: String,
    /// Human-readable prompt, when supplied by the backend.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Input schema or other structured constraints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
    /// Non-secret extension fields preserved for the application handler.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

/// Ordered input requests surfaced by a deferred execution.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct InputRequests(pub Vec<InputRequest>);

/// One application response matched to an input request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct InputResponse {
    /// Identifier of the corresponding [`InputRequest`].
    pub request_id: String,
    /// Structured response value.
    pub value: serde_json::Value,
}

/// Ordered application responses submitted to a deferred execution.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct InputResponses(pub Vec<InputResponse>);

/// Application callback used when a deferred execution needs client input.
///
/// Backends surface only the request kinds negotiated by their client. The
/// handler must return exactly one response for every request identifier.
pub trait DeferredInputHandler: WasmCompatSend + WasmCompatSync {
    /// Fulfil the outstanding requests for one deferred execution.
    fn respond<'a>(
        &'a self,
        descriptor: &'a DeferredToolDescriptor,
        requests: &'a InputRequests,
    ) -> WasmBoxedFuture<'a, Result<InputResponses, ToolExecutionError>>;
}

/// Bounds applied while an agent runner waits for a deferred tool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeferredExecutionPolicy {
    /// Maximum wall-clock time spent resolving one deferred execution.
    pub timeout: Duration,
    /// Delay between state reads while the backend reports `Working`.
    pub working_poll_interval: Duration,
    /// Maximum number of observable state reads, including input rounds.
    pub max_state_reads: usize,
}

impl Default for DeferredExecutionPolicy {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(24 * 60 * 60),
            working_poll_interval: Duration::from_millis(100),
            max_state_reads: 10_000,
        }
    }
}

/// Result of invoking a tool at Rig's erased runtime boundary.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ToolExecution {
    /// The tool reached an immediate terminal result.
    Complete(ToolResult),
    /// The backend created an execution that can be reconstructed later.
    Deferred(DeferredToolDescriptor),
}

impl ToolExecution {
    /// Convert an ordinary in-process result into a complete execution.
    pub fn complete(result: ToolResult) -> Self {
        Self::Complete(result)
    }
}

/// Current state of a live deferred execution.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum DeferredToolState {
    /// The backend is still making progress.
    Working,
    /// Application input is required before work can continue.
    InputRequired(InputRequests),
    /// The tool reached a completed result, including model-visible tool errors.
    Completed(ToolResult),
    /// The deferred backend itself failed to execute the work.
    Failed(ToolExecutionError),
    /// The execution reached its immutable cancelled state.
    Cancelled,
}

/// Serializable lifecycle signal emitted while a deferred tool is driven.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
#[non_exhaustive]
pub enum DeferredToolLifecycleEvent {
    /// A descriptor was accepted for reconstruction.
    Started,
    /// The backend remains in progress.
    Working,
    /// The backend is waiting for application input.
    InputRequired { requests: InputRequests },
    /// Matching application responses were submitted.
    InputSubmitted { request_ids: Vec<String> },
    /// Rig requested cooperative cancellation.
    CancellationRequested,
    /// The execution completed, including model-visible tool errors.
    Completed,
    /// The deferred execution itself failed.
    Failed { message: String },
    /// The execution reached its cancelled terminal state.
    Cancelled,
}

/// Live backend operations for one deferred execution.
pub trait DeferredToolDriver: WasmCompatSend + WasmCompatSync {
    /// Read the current state.
    fn state(&self) -> WasmBoxedFuture<'_, Result<DeferredToolState, ToolExecutionError>>;

    /// Submit responses to the currently requested inputs.
    fn submit_input(
        &self,
        responses: InputResponses,
    ) -> WasmBoxedFuture<'_, Result<DeferredToolState, ToolExecutionError>>;

    /// Request cooperative cancellation.
    fn cancel(&self) -> WasmBoxedFuture<'_, Result<DeferredToolState, ToolExecutionError>>;

    /// Publish backend-specific terminal metadata into the original dispatch
    /// context. Ordinary drivers may keep the default no-op implementation.
    fn publish_result_context(&self, _context: &mut ToolContext) {}
}

/// Reconstructed live access to a deferred tool execution.
#[derive(Clone)]
pub struct DeferredToolHandle {
    descriptor: DeferredToolDescriptor,
    driver: Arc<dyn DeferredToolDriver>,
}

impl std::fmt::Debug for DeferredToolHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeferredToolHandle")
            .field("descriptor", &self.descriptor)
            .finish_non_exhaustive()
    }
}

impl DeferredToolHandle {
    /// Build a live handle from a descriptor and backend driver.
    pub fn new(
        descriptor: DeferredToolDescriptor,
        driver: impl DeferredToolDriver + 'static,
    ) -> Self {
        Self {
            descriptor,
            driver: Arc::new(driver),
        }
    }

    /// Data-only reconstruction descriptor.
    pub fn descriptor(&self) -> &DeferredToolDescriptor {
        &self.descriptor
    }

    /// Read the current backend state.
    pub async fn state(&self) -> Result<DeferredToolState, ToolExecutionError> {
        self.driver.state().await
    }

    /// Submit responses to the currently requested inputs.
    pub async fn submit_input(
        &self,
        responses: InputResponses,
    ) -> Result<DeferredToolState, ToolExecutionError> {
        self.driver.submit_input(responses).await
    }

    /// Request cooperative cancellation.
    pub async fn cancel(&self) -> Result<DeferredToolState, ToolExecutionError> {
        self.driver.cancel().await
    }

    /// Publish backend-specific terminal metadata into a dispatch context.
    pub fn publish_result_context(&self, context: &mut ToolContext) {
        self.driver.publish_result_context(context);
    }
}

/// Backend factory capable of reconstructing serialized descriptors.
pub trait DeferredToolResolver: WasmCompatSend + WasmCompatSync {
    /// Stable backend key stored in [`DeferredToolDescriptor`].
    fn backend_type(&self) -> &str;

    /// Reconstruct a live execution using process-local credentials and
    /// transports owned by this resolver.
    fn resolve<'a>(
        &'a self,
        descriptor: &'a DeferredToolDescriptor,
    ) -> WasmBoxedFuture<'a, Result<DeferredToolHandle, ToolExecutionError>>;
}

/// Resolver registration or descriptor validation failure.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeferredResolverError {
    /// A resolver is already registered for this backend key.
    #[error("a deferred tool resolver is already registered for `{0}`")]
    AlreadyRegistered(String),
    /// No resolver is registered for this backend key.
    #[error("no deferred tool resolver is registered for `{0}`")]
    NotRegistered(String),
    /// The serialized descriptor format is not supported.
    #[error("unsupported deferred tool descriptor version {0}")]
    UnsupportedVersion(u16),
    /// The descriptor has no backend key.
    #[error("deferred tool descriptor has an empty backend type")]
    MissingBackendType,
    /// The descriptor has no execution identifier.
    #[error("deferred tool descriptor has an empty execution ID")]
    MissingExecutionId,
    /// Internal resolver registry synchronization failed.
    #[error("deferred tool resolver registry lock is poisoned")]
    RegistryPoisoned,
}

/// Thread-safe resolver registry keyed by descriptor backend type.
#[derive(Clone, Default)]
pub struct DeferredToolResolverRegistry {
    resolvers: Arc<std::sync::RwLock<HashMap<String, Arc<dyn DeferredToolResolver>>>>,
}

impl DeferredToolResolverRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one backend resolver without replacing an existing resolver.
    pub fn register(
        &self,
        resolver: impl DeferredToolResolver + 'static,
    ) -> Result<(), DeferredResolverError> {
        self.register_arc(Arc::new(resolver))
    }

    /// Register an already shared resolver without replacing an existing one.
    pub fn register_arc(
        &self,
        resolver: Arc<dyn DeferredToolResolver>,
    ) -> Result<(), DeferredResolverError> {
        let key = resolver.backend_type().to_owned();
        let mut resolvers = self
            .resolvers
            .write()
            .map_err(|_| DeferredResolverError::RegistryPoisoned)?;
        if resolvers.contains_key(&key) {
            return Err(DeferredResolverError::AlreadyRegistered(key));
        }
        resolvers.insert(key, resolver);
        Ok(())
    }

    /// Reconstruct a descriptor with the resolver registered for its backend.
    pub async fn resolve(
        &self,
        descriptor: &DeferredToolDescriptor,
    ) -> Result<DeferredToolHandle, ToolExecutionError> {
        descriptor
            .validate()
            .map_err(|error| ToolExecutionError::invalid_args(error.to_string()))?;
        let resolver = self
            .resolvers
            .read()
            .map_err(|_| {
                ToolExecutionError::other(DeferredResolverError::RegistryPoisoned.to_string())
            })?
            .get(descriptor.backend_type())
            .cloned()
            .ok_or_else(|| {
                ToolExecutionError::not_found(
                    DeferredResolverError::NotRegistered(descriptor.backend_type().to_owned())
                        .to_string(),
                )
            })?;
        resolver.resolve(descriptor).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CompletedDriver;

    impl DeferredToolDriver for CompletedDriver {
        fn state(&self) -> WasmBoxedFuture<'_, Result<DeferredToolState, ToolExecutionError>> {
            Box::pin(async {
                Ok(DeferredToolState::Completed(ToolResult::success(
                    rig_core::tool::ToolOutput::text("done"),
                )))
            })
        }

        fn submit_input(
            &self,
            _responses: InputResponses,
        ) -> WasmBoxedFuture<'_, Result<DeferredToolState, ToolExecutionError>> {
            self.state()
        }

        fn cancel(&self) -> WasmBoxedFuture<'_, Result<DeferredToolState, ToolExecutionError>> {
            Box::pin(async { Ok(DeferredToolState::Cancelled) })
        }
    }

    struct FixtureResolver;

    impl DeferredToolResolver for FixtureResolver {
        fn backend_type(&self) -> &str {
            "fixture"
        }

        fn resolve<'a>(
            &'a self,
            descriptor: &'a DeferredToolDescriptor,
        ) -> WasmBoxedFuture<'a, Result<DeferredToolHandle, ToolExecutionError>> {
            Box::pin(
                async move { Ok(DeferredToolHandle::new(descriptor.clone(), CompletedDriver)) },
            )
        }
    }

    #[tokio::test]
    async fn descriptor_round_trip_reconstructs_without_live_state() {
        let descriptor = DeferredToolDescriptor::new(
            "fixture",
            "opaque-123",
            serde_json::json!({"region": "test"}),
        );
        let serialized = serde_json::to_string(&descriptor).expect("serialize descriptor");
        let restored: DeferredToolDescriptor =
            serde_json::from_str(&serialized).expect("deserialize descriptor");
        let registry = DeferredToolResolverRegistry::new();
        registry
            .register(FixtureResolver)
            .expect("register resolver");
        let handle = registry
            .resolve(&restored)
            .await
            .expect("resolve descriptor");

        assert_eq!(handle.descriptor(), &descriptor);
        let DeferredToolState::Completed(result) = handle.state().await.expect("read state") else {
            panic!("expected completed state");
        };
        assert_eq!(result.output().as_text(), Some("done"));
    }

    #[test]
    fn duplicate_backend_registration_is_rejected() {
        let registry = DeferredToolResolverRegistry::new();
        registry.register(FixtureResolver).expect("first resolver");
        assert!(matches!(
            registry.register(FixtureResolver),
            Err(DeferredResolverError::AlreadyRegistered(backend)) if backend == "fixture"
        ));
    }
}
