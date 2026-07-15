//! MCP elicitation (SEP-1686 interactive input): the handler seam that lets a
//! server ask the user for input mid-task.
//!
//! Register an implementation via
//! [`McpClientHandler::with_elicitation_handler`](super::McpClientHandler::with_elicitation_handler).
//! Registration does three things:
//!
//! 1. `elicitation/create` requests route to [`McpElicitationHandler::elicit`]
//!    instead of rmcp's auto-decline default.
//! 2. The handler's [`capability`](McpElicitationHandler::capability) is
//!    advertised in the client handshake — servers only elicit from clients
//!    that declared it.
//! 3. Deferred tasks that enter `input_required` keep waiting for the
//!    elicitation round-trip instead of failing with
//!    `mcp_task_input_required` (see
//!    [`McpTaskHandle::wait`](super::McpTaskHandle)).
//!
//! Unregistered, every behavior is exactly as before: requests are declined,
//! no capability is advertised, and `input_required` is a classified failure.
//!
//! Policy stays with the embedder: whether to prompt a human (see
//! `examples/agent_with_human_in_the_loop` for the fail-closed stdin pattern),
//! call another service, or auto-answer from configuration is entirely the
//! handler's decision. Decline anything you do not support.

use std::future::ready;

use rmcp::model::{
    ElicitRequestParams, ElicitResult, ElicitationCapability, ElicitationResponseNotificationParam,
    FormElicitationCapability, Meta, RelatedTaskMetadata,
};

use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};

/// Answers MCP elicitation requests (SEP-1686 interactive input).
///
/// See the [module docs](self) for what registration changes. Handlers should
/// be **fail-closed**: when input cannot be collected (no user present, EOF,
/// timeout), return [`ElicitationAction::Decline`](rmcp::model::ElicitationAction::Decline)
/// rather than fabricating content.
pub trait McpElicitationHandler: WasmCompatSend + WasmCompatSync {
    /// Answer one elicitation request.
    ///
    /// `request` is form-mode (a message plus a
    /// [`ElicitationSchema`](rmcp::model::ElicitationSchema) describing the
    /// requested fields) or URL-mode (a URL the user must visit); decline
    /// modes you do not support. Use [`related_task_id`] on the request's
    /// `_meta` to correlate it with the pending deferred task that raised it.
    ///
    /// # Errors
    /// An `Err` surfaces to the server as the request's JSON-RPC error;
    /// prefer answering with a `Decline`/`Cancel` action for ordinary
    /// "cannot answer" outcomes.
    fn elicit(
        &self,
        request: ElicitRequestParams,
    ) -> WasmBoxedFuture<'_, Result<ElicitResult, rmcp::ErrorData>>;

    /// The elicitation capability advertised to servers during the handshake.
    ///
    /// Defaults to form-mode only (without schema validation — Rig does not
    /// validate handler content against the requested schema). Override to
    /// declare URL-mode support, and then also override
    /// [`url_elicitation_complete`](Self::url_elicitation_complete) to observe
    /// out-of-band completions.
    fn capability(&self) -> ElicitationCapability {
        ElicitationCapability::new().with_form(FormElicitationCapability::new())
    }

    /// A URL-mode elicitation completed out of band
    /// (`notifications/elicitation/complete`). Default: no-op — only relevant
    /// to handlers that declared URL-mode support.
    fn url_elicitation_complete(
        &self,
        params: ElicitationResponseNotificationParam,
    ) -> WasmBoxedFuture<'_, ()> {
        let _ = params;
        Box::pin(ready(()))
    }
}

/// Extract the related-task id (`io.modelcontextprotocol/related-task`) from a
/// request's `_meta`, correlating an elicitation with the pending deferred
/// task that raised it. rmcp 2.2 ships the [`RelatedTaskMetadata`] type but no
/// parse helper; this is that helper.
///
/// Returns `None` when the `_meta` is absent, carries no related-task key, or
/// the value does not parse as a [`RelatedTaskMetadata`].
pub fn related_task_id(meta: Option<&Meta>) -> Option<String> {
    let value = meta?.get(RelatedTaskMetadata::META_KEY)?;
    serde_json::from_value::<RelatedTaskMetadata>(value.clone())
        .ok()
        .map(|related| related.task_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn related_task_id_parses_the_meta_key() {
        let mut meta = Meta::new();
        meta.0.insert(
            RelatedTaskMetadata::META_KEY.to_string(),
            serde_json::json!({ "taskId": "task-7" }),
        );
        assert_eq!(related_task_id(Some(&meta)).as_deref(), Some("task-7"));
    }

    #[test]
    fn related_task_id_is_none_for_absent_or_malformed_meta() {
        assert_eq!(related_task_id(None), None);

        let empty = Meta::new();
        assert_eq!(related_task_id(Some(&empty)), None);

        let mut malformed = Meta::new();
        malformed.0.insert(
            RelatedTaskMetadata::META_KEY.to_string(),
            serde_json::json!("not an object"),
        );
        assert_eq!(related_task_id(Some(&malformed)), None);
    }

    #[test]
    fn default_capability_is_form_only() {
        struct DeclineAll;
        impl McpElicitationHandler for DeclineAll {
            fn elicit(
                &self,
                _request: ElicitRequestParams,
            ) -> WasmBoxedFuture<'_, Result<ElicitResult, rmcp::ErrorData>> {
                Box::pin(async { Ok(ElicitResult::new(rmcp::model::ElicitationAction::Decline)) })
            }
        }

        let capability = DeclineAll.capability();
        assert!(capability.form.is_some(), "form mode must be declared");
        assert!(capability.url.is_none(), "URL mode is opt-in");
        assert_eq!(
            capability.form.and_then(|form| form.schema_validation),
            None,
            "rig does not claim schema validation"
        );
    }
}
