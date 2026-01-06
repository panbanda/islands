//! Webhook event types

use serde::{Deserialize, Serialize};

use crate::repository::Repository;

/// Represents a webhook event from a git provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookEvent {
    /// Event type (push, pull_request, etc.)
    pub event_type: String,
    /// Repository that triggered the event
    pub repository: Repository,
    /// Git ref (branch/tag) if applicable
    pub ref_name: Option<String>,
    /// Commit SHA before the event
    pub before: Option<String>,
    /// Commit SHA after the event
    pub after: Option<String>,
    /// Raw payload for provider-specific data
    #[serde(default)]
    pub payload: serde_json::Value,
}

impl WebhookEvent {
    /// Create a push event
    #[must_use]
    pub fn push(
        repository: Repository,
        ref_name: String,
        before: String,
        after: String,
    ) -> Self {
        Self {
            event_type: "push".to_string(),
            repository,
            ref_name: Some(ref_name),
            before: Some(before),
            after: Some(after),
            payload: serde_json::Value::Null,
        }
    }

    /// Check if this is a push event
    #[must_use]
    pub fn is_push(&self) -> bool {
        self.event_type == "push" || self.event_type == "push_hook"
    }

    /// Check if this is a pull request event
    #[must_use]
    pub fn is_pull_request(&self) -> bool {
        self.event_type == "pull_request" || self.event_type == "merge_request"
    }

    /// Get the branch name from the ref (if it's a branch ref)
    #[must_use]
    pub fn branch_name(&self) -> Option<&str> {
        self.ref_name.as_ref().and_then(|r| {
            r.strip_prefix("refs/heads/")
        })
    }
}
