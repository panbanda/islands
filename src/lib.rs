#![allow(
    clippy::manual_is_multiple_of,
    clippy::collapsible_if,
    clippy::field_reassign_with_default,
    clippy::approx_constant,
    clippy::manual_range_contains,
    clippy::clone_on_copy,
    clippy::excessive_precision,
    clippy::unnecessary_lazy_evaluations,
    clippy::explicit_auto_deref,
    clippy::uninlined_format_args,
    unexpected_cfgs
)]
//! Islands - LEANN-based codebase indexing with MCP server and AI-powered search
//!
//! This crate provides:
//! - **core**: HNSW/LEANN graph-based vector search with product quantization
//! - **providers**: Git provider implementations (GitHub, GitLab, Bitbucket, Gitea)
//! - **indexer**: Repository indexing service with LEANN integration
//! - **mcp**: Model Context Protocol server for AI assistants
//! - **agent**: AI-powered Q&A agent using LLM providers
//!
//! # Example
//!
//! ```rust,no_run
//! use islands::core::prelude::*;
//!
//! # fn main() -> islands::core::CoreResult<()> {
//! // Create an HNSW graph
//! let mut graph = HnswGraph::with_defaults()?;
//!
//! // Insert vectors
//! graph.insert(vec![1.0, 0.0, 0.0])?;
//! graph.insert(vec![0.0, 1.0, 0.0])?;
//!
//! // Search for nearest neighbors
//! let results = graph.search(&[0.9, 0.1, 0.0], 2, 50)?;
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]

pub mod core;
pub mod indexer;
pub mod providers;

#[cfg(feature = "mcp")]
pub mod mcp;

#[cfg(feature = "agent")]
pub mod agent;

pub mod commands;
pub mod config;
pub mod error;
pub mod output;

// Re-export main types from core
pub use core::{
    CoreError, CoreResult, Distance, DistanceMetric, Embedding, HnswConfig, HnswGraph, HnswNode,
    LeannConfig, LeannIndex,
};

// Re-export provider types
pub use providers::{
    BitbucketProvider, GitHubProvider, GitLabProvider, GitProvider, GiteaProvider, ProviderError,
    ProviderFactory,
};

// Re-export indexer types
pub use indexer::{IndexerConfig, IndexerService, RepositoryManager, RepositoryState};

// Re-export MCP types
#[cfg(feature = "mcp")]
pub use mcp::McpServer;

// Re-export agent types
#[cfg(feature = "agent")]
pub use agent::IslandsAgent;

// Re-export CLI types
pub use config::Config;
pub use error::{Error, Result};

/// Prelude for commonly used types
pub mod prelude {
    pub use crate::core::prelude::*;
    pub use crate::indexer::prelude::*;
    pub use crate::providers::prelude::*;

    #[cfg(feature = "mcp")]
    pub use crate::mcp::prelude::*;

    #[cfg(feature = "agent")]
    pub use crate::agent::prelude::*;

    pub use crate::config::Config;
    pub use crate::error::{Error, Result};
}
