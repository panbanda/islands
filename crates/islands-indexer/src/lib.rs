//! Islands Indexer - Repository Indexing Service
//!
//! This crate provides the indexing service that coordinates:
//! - Repository cloning and synchronization
//! - LEANN index building
//! - Periodic sync and re-indexing
//! - Webhook handling for push events

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::uninlined_format_args)]

pub mod error;
pub mod manager;
pub mod service;
pub mod state;

#[cfg(feature = "watch")]
pub mod watcher;

pub use error::{Error, Result};
pub use manager::RepositoryManager;
pub use service::{IndexInfo, IndexerConfig, IndexerService, StoredIndex};
pub use state::RepositoryState;

/// Re-export commonly used types
pub mod prelude {
    pub use crate::error::{Error, Result};
    pub use crate::manager::RepositoryManager;
    pub use crate::service::{IndexInfo, IndexerConfig, IndexerService, StoredIndex};
    pub use crate::state::RepositoryState;
}
