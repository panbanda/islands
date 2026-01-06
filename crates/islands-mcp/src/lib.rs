//! Islands MCP - Model Context Protocol Server
//!
//! This crate implements the MCP (Model Context Protocol) server for Islands,
//! enabling AI assistants like Claude to search and query indexed codebases.
//!
//! The server communicates via stdio using JSON-RPC 2.0.

#![warn(clippy::all)]
#![allow(
    missing_docs,
    dead_code,
    clippy::collapsible_if,
    clippy::uninlined_format_args
)]

pub mod error;
pub mod protocol;
pub mod server;
pub mod tools;

pub use error::{Error, Result};
pub use server::McpServer;

/// Re-export commonly used types
pub mod prelude {
    pub use crate::error::{Error, Result};
    pub use crate::server::McpServer;
}
