//! Islands CLI - Command-Line Interface
//!
//! This crate provides the CLI for the Islands codebase indexing system.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(
    clippy::uninlined_format_args,
    clippy::field_reassign_with_default
)]

pub mod commands;
pub mod config;
pub mod error;
pub mod output;

pub use config::Config;
pub use error::{Error, Result};
