//! Pythia CLI - Command-Line Interface
//!
//! This crate provides the CLI for the Pythia codebase indexing system.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

pub mod commands;
pub mod config;
pub mod error;
pub mod output;

pub use config::Config;
pub use error::{Error, Result};
