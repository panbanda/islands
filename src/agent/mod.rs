//! Islands Agent - AI-powered Codebase Q&A
//!
//! This crate provides an AI agent for answering questions about indexed
//! codebases using LLM providers (OpenAI, Anthropic).

#![warn(clippy::all)]
#![allow(
    missing_docs,
    unsafe_code,
    clippy::collapsible_if,
    clippy::uninlined_format_args
)]

pub mod agent;
pub mod error;
pub mod llm;
pub mod prompt;

pub use agent::IslandsAgent;
pub use error::{Error, Result};

/// Re-export commonly used types
pub mod prelude {
    pub use super::agent::IslandsAgent;
    pub use super::error::{Error, Result};
    pub use super::llm::LlmProvider;
}
