//! Pythia Agent - AI-powered Codebase Q&A
//!
//! This crate provides an AI agent for answering questions about indexed
//! codebases using LLM providers (OpenAI, Anthropic).

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

pub mod agent;
pub mod error;
pub mod llm;
pub mod prompt;

pub use agent::PythiaAgent;
pub use error::{Error, Result};

/// Re-export commonly used types
pub mod prelude {
    pub use crate::agent::PythiaAgent;
    pub use crate::error::{Error, Result};
    pub use crate::llm::LlmProvider;
}
