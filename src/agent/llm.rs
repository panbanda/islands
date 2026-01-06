//! LLM Provider abstractions

use std::pin::Pin;

use futures::Stream;
use serde::{Deserialize, Serialize};

use super::error::Result;

/// A message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role (system, user, assistant)
    pub role: String,
    /// Message content
    pub content: String,
}

impl Message {
    /// Create a system message
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    /// Create a user message
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    /// Create an assistant message
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

/// Configuration for LLM requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Model name
    pub model: String,
    /// Temperature (0.0 - 1.0)
    pub temperature: f32,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Top-p sampling
    pub top_p: Option<f32>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4o".to_string(),
            temperature: 0.1,
            max_tokens: 4096,
            top_p: None,
        }
    }
}

/// Stream of response chunks
pub type ResponseStream = Pin<Box<dyn Stream<Item = Result<String>> + Send>>;

/// Trait for LLM providers
#[async_trait::async_trait]
pub trait LlmProvider: Send + Sync {
    /// Get the provider name
    fn name(&self) -> &'static str;

    /// Generate a completion
    async fn complete(&self, messages: &[Message], config: &LlmConfig) -> Result<String>;

    /// Generate a streaming completion
    async fn complete_stream(
        &self,
        messages: &[Message],
        config: &LlmConfig,
    ) -> Result<ResponseStream>;
}

/// OpenAI provider implementation
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_system() {
        let msg = Message::system("You are a helpful assistant");
        assert_eq!(msg.role, "system");
        assert_eq!(msg.content, "You are a helpful assistant");
    }

    #[test]
    fn test_message_user() {
        let msg = Message::user("Hello, how are you?");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello, how are you?");
    }

    #[test]
    fn test_message_assistant() {
        let msg = Message::assistant("I'm doing well, thank you!");
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content, "I'm doing well, thank you!");
    }

    #[test]
    fn test_message_from_string() {
        let content = String::from("Dynamic content");
        let msg = Message::user(content);
        assert_eq!(msg.content, "Dynamic content");
    }

    #[test]
    fn test_message_clone() {
        let msg = Message::user("Test message");
        let cloned = msg.clone();
        assert_eq!(cloned.role, msg.role);
        assert_eq!(cloned.content, msg.content);
    }

    #[test]
    fn test_message_debug() {
        let msg = Message::system("Test");
        let debug_str = format!("{:?}", msg);
        assert!(debug_str.contains("Message"));
        assert!(debug_str.contains("system"));
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message::user("Hello");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"Hello\""));
    }

    #[test]
    fn test_message_deserialization() {
        let json = r#"{"role": "assistant", "content": "Hi there"}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content, "Hi there");
    }

    #[test]
    fn test_message_roundtrip() {
        let original = Message::user("Test content");
        let json = serde_json::to_string(&original).unwrap();
        let parsed: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.role, original.role);
        assert_eq!(parsed.content, original.content);
    }

    #[test]
    fn test_message_empty_content() {
        let msg = Message::user("");
        assert_eq!(msg.content, "");
    }

    #[test]
    fn test_message_unicode_content() {
        let msg = Message::user("Hello \u{1F44B} World \u{1F310}");
        assert!(msg.content.contains('\u{1F44B}'));
    }

    #[test]
    fn test_message_multiline_content() {
        let content = "Line 1\nLine 2\nLine 3";
        let msg = Message::system(content);
        assert!(msg.content.contains('\n'));
    }

    #[test]
    fn test_llm_config_default() {
        let config = LlmConfig::default();
        assert_eq!(config.model, "gpt-4o");
        assert!((config.temperature - 0.1).abs() < f32::EPSILON);
        assert_eq!(config.max_tokens, 4096);
        assert!(config.top_p.is_none());
    }

    #[test]
    fn test_llm_config_custom() {
        let config = LlmConfig {
            model: "gpt-3.5-turbo".to_string(),
            temperature: 0.7,
            max_tokens: 2048,
            top_p: Some(0.9),
        };

        assert_eq!(config.model, "gpt-3.5-turbo");
        assert!((config.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(config.max_tokens, 2048);
        assert!((config.top_p.unwrap() - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_llm_config_clone() {
        let config = LlmConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.model, config.model);
        assert_eq!(cloned.temperature, config.temperature);
        assert_eq!(cloned.max_tokens, config.max_tokens);
    }

    #[test]
    fn test_llm_config_debug() {
        let config = LlmConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("LlmConfig"));
        assert!(debug_str.contains("gpt-4o"));
    }

    #[test]
    fn test_llm_config_serialization() {
        let config = LlmConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"model\":\"gpt-4o\""));
        assert!(json.contains("\"temperature\":0.1"));
    }

    #[test]
    fn test_llm_config_deserialization() {
        let json = r#"{
            "model": "claude-3",
            "temperature": 0.5,
            "max_tokens": 1000,
            "top_p": 0.95
        }"#;
        let config: LlmConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.model, "claude-3");
        assert_eq!(config.max_tokens, 1000);
        assert!((config.top_p.unwrap() - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn test_llm_config_roundtrip() {
        let original = LlmConfig {
            model: "test-model".to_string(),
            temperature: 0.3,
            max_tokens: 512,
            top_p: Some(0.8),
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: LlmConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model, original.model);
        assert_eq!(parsed.temperature, original.temperature);
        assert_eq!(parsed.max_tokens, original.max_tokens);
        assert_eq!(parsed.top_p, original.top_p);
    }

    #[test]
    fn test_llm_config_temperature_boundaries() {
        let config1 = LlmConfig {
            temperature: 0.0,
            ..LlmConfig::default()
        };
        assert_eq!(config1.temperature, 0.0);

        let config2 = LlmConfig {
            temperature: 1.0,
            ..LlmConfig::default()
        };
        assert_eq!(config2.temperature, 1.0);
    }

    #[test]
    fn test_llm_config_without_top_p() {
        let json = r#"{
            "model": "gpt-4",
            "temperature": 0.5,
            "max_tokens": 1000
        }"#;
        let config: LlmConfig = serde_json::from_str(json).unwrap();
        assert!(config.top_p.is_none());
    }
}

#[cfg(feature = "openai")]
pub mod openai {
    use super::*;
    use async_stream::stream;
    use futures::StreamExt;
    use reqwest::Client;

    /// OpenAI LLM provider
    pub struct OpenAiProvider {
        client: Client,
        api_key: String,
        base_url: String,
    }

    impl OpenAiProvider {
        /// Create a new OpenAI provider
        pub fn new(api_key: impl Into<String>) -> Self {
            Self {
                client: Client::new(),
                api_key: api_key.into(),
                base_url: "https://api.openai.com/v1".to_string(),
            }
        }

        /// Set a custom base URL (for Azure, etc.)
        #[must_use]
        pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
            self.base_url = url.into();
            self
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for OpenAiProvider {
        fn name(&self) -> &'static str {
            "openai"
        }

        async fn complete(&self, messages: &[Message], config: &LlmConfig) -> Result<String> {
            let body = serde_json::json!({
                "model": config.model,
                "messages": messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            });

            let response = self
                .client
                .post(format!("{}/chat/completions", self.base_url))
                .header("Authorization", format!("Bearer {}", self.api_key))
                .json(&body)
                .send()
                .await?;

            let data: serde_json::Value = response.json().await?;

            data["choices"][0]["message"]["content"]
                .as_str()
                .map(String::from)
                .ok_or_else(|| crate::error::Error::Llm("No content in response".into()))
        }

        async fn complete_stream(
            &self,
            messages: &[Message],
            config: &LlmConfig,
        ) -> Result<ResponseStream> {
            let body = serde_json::json!({
                "model": config.model,
                "messages": messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "stream": true,
            });

            let response = self
                .client
                .post(format!("{}/chat/completions", self.base_url))
                .header("Authorization", format!("Bearer {}", self.api_key))
                .json(&body)
                .send()
                .await?;

            let stream = stream! {
                let mut stream = response.bytes_stream();

                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            let text = String::from_utf8_lossy(&bytes);
                            for line in text.lines() {
                                if let Some(data) = line.strip_prefix("data: ") {
                                    if data == "[DONE]" {
                                        break;
                                    }
                                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                                        if let Some(content) = json["choices"][0]["delta"]["content"].as_str() {
                                            yield Ok(content.to_string());
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            yield Err(crate::error::Error::Http(e));
                        }
                    }
                }
            };

            Ok(Box::pin(stream))
        }
    }
}
