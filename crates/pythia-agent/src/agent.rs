//! Main agent implementation

use std::sync::Arc;

use futures::StreamExt;

use pythia_indexer::IndexerService;

use crate::error::Result;
use crate::llm::{LlmConfig, LlmProvider, Message, ResponseStream};
use crate::prompt::{DEFAULT_SYSTEM_PROMPT, build_messages, format_search_context};

/// The Pythia codebase Q&A agent
pub struct PythiaAgent {
    indexer: Arc<IndexerService>,
    llm: Arc<dyn LlmProvider>,
    config: LlmConfig,
    system_prompt: String,
    conversation: Vec<Message>,
}

impl PythiaAgent {
    /// Create a new agent
    pub fn new(
        indexer: Arc<IndexerService>,
        llm: Arc<dyn LlmProvider>,
        config: Option<LlmConfig>,
    ) -> Self {
        Self {
            indexer,
            llm,
            config: config.unwrap_or_default(),
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            conversation: Vec::new(),
        }
    }

    /// Set a custom system prompt
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Search for relevant context
    async fn search_context(&self, query: &str, top_k: usize) -> Result<Vec<serde_json::Value>> {
        self.indexer
            .search(query, None, top_k)
            .await
            .map_err(|e| crate::error::Error::Indexer(e))
    }

    /// Ask a question and get a response
    pub async fn ask(&mut self, question: &str, search_first: bool) -> Result<String> {
        // Optionally search for context
        let context = if search_first {
            let results = self.search_context(question, 5).await?;
            Some(format_search_context(&results))
        } else {
            None
        };

        // Build messages
        let messages = build_messages(
            &self.system_prompt,
            &self.conversation,
            context.as_deref(),
            question,
        );

        // Get completion
        let response = self.llm.complete(&messages, &self.config).await?;

        // Update conversation
        self.conversation.push(Message::user(question));
        self.conversation.push(Message::assistant(&response));

        Ok(response)
    }

    /// Ask a question and stream the response
    pub async fn ask_stream(
        &mut self,
        question: &str,
        search_first: bool,
    ) -> Result<(ResponseStream, impl FnOnce(String))> {
        // Optionally search for context
        let context = if search_first {
            let results = self.search_context(question, 5).await?;
            Some(format_search_context(&results))
        } else {
            None
        };

        // Build messages
        let messages = build_messages(
            &self.system_prompt,
            &self.conversation,
            context.as_deref(),
            question,
        );

        // Get streaming completion
        let stream = self.llm.complete_stream(&messages, &self.config).await?;

        // Add user message to conversation now
        self.conversation.push(Message::user(question));

        // Return stream and callback to add assistant response
        let conversation = &mut self.conversation as *mut Vec<Message>;

        let finish_callback = move |response: String| {
            // SAFETY: This is only called after stream completes
            unsafe {
                (*conversation).push(Message::assistant(response));
            }
        };

        Ok((stream, finish_callback))
    }

    /// Clear conversation history
    pub fn clear_conversation(&mut self) {
        self.conversation.clear();
    }

    /// Get available index names
    pub async fn list_indexes(&self) -> Vec<String> {
        self.indexer
            .list_indexes()
            .await
            .into_iter()
            .map(|info| info.name)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{LlmConfig, Message};
    use async_trait::async_trait;
    use futures::stream;
    use std::collections::HashMap;

    /// Mock LLM provider for testing
    struct MockLlmProvider {
        response: String,
    }

    impl MockLlmProvider {
        fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
            }
        }
    }

    #[async_trait]
    impl LlmProvider for MockLlmProvider {
        fn name(&self) -> &'static str {
            "mock"
        }

        async fn complete(
            &self,
            _messages: &[Message],
            _config: &LlmConfig,
        ) -> crate::error::Result<String> {
            Ok(self.response.clone())
        }

        async fn complete_stream(
            &self,
            _messages: &[Message],
            _config: &LlmConfig,
        ) -> crate::error::Result<ResponseStream> {
            let response = self.response.clone();
            let stream = stream::once(async move { Ok(response) });
            Ok(Box::pin(stream))
        }
    }

    fn create_test_indexer() -> Arc<IndexerService> {
        use pythia_indexer::service::IndexerConfig;
        let config = IndexerConfig::default();
        Arc::new(IndexerService::new(config, HashMap::new()))
    }

    #[test]
    fn test_agent_new_default_config() {
        let indexer = create_test_indexer();
        let llm = Arc::new(MockLlmProvider::new("test response"));

        let agent = PythiaAgent::new(indexer, llm, None);

        assert!(agent.conversation.is_empty());
        assert_eq!(agent.system_prompt, DEFAULT_SYSTEM_PROMPT);
    }

    #[test]
    fn test_agent_new_custom_config() {
        let indexer = create_test_indexer();
        let llm = Arc::new(MockLlmProvider::new("test response"));
        let config = LlmConfig {
            model: "custom-model".to_string(),
            temperature: 0.5,
            max_tokens: 500,
            top_p: Some(0.9),
        };

        let agent = PythiaAgent::new(indexer, llm, Some(config.clone()));

        assert_eq!(agent.config.model, "custom-model");
        assert_eq!(agent.config.temperature, 0.5);
        assert_eq!(agent.config.max_tokens, 500);
    }

    #[test]
    fn test_agent_with_system_prompt() {
        let indexer = create_test_indexer();
        let llm = Arc::new(MockLlmProvider::new("test"));

        let agent = PythiaAgent::new(indexer, llm, None).with_system_prompt("Custom prompt");

        assert_eq!(agent.system_prompt, "Custom prompt");
    }

    #[test]
    fn test_agent_with_system_prompt_string() {
        let indexer = create_test_indexer();
        let llm = Arc::new(MockLlmProvider::new("test"));

        let agent =
            PythiaAgent::new(indexer, llm, None).with_system_prompt(String::from("String prompt"));

        assert_eq!(agent.system_prompt, "String prompt");
    }

    #[test]
    fn test_agent_clear_conversation() {
        let indexer = create_test_indexer();
        let llm = Arc::new(MockLlmProvider::new("test"));

        let mut agent = PythiaAgent::new(indexer, llm, None);

        // Manually add messages
        agent.conversation.push(Message::user("test question"));
        agent.conversation.push(Message::assistant("test answer"));
        assert_eq!(agent.conversation.len(), 2);

        agent.clear_conversation();
        assert!(agent.conversation.is_empty());
    }

    #[tokio::test]
    async fn test_agent_ask_without_search() {
        let indexer = create_test_indexer();
        let llm = Arc::new(MockLlmProvider::new("This is the answer"));

        let mut agent = PythiaAgent::new(indexer, llm, None);
        let response = agent.ask("What is this?", false).await.unwrap();

        assert_eq!(response, "This is the answer");
        assert_eq!(agent.conversation.len(), 2);
        assert_eq!(agent.conversation[0].role, "user");
        assert_eq!(agent.conversation[0].content, "What is this?");
        assert_eq!(agent.conversation[1].role, "assistant");
        assert_eq!(agent.conversation[1].content, "This is the answer");
    }

    #[tokio::test]
    async fn test_agent_ask_with_search() {
        let indexer = create_test_indexer();
        let llm = Arc::new(MockLlmProvider::new("Search-aware answer"));

        let mut agent = PythiaAgent::new(indexer, llm, None);
        let response = agent.ask("Find the function", true).await.unwrap();

        assert_eq!(response, "Search-aware answer");
        assert_eq!(agent.conversation.len(), 2);
    }

    #[tokio::test]
    async fn test_agent_ask_multiple_turns() {
        let indexer = create_test_indexer();
        let llm = Arc::new(MockLlmProvider::new("Response"));

        let mut agent = PythiaAgent::new(indexer, llm, None);

        agent.ask("First question", false).await.unwrap();
        agent.ask("Second question", false).await.unwrap();
        agent.ask("Third question", false).await.unwrap();

        assert_eq!(agent.conversation.len(), 6);
    }

    #[tokio::test]
    async fn test_agent_list_indexes_empty() {
        let indexer = create_test_indexer();
        let llm = Arc::new(MockLlmProvider::new("test"));

        let agent = PythiaAgent::new(indexer, llm, None);
        let indexes = agent.list_indexes().await;

        assert!(indexes.is_empty());
    }

    #[tokio::test]
    async fn test_agent_search_context() {
        let indexer = create_test_indexer();
        let llm = Arc::new(MockLlmProvider::new("test"));

        let agent = PythiaAgent::new(indexer, llm, None);
        let results = agent.search_context("test query", 5).await.unwrap();

        // Empty indexer returns empty results
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_agent_ask_stream() {
        let indexer = create_test_indexer();
        let llm = Arc::new(MockLlmProvider::new("Streamed response"));

        let mut agent = PythiaAgent::new(indexer, llm, None);
        let (mut stream, finish_callback) = agent.ask_stream("Question", false).await.unwrap();

        // Collect stream
        let mut response = String::new();
        while let Some(chunk) = stream.next().await {
            if let Ok(text) = chunk {
                response.push_str(&text);
            }
        }

        assert_eq!(response, "Streamed response");

        // Call finish callback to add assistant message
        finish_callback(response);

        // Now check conversation after callback is consumed
        assert_eq!(agent.conversation.len(), 2);
        assert_eq!(agent.conversation[0].role, "user");
    }

    #[test]
    fn test_llm_config_in_agent() {
        let indexer = create_test_indexer();
        let llm = Arc::new(MockLlmProvider::new("test"));

        let config = LlmConfig {
            model: "gpt-4".to_string(),
            temperature: 0.7,
            max_tokens: 2000,
            top_p: None,
        };

        let agent = PythiaAgent::new(indexer, llm, Some(config));

        assert_eq!(agent.config.model, "gpt-4");
        assert_eq!(agent.config.temperature, 0.7);
        assert_eq!(agent.config.max_tokens, 2000);
        assert!(agent.config.top_p.is_none());
    }
}
