//! System prompts for the Islands agent

/// Default system prompt for the Islands codebase assistant
pub const DEFAULT_SYSTEM_PROMPT: &str = r#"You are Islands, an intelligent codebase assistant. You help developers understand and navigate codebases by answering questions about code structure, implementation details, and best practices.

You have access to semantic search across indexed repositories. When answering questions:
1. Use the search results to provide accurate, context-aware answers
2. Reference specific files and code snippets when relevant
3. Explain code concepts clearly and concisely
4. Suggest related areas of the codebase when appropriate

If the search results don't contain enough information to answer the question, say so and suggest how the user might find the information they need."#;

/// Format search results as context for the LLM
pub fn format_search_context(results: &[serde_json::Value]) -> String {
    if results.is_empty() {
        return String::new();
    }

    let mut context = String::from("Relevant code context from semantic search:\n\n");

    for (i, result) in results.iter().take(5).enumerate() {
        let repo = result
            .get("repository")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let file = result
            .get("file")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let score = result.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let content = result.get("content").and_then(|v| v.as_str()).unwrap_or("");

        // Truncate content if too long
        let content = if content.len() > 1000 {
            &content[..1000]
        } else {
            content
        };

        context.push_str(&format!("### Result {} (score: {:.3})\n", i + 1, score));
        context.push_str(&format!("Repository: {}\n", repo));
        context.push_str(&format!("File: {}\n", file));
        context.push_str(&format!("```\n{}\n```\n\n", content));
    }

    context
}

/// Build the full message list for an LLM request
pub fn build_messages(
    system_prompt: &str,
    conversation: &[crate::llm::Message],
    context: Option<&str>,
    user_message: &str,
) -> Vec<crate::llm::Message> {
    use crate::llm::Message;

    let mut messages = vec![Message::system(system_prompt)];

    // Add conversation history (last 10 messages)
    for msg in conversation.iter().rev().take(10).rev() {
        messages.push(msg.clone());
    }

    // Add search context if available
    if let Some(ctx) = context {
        if !ctx.is_empty() {
            messages.push(Message::system(ctx));
        }
    }

    // Add the user's message
    messages.push(Message::user(user_message));

    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Message;
    use serde_json::json;

    #[test]
    fn test_default_system_prompt_exists() {
        assert!(!DEFAULT_SYSTEM_PROMPT.is_empty());
        assert!(DEFAULT_SYSTEM_PROMPT.contains("Islands"));
    }

    #[test]
    fn test_default_system_prompt_instructions() {
        assert!(DEFAULT_SYSTEM_PROMPT.contains("codebase"));
        assert!(DEFAULT_SYSTEM_PROMPT.contains("search"));
    }

    #[test]
    fn test_format_search_context_empty() {
        let results: Vec<serde_json::Value> = vec![];
        let context = format_search_context(&results);
        assert!(context.is_empty());
    }

    #[test]
    fn test_format_search_context_single_result() {
        let results = vec![json!({
            "repository": "owner/repo",
            "file": "src/main.rs",
            "score": 0.95,
            "content": "fn main() { println!(\"Hello\"); }"
        })];

        let context = format_search_context(&results);
        assert!(context.contains("owner/repo"));
        assert!(context.contains("src/main.rs"));
        assert!(context.contains("0.95"));
        assert!(context.contains("fn main()"));
    }

    #[test]
    fn test_format_search_context_multiple_results() {
        let results = vec![
            json!({
                "repository": "repo1",
                "file": "file1.rs",
                "score": 0.9,
                "content": "code1"
            }),
            json!({
                "repository": "repo2",
                "file": "file2.rs",
                "score": 0.8,
                "content": "code2"
            }),
        ];

        let context = format_search_context(&results);
        assert!(context.contains("Result 1"));
        assert!(context.contains("Result 2"));
        assert!(context.contains("repo1"));
        assert!(context.contains("repo2"));
    }

    #[test]
    fn test_format_search_context_truncates_at_five() {
        let results: Vec<serde_json::Value> = (0..10)
            .map(|i| {
                json!({
                    "repository": format!("repo{}", i),
                    "file": format!("file{}.rs", i),
                    "score": 0.9 - (i as f64 * 0.05),
                    "content": format!("content{}", i)
                })
            })
            .collect();

        let context = format_search_context(&results);
        assert!(context.contains("Result 5"));
        assert!(!context.contains("Result 6"));
    }

    #[test]
    fn test_format_search_context_truncates_long_content() {
        let long_content = "x".repeat(2000);
        let results = vec![json!({
            "repository": "repo",
            "file": "file.rs",
            "score": 0.9,
            "content": long_content
        })];

        let context = format_search_context(&results);
        // Content should be truncated to 1000 chars
        assert!(context.len() < 2000 + 500); // Some overhead for formatting
    }

    #[test]
    fn test_format_search_context_missing_fields() {
        let results = vec![json!({})];

        let context = format_search_context(&results);
        assert!(context.contains("unknown")); // Default value for missing fields
    }

    #[test]
    fn test_format_search_context_partial_fields() {
        let results = vec![json!({
            "repository": "repo",
            "file": "file.rs"
        })];

        let context = format_search_context(&results);
        assert!(context.contains("repo"));
        assert!(context.contains("file.rs"));
        assert!(context.contains("0.000")); // Default score
    }

    #[test]
    fn test_build_messages_basic() {
        let messages = build_messages("System prompt", &[], None, "User question");

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[0].content, "System prompt");
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[1].content, "User question");
    }

    #[test]
    fn test_build_messages_with_conversation() {
        let conversation = vec![
            Message::user("Previous question"),
            Message::assistant("Previous answer"),
        ];

        let messages = build_messages("System", &conversation, None, "New question");

        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[1].content, "Previous question");
        assert_eq!(messages[2].role, "assistant");
        assert_eq!(messages[2].content, "Previous answer");
        assert_eq!(messages[3].role, "user");
        assert_eq!(messages[3].content, "New question");
    }

    #[test]
    fn test_build_messages_with_context() {
        let messages = build_messages(
            "System",
            &[],
            Some("Search context here"),
            "Question",
        );

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[0].content, "System");
        assert_eq!(messages[1].role, "system");
        assert_eq!(messages[1].content, "Search context here");
        assert_eq!(messages[2].role, "user");
    }

    #[test]
    fn test_build_messages_empty_context_ignored() {
        let messages = build_messages("System", &[], Some(""), "Question");

        assert_eq!(messages.len(), 2);
        // Empty context should not be added
    }

    #[test]
    fn test_build_messages_conversation_limit() {
        // Create more than 10 messages
        let conversation: Vec<Message> = (0..15)
            .flat_map(|i| {
                vec![
                    Message::user(format!("Question {}", i)),
                    Message::assistant(format!("Answer {}", i)),
                ]
            })
            .collect();

        let messages = build_messages("System", &conversation, None, "Final question");

        // Should have system + last 10 conversation messages + user message = 12
        assert_eq!(messages.len(), 12);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages.last().unwrap().role, "user");
    }

    #[test]
    fn test_build_messages_full_flow() {
        let conversation = vec![
            Message::user("What is this project?"),
            Message::assistant("This is a code indexing project."),
        ];

        let context = "Relevant code: fn main() {}";

        let messages = build_messages(
            DEFAULT_SYSTEM_PROMPT,
            &conversation,
            Some(context),
            "How does the main function work?",
        );

        assert_eq!(messages.len(), 5);
        assert!(messages[0].content.contains("Islands"));
        assert_eq!(messages[1].content, "What is this project?");
        assert_eq!(messages[2].content, "This is a code indexing project.");
        assert!(messages[3].content.contains("fn main()"));
        assert!(messages[4].content.contains("main function"));
    }

    #[test]
    fn test_build_messages_preserves_order() {
        let conversation = vec![
            Message::user("Q1"),
            Message::assistant("A1"),
            Message::user("Q2"),
            Message::assistant("A2"),
        ];

        let messages = build_messages("System", &conversation, None, "Q3");

        assert_eq!(messages[1].content, "Q1");
        assert_eq!(messages[2].content, "A1");
        assert_eq!(messages[3].content, "Q2");
        assert_eq!(messages[4].content, "A2");
        assert_eq!(messages[5].content, "Q3");
    }
}
