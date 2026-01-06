//! MCP Tool implementations for Islands

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use islands_indexer::IndexerService;
use islands_providers::Repository;

use crate::error::{Error, Result};
use crate::protocol::{CallToolResult, ContentItem, Tool};

/// Islands tool handler
pub struct IslandsTools {
    indexer: Arc<IndexerService>,
}

impl IslandsTools {
    /// Create a new tools handler
    #[must_use]
    pub fn new(indexer: Arc<IndexerService>) -> Self {
        Self { indexer }
    }

    /// Get the list of available tools
    #[must_use]
    pub fn list_tools(&self) -> Vec<Tool> {
        vec![
            Tool {
                name: "islands_list".to_string(),
                description: "List all indexed codebases available for search".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            },
            Tool {
                name: "islands_search".to_string(),
                description: "Semantic search across indexed codebases. Returns relevant code snippets and documentation.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query to search for in the codebase"
                        },
                        "indexes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of index names to search. If not provided, searches all indexes."
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }),
            },
            Tool {
                name: "islands_add_repo".to_string(),
                description: "Add a repository to be indexed by URL.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Repository URL (e.g., https://github.com/owner/repo)"
                        }
                    },
                    "required": ["url"]
                }),
            },
            Tool {
                name: "islands_sync".to_string(),
                description: "Sync and re-index a repository to get latest changes".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "index_name": {
                            "type": "string",
                            "description": "Index name in format 'provider/owner/repo'"
                        }
                    },
                    "required": ["index_name"]
                }),
            },
            Tool {
                name: "islands_status".to_string(),
                description: "Get the status of a specific index or all indexes".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "index_name": {
                            "type": "string",
                            "description": "Optional index name to get status for"
                        }
                    },
                    "required": []
                }),
            },
        ]
    }

    /// Call a tool by name
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<CallToolResult> {
        match name {
            "islands_list" => self.handle_list().await,
            "islands_search" => self.handle_search(arguments).await,
            "islands_add_repo" => self.handle_add_repo(arguments).await,
            "islands_sync" => self.handle_sync(arguments).await,
            "islands_status" => self.handle_status(arguments).await,
            _ => Err(Error::ToolNotFound(name.to_string())),
        }
    }

    /// Handle islands_list tool
    async fn handle_list(&self) -> Result<CallToolResult> {
        let indexes = self.indexer.list_indexes().await;

        if indexes.is_empty() {
            return Ok(CallToolResult {
                content: vec![ContentItem::text(
                    "No indexes available. Add repositories using islands_add_repo.",
                )],
                is_error: false,
            });
        }

        let mut output = String::from("Available Islands Indexes:\n\n");

        for info in indexes {
            output.push_str(&format!("- **{}**\n", info.name));
            output.push_str(&format!("  Provider: {}\n", info.repository.provider));
            output.push_str(&format!(
                "  Files: {}, Size: {:.2} MB\n",
                info.file_count,
                info.size_bytes as f64 / (1024.0 * 1024.0)
            ));
            output.push_str(&format!("  Updated: {}\n\n", info.updated_at));
        }

        Ok(CallToolResult {
            content: vec![ContentItem::text(output)],
            is_error: false,
        })
    }

    /// Handle islands_search tool
    async fn handle_search(&self, arguments: Value) -> Result<CallToolResult> {
        #[derive(Deserialize)]
        struct SearchArgs {
            query: String,
            indexes: Option<Vec<String>>,
            top_k: Option<usize>,
        }

        let args: SearchArgs =
            serde_json::from_value(arguments).map_err(|e| Error::InvalidParams(e.to_string()))?;

        let results = self
            .indexer
            .search(
                &args.query,
                args.indexes.as_deref(),
                args.top_k.unwrap_or(10),
            )
            .await?;

        if results.is_empty() {
            return Ok(CallToolResult {
                content: vec![ContentItem::text(format!(
                    "No results found for: {}",
                    args.query
                ))],
                is_error: false,
            });
        }

        let mut output = format!("Search results for: **{}**\n\n", args.query);

        for (i, result) in results.iter().enumerate() {
            output.push_str(&format!("### Result {}\n", i + 1));
            output.push_str(&format!(
                "**Repository:** {}\n",
                result
                    .get("repository")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
            ));
            if let Some(file) = result.get("file").and_then(|v| v.as_str()) {
                output.push_str(&format!("**File:** {}\n", file));
            }
            if let Some(score) = result.get("score").and_then(|v| v.as_f64()) {
                output.push_str(&format!("**Score:** {:.4}\n", score));
            }
            if let Some(text) = result.get("content").and_then(|v| v.as_str()) {
                let truncated = if text.len() > 500 {
                    format!("{}...", &text[..500])
                } else {
                    text.to_string()
                };
                output.push_str(&format!("```\n{}\n```\n\n", truncated));
            }
        }

        Ok(CallToolResult {
            content: vec![ContentItem::text(output)],
            is_error: false,
        })
    }

    /// Handle islands_add_repo tool
    async fn handle_add_repo(&self, arguments: Value) -> Result<CallToolResult> {
        #[derive(Deserialize)]
        struct AddRepoArgs {
            url: String,
        }

        let args: AddRepoArgs =
            serde_json::from_value(arguments).map_err(|e| Error::InvalidParams(e.to_string()))?;

        // Parse URL to get provider info
        let (provider_type, owner, name, _base_url) =
            islands_providers::factory::parse_repo_url(&args.url)?;

        // Create minimal repository info
        let repo = Repository::new(
            &format!("{:?}", provider_type).to_lowercase(),
            &owner,
            &name,
            &args.url,
        );

        match self.indexer.add_repository(&repo).await {
            Ok(state) => Ok(CallToolResult {
                content: vec![ContentItem::text(format!(
                    "Successfully added and indexed {}\nCommit: {}\nPath: {:?}",
                    repo.full_name,
                    state.last_commit.unwrap_or_default(),
                    state.local_path
                ))],
                is_error: false,
            }),
            Err(e) => Ok(CallToolResult {
                content: vec![ContentItem::text(format!(
                    "Failed to add repository: {}",
                    e
                ))],
                is_error: true,
            }),
        }
    }

    /// Handle islands_sync tool
    async fn handle_sync(&self, arguments: Value) -> Result<CallToolResult> {
        #[derive(Deserialize)]
        struct SyncArgs {
            index_name: String,
        }

        let args: SyncArgs =
            serde_json::from_value(arguments).map_err(|e| Error::InvalidParams(e.to_string()))?;

        let info = self
            .indexer
            .get_index(&args.index_name)
            .await
            .ok_or_else(|| {
                Error::Indexer(islands_indexer::Error::index_not_found(&args.index_name))
            })?;

        match self.indexer.sync_repository(&info.repository).await {
            Ok(state) => Ok(CallToolResult {
                content: vec![ContentItem::text(format!(
                    "Synced {}\nCommit: {}\nIndexed: {}",
                    info.repository.full_name,
                    state.last_commit.unwrap_or_default(),
                    state.indexed
                ))],
                is_error: false,
            }),
            Err(e) => Ok(CallToolResult {
                content: vec![ContentItem::text(format!("Sync failed: {}", e))],
                is_error: true,
            }),
        }
    }

    /// Handle islands_status tool
    async fn handle_status(&self, arguments: Value) -> Result<CallToolResult> {
        #[derive(Deserialize)]
        struct StatusArgs {
            index_name: Option<String>,
        }

        let args: StatusArgs =
            serde_json::from_value(arguments).map_err(|e| Error::InvalidParams(e.to_string()))?;

        if let Some(name) = args.index_name {
            let info = self.indexer.get_index(&name).await;

            match info {
                Some(info) => {
                    let state = self
                        .indexer
                        .repository_manager()
                        .get_state(&info.repository)
                        .await;

                    let status = json!({
                        "name": info.name,
                        "repository": info.repository.full_name,
                        "file_count": info.file_count,
                        "size_bytes": info.size_bytes,
                        "updated_at": info.updated_at.to_rfc3339(),
                        "last_commit": state.as_ref().and_then(|s| s.last_commit.clone()),
                        "indexed": state.as_ref().map(|s| s.indexed).unwrap_or(false),
                        "error": state.as_ref().and_then(|s| s.error.clone()),
                    });

                    Ok(CallToolResult {
                        content: vec![ContentItem::text(
                            serde_json::to_string_pretty(&status).unwrap(),
                        )],
                        is_error: false,
                    })
                }
                None => Ok(CallToolResult {
                    content: vec![ContentItem::text(format!("Index not found: {}", name))],
                    is_error: true,
                }),
            }
        } else {
            let indexes = self.indexer.list_indexes().await;
            let mut statuses = Vec::new();

            for info in indexes {
                let state = self
                    .indexer
                    .repository_manager()
                    .get_state(&info.repository)
                    .await;

                statuses.push(json!({
                    "name": info.name,
                    "indexed": state.as_ref().map(|s| s.indexed).unwrap_or(false),
                    "error": state.as_ref().and_then(|s| s.error.clone()),
                }));
            }

            Ok(CallToolResult {
                content: vec![ContentItem::text(
                    serde_json::to_string_pretty(&statuses).unwrap(),
                )],
                is_error: false,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use islands_indexer::service::IndexerConfig;
    use std::collections::HashMap;
    use tempfile::tempdir;

    fn create_test_service() -> Arc<IndexerService> {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        Arc::new(IndexerService::new(config, HashMap::new()))
    }

    #[test]
    fn test_islands_tools_new() {
        let indexer = create_test_service();
        let _tools = IslandsTools::new(indexer);
    }

    #[test]
    fn test_list_tools() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);
        let list = tools.list_tools();

        assert_eq!(list.len(), 5);

        let names: Vec<_> = list.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"islands_list"));
        assert!(names.contains(&"islands_search"));
        assert!(names.contains(&"islands_add_repo"));
        assert!(names.contains(&"islands_sync"));
        assert!(names.contains(&"islands_status"));
    }

    #[test]
    fn test_list_tools_have_schemas() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);

        for tool in tools.list_tools() {
            assert!(!tool.description.is_empty());
            assert!(tool.input_schema.is_object());
            assert_eq!(tool.input_schema["type"], "object");
        }
    }

    #[test]
    fn test_islands_search_schema() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);
        let list = tools.list_tools();

        let search_tool = list.iter().find(|t| t.name == "islands_search").unwrap();

        let props = &search_tool.input_schema["properties"];
        assert!(props["query"].is_object());
        assert!(props["indexes"].is_object());
        assert!(props["top_k"].is_object());

        let required = &search_tool.input_schema["required"];
        assert!(required.as_array().unwrap().contains(&json!("query")));
    }

    #[tokio::test]
    async fn test_call_tool_unknown() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);

        let result = tools.call_tool("unknown_tool", json!({})).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::ToolNotFound(_)));
    }

    #[tokio::test]
    async fn test_handle_list_empty() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);

        let result = tools.call_tool("islands_list", json!({})).await.unwrap();

        assert!(!result.is_error);
        assert!(!result.content.is_empty());

        if let ContentItem::Text { text } = &result.content[0] {
            assert!(text.contains("No indexes available"));
        } else {
            panic!("Expected text content");
        }
    }

    #[tokio::test]
    async fn test_handle_search_empty() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);

        let result = tools
            .call_tool("islands_search", json!({"query": "test"}))
            .await
            .unwrap();

        assert!(!result.is_error);
        if let ContentItem::Text { text } = &result.content[0] {
            assert!(text.contains("No results found"));
        }
    }

    #[tokio::test]
    async fn test_handle_search_invalid_params() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);

        // Missing required 'query' field
        let result = tools.call_tool("islands_search", json!({})).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::InvalidParams(_)));
    }

    #[tokio::test]
    async fn test_handle_status_all() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);

        let result = tools.call_tool("islands_status", json!({})).await.unwrap();

        assert!(!result.is_error);
        if let ContentItem::Text { text } = &result.content[0] {
            // Empty list of statuses
            let statuses: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
            assert!(statuses.is_empty());
        }
    }

    #[tokio::test]
    async fn test_handle_status_not_found() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);

        let result = tools
            .call_tool("islands_status", json!({"index_name": "nonexistent"}))
            .await
            .unwrap();

        assert!(result.is_error);
        if let ContentItem::Text { text } = &result.content[0] {
            assert!(text.contains("Index not found"));
        }
    }

    #[tokio::test]
    async fn test_handle_sync_not_found() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);

        let result = tools
            .call_tool("islands_sync", json!({"index_name": "nonexistent"}))
            .await;

        // Should return an error since index doesn't exist
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_add_repo_invalid_url() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);

        let result = tools
            .call_tool("islands_add_repo", json!({"url": "not-a-valid-url"}))
            .await;

        // Should fail to parse invalid URL
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_add_repo_missing_url() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);

        let result = tools.call_tool("islands_add_repo", json!({})).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::InvalidParams(_)));
    }

    #[tokio::test]
    async fn test_handle_sync_missing_index_name() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);

        let result = tools.call_tool("islands_sync", json!({})).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::InvalidParams(_)));
    }

    #[tokio::test]
    async fn test_handle_search_with_top_k() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);

        let result = tools
            .call_tool("islands_search", json!({"query": "test", "top_k": 5}))
            .await
            .unwrap();

        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn test_handle_search_with_indexes() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);

        let result = tools
            .call_tool(
                "islands_search",
                json!({"query": "test", "indexes": ["repo1", "repo2"]}),
            )
            .await
            .unwrap();

        assert!(!result.is_error);
    }

    #[test]
    fn test_tool_list_schema() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);
        let list = tools.list_tools();

        let list_tool = list.iter().find(|t| t.name == "islands_list").unwrap();
        assert_eq!(list_tool.input_schema["type"], "object");
        assert!(list_tool.input_schema["properties"].is_object());
    }

    #[test]
    fn test_tool_add_repo_schema() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);
        let list = tools.list_tools();

        let tool = list.iter().find(|t| t.name == "islands_add_repo").unwrap();
        let props = &tool.input_schema["properties"];
        assert!(props["url"].is_object());
        assert!(tool.input_schema["required"]
            .as_array()
            .unwrap()
            .contains(&json!("url")));
    }

    #[test]
    fn test_tool_sync_schema() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);
        let list = tools.list_tools();

        let tool = list.iter().find(|t| t.name == "islands_sync").unwrap();
        let props = &tool.input_schema["properties"];
        assert!(props["index_name"].is_object());
        assert!(tool.input_schema["required"]
            .as_array()
            .unwrap()
            .contains(&json!("index_name")));
    }

    #[test]
    fn test_tool_status_schema() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);
        let list = tools.list_tools();

        let tool = list.iter().find(|t| t.name == "islands_status").unwrap();
        let props = &tool.input_schema["properties"];
        assert!(props["index_name"].is_object());
        // index_name is optional, so required should be empty
        assert!(tool.input_schema["required"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_handle_status_with_invalid_json() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);

        // Send an array instead of object for index_name
        let result = tools
            .call_tool("islands_status", json!({"index_name": []}))
            .await;

        assert!(result.is_err());
    }

    #[test]
    fn test_tool_descriptions_not_empty() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);

        for tool in tools.list_tools() {
            assert!(!tool.description.is_empty(), "Tool {} has empty description", tool.name);
            assert!(tool.description.len() > 10, "Tool {} has too short description", tool.name);
        }
    }

    #[tokio::test]
    async fn test_all_tools_callable() {
        let indexer = create_test_service();
        let tools = IslandsTools::new(indexer);

        // Test that each tool is callable (may error but shouldn't panic)
        let _ = tools.call_tool("islands_list", json!({})).await;
        let _ = tools.call_tool("islands_search", json!({"query": "test"})).await;
        let _ = tools.call_tool("islands_add_repo", json!({"url": "invalid"})).await;
        let _ = tools.call_tool("islands_sync", json!({"index_name": "test"})).await;
        let _ = tools.call_tool("islands_status", json!({})).await;
    }
}
