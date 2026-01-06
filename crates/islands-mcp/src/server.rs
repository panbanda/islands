//! MCP Server implementation

use std::sync::Arc;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{debug, info};

use islands_indexer::IndexerService;

use crate::error::{Error, Result};
use crate::protocol::{
    CallToolParams, CallToolResult, ContentItem, InitializeResult, MCP_VERSION, Request, Response,
    RpcError, ServerCapabilities, ServerInfo, ToolsCapability,
};
use crate::tools::IslandsTools;

/// MCP Server for Islands
pub struct McpServer {
    indexer: Arc<IndexerService>,
    tools: IslandsTools,
    initialized: bool,
}

impl McpServer {
    /// Create a new MCP server
    #[must_use]
    pub fn new(indexer: Arc<IndexerService>) -> Self {
        let tools = IslandsTools::new(Arc::clone(&indexer));

        Self {
            indexer,
            tools,
            initialized: false,
        }
    }

    /// Run the server using stdio
    pub async fn run_stdio(&mut self) -> Result<()> {
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let reader = BufReader::new(stdin);
        let mut lines = reader.lines();

        info!("Islands MCP server starting on stdio");

        while let Ok(Some(line)) = lines.next_line().await {
            if line.trim().is_empty() {
                continue;
            }

            debug!("Received: {}", line);

            let response = match serde_json::from_str::<Request>(&line) {
                Ok(request) => self.handle_request(request).await,
                Err(e) => Response::error(
                    Value::Null,
                    RpcError {
                        code: Error::PARSE_ERROR,
                        message: format!("Parse error: {}", e),
                        data: None,
                    },
                ),
            };

            let response_json = serde_json::to_string(&response)?;
            debug!("Sending: {}", response_json);

            stdout.write_all(response_json.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }

        info!("MCP server shutting down");
        Ok(())
    }

    /// Handle a single request
    async fn handle_request(&mut self, request: Request) -> Response {
        match request.method.as_str() {
            "initialize" => self.handle_initialize(request.id, request.params).await,
            "initialized" => {
                // Notification, no response needed
                Response::success(request.id, Value::Null)
            }
            "tools/list" => self.handle_tools_list(request.id).await,
            "tools/call" => self.handle_tools_call(request.id, request.params).await,
            "shutdown" => {
                info!("Shutdown requested");
                Response::success(request.id, Value::Null)
            }
            method => Response::error(
                request.id,
                RpcError {
                    code: Error::METHOD_NOT_FOUND,
                    message: format!("Method not found: {}", method),
                    data: None,
                },
            ),
        }
    }

    /// Handle initialize request
    async fn handle_initialize(&mut self, id: Value, _params: Value) -> Response {
        self.initialized = true;

        let result = InitializeResult {
            protocol_version: MCP_VERSION.to_string(),
            capabilities: ServerCapabilities {
                tools: Some(ToolsCapability {
                    list_changed: false,
                }),
                resources: None,
                prompts: None,
            },
            server_info: ServerInfo {
                name: "islands".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
        };

        Response::success(id, serde_json::to_value(result).unwrap())
    }

    /// Handle tools/list request
    async fn handle_tools_list(&self, id: Value) -> Response {
        let tools = self.tools.list_tools();
        Response::success(id, json!({ "tools": tools }))
    }

    /// Handle tools/call request
    async fn handle_tools_call(&self, id: Value, params: Value) -> Response {
        let call_params: CallToolParams = match serde_json::from_value(params) {
            Ok(p) => p,
            Err(e) => {
                return Response::error(
                    id,
                    RpcError {
                        code: Error::INVALID_PARAMS,
                        message: format!("Invalid params: {}", e),
                        data: None,
                    },
                );
            }
        };

        match self
            .tools
            .call_tool(&call_params.name, call_params.arguments)
            .await
        {
            Ok(result) => Response::success(id, serde_json::to_value(result).unwrap()),
            Err(e) => {
                let result = CallToolResult {
                    content: vec![ContentItem::text(format!("Error: {}", e))],
                    is_error: true,
                };
                Response::success(id, serde_json::to_value(result).unwrap())
            }
        }
    }
}

/// Entry point for running the MCP server
pub async fn run_server(indexer: Arc<IndexerService>) -> Result<()> {
    let mut server = McpServer::new(indexer);
    server.run_stdio().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use islands_indexer::service::IndexerConfig;
    use serde_json::json;
    use std::collections::HashMap;

    fn create_test_server() -> McpServer {
        let config = IndexerConfig::default();
        let indexer = Arc::new(IndexerService::new(config, HashMap::new()));
        McpServer::new(indexer)
    }

    #[test]
    fn test_server_new() {
        let server = create_test_server();

        assert!(!server.initialized);
    }

    #[tokio::test]
    async fn test_handle_initialize() {
        let mut server = create_test_server();

        let request = Request {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            method: "initialize".to_string(),
            params: json!({}),
        };

        let response = server.handle_request(request).await;

        assert!(server.initialized);
        assert!(response.result.is_some());
        assert!(response.error.is_none());

        let result = response.result.unwrap();
        // Note: camelCase due to serde(rename_all = "camelCase")
        assert!(result.get("protocolVersion").is_some());
        assert!(result.get("capabilities").is_some());
        assert!(result.get("serverInfo").is_some());

        let server_info = result.get("serverInfo").unwrap();
        assert_eq!(server_info.get("name").unwrap(), "islands");
    }

    #[tokio::test]
    async fn test_handle_initialized_notification() {
        let mut server = create_test_server();

        let request = Request {
            jsonrpc: "2.0".to_string(),
            id: json!(2),
            method: "initialized".to_string(),
            params: json!({}),
        };

        let response = server.handle_request(request).await;

        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[tokio::test]
    async fn test_handle_tools_list() {
        let mut server = create_test_server();

        let request = Request {
            jsonrpc: "2.0".to_string(),
            id: json!(3),
            method: "tools/list".to_string(),
            params: json!({}),
        };

        let response = server.handle_request(request).await;

        assert!(response.result.is_some());
        assert!(response.error.is_none());

        let result = response.result.unwrap();
        let tools = result.get("tools").unwrap().as_array().unwrap();
        assert!(!tools.is_empty());

        // Verify expected tools exist
        let tool_names: Vec<&str> = tools
            .iter()
            .map(|t| t.get("name").unwrap().as_str().unwrap())
            .collect();

        assert!(tool_names.contains(&"islands_search"));
        assert!(tool_names.contains(&"islands_add_repo"));
        assert!(tool_names.contains(&"islands_list"));
    }

    #[tokio::test]
    async fn test_handle_tools_call_valid() {
        let mut server = create_test_server();

        let request = Request {
            jsonrpc: "2.0".to_string(),
            id: json!(4),
            method: "tools/call".to_string(),
            params: json!({
                "name": "islands_list",
                "arguments": {}
            }),
        };

        let response = server.handle_request(request).await;

        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[tokio::test]
    async fn test_handle_tools_call_unknown_tool() {
        let mut server = create_test_server();

        let request = Request {
            jsonrpc: "2.0".to_string(),
            id: json!(5),
            method: "tools/call".to_string(),
            params: json!({
                "name": "unknown_tool",
                "arguments": {}
            }),
        };

        let response = server.handle_request(request).await;

        // Unknown tool returns success with error in content
        assert!(response.result.is_some());
        let result = response.result.unwrap();
        assert_eq!(result.get("is_error").unwrap(), true);
    }

    #[tokio::test]
    async fn test_handle_tools_call_invalid_params() {
        let mut server = create_test_server();

        let request = Request {
            jsonrpc: "2.0".to_string(),
            id: json!(6),
            method: "tools/call".to_string(),
            params: json!("invalid"),
        };

        let response = server.handle_request(request).await;

        assert!(response.error.is_some());
        let error = response.error.unwrap();
        assert_eq!(error.code, Error::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn test_handle_shutdown() {
        let mut server = create_test_server();

        let request = Request {
            jsonrpc: "2.0".to_string(),
            id: json!(7),
            method: "shutdown".to_string(),
            params: json!({}),
        };

        let response = server.handle_request(request).await;

        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[tokio::test]
    async fn test_handle_unknown_method() {
        let mut server = create_test_server();

        let request = Request {
            jsonrpc: "2.0".to_string(),
            id: json!(8),
            method: "unknown/method".to_string(),
            params: json!({}),
        };

        let response = server.handle_request(request).await;

        assert!(response.error.is_some());
        let error = response.error.unwrap();
        assert_eq!(error.code, Error::METHOD_NOT_FOUND);
        assert!(error.message.contains("unknown/method"));
    }

    #[tokio::test]
    async fn test_handle_search_tool() {
        let mut server = create_test_server();

        let request = Request {
            jsonrpc: "2.0".to_string(),
            id: json!(9),
            method: "tools/call".to_string(),
            params: json!({
                "name": "islands_search",
                "arguments": {
                    "query": "test search"
                }
            }),
        };

        let response = server.handle_request(request).await;

        assert!(response.result.is_some());
    }

    #[tokio::test]
    async fn test_handle_status_tool() {
        let mut server = create_test_server();

        let request = Request {
            jsonrpc: "2.0".to_string(),
            id: json!(10),
            method: "tools/call".to_string(),
            params: json!({
                "name": "islands_status",
                "arguments": {}
            }),
        };

        let response = server.handle_request(request).await;

        assert!(response.result.is_some());
    }

    #[tokio::test]
    async fn test_initialize_result_structure() {
        let mut server = create_test_server();

        let response = server.handle_initialize(json!(1), json!({})).await;

        let result = response.result.unwrap();

        // Verify protocol version (camelCase due to serde)
        assert_eq!(
            result.get("protocolVersion").unwrap().as_str().unwrap(),
            MCP_VERSION
        );

        // Verify capabilities structure
        let capabilities = result.get("capabilities").unwrap();
        assert!(capabilities.get("tools").is_some());

        // Verify server info (camelCase)
        let server_info = result.get("serverInfo").unwrap();
        assert_eq!(
            server_info.get("name").unwrap().as_str().unwrap(),
            "islands"
        );
        assert!(server_info.get("version").is_some());
    }

    #[tokio::test]
    async fn test_tools_list_structure() {
        let server = create_test_server();

        let response = server.handle_tools_list(json!(1)).await;

        let result = response.result.unwrap();
        let tools = result.get("tools").unwrap().as_array().unwrap();

        for tool in tools {
            // Each tool should have name and description
            assert!(tool.get("name").is_some());
            assert!(tool.get("description").is_some());
            assert!(tool.get("inputSchema").is_some());
        }
    }

    #[test]
    fn test_server_not_initialized_by_default() {
        let server = create_test_server();
        assert!(!server.initialized);
    }

    #[tokio::test]
    async fn test_multiple_requests_same_server() {
        let mut server = create_test_server();

        // Initialize
        let init_req = Request {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            method: "initialize".to_string(),
            params: json!({}),
        };
        server.handle_request(init_req).await;

        // List tools
        let list_req = Request {
            jsonrpc: "2.0".to_string(),
            id: json!(2),
            method: "tools/list".to_string(),
            params: json!({}),
        };
        let response = server.handle_request(list_req).await;
        assert!(response.result.is_some());

        // Call a tool
        let call_req = Request {
            jsonrpc: "2.0".to_string(),
            id: json!(3),
            method: "tools/call".to_string(),
            params: json!({
                "name": "islands_list",
                "arguments": {}
            }),
        };
        let response = server.handle_request(call_req).await;
        assert!(response.result.is_some());
    }

    #[tokio::test]
    async fn test_response_id_preserved() {
        let mut server = create_test_server();

        let request = Request {
            jsonrpc: "2.0".to_string(),
            id: json!("custom-id-123"),
            method: "tools/list".to_string(),
            params: json!({}),
        };

        let response = server.handle_request(request).await;

        assert_eq!(response.id, json!("custom-id-123"));
    }

    #[tokio::test]
    async fn test_response_jsonrpc_version() {
        let mut server = create_test_server();

        let request = Request {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            method: "tools/list".to_string(),
            params: json!({}),
        };

        let response = server.handle_request(request).await;

        assert_eq!(response.jsonrpc, "2.0");
    }
}
