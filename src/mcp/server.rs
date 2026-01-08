//! MCP Server implementation

use std::sync::Arc;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{debug, info};

use crate::indexer::IndexerService;

use super::error::{Error, Result};
use super::protocol::{
    CallToolParams, CallToolResult, ContentItem, InitializeResult, MCP_VERSION, Request, Response,
    RpcError, ServerCapabilities, ServerInfo, ToolsCapability,
};
use super::tools::IslandsTools;

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
    use crate::indexer::service::IndexerConfig;
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

    // Tests for unusual request IDs
    mod request_id_tests {
        use super::*;

        #[tokio::test]
        async fn test_handle_request_with_null_id() {
            let mut server = create_test_server();

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: Value::Null,
                method: "tools/list".to_string(),
                params: json!({}),
            };

            let response = server.handle_request(request).await;

            assert_eq!(response.id, Value::Null);
            assert!(response.result.is_some());
        }

        #[tokio::test]
        async fn test_handle_request_with_numeric_id() {
            let mut server = create_test_server();

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(42),
                method: "tools/list".to_string(),
                params: json!({}),
            };

            let response = server.handle_request(request).await;

            assert_eq!(response.id, json!(42));
        }

        #[tokio::test]
        async fn test_handle_request_with_string_id() {
            let mut server = create_test_server();

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!("request-abc-123"),
                method: "tools/list".to_string(),
                params: json!({}),
            };

            let response = server.handle_request(request).await;

            assert_eq!(response.id, json!("request-abc-123"));
        }

        #[tokio::test]
        async fn test_handle_request_with_negative_id() {
            let mut server = create_test_server();

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(-1),
                method: "tools/list".to_string(),
                params: json!({}),
            };

            let response = server.handle_request(request).await;

            assert_eq!(response.id, json!(-1));
        }

        #[tokio::test]
        async fn test_handle_request_with_float_id() {
            let mut server = create_test_server();

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(1.23456789),
                method: "tools/list".to_string(),
                params: json!({}),
            };

            let response = server.handle_request(request).await;

            assert_eq!(response.id, json!(1.23456789));
        }

        #[tokio::test]
        async fn test_handle_request_with_large_id() {
            let mut server = create_test_server();

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(9_007_199_254_740_991_i64), // Max safe integer in JS
                method: "tools/list".to_string(),
                params: json!({}),
            };

            let response = server.handle_request(request).await;

            assert_eq!(response.id, json!(9_007_199_254_740_991_i64));
        }

        #[tokio::test]
        async fn test_error_response_preserves_id() {
            let mut server = create_test_server();

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!("error-test-id"),
                method: "nonexistent/method".to_string(),
                params: json!({}),
            };

            let response = server.handle_request(request).await;

            assert_eq!(response.id, json!("error-test-id"));
            assert!(response.error.is_some());
        }
    }

    // Tests for initialization behavior
    mod initialization_tests {
        use super::*;

        #[tokio::test]
        async fn test_multiple_initialize_calls() {
            let mut server = create_test_server();

            // First initialization
            let request1 = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(1),
                method: "initialize".to_string(),
                params: json!({}),
            };
            let response1 = server.handle_request(request1).await;
            assert!(server.initialized);
            assert!(response1.result.is_some());

            // Second initialization (should still work)
            let request2 = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(2),
                method: "initialize".to_string(),
                params: json!({}),
            };
            let response2 = server.handle_request(request2).await;
            assert!(server.initialized);
            assert!(response2.result.is_some());

            // Verify both responses have the same structure
            let result1 = response1.result.unwrap();
            let result2 = response2.result.unwrap();
            assert_eq!(
                result1.get("protocolVersion"),
                result2.get("protocolVersion")
            );
        }

        #[tokio::test]
        async fn test_initialize_with_params() {
            let mut server = create_test_server();

            // Initialize with client info (params are currently ignored but should not error)
            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(1),
                method: "initialize".to_string(),
                params: json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "test-client",
                        "version": "1.0.0"
                    }
                }),
            };

            let response = server.handle_request(request).await;

            assert!(server.initialized);
            assert!(response.result.is_some());
        }

        #[tokio::test]
        async fn test_tools_list_before_initialize() {
            let mut server = create_test_server();
            assert!(!server.initialized);

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(1),
                method: "tools/list".to_string(),
                params: json!({}),
            };

            // Should still work (server doesn't enforce initialization order)
            let response = server.handle_request(request).await;
            assert!(response.result.is_some());
        }

        #[tokio::test]
        async fn test_tools_call_before_initialize() {
            let mut server = create_test_server();
            assert!(!server.initialized);

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(1),
                method: "tools/call".to_string(),
                params: json!({
                    "name": "islands_list",
                    "arguments": {}
                }),
            };

            // Should still work
            let response = server.handle_request(request).await;
            assert!(response.result.is_some());
        }
    }

    // Tests for method handling edge cases
    mod method_handling_tests {
        use super::*;

        #[tokio::test]
        async fn test_handle_empty_method() {
            let mut server = create_test_server();

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(1),
                method: String::new(),
                params: json!({}),
            };

            let response = server.handle_request(request).await;

            assert!(response.error.is_some());
            assert_eq!(
                response.error.as_ref().unwrap().code,
                Error::METHOD_NOT_FOUND
            );
        }

        #[tokio::test]
        async fn test_handle_method_with_whitespace() {
            let mut server = create_test_server();

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(1),
                method: " tools/list ".to_string(),
                params: json!({}),
            };

            let response = server.handle_request(request).await;

            // Method names are matched exactly, so whitespace means not found
            assert!(response.error.is_some());
            assert_eq!(
                response.error.as_ref().unwrap().code,
                Error::METHOD_NOT_FOUND
            );
        }

        #[tokio::test]
        async fn test_handle_case_sensitive_method() {
            let mut server = create_test_server();

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(1),
                method: "TOOLS/LIST".to_string(),
                params: json!({}),
            };

            let response = server.handle_request(request).await;

            // Methods are case sensitive
            assert!(response.error.is_some());
            assert_eq!(
                response.error.as_ref().unwrap().code,
                Error::METHOD_NOT_FOUND
            );
        }

        #[tokio::test]
        async fn test_shutdown_response() {
            let mut server = create_test_server();

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(1),
                method: "shutdown".to_string(),
                params: json!({}),
            };

            let response = server.handle_request(request).await;

            assert!(response.result.is_some());
            assert_eq!(response.result.unwrap(), Value::Null);
        }

        #[tokio::test]
        async fn test_initialized_notification() {
            let mut server = create_test_server();

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(1),
                method: "initialized".to_string(),
                params: json!({}),
            };

            let response = server.handle_request(request).await;

            assert!(response.result.is_some());
            assert_eq!(response.result.unwrap(), Value::Null);
        }
    }

    // Tests for tools/call error handling
    mod tools_call_tests {
        use super::*;

        #[tokio::test]
        async fn test_tools_call_missing_name() {
            let mut server = create_test_server();

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(1),
                method: "tools/call".to_string(),
                params: json!({
                    "arguments": {}
                }),
            };

            let response = server.handle_request(request).await;

            assert!(response.error.is_some());
            assert_eq!(response.error.as_ref().unwrap().code, Error::INVALID_PARAMS);
        }

        #[tokio::test]
        async fn test_tools_call_with_array_params() {
            let mut server = create_test_server();

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(1),
                method: "tools/call".to_string(),
                params: json!(["not", "valid"]),
            };

            let response = server.handle_request(request).await;

            // Serde deserializes arrays positionally: first element -> name, second -> arguments
            // So this is actually a valid request for tool named "not" (which doesn't exist)
            // The response will have is_error=true in the result, not in response.error
            assert!(response.result.is_some());
            let result = response.result.unwrap();
            assert_eq!(result.get("is_error").unwrap(), true);
        }

        #[tokio::test]
        async fn test_tools_call_with_null_params() {
            let mut server = create_test_server();

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(1),
                method: "tools/call".to_string(),
                params: Value::Null,
            };

            let response = server.handle_request(request).await;

            assert!(response.error.is_some());
            assert_eq!(response.error.as_ref().unwrap().code, Error::INVALID_PARAMS);
        }

        #[tokio::test]
        async fn test_tools_call_with_empty_arguments() {
            let mut server = create_test_server();

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(1),
                method: "tools/call".to_string(),
                params: json!({
                    "name": "islands_list"
                    // arguments not provided, should default to null
                }),
            };

            let response = server.handle_request(request).await;

            // Should succeed - islands_list doesn't require arguments
            assert!(response.result.is_some());
        }

        #[tokio::test]
        async fn test_tools_call_unknown_tool_error_format() {
            let mut server = create_test_server();

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(1),
                method: "tools/call".to_string(),
                params: json!({
                    "name": "nonexistent_tool",
                    "arguments": {}
                }),
            };

            let response = server.handle_request(request).await;

            // Unknown tool returns success with is_error=true in result
            assert!(response.result.is_some());
            let result = response.result.unwrap();
            assert_eq!(result.get("is_error").unwrap(), true);

            let content = result.get("content").unwrap().as_array().unwrap();
            assert!(!content.is_empty());
        }
    }

    // Tests for response structure
    mod response_structure_tests {
        use super::*;

        #[tokio::test]
        async fn test_success_response_has_no_error() {
            let mut server = create_test_server();

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(1),
                method: "tools/list".to_string(),
                params: json!({}),
            };

            let response = server.handle_request(request).await;

            assert!(response.result.is_some());
            assert!(response.error.is_none());
        }

        #[tokio::test]
        async fn test_error_response_has_no_result() {
            let mut server = create_test_server();

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(1),
                method: "nonexistent".to_string(),
                params: json!({}),
            };

            let response = server.handle_request(request).await;

            assert!(response.result.is_none());
            assert!(response.error.is_some());
        }

        #[tokio::test]
        async fn test_response_serialization() {
            let mut server = create_test_server();

            let request = Request {
                jsonrpc: "2.0".to_string(),
                id: json!(1),
                method: "tools/list".to_string(),
                params: json!({}),
            };

            let response = server.handle_request(request).await;

            // Should serialize to valid JSON
            let json_str = serde_json::to_string(&response).unwrap();
            assert!(json_str.contains("jsonrpc"));
            assert!(json_str.contains("2.0"));
            assert!(json_str.contains("result"));
        }

        #[tokio::test]
        async fn test_initialize_response_structure() {
            let mut server = create_test_server();

            let response = server.handle_initialize(json!(1), json!({})).await;

            let result = response.result.unwrap();

            // Check required fields exist
            assert!(result.get("protocolVersion").is_some());
            assert!(result.get("capabilities").is_some());
            assert!(result.get("serverInfo").is_some());

            // Check server info
            let info = result.get("serverInfo").unwrap();
            assert_eq!(info.get("name").unwrap(), "islands");
            assert!(info.get("version").is_some());
        }

        #[tokio::test]
        async fn test_tools_list_response_structure() {
            let server = create_test_server();

            let response = server.handle_tools_list(json!(1)).await;

            let result = response.result.unwrap();
            let tools = result.get("tools").unwrap().as_array().unwrap();

            // Should have exactly 6 tools
            assert_eq!(tools.len(), 6);

            // Each tool should have required fields
            for tool in tools {
                assert!(tool.get("name").is_some());
                assert!(tool.get("description").is_some());
                assert!(tool.get("inputSchema").is_some());
            }
        }
    }

    // Tests for server construction
    mod construction_tests {
        use super::*;
        use tempfile::tempdir;

        #[test]
        fn test_server_new_with_custom_config() {
            let dir = tempdir().unwrap();
            let config = IndexerConfig {
                repos_path: dir.path().join("repos"),
                indexes_path: dir.path().join("indexes"),
                ..Default::default()
            };
            let indexer = Arc::new(IndexerService::new(config, HashMap::new()));
            let server = McpServer::new(indexer);

            assert!(!server.initialized);
        }

        #[test]
        fn test_run_server_function_compiles() {
            // Just verify the public API exists and compiles
            #[allow(clippy::type_complexity)]
            let _: fn(
                Arc<IndexerService>,
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<()>> + Send>,
            > = |indexer| Box::pin(async move { run_server(indexer).await });
        }
    }
}
