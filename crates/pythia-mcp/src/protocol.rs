//! MCP Protocol types
//!
//! Defines the JSON-RPC message types for the Model Context Protocol.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON-RPC version string
pub const JSONRPC_VERSION: &str = "2.0";

/// MCP protocol version
pub const MCP_VERSION: &str = "2024-11-05";

/// A JSON-RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    /// JSON-RPC version (always "2.0")
    pub jsonrpc: String,
    /// Request ID (can be string or number)
    pub id: Value,
    /// Method name
    pub method: String,
    /// Method parameters
    #[serde(default)]
    pub params: Value,
}

/// A JSON-RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    /// JSON-RPC version (always "2.0")
    pub jsonrpc: String,
    /// Request ID (matches the request)
    pub id: Value,
    /// Result (present on success)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error (present on failure)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

impl Response {
    /// Create a success response
    #[must_use]
    pub fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Create an error response
    #[must_use]
    pub fn error(id: Value, error: RpcError) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id,
            result: None,
            error: Some(error),
        }
    }
}

/// A JSON-RPC error object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
    /// Error code
    pub code: i32,
    /// Error message
    pub message: String,
    /// Additional error data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// MCP initialization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    /// Protocol version the client supports
    pub protocol_version: String,
    /// Client capabilities
    pub capabilities: ClientCapabilities,
    /// Client information
    pub client_info: ClientInfo,
}

/// Client capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClientCapabilities {
    /// Root capabilities
    #[serde(default)]
    pub roots: Option<RootsCapability>,
    /// Sampling capability
    #[serde(default)]
    pub sampling: Option<Value>,
}

/// Roots capability
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RootsCapability {
    /// Whether list_changed notifications are supported
    #[serde(default)]
    pub list_changed: bool,
}

/// Client information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    /// Client name
    pub name: String,
    /// Client version
    pub version: String,
}

/// MCP initialization result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    /// Protocol version the server is using
    pub protocol_version: String,
    /// Server capabilities
    pub capabilities: ServerCapabilities,
    /// Server information
    pub server_info: ServerInfo,
}

/// Server capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerCapabilities {
    /// Tools capability
    #[serde(default)]
    pub tools: Option<ToolsCapability>,
    /// Resources capability
    #[serde(default)]
    pub resources: Option<Value>,
    /// Prompts capability
    #[serde(default)]
    pub prompts: Option<Value>,
}

/// Tools capability
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsCapability {
    /// Whether list_changed notifications are supported
    #[serde(default)]
    pub list_changed: bool,
}

/// Server information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    /// Server name
    pub name: String,
    /// Server version
    pub version: String,
}

/// A tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Input schema (JSON Schema)
    pub input_schema: Value,
}

/// Tool call parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallToolParams {
    /// Tool name
    pub name: String,
    /// Tool arguments
    #[serde(default)]
    pub arguments: Value,
}

/// Tool call result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallToolResult {
    /// Result content
    pub content: Vec<ContentItem>,
    /// Whether the tool encountered an error
    #[serde(default)]
    pub is_error: bool,
}

/// A content item in a tool result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ContentItem {
    /// Text content
    Text {
        /// The text content
        text: String,
    },
    /// Image content
    Image {
        /// Base64-encoded image data
        data: String,
        /// MIME type
        mime_type: String,
    },
    /// Resource reference
    Resource {
        /// Resource URI
        uri: String,
        /// Resource content
        text: Option<String>,
    },
}

impl ContentItem {
    /// Create a text content item
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_jsonrpc_version() {
        assert_eq!(JSONRPC_VERSION, "2.0");
    }

    #[test]
    fn test_mcp_version() {
        assert_eq!(MCP_VERSION, "2024-11-05");
    }

    #[test]
    fn test_request_serialization() {
        let request = Request {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: json!(1),
            method: "initialize".to_string(),
            params: json!({}),
        };

        let json = serde_json::to_string(&request).unwrap();
        let parsed: Request = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.jsonrpc, "2.0");
        assert_eq!(parsed.id, json!(1));
        assert_eq!(parsed.method, "initialize");
    }

    #[test]
    fn test_request_default_params() {
        let json = r#"{"jsonrpc":"2.0","id":1,"method":"test"}"#;
        let request: Request = serde_json::from_str(json).unwrap();

        assert_eq!(request.params, Value::Null);
    }

    #[test]
    fn test_response_success() {
        let response = Response::success(json!(1), json!({"result": "ok"}));

        assert_eq!(response.jsonrpc, "2.0");
        assert_eq!(response.id, json!(1));
        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[test]
    fn test_response_error() {
        let error = RpcError {
            code: -32600,
            message: "Invalid Request".to_string(),
            data: None,
        };
        let response = Response::error(json!(1), error);

        assert_eq!(response.jsonrpc, "2.0");
        assert!(response.result.is_none());
        assert!(response.error.is_some());
        assert_eq!(response.error.as_ref().unwrap().code, -32600);
    }

    #[test]
    fn test_response_serialization() {
        let response = Response::success(json!(1), json!({"data": "test"}));
        let json_str = serde_json::to_string(&response).unwrap();

        // Error should not be present when None
        assert!(!json_str.contains("error"));

        let parsed: Response = serde_json::from_str(&json_str).unwrap();
        assert!(parsed.result.is_some());
    }

    #[test]
    fn test_rpc_error_with_data() {
        let error = RpcError {
            code: -32000,
            message: "Server error".to_string(),
            data: Some(json!({"details": "more info"})),
        };

        let json = serde_json::to_string(&error).unwrap();
        assert!(json.contains("details"));
    }

    #[test]
    fn test_initialize_params() {
        let params = InitializeParams {
            protocol_version: MCP_VERSION.to_string(),
            capabilities: ClientCapabilities::default(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
        };

        let json = serde_json::to_string(&params).unwrap();
        let parsed: InitializeParams = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.protocol_version, MCP_VERSION);
        assert_eq!(parsed.client_info.name, "test-client");
    }

    #[test]
    fn test_initialize_result() {
        let result = InitializeResult {
            protocol_version: MCP_VERSION.to_string(),
            capabilities: ServerCapabilities {
                tools: Some(ToolsCapability { list_changed: true }),
                resources: None,
                prompts: None,
            },
            server_info: ServerInfo {
                name: "pythia".to_string(),
                version: "0.1.0".to_string(),
            },
        };

        let json = serde_json::to_string(&result).unwrap();
        let parsed: InitializeResult = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.server_info.name, "pythia");
        assert!(parsed.capabilities.tools.is_some());
    }

    #[test]
    fn test_tool_definition() {
        let tool = Tool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }),
        };

        let json = serde_json::to_string(&tool).unwrap();
        let parsed: Tool = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.name, "test_tool");
        assert!(parsed.input_schema["properties"]["query"].is_object());
    }

    #[test]
    fn test_call_tool_params() {
        let params = CallToolParams {
            name: "pythia_search".to_string(),
            arguments: json!({"query": "test"}),
        };

        let json = serde_json::to_string(&params).unwrap();
        let parsed: CallToolParams = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.name, "pythia_search");
        assert_eq!(parsed.arguments["query"], "test");
    }

    #[test]
    fn test_call_tool_params_default_arguments() {
        let json = r#"{"name":"test_tool"}"#;
        let params: CallToolParams = serde_json::from_str(json).unwrap();

        assert_eq!(params.arguments, Value::Null);
    }

    #[test]
    fn test_call_tool_result() {
        let result = CallToolResult {
            content: vec![ContentItem::text("Result text")],
            is_error: false,
        };

        let json = serde_json::to_string(&result).unwrap();
        let parsed: CallToolResult = serde_json::from_str(&json).unwrap();

        assert!(!parsed.is_error);
        assert_eq!(parsed.content.len(), 1);
    }

    #[test]
    fn test_call_tool_result_error() {
        let result = CallToolResult {
            content: vec![ContentItem::text("Error occurred")],
            is_error: true,
        };

        assert!(result.is_error);
    }

    #[test]
    fn test_content_item_text() {
        let item = ContentItem::text("Hello world");

        if let ContentItem::Text { text } = item {
            assert_eq!(text, "Hello world");
        } else {
            panic!("Expected Text variant");
        }
    }

    #[test]
    fn test_content_item_text_from_string() {
        let item = ContentItem::text(String::from("Dynamic string"));

        if let ContentItem::Text { text } = item {
            assert_eq!(text, "Dynamic string");
        } else {
            panic!("Expected Text variant");
        }
    }

    #[test]
    fn test_content_item_image_serialization() {
        let item = ContentItem::Image {
            data: "base64data".to_string(),
            mime_type: "image/png".to_string(),
        };

        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains(r#""type":"image""#));
        assert!(json.contains("base64data"));
    }

    #[test]
    fn test_content_item_resource_serialization() {
        let item = ContentItem::Resource {
            uri: "file:///test.txt".to_string(),
            text: Some("File contents".to_string()),
        };

        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains(r#""type":"resource""#));
        assert!(json.contains("file:///test.txt"));
    }

    #[test]
    fn test_client_capabilities_default() {
        let caps = ClientCapabilities::default();
        assert!(caps.roots.is_none());
        assert!(caps.sampling.is_none());
    }

    #[test]
    fn test_server_capabilities_default() {
        let caps = ServerCapabilities::default();
        assert!(caps.tools.is_none());
        assert!(caps.resources.is_none());
        assert!(caps.prompts.is_none());
    }

    #[test]
    fn test_tools_capability_default() {
        let cap = ToolsCapability::default();
        assert!(!cap.list_changed);
    }

    #[test]
    fn test_roots_capability_default() {
        let cap = RootsCapability::default();
        assert!(!cap.list_changed);
    }
}
