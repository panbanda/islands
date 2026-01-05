//! Error types for pythia-mcp

use thiserror::Error;

/// Result type alias for MCP operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for MCP server operations
#[derive(Error, Debug)]
pub enum Error {
    /// JSON-RPC protocol error
    #[error("JSON-RPC error: {code} - {message}")]
    JsonRpc {
        /// Error code
        code: i32,
        /// Error message
        message: String,
    },

    /// Invalid request
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// Method not found
    #[error("method not found: {0}")]
    MethodNotFound(String),

    /// Invalid params
    #[error("invalid params: {0}")]
    InvalidParams(String),

    /// Internal error
    #[error("internal error: {0}")]
    Internal(String),

    /// Tool not found
    #[error("tool not found: {0}")]
    ToolNotFound(String),

    /// Indexer error
    #[error("indexer error: {0}")]
    Indexer(#[from] pythia_indexer::Error),

    /// Provider error
    #[error("provider error: {0}")]
    Provider(#[from] pythia_providers::Error),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl Error {
    /// JSON-RPC standard error codes
    pub const PARSE_ERROR: i32 = -32700;
    /// Invalid request error code
    pub const INVALID_REQUEST: i32 = -32600;
    /// Method not found error code
    pub const METHOD_NOT_FOUND: i32 = -32601;
    /// Invalid params error code
    pub const INVALID_PARAMS: i32 = -32602;
    /// Internal error code
    pub const INTERNAL_ERROR: i32 = -32603;

    /// Get the JSON-RPC error code for this error
    #[must_use]
    pub fn code(&self) -> i32 {
        match self {
            Self::JsonRpc { code, .. } => *code,
            Self::InvalidRequest(_) => Self::INVALID_REQUEST,
            Self::MethodNotFound(_) => Self::METHOD_NOT_FOUND,
            Self::InvalidParams(_) => Self::INVALID_PARAMS,
            _ => Self::INTERNAL_ERROR,
        }
    }

    /// Create a JSON-RPC error response
    #[must_use]
    pub fn to_json_rpc(&self) -> serde_json::Value {
        serde_json::json!({
            "code": self.code(),
            "message": self.to_string()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        assert_eq!(Error::PARSE_ERROR, -32700);
        assert_eq!(Error::INVALID_REQUEST, -32600);
        assert_eq!(Error::METHOD_NOT_FOUND, -32601);
        assert_eq!(Error::INVALID_PARAMS, -32602);
        assert_eq!(Error::INTERNAL_ERROR, -32603);
    }

    #[test]
    fn test_error_code_method() {
        let err = Error::InvalidRequest("bad request".into());
        assert_eq!(err.code(), Error::INVALID_REQUEST);

        let err = Error::MethodNotFound("unknown".into());
        assert_eq!(err.code(), Error::METHOD_NOT_FOUND);

        let err = Error::InvalidParams("missing param".into());
        assert_eq!(err.code(), Error::INVALID_PARAMS);

        let err = Error::Internal("server error".into());
        assert_eq!(err.code(), Error::INTERNAL_ERROR);

        let err = Error::ToolNotFound("unknown_tool".into());
        assert_eq!(err.code(), Error::INTERNAL_ERROR);

        let err = Error::JsonRpc {
            code: -32000,
            message: "custom".into(),
        };
        assert_eq!(err.code(), -32000);
    }

    #[test]
    fn test_error_display() {
        let err = Error::InvalidRequest("bad request".into());
        assert_eq!(err.to_string(), "invalid request: bad request");

        let err = Error::MethodNotFound("tools/unknown".into());
        assert_eq!(err.to_string(), "method not found: tools/unknown");

        let err = Error::InvalidParams("missing query".into());
        assert_eq!(err.to_string(), "invalid params: missing query");

        let err = Error::Internal("database error".into());
        assert_eq!(err.to_string(), "internal error: database error");

        let err = Error::ToolNotFound("pythia_xyz".into());
        assert_eq!(err.to_string(), "tool not found: pythia_xyz");

        let err = Error::JsonRpc {
            code: -32000,
            message: "custom error".into(),
        };
        assert_eq!(err.to_string(), "JSON-RPC error: -32000 - custom error");
    }

    #[test]
    fn test_error_to_json_rpc() {
        let err = Error::InvalidRequest("bad request".into());
        let json = err.to_json_rpc();

        assert_eq!(json["code"], Error::INVALID_REQUEST);
        assert_eq!(json["message"], "invalid request: bad request");
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Io(_)));
        assert_eq!(err.code(), Error::INTERNAL_ERROR);
    }

    #[test]
    fn test_error_from_json() {
        let json_err = serde_json::from_str::<serde_json::Value>("invalid").unwrap_err();
        let err: Error = json_err.into();
        assert!(matches!(err, Error::Json(_)));
        assert_eq!(err.code(), Error::INTERNAL_ERROR);
    }

    #[test]
    fn test_error_debug() {
        let err = Error::ToolNotFound("test_tool".into());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("ToolNotFound"));
    }
}
