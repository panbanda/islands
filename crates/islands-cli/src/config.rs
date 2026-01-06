//! Configuration management

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use islands_indexer::service::IndexerConfig;

/// CLI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Debug mode
    pub debug: bool,
    /// Log level
    pub log_level: String,
    /// Indexer configuration
    pub indexer: IndexerConfig,
    /// OpenAI API key (optional)
    pub openai_api_key: Option<String>,
    /// MCP server host
    pub mcp_host: String,
    /// MCP server port
    pub mcp_port: u16,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            debug: false,
            log_level: "info".to_string(),
            indexer: IndexerConfig::default(),
            openai_api_key: None,
            mcp_host: "0.0.0.0".to_string(),
            mcp_port: 8080,
        }
    }
}

impl Config {
    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(val) = std::env::var("ISLANDS_DEBUG") {
            config.debug = val == "1" || val.to_lowercase() == "true";
        }

        if let Ok(val) = std::env::var("ISLANDS_LOG_LEVEL") {
            config.log_level = val;
        }

        if let Ok(val) = std::env::var("ISLANDS_REPOS_PATH") {
            config.indexer.repos_path = PathBuf::from(val);
        }

        if let Ok(val) = std::env::var("ISLANDS_INDEXES_PATH") {
            config.indexer.indexes_path = PathBuf::from(val);
        }

        if let Ok(val) = std::env::var("OPENAI_API_KEY") {
            config.openai_api_key = Some(val);
        } else if let Ok(val) = std::env::var("ISLANDS_OPENAI_API_KEY") {
            config.openai_api_key = Some(val);
        }

        config
    }

    /// Load configuration from a file
    pub fn from_file(path: &PathBuf) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)?;

        if path
            .extension()
            .map(|e| e == "yaml" || e == "yml")
            .unwrap_or(false)
        {
            serde_yaml::from_str(&content).map_err(|e| crate::Error::Config(e.to_string()))
        } else {
            serde_json::from_str(&content).map_err(crate::Error::Json)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert!(!config.debug);
        assert_eq!(config.log_level, "info");
        assert!(config.openai_api_key.is_none());
        assert_eq!(config.mcp_host, "0.0.0.0");
        assert_eq!(config.mcp_port, 8080);
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: Config = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.debug, config.debug);
        assert_eq!(parsed.log_level, config.log_level);
        assert_eq!(parsed.mcp_host, config.mcp_host);
        assert_eq!(parsed.mcp_port, config.mcp_port);
    }

    #[test]
    fn test_config_yaml_serialization() {
        let config = Config::default();
        let yaml = serde_yaml::to_string(&config).unwrap();
        let parsed: Config = serde_yaml::from_str(&yaml).unwrap();

        assert_eq!(parsed.debug, config.debug);
        assert_eq!(parsed.log_level, config.log_level);
    }

    #[test]
    fn test_config_from_file_json() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("islands_test_config.json");

        let config_json = r#"{
            "debug": true,
            "log_level": "debug",
            "indexer": {
                "repos_path": "/tmp/repos",
                "indexes_path": "/tmp/indexes",
                "max_concurrent_syncs": 4,
                "sync_interval_secs": 300,
                "index_extensions": ["rs", "py"]
            },
            "openai_api_key": "test-key",
            "mcp_host": "127.0.0.1",
            "mcp_port": 9000
        }"#;

        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(config_json.as_bytes()).unwrap();

        let config = Config::from_file(&file_path).unwrap();
        assert!(config.debug);
        assert_eq!(config.log_level, "debug");
        assert_eq!(config.openai_api_key, Some("test-key".to_string()));
        assert_eq!(config.mcp_host, "127.0.0.1");
        assert_eq!(config.mcp_port, 9000);

        std::fs::remove_file(&file_path).ok();
    }

    #[test]
    fn test_config_from_file_yaml() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("islands_test_config.yaml");

        let config_yaml = r#"
debug: true
log_level: warn
indexer:
  repos_path: /tmp/repos
  indexes_path: /tmp/indexes
  max_concurrent_syncs: 4
  sync_interval_secs: 300
  index_extensions:
    - rs
    - py
openai_api_key: yaml-key
mcp_host: localhost
mcp_port: 8888
"#;

        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(config_yaml.as_bytes()).unwrap();

        let config = Config::from_file(&file_path).unwrap();
        assert!(config.debug);
        assert_eq!(config.log_level, "warn");
        assert_eq!(config.openai_api_key, Some("yaml-key".to_string()));
        assert_eq!(config.mcp_host, "localhost");
        assert_eq!(config.mcp_port, 8888);

        std::fs::remove_file(&file_path).ok();
    }

    #[test]
    fn test_config_from_file_yml_extension() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("islands_test_config.yml");

        let config_yaml = r#"
debug: false
log_level: info
indexer:
  repos_path: /tmp/repos
  indexes_path: /tmp/indexes
  max_concurrent_syncs: 4
  sync_interval_secs: 300
  index_extensions:
    - rs
mcp_host: "0.0.0.0"
mcp_port: 8080
"#;

        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(config_yaml.as_bytes()).unwrap();

        let config = Config::from_file(&file_path).unwrap();
        assert!(!config.debug);
        assert_eq!(config.log_level, "info");

        std::fs::remove_file(&file_path).ok();
    }

    #[test]
    fn test_config_from_file_not_found() {
        let file_path = PathBuf::from("/nonexistent/path/config.json");
        let result = Config::from_file(&file_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_from_file_invalid_json() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("islands_invalid_config.json");

        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(b"{ invalid json }").unwrap();

        let result = Config::from_file(&file_path);
        assert!(result.is_err());

        std::fs::remove_file(&file_path).ok();
    }

    #[test]
    fn test_config_from_file_invalid_yaml() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("islands_invalid_config.yaml");

        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(b"invalid: yaml: content:").unwrap();

        let result = Config::from_file(&file_path);
        assert!(result.is_err());

        std::fs::remove_file(&file_path).ok();
    }

    #[test]
    fn test_config_clone() {
        let mut config = Config::default();
        config.debug = true;
        config.log_level = "error".to_string();
        config.mcp_port = 9999;

        let cloned = config.clone();
        assert_eq!(cloned.debug, config.debug);
        assert_eq!(cloned.log_level, config.log_level);
        assert_eq!(cloned.mcp_port, config.mcp_port);
    }

    #[test]
    fn test_config_debug() {
        let config = Config::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("Config"));
        assert!(debug_str.contains("debug"));
        assert!(debug_str.contains("log_level"));
    }

    #[test]
    fn test_config_with_openai_key() {
        let mut config = Config::default();
        config.openai_api_key = Some("sk-test123".to_string());

        assert!(config.openai_api_key.is_some());
        assert_eq!(config.openai_api_key.as_ref().unwrap(), "sk-test123");
    }
}
