//! CLI command implementations

use std::collections::HashMap;
use std::sync::Arc;

use crate::indexer::service::IndexerService;
use crate::providers::factory::{create_provider, parse_repo_url};

use crate::Result;
use crate::config::Config;
use crate::output;

/// Add a repository to the index
pub async fn add_repository(config: &Config, url: &str, token: Option<&str>) -> Result<()> {
    let spinner = output::spinner(&format!("Adding repository: {}", url));

    // Parse URL
    let (provider_type, owner, name, base_url) = parse_repo_url(url)?;

    // Create provider
    let provider_type_str = format!("{:?}", provider_type).to_lowercase();
    let provider = create_provider(&provider_type_str, base_url.as_deref(), token, None)?;

    // Create indexer
    let mut providers = HashMap::new();
    providers.insert(provider.provider_name().to_string(), provider.clone());

    let indexer = IndexerService::new(config.indexer.clone(), providers);

    // Get repository info
    let repo = provider.get_repository(&owner, &name).await?;

    spinner.set_message(format!("Cloning {}...", repo.full_name));

    // Clone first (spinner for this phase)
    let state = indexer.repository_manager().clone_repository(&repo).await?;
    spinner.finish_and_clear();

    // Now index with progress bar
    let progress = output::progress_bar(0, "Indexing files...");
    indexer
        .index_repository_with_progress(&repo, Some(&progress))
        .await?;
    progress.finish_and_clear();

    output::success(&format!("Successfully indexed {}", repo.full_name));
    println!("  Commit: {}", state.last_commit.unwrap_or_default());
    println!("  Path: {:?}", state.local_path);

    Ok(())
}

/// Search across indexed repositories
pub async fn search(
    config: &Config,
    query: &str,
    indexes: Option<Vec<String>>,
    workspace: Option<&str>,
    top_k: usize,
) -> Result<()> {
    let indexer = IndexerService::new(config.indexer.clone(), HashMap::new());

    // Resolve workspace to index names if specified
    let search_indexes = if let Some(ws_name) = workspace {
        let ws_indexes = indexer
            .get_workspace_index_names(ws_name)
            .await
            .ok_or_else(|| {
                crate::Error::InvalidArgument(format!("Workspace not found: {}", ws_name))
            })?;
        Some(ws_indexes)
    } else {
        indexes
    };

    let results = indexer
        .search(query, search_indexes.as_deref(), top_k)
        .await?;

    if results.is_empty() {
        output::warning(&format!("No results found for: {}", query));
        return Ok(());
    }

    println!("\nSearch Results: {}\n", query);

    for (i, result) in results.iter().enumerate() {
        println!(
            "{} {} - {}",
            console::style(format!("{}.", i + 1)).cyan(),
            result
                .get("repository")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown"),
            result
                .get("file")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown"),
        );

        if let Some(score) = result.get("score").and_then(|v| v.as_f64()) {
            println!("   Score: {:.4}", score);
        }

        if let Some(content) = result.get("content").and_then(|v| v.as_str()) {
            let preview: String = content.chars().take(100).collect();
            println!("   {}", console::style(preview).dim());
        }

        println!();
    }

    Ok(())
}

/// List all indexed repositories
pub async fn list_indexes(config: &Config) -> Result<()> {
    let indexer = IndexerService::new(config.indexer.clone(), HashMap::new());

    let indexes = indexer.list_indexes().await;

    if indexes.is_empty() {
        output::warning("No indexes found. Use 'islands add' to add repositories.");
        return Ok(());
    }

    println!("\nIslands Indexes\n");

    for info in indexes {
        println!(
            "{} ({})",
            console::style(&info.name).cyan().bold(),
            info.repository.provider
        );
        println!(
            "  Files: {}, Size: {:.2} MB",
            info.file_count,
            info.size_bytes as f64 / (1024.0 * 1024.0)
        );
        println!("  Updated: {}", info.updated_at);
        println!();
    }

    Ok(())
}

/// Sync a repository
pub async fn sync_repository(config: &Config, index_name: &str) -> Result<()> {
    let spinner = output::spinner(&format!("Syncing {}", index_name));

    let indexer = IndexerService::new(config.indexer.clone(), HashMap::new());

    let info = indexer
        .get_index(index_name)
        .await
        .ok_or_else(|| crate::Error::InvalidArgument(format!("Index not found: {}", index_name)))?;

    // First update the repository
    let state = indexer
        .repository_manager()
        .update_repository(&info.repository)
        .await?;
    spinner.finish_and_clear();

    // Check if re-indexing is needed
    if indexer
        .repository_manager()
        .needs_reindex(&info.repository)
        .await
    {
        let progress = output::progress_bar(0, "Re-indexing files...");
        indexer
            .index_repository_with_progress(&info.repository, Some(&progress))
            .await?;
        progress.finish_and_clear();
    }

    output::success(&format!("Synced {}", index_name));
    println!("  Commit: {}", state.last_commit.unwrap_or_default());
    println!("  Indexed: {}", state.indexed);

    Ok(())
}

/// Start the MCP server
#[cfg(feature = "mcp")]
pub async fn serve_mcp(config: &Config) -> Result<()> {
    use crate::mcp::server::run_server;

    output::info("Starting Islands MCP server...");

    let indexer = Arc::new(IndexerService::new(config.indexer.clone(), HashMap::new()));

    run_server(indexer)
        .await
        .map_err(|e| crate::Error::Config(e.to_string()))?;

    Ok(())
}

/// Start interactive Q&A session
#[cfg(all(feature = "agent", feature = "openai"))]
pub async fn interactive_ask(config: &Config) -> Result<()> {
    use crate::agent::{IslandsAgent, llm::openai::OpenAiProvider};

    let api_key = config
        .openai_api_key
        .as_ref()
        .ok_or_else(|| crate::Error::Config("OPENAI_API_KEY not set".into()))?;

    let indexer = Arc::new(IndexerService::new(config.indexer.clone(), HashMap::new()));
    let llm = Arc::new(OpenAiProvider::new(api_key));

    let mut agent = IslandsAgent::new(indexer, llm, None);

    output::info("Islands Codebase Assistant");
    println!("Ask questions about your indexed codebases. Type 'quit' to exit.\n");

    let indexes = agent.list_indexes().await;
    if indexes.is_empty() {
        output::warning("No indexes available. Add repositories first.");
    } else {
        println!("Available indexes: {}\n", indexes.join(", "));
    }

    loop {
        print!("\n{} ", console::style("You:").cyan().bold());
        std::io::Write::flush(&mut std::io::stdout())?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "quit" || input == "exit" || input == "q" {
            break;
        }

        if input == "clear" {
            agent.clear_conversation();
            output::info("Conversation cleared.");
            continue;
        }

        println!("\n{}", console::style("Islands:").magenta().bold());

        match agent.ask(input, true).await {
            Ok(response) => {
                println!("{}", response);
            }
            Err(e) => {
                output::error(&format!("Error: {}", e));
            }
        }
    }

    println!("\nGoodbye!");
    Ok(())
}

/// Show system status
pub async fn show_status(config: &Config) -> Result<()> {
    let indexer = IndexerService::new(config.indexer.clone(), HashMap::new());

    println!("\n{}\n", console::style("Islands Status").bold());

    let indexes = indexer.list_indexes().await;
    let total_files: usize = indexes.iter().map(|i| i.file_count).sum();
    let total_size: u64 = indexes.iter().map(|i| i.size_bytes).sum();

    println!("Total Indexes: {}", indexes.len());
    println!("Total Files: {}", total_files);
    println!(
        "Total Size: {:.2} MB",
        total_size as f64 / (1024.0 * 1024.0)
    );
    println!("Repos Path: {:?}", config.indexer.repos_path);
    println!("Indexes Path: {:?}", config.indexer.indexes_path);

    Ok(())
}

/// Remove an index and its associated files
pub async fn remove_index(config: &Config, index_name: &str, force: bool) -> Result<()> {
    let indexer = IndexerService::new(config.indexer.clone(), HashMap::new());

    // Check if index exists
    let info = indexer.get_index(index_name).await;

    if info.is_none() {
        return Err(crate::Error::InvalidArgument(format!(
            "Index not found: {}",
            index_name
        )));
    }

    let info = info.unwrap();

    // Confirm if not forced
    if !force {
        output::warning(&format!(
            "This will delete index '{}' and all associated files.",
            index_name
        ));
        print!("Are you sure? (y/N): ");
        std::io::Write::flush(&mut std::io::stdout())?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim().to_lowercase();

        if input != "y" && input != "yes" {
            output::info("Aborted.");
            return Ok(());
        }
    }

    let spinner = output::spinner(&format!("Removing index: {}", index_name));

    // Delete the index
    indexer.delete_index(index_name).await?;

    spinner.finish_and_clear();

    output::success(&format!("Successfully removed index: {}", index_name));
    println!("  Repository: {}", info.repository.full_name);

    Ok(())
}

/// Show current configuration as YAML
pub async fn config_show(config: &Config) -> Result<()> {
    let yaml = serde_yaml::to_string(config).map_err(|e| crate::Error::Config(e.to_string()))?;
    println!("{}", yaml);
    Ok(())
}

/// Initialize a configuration file with defaults
pub async fn config_init(output_path: Option<std::path::PathBuf>) -> Result<()> {
    let config = Config::default();
    let yaml = serde_yaml::to_string(&config).map_err(|e| crate::Error::Config(e.to_string()))?;

    let path = output_path.unwrap_or_else(|| std::path::PathBuf::from("islands.yaml"));

    if path.exists() {
        return Err(crate::Error::InvalidArgument(format!(
            "File already exists: {}",
            path.display()
        )));
    }

    std::fs::write(&path, &yaml)?;
    output::success(&format!("Configuration file created: {}", path.display()));

    Ok(())
}

/// Create a new workspace with one or more repositories
pub async fn workspace_create(
    config: &Config,
    name: &str,
    repo_urls: &[String],
    token: Option<&str>,
) -> Result<()> {
    let spinner = output::spinner(&format!("Creating workspace: {}", name));

    let indexer = IndexerService::new(config.indexer.clone(), HashMap::new());

    let mut repos = Vec::new();
    for url in repo_urls {
        let (provider_type, owner, repo_name, base_url) = parse_repo_url(url)?;
        let provider_type_str = format!("{:?}", provider_type).to_lowercase();
        let provider = create_provider(&provider_type_str, base_url.as_deref(), token, None)?;
        let repo = provider.get_repository(&owner, &repo_name).await?;
        repos.push(repo);
    }

    let workspace = indexer.create_workspace(name, &repos).await?;
    spinner.finish_and_clear();

    output::success(&format!("Created workspace: {}", name));
    println!("  Repositories: {}", workspace.repositories.len());
    for repo in &workspace.repositories {
        println!("    - {}", repo.full_name);
    }

    Ok(())
}

/// List all workspaces
pub async fn workspace_list(config: &Config) -> Result<()> {
    let indexer = IndexerService::new(config.indexer.clone(), HashMap::new());
    let workspaces = indexer.list_workspaces().await;

    if workspaces.is_empty() {
        output::warning("No workspaces found. Use 'islands workspace create' to create one.");
        return Ok(());
    }

    println!("\nIslands Workspaces\n");

    for ws in workspaces {
        println!(
            "{} ({} repositories)",
            console::style(&ws.name).cyan().bold(),
            ws.repositories.len()
        );
        for repo in &ws.repositories {
            println!("    - {}", repo.full_name);
        }
        println!("  Created: {}", ws.created_at);
        println!("  Updated: {}", ws.updated_at);
        println!();
    }

    Ok(())
}

/// Delete a workspace
pub async fn workspace_delete(config: &Config, name: &str, force: bool) -> Result<()> {
    let indexer = IndexerService::new(config.indexer.clone(), HashMap::new());

    if !force {
        output::warning(&format!("This will delete workspace '{}'.", name));
        print!("Are you sure? (y/N): ");
        std::io::Write::flush(&mut std::io::stdout())?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim().to_lowercase();

        if input != "y" && input != "yes" {
            output::info("Aborted.");
            return Ok(());
        }
    }

    let spinner = output::spinner(&format!("Deleting workspace: {}", name));
    indexer.delete_workspace(name).await?;
    spinner.finish_and_clear();

    output::success(&format!("Deleted workspace: {}", name));

    Ok(())
}

/// Add a repository to an existing workspace
pub async fn workspace_add_repo(
    config: &Config,
    workspace_name: &str,
    repo_url: &str,
    token: Option<&str>,
) -> Result<()> {
    let spinner = output::spinner(&format!(
        "Adding repository to workspace: {}",
        workspace_name
    ));

    let (provider_type, owner, repo_name, base_url) = parse_repo_url(repo_url)?;
    let provider_type_str = format!("{:?}", provider_type).to_lowercase();
    let provider = create_provider(&provider_type_str, base_url.as_deref(), token, None)?;
    let repo = provider.get_repository(&owner, &repo_name).await?;

    let indexer = IndexerService::new(config.indexer.clone(), HashMap::new());
    indexer.add_repo_to_workspace(workspace_name, &repo).await?;
    spinner.finish_and_clear();

    output::success(&format!(
        "Added {} to workspace: {}",
        repo.full_name, workspace_name
    ));

    Ok(())
}

/// Remove a repository from a workspace
pub async fn workspace_remove_repo(
    config: &Config,
    workspace_name: &str,
    repo_id: &str,
) -> Result<()> {
    let spinner = output::spinner(&format!(
        "Removing repository from workspace: {}",
        workspace_name
    ));

    let indexer = IndexerService::new(config.indexer.clone(), HashMap::new());
    indexer
        .remove_repo_from_workspace(workspace_name, repo_id)
        .await?;
    spinner.finish_and_clear();

    output::success(&format!(
        "Removed {} from workspace: {}",
        repo_id, workspace_name
    ));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::IndexerConfig;
    use tempfile::tempdir;

    /// Helper to create a test config with temp directories
    fn test_config() -> (Config, tempfile::TempDir) {
        let dir = tempdir().expect("Failed to create temp dir");
        let config = Config {
            debug: false,
            log_level: "error".to_string(),
            indexer: IndexerConfig {
                repos_path: dir.path().join("repos"),
                indexes_path: dir.path().join("indexes"),
                max_concurrent_syncs: 1,
                sync_interval_secs: 60,
                index_extensions: vec!["rs".to_string()],
                #[cfg(feature = "embeddings")]
                embedding: crate::indexer::service::EmbeddingConfig::default(),
            },
            openai_api_key: None,
            mcp_host: "127.0.0.1".to_string(),
            mcp_port: 8080,
        };
        (config, dir)
    }

    // =========================================================================
    // search() tests
    // =========================================================================

    #[tokio::test]
    async fn test_search_with_no_indexes_returns_ok() {
        let (config, _dir) = test_config();

        // Searching with no indexes should return Ok (with warning printed)
        let result = search(&config, "test query", None, None, 10).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_search_with_empty_query_returns_ok() {
        let (config, _dir) = test_config();

        let result = search(&config, "", None, None, 10).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_search_with_specific_indexes_returns_ok() {
        let (config, _dir) = test_config();

        let indexes = vec!["nonexistent/repo".to_string()];
        let result = search(&config, "query", Some(indexes), None, 5).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_search_with_top_k_zero() {
        let (config, _dir) = test_config();

        let result = search(&config, "test", None, None, 0).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_search_with_large_top_k() {
        let (config, _dir) = test_config();

        let result = search(&config, "test", None, None, 10000).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_search_with_unicode_query() {
        let (config, _dir) = test_config();

        let result = search(&config, "unicode test", None, None, 10).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_search_with_special_chars_query() {
        let (config, _dir) = test_config();

        let result = search(
            &config,
            "fn main() { println!(\"hello\"); }",
            None,
            None,
            10,
        )
        .await;
        assert!(result.is_ok());
    }

    // =========================================================================
    // list_indexes() tests
    // =========================================================================

    #[tokio::test]
    async fn test_list_indexes_empty_returns_ok() {
        let (config, _dir) = test_config();

        let result = list_indexes(&config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_list_indexes_creates_directories() {
        let (config, dir) = test_config();

        let _ = list_indexes(&config).await;

        // Directories should be created by IndexerService::new
        assert!(dir.path().join("repos").exists());
        assert!(dir.path().join("indexes").exists());
    }

    // =========================================================================
    // show_status() tests
    // =========================================================================

    #[tokio::test]
    async fn test_show_status_empty_returns_ok() {
        let (config, _dir) = test_config();

        let result = show_status(&config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_show_status_creates_directories() {
        let (config, dir) = test_config();

        let _ = show_status(&config).await;

        assert!(dir.path().join("repos").exists());
        assert!(dir.path().join("indexes").exists());
    }

    #[tokio::test]
    async fn test_show_status_with_custom_paths() {
        let dir = tempdir().expect("Failed to create temp dir");
        let custom_repos = dir.path().join("custom_repos");
        let custom_indexes = dir.path().join("custom_indexes");

        let config = Config {
            debug: true,
            log_level: "debug".to_string(),
            indexer: IndexerConfig {
                repos_path: custom_repos.clone(),
                indexes_path: custom_indexes.clone(),
                ..Default::default()
            },
            openai_api_key: Some("test-key".to_string()),
            mcp_host: "localhost".to_string(),
            mcp_port: 9000,
        };

        let result = show_status(&config).await;
        assert!(result.is_ok());

        // Custom directories should be created
        assert!(custom_repos.exists());
        assert!(custom_indexes.exists());
    }

    // =========================================================================
    // sync_repository() tests
    // =========================================================================

    #[tokio::test]
    async fn test_sync_repository_nonexistent_index_returns_error() {
        let (config, _dir) = test_config();

        let result = sync_repository(&config, "nonexistent/owner/repo").await;

        // Should fail because the index doesn't exist
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, crate::Error::InvalidArgument(_)));
    }

    #[tokio::test]
    async fn test_sync_repository_with_empty_name() {
        let (config, _dir) = test_config();

        let result = sync_repository(&config, "").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_sync_repository_with_special_chars() {
        let (config, _dir) = test_config();

        let result = sync_repository(&config, "github/test/repo-with-dashes").await;
        assert!(result.is_err()); // Index doesn't exist
    }

    #[tokio::test]
    async fn test_sync_repository_error_contains_index_name() {
        let (config, _dir) = test_config();

        let result = sync_repository(&config, "test-index-name").await;
        let err = result.unwrap_err();
        let err_msg = err.to_string();

        assert!(err_msg.contains("test-index-name") || err_msg.contains("not found"));
    }

    // =========================================================================
    // add_repository() tests - URL parsing errors
    // =========================================================================

    #[tokio::test]
    async fn test_add_repository_invalid_url_returns_error() {
        let (config, _dir) = test_config();

        let result = add_repository(&config, "not-a-valid-url", None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_add_repository_url_without_repo_path() {
        let (config, _dir) = test_config();

        let result = add_repository(&config, "https://github.com", None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_add_repository_url_missing_scheme() {
        let (config, _dir) = test_config();

        let result = add_repository(&config, "github.com/owner/repo", None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_add_repository_empty_url() {
        let (config, _dir) = test_config();

        let result = add_repository(&config, "", None).await;
        assert!(result.is_err());
    }

    // =========================================================================
    // parse_repo_url delegation tests
    // =========================================================================

    #[test]
    fn test_parse_repo_url_github() {
        use crate::providers::factory::{ProviderType, parse_repo_url};

        let result = parse_repo_url("https://github.com/rust-lang/rust");
        assert!(result.is_ok());

        let (provider, owner, name, base_url) = result.unwrap();
        assert_eq!(provider, ProviderType::GitHub);
        assert_eq!(owner, "rust-lang");
        assert_eq!(name, "rust");
        assert!(base_url.is_none());
    }

    #[test]
    fn test_parse_repo_url_gitlab() {
        use crate::providers::factory::{ProviderType, parse_repo_url};

        let result = parse_repo_url("https://gitlab.com/group/project");
        assert!(result.is_ok());

        let (provider, owner, name, _) = result.unwrap();
        assert_eq!(provider, ProviderType::GitLab);
        assert_eq!(owner, "group");
        assert_eq!(name, "project");
    }

    #[test]
    fn test_parse_repo_url_bitbucket() {
        use crate::providers::factory::{ProviderType, parse_repo_url};

        let result = parse_repo_url("https://bitbucket.org/team/repo");
        assert!(result.is_ok());

        let (provider, owner, name, _) = result.unwrap();
        assert_eq!(provider, ProviderType::Bitbucket);
        assert_eq!(owner, "team");
        assert_eq!(name, "repo");
    }

    #[test]
    fn test_parse_repo_url_self_hosted_gitea() {
        use crate::providers::factory::{ProviderType, parse_repo_url};

        let result = parse_repo_url("https://git.example.com/user/project");
        assert!(result.is_ok());

        let (provider, owner, name, base_url) = result.unwrap();
        assert_eq!(provider, ProviderType::Gitea);
        assert_eq!(owner, "user");
        assert_eq!(name, "project");
        assert_eq!(base_url, Some("https://git.example.com".to_string()));
    }

    #[test]
    fn test_parse_repo_url_with_git_suffix() {
        use crate::providers::factory::parse_repo_url;

        let result = parse_repo_url("https://github.com/owner/repo.git");
        assert!(result.is_ok());

        let (_, owner, name, _) = result.unwrap();
        assert_eq!(owner, "owner");
        assert_eq!(name, "repo");
    }

    #[test]
    fn test_parse_repo_url_invalid() {
        use crate::providers::factory::parse_repo_url;

        let result = parse_repo_url("invalid-url");
        assert!(result.is_err());
    }

    // =========================================================================
    // create_provider delegation tests
    // =========================================================================

    #[test]
    fn test_create_provider_github() {
        use crate::providers::factory::create_provider;

        let provider = create_provider("github", None, Some("test-token"), None);
        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().provider_name(), "github");
    }

    #[test]
    fn test_create_provider_gitlab() {
        use crate::providers::factory::create_provider;

        let provider = create_provider("gitlab", None, Some("test-token"), None);
        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().provider_name(), "gitlab");
    }

    #[test]
    fn test_create_provider_bitbucket() {
        use crate::providers::factory::create_provider;

        let provider = create_provider("bitbucket", None, Some("test-token"), None);
        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().provider_name(), "bitbucket");
    }

    #[test]
    fn test_create_provider_unsupported() {
        use crate::providers::factory::create_provider;

        let result = create_provider("unsupported", None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_create_provider_gitea_requires_base_url() {
        use crate::providers::factory::create_provider;

        let result = create_provider("gitea", None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_create_provider_gitea_with_base_url() {
        use crate::providers::factory::create_provider;

        let result = create_provider(
            "gitea",
            Some("https://gitea.example.com"),
            Some("token"),
            None,
        );
        assert!(result.is_ok());
    }

    // =========================================================================
    // Config interaction tests
    // =========================================================================

    #[test]
    fn test_config_debug_mode() {
        let (config, _dir) = test_config();
        assert!(!config.debug);
    }

    #[test]
    fn test_config_log_level() {
        let (config, _dir) = test_config();
        assert_eq!(config.log_level, "error");
    }

    #[test]
    fn test_config_indexer_paths() {
        let (config, dir) = test_config();
        assert_eq!(config.indexer.repos_path, dir.path().join("repos"));
        assert_eq!(config.indexer.indexes_path, dir.path().join("indexes"));
    }

    // =========================================================================
    // Edge cases
    // =========================================================================

    #[tokio::test]
    async fn test_multiple_operations_same_config() {
        let (config, _dir) = test_config();

        // Multiple operations should work with same config
        let _ = show_status(&config).await;
        let _ = list_indexes(&config).await;
        let _ = search(&config, "test", None, None, 10).await;

        // All should complete without panicking
    }

    #[tokio::test]
    async fn test_operations_with_default_config() {
        // Create config with default indexer (will use ~/.local/share/islands paths)
        // These may fail due to permissions, but shouldn't panic
        let config = Config::default();

        // These operations should not panic, even if they fail
        let _ = show_status(&config).await;
        let _ = list_indexes(&config).await;
    }

    #[tokio::test]
    async fn test_search_empty_indexes_list() {
        let (config, _dir) = test_config();

        let empty_indexes: Vec<String> = vec![];
        let result = search(&config, "query", Some(empty_indexes), None, 10).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_sync_repository_various_invalid_names() {
        let (config, _dir) = test_config();

        // Various invalid index names
        let invalid_names = vec![
            "a",              // Too short
            "../escape",      // Path traversal attempt
            "with spaces",    // Spaces
            "owner//repo",    // Double slash
            "/absolute/path", // Absolute path
        ];

        for name in invalid_names {
            let result = sync_repository(&config, name).await;
            assert!(result.is_err(), "Expected error for name: {}", name);
        }
    }

    // =========================================================================
    // MCP server tests (feature-gated)
    // =========================================================================

    #[cfg(feature = "mcp")]
    mod mcp_tests {
        use super::*;

        // Note: serve_mcp is difficult to test directly because it:
        // 1. Runs indefinitely
        // 2. Uses stdin/stdout for MCP protocol
        // We can only test that the function exists and compiles correctly.

        #[test]
        fn test_serve_mcp_exists() {
            // This test verifies the function exists and has the right signature
            async fn _check_signature(_config: &Config) -> Result<()> {
                serve_mcp(&Config::default()).await
            }
            // Type check that function has correct signature
            let _ = _check_signature;
        }
    }

    // =========================================================================
    // Interactive ask tests (feature-gated)
    // =========================================================================

    #[cfg(all(feature = "agent", feature = "openai"))]
    mod agent_tests {
        use super::*;

        #[tokio::test]
        async fn test_interactive_ask_requires_api_key() {
            let (mut config, _dir) = test_config();
            config.openai_api_key = None;

            // Should fail without API key
            let result = interactive_ask(&config).await;
            assert!(result.is_err());
        }
    }

    // =========================================================================
    // Concurrent operation tests
    // =========================================================================

    #[tokio::test]
    async fn test_concurrent_searches() {
        let (config, _dir) = test_config();

        let config1 = config.clone();
        let config2 = config.clone();
        let config3 = config.clone();

        let (r1, r2, r3) = tokio::join!(
            search(&config1, "query1", None, None, 5),
            search(&config2, "query2", None, None, 5),
            search(&config3, "query3", None, None, 5),
        );

        assert!(r1.is_ok());
        assert!(r2.is_ok());
        assert!(r3.is_ok());
    }

    #[tokio::test]
    async fn test_concurrent_status_and_list() {
        let (config, _dir) = test_config();

        let config1 = config.clone();
        let config2 = config.clone();

        let (r1, r2) = tokio::join!(show_status(&config1), list_indexes(&config2),);

        assert!(r1.is_ok());
        assert!(r2.is_ok());
    }

    // =========================================================================
    // Full workflow tests
    // =========================================================================

    #[tokio::test]
    async fn test_full_workflow_empty_state() {
        let (config, _dir) = test_config();

        // Show status (empty)
        assert!(show_status(&config).await.is_ok());

        // List indexes (empty)
        assert!(list_indexes(&config).await.is_ok());

        // Search (empty)
        assert!(search(&config, "test", None, None, 10).await.is_ok());

        // Sync non-existent (error)
        assert!(sync_repository(&config, "nonexistent").await.is_err());
    }

    // =========================================================================
    // remove_index() tests
    // =========================================================================

    #[tokio::test]
    async fn test_remove_index_nonexistent_returns_error() {
        let (config, _dir) = test_config();

        let result = remove_index(&config, "nonexistent/index", true).await;
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, crate::Error::InvalidArgument(_)));
    }

    #[tokio::test]
    async fn test_remove_index_with_empty_name() {
        let (config, _dir) = test_config();

        let result = remove_index(&config, "", true).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_remove_index_with_force_flag() {
        let (config, _dir) = test_config();

        // With force=true, should skip confirmation (but still fail for nonexistent)
        let result = remove_index(&config, "github/owner/repo", true).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_remove_index_error_contains_index_name() {
        let (config, _dir) = test_config();

        let result = remove_index(&config, "my-missing-index", true).await;
        let err = result.unwrap_err();
        let err_msg = err.to_string();

        assert!(err_msg.contains("my-missing-index") || err_msg.contains("not found"));
    }

    // =========================================================================
    // config_show() tests
    // =========================================================================

    #[tokio::test]
    async fn test_config_show_returns_ok() {
        let (config, _dir) = test_config();

        let result = config_show(&config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_config_show_with_default_config() {
        let config = Config::default();

        let result = config_show(&config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_config_show_with_custom_config() {
        let dir = tempdir().expect("Failed to create temp dir");
        let config = Config {
            debug: true,
            log_level: "trace".to_string(),
            indexer: IndexerConfig {
                repos_path: dir.path().join("custom_repos"),
                indexes_path: dir.path().join("custom_indexes"),
                max_concurrent_syncs: 10,
                sync_interval_secs: 120,
                index_extensions: vec!["py".to_string(), "rs".to_string()],
                #[cfg(feature = "embeddings")]
                embedding: crate::indexer::service::EmbeddingConfig::default(),
            },
            openai_api_key: Some("sk-test".to_string()),
            mcp_host: "localhost".to_string(),
            mcp_port: 9999,
        };

        let result = config_show(&config).await;
        assert!(result.is_ok());
    }

    // =========================================================================
    // config_init() tests
    // =========================================================================

    #[tokio::test]
    async fn test_config_init_creates_file() {
        let dir = tempdir().expect("Failed to create temp dir");
        let output_path = dir.path().join("islands.yaml");

        let result = config_init(Some(output_path.clone())).await;
        assert!(result.is_ok());
        assert!(output_path.exists());

        // Verify the file contains valid YAML
        let content = std::fs::read_to_string(&output_path).unwrap();
        let parsed: std::result::Result<Config, _> = serde_yaml::from_str(&content);
        assert!(parsed.is_ok());
    }

    #[tokio::test]
    async fn test_config_init_fails_if_file_exists() {
        let dir = tempdir().expect("Failed to create temp dir");
        let output_path = dir.path().join("existing.yaml");

        // Create the file first
        std::fs::write(&output_path, "existing content").unwrap();

        let result = config_init(Some(output_path)).await;
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, crate::Error::InvalidArgument(_)));
    }

    #[tokio::test]
    async fn test_config_init_creates_valid_default_config() {
        let dir = tempdir().expect("Failed to create temp dir");
        let output_path = dir.path().join("default-config.yaml");

        let _ = config_init(Some(output_path.clone())).await;

        let content = std::fs::read_to_string(&output_path).unwrap();
        let parsed: Config = serde_yaml::from_str(&content).unwrap();

        // Verify default values
        assert!(!parsed.debug);
        assert_eq!(parsed.log_level, "info");
        assert_eq!(parsed.mcp_host, "0.0.0.0");
        assert_eq!(parsed.mcp_port, 8080);
    }

    #[tokio::test]
    async fn test_config_init_in_nested_directory() {
        let dir = tempdir().expect("Failed to create temp dir");
        let nested = dir.path().join("nested").join("deep");
        std::fs::create_dir_all(&nested).unwrap();
        let output_path = nested.join("config.yaml");

        let result = config_init(Some(output_path.clone())).await;

        assert!(result.is_ok());
        assert!(output_path.exists());
    }
}
