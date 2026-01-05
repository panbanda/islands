//! Local repository indexing example
//!
//! Index local directories without needing to clone from a remote.

use std::path::PathBuf;

use pythia_indexer::{IndexerConfig, IndexerService};
use pythia_providers::Repository;

#[cfg(feature = "embeddings")]
use pythia_indexer::EmbeddingConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    // Get path from args or use current directory
    let project_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| ".".to_string());

    let project_path = PathBuf::from(&project_path).canonicalize()?;
    let project_name = project_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("project");

    println!("Indexing local project: {}", project_path.display());

    let config = IndexerConfig {
        repos_path: PathBuf::from("./data/repos"),
        indexes_path: PathBuf::from("./data/indexes"),
        index_extensions: vec![
            "rs", "py", "js", "ts", "jsx", "tsx", "go", "java",
            "c", "cpp", "h", "hpp", "md", "txt", "toml", "yaml",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
        #[cfg(feature = "embeddings")]
        embedding: EmbeddingConfig::Local {
            model: "bge-small".to_string(),
            batch_size: 32,
        },
        ..Default::default()
    };

    let mut service = IndexerService::new(config);

    #[cfg(feature = "embeddings")]
    {
        println!("Loading embedding model...");
        service.init_embedder().await?;
    }

    // Create a "local" repository pointing to the directory
    let local_repo = Repository {
        provider: "local".to_string(),
        owner: "local".to_string(),
        name: project_name.to_string(),
        clone_url: Some(project_path.to_string_lossy().to_string()),
        default_branch: Some("main".to_string()),
        ..Default::default()
    };

    println!("Building index...");
    let info = service.add_repository(local_repo).await?;

    println!(
        "Indexed {} files ({:.2} KB)",
        info.file_count,
        info.size_bytes as f64 / 1024.0
    );

    // Interactive search loop
    println!("\nEnter search queries (Ctrl+C to exit):\n");

    let stdin = std::io::stdin();
    let mut query = String::new();

    loop {
        print!("> ");
        std::io::Write::flush(&mut std::io::stdout())?;

        query.clear();
        if stdin.read_line(&mut query)? == 0 {
            break;
        }

        let query = query.trim();
        if query.is_empty() {
            continue;
        }

        let results = service.search(query, 5, None).await?;

        if results.is_empty() {
            println!("No results found.\n");
            continue;
        }

        for (i, result) in results.iter().enumerate() {
            println!(
                "\n{}. {} (score: {:.4})",
                i + 1,
                result.file_path,
                result.score
            );

            // Show snippet
            let lines: Vec<&str> = result.content.lines().take(5).collect();
            for line in lines {
                println!("   {}", line);
            }
        }
        println!();
    }

    Ok(())
}
