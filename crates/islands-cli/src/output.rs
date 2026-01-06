//! Output formatting utilities

use console::{Term, style};
use indicatif::{ProgressBar, ProgressStyle};
use tabled::{Table, Tabled};

/// Create a progress bar for operations
pub fn progress_bar(len: u64, message: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .expect("valid template")
            .progress_chars("#>-"),
    );
    pb.set_message(message.to_string());
    pb
}

/// Create a spinner for indeterminate operations
pub fn spinner(message: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("valid template"),
    );
    pb.set_message(message.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    pb
}

/// Print a success message
pub fn success(message: &str) {
    println!("{} {}", style("[OK]").green().bold(), message);
}

/// Print an error message
pub fn error(message: &str) {
    eprintln!("{} {}", style("[ERROR]").red().bold(), message);
}

/// Print a warning message
pub fn warning(message: &str) {
    println!("{} {}", style("[WARN]").yellow().bold(), message);
}

/// Print an info message
pub fn info(message: &str) {
    println!("{} {}", style("[INFO]").blue().bold(), message);
}

/// Print a table
pub fn table<T: Tabled>(items: &[T], title: Option<&str>) {
    if let Some(t) = title {
        println!("\n{}\n", style(t).bold());
    }

    if items.is_empty() {
        println!("{}", style("  (no items)").dim());
    } else {
        let table = Table::new(items);
        println!("{}", table);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_bar_creation() {
        let pb = progress_bar(100, "Processing files");
        assert_eq!(pb.length(), Some(100));
        assert_eq!(pb.position(), 0);
    }

    #[test]
    fn test_progress_bar_increment() {
        let pb = progress_bar(10, "Testing");
        pb.inc(1);
        assert_eq!(pb.position(), 1);
        pb.inc(5);
        assert_eq!(pb.position(), 6);
    }

    #[test]
    fn test_progress_bar_set_position() {
        let pb = progress_bar(100, "Testing");
        pb.set_position(50);
        assert_eq!(pb.position(), 50);
    }

    #[test]
    fn test_progress_bar_finish() {
        let pb = progress_bar(100, "Testing");
        pb.set_position(100);
        pb.finish();
        assert!(pb.is_finished());
    }

    #[test]
    fn test_progress_bar_finish_and_clear() {
        let pb = progress_bar(100, "Testing");
        pb.finish_and_clear();
        assert!(pb.is_finished());
    }

    #[test]
    fn test_progress_bar_message_update() {
        let pb = progress_bar(100, "Initial");
        pb.set_message("Updated message");
        // Message updated without error
    }

    #[test]
    fn test_progress_bar_zero_length() {
        let pb = progress_bar(0, "Empty");
        assert_eq!(pb.length(), Some(0));
    }

    #[test]
    fn test_progress_bar_large_length() {
        let pb = progress_bar(1_000_000, "Large");
        assert_eq!(pb.length(), Some(1_000_000));
    }

    #[test]
    fn test_spinner_creation() {
        let sp = spinner("Loading...");
        assert!(!sp.is_finished());
    }

    #[test]
    fn test_spinner_message_update() {
        let sp = spinner("Initial");
        sp.set_message("New message");
        // Message updated without error
    }

    #[test]
    fn test_spinner_finish() {
        let sp = spinner("Processing");
        sp.finish();
        assert!(sp.is_finished());
    }

    #[test]
    fn test_spinner_finish_with_message() {
        let sp = spinner("Processing");
        sp.finish_with_message("Done!");
        assert!(sp.is_finished());
    }

    #[test]
    fn test_spinner_finish_and_clear() {
        let sp = spinner("Loading");
        sp.finish_and_clear();
        assert!(sp.is_finished());
    }

    #[test]
    fn test_success_does_not_panic() {
        // Output functions that print to stdout/stderr
        // We verify they don't panic with various inputs
        success("Operation completed");
        success("");
        success("with special chars: <>&\"'");
    }

    #[test]
    fn test_error_does_not_panic() {
        error("Something went wrong");
        error("");
        error("with special chars: <>&\"'");
    }

    #[test]
    fn test_warning_does_not_panic() {
        warning("Potential issue detected");
        warning("");
        warning("with unicode: \u{2022} \u{2013} \u{2014}");
    }

    #[test]
    fn test_info_does_not_panic() {
        info("System information");
        info("");
        info("with numbers: 123456789");
    }

    #[derive(Tabled)]
    struct TestRow {
        name: String,
        value: i32,
    }

    #[test]
    fn test_table_with_items() {
        let items = vec![
            TestRow {
                name: "foo".to_string(),
                value: 1,
            },
            TestRow {
                name: "bar".to_string(),
                value: 2,
            },
        ];
        table(&items, Some("Test Table"));
        // Verify it doesn't panic
    }

    #[test]
    fn test_table_empty() {
        let items: Vec<TestRow> = vec![];
        table(&items, Some("Empty Table"));
        // Should print "(no items)"
    }

    #[test]
    fn test_table_without_title() {
        let items = vec![TestRow {
            name: "test".to_string(),
            value: 42,
        }];
        table(&items, None);
        // Should work without title
    }

    #[test]
    fn test_table_empty_without_title() {
        let items: Vec<TestRow> = vec![];
        table(&items, None);
    }

    #[test]
    fn test_table_single_row() {
        let items = vec![TestRow {
            name: "single".to_string(),
            value: 100,
        }];
        table(&items, Some("Single Row"));
    }

    #[test]
    fn test_table_many_rows() {
        let items: Vec<TestRow> = (0..100)
            .map(|i| TestRow {
                name: format!("item_{}", i),
                value: i,
            })
            .collect();
        table(&items, Some("Many Rows"));
    }

    #[derive(Tabled)]
    struct ComplexRow {
        id: u64,
        name: String,
        description: String,
        active: bool,
    }

    #[test]
    fn test_table_complex_struct() {
        let items = vec![
            ComplexRow {
                id: 1,
                name: "First".to_string(),
                description: "Description one".to_string(),
                active: true,
            },
            ComplexRow {
                id: 2,
                name: "Second".to_string(),
                description: "Description two".to_string(),
                active: false,
            },
        ];
        table(&items, Some("Complex Table"));
    }
}
