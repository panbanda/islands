# Contributing to Islands

Thank you for your interest in contributing to Islands!

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/islands.git
   cd islands
   ```

2. Install Rust (1.85+):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

3. Build and test:
   ```bash
   cargo build
   cargo test
   ```

## Code Style

- Run `cargo fmt` before committing
- Run `cargo clippy --all-targets -- -D warnings` and fix any warnings
- Keep functions focused and under 50 lines where practical
- Add doc comments for public APIs

## Testing

- Write tests for new functionality
- Run the full test suite before submitting:
  ```bash
  cargo test
  cargo test --doc
  ```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with appropriate tests
3. Ensure `cargo fmt`, `cargo clippy`, and `cargo test` pass
4. Submit a pull request with a clear description

## Commit Messages

Follow conventional commit format:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions or modifications
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

## Questions?

Open an issue for questions or discussions about contributions.
