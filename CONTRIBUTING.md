# Contributing to Pythia

Thank you for your interest in contributing to Pythia! This document provides guidelines and information about contributing.

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/pythia.git
   cd pythia
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

- We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Type hints are required for all public functions
- Docstrings follow Google style

Run code quality checks:
```bash
ruff check src tests
ruff format src tests
mypy src
```

## Testing

- Write tests for all new functionality
- Maintain test coverage above 80%
- Use pytest fixtures for common setup

Run tests:
```bash
pytest
pytest --cov=pythia --cov-report=html
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with appropriate tests
3. Ensure all tests pass and code quality checks succeed
4. Update documentation as needed
5. Submit a pull request with a clear description

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
