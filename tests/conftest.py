"""Pytest configuration and fixtures for Pythia tests."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest

from pythia.config.settings import Settings, StorageConfig, ProviderConfig, ProviderType
from pythia.providers.base import AuthType, ProviderAuth, Repository


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def storage_config(temp_dir: Path) -> StorageConfig:
    """Create a storage config using temp directory."""
    return StorageConfig(
        base_path=temp_dir,
        repos_path=temp_dir / "repos",
        indexes_path=temp_dir / "indexes",
        cache_path=temp_dir / "cache",
    )


@pytest.fixture
def settings(storage_config: StorageConfig) -> Settings:
    """Create test settings."""
    return Settings(
        debug=True,
        log_level="DEBUG",
        storage=storage_config,
        providers=[],
        sync_interval=60,
        max_concurrent_syncs=2,
    )


@pytest.fixture
def github_provider_config() -> ProviderConfig:
    """Create a GitHub provider config for testing."""
    return ProviderConfig(
        type=ProviderType.GITHUB,
        base_url="https://api.github.com",
        token=None,
    )


@pytest.fixture
def mock_repository() -> Repository:
    """Create a mock repository for testing."""
    return Repository(
        provider="github",
        owner="test-owner",
        name="test-repo",
        full_name="test-owner/test-repo",
        clone_url="https://github.com/test-owner/test-repo.git",
        ssh_url="git@github.com:test-owner/test-repo.git",
        default_branch="main",
        description="A test repository",
        language="Python",
        size_kb=1024,
        is_private=False,
        topics=["test", "example"],
    )


@pytest.fixture
def provider_auth() -> ProviderAuth:
    """Create a test provider auth."""
    return ProviderAuth(
        auth_type=AuthType.TOKEN,
        token="test-token-12345",
    )
