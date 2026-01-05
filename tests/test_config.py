"""Tests for configuration module."""

from __future__ import annotations

import json
from pathlib import Path

from pythia.config.settings import (
    AgentConfig,
    LeannBackend,
    LeannConfig,
    ProviderConfig,
    ProviderType,
    Settings,
    StorageConfig,
)


class TestStorageConfig:
    """Tests for StorageConfig."""

    def test_default_paths(self):
        config = StorageConfig()
        assert config.base_path == Path("/data/pythia")
        assert config.repos_path == Path("/data/pythia/repos")
        assert config.indexes_path == Path("/data/pythia/indexes")
        assert config.cache_path == Path("/data/pythia/cache")

    def test_custom_paths(self, temp_dir):
        config = StorageConfig(
            base_path=temp_dir,
            repos_path=temp_dir / "repos",
            indexes_path=temp_dir / "indexes",
            cache_path=temp_dir / "cache",
        )
        assert config.base_path == temp_dir

    def test_ensure_directories(self, temp_dir):
        config = StorageConfig(
            base_path=temp_dir / "pythia",
            repos_path=temp_dir / "pythia" / "repos",
            indexes_path=temp_dir / "pythia" / "indexes",
            cache_path=temp_dir / "pythia" / "cache",
        )
        config.ensure_directories()

        assert config.base_path.exists()
        assert config.repos_path.exists()
        assert config.indexes_path.exists()
        assert config.cache_path.exists()


class TestProviderConfig:
    """Tests for ProviderConfig."""

    def test_github_default_url(self):
        config = ProviderConfig(type=ProviderType.GITHUB)
        assert config.type == ProviderType.GITHUB

    def test_custom_base_url(self):
        config = ProviderConfig(
            type=ProviderType.GITHUB,
            base_url="https://github.enterprise.com/api/v3",
        )
        assert config.base_url == "https://github.enterprise.com/api/v3"

    def test_rate_limits(self):
        config = ProviderConfig(
            type=ProviderType.GITHUB,
            rate_limit_requests=1000,
            rate_limit_window=600,
        )
        assert config.rate_limit_requests == 1000
        assert config.rate_limit_window == 600


class TestLeannConfig:
    """Tests for LeannConfig."""

    def test_defaults(self):
        config = LeannConfig()
        assert config.backend == LeannBackend.HNSW
        assert config.no_recompute is True
        assert config.chunk_size == 512
        assert config.chunk_overlap == 64

    def test_diskann_backend(self):
        config = LeannConfig(backend=LeannBackend.DISKANN)
        assert config.backend == LeannBackend.DISKANN


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_defaults(self):
        config = AgentConfig()
        assert config.model == "gpt-4o"
        assert config.temperature == 0.1
        assert config.max_tokens == 4096

    def test_custom_model(self):
        config = AgentConfig(model="gpt-4-turbo", temperature=0.5)
        assert config.model == "gpt-4-turbo"
        assert config.temperature == 0.5


class TestSettings:
    """Tests for main Settings class."""

    def test_default_settings(self):
        settings = Settings()
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.sync_interval == 300
        assert settings.max_concurrent_syncs == 4

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("PYTHIA_DEBUG", "true")
        monkeypatch.setenv("PYTHIA_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("PYTHIA_SYNC_INTERVAL", "600")

        settings = Settings.from_env()
        assert settings.debug is True
        assert settings.log_level == "DEBUG"
        assert settings.sync_interval == 600

    def test_from_file_json(self, temp_dir):
        config_file = temp_dir / "config.json"
        config_data = {
            "debug": True,
            "log_level": "DEBUG",
            "sync_interval": 120,
            "providers": [{"type": "github", "token": "test-token"}],
        }
        config_file.write_text(json.dumps(config_data))

        settings = Settings.from_file(config_file)
        assert settings.debug is True
        assert settings.log_level == "DEBUG"
        assert settings.sync_interval == 120
        assert len(settings.providers) == 1

    def test_multiple_providers(self):
        settings = Settings(
            providers=[
                ProviderConfig(type=ProviderType.GITHUB, token="github-token"),
                ProviderConfig(type=ProviderType.GITLAB, token="gitlab-token"),
            ]
        )
        assert len(settings.providers) == 2
        assert settings.providers[0].type == ProviderType.GITHUB
        assert settings.providers[1].type == ProviderType.GITLAB
