"""Configuration settings for Pythia using Pydantic Settings."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderType(str, Enum):
    """Supported git provider types."""

    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"
    GITEA = "gitea"


class LeannBackend(str, Enum):
    """LEANN backend types."""

    HNSW = "hnsw"
    DISKANN = "diskann"


class ProviderConfig(BaseModel):
    """Configuration for a git provider."""

    type: ProviderType
    base_url: str | None = None
    token: str | None = None
    ssh_key_path: Path | None = None
    webhook_secret: str | None = None
    rate_limit_requests: int = 5000
    rate_limit_window: int = 3600

    @field_validator("base_url", mode="before")
    @classmethod
    def set_default_base_url(cls, v: str | None, info: Any) -> str | None:
        if v is not None:
            return v
        provider_type = info.data.get("type")
        defaults = {
            ProviderType.GITHUB: "https://api.github.com",
            ProviderType.GITLAB: "https://gitlab.com/api/v4",
            ProviderType.BITBUCKET: "https://api.bitbucket.org/2.0",
        }
        return defaults.get(provider_type)


class StorageConfig(BaseModel):
    """Configuration for storage locations."""

    base_path: Path = Field(default_factory=lambda: Path("/data/pythia"))
    repos_path: Path = Field(default_factory=lambda: Path("/data/pythia/repos"))
    indexes_path: Path = Field(default_factory=lambda: Path("/data/pythia/indexes"))
    cache_path: Path = Field(default_factory=lambda: Path("/data/pythia/cache"))

    def ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        for path in [self.base_path, self.repos_path, self.indexes_path, self.cache_path]:
            path.mkdir(parents=True, exist_ok=True)


class LeannConfig(BaseModel):
    """Configuration for LEANN indexing."""

    backend: LeannBackend = LeannBackend.HNSW
    no_recompute: bool = True
    embedding_model: str = "default"
    chunk_size: int = 512
    chunk_overlap: int = 64


class AgentConfig(BaseModel):
    """Configuration for the OpenAI-based agent."""

    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 4096
    system_prompt: str | None = None


class Settings(BaseSettings):
    """Main settings for Pythia."""

    model_config = SettingsConfigDict(
        env_prefix="PYTHIA_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    debug: bool = False
    log_level: str = "INFO"

    storage: StorageConfig = Field(default_factory=StorageConfig)
    leann: LeannConfig = Field(default_factory=LeannConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)

    providers: list[ProviderConfig] = Field(default_factory=list)

    openai_api_key: str | None = None
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8080

    sync_interval: int = 300
    max_concurrent_syncs: int = 4
    webhook_enabled: bool = True
    webhook_port: int = 9000

    @classmethod
    def from_env(cls) -> Settings:
        """Load settings from environment variables."""
        return cls()

    @classmethod
    def from_file(cls, path: Path) -> Settings:
        """Load settings from a YAML or JSON file."""
        import json

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml

                with open(path) as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required for YAML config files")
        else:
            with open(path) as f:
                data = json.load(f)

        return cls(**data)
