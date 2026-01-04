"""Base class and interfaces for git providers."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import AsyncIterator

import httpx


class AuthType(str, Enum):
    """Authentication type for git providers."""

    TOKEN = "token"
    SSH = "ssh"
    OAUTH = "oauth"
    BASIC = "basic"


@dataclass
class ProviderAuth:
    """Authentication credentials for a git provider."""

    auth_type: AuthType
    token: str | None = None
    username: str | None = None
    password: str | None = None
    ssh_key_path: Path | None = None
    oauth_client_id: str | None = None
    oauth_client_secret: str | None = None


@dataclass
class Repository:
    """Represents a git repository."""

    provider: str
    owner: str
    name: str
    full_name: str
    clone_url: str
    ssh_url: str | None = None
    default_branch: str = "main"
    description: str | None = None
    language: str | None = None
    size_kb: int = 0
    last_updated: datetime | None = None
    is_private: bool = False
    topics: list[str] = field(default_factory=list)

    @property
    def local_path(self) -> Path:
        """Get the local path where this repo would be cloned."""
        return Path(f"{self.provider}/{self.owner}/{self.name}")


@dataclass
class WebhookEvent:
    """Represents a webhook event from a git provider."""

    event_type: str
    repository: Repository
    ref: str | None = None
    before: str | None = None
    after: str | None = None
    payload: dict = field(default_factory=dict)


class GitProvider(ABC):
    """Abstract base class for git providers."""

    provider_name: str = "base"

    def __init__(
        self,
        base_url: str,
        auth: ProviderAuth | None = None,
        rate_limit_requests: int = 5000,
        rate_limit_window: int = 3600,
    ):
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window = rate_limit_window
        self._client: httpx.AsyncClient | None = None
        self._request_count = 0
        self._window_start = datetime.now()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = self._build_auth_headers()
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=30.0,
                follow_redirects=True,
            )
        return self._client

    @abstractmethod
    def _build_auth_headers(self) -> dict[str, str]:
        """Build authentication headers for API requests."""
        pass

    async def _check_rate_limit(self) -> None:
        """Check and respect rate limits."""
        now = datetime.now()
        elapsed = (now - self._window_start).total_seconds()

        if elapsed >= self.rate_limit_window:
            self._request_count = 0
            self._window_start = now

        if self._request_count >= self.rate_limit_requests:
            wait_time = self.rate_limit_window - elapsed
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                self._request_count = 0
                self._window_start = datetime.now()

        self._request_count += 1

    async def _request(
        self, method: str, path: str, **kwargs
    ) -> httpx.Response:
        """Make an API request with rate limiting."""
        await self._check_rate_limit()
        client = await self._get_client()
        response = await client.request(method, path, **kwargs)
        response.raise_for_status()
        return response

    @abstractmethod
    async def list_repositories(
        self,
        owner: str | None = None,
        visibility: str | None = None,
        per_page: int = 100,
    ) -> AsyncIterator[Repository]:
        """List repositories for the authenticated user or organization."""
        pass

    @abstractmethod
    async def get_repository(self, owner: str, name: str) -> Repository:
        """Get a specific repository by owner and name."""
        pass

    @abstractmethod
    async def get_default_branch(self, owner: str, name: str) -> str:
        """Get the default branch for a repository."""
        pass

    @abstractmethod
    async def get_latest_commit(self, owner: str, name: str, ref: str | None = None) -> str:
        """Get the latest commit SHA for a branch or ref."""
        pass

    @abstractmethod
    async def parse_webhook(self, headers: dict, body: bytes) -> WebhookEvent | None:
        """Parse a webhook payload and return an event."""
        pass

    def get_clone_url(self, repo: Repository) -> str:
        """Get the appropriate clone URL based on auth type."""
        if self.auth and self.auth.auth_type == AuthType.SSH and repo.ssh_url:
            return repo.ssh_url
        if self.auth and self.auth.token:
            if "github" in repo.clone_url:
                return repo.clone_url.replace(
                    "https://", f"https://{self.auth.token}@"
                )
            if "gitlab" in repo.clone_url:
                return repo.clone_url.replace(
                    "https://", f"https://oauth2:{self.auth.token}@"
                )
        return repo.clone_url

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> GitProvider:
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
