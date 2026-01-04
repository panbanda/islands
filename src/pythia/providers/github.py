"""GitHub provider implementation."""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime
from typing import AsyncIterator

from pythia.providers.base import (
    AuthType,
    GitProvider,
    ProviderAuth,
    Repository,
    WebhookEvent,
)


class GitHubProvider(GitProvider):
    """GitHub API provider implementation."""

    provider_name = "github"

    def __init__(
        self,
        base_url: str = "https://api.github.com",
        auth: ProviderAuth | None = None,
        webhook_secret: str | None = None,
        **kwargs,
    ):
        super().__init__(base_url, auth, **kwargs)
        self.webhook_secret = webhook_secret

    def _build_auth_headers(self) -> dict[str, str]:
        """Build GitHub authentication headers."""
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.auth:
            if self.auth.auth_type == AuthType.TOKEN and self.auth.token:
                headers["Authorization"] = f"Bearer {self.auth.token}"
            elif self.auth.auth_type == AuthType.BASIC:
                import base64

                credentials = f"{self.auth.username}:{self.auth.password}"
                encoded = base64.b64encode(credentials.encode()).decode()
                headers["Authorization"] = f"Basic {encoded}"
        return headers

    def _parse_repository(self, data: dict) -> Repository:
        """Parse GitHub API response into Repository object."""
        return Repository(
            provider=self.provider_name,
            owner=data["owner"]["login"],
            name=data["name"],
            full_name=data["full_name"],
            clone_url=data["clone_url"],
            ssh_url=data.get("ssh_url"),
            default_branch=data.get("default_branch", "main"),
            description=data.get("description"),
            language=data.get("language"),
            size_kb=data.get("size", 0),
            last_updated=datetime.fromisoformat(
                data["updated_at"].replace("Z", "+00:00")
            ) if data.get("updated_at") else None,
            is_private=data.get("private", False),
            topics=data.get("topics", []),
        )

    async def list_repositories(
        self,
        owner: str | None = None,
        visibility: str | None = None,
        per_page: int = 100,
    ) -> AsyncIterator[Repository]:
        """List repositories for the authenticated user or organization."""
        page = 1

        while True:
            params = {"per_page": per_page, "page": page}
            if visibility:
                params["visibility"] = visibility

            if owner:
                path = f"/orgs/{owner}/repos"
            else:
                path = "/user/repos"

            try:
                response = await self._request("GET", path, params=params)
                repos = response.json()

                if not repos:
                    break

                for repo_data in repos:
                    yield self._parse_repository(repo_data)

                if len(repos) < per_page:
                    break

                page += 1
            except Exception:
                break

    async def get_repository(self, owner: str, name: str) -> Repository:
        """Get a specific repository by owner and name."""
        response = await self._request("GET", f"/repos/{owner}/{name}")
        return self._parse_repository(response.json())

    async def get_default_branch(self, owner: str, name: str) -> str:
        """Get the default branch for a repository."""
        repo = await self.get_repository(owner, name)
        return repo.default_branch

    async def get_latest_commit(
        self, owner: str, name: str, ref: str | None = None
    ) -> str:
        """Get the latest commit SHA for a branch or ref."""
        if ref is None:
            ref = await self.get_default_branch(owner, name)

        response = await self._request(
            "GET", f"/repos/{owner}/{name}/commits/{ref}"
        )
        return response.json()["sha"]

    async def parse_webhook(
        self, headers: dict, body: bytes
    ) -> WebhookEvent | None:
        """Parse a GitHub webhook payload."""
        event_type = headers.get("x-github-event")
        if not event_type:
            return None

        if self.webhook_secret:
            signature = headers.get("x-hub-signature-256", "")
            expected = "sha256=" + hmac.new(
                self.webhook_secret.encode(),
                body,
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError("Invalid webhook signature")

        payload = json.loads(body)
        repo_data = payload.get("repository", {})

        if not repo_data:
            return None

        repo = Repository(
            provider=self.provider_name,
            owner=repo_data.get("owner", {}).get("login", ""),
            name=repo_data.get("name", ""),
            full_name=repo_data.get("full_name", ""),
            clone_url=repo_data.get("clone_url", ""),
            ssh_url=repo_data.get("ssh_url"),
            default_branch=repo_data.get("default_branch", "main"),
        )

        return WebhookEvent(
            event_type=event_type,
            repository=repo,
            ref=payload.get("ref"),
            before=payload.get("before"),
            after=payload.get("after"),
            payload=payload,
        )
