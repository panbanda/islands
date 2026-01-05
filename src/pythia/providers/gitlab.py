"""GitLab provider implementation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import datetime
from urllib.parse import quote_plus

from pythia.providers.base import (
    AuthType,
    GitProvider,
    ProviderAuth,
    Repository,
    WebhookEvent,
)


class GitLabProvider(GitProvider):
    """GitLab API provider implementation."""

    provider_name = "gitlab"

    def __init__(
        self,
        base_url: str = "https://gitlab.com/api/v4",
        auth: ProviderAuth | None = None,
        webhook_secret: str | None = None,
        **kwargs,
    ):
        super().__init__(base_url, auth, **kwargs)
        self.webhook_secret = webhook_secret

    def _build_auth_headers(self) -> dict[str, str]:
        """Build GitLab authentication headers."""
        headers = {"Accept": "application/json"}
        if self.auth:
            if self.auth.auth_type in (AuthType.TOKEN, AuthType.OAUTH) and self.auth.token:
                headers["PRIVATE-TOKEN"] = self.auth.token
        return headers

    def _parse_repository(self, data: dict) -> Repository:
        """Parse GitLab API response into Repository object."""
        namespace = data.get("namespace", {})
        return Repository(
            provider=self.provider_name,
            owner=namespace.get("path", namespace.get("full_path", "").split("/")[0]),
            name=data["path"],
            full_name=data["path_with_namespace"],
            clone_url=data["http_url_to_repo"],
            ssh_url=data.get("ssh_url_to_repo"),
            default_branch=data.get("default_branch", "main"),
            description=data.get("description"),
            language=None,
            size_kb=data.get("statistics", {}).get("repository_size", 0) // 1024,
            last_updated=datetime.fromisoformat(data["last_activity_at"].replace("Z", "+00:00"))
            if data.get("last_activity_at")
            else None,
            is_private=data.get("visibility") == "private",
            topics=data.get("topics", []),
        )

    async def list_repositories(
        self,
        owner: str | None = None,
        visibility: str | None = None,
        per_page: int = 100,
    ) -> AsyncIterator[Repository]:
        """List repositories for the authenticated user or group."""
        page = 1

        while True:
            params = {"per_page": per_page, "page": page}
            if visibility:
                params["visibility"] = visibility

            if owner:
                encoded_owner = quote_plus(owner)
                path = f"/groups/{encoded_owner}/projects"
            else:
                path = "/projects"
                params["membership"] = "true"

            try:
                response = await self._request("GET", path, params=params)
                projects = response.json()

                if not projects:
                    break

                for project_data in projects:
                    yield self._parse_repository(project_data)

                if len(projects) < per_page:
                    break

                page += 1
            except Exception:
                break

    async def get_repository(self, owner: str, name: str) -> Repository:
        """Get a specific repository by owner and name."""
        project_path = quote_plus(f"{owner}/{name}")
        response = await self._request("GET", f"/projects/{project_path}")
        return self._parse_repository(response.json())

    async def get_default_branch(self, owner: str, name: str) -> str:
        """Get the default branch for a repository."""
        repo = await self.get_repository(owner, name)
        return repo.default_branch

    async def get_latest_commit(self, owner: str, name: str, ref: str | None = None) -> str:
        """Get the latest commit SHA for a branch or ref."""
        project_path = quote_plus(f"{owner}/{name}")

        if ref is None:
            ref = await self.get_default_branch(owner, name)

        response = await self._request(
            "GET",
            f"/projects/{project_path}/repository/commits/{ref}",
        )
        return response.json()["id"]

    async def parse_webhook(self, headers: dict, body: bytes) -> WebhookEvent | None:
        """Parse a GitLab webhook payload."""
        event_type = headers.get("x-gitlab-event", "").lower().replace(" ", "_")
        if not event_type:
            return None

        if self.webhook_secret:
            token = headers.get("x-gitlab-token", "")
            if token != self.webhook_secret:
                raise ValueError("Invalid webhook token")

        payload = json.loads(body)
        project = payload.get("project", {})

        if not project:
            return None

        repo = Repository(
            provider=self.provider_name,
            owner=project.get("namespace", ""),
            name=project.get("name", ""),
            full_name=project.get("path_with_namespace", ""),
            clone_url=project.get("http_url", ""),
            ssh_url=project.get("ssh_url"),
            default_branch=project.get("default_branch", "main"),
        )

        return WebhookEvent(
            event_type=event_type,
            repository=repo,
            ref=payload.get("ref"),
            before=payload.get("before"),
            after=payload.get("after"),
            payload=payload,
        )
