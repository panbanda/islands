"""Bitbucket provider implementation."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from collections.abc import AsyncIterator
from datetime import datetime

from pythia.providers.base import (
    AuthType,
    GitProvider,
    ProviderAuth,
    Repository,
    WebhookEvent,
)


class BitbucketProvider(GitProvider):
    """Bitbucket Cloud API provider implementation."""

    provider_name = "bitbucket"

    def __init__(
        self,
        base_url: str = "https://api.bitbucket.org/2.0",
        auth: ProviderAuth | None = None,
        webhook_secret: str | None = None,
        **kwargs,
    ):
        super().__init__(base_url, auth, **kwargs)
        self.webhook_secret = webhook_secret

    def _build_auth_headers(self) -> dict[str, str]:
        """Build Bitbucket authentication headers."""
        headers = {"Accept": "application/json"}
        if self.auth:
            if self.auth.auth_type == AuthType.TOKEN and self.auth.token:
                headers["Authorization"] = f"Bearer {self.auth.token}"
            elif self.auth.auth_type == AuthType.BASIC:
                credentials = f"{self.auth.username}:{self.auth.password}"
                encoded = base64.b64encode(credentials.encode()).decode()
                headers["Authorization"] = f"Basic {encoded}"
        return headers

    def _parse_repository(self, data: dict) -> Repository:
        """Parse Bitbucket API response into Repository object."""
        links = data.get("links", {})
        clone_links = links.get("clone", [])

        https_url = ""
        ssh_url = None
        for link in clone_links:
            if link.get("name") == "https":
                https_url = link.get("href", "")
            elif link.get("name") == "ssh":
                ssh_url = link.get("href")

        mainbranch = data.get("mainbranch", {})

        return Repository(
            provider=self.provider_name,
            owner=data.get("owner", {}).get("username", data.get("workspace", {}).get("slug", "")),
            name=data["slug"],
            full_name=data["full_name"],
            clone_url=https_url,
            ssh_url=ssh_url,
            default_branch=mainbranch.get("name", "main"),
            description=data.get("description"),
            language=data.get("language"),
            size_kb=data.get("size", 0) // 1024,
            last_updated=datetime.fromisoformat(data["updated_on"].replace("Z", "+00:00"))
            if data.get("updated_on")
            else None,
            is_private=data.get("is_private", False),
            topics=[],
        )

    async def list_repositories(
        self,
        owner: str | None = None,
        visibility: str | None = None,
        per_page: int = 100,
    ) -> AsyncIterator[Repository]:
        """List repositories for the authenticated user or workspace."""
        params = {"pagelen": min(per_page, 100)}

        if owner:
            path = f"/repositories/{owner}"
        else:
            path = "/user/permissions/repositories"

        next_url: str | None = path

        while next_url:
            try:
                if next_url.startswith("http"):
                    client = await self._get_client()
                    response = await client.get(next_url, params=params)
                    response.raise_for_status()
                else:
                    response = await self._request("GET", next_url, params=params)

                data = response.json()
                values = data.get("values", [])

                for item in values:
                    repo_data = item.get("repository", item)
                    if visibility:
                        is_private = repo_data.get("is_private", False)
                        if visibility == "private" and not is_private:
                            continue
                        if visibility == "public" and is_private:
                            continue
                    yield self._parse_repository(repo_data)

                next_url = data.get("next")
                params = {}
            except Exception:
                break

    async def get_repository(self, owner: str, name: str) -> Repository:
        """Get a specific repository by owner and name."""
        response = await self._request("GET", f"/repositories/{owner}/{name}")
        return self._parse_repository(response.json())

    async def get_default_branch(self, owner: str, name: str) -> str:
        """Get the default branch for a repository."""
        repo = await self.get_repository(owner, name)
        return repo.default_branch

    async def get_latest_commit(self, owner: str, name: str, ref: str | None = None) -> str:
        """Get the latest commit SHA for a branch or ref."""
        if ref is None:
            ref = await self.get_default_branch(owner, name)

        response = await self._request(
            "GET",
            f"/repositories/{owner}/{name}/commits/{ref}",
            params={"pagelen": 1},
        )
        commits = response.json().get("values", [])
        if commits:
            return commits[0]["hash"]
        raise ValueError(f"No commits found for {ref}")

    async def parse_webhook(self, headers: dict, body: bytes) -> WebhookEvent | None:
        """Parse a Bitbucket webhook payload."""
        event_key = headers.get("x-event-key", "")
        if not event_key:
            return None

        if self.webhook_secret:
            expected = hmac.new(
                self.webhook_secret.encode(),
                body,
                hashlib.sha256,
            ).hexdigest()
            signature = headers.get("x-hub-signature", "")
            if signature and not hmac.compare_digest(signature, f"sha256={expected}"):
                raise ValueError("Invalid webhook signature")

        payload = json.loads(body)
        repo_data = payload.get("repository", {})

        if not repo_data:
            return None

        repo = self._parse_repository(repo_data)

        push_changes = payload.get("push", {}).get("changes", [])
        ref = None
        before = None
        after = None
        if push_changes:
            change = push_changes[0]
            new_state = change.get("new", {})
            old_state = change.get("old", {})
            ref = new_state.get("name")
            after = new_state.get("target", {}).get("hash")
            before = old_state.get("target", {}).get("hash") if old_state else None

        return WebhookEvent(
            event_type=event_key,
            repository=repo,
            ref=ref,
            before=before,
            after=after,
            payload=payload,
        )
