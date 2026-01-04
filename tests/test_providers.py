"""Tests for git provider implementations."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import json

from pythia.providers.base import AuthType, GitProvider, ProviderAuth, Repository
from pythia.providers.github import GitHubProvider
from pythia.providers.gitlab import GitLabProvider
from pythia.providers.bitbucket import BitbucketProvider
from pythia.providers.gitea import GiteaProvider
from pythia.providers.factory import create_provider
from pythia.config.settings import ProviderConfig, ProviderType


class TestProviderAuth:
    """Tests for ProviderAuth."""

    def test_token_auth(self):
        auth = ProviderAuth(auth_type=AuthType.TOKEN, token="test-token")
        assert auth.auth_type == AuthType.TOKEN
        assert auth.token == "test-token"

    def test_ssh_auth(self, temp_dir):
        key_path = temp_dir / "id_rsa"
        auth = ProviderAuth(auth_type=AuthType.SSH, ssh_key_path=key_path)
        assert auth.auth_type == AuthType.SSH
        assert auth.ssh_key_path == key_path


class TestRepository:
    """Tests for Repository dataclass."""

    def test_local_path(self, mock_repository):
        expected = "github/test-owner/test-repo"
        assert str(mock_repository.local_path) == expected

    def test_repository_attributes(self, mock_repository):
        assert mock_repository.provider == "github"
        assert mock_repository.owner == "test-owner"
        assert mock_repository.name == "test-repo"
        assert mock_repository.full_name == "test-owner/test-repo"
        assert mock_repository.default_branch == "main"


class TestGitHubProvider:
    """Tests for GitHubProvider."""

    def test_init_default(self):
        provider = GitHubProvider()
        assert provider.base_url == "https://api.github.com"
        assert provider.provider_name == "github"

    def test_init_with_auth(self, provider_auth):
        provider = GitHubProvider(auth=provider_auth)
        headers = provider._build_auth_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-token-12345"

    def test_build_auth_headers_no_auth(self):
        provider = GitHubProvider()
        headers = provider._build_auth_headers()
        assert "Accept" in headers
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_parse_webhook_push(self):
        provider = GitHubProvider()
        headers = {"x-github-event": "push"}
        payload = {
            "ref": "refs/heads/main",
            "before": "abc123",
            "after": "def456",
            "repository": {
                "full_name": "owner/repo",
                "name": "repo",
                "clone_url": "https://github.com/owner/repo.git",
                "default_branch": "main",
                "owner": {"login": "owner"},
            },
        }
        body = json.dumps(payload).encode()

        event = await provider.parse_webhook(headers, body)
        assert event is not None
        assert event.event_type == "push"
        assert event.ref == "refs/heads/main"
        assert event.repository.full_name == "owner/repo"


class TestGitLabProvider:
    """Tests for GitLabProvider."""

    def test_init_default(self):
        provider = GitLabProvider()
        assert provider.base_url == "https://gitlab.com/api/v4"
        assert provider.provider_name == "gitlab"

    def test_build_auth_headers_with_token(self, provider_auth):
        provider = GitLabProvider(auth=provider_auth)
        headers = provider._build_auth_headers()
        assert "PRIVATE-TOKEN" in headers
        assert headers["PRIVATE-TOKEN"] == "test-token-12345"


class TestBitbucketProvider:
    """Tests for BitbucketProvider."""

    def test_init_default(self):
        provider = BitbucketProvider()
        assert provider.base_url == "https://api.bitbucket.org/2.0"
        assert provider.provider_name == "bitbucket"

    def test_build_auth_headers_with_token(self, provider_auth):
        provider = BitbucketProvider(auth=provider_auth)
        headers = provider._build_auth_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-token-12345"


class TestGiteaProvider:
    """Tests for GiteaProvider."""

    def test_init_requires_base_url(self):
        provider = GiteaProvider(base_url="https://gitea.example.com")
        assert provider.base_url == "https://gitea.example.com/api/v1"
        assert provider.provider_name == "gitea"

    def test_init_appends_api_path(self):
        provider = GiteaProvider(base_url="https://gitea.example.com")
        assert provider.base_url.endswith("/api/v1")

    def test_init_preserves_api_path(self):
        provider = GiteaProvider(base_url="https://gitea.example.com/api/v1")
        assert provider.base_url == "https://gitea.example.com/api/v1"


class TestProviderFactory:
    """Tests for provider factory."""

    def test_create_github_provider(self):
        config = ProviderConfig(type=ProviderType.GITHUB, token="test-token")
        provider = create_provider(config)
        assert isinstance(provider, GitHubProvider)

    def test_create_gitlab_provider(self):
        config = ProviderConfig(type=ProviderType.GITLAB, token="test-token")
        provider = create_provider(config)
        assert isinstance(provider, GitLabProvider)

    def test_create_bitbucket_provider(self):
        config = ProviderConfig(type=ProviderType.BITBUCKET, token="test-token")
        provider = create_provider(config)
        assert isinstance(provider, BitbucketProvider)

    def test_create_gitea_provider(self):
        config = ProviderConfig(
            type=ProviderType.GITEA,
            base_url="https://gitea.example.com",
            token="test-token",
        )
        provider = create_provider(config)
        assert isinstance(provider, GiteaProvider)

    def test_create_gitea_requires_base_url(self):
        config = ProviderConfig(type=ProviderType.GITEA, token="test-token")
        with pytest.raises(ValueError, match="requires a base_url"):
            create_provider(config)
