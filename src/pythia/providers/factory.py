"""Factory for creating git provider instances."""

from __future__ import annotations

from pythia.config.settings import ProviderConfig, ProviderType
from pythia.providers.base import AuthType, GitProvider, ProviderAuth
from pythia.providers.bitbucket import BitbucketProvider
from pythia.providers.gitea import GiteaProvider
from pythia.providers.github import GitHubProvider
from pythia.providers.gitlab import GitLabProvider


def create_provider(config: ProviderConfig) -> GitProvider:
    """Create a git provider instance from configuration.

    Args:
        config: Provider configuration with type, credentials, and settings.

    Returns:
        Configured GitProvider instance.

    Raises:
        ValueError: If provider type is not supported.
    """
    auth = None
    if config.token:
        auth = ProviderAuth(auth_type=AuthType.TOKEN, token=config.token)
    elif config.ssh_key_path:
        auth = ProviderAuth(auth_type=AuthType.SSH, ssh_key_path=config.ssh_key_path)

    common_kwargs = {
        "auth": auth,
        "webhook_secret": config.webhook_secret,
        "rate_limit_requests": config.rate_limit_requests,
        "rate_limit_window": config.rate_limit_window,
    }

    if config.type == ProviderType.GITHUB:
        return GitHubProvider(
            base_url=config.base_url or "https://api.github.com",
            **common_kwargs,
        )
    elif config.type == ProviderType.GITLAB:
        return GitLabProvider(
            base_url=config.base_url or "https://gitlab.com/api/v4",
            **common_kwargs,
        )
    elif config.type == ProviderType.BITBUCKET:
        return BitbucketProvider(
            base_url=config.base_url or "https://api.bitbucket.org/2.0",
            **common_kwargs,
        )
    elif config.type == ProviderType.GITEA:
        if not config.base_url:
            raise ValueError("Gitea provider requires a base_url")
        return GiteaProvider(
            base_url=config.base_url,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unsupported provider type: {config.type}")
