"""Tests for CLI module."""

from __future__ import annotations

import pytest

from pythia.cli import parse_repo_url


class TestParseRepoUrl:
    """Tests for parse_repo_url function."""

    def test_github_https(self):
        provider, owner, repo, base_url = parse_repo_url("https://github.com/owner/repo")
        assert provider == "github"
        assert owner == "owner"
        assert repo == "repo"
        assert base_url is None

    def test_github_https_with_git_suffix(self):
        provider, owner, repo, base_url = parse_repo_url("https://github.com/owner/repo.git")
        assert provider == "github"
        assert owner == "owner"
        assert repo == "repo"
        assert base_url is None

    def test_github_ssh(self):
        provider, owner, repo, base_url = parse_repo_url("git@github.com:owner/repo.git")
        assert provider == "github"
        assert owner == "owner"
        assert repo == "repo"
        assert base_url is None

    def test_gitlab_https(self):
        provider, owner, repo, base_url = parse_repo_url("https://gitlab.com/owner/repo")
        assert provider == "gitlab"
        assert owner == "owner"
        assert repo == "repo"
        assert base_url is None

    def test_bitbucket_https(self):
        provider, owner, repo, base_url = parse_repo_url("https://bitbucket.org/owner/repo")
        assert provider == "bitbucket"
        assert owner == "owner"
        assert repo == "repo"
        assert base_url is None

    def test_gitea_custom_domain(self):
        provider, owner, repo, base_url = parse_repo_url("https://gitea.example.com/owner/repo")
        assert provider == "gitea"
        assert owner == "owner"
        assert repo == "repo"
        assert base_url == "https://gitea.example.com"

    def test_gitea_ssh_custom_domain(self):
        provider, owner, repo, base_url = parse_repo_url("git@gitea.example.com:owner/repo.git")
        assert provider == "gitea"
        assert owner == "owner"
        assert repo == "repo"
        assert base_url == "https://gitea.example.com"

    def test_url_with_trailing_slash(self):
        provider, owner, repo, base_url = parse_repo_url("https://github.com/owner/repo/")
        assert provider == "github"
        assert owner == "owner"
        assert repo == "repo"

    def test_url_with_whitespace(self):
        provider, owner, repo, base_url = parse_repo_url("  https://github.com/owner/repo  ")
        assert provider == "github"
        assert owner == "owner"
        assert repo == "repo"

    def test_invalid_url_no_netloc(self):
        with pytest.raises(ValueError, match="Invalid repository URL"):
            parse_repo_url("not-a-url")

    def test_invalid_url_missing_repo(self):
        with pytest.raises(ValueError, match="Invalid repository path"):
            parse_repo_url("https://github.com/owner")

    def test_github_enterprise(self):
        provider, owner, repo, base_url = parse_repo_url("https://github.mycompany.com/owner/repo")
        assert provider == "gitea"  # Falls back to gitea for unknown domains
        assert owner == "owner"
        assert repo == "repo"
        assert base_url == "https://github.mycompany.com"
