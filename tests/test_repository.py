"""Tests for repository management."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from pythia.indexer.repository import RepositoryManager, RepositoryState
from pythia.providers.base import Repository, GitProvider


class TestRepositoryState:
    """Tests for RepositoryState."""

    def test_state_creation(self, mock_repository, temp_dir):
        state = RepositoryState(
            repository=mock_repository,
            local_path=temp_dir / "repo",
            last_commit="abc123",
            indexed=True,
        )
        assert state.repository == mock_repository
        assert state.last_commit == "abc123"
        assert state.indexed is True
        assert state.error is None


class TestRepositoryManager:
    """Tests for RepositoryManager."""

    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock(spec=GitProvider)
        provider.provider_name = "github"
        provider.get_clone_url = MagicMock(return_value="https://github.com/test/repo.git")
        return provider

    @pytest.fixture
    def repo_manager(self, temp_dir, mock_provider):
        return RepositoryManager(
            storage_path=temp_dir,
            providers={"github": mock_provider},
            max_concurrent=2,
        )

    def test_get_local_path(self, repo_manager, mock_repository):
        path = repo_manager._get_local_path(mock_repository)
        assert "github" in str(path)
        assert "test-owner" in str(path)
        assert "test-repo" in str(path)

    def test_get_state_key(self, repo_manager, mock_repository):
        key = repo_manager._get_state_key(mock_repository)
        assert key == "github/test-owner/test-repo"

    @pytest.mark.asyncio
    async def test_get_state_not_found(self, repo_manager, mock_repository):
        state = await repo_manager.get_state(mock_repository)
        assert state is None

    @pytest.mark.asyncio
    async def test_needs_reindex_new_repo(self, repo_manager, mock_repository):
        needs = await repo_manager.needs_reindex(mock_repository)
        assert needs is True

    @pytest.mark.asyncio
    async def test_list_states_empty(self, repo_manager):
        states = []
        async for state in repo_manager.list_states():
            states.append(state)
        assert len(states) == 0

    @pytest.mark.asyncio
    async def test_mark_indexed(self, repo_manager, mock_repository, temp_dir):
        state = RepositoryState(
            repository=mock_repository,
            local_path=temp_dir / "repo",
            indexed=False,
        )
        repo_manager._states[repo_manager._get_state_key(mock_repository)] = state

        await repo_manager.mark_indexed(mock_repository)

        updated_state = await repo_manager.get_state(mock_repository)
        assert updated_state.indexed is True


class TestRepositoryManagerClone:
    """Tests for repository cloning."""

    @pytest.fixture
    def mock_git_repo(self):
        mock_repo = MagicMock()
        mock_repo.head.commit.hexsha = "abc123def456"
        return mock_repo

    @pytest.mark.asyncio
    async def test_clone_creates_state(self, temp_dir, mock_repository, mock_git_repo):
        mock_provider = MagicMock(spec=GitProvider)
        mock_provider.provider_name = "github"
        mock_provider.get_clone_url = MagicMock(return_value="https://github.com/test/repo.git")

        manager = RepositoryManager(
            storage_path=temp_dir,
            providers={"github": mock_provider},
        )

        with patch("pythia.indexer.repository.Repo") as mock_repo_class:
            mock_repo_class.clone_from.return_value = mock_git_repo

            state = await manager.clone_repository(mock_repository)

            assert state is not None
            assert state.last_commit == "abc123def456"
            assert state.error is None
