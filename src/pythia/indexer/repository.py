"""Repository management for cloning and updating repositories."""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

import git
from git import Repo

from pythia.providers.base import GitProvider, Repository

logger = logging.getLogger(__name__)


@dataclass
class RepositoryState:
    """Tracks the state of a managed repository."""

    repository: Repository
    local_path: Path
    last_commit: str | None = None
    last_synced: datetime | None = None
    indexed: bool = False
    error: str | None = None


class RepositoryManager:
    """Manages repository cloning, updating, and state tracking."""

    def __init__(
        self,
        storage_path: Path,
        providers: dict[str, GitProvider],
        max_concurrent: int = 4,
    ):
        self.storage_path = storage_path
        self.providers = providers
        self.max_concurrent = max_concurrent
        self._states: dict[str, RepositoryState] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()

    def _get_local_path(self, repo: Repository) -> Path:
        """Get the local filesystem path for a repository."""
        return self.storage_path / repo.provider / repo.owner / repo.name

    def _get_state_key(self, repo: Repository) -> str:
        """Get a unique key for a repository."""
        return f"{repo.provider}/{repo.full_name}"

    async def get_state(self, repo: Repository) -> RepositoryState | None:
        """Get the current state of a repository."""
        return self._states.get(self._get_state_key(repo))

    async def list_states(self) -> AsyncIterator[RepositoryState]:
        """List all tracked repository states."""
        for state in self._states.values():
            yield state

    async def clone_repository(self, repo: Repository) -> RepositoryState:
        """Clone a repository to local storage."""
        async with self._semaphore:
            local_path = self._get_local_path(repo)
            state_key = self._get_state_key(repo)

            try:
                if local_path.exists():
                    shutil.rmtree(local_path)

                local_path.parent.mkdir(parents=True, exist_ok=True)

                provider = self.providers.get(repo.provider)
                if not provider:
                    raise ValueError(f"Unknown provider: {repo.provider}")

                clone_url = provider.get_clone_url(repo)

                logger.info(f"Cloning {repo.full_name} to {local_path}")

                def _clone():
                    return Repo.clone_from(
                        clone_url,
                        local_path,
                        branch=repo.default_branch,
                        depth=1,
                    )

                loop = asyncio.get_event_loop()
                git_repo = await loop.run_in_executor(None, _clone)

                last_commit = git_repo.head.commit.hexsha

                state = RepositoryState(
                    repository=repo,
                    local_path=local_path,
                    last_commit=last_commit,
                    last_synced=datetime.now(),
                    indexed=False,
                    error=None,
                )

                async with self._lock:
                    self._states[state_key] = state

                logger.info(f"Successfully cloned {repo.full_name}")
                return state

            except Exception as e:
                logger.error(f"Failed to clone {repo.full_name}: {e}")
                state = RepositoryState(
                    repository=repo,
                    local_path=local_path,
                    error=str(e),
                )
                async with self._lock:
                    self._states[state_key] = state
                raise

    async def update_repository(self, repo: Repository) -> RepositoryState:
        """Pull latest changes for a repository."""
        async with self._semaphore:
            local_path = self._get_local_path(repo)
            state_key = self._get_state_key(repo)

            if not local_path.exists():
                return await self.clone_repository(repo)

            try:
                def _pull():
                    git_repo = Repo(local_path)
                    origin = git_repo.remote("origin")
                    origin.pull()
                    return git_repo.head.commit.hexsha

                loop = asyncio.get_event_loop()
                last_commit = await loop.run_in_executor(None, _pull)

                current_state = self._states.get(state_key)
                needs_reindex = (
                    current_state is None
                    or current_state.last_commit != last_commit
                )

                state = RepositoryState(
                    repository=repo,
                    local_path=local_path,
                    last_commit=last_commit,
                    last_synced=datetime.now(),
                    indexed=not needs_reindex and (current_state.indexed if current_state else False),
                    error=None,
                )

                async with self._lock:
                    self._states[state_key] = state

                logger.info(f"Updated {repo.full_name}, needs_reindex={needs_reindex}")
                return state

            except Exception as e:
                logger.error(f"Failed to update {repo.full_name}: {e}")
                current_state = self._states.get(state_key)
                if current_state:
                    current_state.error = str(e)
                raise

    async def mark_indexed(self, repo: Repository) -> None:
        """Mark a repository as indexed."""
        state_key = self._get_state_key(repo)
        async with self._lock:
            if state_key in self._states:
                self._states[state_key].indexed = True
                self._states[state_key].error = None

    async def remove_repository(self, repo: Repository) -> None:
        """Remove a repository from local storage."""
        local_path = self._get_local_path(repo)
        state_key = self._get_state_key(repo)

        if local_path.exists():
            shutil.rmtree(local_path)

        async with self._lock:
            self._states.pop(state_key, None)

        logger.info(f"Removed {repo.full_name}")

    async def needs_reindex(self, repo: Repository) -> bool:
        """Check if a repository needs to be re-indexed."""
        state = await self.get_state(repo)
        if state is None:
            return True
        return not state.indexed or state.error is not None
