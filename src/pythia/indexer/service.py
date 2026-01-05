"""Main indexer service that coordinates repository management and LEANN indexing."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pythia.config.settings import Settings
from pythia.indexer.repository import RepositoryManager, RepositoryState
from pythia.providers.base import GitProvider, Repository, WebhookEvent
from pythia.providers.factory import create_provider

logger = logging.getLogger(__name__)


@dataclass
class IndexInfo:
    """Information about a LEANN index."""

    name: str
    path: Path
    repository: Repository
    created_at: datetime
    updated_at: datetime
    file_count: int
    size_bytes: int


class IndexerService:
    """Main service for indexing repositories with LEANN."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.storage.ensure_directories()

        self._providers: dict[str, GitProvider] = {}
        for config in settings.providers:
            provider = create_provider(config)
            self._providers[provider.provider_name] = provider

        self._repo_manager = RepositoryManager(
            storage_path=settings.storage.repos_path,
            providers=self._providers,
            max_concurrent=settings.max_concurrent_syncs,
        )

        self._indexes: dict[str, IndexInfo] = {}
        self._running = False
        self._sync_task: asyncio.Task | None = None

    @property
    def providers(self) -> dict[str, GitProvider]:
        """Get configured providers."""
        return self._providers

    @property
    def repository_manager(self) -> RepositoryManager:
        """Get the repository manager."""
        return self._repo_manager

    def _get_index_path(self, repo: Repository) -> Path:
        """Get the LEANN index path for a repository."""
        return (
            self.settings.storage.indexes_path / repo.provider / repo.owner / f"{repo.name}.leann"
        )

    def _get_index_name(self, repo: Repository) -> str:
        """Get a unique index name for a repository."""
        return f"{repo.provider}/{repo.full_name}"

    async def add_repository(self, repo: Repository) -> RepositoryState:
        """Add and clone a repository."""
        state = await self._repo_manager.clone_repository(repo)
        await self._index_repository(repo)
        return state

    async def sync_repository(self, repo: Repository) -> RepositoryState:
        """Sync a repository and re-index if needed."""
        state = await self._repo_manager.update_repository(repo)
        if await self._repo_manager.needs_reindex(repo):
            await self._index_repository(repo)
        return state

    async def _index_repository(self, repo: Repository) -> IndexInfo:
        """Build a LEANN index for a repository."""
        try:
            from leann import LeannBuilder
        except ImportError:
            logger.warning("LEANN not installed, using mock indexer")
            return await self._mock_index_repository(repo)

        state = await self._repo_manager.get_state(repo)
        if state is None or state.error:
            raise ValueError(f"Repository {repo.full_name} not available for indexing")

        index_path = self._get_index_path(repo)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        backend = self.settings.leann.backend.value
        builder = LeannBuilder(backend_name=backend)

        file_count = 0
        code_extensions = {
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".go",
            ".rs",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".cs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".sql",
            ".sh",
            ".bash",
            ".yaml",
            ".yml",
            ".json",
            ".toml",
            ".md",
            ".rst",
            ".txt",
        }

        for file_path in state.local_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in code_extensions:
                if ".git" in file_path.parts:
                    continue
                try:
                    content = file_path.read_text(errors="ignore")
                    relative_path = file_path.relative_to(state.local_path)
                    builder.add_text(
                        content,
                        metadata={"file": str(relative_path), "repo": repo.full_name},
                    )
                    file_count += 1
                except Exception as e:
                    logger.warning(f"Failed to index {file_path}: {e}")

        if self.settings.leann.no_recompute:
            builder.build_index(str(index_path), no_recompute=True)
        else:
            builder.build_index(str(index_path))

        await self._repo_manager.mark_indexed(repo)

        size_bytes = sum(f.stat().st_size for f in index_path.parent.rglob("*") if f.is_file())

        index_info = IndexInfo(
            name=self._get_index_name(repo),
            path=index_path,
            repository=repo,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            file_count=file_count,
            size_bytes=size_bytes,
        )
        self._indexes[index_info.name] = index_info

        logger.info(f"Indexed {repo.full_name}: {file_count} files")
        return index_info

    async def _mock_index_repository(self, repo: Repository) -> IndexInfo:
        """Mock indexer for when LEANN is not installed."""
        state = await self._repo_manager.get_state(repo)
        if state is None:
            raise ValueError(f"Repository {repo.full_name} not available")

        index_path = self._get_index_path(repo)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(f"Mock index for {repo.full_name}")

        await self._repo_manager.mark_indexed(repo)

        index_info = IndexInfo(
            name=self._get_index_name(repo),
            path=index_path,
            repository=repo,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            file_count=0,
            size_bytes=0,
        )
        self._indexes[index_info.name] = index_info
        return index_info

    async def search(
        self,
        query: str,
        index_names: list[str] | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        """Search across one or more indexes."""
        try:
            from leann import LeannSearcher
        except ImportError:
            logger.warning("LEANN not installed, returning mock results")
            return [{"mock": True, "query": query}]

        results = []
        target_indexes = index_names or list(self._indexes.keys())

        for index_name in target_indexes:
            if index_name not in self._indexes:
                continue

            index_info = self._indexes[index_name]
            try:
                searcher = LeannSearcher(str(index_info.path))
                search_results = searcher.search(query, top_k=top_k)
                for result in search_results:
                    result["index"] = index_name
                    result["repository"] = index_info.repository.full_name
                    results.append(result)
            except Exception as e:
                logger.error(f"Search failed for {index_name}: {e}")

        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:top_k]

    async def list_indexes(self) -> AsyncIterator[IndexInfo]:
        """List all available indexes."""
        for info in self._indexes.values():
            yield info

    async def get_index(self, name: str) -> IndexInfo | None:
        """Get information about a specific index."""
        return self._indexes.get(name)

    async def handle_webhook(self, event: WebhookEvent) -> None:
        """Handle a webhook event from a git provider."""
        if event.event_type in ("push", "push_hook"):
            logger.info(f"Webhook push event for {event.repository.full_name}")
            await self.sync_repository(event.repository)

    async def start_sync_loop(self) -> None:
        """Start the background sync loop."""
        if self._running:
            return

        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info("Started sync loop")

    async def stop_sync_loop(self) -> None:
        """Stop the background sync loop."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped sync loop")

    async def _sync_loop(self) -> None:
        """Background loop to sync repositories periodically."""
        while self._running:
            try:
                async for state in self._repo_manager.list_states():
                    if not self._running:
                        break
                    try:
                        await self.sync_repository(state.repository)
                    except Exception as e:
                        logger.error(f"Sync failed for {state.repository.full_name}: {e}")

                await asyncio.sleep(self.settings.sync_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(60)

    async def close(self) -> None:
        """Clean up resources."""
        await self.stop_sync_loop()
        for provider in self._providers.values():
            await provider.close()
