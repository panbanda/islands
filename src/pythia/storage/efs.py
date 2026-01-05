"""EFS storage management for Kubernetes environments."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from pythia.config.settings import StorageConfig

logger = logging.getLogger(__name__)


@dataclass
class IndexMetadata:
    """Metadata for a stored index."""

    name: str
    provider: str
    owner: str
    repo: str
    commit: str
    indexed_at: str
    file_count: int
    size_bytes: int


class EFSStorage:
    """Manage index storage on EFS for Kubernetes deployments."""

    def __init__(self, config: StorageConfig):
        self.config = config
        self.metadata_file = config.base_path / "metadata.json"
        self._metadata: dict[str, IndexMetadata] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load metadata from disk."""
        if self.metadata_file.exists():
            try:
                data = json.loads(self.metadata_file.read_text())
                for name, meta in data.items():
                    self._metadata[name] = IndexMetadata(**meta)
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")

    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        self.config.base_path.mkdir(parents=True, exist_ok=True)
        data = {name: asdict(meta) for name, meta in self._metadata.items()}
        self.metadata_file.write_text(json.dumps(data, indent=2))

    def register_index(
        self,
        name: str,
        provider: str,
        owner: str,
        repo: str,
        commit: str,
        file_count: int,
        size_bytes: int,
    ) -> IndexMetadata:
        """Register an index in the metadata store."""
        metadata = IndexMetadata(
            name=name,
            provider=provider,
            owner=owner,
            repo=repo,
            commit=commit,
            indexed_at=datetime.now().isoformat(),
            file_count=file_count,
            size_bytes=size_bytes,
        )
        self._metadata[name] = metadata
        self._save_metadata()
        logger.info(f"Registered index: {name}")
        return metadata

    def get_index(self, name: str) -> IndexMetadata | None:
        """Get metadata for an index."""
        return self._metadata.get(name)

    def list_indexes(self) -> Iterator[IndexMetadata]:
        """List all registered indexes."""
        yield from self._metadata.values()

    def remove_index(self, name: str) -> bool:
        """Remove an index from the metadata store."""
        if name in self._metadata:
            del self._metadata[name]
            self._save_metadata()
            logger.info(f"Removed index: {name}")
            return True
        return False

    def get_index_path(self, provider: str, owner: str, repo: str) -> Path:
        """Get the storage path for an index."""
        return self.config.indexes_path / provider / owner / f"{repo}.leann"

    def get_repo_path(self, provider: str, owner: str, repo: str) -> Path:
        """Get the storage path for a repository."""
        return self.config.repos_path / provider / owner / repo

    def cleanup_orphaned_indexes(self) -> int:
        """Remove indexes that don't have corresponding metadata."""
        removed = 0
        for index_path in self.config.indexes_path.rglob("*.leann"):
            parts = index_path.relative_to(self.config.indexes_path).parts
            if len(parts) >= 3:
                name = f"{parts[0]}/{parts[1]}/{parts[2].replace('.leann', '')}"
                if name not in self._metadata:
                    logger.info(f"Removing orphaned index: {index_path}")
                    index_path.unlink(missing_ok=True)
                    removed += 1
        return removed

    def get_storage_stats(self) -> dict:
        """Get storage statistics."""
        total_indexes = len(self._metadata)
        total_size = sum(m.size_bytes for m in self._metadata.values())
        total_files = sum(m.file_count for m in self._metadata.values())

        return {
            "total_indexes": total_indexes,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "total_files": total_files,
            "repos_path": str(self.config.repos_path),
            "indexes_path": str(self.config.indexes_path),
        }
