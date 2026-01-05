"""Tests for storage module."""

from __future__ import annotations

import pytest

from pythia.storage.efs import EFSStorage, IndexMetadata


class TestIndexMetadata:
    """Tests for IndexMetadata."""

    def test_metadata_creation(self):
        metadata = IndexMetadata(
            name="github/owner/repo",
            provider="github",
            owner="owner",
            repo="repo",
            commit="abc123",
            indexed_at="2025-01-01T00:00:00",
            file_count=100,
            size_bytes=1024000,
        )
        assert metadata.name == "github/owner/repo"
        assert metadata.file_count == 100


class TestEFSStorage:
    """Tests for EFSStorage."""

    @pytest.fixture
    def efs_storage(self, storage_config):
        storage_config.ensure_directories()
        return EFSStorage(storage_config)

    def test_register_index(self, efs_storage):
        metadata = efs_storage.register_index(
            name="github/test/repo",
            provider="github",
            owner="test",
            repo="repo",
            commit="abc123",
            file_count=50,
            size_bytes=512000,
        )

        assert metadata.name == "github/test/repo"
        assert metadata.file_count == 50

    def test_get_index(self, efs_storage):
        efs_storage.register_index(
            name="github/test/repo",
            provider="github",
            owner="test",
            repo="repo",
            commit="abc123",
            file_count=50,
            size_bytes=512000,
        )

        retrieved = efs_storage.get_index("github/test/repo")
        assert retrieved is not None
        assert retrieved.commit == "abc123"

    def test_get_index_not_found(self, efs_storage):
        retrieved = efs_storage.get_index("nonexistent")
        assert retrieved is None

    def test_list_indexes(self, efs_storage):
        efs_storage.register_index(
            name="github/test/repo1",
            provider="github",
            owner="test",
            repo="repo1",
            commit="abc123",
            file_count=50,
            size_bytes=512000,
        )
        efs_storage.register_index(
            name="github/test/repo2",
            provider="github",
            owner="test",
            repo="repo2",
            commit="def456",
            file_count=100,
            size_bytes=1024000,
        )

        indexes = list(efs_storage.list_indexes())
        assert len(indexes) == 2

    def test_remove_index(self, efs_storage):
        efs_storage.register_index(
            name="github/test/repo",
            provider="github",
            owner="test",
            repo="repo",
            commit="abc123",
            file_count=50,
            size_bytes=512000,
        )

        result = efs_storage.remove_index("github/test/repo")
        assert result is True
        assert efs_storage.get_index("github/test/repo") is None

    def test_remove_nonexistent_index(self, efs_storage):
        result = efs_storage.remove_index("nonexistent")
        assert result is False

    def test_get_storage_stats(self, efs_storage):
        efs_storage.register_index(
            name="github/test/repo",
            provider="github",
            owner="test",
            repo="repo",
            commit="abc123",
            file_count=50,
            size_bytes=1048576,  # 1 MB
        )

        stats = efs_storage.get_storage_stats()
        assert stats["total_indexes"] == 1
        assert stats["total_files"] == 50
        assert stats["total_size_mb"] == 1.0

    def test_persistence(self, storage_config):
        storage_config.ensure_directories()

        storage1 = EFSStorage(storage_config)
        storage1.register_index(
            name="github/test/repo",
            provider="github",
            owner="test",
            repo="repo",
            commit="abc123",
            file_count=50,
            size_bytes=512000,
        )

        storage2 = EFSStorage(storage_config)
        retrieved = storage2.get_index("github/test/repo")
        assert retrieved is not None
        assert retrieved.commit == "abc123"

    def test_get_index_path(self, efs_storage):
        path = efs_storage.get_index_path("github", "owner", "repo")
        assert "github" in str(path)
        assert "owner" in str(path)
        assert "repo.leann" in str(path)

    def test_get_repo_path(self, efs_storage):
        path = efs_storage.get_repo_path("github", "owner", "repo")
        assert "github" in str(path)
        assert "owner" in str(path)
        assert "repo" in str(path)
