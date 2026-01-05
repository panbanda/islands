"""Tests for MCP server."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from pythia.indexer.service import IndexerService, IndexInfo
from pythia.mcp.server import PythiaMCPServer
from pythia.providers.base import Repository


class TestPythiaMCPServer:
    """Tests for PythiaMCPServer."""

    @pytest.fixture
    def mock_indexer(self):
        indexer = MagicMock(spec=IndexerService)
        indexer.providers = {"github": MagicMock()}
        indexer.list_indexes = AsyncMock()
        indexer.search = AsyncMock(return_value=[])
        indexer.get_index = AsyncMock(return_value=None)
        indexer.add_repository = AsyncMock()
        indexer.sync_repository = AsyncMock()
        indexer.repository_manager = MagicMock()
        indexer.repository_manager.get_state = AsyncMock(return_value=None)
        return indexer

    @pytest.fixture
    def mcp_server(self, mock_indexer):
        return PythiaMCPServer(mock_indexer)

    @pytest.mark.asyncio
    async def test_handle_list_empty(self, mcp_server, mock_indexer):
        async def empty_gen():
            return
            yield  # Make it an async generator

        mock_indexer.list_indexes = empty_gen

        result = await mcp_server._handle_list()
        assert len(result) == 1
        assert "No indexes available" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_list_with_indexes(self, mcp_server, mock_indexer):
        repo = Repository(
            provider="github",
            owner="test",
            name="repo",
            full_name="test/repo",
            clone_url="https://github.com/test/repo.git",
        )
        index_info = IndexInfo(
            name="github/test/repo",
            path="/data/indexes/github/test/repo.leann",
            repository=repo,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            file_count=100,
            size_bytes=1048576,
        )

        async def gen_indexes():
            yield index_info

        mock_indexer.list_indexes = gen_indexes

        result = await mcp_server._handle_list()
        assert len(result) == 1
        assert "github/test/repo" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_search_no_results(self, mcp_server, mock_indexer):
        mock_indexer.search = AsyncMock(return_value=[])

        result = await mcp_server._handle_search({"query": "test query"})
        assert len(result) == 1
        assert "No results found" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_search_with_results(self, mcp_server, mock_indexer):
        mock_indexer.search = AsyncMock(
            return_value=[
                {
                    "repository": "test/repo",
                    "metadata": {"file": "src/main.py"},
                    "score": 0.95,
                    "text": "def main():\n    pass",
                }
            ]
        )

        result = await mcp_server._handle_search({"query": "main function"})
        assert len(result) == 1
        assert "test/repo" in result[0].text
        assert "0.95" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_add_repo_invalid_url(self, mcp_server, mock_indexer):
        result = await mcp_server._handle_add_repo({"url": "not-a-url"})
        assert len(result) == 1
        assert "Invalid URL" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_add_repo_unknown_provider(self, mcp_server, mock_indexer):
        mock_indexer.providers = {}

        result = await mcp_server._handle_add_repo({"url": "https://github.com/test/repo"})
        assert len(result) == 1
        assert "not configured" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_sync_not_found(self, mcp_server, mock_indexer):
        mock_indexer.get_index = AsyncMock(return_value=None)

        result = await mcp_server._handle_sync({"index_name": "nonexistent"})
        assert len(result) == 1
        assert "not found" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_status_all(self, mcp_server, mock_indexer):
        async def empty_gen():
            return
            yield

        mock_indexer.list_indexes = empty_gen

        result = await mcp_server._handle_status({})
        assert len(result) == 1
