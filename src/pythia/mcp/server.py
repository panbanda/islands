"""MCP Server implementation for Pythia codebase inquiries."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from pythia.config.settings import Settings
from pythia.indexer.service import IndexerService
from pythia.cli import parse_repo_url

logger = logging.getLogger(__name__)


class PythiaMCPServer:
    """MCP Server for Pythia codebase semantic search."""

    def __init__(self, indexer: IndexerService):
        self.indexer = indexer
        self.server = Server("pythia")
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Set up MCP tool handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="pythia_list",
                    description="List all indexed codebases available for search",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                ),
                Tool(
                    name="pythia_search",
                    description="Semantic search across indexed codebases. Returns relevant code snippets and documentation.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language query to search for in the codebase",
                            },
                            "indexes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional list of index names to search. If not provided, searches all indexes.",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return (default: 10)",
                                "default": 10,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="pythia_add_repo",
                    description="Add a repository to be indexed by URL.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "Repository URL (e.g., https://github.com/owner/repo)",
                            },
                        },
                        "required": ["url"],
                    },
                ),
                Tool(
                    name="pythia_sync",
                    description="Sync and re-index a repository to get latest changes",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "index_name": {
                                "type": "string",
                                "description": "Index name in format 'provider/owner/repo'",
                            },
                        },
                        "required": ["index_name"],
                    },
                ),
                Tool(
                    name="pythia_status",
                    description="Get the status of a specific index or all indexes",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "index_name": {
                                "type": "string",
                                "description": "Optional index name to get status for",
                            },
                        },
                        "required": [],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            try:
                if name == "pythia_list":
                    return await self._handle_list()
                elif name == "pythia_search":
                    return await self._handle_search(arguments)
                elif name == "pythia_add_repo":
                    return await self._handle_add_repo(arguments)
                elif name == "pythia_sync":
                    return await self._handle_sync(arguments)
                elif name == "pythia_status":
                    return await self._handle_status(arguments)
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
            except Exception as e:
                logger.error(f"Tool {name} failed: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def _handle_list(self) -> list[TextContent]:
        """Handle pythia_list tool call."""
        indexes = []
        async for info in self.indexer.list_indexes():
            indexes.append({
                "name": info.name,
                "repository": info.repository.full_name,
                "provider": info.repository.provider,
                "file_count": info.file_count,
                "size_mb": round(info.size_bytes / (1024 * 1024), 2),
                "updated_at": info.updated_at.isoformat(),
            })

        if not indexes:
            return [TextContent(type="text", text="No indexes available. Add repositories using pythia_add_repo.")]

        result = "Available Pythia Indexes:\n\n"
        for idx in indexes:
            result += f"- **{idx['name']}**\n"
            result += f"  Provider: {idx['provider']}\n"
            result += f"  Files: {idx['file_count']}, Size: {idx['size_mb']} MB\n"
            result += f"  Updated: {idx['updated_at']}\n\n"

        return [TextContent(type="text", text=result)]

    async def _handle_search(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle pythia_search tool call."""
        query = arguments["query"]
        indexes = arguments.get("indexes")
        top_k = arguments.get("top_k", 10)

        results = await self.indexer.search(query, index_names=indexes, top_k=top_k)

        if not results:
            return [TextContent(type="text", text=f"No results found for: {query}")]

        output = f"Search results for: **{query}**\n\n"
        for i, result in enumerate(results, 1):
            output += f"### Result {i}\n"
            output += f"**Repository:** {result.get('repository', 'unknown')}\n"
            if "file" in result.get("metadata", {}):
                output += f"**File:** {result['metadata']['file']}\n"
            output += f"**Score:** {result.get('score', 0):.4f}\n"
            if "text" in result:
                text = result["text"][:500] + "..." if len(result.get("text", "")) > 500 else result.get("text", "")
                output += f"```\n{text}\n```\n\n"

        return [TextContent(type="text", text=output)]

    async def _handle_add_repo(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle pythia_add_repo tool call."""
        url = arguments["url"]

        try:
            provider, owner, name, base_url = parse_repo_url(url)
        except ValueError as e:
            return [TextContent(type="text", text=f"Invalid URL: {e}")]

        if provider not in self.indexer.providers:
            return [TextContent(
                type="text",
                text=f"Provider '{provider}' is not configured. Available providers: {list(self.indexer.providers.keys())}",
            )]

        git_provider = self.indexer.providers[provider]

        try:
            repo = await git_provider.get_repository(owner, name)
            state = await self.indexer.add_repository(repo)

            return [TextContent(
                type="text",
                text=f"Successfully added and indexed {repo.full_name}\n"
                     f"Commit: {state.last_commit}\n"
                     f"Path: {state.local_path}",
            )]
        except Exception as e:
            return [TextContent(type="text", text=f"Failed to add repository: {e}")]

    async def _handle_sync(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle pythia_sync tool call."""
        index_name = arguments["index_name"]
        info = await self.indexer.get_index(index_name)

        if not info:
            return [TextContent(type="text", text=f"Index not found: {index_name}")]

        try:
            state = await self.indexer.sync_repository(info.repository)
            return [TextContent(
                type="text",
                text=f"Synced {info.repository.full_name}\n"
                     f"Commit: {state.last_commit}\n"
                     f"Indexed: {state.indexed}",
            )]
        except Exception as e:
            return [TextContent(type="text", text=f"Sync failed: {e}")]

    async def _handle_status(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle pythia_status tool call."""
        index_name = arguments.get("index_name")

        if index_name:
            info = await self.indexer.get_index(index_name)
            if not info:
                return [TextContent(type="text", text=f"Index not found: {index_name}")]

            state = await self.indexer.repository_manager.get_state(info.repository)
            status = {
                "name": info.name,
                "repository": info.repository.full_name,
                "file_count": info.file_count,
                "size_bytes": info.size_bytes,
                "updated_at": info.updated_at.isoformat(),
                "last_commit": state.last_commit if state else None,
                "indexed": state.indexed if state else False,
                "error": state.error if state else None,
            }
            return [TextContent(type="text", text=json.dumps(status, indent=2))]

        statuses = []
        async for info in self.indexer.list_indexes():
            state = await self.indexer.repository_manager.get_state(info.repository)
            statuses.append({
                "name": info.name,
                "indexed": state.indexed if state else False,
                "error": state.error if state else None,
            })

        return [TextContent(type="text", text=json.dumps(statuses, indent=2))]

    async def run(self) -> None:
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


async def main() -> None:
    """Entry point for the MCP server."""
    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    indexer = IndexerService(settings)

    try:
        await indexer.start_sync_loop()
        server = PythiaMCPServer(indexer)
        await server.run()
    finally:
        await indexer.close()


if __name__ == "__main__":
    asyncio.run(main())
