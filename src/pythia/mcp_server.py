"""Entry point for the Pythia MCP server."""

import asyncio

from pythia.mcp.server import main

if __name__ == "__main__":
    asyncio.run(main())
