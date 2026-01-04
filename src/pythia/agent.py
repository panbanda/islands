"""Entry point for the Pythia agent."""

import asyncio

from pythia.agent.assistant import main

if __name__ == "__main__":
    asyncio.run(main())
