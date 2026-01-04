"""OpenAI SDK-based agent for answering questions about codebases."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator

from openai import AsyncOpenAI

from pythia.config.settings import Settings, AgentConfig
from pythia.indexer.service import IndexerService

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are Pythia, an intelligent codebase assistant. You help developers understand and navigate codebases by answering questions about code structure, implementation details, and best practices.

You have access to semantic search across indexed repositories. When answering questions:
1. Use the search results to provide accurate, context-aware answers
2. Reference specific files and code snippets when relevant
3. Explain code concepts clearly and concisely
4. Suggest related areas of the codebase when appropriate

If the search results don't contain enough information to answer the question, say so and suggest how the user might find the information they need."""


@dataclass
class Message:
    """A message in the conversation."""

    role: str
    content: str


@dataclass
class SearchContext:
    """Context from a semantic search."""

    query: str
    results: list[dict]


class PythiaAgent:
    """OpenAI SDK-based agent for codebase Q&A."""

    def __init__(
        self,
        indexer: IndexerService,
        config: AgentConfig | None = None,
        api_key: str | None = None,
    ):
        self.indexer = indexer
        self.config = config or AgentConfig()
        self.client = AsyncOpenAI(api_key=api_key)
        self.conversation: list[Message] = []
        self._system_prompt = self.config.system_prompt or DEFAULT_SYSTEM_PROMPT

    def _build_messages(
        self,
        user_message: str,
        context: SearchContext | None = None,
    ) -> list[dict]:
        """Build the messages list for the API call."""
        messages = [{"role": "system", "content": self._system_prompt}]

        for msg in self.conversation[-10:]:
            messages.append({"role": msg.role, "content": msg.content})

        if context and context.results:
            context_text = self._format_search_context(context)
            messages.append({
                "role": "system",
                "content": f"Relevant code context from semantic search:\n\n{context_text}",
            })

        messages.append({"role": "user", "content": user_message})
        return messages

    def _format_search_context(self, context: SearchContext) -> str:
        """Format search results as context for the LLM."""
        parts = []
        for i, result in enumerate(context.results[:5], 1):
            repo = result.get("repository", "unknown")
            file = result.get("metadata", {}).get("file", "unknown")
            text = result.get("text", "")[:1000]
            score = result.get("score", 0)

            parts.append(
                f"### Result {i} (score: {score:.3f})\n"
                f"Repository: {repo}\n"
                f"File: {file}\n"
                f"```\n{text}\n```"
            )

        return "\n\n".join(parts)

    async def search_context(self, query: str, top_k: int = 5) -> SearchContext:
        """Search for relevant code context."""
        results = await self.indexer.search(query, top_k=top_k)
        return SearchContext(query=query, results=results)

    async def ask(
        self,
        question: str,
        search_first: bool = True,
        indexes: list[str] | None = None,
    ) -> str:
        """Ask a question about the codebase."""
        context = None
        if search_first:
            context = await self.search_context(question)

        messages = self._build_messages(question, context)

        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        answer = response.choices[0].message.content or ""

        self.conversation.append(Message(role="user", content=question))
        self.conversation.append(Message(role="assistant", content=answer))

        return answer

    async def ask_stream(
        self,
        question: str,
        search_first: bool = True,
        indexes: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Ask a question and stream the response."""
        context = None
        if search_first:
            context = await self.search_context(question)

        messages = self._build_messages(question, context)

        stream = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )

        full_response = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content

        self.conversation.append(Message(role="user", content=question))
        self.conversation.append(Message(role="assistant", content=full_response))

    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.conversation.clear()

    async def list_indexes(self) -> list[str]:
        """List available indexes."""
        indexes = []
        async for info in self.indexer.list_indexes():
            indexes.append(info.name)
        return indexes


async def main() -> None:
    """Entry point for the interactive agent."""
    import sys

    from rich.console import Console
    from rich.markdown import Markdown
    from rich.prompt import Prompt

    logging.basicConfig(level=logging.INFO)
    console = Console()
    settings = Settings.from_env()

    if not settings.openai_api_key:
        console.print("[red]Error: PYTHIA_OPENAI_API_KEY or OPENAI_API_KEY not set[/red]")
        sys.exit(1)

    indexer = IndexerService(settings)
    agent = PythiaAgent(indexer, settings.agent, settings.openai_api_key)

    console.print("[bold green]Pythia Codebase Assistant[/bold green]")
    console.print("Ask questions about your indexed codebases. Type 'quit' to exit.\n")

    indexes = await agent.list_indexes()
    if indexes:
        console.print(f"[dim]Available indexes: {', '.join(indexes)}[/dim]\n")
    else:
        console.print("[yellow]No indexes available. Add repositories first.[/yellow]\n")

    try:
        while True:
            question = Prompt.ask("\n[bold cyan]You[/bold cyan]")

            if question.lower() in ("quit", "exit", "q"):
                break

            if question.lower() == "clear":
                agent.clear_conversation()
                console.print("[dim]Conversation cleared.[/dim]")
                continue

            console.print("\n[bold magenta]Pythia[/bold magenta]")

            response = ""
            async for chunk in agent.ask_stream(question):
                response += chunk

            console.print(Markdown(response))

    except KeyboardInterrupt:
        pass
    finally:
        await indexer.close()
        console.print("\n[dim]Goodbye![/dim]")


if __name__ == "__main__":
    asyncio.run(main())
