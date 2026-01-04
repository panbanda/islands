"""Command-line interface for Pythia."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from pythia.config.settings import Settings, ProviderConfig, ProviderType


console = Console()


def setup_logging(debug: bool) -> None:
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--config", type=click.Path(exists=True), help="Path to config file")
@click.pass_context
def cli(ctx: click.Context, debug: bool, config: str | None) -> None:
    """Pythia - Codebase Indexing and Inquiry System."""
    setup_logging(debug)
    ctx.ensure_object(dict)

    if config:
        ctx.obj["settings"] = Settings.from_file(Path(config))
    else:
        ctx.obj["settings"] = Settings.from_env()


@cli.command()
@click.argument("provider", type=click.Choice(["github", "gitlab", "bitbucket", "gitea"]))
@click.argument("owner")
@click.argument("name")
@click.option("--token", envvar="PYTHIA_GIT_TOKEN", help="Git provider token")
@click.pass_context
def add(ctx: click.Context, provider: str, owner: str, name: str, token: str | None) -> None:
    """Add and index a repository."""
    from pythia.indexer.service import IndexerService
    from pythia.config.settings import ProviderConfig, ProviderType

    settings: Settings = ctx.obj["settings"]

    if token:
        provider_config = ProviderConfig(
            type=ProviderType(provider),
            token=token,
        )
        settings.providers.append(provider_config)

    async def _add():
        indexer = IndexerService(settings)
        try:
            git_provider = indexer.providers.get(provider)
            if not git_provider:
                console.print(f"[red]Provider '{provider}' not configured[/red]")
                return

            console.print(f"Fetching repository info for {owner}/{name}...")
            repo = await git_provider.get_repository(owner, name)

            console.print(f"Cloning and indexing {repo.full_name}...")
            state = await indexer.add_repository(repo)

            console.print(f"[green]Successfully indexed {repo.full_name}[/green]")
            console.print(f"  Commit: {state.last_commit}")
            console.print(f"  Path: {state.local_path}")
        finally:
            await indexer.close()

    asyncio.run(_add())


@cli.command()
@click.argument("query")
@click.option("--index", "-i", multiple=True, help="Specific indexes to search")
@click.option("--top-k", "-k", default=10, help="Number of results")
@click.pass_context
def search(ctx: click.Context, query: str, index: tuple, top_k: int) -> None:
    """Search across indexed codebases."""
    from pythia.indexer.service import IndexerService

    settings: Settings = ctx.obj["settings"]

    async def _search():
        indexer = IndexerService(settings)
        try:
            indexes = list(index) if index else None
            results = await indexer.search(query, index_names=indexes, top_k=top_k)

            if not results:
                console.print(f"[yellow]No results found for: {query}[/yellow]")
                return

            table = Table(title=f"Search Results: {query}")
            table.add_column("Score", style="cyan", width=8)
            table.add_column("Repository", style="green")
            table.add_column("File", style="blue")
            table.add_column("Preview", max_width=60)

            for result in results:
                score = f"{result.get('score', 0):.3f}"
                repo = result.get("repository", "unknown")
                file = result.get("metadata", {}).get("file", "unknown")
                text = result.get("text", "")[:100].replace("\n", " ")

                table.add_row(score, repo, file, text + "...")

            console.print(table)
        finally:
            await indexer.close()

    asyncio.run(_search())


@cli.command(name="list")
@click.pass_context
def list_indexes(ctx: click.Context) -> None:
    """List all indexed repositories."""
    from pythia.indexer.service import IndexerService

    settings: Settings = ctx.obj["settings"]

    async def _list():
        indexer = IndexerService(settings)
        try:
            table = Table(title="Pythia Indexes")
            table.add_column("Name", style="cyan")
            table.add_column("Provider", style="green")
            table.add_column("Files", justify="right")
            table.add_column("Size", justify="right")
            table.add_column("Updated", style="dim")

            count = 0
            async for info in indexer.list_indexes():
                size_mb = f"{info.size_bytes / (1024 * 1024):.2f} MB"
                table.add_row(
                    info.name,
                    info.repository.provider,
                    str(info.file_count),
                    size_mb,
                    info.updated_at.strftime("%Y-%m-%d %H:%M"),
                )
                count += 1

            if count == 0:
                console.print("[yellow]No indexes found. Use 'pythia add' to add repositories.[/yellow]")
            else:
                console.print(table)
        finally:
            await indexer.close()

    asyncio.run(_list())


@cli.command()
@click.argument("index_name")
@click.pass_context
def sync(ctx: click.Context, index_name: str) -> None:
    """Sync and re-index a repository."""
    from pythia.indexer.service import IndexerService

    settings: Settings = ctx.obj["settings"]

    async def _sync():
        indexer = IndexerService(settings)
        try:
            info = await indexer.get_index(index_name)
            if not info:
                console.print(f"[red]Index not found: {index_name}[/red]")
                return

            console.print(f"Syncing {index_name}...")
            state = await indexer.sync_repository(info.repository)

            console.print(f"[green]Synced {index_name}[/green]")
            console.print(f"  Commit: {state.last_commit}")
            console.print(f"  Indexed: {state.indexed}")
        finally:
            await indexer.close()

    asyncio.run(_sync())


@cli.command()
@click.pass_context
def serve(ctx: click.Context) -> None:
    """Start the MCP server."""
    from pythia.mcp.server import main as mcp_main

    console.print("[green]Starting Pythia MCP server...[/green]")
    asyncio.run(mcp_main())


@cli.command()
@click.pass_context
def ask(ctx: click.Context) -> None:
    """Start interactive Q&A session."""
    from pythia.agent.assistant import main as agent_main

    asyncio.run(agent_main())


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show system status and statistics."""
    from pythia.indexer.service import IndexerService
    from pythia.storage.efs import EFSStorage

    settings: Settings = ctx.obj["settings"]

    async def _status():
        indexer = IndexerService(settings)
        efs = EFSStorage(settings.storage)

        try:
            stats = efs.get_storage_stats()

            console.print("[bold]Pythia Status[/bold]\n")

            table = Table(show_header=False)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Total Indexes", str(stats["total_indexes"]))
            table.add_row("Total Files", str(stats["total_files"]))
            table.add_row("Total Size", f"{stats['total_size_mb']} MB")
            table.add_row("Repos Path", stats["repos_path"])
            table.add_row("Indexes Path", stats["indexes_path"])
            table.add_row("Providers", ", ".join(indexer.providers.keys()) or "None")

            console.print(table)
        finally:
            await indexer.close()

    asyncio.run(_status())


def main() -> None:
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
