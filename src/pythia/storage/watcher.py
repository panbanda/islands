"""File system watcher for detecting repository changes."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, Awaitable

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class IndexWatcher:
    """Watch for changes in indexed repositories and trigger re-indexing."""

    def __init__(
        self,
        watch_path: Path,
        on_change: Callable[[Path], Awaitable[None]],
        debounce_seconds: float = 5.0,
    ):
        self.watch_path = watch_path
        self.on_change = on_change
        self.debounce_seconds = debounce_seconds
        self._observer: Observer | None = None
        self._pending: dict[str, asyncio.Task] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_repo_path(self, path: Path) -> Path | None:
        """Extract the repository root path from a changed file path."""
        try:
            relative = path.relative_to(self.watch_path)
            parts = relative.parts
            if len(parts) >= 3:
                return self.watch_path / parts[0] / parts[1] / parts[2]
        except ValueError:
            pass
        return None

    def _schedule_callback(self, repo_path: Path) -> None:
        """Schedule a debounced callback for a repository change."""
        if self._loop is None:
            return

        key = str(repo_path)

        if key in self._pending:
            self._pending[key].cancel()

        async def delayed_callback():
            await asyncio.sleep(self.debounce_seconds)
            try:
                await self.on_change(repo_path)
            except Exception as e:
                logger.error(f"Change callback failed for {repo_path}: {e}")
            finally:
                self._pending.pop(key, None)

        self._pending[key] = self._loop.create_task(delayed_callback())

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Start watching for file changes."""
        self._loop = loop or asyncio.get_event_loop()

        class Handler(FileSystemEventHandler):
            def __init__(handler_self):
                super().__init__()

            def on_any_event(handler_self, event: FileSystemEvent):
                if event.is_directory:
                    return
                if ".git" in str(event.src_path):
                    return

                path = Path(event.src_path)
                repo_path = self._get_repo_path(path)
                if repo_path:
                    self._loop.call_soon_threadsafe(
                        lambda: self._schedule_callback(repo_path)
                    )

        self._observer = Observer()
        self._observer.schedule(Handler(), str(self.watch_path), recursive=True)
        self._observer.start()
        logger.info(f"Started watching: {self.watch_path}")

    def stop(self) -> None:
        """Stop watching for file changes."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

        for task in self._pending.values():
            task.cancel()
        self._pending.clear()

        logger.info("Stopped file watcher")
