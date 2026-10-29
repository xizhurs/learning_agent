"""Minimum-interval request throttle for rate-limited hosted APIs (shared by all workers)."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable


class Throttle:
    """Spaces calls at least `min_interval` seconds apart. Thread-safe; also usable from asyncio."""

    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.min_interval = min_interval
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()
        self._next = 0.0

    def _reserve(self) -> float:
        """Claim the next free slot and return how long the caller must wait for it."""
        with self._lock:
            now = self._clock()
            slot = max(now, self._next)
            self._next = slot + self.min_interval
            return slot - now

    def wait(self) -> None:
        delay = self._reserve()
        if delay > 0:
            self._sleep(delay)

    async def await_slot(self) -> None:
        delay = self._reserve()
        if delay > 0:
            await asyncio.sleep(delay)


def http_clients(throttle: Throttle):
    """(sync, async) httpx clients whose every request (including SDK retries) waits for a throttle slot."""
    import httpx

    def before(_request):
        throttle.wait()

    async def abefore(_request):
        await throttle.await_slot()

    return (
        httpx.Client(event_hooks={"request": [before]}, timeout=180),
        httpx.AsyncClient(event_hooks={"request": [abefore]}, timeout=180),
    )
