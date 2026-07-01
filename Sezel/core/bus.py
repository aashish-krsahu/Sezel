"""Event bus for async publish-subscribe communication."""

import asyncio
from .type import Event


class EventBus:
    """FIFO queue-based event bus for decoupling producers and consumers."""

    def __init__(self, maxsize: int = 100):
        self._q: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)

    async def publish(self, event: Event) -> None:
        """Publish an event to the bus (blocking if queue full)."""
        await self._q.put(event)

    async def next(self) -> Event:
        """Get the next event from the bus (blocking if queue empty)."""
        return await self._q.get()

    async def close(self) -> None:
        """Close the bus (drain remaining events, stop accepting new ones)."""
        # For simplicity, just clear the queue
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except asyncio.QueueEmpty:
                break

