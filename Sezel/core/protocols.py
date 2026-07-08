"""Protocol definitions for pluggable components."""

from typing import Protocol, Awaitable
from .type import Context, Plan


class LLM(Protocol):
    """Interface for language models."""

    async def complete(
        self,
        ctx: Context,
        *,
        tools: list[dict] | None = None
    ) -> Plan:
        """Generate a response given a context."""
        ...

    async def score_complexity(self, ctx: Context) -> float:
        """Score the complexity of a context (0.0 to 1.0)."""
        ...


class Embedder(Protocol):
    """Interface for embedder models."""

    async def embed(self, text: str) -> list[float]:
        """Embed text into a vector."""
        ...


class MemoryStore(Protocol):
    """Interface for memory storage backends."""

    async def write(self, data: dict) -> None:
        """Write data to the store."""
        ...

    async def retrieve(self, query: str, limit: int = 5) -> list[dict]:
        """Retrieve data from the store."""
        ...

