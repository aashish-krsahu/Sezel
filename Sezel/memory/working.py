"""Working memory: in-RAM ring buffer for recent conversational history."""

from ..core.type import Turn


class WorkingMemory:
    """Ring buffer for the current conversation context."""

    def __init__(self, size: int = 20):
        self._buffer: list[Turn] = []
        self._max = size

    def append(self, turn: Turn) -> None:
        """Add a turn to the buffer, dropping oldest if full."""
        self._buffer.append(turn)
        if len(self._buffer) > self._max:
            self._buffer.pop(0)

    def recent(self, n: int = 10) -> list[Turn]:
        """Return the last n turns."""
        return self._buffer[-n:]

    def all(self) -> list[Turn]:
        """Return all turns in the buffer."""
        return self._buffer.copy()

    def clear(self) -> None:
        """Clear all turns from the buffer."""
        self._buffer.clear()

    def size(self) -> int:
        """Return the number of turns in the buffer."""
        return len(self._buffer)

    def is_empty(self) -> bool:
        """Check if the buffer is empty."""
        return len(self._buffer) == 0

