"""Tracks which large model is currently resident and ensures
only one is loaded at a time"""

from __future__ import annotations

from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

@dataclass
class VRAMbudget:
    total_gb: float = 6.0
    reserved_gb: float = 0.5
    ollama_overhead: float = 0.5

    @property
    def available_gb(self) -> float:
        return self.total_gb - self.reserved_gb - self.ollama_overhead

@dataclass
class ModelSlot:
    name: str
    vram_gb: float
    loaded: bool = False


class ModelManager:

    def __int__(self, vram_budget: VRAMbudget | None = None) -> None:
        self.budget = vram_budget or VRAMbudget()
        self._active: str | None = None
        self._slots: dict[str, ModelSlot] = {}

    def register(self, name:str, vram_gb:float):

        self._slots[name] = ModelSlot(name, vram_gb)

    @property
    def active(self) -> str | None:
        return self._active

    @property
    def can_swap_to(self, name:str) -> bool:
        """Check if the model can fit in VRAM (possibly after unloading another)."""
        needed = self._slots.get(name, ModelSlot(name, 0)).vram_gb
        available = self.budget.available_gb

        if self._active and self.active in self._slots:
            available += self._slots[self.active].vram_gb

        return needed <= available

    async def acquire(self, name: str) -> bool:
        """
        Mark a model as active. If another model is active, release it first.

        Returns True if the model was acquired, False if it wouldn't fit.
        """

        if not self.can_swap_to(name):
            return False

        if self._active:
            await self.release(self._active)

        slot = self._slots.get(name)
        if slot:
            slot.loaded = True

        self._active = name
        return True

    async def release(self, name:str) -> None:
        """Mark a model as released."""
        if name in self._slots:
            self._slots[name].loaded = False
        if self._active == name:
            self._active = None

    @asynccontextmanager
    async def using(self, name:str, vram_gb:float | None = None) -> AsyncIterator[bool]:
        """
        Context manager: acquire model on entry, release on exit.

        Yields:
            True if the model was acquired, False if it wouldn't fit.
        """

        if vram_gb is not None and name not in self._slots:
            self.register(name, vram_gb)

        acquired = await self.acquire(name)
        try:
            yield acquired
        finally:
            await self.release(name)