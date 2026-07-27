"""Core data types for the Sezel system."""

from dataclasses import dataclass, field
from typing import Any
import time
from enum import Enum


@dataclass
class Event:
    """An event published to the bus."""
    kind: str
    payload: dict[str, Any]
    ts: float = field(default_factory=time.time)


@dataclass
class Perception:
    """Raw sensory input parsed from an Event."""
    text: str
    user_affect: "Affect | None" = None
    visual_context: "VisualContext | None" = None
    embeddings: list[float] | None = None


@dataclass
class Affect:
    """Emotional state (PAD model: Pleasure, Arousal, Dominance)."""
    valence: float = 0.0      # pleasure
    arousal: float = 0.0      # excitement
    dominance: float = 0.0    # control

    def as_prompt(self) -> str:
        """Return affect as a string for LLM context."""
        moods = []
        if self.valence > 0.3:
            moods.append("positive")
        elif self.valence < -0.3:
            moods.append("negative")
        if self.arousal > 0.3:
            moods.append("energetic")
        elif self.arousal < -0.3:
            moods.append("calm")
        return ", ".join(moods) if moods else "neutral"


@dataclass
class Turn:
    """A single conversational turn."""
    role: str              # "user" | "assistant"
    content: str
    ts: float = field(default_factory=time.time)


@dataclass
class Context:
    """The full context fed into the LLM."""
    working: list[Turn]
    retrieved: list[Turn]
    mood: Affect
    perception: Perception
    token_estimate: int = 0


@dataclass
class Plan:
    """The LLM's response plan."""
    text: str
    tool_calls: list[dict] = field(default_factory=list)

@dataclass
class MemoryHit:
    """Attributes:
        text: The actual memory content (e.g. "User's name is Alice")
        score: How relevant this is (0.0 = not relevant, 1.0 = perfect match)
        meta: Extra info like when it was saved, what category, etc.
    """
    text: str
    score: float
    meta: dict[str, Any] = field(default_factory=dict)

@dataclass
class UIElement:
    """A single UI element from the accessibility tree."""
    role: str
    name: str
    bbox: tuple[int, int, int, int]
    clickable: bool = False

@dataclass
class VisualContext:
    """Structured representation of what's on screen."""
    text_on_screen: str = ""
    elements: list[UIElement] = field(default_factory=list)
    caption: str | None = None
    image_ref: str | None = None

class Route:

    LOCAL = "local"
    CLOUD = "cloud"
    TOOL_ONLY = "tool_only"
    VISION_LOCAL = "vision_local"
    VISION_CLOUD = "vision_cloud"