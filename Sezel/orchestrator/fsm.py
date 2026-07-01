"""FSM for orchestrator state management."""

from enum import Enum, auto


class State(Enum):
    """FSM states for the orchestrator."""
    IDLE = auto()
    PERCEIVING = auto()
    ASSEMBLING = auto()
    REASONING = auto()
    RESPONDING = auto()
    CONSOLIDATING = auto()


class FSM:
    """Finite State Machine for orchestrator."""

    def __init__(self):
        self.state = State.IDLE
        self._history: list[State] = []

    def to(self, new: State) -> None:
        """Transition to a new state with optional guard checks."""
        # Phase 1: permissive guards — future phases will enforce strict transitions
        prev = self.state
        self._history.append(prev)
        self.state = new
        # Optional: log transitions
        # print(f"FSM: {prev.name} → {new.name}")

    def can_transition(self, target: State) -> bool:
        """Check if a transition is valid (Phase 1: always True)."""
        return True

    def last_state(self) -> State | None:
        """Get the previous state."""
        return self._history[-1] if self._history else None

    def history(self) -> list[State]:
        """Get the complete transition history."""
        return self._history.copy()

    def reset(self) -> None:
        """Reset FSM to IDLE."""
        self.state = State.IDLE
        self._history.clear()

