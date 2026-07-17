import json
import math
import time
from pathlib import Path
from ..core.type import Affect

class AffectiveState:
    """
        The state drifts toward baseline on every decay_tick() call.

        Usage:
            mood = AffectiveState(decay_per_sec=0.95)
            mood.apply(Affect(valence=-0.5, arousal=0.3))  # user said something upsetting
            current = mood.current()                         # returns the current Affect
            mood.decay_tick()                                # drift toward baseline
            mood.persist()                                   # save to disk
    """

    def __init__(
        self,
        baseline:Affect,
        decay_per_sec:float=0.95,
        persist_path: str | Path = "sezel_mood.json"
    ):
        self.baseline = baseline
        self.decay_per_sec = decay_per_sec
        self.persist_path = persist_path

        self._valence = self.baseline.valence
        self._arousal = self.baseline.arousal
        self._dominance = self.baseline.dominance
        self._last_decay_ts: float = time.time()

        self._load()

    def current(self) -> Affect:
        """Return the current affective state as an Affect dataclass."""
        return Affect(
            valence=self._valence,
            arousal=self._arousal,
            dominance=self._dominance,
        )

    def apply(self, delta) -> None:
        """
        Apply an affect delta, then clamp to [-1.0, 1.0].
        Positive valence delta = good news; negative = bad news.
        :param delta: represents the emotional impact of the latest event.
        """
        self._valence = AffectiveState._clamp(self._valence + delta.valence)
        self._arousal = AffectiveState._clamp(self._arousal + delta.arousal)
        self._dominance = AffectiveState._clamp(self._dominance + delta.dominance)

    def decay_tick(self) -> None:
        """
        Pull the current state toward baseline.

        Called on a timer (e.g. every 30 seconds of inactivity).
        Uses the formula: new = baseline + (current - baseline) * decay

        This means:
        - decay=1.0 → stays at current forever (no decay)
        - decay=0.0 → snaps to baseline instantly
        - decay=0.95 → drifts 5% toward baseline per tick
        """

        now = time.time()
        elapsed = now - self._last_decay_ts
        self._last_decay_ts = now

        if elapsed > 0:
            factor = self.decay_per_sec ** elapsed
            self._valence = AffectiveState._clamp(self.baseline.valence + (self._valence - self.baseline.valence) * factor)
            self._dominance = AffectiveState._clamp(self.baseline.dominance + (self._dominance - self.baseline.dominance) * factor)
            self._arousal = AffectiveState._clamp(self.baseline.arousal + (self._arousal - self.baseline.arousal) * factor)

    def persist(self) -> None:
        """Save the current affective state to a JSON file to remember mood after restart."""

        data = {
            "valence": self._valence,
            "arousal": self._arousal,
            "dominance": self._dominance,
            "baseline_valence": self.baseline.valence,
            "baseline_arousal": self.baseline.arousal,
            "baseline_dominance": self.baseline.dominance,
            "timestamp": time.time()
        }
        try:
            self.persist_path.write_text(json.dumps(data, indent=2))
        except OSError as e:
            print(f"  [AffectiveState: could not persist mood — {e}]")

    def _load(self) -> None:
        """
            Load previously persisted state from disk.

            Only loads if the file exists and is valid JSON.
            If the file is corrupt, silently starts fresh at baseline.
        """

        if not Path(self.persist_path).exists():
            return
        try:
            data = json.loads(self.persist_path.read_text())
            self._valence = AffectiveState._clamp(data.get("valence", self.baseline.valence))
            self._arousal = AffectiveState._clamp(data.get("arousal", self.baseline.arousal))
            self._dominance = AffectiveState._clamp(data.get("dominance", self.baseline.dominance))

            self.baseline = Affect(
                valence= AffectiveState._clamp(data.get("baseline_valence", self.baseline.valence)),
                arousal= AffectiveState._clamp(data.get("baseline_arousal", self.baseline.arousal)),
                dominance= AffectiveState._clamp(data.get("baseline_dominance", self.baseline.dominance))
            )

        except (json.JSONDecodeError, KeyError, OSError):
            self._valence = self.baseline.valence
            self._arousal = self.baseline.arousal
            self._dominance = self.baseline.dominance

    @staticmethod
    def _clamp(value: float) -> float:
        hi: float = 1.0
        lo: float = -1.0
        return max(min(value,hi), lo)