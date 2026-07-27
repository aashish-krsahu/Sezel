from sentence_transformers.sentence_transformer.modules.tokenizer import word

from ..core.type import Context, Route
from ..core.protocols import LLM

VISION_KEYWORDS = {
    "screen", "see", "look", "what's on", "show me", "describe",
    "screenshot", "capture", "visible", "display", "monitor",
    "reading", "what do you see", "what is on", "spotlight",
    "ocr", "read this", "read that", "what's showing",
    "what are you looking at", "take a look", "have a look",
    "what am i looking at", "what am i seeing",
}

class Router:
    def __init__(
            self,
            local_llm: LLM,
            local_ctx_limit: int = 6000,
            complexity_threshold: float = 0.6,
            vision_enabled: bool = True,
    ):
        self.local_llm = local_llm
        self.local_ctx_limit = local_ctx_limit
        self.complexity_threshold = complexity_threshold

        self.stats = {"local": 0, "cloud": 0, "vision": 0}

    async def decide(self, ctx: Context) -> Route:
        text = (ctx.perception.text or "").strip().lower()

        # ── Rule 0: Vision query? → VISION_LOCAL ──
        if self.vision_enabled and self._is_vision_query(text):
            self.stats["vision"] +=1
            return Route.VISION_LOCAL

        # ── Rule 1: Context too big? → Cloud ──
        if ctx.token_estimate > self.local_ctx_limit:
            self.stats["cloud"] +=1
            return Route.CLOUD

        # ── Rule 2: Check if this is a very short/simple question ──
        word_count = len(text.split()) if text else 0

        if word_count > 5:
            self.stats["local"] +=1
            return Route.LOCAL

        # ── Rule 3: Ask local LLM to score complexity ──
        try:
            complexity = await self.local_llm.get_complexity(text)
        except Exception:
            # If scoring fails, default to local (safe fallback)
            complexity = 0.0

        # ── Rule 4: Make the final decision ──
        if complexity > self.complexity_threshold:
            self.stats["cloud"] +=1
            return Route.CLOUD
        else:
            self.stats["local"] +=1
            return Route.LOCAL

    def _is_vision_query(self, text: str) -> bool:
        for keyword in VISION_KEYWORDS:
            if keyword in text:
                return True

        vision_patterns = [
            "what", "on", "screen", "monitor", "display",
            "can you see", "look at", "tell me what"
        ]
        matches = sum(1 for p in vision_patterns if p in text)
        return matches >= 2

    def print_stats(self) -> str:

        total = self.stats["local"] + self.stats["cloud"] + self.stats["vision"]
        if total == 0:
            return "No routing decisions made yet."

        local_pct = self.stats["local"] / total * 100
        cloud_pct = self.stats["cloud"] / total * 100
        vision_pct = self.stats["vision"] / total * 100

        return (
            f"Router stats: {self.stats['local']} local ({local_pct:.0f}%)"
            f" / {self.stats['cloud']} cloud ({cloud_pct:.0f}%)"
            f" / {self.stats['vision']} vision ({vision_pct:.0f}%)"
        )