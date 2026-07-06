from sentence_transformers.sentence_transformer.modules.tokenizer import word

from ..core.type import Context, Route
from ..core.protocols import LLM

class Router:
    def __init__(
            self,
            local_llm: LLM,
            local_ctx_limit: int = 6000,
            complexity_threshold: float = 0.6
    ):
        self.local_llm = local_llm
        self.local_ctx_limit = local_ctx_limit
        self.complexity_threshold = complexity_threshold

        self.stats = {"local": 0, "cloud": 0}

    async def decide(self, ctx: Context) -> Route:

        # ── Rule 1: Context too big? → Cloud ──
        if ctx.token_estimate > self.local_ctx_limit:
            self.stats["cloud"] +=1
            return Route.CLOUD

        # ── Rule 2: Check if this is a very short/simple question ──
        text = (ctx.perception.text or "").strip()
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

    def print_stats(self) -> str:

        total = self.stats["local"] + self.stats["cloud"]
        if total == 0:
            return "No Routing decision made yet."

        local_pct = self.stats["local"] / total * 100
        cloud_pct = self.stats["cloud"] / total * 100

        return (
            f"Router stats: {self.stats['local']} local ({local_pct:.0f}%)"
            f"/ {self.stats['cloud']} cloud ({cloud_pct: .0f}%)"
        )