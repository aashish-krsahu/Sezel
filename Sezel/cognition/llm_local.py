"""Local LLM via Ollama HTTP API."""

import httpx
from ..core.type import Context, Plan


class OllamaLLM:
    """LLM interface using Ollama local API."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = "http://localhost:11434"
    ):
        self.model = model
        self.host = host
        self.client = httpx.AsyncClient(base_url=host, timeout=300.0)

    async def complete(
        self,
        ctx: Context,
        *,
        tools: list[dict] | None = None
    ) -> Plan:
        """Generate a response given a context."""
        messages = self._render(ctx)
        text = await self._chat(messages)
        return Plan(text=text, tool_calls=[])

    async def score_complexity(self, ctx: Context) -> float:
        """Score complexity (Phase 1: stub, always 0.5)."""
        return 0.5

    def _render(self, ctx: Context) -> list[dict[str, str]]:
        """Convert Context into Ollama chat messages."""
        system_prompt = (
            "You are Sezel, a helpful AI assistant. "
            f"Current mood: {ctx.mood.as_prompt()}\n"
            "Be concise and natural."
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add working memory history
        for turn in ctx.working:
            messages.append({
                "role": turn.role,
                "content": turn.content
            })
        
        # Add current perception as the final user message
        if ctx.perception.text:
            messages.append({
                "role": "user",
                "content": ctx.perception.text
            })
        
        return messages

    async def _chat(self, messages: list[dict[str, str]]) -> str:
        """Call Ollama API and return the assistant's response."""
        try:
            resp = await self.client.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_ctx": 4096,
                    }
                }
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except httpx.RequestError as e:
            raise RuntimeError(f"Ollama API error: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to parse Ollama response: {e}") from e

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

