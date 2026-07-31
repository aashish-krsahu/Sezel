import os

from langchain_core.messages import content
from scipy._lib.pyprima.common import message
from sqlalchemy.ext.asyncio import result

from core.type import Context,Plan

class ClaudeLLM:
    def __init__(
            self,
            model: str = "claude-sonnet-4-6",
            api_key: str | None = None,
            max_tokens: int = 1024
    ):
        self.model = model
        self.max_tokens = max_tokens

        resolved_key = api_key or os.environ.get("ANTROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Antropic Key Found "
                "Set the ANTHROPIC_API_KEY environment variable "
                "or pass api_key directly."
            )
        self.api_key = resolved_key
        self._client = self._create_client()

    def _create_client(self):
        from anthropic import AsyncAnthropic
        return AsyncAnthropic(api_key=self.api_key)

    async def complete(
            self,
            ctx: Context,
            *,
            tools: list[dict] | None = None
    ) -> Plan:

        system_prompt, messages = self._render(ctx)

        kwangs = {
            "model": self.model,
            "system": system_prompt,
            "messages": messages,
            "max_tokens": self.max_tokens
        }

        try:
            response = await self._client.messages.create(**kwangs)

            text = ""
            for block in response.blocks:
                if block.type == "text":
                    text += block.text

            return Plan(text=text, tool_calls=[])

        except Exception as e:
            raise RuntimeError(f"Claude API error: {e}") from e

    async def score_complexity(
            self,
            ctx: Context,
    ) -> float:

        return 1.0

    def _render(self, ctx: Context) -> tuple[str, list[dict]]:

        system_prompt = (
            "You are Sezel, a helpful AI assistant running in the cloud with screen capture capabilities. "
            "You handle the complex questions that the local model can't.\n"
            f"Current mood: {ctx.mood.as_prompt()}\n"
            "Be natural, helpful, and concise."
        )

        if ctx.perception.visual_context:
            vc = ctx.perception.visual_context
            visual_block = "\n\n[Screen Content - You have captured and can now see the user's screen]\n"
            if vc.text_on_screen:
                visual_block += f"Visible text: {vc.text_on_screen}\n"
            if vc.elements:
                element_summary = ", ".join(
                    f"{el.role}: '{el.name}'" for el in vc.elements[:20]
                )
                visual_block += f"UI elements: {element_summary}\n"
            if vc.caption:
                visual_block += f"Visual description: {vc.caption}\n"
            system_prompt += visual_block
        else:
            system_prompt += "\n\nYou have the ability to capture and analyze the user's screen when they ask you to look at it."

        if ctx.retrieved:
            memory_content = "\nRelevant memories from past sessions:\n"
            for hit in ctx.retrieved:
                memory_content += f" - {hit.text}\n"
            system_prompt += memory_content

        messages = []

        for turn in ctx.working:
            messages.append({
                "role": turn.role if turn.role != "assistant" else "assistant",
                "context": turn.content
            })

        if ctx.perception.text:
            messages.append({
                "role": "user",
                "content": ctx.perception.text
            })

        return system_prompt, messages

    async def close(self):
        await self._client.close()