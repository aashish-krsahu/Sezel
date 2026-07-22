"""Vision Language Model interface — caption and locate via Ollama vision models."""
from __future__ import annotations

from typing import Optional
from PIL import Image
import httpx

from core.protocols import VLM

class OllamaVLM(VLM):

    """
        Usage:
        vlm = OllamaVLM(model="llava")
        caption = await vlm.caption(screenshot_pil)
        bbox = await vlm.locate(screenshot_pil, "the search button")
    """
    def __init__(
            self,
            model: str = "llava",
            host: str = "https://localhost:11434",
            timeout: float = 120.0
    ):
        self.model = model
        self.host = host
        self.client = httpx.AsyncClient(base_url=host, timeout=timeout)

    async def caption(self, image: Image.Image, prompt: str | None = None) -> str:
        """
        Generate a caption/description of the image.

        Args:
            image: PIL Image to describe
            prompt: Optional specific instruction (e.g., "What's in this image?")

        Returns:
            A text description of the image.
        """
        import base64
        from io import BytesIO

        buf = BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        user_prompt = prompt or "Describe this screenshot in detail. What windows, applications, and content do you see? Be specific about visible text, buttons, and layout."

        messages = [
            {
                "role": "user",
                "content": user_prompt,
                "images": [b64],
            }
        ]

        try:
            resp = await self.client.post(
                "/api/chat",
                json = {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_ctx": 4096,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("messages", {}).get("content", "").strip()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama VLM API error: {e}") from e

    async def locate(
            self,
            image: Image.Image,
            description: str,
    ) -> tuple[int, int, int, int] | None:
        """
        Locate an element described in natural language.

        Args:
            image: PIL Image of the screen
            description: e.g., "the search button", "the login form"

        Returns:
            (left, top, right, bottom) bounding box, or None if not found.
        """

        import base64
        from io import BytesIO
        import json

        buf = BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = (
            f"Find '{description}' in this screenshot. "
            "Return ONLY a JSON object with the bounding box "
            "in the format: {\"left\": X, \"top\": Y, \"right\": X2, \"bottom\": Y2}. "
            "If not found, return {\"found\": false}."
        )

        messages = [
            {
                "role": "user",
                "content": prompt,
                "images": [b64],
            }
        ]

        try:
            resp = await self.client.post(
                "/api/chat",
                json = {
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
            text = data.get("messages", {}).get("content", "").strip()

            # Try to parse JSON from the response
            try:
                result = json.loads(text)
                if result["found"] is False:
                    return None
                return (
                    result["left"],
                    result["top"],
                    result["right"],
                    result["bottom"],
                )
            except (json.JSONDecodeError, KeyError):
                return None

        except httpx.RequestError as e:
            raise RuntimeError(f"Ollama VLM locate error: {e}") from e

async def close(self) -> None:
        await self.client.aclose()