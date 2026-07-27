"""Visual memory: store and recall screenshots with captions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from PIL import Image
from typing import Optional

from langgraph.func import entrypoint
from sqlalchemy.ext.asyncio import result

from ..core.type import MemoryHit

class VisualStore:
    """

    """
    def __init__(
            self,
            storage_dir: str | Path = "visual_memory",
            embedder = None,
            semantic_store = None,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.embedder = embedder
        self.semantic_store = semantic_store
        self._index_path = self.storage_dir / "index.json"
        self._index: list[dict] = []
        self._load_index()

    def _load_index(self) -> None:
        """Persist the metadata index."""
        try:
            self._index_path.write_text(
                json.dumps(self._index[-500:], indent = 2)
            )
        except OSError as e:
            print(f"  [VisualStore: could not save index — {e}]")
    async def save(
        self,
        image: Image.Image,
        caption: str,
        text: str
    ):
        """

        """

        timestamp = time.time()
        filename = f"screen_{int(timestamp)}.png"
        file_path = self.storage_dir / filename

        image.save(file_path, "PNG")

        entry = {
            "ref": filename,
            "timestamp": timestamp,
            "caption": caption,
            "ocr_text": text,
        }
        self._index.append(entry)
        self._save_index()

        if self.semantic_store is not None and self.embedder is not None:
            searchable_text = f"{caption}\n{text}"
            await self.semantic_store.upsert(
                text = searchable_text,
                meta = {
                    "type": "visual",
                    "ref": filename,
                    "timestamp": timestamp,
                },
            )
        return filename

    async def recall(self, query: str, k: int = 5):
        """

        """
        if self.semantic_store is not None:
            return await self.semantic_store

        query_lower = query.lower()
        results = []
        for entry in reversed(self._index):
            score = 0.0
            if query_lower in entry.get("caption", "").lower():
                score += 0.5
            if query_lower in entry.get("ocr_text", "").lower():
                score += 0.3
            if score > 0:
                results.append(MemoryHit(
                    text = entry.get("caption", ""),
                    score = score,
                    meta = {
                        "ref": entry["ref"],
                        "timestamp": entry["timestamp"],
                    }
                ))
                if len(results) >= k:
                    break

        return results

    def get_image_path(self, ref: str) -> Path | None:
        """Get the full path to a saved screenshot by its reference."""
        path = self.storage_dir / ref
        return path if path.exists() else None