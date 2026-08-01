from __future__ import annotations

import json
import time
from pathlib import Path
from PIL import Image
from typing import Optional

from ..core.type import MemoryHit


class VisualStore:
    """
    Persistent visual memory.

    Screenshots are saved as PNG files in a directory.
    Metadata (caption, OCR text, timestamp) is stored in an
    accompanying JSON index and also indexed in ChromaDB
    for semantic search.

    Usage:
        store = VisualStore()
        ref = await store.save(image, caption="A web browser", text="...")
        hits = await store.recall("browser with login page", k=3)
    """

    def __init__(
        self,
        storage_dir: str | Path = "visual_memory",
        embedder=None,       # Reuse BGEmbedder for semantic search
        semantic_store=None, # Reuse SemanticStore for indexing
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.embedder = embedder
        self.semantic_store = semantic_store
        self._index_path = self.storage_dir / "index.json"
        self._index: list[dict] = []
        self._load_index()

    def _load_index(self) -> None:
        """Load the metadata index from disk."""
        if self._index_path.exists():
            try:
                self._index = json.loads(self._index_path.read_text())
            except (json.JSONDecodeError, OSError):
                self._index = []

    def _save_index(self) -> None:
        """Persist the metadata index."""
        try:
            self._index_path.write_text(
                json.dumps(self._index[-500:], indent=2)  # Keep last 500 entries
            )
        except OSError as e:
            print(f"  [VisualStore: could not save index — {e}]")

    async def save(
        self,
        image: Image.Image,
        caption: str,
        text: str,
    ) -> str:
        """
        Save a screenshot with its metadata.

        Args:
            image: PIL Image to save
            caption: VLM-generated description (or OCR text if no VLM)
            text: Raw OCR text

        Returns:
            A reference string (filename) for the saved image.
        """
        timestamp = time.time()
        filename = f"screen_{int(timestamp)}.png"
        filepath = self.storage_dir / filename

        # Save the image
        image.save(filepath, "PNG")

        # Build metadata entry
        entry = {
            "ref": filename,
            "timestamp": timestamp,
            "caption": caption,
            "ocr_text": text,
        }
        self._index.append(entry)
        self._save_index()

        # Also index in semantic store for search
        if self.semantic_store is not None and self.embedder is not None:
            searchable_text = f"{caption}\n{text}"
            await self.semantic_store.upsert(
                text=searchable_text,
                meta={
                    "type": "visual",
                    "ref": filename,
                    "timestamp": timestamp,
                },
            )

        return filename

    async def recall(self, query: str, k: int = 5) -> list[MemoryHit]:
        """
        Search visual memory for relevant screenshots.

        Args:
            query: Natural language query (e.g., "the browser I had open")
            k: Number of results

        Returns:
            List of MemoryHit objects with the image ref in meta.
        """
        if self.semantic_store is not None:
            # Use semantic search
            return await self.semantic_store.recall(query, k=k)

        # Fallback: keyword search in index
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
                    text=entry.get("caption", ""),
                    score=score,
                    meta={"ref": entry["ref"], "timestamp": entry["timestamp"]},
                ))
                if len(results) >= k:
                    break

        return results

    def get_image_path(self, ref: str) -> Path | None:
        """Get the full path to a saved screenshot by its reference."""
        path = self.storage_dir / ref
        return path if path.exists() else None