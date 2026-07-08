from pathlib import Path

from sqlalchemy.dialects.oracle import vector
from sqlalchemy.ext.asyncio import result
from sqlalchemy.orm.collections import collection
from sympy.plotting.backends.textbackend import text

from ..core.type import MemoryHit
from ..cognition.embedder import BGEmbedder

class SemanticStore:

    def __init__(
        self,
        embedder: BGEmbedder,
        collection_name: str = "sezel_memory",
        persist_directory: str | Path = "sezel_memory"
    ):
        self.embedder = embedder
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)

        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path= (self.persist_directory),
                settings = Settings(anonymized_telemetry=False)
            )

            self._collection = client.get_or_create_collection(
                name= self.collection_name,
                metadata= {"hnsw:space": "cosine"}
            )

        return self._collection

    async def upsert(self, text: str, meta: dict | None = None):

        """
        'upsert' = UPdate + inSERT. If the ID already exists, update it.
        If not, insert it as new.

        Args:
            text: The memory content (e.g. "User's favorite color is blue")
            meta: Extra info (timestamps, source, category, etc.)

        Returns:
            The unique ID of the stored memory
        """
        collection = self._get_collection()

        vector = await self.embedder.embed(text)

        import hashlib
        doc_id = hashlib.md5(text.encode()).hexdigest()

        if meta is None:
            meta = {}

        collection.upsert(
            ids = [doc_id],
            embeddings = [vector],
            documents= [text],
            metadatas= [meta]
        )

        return doc_id

    async def recall(self, query: str, k: int = 5) -> list[MemoryHit]:

        """
        Args:
            query: What to search for (e.g. "What name did user say?")
            k: How many results to return (default 5)

        Returns:
            List of MemoryHit objects, sorted by relevance (best first)
        """
        collection = self._get_collection()

        count = collection.count()
        if count == 0:
            return []

        query_vector = await self.embedder.embed(query)

        results = collection.query(
            query_embeddings=[query_vector],
            n_results= min(k, count),
            include= ["document", "metadata", "distances"]
        )

        memories = []
        if results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                text = results["documents"][0][i]
                distance = results["distances"][0][i]
                score = max(0.0, 1.0 - distance)
                meta = results["metadatas"][0][i] if results["metadatas"] else {}

                memories.append(MemoryHit(
                    text = text,
                    score = score,
                    meta = meta
                ))

        return memories

    async def count(self) -> int:

        return self._get_collection().count()

    async def clear(self) -> None:
        """delete all memory"""
        collection = self._get_collection()
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)

    async def close(self) -> None:
        """Clean up (ChromaDB persists automatically on disk)."""
        self._collection = None
