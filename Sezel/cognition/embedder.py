from sentence_transformers.util import normalize_embeddings

from ..core.protocols import Embedder

class BGEmbedder(Embedder):


    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):

        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    async def embed(self, text: str) -> list[float]:

        self._load()

        import asyncio
        loop = asyncio.get_event_loop()

        def _embed():
            vec = self._model.encode(text)
            return vec.tolist()

        return await loop.run_in_executor(None, _embed)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:

        self._load()

        import asyncio
        loop = asyncio.get_event_loop()

        def _embed_batch():
            vecs = self._model.encode(texts, normalize_embeddings=True)
            return [v.tolist() for v in vecs]

        return await loop.run_in_executor(None, _embed_batch)