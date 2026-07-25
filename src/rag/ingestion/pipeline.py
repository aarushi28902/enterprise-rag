import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, SparseVector

from src.rag.config import get_settings
from src.rag.ingestion.chunker import ParentChildChunker
from src.rag.retrieval.embedder import EmbedderService

logger = structlog.get_logger()


class IngestionPipeline:
    def __init__(self):
        self.settings = get_settings()
        self.chunker = ParentChildChunker()
        self.embedder = EmbedderService()

    async def run_pdf(self, content: bytes, filename: str, collection: str) -> None:
        import asyncio
        from pypdf import PdfReader
        import io

        logger.info("ingestion_start", filename=filename)

        reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        await self._ingest_text(text=text, source=filename, collection=collection)
        logger.info("ingestion_complete", filename=filename)

    async def _ingest_text(self, text: str, source: str, collection: str) -> None:
        import asyncio
        from qdrant_client import AsyncQdrantClient

        client = AsyncQdrantClient(url=self.settings.qdrant_url)
        await self.embedder.load()

        pairs = list(self.chunker.chunk(text, metadata={"source": source}))
        points = []

        for parent, children in pairs:
            for child in children:
                dense, sparse = await self.embedder.embed_query(child.text)
                points.append(
                    PointStruct(
                        id=child.chunk_id,
                        vector={
                            "dense": dense,
                            "sparse": SparseVector(
                                indices=sparse.indices,
                                values=sparse.values,
                            ),
                        },
                        payload={
                            "text": child.text,
                            "chunk_id": child.chunk_id,
                            "parent_id": child.parent_id,
                            "source": source,
                        },
                    )
                )

        await client.upsert(collection_name=collection, points=points)
        logger.info("upsert_complete", n_points=len(points))
