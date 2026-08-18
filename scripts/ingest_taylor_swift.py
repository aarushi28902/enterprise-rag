"""Ingest Taylor Swift Wikipedia articles into Qdrant."""
import asyncio
from pathlib import Path
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    VectorParams, Distance, SparseVectorParams, PointStruct, SparseVector
)
from src.rag.ingestion.chunker import ParentChildChunker
from src.rag.retrieval.embedder import EmbedderService

FILES = [
    ("data/folklore_album.txt", "Folklore Album - Wikipedia"),
    ("data/evermore_album.txt", "Evermore Album - Wikipedia"),
    ("data/betty_song.txt", "Betty Song - Wikipedia"),
    ("data/cardigan_song.txt", "Cardigan Song - Wikipedia"),
    ("data/willow_song.txt", "Willow Song - Wikipedia"),
]


async def main():
    client = AsyncQdrantClient(url="http://localhost:6333")
    embedder = EmbedderService()

    print("Loading embedder...")
    await embedder.load()

    print("Creating collection...")
    await client.create_collection(
        collection_name="enterprise_rag",
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )

    chunker = ParentChildChunker(parent_size=1024, child_size=256, overlap=32)

    total_points = 0
    for filepath, source in FILES:
        text = Path(filepath).read_text()
        print(f"\nIngesting: {source} ({len(text)} chars)")

        pairs = list(chunker.chunk(text, metadata={"source": source}))
        points = []

        for parent, children in pairs:
            for child in children:
                dense, sparse = await embedder.embed_query(child.text)
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

        await client.upsert(collection_name="enterprise_rag", points=points)
        total_points += len(points)
        print(f"  → {len(pairs)} parents, {len(points)} children ingested")

    count = await client.count(collection_name="enterprise_rag")
    print(f"\nTotal points in collection: {count.count}")


asyncio.run(main())
