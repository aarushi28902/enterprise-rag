import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, PointStruct, SparseVector

from src.rag.ingestion.chunker import ParentChildChunker
from src.rag.retrieval.embedder import EmbedderService


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

    text = "Retrieval Augmented Generation combines a retriever and a generator. " * 50
    chunker = ParentChildChunker()
    pairs = list(chunker.chunk(text, metadata={"source": "test.pdf"}))

    print(f"Generated {len(pairs)} parent chunks")

    points = []
    for parent, children in pairs:
        for child in children:
            dense, sparse = await embedder.embed_query(child.text)
            points.append(
                PointStruct(
                    id=child.chunk_id,
                    vector={
                        "dense": dense,
                        "sparse": SparseVector(indices=sparse.indices, values=sparse.values),
                    },
                    payload={
                        "text": child.text,
                        "chunk_id": child.chunk_id,
                        "parent_id": child.parent_id,
                        "source": child.metadata["source"],
                    },
                )
            )

    print(f"Upserting {len(points)} points...")
    await client.upsert(collection_name="enterprise_rag", points=points)

    count = await client.count(collection_name="enterprise_rag")
    print(f"Total points in collection: {count.count}")


asyncio.run(main())
