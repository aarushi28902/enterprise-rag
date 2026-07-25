import time
import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.rag.api.dependencies import get_embedder, get_qdrant
from src.rag.generation.generator import StreamingGenerator
from src.rag.monitoring.metrics import (
    REQUEST_LATENCY,
    RETRIEVAL_LATENCY,
    GENERATION_LATENCY,
    ACTIVE_REQUESTS,
    QUERY_ERRORS,
)
from src.rag.retrieval.hybrid_search import HybridSearcher

logger = structlog.get_logger()
router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=20)
    collection: str = "enterprise_rag"


@router.post("/stream")
async def query_stream(
    req: QueryRequest,
    embedder=Depends(get_embedder),
    qdrant=Depends(get_qdrant),
):
    async def stream():
        ACTIVE_REQUESTS.inc()
        t0 = time.perf_counter()
        try:
            # retrieval
            t_ret = time.perf_counter()
            searcher = HybridSearcher(qdrant_client=qdrant, embedder=embedder)
            chunks = await searcher.search(
                query=req.query,
                collection=req.collection,
                top_k=req.top_k,
            )
            RETRIEVAL_LATENCY.observe(time.perf_counter() - t_ret)
            logger.info("retrieval_done", n_chunks=len(chunks), query=req.query[:80])

            if not chunks:
                yield "data: I couldn't find relevant information.\n\n"
                return

            # generation
            t_gen = time.perf_counter()
            generator = StreamingGenerator()
            async for token in generator.stream(query=req.query, chunks=chunks):
                yield f"data: {token}\n\n"
            GENERATION_LATENCY.observe(time.perf_counter() - t_gen)

            yield "data: [DONE]\n\n"
            REQUEST_LATENCY.observe(time.perf_counter() - t0)

        except Exception as e:
            QUERY_ERRORS.inc()
            logger.error("query_failed", error=str(e))
            yield f"data: [ERROR] {e}\n\n"

        finally:
            ACTIVE_REQUESTS.dec()

    return StreamingResponse(stream(), media_type="text/event-stream")
