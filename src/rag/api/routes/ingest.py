import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from pydantic import BaseModel

from src.rag.retrieval.embedder import EmbedderService
from src.rag.ingestion.pipeline import IngestionPipeline

logger = structlog.get_logger()
router = APIRouter()


class IngestResponse(BaseModel):
    status: str
    filename: str
    collection: str


@router.post("/pdf", response_model=IngestResponse)
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection: str = "enterprise_rag",
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    content = await file.read()
    background_tasks.add_task(
        IngestionPipeline().run_pdf,
        content=content,
        filename=file.filename,
        collection=collection,
    )
    logger.info("ingest_queued", filename=file.filename)
    return IngestResponse(status="queued", filename=file.filename, collection=collection)
