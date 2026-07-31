FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# deps layer - cached unless pyproject changes
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# source layer
COPY src/ src/
COPY scripts/ scripts/

# non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "src.rag.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
