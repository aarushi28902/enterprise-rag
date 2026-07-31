import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_embedder():
    emb = AsyncMock()
    emb.embed_query.return_value = (
        [0.1] * 1024,
        MagicMock(indices=[1, 2], values=[0.5, 0.3]),
    )
    return emb


@pytest_asyncio.fixture
async def client(mock_embedder):
    from src.rag.api.main import create_app
    from src.rag.api.dependencies import get_embedder, get_qdrant

    app = create_app()
    app.state.embedder = mock_embedder

    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_qdrant] = lambda: AsyncMock()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_health_check(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_query_too_short_rejected(client):
    r = await client.post(
        "/api/v1/query/stream",
        json={"query": "hi"},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_query_valid_returns_stream(client):
    r = await client.post(
        "/api/v1/query/stream",
        json={"query": "What is retrieval augmented generation?"},
    )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
