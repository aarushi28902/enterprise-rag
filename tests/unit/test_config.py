import pytest
from src.rag.config import get_settings


def test_settings_load():
    settings = get_settings()
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.top_k == 6


def test_settings_are_cached():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2   # same object = lru_cache working
