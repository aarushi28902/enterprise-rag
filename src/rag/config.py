from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "enterprise_rag"
    embed_model: str = "BAAI/bge-m3"
    generation_model: str = "llama3.2"
    ollama_url: str = "http://localhost:11434"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 6
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
