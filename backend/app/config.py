from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseModel):
    app_name: str = os.getenv("APP_NAME", "research-copilot-runtime")
    app_version: str = os.getenv("APP_VERSION", "0.2.0")
    environment: str = os.getenv("ENVIRONMENT", "local")
    host: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    port: int = int(os.getenv("BACKEND_PORT", "8001"))
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    mysql_host: str = os.getenv("MYSQL_HOST", "127.0.0.1")
    mysql_port: int = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_database: str = os.getenv("MYSQL_DATABASE", "agent_studio")
    mysql_user: str = os.getenv("MYSQL_USER", "agent")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "agent")
    redis_host: str = os.getenv("REDIS_HOST", "redis")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    minio_host: str = os.getenv("MINIO_HOST", "minio")
    minio_port: int = int(os.getenv("MINIO_PORT", "9000"))
    qdrant_host: str = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "knowledge_chunks")
    semantic_memory_collection: str = os.getenv("SEMANTIC_MEMORY_COLLECTION", "semantic_memory_facts")
    llm_provider: str = os.getenv("LLM_PROVIDER", "deepseek")
    llm_model: str = os.getenv("LLM_MODEL", "deepseek-chat")
    llm_api_base: str = os.getenv("LLM_API_BASE", "https://api.deepseek.com")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    embedding_model_dir: Path = Path(os.getenv("EMBEDDING_MODEL_DIR", "/models"))
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "4"))
    embedding_max_length: int = int(os.getenv("EMBEDDING_MAX_LENGTH", "2048"))
    reranker_provider: str = os.getenv("RERANKER_PROVIDER", "local")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    reranker_model_dir: Path = Path(os.getenv("RERANKER_MODEL_DIR", "/models"))
    reranker_device: str = os.getenv("RERANKER_DEVICE", "cpu")
    reranker_batch_size: int = int(os.getenv("RERANKER_BATCH_SIZE", "2"))
    reranker_max_length: int = int(os.getenv("RERANKER_MAX_LENGTH", "1024"))
    execution_mode: str = os.getenv("EXECUTION_MODE", "plan_and_solve")
    retrieval_limit: int = int(os.getenv("RETRIEVAL_LIMIT", "5"))
    retrieval_candidate_limit: int = int(os.getenv("RETRIEVAL_CANDIDATE_LIMIT", "16"))
    asset_chunk_size: int = int(os.getenv("ASSET_CHUNK_SIZE", "500"))
    working_memory_limit: int = int(os.getenv("WORKING_MEMORY_LIMIT", "8"))
    episodic_memory_limit: int = int(os.getenv("EPISODIC_MEMORY_LIMIT", "4"))
    semantic_memory_limit: int = int(os.getenv("SEMANTIC_MEMORY_LIMIT", "6"))
    upload_max_bytes: int = int(os.getenv("UPLOAD_MAX_BYTES", str(2 * 1024 * 1024)))
    cors_origins: list[str] = _split_csv(os.getenv("CORS_ORIGINS", "*"))
    database_url: str = os.getenv("DATABASE_URL", "")
    vector_store_provider: str = os.getenv("VECTOR_STORE_PROVIDER", "qdrant")

    @property
    def resolved_database_url(self) -> str:
        return self.database_url or (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )


settings = Settings()
