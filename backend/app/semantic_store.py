from __future__ import annotations

from functools import lru_cache
from typing import Any

try:
    from qdrant_client import QdrantClient, models
except ModuleNotFoundError:  # pragma: no cover - optional dependency in stub tests
    QdrantClient = None
    models = None

from app.config import settings
from app.vector_store import (
    BgeM3Embedder,
    BgeReranker,
    bm25_search,
    hybrid_candidate_limit,
    merge_rankings,
    point_id,
)


class StubSemanticMemoryStore:
    def __init__(self) -> None:
        self.facts: dict[str, dict[str, Any]] = {}

    def ensure_collection(self) -> None:
        return None

    def reset(self) -> None:
        self.facts.clear()

    def upsert_facts(self, project_id: str, facts: list[dict[str, Any]]) -> None:
        for fact in facts:
            self.facts[fact["fact_id"]] = {**fact, "project_id": project_id}

    def delete_project(self, project_id: str) -> None:
        self.facts = {key: value for key, value in self.facts.items() if value["project_id"] != project_id}

    def search(self, project_id: str, query: str, limit: int, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        corpus = [fact for fact in (facts or list(self.facts.values())) if fact["project_id"] == project_id]
        candidate_limit = hybrid_candidate_limit(limit)
        dense_hits = bm25_search(corpus, query, candidate_limit)
        sparse_hits = bm25_search(corpus, query, candidate_limit)
        ranked = merge_rankings(dense_hits, sparse_hits, candidate_limit)
        for item in ranked:
            item["rerank_score"] = item["fusion_score"] + item["dense_score"] + item["sparse_score"]
            item["score"] = item["rerank_score"]
        ranked.sort(key=lambda item: (item["score"], item["importance"]), reverse=True)
        return ranked[:limit]


class QdrantSemanticMemoryStore:
    def __init__(self) -> None:
        if QdrantClient is None or models is None:
            raise RuntimeError("qdrant_client is required when VECTOR_STORE_PROVIDER=qdrant")
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self._embedder: BgeM3Embedder | None = None
        self._reranker: BgeReranker | None = None

    @property
    def embedder(self) -> BgeM3Embedder:
        if self._embedder is None:
            self._embedder = BgeM3Embedder()
        return self._embedder

    @property
    def reranker(self) -> BgeReranker:
        if self._reranker is None:
            self._reranker = BgeReranker()
        return self._reranker

    def ensure_collection(self) -> None:
        if self.client.collection_exists(settings.semantic_memory_collection):
            return
        self.client.create_collection(
            collection_name=settings.semantic_memory_collection,
            vectors_config=models.VectorParams(
                size=settings.embedding_dimension,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert_facts(self, project_id: str, facts: list[dict[str, Any]]) -> None:
        self.ensure_collection()
        if not facts:
            return
        vectors = self.embedder.encode([fact["statement"] for fact in facts])
        points = [
            models.PointStruct(
                id=point_id(str(fact["fact_id"])),
                vector=vector,
                payload={**fact, "project_id": project_id},
            )
            for fact, vector in zip(facts, vectors, strict=False)
        ]
        self.client.upsert(collection_name=settings.semantic_memory_collection, points=points, wait=True)

    def delete_project(self, project_id: str) -> None:
        self.ensure_collection()
        self.client.delete(
            collection_name=settings.semantic_memory_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="project_id",
                            match=models.MatchValue(value=project_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def dense_search(self, project_id: str, query: str, limit: int) -> list[dict[str, Any]]:
        self.ensure_collection()
        response = self.client.query_points(
            collection_name=settings.semantic_memory_collection,
            query=self.embedder.encode([query])[0],
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="project_id",
                        match=models.MatchValue(value=project_id),
                    )
                ]
            ),
            with_payload=True,
            limit=limit,
        )
        hits = []
        for hit in response.points:
            payload = dict(hit.payload or {})
            payload.setdefault("chunk_id", payload.get("fact_id", ""))
            payload.setdefault("title", f"{payload.get('fact_type', 'fact')}:{payload.get('memory_key', 'memory')}")
            payload.setdefault("content", payload.get("statement", ""))
            hits.append({**payload, "score": float(hit.score)})
        return hits

    def rerank(self, query: str, hits: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        passages = [f"{hit['fact_type']}\n{hit['statement']}" for hit in hits]
        scores = self.reranker.score(query, passages)
        ranked = []
        for hit, score in zip(hits, scores, strict=False):
            ranked.append({**hit, "rerank_score": float(score), "score": float(score)})
        ranked.sort(key=lambda item: (item["score"], item["importance"]), reverse=True)
        return ranked[:limit]

    def search(self, project_id: str, query: str, limit: int, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidate_limit = hybrid_candidate_limit(limit)
        dense_hits = self.dense_search(project_id, query, candidate_limit)
        sparse_hits = bm25_search(facts, query, candidate_limit)
        fused_hits = merge_rankings(dense_hits, sparse_hits, candidate_limit)
        return self.rerank(query, fused_hits, limit)


@lru_cache
def get_semantic_memory_store() -> StubSemanticMemoryStore | QdrantSemanticMemoryStore:
    return StubSemanticMemoryStore() if settings.vector_store_provider == "stub" else QdrantSemanticMemoryStore()


def ensure_semantic_memory_store() -> None:
    get_semantic_memory_store().ensure_collection()


def reset_semantic_memory_store() -> None:
    store = get_semantic_memory_store()
    if isinstance(store, StubSemanticMemoryStore):
        store.reset()
