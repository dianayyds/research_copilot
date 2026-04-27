from __future__ import annotations

import re
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import jieba
except ModuleNotFoundError:  # pragma: no cover - optional dependency in stub tests
    jieba = None

try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError:  # pragma: no cover - optional dependency in stub tests
    snapshot_download = None

try:
    from qdrant_client import QdrantClient, models
except ModuleNotFoundError:  # pragma: no cover - optional dependency in stub tests
    QdrantClient = None
    models = None

try:
    from rank_bm25 import BM25Okapi
except ModuleNotFoundError:  # pragma: no cover - optional dependency in stub tests
    BM25Okapi = None

from app.config import settings


MODEL_FILES = [
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "sentencepiece.bpe.model",
    "spiece.model",
    "vocab.txt",
    "merges.txt",
]
WEIGHT_FILES = [
    "model.safetensors",
    "model.safetensors.index.json",
]
CORE_MODEL_FILES = ["config.json", "tokenizer_config.json"]
TOKENIZER_FILES = ["tokenizer.json", "sentencepiece.bpe.model", "spiece.model", "vocab.txt"]


def tokenize(text: str) -> list[str]:
    raw_tokens = (
        [token.strip().lower() for token in jieba.cut_for_search(text) if token.strip()]
        if jieba is not None
        else [token.strip().lower() for token in re.findall(r"[\u4e00-\u9fff]+", text) if token.strip()]
    )
    latin_tokens = re.findall(r"[a-zA-Z0-9_./-]+", text.lower())
    return list(dict.fromkeys(raw_tokens + latin_tokens))


def bm25_tokens(text: str) -> list[str]:
    return tokenize(text) or [text.strip().lower() or "__empty__"]


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def local_model_path(repo_id: str, root: Path) -> Path:
    return root / repo_id.split("/")[-1]


def ensure_model_snapshot(repo_id: str, root: Path) -> str:
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub is required for local embedding models")
    target = local_model_path(repo_id, root)
    weights = [target / name for name in WEIGHT_FILES]
    if (
        all((target / name).exists() for name in CORE_MODEL_FILES)
        and any((target / name).exists() for name in TOKENIZER_FILES)
        and any(path.exists() for path in weights)
    ):
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        allow_patterns=MODEL_FILES + WEIGHT_FILES,
    )
    return str(target)


def resolve_device(requested: str, torch: Any) -> str:
    return requested if requested != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")


def hybrid_candidate_limit(limit: int) -> int:
    return max(settings.retrieval_candidate_limit, limit * 4)


def effective_model_max_length(
    configured_max_length: int,
    tokenizer_max_length: int | None,
    model_max_positions: int | None,
) -> int:
    candidates = [configured_max_length]
    if tokenizer_max_length and tokenizer_max_length < 100000:
        candidates.append(tokenizer_max_length)
    if model_max_positions and model_max_positions > 0:
        candidates.append(model_max_positions)
    return max(32, min(candidates))


def filter_chunks(chunks: list[dict[str, Any]], asset_ids: list[str]) -> list[dict[str, Any]]:
    if not asset_ids:
        return chunks
    allowed = set(asset_ids)
    return [chunk for chunk in chunks if chunk["asset_id"] in allowed]


def lexical_overlap_search(chunks: list[dict[str, Any]], query: str, limit: int) -> list[dict[str, Any]]:
    query_tokens = set(tokenize(query))
    ranked = []
    for chunk in chunks:
        score = float(len(query_tokens.intersection(tokenize(chunk["content"]))))
        if score > 0:
            ranked.append({**chunk, "score": score})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:limit]


def bm25_search(chunks: list[dict[str, Any]], query: str, limit: int) -> list[dict[str, Any]]:
    if not chunks:
        return []
    if BM25Okapi is None:
        return lexical_overlap_search(chunks, query, limit)
    bm25 = BM25Okapi([bm25_tokens(chunk["content"]) for chunk in chunks])
    scores = bm25.get_scores(bm25_tokens(query))
    ranked = [
        {**chunk, "score": float(score)}
        for chunk, score in zip(chunks, scores, strict=False)
        if float(score) > 0
    ]
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:limit]


def merge_rankings(
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for score_key, hits in (("dense_score", dense_hits), ("sparse_score", sparse_hits)):
        for rank, hit in enumerate(hits, start=1):
            current = merged.setdefault(
                str(hit["chunk_id"]),
                {
                    **hit,
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                    "fusion_score": 0.0,
                    "rerank_score": 0.0,
                },
            )
            current[score_key] = max(float(hit["score"]), float(current[score_key]))
            current["fusion_score"] += 1.0 / (60.0 + rank)
    ranked = list(merged.values())
    ranked.sort(key=lambda item: (item["fusion_score"], item["dense_score"], item["sparse_score"]), reverse=True)
    return ranked[:limit]


class StubVectorStore:
    def __init__(self) -> None:
        self.chunks: dict[str, dict[str, Any]] = {}

    def ensure_collection(self) -> None:
        return None

    def reset(self) -> None:
        self.chunks.clear()

    def upsert_chunks(self, chunks: list[dict[str, str]]) -> None:
        for chunk in chunks:
            self.chunks[chunk["chunk_id"]] = dict(chunk)

    def delete_asset(self, asset_id: str) -> None:
        self.chunks = {key: value for key, value in self.chunks.items() if value["asset_id"] != asset_id}

    def delete_project(self, project_id: str) -> None:
        return None

    def search(
        self,
        query: str,
        limit: int,
        asset_ids: list[str],
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        corpus = filter_chunks(chunks or list(self.chunks.values()), asset_ids)
        candidate_limit = hybrid_candidate_limit(limit)
        dense_hits = lexical_overlap_search(corpus, query, candidate_limit)
        sparse_hits = bm25_search(corpus, query, candidate_limit)
        ranked = merge_rankings(dense_hits, sparse_hits, candidate_limit)
        for item in ranked:
            item["rerank_score"] = item["fusion_score"] + item["dense_score"] + item["sparse_score"]
            item["score"] = item["rerank_score"]
        ranked.sort(key=lambda item: (item["score"], item["fusion_score"]), reverse=True)
        return ranked[:limit]


class BgeM3Embedder:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_path = ensure_model_snapshot(settings.embedding_model, settings.embedding_model_dir)
        self.torch = torch
        self.device = resolve_device(settings.embedding_device.lower(), torch)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), settings.embedding_batch_size):
            batch_texts = texts[start : start + settings.embedding_batch_size]
            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=settings.embedding_max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with self.torch.no_grad():
                outputs = self.model(**batch)
                dense = outputs.last_hidden_state[:, 0]
                dense = self.torch.nn.functional.normalize(dense, p=2, dim=1)
            vectors.extend(dense.cpu().tolist())
        return vectors


class BgeReranker:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_path = ensure_model_snapshot(settings.reranker_model, settings.reranker_model_dir)
        self.torch = torch
        self.device = resolve_device(settings.reranker_device.lower(), torch)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = effective_model_max_length(
            settings.reranker_max_length,
            getattr(self.tokenizer, "model_max_length", None),
            getattr(self.model.config, "max_position_embeddings", None),
        )

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores: list[float] = []
        pairs = [[query, passage] for passage in passages]
        for start in range(0, len(pairs), settings.reranker_batch_size):
            batch_pairs = pairs[start : start + settings.reranker_batch_size]
            batch = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with self.torch.no_grad():
                logits = self.model(**batch).logits.view(-1)
            scores.extend(logits.cpu().tolist())
        return scores


class QdrantVectorStore:
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
        if self.client.collection_exists(settings.qdrant_collection):
            return
        self.client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=models.VectorParams(
                size=settings.embedding_dimension,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert_chunks(self, chunks: list[dict[str, str]]) -> None:
        self.ensure_collection()
        if not chunks:
            return
        vectors = self.embedder.encode([chunk["content"] for chunk in chunks])
        points = [
            models.PointStruct(
                id=point_id(chunk["chunk_id"]),
                vector=vector,
                payload=dict(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=False)
        ]
        self.client.upsert(collection_name=settings.qdrant_collection, points=points, wait=True)

    def delete_asset(self, asset_id: str) -> None:
        self.ensure_collection()
        self.client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="asset_id",
                            match=models.MatchValue(value=asset_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def delete_project(self, project_id: str) -> None:
        self.ensure_collection()
        self.client.delete(
            collection_name=settings.qdrant_collection,
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

    def dense_search(self, query: str, limit: int, asset_ids: list[str]) -> list[dict[str, Any]]:
        self.ensure_collection()
        must_conditions: list[models.Condition] = []
        if asset_ids:
            must_conditions.append(
                models.FieldCondition(
                    key="asset_id",
                    match=models.MatchAny(any=asset_ids),
                )
            )
        response = self.client.query_points(
            collection_name=settings.qdrant_collection,
            query=self.embedder.encode([query])[0],
            query_filter=models.Filter(must=must_conditions),
            with_payload=True,
            limit=limit,
        )
        return [{**dict(hit.payload or {}), "score": float(hit.score)} for hit in response.points]

    def rerank(self, query: str, hits: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        passages = [f"{hit['title']}\n{hit['content']}" for hit in hits]
        scores = self.reranker.score(query, passages)
        ranked = []
        for hit, score in zip(hits, scores, strict=False):
            ranked.append({**hit, "rerank_score": float(score), "score": float(score)})
        ranked.sort(key=lambda item: (item["score"], item["fusion_score"]), reverse=True)
        return ranked[:limit]

    def search(
        self,
        query: str,
        limit: int,
        asset_ids: list[str],
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidate_limit = hybrid_candidate_limit(limit)
        corpus = filter_chunks(chunks, asset_ids)
        if not corpus:
            return []
        dense_hits = self.dense_search(query, candidate_limit, asset_ids)
        sparse_hits = bm25_search(corpus, query, candidate_limit)
        fused_hits = merge_rankings(dense_hits, sparse_hits, candidate_limit)
        return self.rerank(query, fused_hits, limit)


@lru_cache
def get_vector_store() -> StubVectorStore | QdrantVectorStore:
    return StubVectorStore() if settings.vector_store_provider == "stub" else QdrantVectorStore()


def ensure_vector_store() -> None:
    get_vector_store().ensure_collection()


def reset_vector_store() -> None:
    store = get_vector_store()
    if isinstance(store, StubVectorStore):
        store.reset()
