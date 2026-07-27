from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db_models import EpisodicMemoryItem, SemanticMemoryFact, WorkingMemoryItem
from app.document_processing import token_count
from app.models import MemoryItem, MemoryRecordResponse
from app.semantic_store import get_semantic_memory_store
from app.vector_store import tokenize


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_datetime(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def compact_text(text: str, limit: int = 240) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    return clean[:limit]


def slugify(text: str, limit: int = 48) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", text.strip().lower()).strip("_")
    return (normalized or "item")[:limit]


def split_sentences(text: str, limit: int = 3) -> list[str]:
    pieces = [item.strip() for item in re.split(r"[。\n!?；;]+", text) if item.strip()]
    return pieces[:limit]


def working_memory_text(item: WorkingMemoryItem) -> str:
    return f"{item.memory_key}: {item.content}"


def working_memory_token_count(item: WorkingMemoryItem) -> int:
    return max(1, token_count(working_memory_text(item)))


def working_memory_token_total(items: list[WorkingMemoryItem]) -> int:
    return sum(working_memory_token_count(item) for item in items)


def normalized_compaction_ratio() -> float:
    return min(max(settings.working_memory_compaction_ratio, 0.05), 0.9)


def llm_memory_available() -> bool:
    return settings.llm_provider == "deepseek" and bool(settings.llm_api_key)


def llm_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response does not contain a JSON object")
    return stripped[start : end + 1]


def working_memories_chronological(db: Session, project_id: str, session_id: str) -> list[WorkingMemoryItem]:
    return db.scalars(
        select(WorkingMemoryItem)
        .where(WorkingMemoryItem.project_id == project_id, WorkingMemoryItem.session_id == session_id)
        .order_by(WorkingMemoryItem.sequence_id, WorkingMemoryItem.created_at)
    ).all()


def select_working_memories_for_context(items: list[WorkingMemoryItem]) -> list[WorkingMemoryItem]:
    threshold = max(1, settings.working_memory_token_threshold)
    if working_memory_token_total(items) <= threshold:
        return items
    selected: list[WorkingMemoryItem] = []
    selected_tokens = 0
    for item in reversed(items):
        item_tokens = working_memory_token_count(item)
        if selected and selected_tokens + item_tokens > threshold:
            continue
        selected.append(item)
        selected_tokens += item_tokens
        if selected_tokens >= threshold:
            break
    return list(reversed(selected))


def trim_working_memory(db: Session, project_id: str, session_id: str) -> None:
    compact_working_memory_if_needed(db, project_id, session_id, 0)


def serialize_working(item: WorkingMemoryItem) -> MemoryRecordResponse:
    return MemoryRecordResponse(
        id=item.id,
        project_id=item.project_id,
        memory_type="working",
        memory_key=item.memory_key,
        memory_value=item.content,
        source=item.source,
        session_id=item.session_id,
        sequence_id=item.sequence_id,
        importance=item.importance,
        metadata=item.meta_payload,
        updated_at=item.updated_at,
    )


def serialize_episodic(item: EpisodicMemoryItem) -> MemoryRecordResponse:
    return MemoryRecordResponse(
        id=item.id,
        project_id=item.project_id,
        memory_type="episodic",
        memory_key=item.event_type,
        memory_value=item.summary,
        source=item.source,
        session_id=item.session_id,
        sequence_id=item.sequence_id,
        importance=item.importance,
        metadata=item.details,
        updated_at=item.updated_at,
    )


def serialize_semantic(item: SemanticMemoryFact) -> MemoryRecordResponse:
    return MemoryRecordResponse(
        id=item.id,
        project_id=item.project_id,
        memory_type=f"semantic.{item.fact_type}",
        memory_key=item.memory_key,
        memory_value=item.statement,
        source=item.source,
        session_id=item.session_id,
        sequence_id=item.sequence_id,
        importance=item.importance,
        metadata=item.meta_payload,
        updated_at=item.updated_at,
    )


def list_layered_memories(db: Session, project_id: str) -> list[MemoryRecordResponse]:
    working = [serialize_working(item) for item in db.scalars(select(WorkingMemoryItem).where(WorkingMemoryItem.project_id == project_id)).all()]
    episodic = [serialize_episodic(item) for item in db.scalars(select(EpisodicMemoryItem).where(EpisodicMemoryItem.project_id == project_id)).all()]
    semantic = [serialize_semantic(item) for item in db.scalars(select(SemanticMemoryFact).where(SemanticMemoryFact.project_id == project_id)).all()]
    return sorted(working + episodic + semantic, key=lambda item: item.updated_at, reverse=True)


def recent_working_memories(db: Session, project_id: str, session_id: str) -> list[WorkingMemoryItem]:
    return select_working_memories_for_context(working_memories_chronological(db, project_id, session_id))


def episodic_score(query: str, item: EpisodicMemoryItem) -> float:
    tokens = set(tokenize(query))
    event_text = f"{item.summary} {json.dumps(item.details, ensure_ascii=False)}"
    overlap = len(tokens.intersection(tokenize(event_text)))
    age_hours = max((utc_now() - normalize_datetime(item.updated_at)).total_seconds() / 3600.0, 0.0)
    recency = 1.0 / (1.0 + age_hours / 24.0)
    return overlap + recency * 0.6 + item.importance


def relevant_episodic_memories(db: Session, project_id: str, query: str) -> list[EpisodicMemoryItem]:
    episodes = db.scalars(
        select(EpisodicMemoryItem)
        .where(EpisodicMemoryItem.project_id == project_id)
        .order_by(desc(EpisodicMemoryItem.updated_at))
        .limit(max(settings.episodic_memory_limit * 4, 12))
    ).all()
    ranked = sorted(episodes, key=lambda item: episodic_score(query, item), reverse=True)
    return ranked[: settings.episodic_memory_limit]


def semantic_fact_payload(item: SemanticMemoryFact) -> dict[str, Any]:
    return {
        "fact_id": item.id,
        "chunk_id": item.id,
        "project_id": item.project_id,
        "session_id": item.session_id or "",
        "sequence_id": item.sequence_id,
        "fact_type": item.fact_type,
        "memory_key": item.memory_key,
        "title": f"{item.fact_type}:{item.memory_key}",
        "statement": item.statement,
        "content": item.statement,
        "subject": item.subject,
        "predicate": item.predicate,
        "object": item.object,
        "importance": item.importance,
        "source": item.source,
        "metadata": item.meta_payload,
    }


def relevant_semantic_memories(db: Session, project_id: str, query: str) -> list[SemanticMemoryFact]:
    facts = db.scalars(
        select(SemanticMemoryFact)
        .where(SemanticMemoryFact.project_id == project_id)
        .order_by(desc(SemanticMemoryFact.updated_at))
        .limit(max(settings.semantic_memory_limit * 6, 24))
    ).all()
    if not facts:
        return []
    ranked = get_semantic_memory_store().search(
        project_id,
        query,
        settings.semantic_memory_limit,
        [semantic_fact_payload(item) for item in facts],
    )
    if not ranked:
        return facts[: settings.semantic_memory_limit]
    fact_ids = [str(item["fact_id"]) for item in ranked]
    fact_map = {item.id: item for item in facts}
    selected = [fact_map[fact_id] for fact_id in fact_ids if fact_id in fact_map]
    return selected or facts[: settings.semantic_memory_limit]


@dataclass
class MemoryContextBundle:
    working_lines: list[str]
    episodic_lines: list[str]
    semantic_lines: list[str]

    @property
    def working_text(self) -> str:
        return "\n".join(self.working_lines) or "- no working memory yet"

    @property
    def episodic_text(self) -> str:
        return "\n".join(self.episodic_lines) or "- no episodic memory yet"

    @property
    def semantic_text(self) -> str:
        return "\n".join(self.semantic_lines) or "- no semantic memory yet"

    @property
    def combined_text(self) -> str:
        return "\n".join(
            [
                "### Working Memory",
                self.working_text,
                "### Episodic Memory",
                self.episodic_text,
                "### Semantic Memory",
                self.semantic_text,
            ]
        )


def build_memory_context_bundle(db: Session, project_id: str, session_id: str, query: str) -> MemoryContextBundle:
    working = recent_working_memories(db, project_id, session_id)
    episodic = relevant_episodic_memories(db, project_id, query)
    semantic = relevant_semantic_memories(db, project_id, query)
    return MemoryContextBundle(
        working_lines=[f"- {item.memory_key}: {compact_text(item.content, 240)}" for item in working],
        episodic_lines=[f"- [{item.event_type}] {compact_text(item.summary, 160)}" for item in episodic],
        semantic_lines=[f"- {item.fact_type}.{item.memory_key}: {compact_text(item.statement, 160)}" for item in semantic],
    )


def store_working_memory(
    db: Session,
    project_id: str,
    session_id: str,
    sequence_id: int,
    user_query: str,
    answer: str,
    citations: list[str],
) -> list[WorkingMemoryItem]:
    items = [
        WorkingMemoryItem(
            id=make_id("wm"),
            project_id=project_id,
            session_id=session_id,
            sequence_id=sequence_id,
            memory_key=f"turn_{sequence_id}_query",
            content=compact_text(user_query, 240),
            importance=0.7,
            meta_payload={"kind": "query", "sequence_id": sequence_id},
        ),
        WorkingMemoryItem(
            id=make_id("wm"),
            project_id=project_id,
            session_id=session_id,
            sequence_id=sequence_id,
            memory_key=f"turn_{sequence_id}_answer",
            content=compact_text(answer, 240),
            importance=0.8,
            meta_payload={"kind": "answer_summary", "citations": citations, "sequence_id": sequence_id},
        ),
    ]
    for item in items:
        db.add(item)
    db.flush()
    return items


def store_episodic_memory(
    db: Session,
    project_id: str,
    session_id: str,
    sequence_id: int,
    user_query: str,
    answer: str,
    citations: list[str],
    todo_title: str | None = None,
) -> EpisodicMemoryItem:
    episode = EpisodicMemoryItem(
        id=make_id("ep"),
        project_id=project_id,
        session_id=session_id,
        sequence_id=sequence_id,
        event_type="research_run",
        summary=compact_text(f"{todo_title or user_query} -> {answer}", 220),
        details={
            "sequence_id": sequence_id,
            "query": user_query,
            "todo_title": todo_title or "",
            "answer_summary": compact_text(answer, 240),
            "citations": citations,
        },
        importance=0.75,
    )
    db.add(episode)
    db.flush()
    return episode


def preference_hints(query: str) -> list[str]:
    patterns = ["优先", "尽量", "希望", "不要", "最好", "必须"]
    return [pattern for pattern in patterns if pattern in query]


def semantic_fact_drafts(user_query: str, answer: str, citations: list[str], todo_title: str | None = None) -> list[dict[str, Any]]:
    answer_sentences = split_sentences(answer, limit=3)
    drafts = [
        {
            "fact_type": "fact",
            "memory_key": f"answer_{slugify(answer_sentences[0] if answer_sentences else user_query)}",
            "statement": compact_text(answer_sentences[0] if answer_sentences else answer, 220),
            "subject": "project",
            "predicate": "latest_fact",
            "object": compact_text(answer_sentences[0] if answer_sentences else answer, 120),
            "importance": 0.72,
            "metadata": {"citations": citations},
        },
        {
            "fact_type": "open_question",
            "memory_key": f"question_{slugify(user_query)}",
            "statement": compact_text(user_query, 220),
            "subject": "project",
            "predicate": "active_question",
            "object": compact_text(user_query, 120),
            "importance": 0.66,
            "metadata": {"todo_title": todo_title or ""},
        },
    ]
    if todo_title:
        drafts.append(
            {
                "fact_type": "progress",
                "memory_key": f"todo_{slugify(todo_title)}",
                "statement": compact_text(f"Completed or advanced TODO: {todo_title}", 220),
                "subject": "project",
                "predicate": "progress",
                "object": compact_text(todo_title, 120),
                "importance": 0.7,
                "metadata": {"query": compact_text(user_query, 160)},
            }
        )
    if answer_sentences:
        drafts.append(
            {
                "fact_type": "decision",
                "memory_key": f"decision_{slugify(answer_sentences[0])}",
                "statement": compact_text(answer_sentences[0], 220),
                "subject": "project",
                "predicate": "decision",
                "object": compact_text(answer_sentences[0], 120),
                "importance": 0.78,
                "metadata": {"citations": citations},
            }
        )
    for hint in preference_hints(user_query):
        drafts.append(
            {
                "fact_type": "preference",
                "memory_key": f"preference_{slugify(hint)}",
                "statement": compact_text(user_query, 220),
                "subject": "user",
                "predicate": "preference",
                "object": hint,
                "importance": 0.74,
                "metadata": {"hint": hint},
            }
        )
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for draft in drafts:
        unique[(draft["fact_type"], draft["memory_key"])] = draft
    return list(unique.values())


def upsert_semantic_fact(
    db: Session,
    project_id: str,
    session_id: str,
    sequence_id: int,
    draft: dict[str, Any],
) -> SemanticMemoryFact:
    fact = db.scalar(
        select(SemanticMemoryFact).where(
            SemanticMemoryFact.project_id == project_id,
            SemanticMemoryFact.fact_type == draft["fact_type"],
            SemanticMemoryFact.memory_key == draft["memory_key"],
        )
    )
    target = fact or SemanticMemoryFact(
        id=make_id("sm"),
        project_id=project_id,
        session_id=session_id,
        sequence_id=sequence_id,
        fact_type=draft["fact_type"],
        memory_key=draft["memory_key"],
        statement=draft["statement"],
        subject=draft["subject"],
        predicate=draft["predicate"],
        object=draft["object"],
        importance=draft["importance"],
        source="runtime_api",
        meta_payload=draft["metadata"],
    )
    target.session_id = session_id
    target.sequence_id = sequence_id
    target.statement = draft["statement"]
    target.subject = draft["subject"]
    target.predicate = draft["predicate"]
    target.object = draft["object"]
    target.importance = draft["importance"]
    target.meta_payload = draft["metadata"]
    db.add(target)
    db.flush()
    return target


def store_semantic_memories(
    db: Session,
    project_id: str,
    session_id: str,
    sequence_id: int,
    user_query: str,
    answer: str,
    citations: list[str],
    todo_title: str | None = None,
) -> list[SemanticMemoryFact]:
    stored = [
        upsert_semantic_fact(db, project_id, session_id, sequence_id, draft)
        for draft in semantic_fact_drafts(user_query, answer, citations, todo_title=todo_title)
    ]
    get_semantic_memory_store().upsert_facts(project_id, [semantic_fact_payload(item) for item in stored])
    return stored


@dataclass
class WorkingMemoryCompactionResult:
    compacted_items: list[WorkingMemoryItem]
    compacted_tokens: int = 0
    total_tokens: int = 0
    summary: str = ""
    episodic: list[EpisodicMemoryItem] | None = None
    semantic: list[SemanticMemoryFact] | None = None

    @property
    def episodic_items(self) -> list[EpisodicMemoryItem]:
        return self.episodic or []

    @property
    def semantic_items(self) -> list[SemanticMemoryFact]:
        return self.semantic or []


def select_working_memories_for_compaction(items: list[WorkingMemoryItem]) -> tuple[list[WorkingMemoryItem], int, int]:
    total_tokens = working_memory_token_total(items)
    threshold = max(1, settings.working_memory_token_threshold)
    if total_tokens <= threshold:
        return [], total_tokens, 0

    target_tokens = max(1, math.ceil(threshold * normalized_compaction_ratio()))
    selected: list[WorkingMemoryItem] = []
    selected_tokens = 0
    for item in items:
        selected.append(item)
        selected_tokens += working_memory_token_count(item)
        if selected_tokens >= target_tokens:
            break
    return selected, total_tokens, selected_tokens


def working_memory_payload(items: list[WorkingMemoryItem]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.id,
            "memory_key": item.memory_key,
            "content": item.content,
            "kind": item.meta_payload.get("kind", ""),
            "sequence_id": item.sequence_id,
            "importance": item.importance,
            "metadata": item.meta_payload,
        }
        for item in items
    ]


def llm_summarize_working_memories(
    project_id: str,
    session_id: str,
    sequence_id: int,
    compacted_items: list[WorkingMemoryItem],
    compacted_tokens: int,
    total_tokens: int,
) -> dict[str, Any]:
    payload = {
        "project_id": project_id,
        "session_id": session_id,
        "trigger_sequence_id": sequence_id,
        "compacted_tokens": compacted_tokens,
        "total_working_memory_tokens": total_tokens,
        "working_memories": working_memory_payload(compacted_items),
    }
    response = httpx.post(
        f"{settings.llm_api_base.rstrip('/')}/chat/completions",
        headers=llm_headers(),
        json={
            "model": settings.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是 Research Copilot 的长期记忆整理器。"
                        "你只能返回一个 JSON 对象，不要 Markdown，不要解释。"
                        "根据输入的 working memories，提炼值得长期保存的 episodic memory 和 semantic memory。"
                        "不要对每条候选执行 ADD/UPDATE/DELETE/NOOP 判断；只输出本批应新增保存的长期记忆。"
                        "忽略寒暄、临时表述、重复内容和没有长期价值的细节。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "输出 JSON schema：\n"
                        "{"
                        '"episodic_memories": ['
                        '{"event_type": "working_memory_summary", "summary": "事件摘要", '
                        '"details": {"key": "value"}, "importance": 0.7}'
                        "], "
                        '"semantic_memories": ['
                        '{"fact_type": "fact|decision|preference|open_question|progress", '
                        '"memory_key": "stable_key", "statement": "可复用事实", '
                        '"subject": "project|user", "predicate": "关系", "object": "宾语", '
                        '"importance": 0.7, "metadata": {"key": "value"}}'
                        "]"
                        "}\n\n"
                        "输入 JSON：\n"
                        f"{json.dumps(payload, ensure_ascii=False, default=str)}"
                    ),
                },
            ],
            "temperature": 0.1,
            "max_tokens": 1200,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    content = str(response.json()["choices"][0]["message"]["content"]).strip()
    data = json.loads(extract_json_object(content))
    if not isinstance(data, dict):
        raise ValueError("LLM memory summary must be a JSON object")
    return data


def coerce_importance(value: Any, default: float = 0.7) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return max(0.0, min(1.0, score))


def compaction_metadata(
    compacted_items: list[WorkingMemoryItem],
    compacted_tokens: int,
    total_tokens: int,
) -> dict[str, Any]:
    return {
        "source": "llm_working_memory_summary",
        "compacted_item_ids": [item.id for item in compacted_items],
        "compacted_memory_keys": [item.memory_key for item in compacted_items],
        "compacted_tokens": compacted_tokens,
        "total_working_memory_tokens": total_tokens,
        "compaction_ratio": normalized_compaction_ratio(),
    }


def store_llm_episodic_memories(
    db: Session,
    project_id: str,
    session_id: str,
    sequence_id: int,
    payload: dict[str, Any],
    compacted_items: list[WorkingMemoryItem],
    compacted_tokens: int,
    total_tokens: int,
) -> list[EpisodicMemoryItem]:
    base_metadata = compaction_metadata(compacted_items, compacted_tokens, total_tokens)
    memories = payload.get("episodic_memories") or []
    if not isinstance(memories, list):
        return []
    stored: list[EpisodicMemoryItem] = []
    for index, item in enumerate(memories, start=1):
        if not isinstance(item, dict):
            continue
        summary = compact_text(str(item.get("summary") or ""), 900)
        if not summary:
            continue
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        episode = EpisodicMemoryItem(
            id=make_id("ep"),
            project_id=project_id,
            session_id=session_id,
            sequence_id=sequence_id,
            event_type=compact_text(str(item.get("event_type") or "working_memory_summary"), 48),
            summary=summary,
            details={**base_metadata, **details, "llm_memory_index": index},
            importance=coerce_importance(item.get("importance"), 0.74),
        )
        db.add(episode)
        stored.append(episode)
    db.flush()
    return stored


def semantic_draft_from_llm_item(
    item: dict[str, Any],
    index: int,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    statement = compact_text(str(item.get("statement") or ""), 900)
    if not statement:
        return None
    fact_type = compact_text(str(item.get("fact_type") or "fact"), 32)
    if fact_type not in {"fact", "decision", "preference", "open_question", "progress"}:
        fact_type = "fact"
    raw_metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return {
        "fact_type": fact_type,
        "memory_key": compact_text(str(item.get("memory_key") or f"{fact_type}_{slugify(statement)}"), 64),
        "statement": statement,
        "subject": compact_text(str(item.get("subject") or "project"), 120),
        "predicate": compact_text(str(item.get("predicate") or "long_term_memory"), 120),
        "object": compact_text(str(item.get("object") or statement), 240),
        "importance": coerce_importance(item.get("importance"), 0.72),
        "metadata": {**metadata, **raw_metadata, "llm_memory_index": index},
    }


def store_llm_semantic_memories(
    db: Session,
    project_id: str,
    session_id: str,
    sequence_id: int,
    payload: dict[str, Any],
    compacted_items: list[WorkingMemoryItem],
    compacted_tokens: int,
    total_tokens: int,
) -> list[SemanticMemoryFact]:
    base_metadata = compaction_metadata(compacted_items, compacted_tokens, total_tokens)
    memories = payload.get("semantic_memories") or []
    if not isinstance(memories, list):
        return []
    drafts = [
        draft
        for index, item in enumerate(memories, start=1)
        if isinstance(item, dict)
        for draft in [semantic_draft_from_llm_item(item, index, base_metadata)]
        if draft is not None
    ]
    stored = [
        upsert_semantic_fact(db, project_id, session_id, sequence_id, draft)
        for draft in drafts
    ]
    get_semantic_memory_store().upsert_facts(project_id, [semantic_fact_payload(item) for item in stored])
    return stored


def compact_working_memory_if_needed(
    db: Session,
    project_id: str,
    session_id: str,
    sequence_id: int,
) -> WorkingMemoryCompactionResult:
    items = working_memories_chronological(db, project_id, session_id)
    selected, total_tokens, selected_tokens = select_working_memories_for_compaction(items)
    if not selected or not llm_memory_available():
        return WorkingMemoryCompactionResult([], total_tokens=total_tokens)

    payload = llm_summarize_working_memories(project_id, session_id, sequence_id, selected, selected_tokens, total_tokens)
    episodic = store_llm_episodic_memories(
        db,
        project_id,
        session_id,
        sequence_id,
        payload,
        selected,
        selected_tokens,
        total_tokens,
    )
    semantic = store_llm_semantic_memories(
        db,
        project_id,
        session_id,
        sequence_id,
        payload,
        selected,
        selected_tokens,
        total_tokens,
    )
    if not episodic and not semantic:
        return WorkingMemoryCompactionResult([], total_tokens=total_tokens)
    for item in selected:
        db.delete(item)
    db.flush()
    return WorkingMemoryCompactionResult(
        compacted_items=selected,
        compacted_tokens=selected_tokens,
        total_tokens=total_tokens,
        summary="llm_working_memory_summary",
        episodic=episodic,
        semantic=semantic,
    )


def consolidate_layered_memories(
    db: Session,
    project_id: str,
    session_id: str,
    sequence_id: int,
    user_query: str,
    answer: str,
    citations: list[str],
    todo_title: str | None = None,
) -> list[MemoryItem]:
    working = store_working_memory(db, project_id, session_id, sequence_id, user_query, answer, citations)
    return [
        MemoryItem(
            memory_type="working",
            key=item.memory_key,
            value=item.content,
            source=item.source,
            session_id=item.session_id,
            sequence_id=item.sequence_id,
            importance=item.importance,
            metadata=item.meta_payload,
        )
        for item in working
    ]


def consolidate_long_term_memories(
    db: Session,
    project_id: str,
    session_id: str,
    sequence_id: int,
) -> list[MemoryItem]:
    compaction = compact_working_memory_if_needed(db, project_id, session_id, sequence_id)
    updates = [
        MemoryItem(
            memory_type="episodic",
            key=item.event_type,
            value=item.summary,
            source=item.source,
            session_id=item.session_id,
            sequence_id=item.sequence_id,
            importance=item.importance,
            metadata=item.details,
        )
        for item in compaction.episodic_items
    ]
    updates.extend(
        MemoryItem(
            memory_type=f"semantic.{item.fact_type}",
            key=item.memory_key,
            value=item.statement,
            source=item.source,
            session_id=item.session_id,
            sequence_id=item.sequence_id,
            importance=item.importance,
            metadata=item.meta_payload,
        )
        for item in compaction.semantic_items
    )
    return updates
