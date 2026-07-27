from __future__ import annotations

import os

os.environ["DATABASE_URL"] = "sqlite:////tmp/research_copilot_memory_test.db"
os.environ["VECTOR_STORE_PROVIDER"] = "stub"
os.environ["EMBEDDING_PROVIDER"] = "stub"
os.environ["LLM_PROVIDER"] = "stub"
os.environ["LLM_API_KEY"] = ""

from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import Base, SessionLocal, engine  # noqa: E402
from app.db_models import ChatSession, EpisodicMemoryItem, Project, SemanticMemoryFact, WorkingMemoryItem  # noqa: E402
import app.memory_manager as memory_manager  # noqa: E402
from app.memory_manager import (  # noqa: E402
    build_memory_context_bundle,
    consolidate_layered_memories,
    consolidate_long_term_memories,
    working_memory_token_total,
)
from app.semantic_store import reset_semantic_memory_store  # noqa: E402
from app.vector_store import reset_vector_store  # noqa: E402


def reset_db() -> None:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    reset_vector_store()
    reset_semantic_memory_store()


def seed_project_session() -> tuple[str, str]:
    project_id = "proj-memory"
    session_id = "sess-memory"
    with SessionLocal() as db:
        db.add(Project(id=project_id, title="Memory", description="", status="active"))
        db.add(ChatSession(id=session_id, project_id=project_id, title="Session"))
        db.commit()
    return project_id, session_id


def test_memory_consolidation_keeps_short_working_memory_without_long_term(monkeypatch) -> None:
    reset_db()
    project_id, session_id = seed_project_session()
    monkeypatch.setattr(settings, "working_memory_token_threshold", 10000)

    with SessionLocal() as db:
        updates = consolidate_layered_memories(
            db,
            project_id,
            session_id,
            1,
            "remember the retrieval plan",
            "hybrid retrieval uses dense search, BM25, and rerank.",
            [],
        )
        db.commit()

        working = db.scalars(select(WorkingMemoryItem)).all()
        episodic = db.scalars(select(EpisodicMemoryItem)).all()
        semantic = db.scalars(select(SemanticMemoryFact)).all()
        bundle = build_memory_context_bundle(db, project_id, session_id, "retrieval plan")

    assert [item.memory_type for item in updates] == ["working", "working"]
    assert len(working) == 2
    assert episodic == []
    assert semantic == []
    assert "turn_1_query" in bundle.working_text
    assert "turn_1_answer" in bundle.working_text


def test_memory_consolidation_skips_long_term_when_llm_unavailable(monkeypatch) -> None:
    reset_db()
    project_id, session_id = seed_project_session()
    monkeypatch.setattr(settings, "working_memory_token_threshold", 40)

    with SessionLocal() as db:
        consolidate_layered_memories(
            db,
            project_id,
            session_id,
            1,
            "first research question",
            " ".join(f"alpha{i}" for i in range(18)),
            [],
        )
        consolidate_layered_memories(
            db,
            project_id,
            session_id,
            2,
            "second research question",
            " ".join(f"beta{i}" for i in range(70)),
            [],
        )
        db.commit()

        working = db.scalars(select(WorkingMemoryItem)).all()
        episodic = db.scalars(select(EpisodicMemoryItem)).all()
        semantic = db.scalars(select(SemanticMemoryFact)).all()

    assert working
    assert working_memory_token_total(working) > settings.working_memory_token_threshold
    assert episodic == []
    assert semantic == []
    assert any(item.memory_key == "turn_1_query" for item in working)


def test_memory_consolidation_uses_llm_summary_for_long_term_memory(monkeypatch) -> None:
    reset_db()
    project_id, session_id = seed_project_session()
    captured: dict[str, object] = {}
    monkeypatch.setattr(settings, "working_memory_token_threshold", 40)
    monkeypatch.setattr(settings, "working_memory_compaction_ratio", 0.75)
    monkeypatch.setattr(memory_manager, "llm_memory_available", lambda: True)

    def fake_llm_summary(project_id, session_id, sequence_id, compacted_items, compacted_tokens, total_tokens):
        captured["memory_keys"] = [item.memory_key for item in compacted_items]
        captured["compacted_tokens"] = compacted_tokens
        return {
            "episodic_memories": [
                {
                    "event_type": "working_memory_summary",
                    "summary": "用户围绕检索方案连续追问，系统形成了混合检索设计结论。",
                    "details": {"topic": "retrieval"},
                    "importance": 0.82,
                }
            ],
            "semantic_memories": [
                {
                    "fact_type": "decision",
                    "memory_key": "retrieval_design_decision",
                    "statement": "项目采用 dense search、BM25 和 rerank 的混合检索设计。",
                    "subject": "project",
                    "predicate": "uses",
                    "object": "hybrid retrieval",
                    "importance": 0.86,
                    "metadata": {"topic": "retrieval"},
                }
            ],
        }

    monkeypatch.setattr(memory_manager, "llm_summarize_working_memories", fake_llm_summary)

    with SessionLocal() as db:
        consolidate_layered_memories(
            db,
            project_id,
            session_id,
            1,
            "first research question",
            " ".join(f"alpha{i}" for i in range(18)),
            [],
        )
        updates = consolidate_layered_memories(
            db,
            project_id,
            session_id,
            2,
            "second research question",
            " ".join(f"beta{i}" for i in range(16)),
            [],
        )
        updates = consolidate_long_term_memories(db, project_id, session_id, 2)
        db.commit()

        working = db.scalars(select(WorkingMemoryItem)).all()
        episodic = db.scalars(select(EpisodicMemoryItem)).all()
        semantic = db.scalars(select(SemanticMemoryFact)).all()

    assert captured["compacted_tokens"] >= int(settings.working_memory_token_threshold * 0.75)
    assert "turn_1_query" in captured["memory_keys"]
    assert any(item.memory_type == "episodic" for item in updates)
    assert any(item.memory_type == "semantic.decision" for item in updates)
    assert any(item.event_type == "working_memory_summary" for item in episodic)
    assert any(item.memory_key == "retrieval_design_decision" for item in semantic)
    assert not any(item.memory_key == "turn_1_query" for item in working)
