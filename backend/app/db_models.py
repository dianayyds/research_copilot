from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, JSON, String, Text, UniqueConstraint
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="active", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    sessions: Mapped[list["ChatSession"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    todos: Mapped[list["Todo"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    runs: Mapped[list["ResearchRun"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    working_memories: Mapped[list["WorkingMemoryItem"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    episodic_memories: Mapped[list["EpisodicMemoryItem"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    semantic_memories: Mapped[list["SemanticMemoryFact"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(160), default="新会话", nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="active", nullable=False)
    last_sequence_id: Mapped[int] = mapped_column(default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    project: Mapped[Project] = relationship(back_populates="sessions")
    runs: Mapped[list["ResearchRun"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    asset_type: Mapped[str] = mapped_column(String(24), default="note", nullable=False)
    content: Mapped[str] = mapped_column(Text().with_variant(LONGTEXT(), "mysql"), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )


class Todo(Base):
    __tablename__ = "todos"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="todo", nullable=False)
    priority: Mapped[str] = mapped_column(String(16), default="medium", nullable=False)
    last_run_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    project: Mapped[Project] = relationship(back_populates="todos")
    runs: Mapped[list["ResearchRun"]] = relationship(back_populates="todo")


class ResearchRun(Base):
    __tablename__ = "research_runs"
    __table_args__ = (UniqueConstraint("session_id", "sequence_id", name="uq_research_runs_session_sequence"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    todo_id: Mapped[str | None] = mapped_column(ForeignKey("todos.id", ondelete="SET NULL"), nullable=True)
    sequence_id: Mapped[int] = mapped_column(nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="completed", nullable=False)
    trace_id: Mapped[str] = mapped_column(String(48), nullable=False)
    answer_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    context_payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    plan_payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    retrieval_payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    answer_payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    memory_payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    project: Mapped[Project] = relationship(back_populates="runs")
    session: Mapped[ChatSession] = relationship(back_populates="runs")
    todo: Mapped[Todo | None] = relationship(back_populates="runs")


class WorkingMemoryItem(Base):
    __tablename__ = "working_memory_items"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    session_id: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    sequence_id: Mapped[int] = mapped_column(default=0, nullable=False)
    memory_key: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    importance: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    source: Mapped[str] = mapped_column(String(32), default="runtime_api", nullable=False)
    meta_payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    project: Mapped[Project] = relationship(back_populates="working_memories")


class EpisodicMemoryItem(Base):
    __tablename__ = "episodic_memory_items"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    session_id: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    sequence_id: Mapped[int] = mapped_column(default=0, nullable=False)
    event_type: Mapped[str] = mapped_column(String(48), default="research_run", nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    importance: Mapped[float] = mapped_column(Float, default=0.7, nullable=False)
    source: Mapped[str] = mapped_column(String(32), default="runtime_api", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    project: Mapped[Project] = relationship(back_populates="episodic_memories")


class SemanticMemoryFact(Base):
    __tablename__ = "semantic_memory_facts"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    session_id: Mapped[str | None] = mapped_column(String(48), nullable=True, index=True)
    sequence_id: Mapped[int] = mapped_column(default=0, nullable=False)
    fact_type: Mapped[str] = mapped_column(String(32), default="fact", nullable=False)
    memory_key: Mapped[str] = mapped_column(String(64), nullable=False)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    subject: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    predicate: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    object: Mapped[str] = mapped_column(String(240), default="", nullable=False)
    importance: Mapped[float] = mapped_column(Float, default=0.7, nullable=False)
    source: Mapped[str] = mapped_column(String(32), default="runtime_api", nullable=False)
    meta_payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    project: Mapped[Project] = relationship(back_populates="semantic_memories")
