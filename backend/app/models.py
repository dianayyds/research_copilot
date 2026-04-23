from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProviderInfo(BaseModel):
    provider: str
    model: str
    mode: str


class ProviderConfigResponse(BaseModel):
    llm: ProviderInfo
    embedding: ProviderInfo
    reranker: ProviderInfo | None = None
    execution_mode: str


class ProjectBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)
    description: str = ""
    status: str = "active"


class ProjectCreate(ProjectBase):
    pass


class ProjectUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = None
    status: str | None = None


class ProjectResponse(ProjectBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    asset_count: int = 0
    todo_count: int = 0
    run_count: int = 0
    created_at: datetime
    updated_at: datetime


class AssetBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)
    asset_type: str = "note"
    content: str = Field(..., min_length=1)


class AssetCreate(AssetBase):
    pass


class AssetUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    asset_type: str | None = None
    content: str | None = Field(default=None, min_length=1)


class AssetResponse(AssetBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    created_at: datetime
    updated_at: datetime


class TodoBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=160)
    description: str = ""
    status: str = "todo"
    priority: str = "medium"


class TodoCreate(TodoBase):
    pass


class TodoUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = None
    status: str | None = None
    priority: str | None = None


class TodoResponse(TodoBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    last_run_id: str | None = None
    created_at: datetime
    updated_at: datetime


class MemoryRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    memory_type: str
    memory_key: str
    memory_value: str
    source: str
    session_id: str | None = None
    importance: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime


class ProjectScopedRequest(BaseModel):
    project_id: str = Field(..., min_length=1)
    user_query: str = Field(..., min_length=1)
    asset_ids: list[str] = Field(default_factory=list)
    session_id: str | None = None
    todo_id: str | None = None


class BuildContextResponse(BaseModel):
    project_id: str
    session_id: str
    instruction_context: str
    evidence_context: str
    task_state_context: str
    memory_context: str
    working_memory_context: str = ""
    episodic_memory_context: str = ""
    semantic_memory_context: str = ""
    packed_context: str
    generated_at: str = Field(default_factory=utc_now)


class ResearchTask(BaseModel):
    task_id: str
    title: str
    goal: str
    status: str = "pending"


class PlanTasksResponse(BaseModel):
    project_id: str
    session_id: str
    tasks: list[ResearchTask]
    generated_at: str = Field(default_factory=utc_now)


class EvidenceItem(BaseModel):
    asset_id: str
    chunk_id: str
    label: str
    title: str
    snippet: str
    source_path: str
    score: float
    tags: list[str] = Field(default_factory=list)


class RetrieveResponse(BaseModel):
    project_id: str
    session_id: str
    retrieval_mode: str = "local_hybrid"
    evidence_items: list[EvidenceItem]
    generated_at: str = Field(default_factory=utc_now)


class AnswerWithCitationsRequest(ProjectScopedRequest):
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    packed_context: str = ""


class Citation(BaseModel):
    asset_id: str
    chunk_id: str
    label: str
    score: float


class AnswerResponse(BaseModel):
    project_id: str
    session_id: str
    answer: str
    citations: list[Citation]
    generated_at: str = Field(default_factory=utc_now)


class MemoryItem(BaseModel):
    memory_type: str
    key: str
    value: str
    source: str
    session_id: str | None = None
    importance: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConsolidateMemoryRequest(ProjectScopedRequest):
    answer: str = Field(..., min_length=1)
    citations: list[Citation] = Field(default_factory=list)


class ConsolidateMemoryResponse(BaseModel):
    project_id: str
    session_id: str
    memory_updates: list[MemoryItem]
    generated_at: str = Field(default_factory=utc_now)


class RunResearchResponse(BaseModel):
    project_id: str
    session_id: str
    context: BuildContextResponse
    plan: PlanTasksResponse
    retrieval: RetrieveResponse
    answer: AnswerResponse
    memory: ConsolidateMemoryResponse
    trace_id: str
    meta: dict[str, Any] = Field(default_factory=dict)


class ResearchRunRequest(BaseModel):
    user_query: str = Field(..., min_length=1)
    todo_id: str | None = None
    session_id: str | None = None
    asset_ids: list[str] = Field(default_factory=list)


class ResearchRunSummaryResponse(BaseModel):
    id: str
    project_id: str
    todo_id: str | None = None
    session_id: str
    query: str
    status: str
    answer_preview: str
    trace_id: str
    created_at: datetime
    updated_at: datetime


class ResearchRunDetailResponse(BaseModel):
    id: str
    project_id: str
    todo_id: str | None = None
    session_id: str
    query: str
    status: str
    trace_id: str
    answer_text: str
    created_at: datetime
    updated_at: datetime
    context: BuildContextResponse
    plan: PlanTasksResponse
    retrieval: RetrieveResponse
    answer: AnswerResponse
    memory: ConsolidateMemoryResponse


class DashboardResponse(BaseModel):
    project_count: int
    todo_count: int
    open_todo_count: int
    run_count: int
    latest_runs: list[ResearchRunSummaryResponse]
