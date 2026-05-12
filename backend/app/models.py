from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    session_count: int = 0
    todo_count: int = 0
    run_count: int = 0
    created_at: datetime
    updated_at: datetime


class ChatSessionBase(BaseModel):
    title: str = Field(default="新会话", min_length=1, max_length=160)
    summary: str = ""
    status: str = "active"


class ChatSessionCreate(BaseModel):
    title: str = Field(default="新会话", min_length=1, max_length=160)


class ChatSessionUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=160)
    summary: str | None = None
    status: str | None = None


class ChatSessionResponse(ChatSessionBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    last_sequence_id: int = 0
    turn_count: int = 0
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
    created_at: datetime
    updated_at: datetime


class ResumableUploadInitRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255)
    file_size: int = Field(..., ge=1)
    file_md5: str = Field(..., min_length=32, max_length=32)
    chunk_size: int = Field(..., ge=1)
    title: str | None = Field(default=None, max_length=120)
    asset_type: str | None = None


class ResumableUploadCompleteRequest(BaseModel):
    title: str | None = Field(default=None, max_length=120)
    asset_type: str | None = None


class ResumableUploadStatusResponse(BaseModel):
    upload_id: str
    file_md5: str
    filename: str
    file_size: int
    chunk_size: int
    total_chunks: int
    uploaded_chunks: list[int] = Field(default_factory=list)
    missing_chunks: list[int] = Field(default_factory=list)
    uploaded_count: int = 0
    complete: bool = False
    finalized: bool = False
    asset: AssetResponse | None = None


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
    id: str
    project_id: str
    memory_type: str
    memory_key: str
    memory_value: str
    source: str
    session_id: str | None = None
    sequence_id: int = 0
    importance: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime


class TurnScopedRequest(BaseModel):
    project_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    sequence_id: int = Field(..., ge=1)
    user_query: str = Field(..., min_length=1)
    asset_ids: list[str] = Field(default_factory=list)
    todo_id: str | None = None


class BuildContextResponse(BaseModel):
    project_id: str
    session_id: str
    sequence_id: int
    instruction_context: str
    session_context: str
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
    task_type: str = "reason"
    depends_on: list[str] = Field(default_factory=list)
    output_key: str = ""
    status: str = "pending"


class PlanExecutionStep(BaseModel):
    step_id: str
    task_id: str
    title: str
    action: str
    summary: str
    status: str = "completed"
    search_queries: list[str] = Field(default_factory=list)
    evidence_labels: list[str] = Field(default_factory=list)


class QueryRewriteVariant(BaseModel):
    strategy: str
    query: str
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: str = ""


class QueryRewriteResponse(BaseModel):
    original_query: str = ""
    standalone_query: str = ""
    intent: str = "general"
    subject: str = ""
    variants: list[QueryRewriteVariant] = Field(default_factory=list)
    generated_by: str = "deterministic_hybrid"


class PlanTasksResponse(BaseModel):
    project_id: str
    session_id: str
    sequence_id: int
    planner_mode: str = "two_stage"
    plan_summary: str = ""
    search_queries: list[str] = Field(default_factory=list)
    query_rewrite: QueryRewriteResponse = Field(default_factory=QueryRewriteResponse)
    tasks: list[ResearchTask]
    execution_trace: list[PlanExecutionStep] = Field(default_factory=list)
    solver_summary: str = ""
    replan_count: int = 0
    replan_reason: str = ""
    trace_tree: dict[str, Any] = Field(default_factory=dict)
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
    sequence_id: int
    retrieval_mode: str = "local_hybrid"
    evidence_items: list[EvidenceItem]
    generated_at: str = Field(default_factory=utc_now)


class AnswerWithCitationsRequest(TurnScopedRequest):
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    plan_summary: str = ""
    plan_tasks: list[ResearchTask] = Field(default_factory=list)
    execution_trace: list[PlanExecutionStep] = Field(default_factory=list)
    packed_context: str = ""


class Citation(BaseModel):
    asset_id: str
    chunk_id: str
    label: str
    score: float


class AnswerQualityReport(BaseModel):
    score: float = Field(default=0.0, ge=0.0, le=5.0)
    level: str = "unknown"
    evidence_count: int = 0
    citation_count: int = 0
    answer_length: int = 0
    grounded: bool = False
    fallback_used: bool = False
    signals: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class AnswerResponse(BaseModel):
    project_id: str
    session_id: str
    sequence_id: int
    answer: str
    citations: list[Citation]
    quality: AnswerQualityReport = Field(default_factory=AnswerQualityReport)
    generated_at: str = Field(default_factory=utc_now)


class MemoryItem(BaseModel):
    memory_type: str
    key: str
    value: str
    source: str
    session_id: str | None = None
    sequence_id: int = 0
    importance: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConsolidateMemoryRequest(TurnScopedRequest):
    answer: str = Field(..., min_length=1)
    citations: list[Citation] = Field(default_factory=list)


class ConsolidateMemoryResponse(BaseModel):
    project_id: str
    session_id: str
    sequence_id: int
    memory_updates: list[MemoryItem]
    generated_at: str = Field(default_factory=utc_now)


class RunResearchResponse(BaseModel):
    project_id: str
    session_id: str
    sequence_id: int
    context: BuildContextResponse
    plan: PlanTasksResponse
    retrieval: RetrieveResponse
    answer: AnswerResponse
    memory: ConsolidateMemoryResponse
    trace_id: str
    meta: dict[str, Any] = Field(default_factory=dict)


class ResearchTurnRequest(BaseModel):
    user_query: str = Field(..., min_length=1)
    sequence_id: int = Field(..., ge=1)
    todo_id: str | None = None
    asset_ids: list[str] = Field(default_factory=list)


class ResearchRunDetailResponse(BaseModel):
    id: str
    project_id: str
    session_id: str
    sequence_id: int
    todo_id: str | None = None
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
