from __future__ import annotations

import ast
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from time import perf_counter
from typing import Callable, Iterator

import httpx
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db_models import Asset, ChatSession, Project, ResearchRun, Todo, WorkingMemoryItem
from app.live_tools import LiveToolResult, LiveToolRoute, execute_live_tool, select_live_tool
from app.memory_manager import build_memory_context_bundle, consolidate_layered_memories, list_layered_memories
from app.models import (
    AnswerResponse,
    AnswerWithCitationsRequest,
    AssetCreate,
    AssetResponse,
    AssetUpdate,
    BuildContextResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    ChatSessionUpdate,
    Citation,
    ConsolidateMemoryRequest,
    ConsolidateMemoryResponse,
    EvidenceItem,
    MemoryRecordResponse,
    PlanExecutionStep,
    PlanTasksResponse,
    ProjectCreate,
    ProjectResponse,
    ProjectUpdate,
    ResearchRunDetailResponse,
    ResearchTask,
    ResearchTurnRequest,
    RetrieveResponse,
    RunResearchResponse,
    TodoCreate,
    TodoResponse,
    TodoUpdate,
    TurnScopedRequest,
)
from app.semantic_store import get_semantic_memory_store
from app.vector_store import get_vector_store


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
TITLE_QUERY_RE = re.compile(r"[《“\"]([^》”\"]+)[》”\"]")
LOOKUP_TERMS = ("谁", "作者", "第一作者", "通讯作者", "年份", "哪一年", "单位", "机构", "在哪里")
logger = logging.getLogger("uvicorn.error")


@dataclass
class ChunkRecord:
    asset_id: str
    chunk_id: str
    title: str
    asset_type: str
    content: str
    source_path: str


@dataclass(frozen=True)
class ToolPlannerDecision:
    route: LiveToolRoute | None
    planner_mode: str
    reason: str
    fallback_reason: str = ""


@dataclass(frozen=True)
class AgentToolSpec:
    name: str
    description: str
    input_schema: dict[str, object]


@dataclass(frozen=True)
class AgentDecision:
    thought: str
    action: str
    arguments: dict[str, object]


@dataclass
class AgentToolObservation:
    tool_name: str
    summary: str
    content: str
    evidence_items: list[EvidenceItem]
    metadata: dict[str, object]


def make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def compact_text(text: str, limit: int = 80) -> str:
    return " ".join(text.split())[:limit].strip() or "未命名"


def slugify(text: str) -> str:
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "-", text.strip().lower())
    return normalized.strip("-")[:48] or "chunk"


def split_chunks(text: str) -> list[str]:
    paragraphs = [item.strip() for item in text.splitlines() if item.strip()]
    joined = paragraphs or [text.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for paragraph in joined:
        next_length = current_length + len(paragraph)
        if current and next_length > settings.asset_chunk_size:
            chunks.append("\n".join(current))
            current = [paragraph]
            current_length = len(paragraph)
            continue
        current.append(paragraph)
        current_length = next_length
    if current:
        chunks.append("\n".join(current))
    return chunks


def markdown_sections(text: str) -> list[tuple[str, str]]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    sections: list[tuple[str, str]] = []
    heading_stack: dict[int, str] = {}
    current_heading = "正文"
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_heading, body))

    for raw_line in lines:
        stripped = raw_line.strip()
        match = HEADING_RE.match(stripped)
        if match:
            flush()
            current_lines = []
            level = len(match.group(1))
            heading_stack[level] = match.group(2).strip()
            for key in list(heading_stack):
                if key > level:
                    del heading_stack[key]
            current_heading = " / ".join(heading_stack[index] for index in sorted(heading_stack))
            continue
        current_lines.append(raw_line)
    flush()
    return sections or [("正文", text.strip())]


def project_to_response(db: Session, project: Project) -> ProjectResponse:
    return ProjectResponse(
        id=project.id,
        title=project.title,
        description=project.description,
        status=project.status,
        session_count=db.query(ChatSession).filter(ChatSession.project_id == project.id).count(),
        todo_count=db.query(Todo).filter(Todo.project_id == project.id).count(),
        run_count=db.query(ResearchRun).filter(ResearchRun.project_id == project.id).count(),
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


def session_to_response(db: Session, session: ChatSession) -> ChatSessionResponse:
    return ChatSessionResponse(
        id=session.id,
        project_id=session.project_id,
        title=session.title,
        summary=session.summary,
        status=session.status,
        last_sequence_id=session.last_sequence_id,
        turn_count=db.query(ResearchRun).filter(ResearchRun.session_id == session.id).count(),
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def todo_to_response(todo: Todo) -> TodoResponse:
    return TodoResponse.model_validate(todo)


def asset_to_response(asset: Asset) -> AssetResponse:
    return AssetResponse.model_validate(asset)


def decode_text_content(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Uploaded file must be a UTF-8 or GB18030 text file")


def uploaded_asset_title(filename: str, title: str | None = None) -> str:
    return (title or filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]).strip()


def touch_project(project: Project) -> None:
    project.updated_at = utc_now()


def touch_session(session: ChatSession) -> None:
    session.updated_at = utc_now()


def get_project(db: Session, project_id: str) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError("Project not found")
    return project


def get_chat_session(db: Session, project_id: str, session_id: str) -> ChatSession:
    session = db.scalar(select(ChatSession).where(ChatSession.id == session_id, ChatSession.project_id == project_id))
    if session is None:
        raise ValueError("Session not found in project")
    return session


def get_todo(db: Session, todo_id: str) -> Todo:
    todo = db.get(Todo, todo_id)
    if todo is None:
        raise ValueError("Todo not found")
    return todo


def get_project_todo(db: Session, project_id: str, todo_id: str) -> Todo:
    todo = db.scalar(select(Todo).where(Todo.id == todo_id, Todo.project_id == project_id))
    if todo is None:
        raise ValueError("Todo not found in project")
    return todo


def ensure_next_sequence(session: ChatSession, sequence_id: int) -> None:
    expected = session.last_sequence_id + 1
    if sequence_id != expected:
        raise ValueError(f"Sequence id must be {expected} for this session")


def list_projects(db: Session) -> list[ProjectResponse]:
    projects = db.scalars(select(Project).order_by(desc(Project.updated_at))).all()
    return [project_to_response(db, project) for project in projects]


def create_project(db: Session, payload: ProjectCreate) -> ProjectResponse:
    project = Project(
        id=make_id("proj"),
        title=payload.title,
        description=payload.description,
        status=payload.status,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project_to_response(db, project)


def update_project(db: Session, project_id: str, payload: ProjectUpdate) -> ProjectResponse:
    project = get_project(db, project_id)
    for key, value in payload.model_dump(exclude_none=True).items():
        setattr(project, key, value)
    db.commit()
    db.refresh(project)
    return project_to_response(db, project)


def delete_project(db: Session, project_id: str) -> None:
    project = get_project(db, project_id)
    get_semantic_memory_store().delete_project(project_id)
    db.delete(project)
    db.commit()


def list_sessions(db: Session, project_id: str) -> list[ChatSessionResponse]:
    get_project(db, project_id)
    sessions = db.scalars(
        select(ChatSession).where(ChatSession.project_id == project_id).order_by(desc(ChatSession.updated_at))
    ).all()
    return [session_to_response(db, session) for session in sessions]


def create_session(db: Session, project_id: str, payload: ChatSessionCreate) -> ChatSessionResponse:
    project = get_project(db, project_id)
    session = ChatSession(
        id=make_id("sess"),
        project_id=project_id,
        title=payload.title,
    )
    db.add(session)
    touch_project(project)
    db.commit()
    db.refresh(session)
    return session_to_response(db, session)


def update_session(db: Session, session_id: str, payload: ChatSessionUpdate) -> ChatSessionResponse:
    session = db.get(ChatSession, session_id)
    if session is None:
        raise ValueError("Session not found")
    for key, value in payload.model_dump(exclude_none=True).items():
        setattr(session, key, value)
    touch_project(get_project(db, session.project_id))
    db.commit()
    db.refresh(session)
    return session_to_response(db, session)


def delete_session(db: Session, project_id: str, session_id: str) -> None:
    session = get_chat_session(db, project_id, session_id)
    run_ids = [run.id for run in session.runs]
    if run_ids:
        todos = db.scalars(select(Todo).where(Todo.project_id == project_id, Todo.last_run_id.in_(run_ids))).all()
        for todo in todos:
            todo.last_run_id = None
    memories = db.scalars(
        select(WorkingMemoryItem).where(
            WorkingMemoryItem.project_id == project_id,
            WorkingMemoryItem.session_id == session_id,
        )
    ).all()
    for item in memories:
        db.delete(item)
    project = get_project(db, project_id)
    db.delete(session)
    touch_project(project)
    db.commit()


def list_assets(db: Session) -> list[AssetResponse]:
    assets = db.scalars(select(Asset).order_by(desc(Asset.updated_at))).all()
    return [asset_to_response(asset) for asset in assets]


def build_chunks(assets: list[Asset], asset_ids: list[str]) -> list[ChunkRecord]:
    allowed = {asset_id for asset_id in asset_ids if asset_id}
    selected_assets = [asset for asset in assets if not allowed or asset.id in allowed]
    chunks: list[ChunkRecord] = []
    for asset in selected_assets:
        section_index = 0
        for heading, body in markdown_sections(asset.content):
            for chunk_index, chunk_text in enumerate(split_chunks(body), start=1):
                section_index += 1
                heading_slug = slugify(heading)
                source_path = f"/assets/{asset.id}#{heading_slug}" if heading != "正文" else f"/assets/{asset.id}"
                display_title = asset.title if heading == "正文" else f"{asset.title} · {heading}"
                chunks.append(
                    ChunkRecord(
                        asset_id=asset.id,
                        chunk_id=f"{asset.id}-{heading_slug}-{section_index:03d}-{chunk_index:03d}",
                        title=display_title,
                        asset_type=asset.asset_type,
                        content=chunk_text,
                        source_path=source_path,
                    )
                )
    return chunks


def chunk_payloads(assets: list[Asset], asset_ids: list[str]) -> list[dict[str, str]]:
    return [
        {
            "asset_id": chunk.asset_id,
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "asset_type": chunk.asset_type,
            "content": chunk.content,
            "source_path": chunk.source_path,
        }
        for chunk in build_chunks(assets, asset_ids)
    ]


def sync_asset_chunks(assets: list[Asset], asset_ids: list[str]) -> int:
    payloads = chunk_payloads(assets, asset_ids)
    if payloads:
        started_at = perf_counter()
        get_vector_store().upsert_chunks(payloads)
        logger.info(
            "asset_chunk_sync_completed asset_ids=%s chunk_count=%s elapsed_ms=%.1f",
            ",".join(asset_ids),
            len(payloads),
            (perf_counter() - started_at) * 1000,
        )
    else:
        logger.info("asset_chunk_sync_skipped asset_ids=%s chunk_count=0", ",".join(asset_ids))
    return len(payloads)


def create_asset(db: Session, payload: AssetCreate) -> AssetResponse:
    asset = Asset(
        id=make_id("asset"),
        title=payload.title,
        asset_type=payload.asset_type,
        content=payload.content,
    )
    db.add(asset)
    db.flush()
    sync_asset_chunks([asset], [asset.id])
    db.commit()
    db.refresh(asset)
    return asset_to_response(asset)


def update_asset(db: Session, asset_id: str, payload: AssetUpdate) -> AssetResponse:
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise ValueError("Asset not found")
    for key, value in payload.model_dump(exclude_none=True).items():
        setattr(asset, key, value)
    db.flush()
    get_vector_store().delete_asset(asset.id)
    sync_asset_chunks([asset], [asset.id])
    db.commit()
    db.refresh(asset)
    return asset_to_response(asset)


def delete_asset(db: Session, asset_id: str) -> None:
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise ValueError("Asset not found")
    get_vector_store().delete_asset(asset.id)
    db.delete(asset)
    db.commit()


def list_todos(db: Session, project_id: str) -> list[TodoResponse]:
    get_project(db, project_id)
    todos = db.scalars(select(Todo).where(Todo.project_id == project_id).order_by(desc(Todo.updated_at))).all()
    return [todo_to_response(todo) for todo in todos]


def create_todo(db: Session, project_id: str, payload: TodoCreate) -> TodoResponse:
    project = get_project(db, project_id)
    todo = Todo(
        id=make_id("todo"),
        project_id=project_id,
        title=payload.title,
        description=payload.description,
        status=payload.status,
        priority=payload.priority,
    )
    db.add(todo)
    touch_project(project)
    db.commit()
    db.refresh(todo)
    return todo_to_response(todo)


def update_todo(db: Session, todo_id: str, payload: TodoUpdate) -> TodoResponse:
    todo = get_todo(db, todo_id)
    for key, value in payload.model_dump(exclude_none=True).items():
        setattr(todo, key, value)
    touch_project(get_project(db, todo.project_id))
    db.commit()
    db.refresh(todo)
    return todo_to_response(todo)


def delete_todo(db: Session, todo_id: str) -> None:
    todo = get_todo(db, todo_id)
    project = get_project(db, todo.project_id)
    db.delete(todo)
    touch_project(project)
    db.commit()


def list_memory(db: Session, project_id: str) -> list[MemoryRecordResponse]:
    get_project(db, project_id)
    return list_layered_memories(db, project_id)


def run_to_detail(run: ResearchRun) -> ResearchRunDetailResponse:
    return ResearchRunDetailResponse(
        id=run.id,
        project_id=run.project_id,
        session_id=run.session_id,
        sequence_id=run.sequence_id,
        todo_id=run.todo_id,
        query=run.query,
        status=run.status,
        trace_id=run.trace_id,
        answer_text=run.answer_text,
        created_at=run.created_at,
        updated_at=run.updated_at,
        context=BuildContextResponse(**run.context_payload),
        plan=PlanTasksResponse(**run.plan_payload),
        retrieval=RetrieveResponse(**run.retrieval_payload),
        answer=AnswerResponse(**run.answer_payload),
        memory=ConsolidateMemoryResponse(**run.memory_payload),
    )


def list_session_runs(db: Session, project_id: str, session_id: str) -> list[ResearchRunDetailResponse]:
    get_chat_session(db, project_id, session_id)
    runs = db.scalars(
        select(ResearchRun)
        .where(ResearchRun.project_id == project_id, ResearchRun.session_id == session_id)
        .order_by(ResearchRun.sequence_id)
    ).all()
    return [run_to_detail(run) for run in runs]


def get_run(db: Session, run_id: str) -> ResearchRunDetailResponse:
    run = db.get(ResearchRun, run_id)
    if run is None:
        raise ValueError("Run not found")
    return run_to_detail(run)


def recent_session_context(db: Session, project_id: str, session_id: str) -> str:
    runs = db.scalars(
        select(ResearchRun)
        .where(ResearchRun.project_id == project_id, ResearchRun.session_id == session_id)
        .order_by(desc(ResearchRun.sequence_id))
        .limit(4)
    ).all()
    if not runs:
        return "- no prior turns yet"
    lines: list[str] = []
    for run in reversed(runs):
        lines.append(f"- 用户#{run.sequence_id}: {compact_text(run.query, 120)}")
        lines.append(f"  助手#{run.sequence_id}: {compact_text(run.answer_text, 160)}")
    return "\n".join(lines)


def context_excerpt(text: str, empty_text: str, limit: int = 1200) -> str:
    compact = text.strip()
    if not compact:
        return empty_text
    return compact[:limit].strip()


def asset_scope_summary(assets: list[Asset], limit: int = 8) -> str:
    if not assets:
        return "- no assets yet"
    return "\n".join(
        f"- {asset.title} ({asset.asset_type})"
        for asset in assets[:limit]
    )


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = " ".join(value.split()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def quoted_title(text: str) -> str:
    match = TITLE_QUERY_RE.search(text)
    return match.group(1).strip() if match else ""


def is_lookup_query(text: str) -> bool:
    return any(term in text for term in LOOKUP_TERMS)


def planned_search_queries(
    query: str,
    subject: str,
    *,
    todo_description: str = "",
    broaden: bool = False,
) -> list[str]:
    focus = quoted_title(query) or compact_text(subject, 96)
    normalized_query = compact_text(re.sub(r"[《》“”\"'，。！？?、]", " ", query), 96)
    suffixes = ["作者", "标题页", "摘要"] if is_lookup_query(query) else ["关键概念", "结论 依据", "摘要"]
    if broaden:
        suffixes = dedupe_preserve_order(suffixes + ["第一页", "全文", "介绍"])
    return dedupe_preserve_order(
        [
            query,
            focus,
            normalized_query,
            *(f"{focus} {suffix}" for suffix in suffixes),
            todo_description,
        ]
    )


def merge_retrieval_hits(hit_groups: list[tuple[str, list[dict[str, object]]]], limit: int) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for query_index, (query, hits) in enumerate(hit_groups, start=1):
        for rank, hit in enumerate(hits, start=1):
            chunk_id = str(hit["chunk_id"])
            boosted_score = float(hit["score"]) + (0.02 / query_index) + (0.01 / rank)
            current = merged.get(chunk_id)
            if current is None or boosted_score > float(current["score"]):
                merged[chunk_id] = {
                    **hit,
                    "score": boosted_score,
                    "matched_query": query,
                }
    ranked = sorted(merged.values(), key=lambda item: float(item["score"]), reverse=True)
    return ranked[:limit]


def plan_text(tasks: list[ResearchTask]) -> str:
    lines = []
    for index, task in enumerate(tasks, start=1):
        deps = f" | deps={','.join(task.depends_on)}" if task.depends_on else ""
        output = f" | output={task.output_key}" if task.output_key else ""
        lines.append(f"{index}. {task.title}: {task.goal}{deps}{output}")
    return "\n".join(lines)


def execution_trace_text(steps: list[PlanExecutionStep]) -> str:
    return "\n".join(
        f"{index}. [{step.action}] {step.title}: {step.summary}"
        for index, step in enumerate(steps, start=1)
    )


def evidence_labels(retrieval: RetrieveResponse) -> list[str]:
    return [item.label for item in retrieval.evidence_items]


def retrieval_strength(retrieval: RetrieveResponse) -> float:
    top_score = retrieval.evidence_items[0].score if retrieval.evidence_items else 0.0
    return top_score + (0.05 * min(len(retrieval.evidence_items), settings.retrieval_limit))


def should_replan(query: str, retrieval: RetrieveResponse) -> bool:
    if not retrieval.evidence_items:
        return True
    threshold = 0.28 if is_lookup_query(query) else 0.18
    return retrieval.evidence_items[0].score < threshold


def mark_tasks_completed(tasks: list[ResearchTask]) -> list[ResearchTask]:
    return [task.model_copy(update={"status": "completed"}) for task in tasks]


def build_context(
    db: Session,
    request: TurnScopedRequest,
    *,
    project: Project,
    session: ChatSession,
    assets: list[Asset],
    todo: Todo | None,
) -> BuildContextResponse:
    memory_bundle = build_memory_context_bundle(db, project.id, session.id, request.user_query)
    session_context = context_excerpt(
        recent_session_context(db, project.id, session.id),
        "- no prior turns yet",
        limit=900,
    )
    working_context = context_excerpt(memory_bundle.working_text, "- no working memory yet", limit=1000)
    episodic_context = context_excerpt(memory_bundle.episodic_text, "- no episodic memory yet", limit=1200)
    semantic_context = context_excerpt(memory_bundle.semantic_text, "- no semantic memory yet", limit=1200)
    asset_scope = asset_scope_summary(assets)
    todo_line = todo.title if todo else "未指定 TODO"
    task_goal = todo.description if todo and todo.description else request.user_query
    packed_context = "\n\n".join(
        [
            "## System Rules",
            "你是一个本地知识型 Research Copilot。遵循 plan-and-solve：先定义目标，再检索证据，再结合记忆求解。",
            "只能使用当前项目记忆和全局知识资产回答，不要编造引用。",
            "## Task Goal",
            f"项目：{project.title}\n任务目标：{task_goal}\n当前 TODO：{todo_line}\n轮次：{request.sequence_id}",
            "## Session Working Set",
            session_context,
            "## Working Memory",
            working_context,
            "## Episodic Memory",
            episodic_context,
            "## Semantic Memory",
            semantic_context,
            "## Knowledge Scope",
            f"当前全局资产数：{len(assets)}\n{asset_scope}",
            "## Current Query",
            request.user_query,
        ]
    )
    return BuildContextResponse(
        project_id=project.id,
        session_id=session.id,
        sequence_id=request.sequence_id,
        instruction_context="plan-and-solve：先定义目标，再检索证据，再结合记忆给出带引用答案。",
        session_context=session_context,
        evidence_context=f"Global assets available: {len(assets)}\n{asset_scope}",
        task_state_context=f"Task goal: {task_goal}\nTODO: {todo_line}\nUser query: {request.user_query}",
        memory_context="\n\n".join(
            [
                "Working Memory",
                working_context,
                "Episodic Memory",
                episodic_context,
                "Semantic Memory",
                semantic_context,
            ]
        ),
        working_memory_context=working_context,
        episodic_memory_context=episodic_context,
        semantic_memory_context=semantic_context,
        packed_context=packed_context,
    )


def plan_tasks(request: TurnScopedRequest, *, todo: Todo | None = None) -> PlanTasksResponse:
    subject = todo.description if todo and todo.description else (todo.title if todo else request.user_query)
    search_queries = planned_search_queries(
        request.user_query,
        subject,
        todo_description=todo.description if todo else "",
    )[:3]
    tasks = [
        ResearchTask(
            task_id="task-1",
            title="Clarify objective",
            goal=f"明确本轮问题的求解目标与答案类型：{subject}",
            task_type="scope",
            output_key="objective",
        ),
        ResearchTask(
            task_id="task-2",
            title="Collect evidence",
            goal="根据目标生成检索查询，在全局知识库中收集候选证据。",
            task_type="retrieve",
            depends_on=["task-1"],
            output_key="evidence",
        ),
        ResearchTask(
            task_id="task-3",
            title="Validate and align",
            goal="把候选证据与工作记忆、情景记忆、语义记忆对齐，筛掉弱相关材料。",
            task_type="reason",
            depends_on=["task-2"],
            output_key="aligned_evidence",
        ),
        ResearchTask(
            task_id="task-4",
            title="Synthesize answer",
            goal="基于对齐后的证据生成带引用答案，并沉淀记忆。",
            task_type="synthesize",
            depends_on=["task-3"],
            output_key="final_answer",
        ),
    ]
    return PlanTasksResponse(
        project_id=request.project_id,
        session_id=request.session_id,
        sequence_id=request.sequence_id,
        planner_mode="two_stage",
        plan_summary="Planner 先定义目标与证据需求，Solver 再按步骤检索、校验、综合并沉淀记忆。",
        search_queries=search_queries,
        tasks=tasks,
    )


def resolve_assets(db: Session) -> list[Asset]:
    return db.scalars(select(Asset).order_by(desc(Asset.updated_at))).all()


def hybrid_retrieve(
    request: TurnScopedRequest,
    *,
    assets: list[Asset],
    search_queries: list[str],
) -> RetrieveResponse:
    payloads = chunk_payloads(assets, request.asset_ids)
    query_hits = [
        (
            search_query,
            get_vector_store().search(
                search_query,
                settings.retrieval_limit,
                request.asset_ids,
                payloads,
            ),
        )
        for search_query in (search_queries or [request.user_query])
    ]
    ranked = merge_retrieval_hits(query_hits, settings.retrieval_limit)
    evidence_items = [
        EvidenceItem(
            asset_id=str(chunk["asset_id"]),
            chunk_id=str(chunk["chunk_id"]),
            label=f"C{index}",
            title=str(chunk["title"]),
            snippet=str(chunk["content"])[:320],
            source_path=str(chunk["source_path"]),
            score=round(float(chunk["score"]), 4),
            tags=[str(chunk["asset_type"]), "plan", "hybrid", "reranked", compact_text(str(chunk["matched_query"]), 48)],
        )
        for index, chunk in enumerate(ranked, start=1)
    ]
    return RetrieveResponse(
        project_id=request.project_id,
        session_id=request.session_id,
        sequence_id=request.sequence_id,
        retrieval_mode=f"{settings.vector_store_provider}_plan_hybrid_rerank",
        evidence_items=evidence_items,
    )


def execute_plan(
    request: TurnScopedRequest,
    *,
    todo: Todo | None,
    plan: PlanTasksResponse,
    assets: list[Asset],
    on_step: Callable[[PlanExecutionStep], None] | None = None,
) -> tuple[PlanTasksResponse, RetrieveResponse]:
    subject = todo.description if todo and todo.description else (todo.title if todo else request.user_query)
    trace: list[PlanExecutionStep] = []

    def append_step(step: PlanExecutionStep) -> None:
        trace.append(step)
        if on_step is not None:
            on_step(step)

    append_step(
        PlanExecutionStep(
            step_id="step-1",
            task_id="task-1",
            title="Scope objective",
            action="scope",
            summary=f"将本轮问题约束为：{compact_text(subject, 140)}",
        )
    )
    retrieval = hybrid_retrieve(request, assets=assets, search_queries=plan.search_queries)
    append_step(
        PlanExecutionStep(
            step_id="step-2",
            task_id="task-2",
            title="Primary retrieval",
            action="retrieve",
            summary=f"首轮按 {len(plan.search_queries)} 条计划查询检索，得到 {len(retrieval.evidence_items)} 条候选证据。",
            search_queries=plan.search_queries,
            evidence_labels=evidence_labels(retrieval),
        )
    )

    replan_count = 0
    replan_reason = ""
    final_queries = list(plan.search_queries)
    final_retrieval = retrieval

    if should_replan(request.user_query, retrieval):
        replan_count = 1
        replan_reason = "首轮证据覆盖不足，Solver 扩展到标题页/摘要/全文定位查询并再次检索。"
        replan_queries = planned_search_queries(
            request.user_query,
            subject,
            todo_description=todo.description if todo else "",
            broaden=True,
        )[:5]
        append_step(
            PlanExecutionStep(
                step_id="step-3",
                task_id="task-2",
                title="Replan retrieval",
                action="replan",
                summary=replan_reason,
                search_queries=replan_queries,
                evidence_labels=evidence_labels(retrieval),
            )
        )
        retry_retrieval = hybrid_retrieve(request, assets=assets, search_queries=replan_queries)
        if retrieval_strength(retry_retrieval) >= retrieval_strength(retrieval):
            final_retrieval = retry_retrieval
            final_queries = replan_queries

    final_labels = evidence_labels(final_retrieval)
    append_step(
        PlanExecutionStep(
            step_id="step-4",
            task_id="task-3",
            title="Validate evidence",
            action="reason",
            summary=(
                "用分层记忆和问题目标过滤证据，"
                f"保留 {len(final_retrieval.evidence_items)} 条高相关材料。"
            ),
            evidence_labels=final_labels,
        )
    )
    append_step(
        PlanExecutionStep(
            step_id="step-5",
            task_id="task-4",
            title="Synthesize answer",
            action="synthesize",
            summary=(
                "基于最终证据生成答案。"
                if final_labels
                else "缺少直接证据，尝试启用直接模型兜底回答。"
            ),
            evidence_labels=final_labels[:3],
        )
    )
    updated_plan = plan.model_copy(
        update={
            "tasks": mark_tasks_completed(plan.tasks),
            "execution_trace": trace,
            "solver_summary": (
                f"Solver 完成 {len(trace)} 个步骤，使用 {len(dedupe_preserve_order(final_queries))} 条查询，"
                f"最终保留 {len(final_retrieval.evidence_items)} 条证据。"
            ),
            "replan_count": replan_count,
            "replan_reason": replan_reason,
            "search_queries": dedupe_preserve_order(plan.search_queries + final_queries),
        }
    )
    return updated_plan, final_retrieval


def plan_live_tool_route(request: TurnScopedRequest, route: LiveToolRoute) -> PlanTasksResponse:
    planner_label = "LLM Tool Planner" if route.planner_mode == "llm_tool_planner" else "Router"
    tasks = [
        ResearchTask(
            task_id="task-1",
            title="Select skill",
            goal=f"识别问题意图并选择技能：{route.skill}",
            task_type="route",
            output_key="selected_skill",
            status="completed",
        ),
        ResearchTask(
            task_id="task-2",
            title="Call live tool",
            goal=f"调用工具 {route.tool_name} 获取实时或公开数据。",
            task_type="tool",
            depends_on=["task-1"],
            output_key="tool_result",
        ),
        ResearchTask(
            task_id="task-3",
            title="Synthesize answer",
            goal="基于工具返回结果生成答案并保留来源。",
            task_type="synthesize",
            depends_on=["task-2"],
            output_key="final_answer",
        ),
    ]
    return PlanTasksResponse(
        project_id=request.project_id,
        session_id=request.session_id,
        sequence_id=request.sequence_id,
        planner_mode=route.planner_mode,
        plan_summary=(
            f"{planner_label} 判断本轮属于 {route.intent}，选择 skill={route.skill}，"
            f"tool={route.tool_name}。理由：{route.reason}"
        ),
        search_queries=[request.user_query],
        tasks=tasks,
    )


def live_tool_failure(route: LiveToolRoute, error: Exception) -> LiveToolResult:
    return LiveToolResult(
        answer=(
            f"我识别到这个问题应使用 {route.tool_name} 工具处理，但工具调用失败：{error}。\n"
            "请稍后重试，或补充本地资料后改用项目知识库检索。"
        ),
        evidence=[],
        metadata={"error": str(error), "tool": route.tool_name, "skill": route.skill},
    )


def evidence_from_live_tool(
    request: TurnScopedRequest,
    result: LiveToolResult,
) -> RetrieveResponse:
    evidence_items = [
        EvidenceItem(
            asset_id="live-tool",
            chunk_id=f"live-{index}-{slugify(item.title)}",
            label=f"C{index}",
            title=item.title,
            snippet=item.snippet[:320],
            source_path=item.source_path,
            score=round(item.score, 4),
            tags=item.tags,
        )
        for index, item in enumerate(result.evidence, start=1)
    ]
    return RetrieveResponse(
        project_id=request.project_id,
        session_id=request.session_id,
        sequence_id=request.sequence_id,
        retrieval_mode="live_tool",
        evidence_items=evidence_items,
    )


def answer_from_live_tool(
    request: TurnScopedRequest,
    route: LiveToolRoute,
    result: LiveToolResult,
    retrieval: RetrieveResponse,
) -> AnswerResponse:
    answer = result.answer
    tool_error = bool(result.metadata.get("error"))
    if tool_error or not retrieval.evidence_items:
        fallback = try_direct_llm_answer(
            request,
            reason=f"{route.tool_name} did not return usable evidence",
            tool_payload={
                "route": {
                    "intent": route.intent,
                    "skill": route.skill,
                    "tool_name": route.tool_name,
                    "reason": route.reason,
                },
                "tool_answer": result.answer,
                "tool_metadata": result.metadata,
            },
        )
        if fallback:
            result.metadata["direct_llm_fallback"] = True
            answer = fallback
    elif settings.llm_provider == "deepseek" and settings.llm_api_key:
        try:
            answer = llm_live_tool_answer(route, request, result, retrieval)
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            answer = result.answer
    if contains_restrictive_fallback_text(answer):
        fallback = try_direct_llm_answer(
            request,
            reason=f"{route.tool_name} answer was not usable",
            tool_payload={
                "route": {
                    "intent": route.intent,
                    "skill": route.skill,
                    "tool_name": route.tool_name,
                    "reason": route.reason,
                },
                "tool_answer": result.answer,
                "tool_metadata": result.metadata,
            },
        )
        if fallback:
            result.metadata["direct_llm_fallback"] = True
            answer = fallback
    return AnswerResponse(
        project_id=request.project_id,
        session_id=request.session_id,
        sequence_id=request.sequence_id,
        answer=answer,
        citations=[] if result.metadata.get("direct_llm_fallback") else citations_from_evidence(retrieval.evidence_items),
    )


def execute_live_tool_plan(
    request: TurnScopedRequest,
    route: LiveToolRoute,
    *,
    on_step: Callable[[PlanExecutionStep], None] | None = None,
) -> tuple[PlanTasksResponse, RetrieveResponse, AnswerResponse]:
    plan = plan_live_tool_route(request, route)
    trace: list[PlanExecutionStep] = []

    def append_step(step: PlanExecutionStep) -> None:
        trace.append(step)
        if on_step is not None:
            on_step(step)

    append_step(
        PlanExecutionStep(
            step_id="step-1",
            task_id="task-1",
            title="Route request",
            action="route",
            summary=(
                f"识别为 {route.intent}，选择 skill={route.skill}，"
                f"tool={route.tool_name}，confidence={route.confidence:.2f}。"
            ),
        )
    )
    append_step(
        PlanExecutionStep(
            step_id="step-2",
            task_id="task-2",
            title="Call live tool",
            action="tool_call",
            summary=f"调用 {route.tool_name}，因为：{route.reason}",
            search_queries=[request.user_query],
        )
    )
    failed = False
    try:
        result = execute_live_tool(route, request.user_query)
    except (httpx.HTTPError, ValueError) as exc:
        failed = True
        result = live_tool_failure(route, exc)
    retrieval = evidence_from_live_tool(request, result)
    append_step(
        PlanExecutionStep(
            step_id="step-3",
            task_id="task-2",
            title="Tool result",
            action="tool_result",
            summary=(
                f"工具返回 {len(retrieval.evidence_items)} 条来源。"
                if not failed
                else f"工具调用失败：{result.metadata.get('error', '')}；将尝试直接模型兜底。"
            ),
            status="failed" if failed else "completed",
            evidence_labels=evidence_labels(retrieval),
        )
    )
    answer = answer_from_live_tool(request, route, result, retrieval)
    synthesize_summary = "基于实时/公开工具结果生成答案。"
    if result.metadata.get("direct_llm_fallback"):
        synthesize_summary = "工具未返回可用结果，已启用直接模型兜底回答。"
    elif failed:
        synthesize_summary = "工具未返回可用结果，生成可解释的失败答复。"
    append_step(
        PlanExecutionStep(
            step_id="step-4",
            task_id="task-3",
            title="Synthesize answer",
            action="synthesize",
            summary=synthesize_summary,
            evidence_labels=evidence_labels(retrieval),
        )
    )
    updated_tasks = [
        task.model_copy(update={"status": "failed" if failed and task.task_id == "task-2" else "completed"})
        for task in plan.tasks
    ]
    plan = plan.model_copy(
        update={
            "tasks": updated_tasks,
            "execution_trace": trace,
            "solver_summary": (
                f"Router 完成工具路由，使用 skill={route.skill}，tool={route.tool_name}，"
                f"最终保留 {len(retrieval.evidence_items)} 条来源。"
            ),
            "replan_count": 0,
            "replan_reason": "",
        }
    )
    return plan, retrieval, answer


def llm_answer_markdown(request: AnswerWithCitationsRequest) -> str:
    response = httpx.post(
        f"{settings.llm_api_base.rstrip('/')}/chat/completions",
        headers=llm_headers(),
        json=answer_prompt(request),
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def llm_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }


DIRECT_FALLBACK_RESTRICTIVE_TERMS = (
    "当前的知识库",
    "当前知识库",
    "当前的知识库和网络搜索",
    "当前知识库和网络搜索",
    "无法从当前",
    "网络搜索中找不到",
    "网络搜索没有返回",
    "公开搜索没有返回",
    "没有命中文档证据",
    "本地知识库没有命中",
    "补充本地资料",
    "工具调用失败",
    "当前没有足够的工具观察结果",
)


def llm_available() -> bool:
    return settings.llm_provider == "deepseek" and bool(settings.llm_api_key)


def contains_restrictive_fallback_text(text: str) -> bool:
    return any(term in text for term in DIRECT_FALLBACK_RESTRICTIVE_TERMS)


def direct_llm_answer(
    request: TurnScopedRequest,
    *,
    reason: str,
    packed_context: str = "",
    tool_payload: dict[str, object] | None = None,
    observations: list[dict[str, object]] | None = None,
) -> str:
    payload = {
        "current_date": utc_now().date().isoformat(),
        "fallback_reason": reason,
        "user_query": request.user_query,
        "project_context_preview": compact_text(packed_context, 1200) if packed_context else "",
        "tool_payload": tool_payload or {},
        "recent_observations": observations[-6:] if observations else [],
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
                        "你是 Research Copilot 的直接回答兜底。当前工具或本地检索没有给出可用结果时，"
                        "基于模型已有知识、用户问题和少量上下文直接回答。不要暴露内部检索限制，"
                        "不要说“当前知识库找不到”“网络搜索找不到”或“只能基于证据”。"
                        "不要提及内部记忆、工具调用或检索流程，除非用户明确询问过程。"
                        "不要编造引用，不要输出 [C1] 这类引用标签。"
                        "涉及年龄、年份、任期、相对日期时，按 current_date 计算并写明日期基准。"
                        "涉及天气、股价、赛事比分等强实时数据且没有有效工具结果时，不要假装拿到实时数据；"
                        "简短说明无法确认实时值，并给出一般判断方式。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "请直接回答用户问题。输入 JSON：\n"
                        f"{json.dumps(payload, ensure_ascii=False, default=str)}"
                    ),
                },
            ],
            "temperature": 0.2,
            "max_tokens": 700,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def try_direct_llm_answer(
    request: TurnScopedRequest,
    *,
    reason: str,
    packed_context: str = "",
    tool_payload: dict[str, object] | None = None,
    observations: list[dict[str, object]] | None = None,
) -> str:
    if not llm_available():
        return ""
    try:
        return direct_llm_answer(
            request,
            reason=reason,
            packed_context=packed_context,
            tool_payload=tool_payload,
            observations=observations,
        )
    except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("direct_llm_fallback_failed error=%s", exc)
        return ""


def agent_should_use_direct_llm_fallback(
    answer_text: str,
    history: list[dict[str, object]],
    evidence_items: list[EvidenceItem],
) -> bool:
    if not answer_text.strip() or contains_restrictive_fallback_text(answer_text):
        return True
    if evidence_items:
        return False
    observations = [item for item in history if item.get("kind") == "observation"]
    if observations and all(isinstance(item.get("metadata"), dict) and item["metadata"].get("error") for item in observations):
        return True
    return False


def agent_tool_catalog() -> list[AgentToolSpec]:
    return [
        AgentToolSpec(
            name="local_rag_search",
            description="检索本项目/全局知识库中的资产内容，适用于论文、代码说明、项目资料和需要引用本地证据的问题。",
            input_schema={"query": "检索问题，字符串"},
        ),
        AgentToolSpec(
            name="weather_lookup",
            description="查询实时天气事实，适用于天气、气温、降水、风速、出行或活动适宜性判断。",
            input_schema={"query": "包含城市/地点的天气问题，字符串"},
        ),
        AgentToolSpec(
            name="public_web_search",
            description="查询公开网络摘要，适用于最新信息、官网、公开事实或用户明确要求联网搜索的问题。",
            input_schema={"query": "搜索问题，字符串"},
        ),
        AgentToolSpec(
            name="memory_read",
            description="读取当前项目的 working/episodic/semantic memory，适用于刚才说过什么、项目记住了什么、用户偏好等问题。",
            input_schema={"query": "记忆检索问题，字符串，可省略"},
        ),
        AgentToolSpec(
            name="memory_write",
            description="写入一条短期工作记忆，适用于用户明确要求记住某个偏好、约束或事实。",
            input_schema={"key": "记忆键", "content": "要记住的内容", "importance": "0-1，可选"},
        ),
        AgentToolSpec(
            name="todo_create",
            description="为当前项目创建 TODO。",
            input_schema={"title": "TODO 标题", "description": "描述，可选", "priority": "low|medium|high，可选"},
        ),
        AgentToolSpec(
            name="todo_list",
            description="列出当前项目 TODO。",
            input_schema={"status": "可选，按状态过滤"},
        ),
        AgentToolSpec(
            name="asset_list",
            description="列出全局知识库资产，适用于用户询问有哪些资料/文档/资产。",
            input_schema={},
        ),
        AgentToolSpec(
            name="calculator",
            description="计算简单算术表达式，支持 + - * / // % ** 和括号。",
            input_schema={"expression": "算术表达式，字符串"},
        ),
    ]


def agent_tool_catalog_payload() -> list[dict[str, object]]:
    return [
        {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
        for tool in agent_tool_catalog()
    ]


def agent_plan(request: TurnScopedRequest) -> PlanTasksResponse:
    tasks = [
        ResearchTask(
            task_id="task-1",
            title="Plan next action",
            goal="理解用户目标，选择下一步工具或最终回答。",
            task_type="agent_reason",
            output_key="agent_action",
        ),
        ResearchTask(
            task_id="task-2",
            title="Execute tool",
            goal="按结构化参数调用工具，并记录 observation。",
            task_type="agent_tool",
            depends_on=["task-1"],
            output_key="observation",
        ),
        ResearchTask(
            task_id="task-3",
            title="Decide continuation",
            goal="根据 observation 决定继续调用工具、纠错或结束。",
            task_type="agent_loop",
            depends_on=["task-2"],
            output_key="next_step",
        ),
        ResearchTask(
            task_id="task-4",
            title="Final answer",
            goal="基于所有工具观察生成最终回答。",
            task_type="synthesize",
            depends_on=["task-3"],
            output_key="final_answer",
        ),
    ]
    return PlanTasksResponse(
        project_id=request.project_id,
        session_id=request.session_id,
        sequence_id=request.sequence_id,
        planner_mode="agent_loop",
        plan_summary=(
            f"Agent 使用最多 {settings.agent_max_steps} 步 Plan-Act-Observe 循环，"
            "可在 local_rag_search、weather_lookup、public_web_search、memory、TODO、asset、calculator 工具之间自主选择。"
        ),
        search_queries=[request.user_query],
        tasks=tasks,
    )


def agent_history_text(history: list[dict[str, object]]) -> str:
    if not history:
        return "[]"
    return json.dumps(history[-8:], ensure_ascii=False, default=str)


def llm_agent_next_step(
    request: TurnScopedRequest,
    context: BuildContextResponse,
    history: list[dict[str, object]],
) -> AgentDecision:
    response = httpx.post(
        f"{settings.llm_api_base.rstrip('/')}/chat/completions",
        headers=llm_headers(),
        json={
            "model": settings.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是 Research Copilot 的多步工具调用 Agent。"
                        "你只能返回一个 JSON 对象，不要 Markdown，不要解释。"
                        "如果还需要信息，选择一个工具；如果已经足够，action 使用 final_answer。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "可用工具：\n"
                        f"{json.dumps(agent_tool_catalog_payload(), ensure_ascii=False)}\n\n"
                        "输出 JSON schema：\n"
                        "{"
                        '"thought": "简短思考", '
                        '"action": "工具名或 final_answer", '
                        '"arguments": {"参数名": "参数值"}'
                        "}\n\n"
                        "约束：\n"
                        "- 每一步只能选择一个 action。\n"
                        "- 需要实时天气时用 weather_lookup，不要用旧记忆替代。\n"
                        "- 需要本地资料证据时用 local_rag_search。\n"
                        "- 需要创建任务时用 todo_create。\n"
                        "- final_answer 的 arguments 必须包含 answer 字段。\n"
                        "- 工具失败或证据不足时可以换工具重试。\n\n"
                        f"用户问题：{request.user_query}\n\n"
                        f"项目上下文：\n{context.packed_context[:3000]}\n\n"
                        f"已有步骤和 observations：\n{agent_history_text(history)}"
                    ),
                },
            ],
            "temperature": 0.1,
            "max_tokens": 700,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    content = str(response.json()["choices"][0]["message"]["content"]).strip()
    return agent_decision_from_payload(json.loads(extract_json_object(content)))


def agent_decision_from_payload(payload: dict[str, object]) -> AgentDecision:
    action = str(payload.get("action") or "").strip()
    if not action:
        raise ValueError("Agent decision missing action")
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise ValueError("Agent decision arguments must be an object")
    return AgentDecision(
        thought=str(payload.get("thought") or "").strip(),
        action=normalize_agent_action(action),
        arguments=dict(arguments),
    )


def normalize_agent_action(action: str) -> str:
    aliases = {
        "finish": "final_answer",
        "final": "final_answer",
        "answer": "final_answer",
        "rag": "local_rag_search",
        "search_local": "local_rag_search",
        "web_search": "public_web_search",
        "weather": "weather_lookup",
        "create_todo": "todo_create",
        "list_todo": "todo_list",
        "list_todos": "todo_list",
        "list_assets": "asset_list",
        "memory": "memory_read",
        "remember": "memory_write",
        "calculate": "calculator",
    }
    return aliases.get(action.strip().lower(), action.strip())


def choose_agent_next_step(
    request: TurnScopedRequest,
    context: BuildContextResponse,
    history: list[dict[str, object]],
) -> AgentDecision:
    if settings.llm_provider == "deepseek" and settings.llm_api_key:
        try:
            return llm_agent_next_step(request, context, history)
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("llm_agent_next_step_failed error=%s", exc)
    return fallback_agent_next_step(request, history)


def fallback_agent_next_step(request: TurnScopedRequest, history: list[dict[str, object]]) -> AgentDecision:
    query = request.user_query
    called_tools = [str(item.get("action") or "") for item in history if item.get("kind") == "action"]
    if called_tools:
        if "weather_lookup" in called_tools and wants_todo_creation(query) and "todo_create" not in called_tools:
            return AgentDecision(
                thought="天气信息已获取，用户还要求创建 TODO。",
                action="todo_create",
                arguments={
                    "title": extract_todo_title(query),
                    "description": "由多步 Agent 根据用户请求创建。",
                    "priority": "medium",
                },
            )
        return AgentDecision(
            thought="已有 observation，生成最终回答。",
            action="final_answer",
            arguments={"answer": summarize_agent_observations(history)},
        )
    if wants_memory_write(query):
        return AgentDecision(
            thought="用户明确要求记住信息，先写入工作记忆。",
            action="memory_write",
            arguments={"key": "user_note", "content": query, "importance": 0.8},
        )
    if "记忆" in query or "刚才" in query or "记住" in query:
        return AgentDecision(thought="用户询问记忆内容。", action="memory_read", arguments={"query": query})
    if wants_todo_list(query):
        return AgentDecision(thought="用户询问 TODO 列表。", action="todo_list", arguments={})
    if wants_todo_creation(query) and not ("天气" in query or "出游" in query):
        return AgentDecision(
            thought="用户要求创建 TODO。",
            action="todo_create",
            arguments={"title": extract_todo_title(query), "description": "", "priority": "medium"},
        )
    if "资产" in query or "文档" in query or "资料" in query:
        return AgentDecision(thought="用户询问资产资料。", action="asset_list", arguments={})
    if "计算" in query or re.search(r"\d+\s*[-+*/%]", query):
        return AgentDecision(thought="用户需要算术计算。", action="calculator", arguments={"expression": extract_expression(query)})
    if any(term in query for term in ("天气", "气温", "温度", "出游", "下雨", "降水")):
        return AgentDecision(thought="用户需要实时天气或出行判断。", action="weather_lookup", arguments={"query": query})
    if any(term in query for term in ("联网", "搜索", "最新", "新闻", "官网", "公开资料")):
        return AgentDecision(thought="用户需要公开网络资料。", action="public_web_search", arguments={"query": query})
    return AgentDecision(thought="默认先检索本地知识库。", action="local_rag_search", arguments={"query": query})


def wants_todo_creation(query: str) -> bool:
    return any(term in query.lower() for term in ("todo", "待办")) and any(term in query for term in ("创建", "新建", "加入", "添加", "生成"))


def wants_todo_list(query: str) -> bool:
    return any(term in query.lower() for term in ("todo", "待办")) and any(term in query for term in ("列出", "有哪些", "查看", "列表"))


def wants_memory_write(query: str) -> bool:
    return any(term in query for term in ("记住", "请记住", "帮我记")) and not any(term in query for term in ("记住了什么", "刚才"))


def extract_todo_title(query: str) -> str:
    match = re.search(r"(?:TODO|todo|待办)[：: ]*([^，。；;\n]+)", query)
    if match:
        return compact_text(match.group(1), 80)
    match = re.search(r"[“\"]([^”\"]+)[”\"]", query)
    if match:
        return compact_text(match.group(1), 80)
    return compact_text(query, 80)


def extract_expression(query: str) -> str:
    matches = re.findall(r"[0-9][0-9\s+\-*/().%]*", query)
    return max(matches, key=len).strip() if matches else query


def summarize_agent_observations(history: list[dict[str, object]]) -> str:
    observations = [str(item.get("content") or item.get("summary") or "") for item in history if item.get("kind") == "observation"]
    if not observations:
        return "当前没有足够的工具观察结果。"
    return "\n\n".join(observations[-3:])


def append_agent_evidence(existing: list[EvidenceItem], new_items: list[EvidenceItem]) -> list[EvidenceItem]:
    updated = list(existing)
    for item in new_items:
        next_label = f"C{len(updated) + 1}"
        updated.append(item.model_copy(update={"label": next_label}))
    return updated


def observation_from_live_result(
    request: TurnScopedRequest,
    result: LiveToolResult,
    *,
    tool_name: str,
) -> AgentToolObservation:
    retrieval = evidence_from_live_tool(request, result)
    return AgentToolObservation(
        tool_name=tool_name,
        summary=f"{tool_name} 返回 {len(retrieval.evidence_items)} 条来源。",
        content=result.answer,
        evidence_items=retrieval.evidence_items,
        metadata=result.metadata,
    )


def execute_agent_tool(
    db: Session,
    request: TurnScopedRequest,
    context: BuildContextResponse,
    decision: AgentDecision,
) -> AgentToolObservation:
    action = decision.action
    arguments = decision.arguments
    if action == "local_rag_search":
        query = compact_text(str(arguments.get("query") or request.user_query), 240)
        assets = resolve_assets(db)
        retrieval = hybrid_retrieve(request.model_copy(update={"user_query": query}), assets=assets, search_queries=[query])
        lines = [f"[{item.label}] {item.title}: {item.snippet}" for item in retrieval.evidence_items]
        return AgentToolObservation(
            tool_name=action,
            summary=f"本地知识库检索返回 {len(retrieval.evidence_items)} 条证据。",
            content="\n".join(lines) or "本地知识库没有命中证据。",
            evidence_items=retrieval.evidence_items,
            metadata={"query": query, "retrieval_mode": retrieval.retrieval_mode},
        )
    if action == "weather_lookup":
        query = str(arguments.get("query") or request.user_query)
        route = LiveToolRoute(
            intent="realtime_weather",
            skill="weather_qa",
            tool_name="weather_lookup",
            reason="Agent selected weather_lookup",
            confidence=0.9,
            planner_mode="agent_loop",
        )
        return observation_from_live_result(request, execute_live_tool(route, query), tool_name=action)
    if action == "public_web_search":
        query = str(arguments.get("query") or request.user_query)
        route = LiveToolRoute(
            intent="public_web_lookup",
            skill="public_web_qa",
            tool_name="public_web_search",
            reason="Agent selected public_web_search",
            confidence=0.8,
            planner_mode="agent_loop",
        )
        return observation_from_live_result(request, execute_live_tool(route, query), tool_name=action)
    if action == "memory_read":
        query = str(arguments.get("query") or request.user_query)
        bundle = build_memory_context_bundle(db, request.project_id, request.session_id, query)
        return AgentToolObservation(
            tool_name=action,
            summary="读取项目分层记忆。",
            content=bundle.combined_text,
            evidence_items=[],
            metadata={"query": query},
        )
    if action == "memory_write":
        key = compact_text(str(arguments.get("key") or "agent_note"), 64)
        content = compact_text(str(arguments.get("content") or request.user_query), 500)
        importance = clamp_confidence(arguments.get("importance", 0.7))
        item = WorkingMemoryItem(
            id=make_id("wm"),
            project_id=request.project_id,
            session_id=request.session_id,
            sequence_id=request.sequence_id,
            memory_key=key,
            content=content,
            importance=importance,
            source="agent_tool",
            meta_payload={"kind": "agent_memory_write"},
        )
        db.add(item)
        db.flush()
        return AgentToolObservation(
            tool_name=action,
            summary=f"写入工作记忆：{key}",
            content=f"已记住：{content}",
            evidence_items=[],
            metadata={"memory_key": key, "importance": importance},
        )
    if action == "todo_create":
        title = compact_text(str(arguments.get("title") or extract_todo_title(request.user_query)), 160)
        description = str(arguments.get("description") or "")
        priority = str(arguments.get("priority") or "medium")
        todo = create_todo(
            db,
            request.project_id,
            TodoCreate(title=title, description=description, priority=priority, status="todo"),
        )
        return AgentToolObservation(
            tool_name=action,
            summary=f"创建 TODO：{todo.title}",
            content=f"已创建 TODO：{todo.title}（priority={todo.priority}, id={todo.id}）",
            evidence_items=[],
            metadata={"todo_id": todo.id, "title": todo.title, "priority": todo.priority},
        )
    if action == "todo_list":
        status = str(arguments.get("status") or "")
        todos = list_todos(db, request.project_id)
        if status:
            todos = [todo for todo in todos if todo.status == status]
        lines = [f"- {todo.title} [{todo.status}, {todo.priority}] id={todo.id}" for todo in todos]
        return AgentToolObservation(
            tool_name=action,
            summary=f"读取 TODO 列表，共 {len(todos)} 条。",
            content="\n".join(lines) or "当前没有 TODO。",
            evidence_items=[],
            metadata={"count": len(todos), "status": status},
        )
    if action == "asset_list":
        assets = list_assets(db)
        lines = [f"- {asset.title} ({asset.asset_type}) id={asset.id}" for asset in assets[:20]]
        return AgentToolObservation(
            tool_name=action,
            summary=f"读取资产列表，共 {len(assets)} 条。",
            content="\n".join(lines) or "当前没有资产。",
            evidence_items=[],
            metadata={"count": len(assets)},
        )
    if action == "calculator":
        expression = str(arguments.get("expression") or extract_expression(request.user_query))
        value = safe_calculate(expression)
        return AgentToolObservation(
            tool_name=action,
            summary=f"计算表达式：{expression}",
            content=f"{expression} = {value}",
            evidence_items=[],
            metadata={"expression": expression, "value": value},
        )
    raise ValueError(f"Unsupported agent action: {action}")


def safe_calculate(expression: str) -> float:
    allowed_binary = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a**b,
    }
    allowed_unary = {
        ast.UAdd: lambda a: a,
        ast.USub: lambda a: -a,
    }

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_binary:
            left = eval_node(node.left)
            right = eval_node(node.right)
            return float(allowed_binary[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary:
            return float(allowed_unary[type(node.op)](eval_node(node.operand)))
        raise ValueError("只支持简单算术表达式")

    if not re.fullmatch(r"[0-9\s+\-*/().%]+", expression):
        raise ValueError("算术表达式包含不支持的字符")
    return eval_node(ast.parse(expression, mode="eval"))


def agent_final_answer_from_history(
    request: TurnScopedRequest,
    history: list[dict[str, object]],
    final_text: str,
    evidence_items: list[EvidenceItem],
) -> str:
    if final_text.strip():
        return final_text.strip()
    lines = ["Agent 已完成多步工具调用。"]
    observation_text = summarize_agent_observations(history)
    if observation_text:
        lines.extend(["", observation_text])
    if evidence_items:
        labels = ", ".join(item.label for item in evidence_items[:3])
        lines.append(f"\n引用：{labels}")
    return "\n".join(lines)


def execute_agent_plan(
    db: Session,
    request: TurnScopedRequest,
    context: BuildContextResponse,
    *,
    on_step: Callable[[PlanExecutionStep], None] | None = None,
) -> tuple[PlanTasksResponse, RetrieveResponse, AnswerResponse]:
    plan = agent_plan(request)
    trace: list[PlanExecutionStep] = []
    history: list[dict[str, object]] = []
    evidence_items: list[EvidenceItem] = []
    final_text = ""

    def append_step(step: PlanExecutionStep) -> None:
        trace.append(step)
        if on_step is not None:
            on_step(step)

    for step_number in range(1, max(settings.agent_max_steps, 1) + 1):
        decision = choose_agent_next_step(request, context, history)
        action_summary = (
            f"thought={decision.thought or '未提供'}; "
            f"action={decision.action}; args={json.dumps(decision.arguments, ensure_ascii=False, default=str)}"
        )
        append_step(
            PlanExecutionStep(
                step_id=f"agent-step-{step_number}-action",
                task_id="task-1",
                title=f"Agent action: {decision.action}",
                action="agent_action",
                summary=action_summary,
            )
        )
        history.append(
            {
                "kind": "action",
                "step": step_number,
                "thought": decision.thought,
                "action": decision.action,
                "arguments": decision.arguments,
            }
        )
        if decision.action == "final_answer":
            final_text = str(decision.arguments.get("answer") or "")
            append_step(
                PlanExecutionStep(
                    step_id=f"agent-step-{step_number}-final",
                    task_id="task-4",
                    title="Agent final answer",
                    action="agent_final",
                    summary=compact_text(final_text, 260) or "Agent 决定结束并生成最终回答。",
                    evidence_labels=[item.label for item in evidence_items],
                )
            )
            break
        try:
            observation = execute_agent_tool(db, request, context, decision)
            evidence_items = append_agent_evidence(evidence_items, observation.evidence_items)
            observation_summary = observation.summary
            observation_status = "completed"
            observation_content = observation.content
        except (httpx.HTTPError, ValueError, ZeroDivisionError) as exc:
            observation = AgentToolObservation(
                tool_name=decision.action,
                summary=f"工具调用失败：{exc}",
                content=f"工具 {decision.action} 调用失败：{exc}",
                evidence_items=[],
                metadata={"error": str(exc)},
            )
            observation_summary = observation.summary
            observation_status = "failed"
            observation_content = observation.content
        history.append(
            {
                "kind": "observation",
                "step": step_number,
                "tool": observation.tool_name,
                "summary": observation.summary,
                "content": observation.content[:1200],
                "metadata": observation.metadata,
                "evidence_labels": [item.label for item in evidence_items],
            }
        )
        append_step(
            PlanExecutionStep(
                step_id=f"agent-step-{step_number}-observation",
                task_id="task-2",
                title=f"Observation: {observation.tool_name}",
                action="agent_observation",
                summary=f"{observation_summary} {compact_text(observation_content, 220)}",
                status=observation_status,
                evidence_labels=[item.label for item in evidence_items],
            )
        )
    if not final_text.strip():
        final_text = agent_final_answer_from_history(request, history, "", evidence_items)
        append_step(
            PlanExecutionStep(
                step_id="agent-final-fallback",
                task_id="task-4",
                title="Agent final fallback",
                action="agent_final",
                summary=compact_text(final_text, 260),
                evidence_labels=[item.label for item in evidence_items],
            )
        )
    retrieval = RetrieveResponse(
        project_id=request.project_id,
        session_id=request.session_id,
        sequence_id=request.sequence_id,
        retrieval_mode="agent_tool_loop",
        evidence_items=evidence_items,
    )
    answer_text = agent_final_answer_from_history(request, history, final_text, evidence_items)
    direct_fallback_used = False
    if agent_should_use_direct_llm_fallback(answer_text, history, evidence_items):
        fallback = try_direct_llm_answer(
            request,
            reason="agent tools did not produce usable evidence or answer",
            packed_context=context.packed_context,
            observations=history,
        )
        if fallback:
            answer_text = fallback
            direct_fallback_used = True
            append_step(
                PlanExecutionStep(
                    step_id="agent-direct-llm-fallback",
                    task_id="task-4",
                    title="Agent direct LLM fallback",
                    action="agent_final",
                    summary=compact_text(answer_text, 260),
                    evidence_labels=[],
                )
            )
    answer = AnswerResponse(
        project_id=request.project_id,
        session_id=request.session_id,
        sequence_id=request.sequence_id,
        answer=answer_text,
        citations=[] if direct_fallback_used else citations_from_evidence(evidence_items),
    )
    plan = plan.model_copy(
        update={
            "tasks": mark_tasks_completed(plan.tasks),
            "execution_trace": trace,
            "solver_summary": (
                f"Agent 完成 {len([item for item in history if item.get('kind') == 'action'])} 个决策步骤，"
                f"调用 {len([item for item in history if item.get('kind') == 'observation'])} 次工具，"
                f"保留 {len(evidence_items)} 条证据。"
            ),
            "replan_count": 0,
            "replan_reason": "",
        }
    )
    return plan, retrieval, answer


def live_tool_catalog() -> list[dict[str, object]]:
    catalog: list[dict[str, object]] = [
        {
            "tool_name": "weather_lookup",
            "skill": "weather_qa",
            "intent": "realtime_weather",
            "description": "查询当前或近期天气事实。适用于天气、气温、降水、风速，以及基于天气事实的出行/穿衣/活动适宜性判断。",
            "arguments": {"query": "用户原始问题，必须包含或可从上下文推断地点。"},
        }
    ]
    if settings.public_web_search_enabled:
        catalog.append(
            {
                "tool_name": "public_web_search",
                "skill": "public_web_qa",
                "intent": "public_web_lookup",
                "description": "查询公开网络资料。适用于最新消息、官网信息、公开事实、非本项目资料的问题。",
                "arguments": {"query": "用户原始问题或等价检索问题。"},
            }
        )
    catalog.append(
        {
            "tool_name": "local_project_rag",
            "skill": "project_research_qa",
            "intent": "local_project_research",
            "description": "不调用实时工具，转入本项目知识库、上传资产、会话记忆、TODO 和本地 RAG/plan-and-solve 流程。",
            "arguments": {},
        }
    )
    return catalog


def tool_planner_context_summary(request: TurnScopedRequest, context: BuildContextResponse) -> dict[str, object]:
    return {
        "has_selected_assets": bool(request.asset_ids),
        "selected_asset_count": len(request.asset_ids),
        "has_todo": bool(request.todo_id),
        "session_context_preview": compact_text(context.session_context, 180),
        "task_state_preview": compact_text(context.task_state_context, 180),
        "knowledge_scope_preview": compact_text(context.evidence_context, 220),
    }


def choose_live_tool(request: TurnScopedRequest, context: BuildContextResponse) -> ToolPlannerDecision:
    if not settings.live_tools_enabled:
        return ToolPlannerDecision(None, "disabled", "live tools are disabled")
    if settings.llm_tool_planner_enabled and settings.llm_provider == "deepseek" and settings.llm_api_key:
        try:
            route = llm_select_live_tool(request, context)
            if route is None:
                return ToolPlannerDecision(None, "llm_tool_planner", "LLM planner selected local_project_rag")
            return ToolPlannerDecision(route, route.planner_mode, route.reason)
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            logger.warning("llm_tool_planner_failed error=%s", exc)
            fallback = select_live_tool(request.user_query)
            if fallback is not None:
                return ToolPlannerDecision(
                    fallback,
                    fallback.planner_mode,
                    fallback.reason,
                    fallback_reason=f"LLM planner failed; used rule fallback: {exc}",
                )
            return ToolPlannerDecision(
                None,
                "tool_router",
                "LLM planner failed and rule fallback did not select a live tool",
                fallback_reason=str(exc),
            )
    fallback = select_live_tool(request.user_query)
    if fallback is not None:
        return ToolPlannerDecision(fallback, fallback.planner_mode, fallback.reason)
    return ToolPlannerDecision(None, "tool_router", "rule router did not select a live tool")


def llm_select_live_tool(
    request: TurnScopedRequest,
    context: BuildContextResponse,
) -> LiveToolRoute | None:
    response = httpx.post(
        f"{settings.llm_api_base.rstrip('/')}/chat/completions",
        headers=llm_headers(),
        json={
            "model": settings.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是 Research Copilot 的工具规划器。你的唯一任务是决定是否调用工具，"
                        "不要回答用户问题。只能返回一个 JSON 对象，不要 Markdown。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "可用工具目录：\n"
                        f"{json.dumps(live_tool_catalog(), ensure_ascii=False)}\n\n"
                        "输出 JSON schema：\n"
                        "{"
                        '"use_tool": true|false, '
                        '"tool_name": "weather_lookup|public_web_search|local_project_rag", '
                        '"skill": "weather_qa|public_web_qa|project_research_qa", '
                        '"intent": "realtime_weather|public_web_lookup|local_project_research", '
                        '"confidence": 0.0, '
                        '"reason": "简短说明"'
                        "}\n\n"
                        "决策规则：\n"
                        "- 如果用户问天气事实，或问出游/穿衣/活动是否适合且需要天气事实，选择 weather_lookup。\n"
                        "- 对“今天/现在/当前/明天”等时效问题，不要用上下文里的旧天气记忆替代实时工具。\n"
                        "- 示例：“今天上海适合出游吗？”必须选择 weather_lookup，因为需要先查询天气事实。\n"
                        "- 如果用户问最新消息、官网、公开网络资料或通用公开事实，选择 public_web_search。\n"
                        "- 如果问题绑定本项目、上传资产、代码、论文库、TODO、会话记忆或刚才内容，选择 local_project_rag。\n"
                        "- 如果不需要实时/公开外部信息，也选择 local_project_rag。\n"
                        "- 不要因为问题包含“论文/项目/刚才/本项目”而选择公开搜索，除非用户明确要求联网查公开资料。\n\n"
                        f"用户问题：{request.user_query}\n\n"
                        "当前项目上下文只作为判断是否绑定本项目的信号，不可当作事实答案：\n"
                        f"{json.dumps(tool_planner_context_summary(request, context), ensure_ascii=False)}"
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 300,
        },
        timeout=settings.llm_tool_planner_timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()
    content = str(data["choices"][0]["message"]["content"]).strip()
    decision = json.loads(extract_json_object(content))
    return route_from_tool_planner_decision(decision)


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM tool planner did not return a JSON object")
    return stripped[start : end + 1]


def route_from_tool_planner_decision(decision: dict[str, object]) -> LiveToolRoute | None:
    use_tool = planner_truthy(decision.get("use_tool", True))
    raw_tool_name = str(decision.get("tool_name") or "").strip()
    tool_name = normalize_planner_tool_name(raw_tool_name)
    if not use_tool or tool_name == "local_project_rag":
        return None
    catalog_by_tool = {
        str(item["tool_name"]): item
        for item in live_tool_catalog()
        if str(item["tool_name"]) != "local_project_rag"
    }
    if tool_name not in catalog_by_tool:
        raise ValueError(f"Unsupported planner tool: {raw_tool_name or tool_name}")
    tool_spec = catalog_by_tool[tool_name]
    confidence = clamp_confidence(decision.get("confidence", 0.6))
    reason = str(decision.get("reason") or tool_spec["description"]).strip()
    return LiveToolRoute(
        intent=str(tool_spec["intent"]),
        skill=str(tool_spec["skill"]),
        tool_name=tool_name,
        reason=reason,
        confidence=confidence,
        planner_mode="llm_tool_planner",
    )


def normalize_planner_tool_name(tool_name: str) -> str:
    aliases = {
        "": "local_project_rag",
        "none": "local_project_rag",
        "no_tool": "local_project_rag",
        "local": "local_project_rag",
        "local_rag": "local_project_rag",
        "project_rag": "local_project_rag",
        "weather": "weather_lookup",
        "weather_qa": "weather_lookup",
        "web_search": "public_web_search",
        "search": "public_web_search",
        "public_web": "public_web_search",
    }
    return aliases.get(tool_name.strip().lower(), tool_name.strip())


def planner_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", "否", "不"}
    return bool(value)


def clamp_confidence(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.6
    return max(0.0, min(1.0, number))


def llm_live_tool_answer(
    route: LiveToolRoute,
    request: TurnScopedRequest,
    result: LiveToolResult,
    retrieval: RetrieveResponse,
) -> str:
    evidence_block = "\n".join(
        f"[{item.label}] {item.title}\n{item.snippet}\nsource={item.source_path}"
        for item in retrieval.evidence_items
    )
    tool_payload = {
        "intent": route.intent,
        "skill": route.skill,
        "tool": route.tool_name,
        "tool_metadata": result.metadata,
        "tool_facts": result.answer,
        "evidence": [item.model_dump() for item in retrieval.evidence_items],
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
                        "你是一个工具增强的 Research Copilot。工具已经完成事实查询，"
                        "你负责理解用户完整问题并基于工具结果推理作答。"
                        "不要把工具事实机械复述完就结束；如果用户询问建议、是否适合、风险、选择，"
                        "要给出明确判断、理由和注意事项。"
                        "只能使用工具结果和给定上下文，不要编造来源。引用来源时使用 [C1]。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"用户问题：{request.user_query}\n\n"
                        f"已选择：skill={route.skill}, tool={route.tool_name}\n\n"
                        f"工具结果 JSON：\n{json.dumps(tool_payload, ensure_ascii=False, default=str)}\n\n"
                        f"可引用来源：\n{evidence_block or '无'}\n\n"
                        "请用简洁中文回答用户完整问题。"
                    ),
                },
            ],
            "temperature": 0.2,
            "max_tokens": 700,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def answer_prompt(request: AnswerWithCitationsRequest) -> dict[str, object]:
    plan_block = request.plan_summary or "未提供显式计划。"
    task_block = plan_text(request.plan_tasks)
    trace_block = execution_trace_text(request.execution_trace)
    evidence_block = "\n".join(
        f"[{item.label}] {item.title}\n{item.snippet}\nsource={item.source_path}"
        for item in request.evidence_items[: settings.retrieval_limit]
    )
    return {
        "model": settings.llm_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a plan-and-solve research copilot. Answer in concise Chinese markdown. "
                    "Use only the supplied project context and evidence. "
                    "Do not invent citations. Refer to evidence labels like [C1]."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"项目上下文：\n{request.packed_context}\n\n"
                    f"执行计划摘要：\n{plan_block}\n\n"
                    f"执行步骤：\n{task_block or '1. 直接回答'}\n\n"
                    f"Solver 执行痕迹：\n{trace_block or '1. 直接回答'}\n\n"
                    f"用户问题：{request.user_query}\n\n"
                    f"检索证据：\n{evidence_block or '暂无命中文档证据'}\n\n"
                    "请输出：\n"
                    "1. 结论\n"
                    "2. 关键依据\n"
                    "3. 下一步建议\n"
                    "要求：只基于给定证据回答，引用使用 [C1] 这种格式。"
                ),
            },
        ],
        "temperature": 0.2,
        "max_tokens": 900,
    }


def llm_answer_stream_chunks(request: AnswerWithCitationsRequest) -> Iterator[str]:
    prompt = answer_prompt(request)
    prompt["stream"] = True
    with httpx.stream(
        "POST",
        f"{settings.llm_api_base.rstrip('/')}/chat/completions",
        headers=llm_headers(),
        json=prompt,
        timeout=120.0,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = str(chunk.get("choices", [{}])[0].get("delta", {}).get("content", "") or "")
            if delta:
                yield delta


def citations_from_evidence(evidence_items: list[EvidenceItem]) -> list[Citation]:
    return [
        Citation(asset_id=item.asset_id, chunk_id=item.chunk_id, label=item.label, score=item.score)
        for item in evidence_items
    ]


def answer_with_citations(request: AnswerWithCitationsRequest) -> AnswerResponse:
    citations = citations_from_evidence(request.evidence_items)
    if not request.evidence_items:
        fallback = try_direct_llm_answer(
            request,
            reason="local retrieval returned no evidence",
            packed_context=request.packed_context,
        )
        if fallback:
            return AnswerResponse(
                project_id=request.project_id,
                session_id=request.session_id,
                sequence_id=request.sequence_id,
                answer=fallback,
                citations=citations,
            )
    task_block = plan_text(request.plan_tasks)
    trace_block = execution_trace_text(request.execution_trace[:3])
    evidence_lines = [f"- [{item.label}] {item.title}: {item.snippet}" for item in request.evidence_items[:3]]
    answer = "\n".join(
        [
            f"问题：{request.user_query}",
            "",
            "计划：",
            request.plan_summary or "按当前上下文直接求解。",
            task_block or "1. 直接回答",
            "",
            "执行：",
            trace_block or "1. 直接回答",
            "",
            "结论：",
            "当前已完成项目级检索，并整理出可用证据。",
            "",
            "关键依据：",
            "\n".join(evidence_lines) or "- 当前没有命中文档证据。",
            "",
            "下一步建议：",
            "- 继续补充全局知识库中的论文、代码说明或笔记。",
            "- 结合当前项目的情景记忆与语义记忆继续追问。",
        ]
    )
    if settings.llm_provider == "deepseek" and settings.llm_api_key:
        try:
            answer = llm_answer_markdown(request)
        except httpx.HTTPError:
            pass
    if contains_restrictive_fallback_text(answer):
        fallback = try_direct_llm_answer(
            request,
            reason="retrieved evidence did not support a usable answer",
            packed_context=request.packed_context,
            tool_payload={"evidence_count": len(request.evidence_items)},
        )
        if fallback:
            answer = fallback
            citations = []
    return AnswerResponse(
        project_id=request.project_id,
        session_id=request.session_id,
        sequence_id=request.sequence_id,
        answer=answer,
        citations=citations,
    )


def persist_research_run(
    db: Session,
    *,
    project: Project,
    session: ChatSession,
    request: TurnScopedRequest,
    context: BuildContextResponse,
    plan: PlanTasksResponse,
    retrieval: RetrieveResponse,
    answer: AnswerResponse,
    memory: ConsolidateMemoryResponse,
    todo: Todo | None,
    trace_id: str,
) -> ResearchRun:
    run = ResearchRun(
        id=make_id("run"),
        project_id=project.id,
        session_id=session.id,
        todo_id=request.todo_id,
        sequence_id=request.sequence_id,
        query=request.user_query,
        status="completed",
        trace_id=trace_id,
        answer_text=answer.answer,
        context_payload=context.model_dump(),
        plan_payload=plan.model_dump(),
        retrieval_payload=retrieval.model_dump(),
        answer_payload=answer.model_dump(),
        memory_payload=memory.model_dump(),
    )
    db.add(run)
    session.last_sequence_id = request.sequence_id
    session.summary = compact_text(answer.answer, 160)
    if request.sequence_id == 1 and session.title == "新会话":
        session.title = compact_text(request.user_query, 48)
    if todo:
        todo.status = "done"
        todo.last_run_id = run.id
    touch_session(session)
    touch_project(project)
    db.commit()
    db.refresh(run)
    return run


def consolidate_memory(
    db: Session,
    request: ConsolidateMemoryRequest,
    *,
    todo: Todo | None = None,
) -> ConsolidateMemoryResponse:
    updates = consolidate_layered_memories(
        db,
        request.project_id,
        request.session_id,
        request.sequence_id,
        request.user_query,
        request.answer,
        [citation.label for citation in request.citations],
        todo_title=todo.title if todo else None,
    )
    return ConsolidateMemoryResponse(
        project_id=request.project_id,
        session_id=request.session_id,
        sequence_id=request.sequence_id,
        memory_updates=updates,
    )


def run_research(db: Session, request: TurnScopedRequest) -> RunResearchResponse:
    project = get_project(db, request.project_id)
    session = get_chat_session(db, request.project_id, request.session_id)
    todo = get_project_todo(db, project.id, request.todo_id) if request.todo_id else None
    ensure_next_sequence(session, request.sequence_id)
    assets = resolve_assets(db)
    context = build_context(db, request, project=project, session=session, assets=assets, todo=todo)
    tool_decision = choose_live_tool(request, context)
    live_route = tool_decision.route
    if live_route is not None:
        plan, retrieval, answer = execute_live_tool_plan(request, live_route)
    else:
        plan = plan_tasks(request, todo=todo)
        plan, retrieval = execute_plan(request, todo=todo, plan=plan, assets=assets)
        answer = answer_with_citations(
            AnswerWithCitationsRequest(
                project_id=request.project_id,
                session_id=request.session_id,
                sequence_id=request.sequence_id,
                user_query=request.user_query,
                asset_ids=request.asset_ids,
                todo_id=request.todo_id,
                evidence_items=retrieval.evidence_items,
                plan_summary=plan.plan_summary,
                plan_tasks=plan.tasks,
                execution_trace=plan.execution_trace,
                packed_context=context.packed_context,
            )
        )
    memory = consolidate_memory(
        db,
        ConsolidateMemoryRequest(
            project_id=request.project_id,
            session_id=request.session_id,
            sequence_id=request.sequence_id,
            user_query=request.user_query,
            asset_ids=request.asset_ids,
            todo_id=request.todo_id,
            answer=answer.answer,
            citations=answer.citations,
        ),
        todo=todo,
    )
    trace_id = f"trace-{uuid.uuid4().hex[:12]}"
    persist_research_run(
        db,
        project=project,
        session=session,
        request=request,
        context=context,
        plan=plan,
        retrieval=retrieval,
        answer=answer,
        memory=memory,
        todo=todo,
        trace_id=trace_id,
    )
    return RunResearchResponse(
        project_id=project.id,
        session_id=session.id,
        sequence_id=request.sequence_id,
        context=context,
        plan=plan,
        retrieval=retrieval,
        answer=answer,
        memory=memory,
        trace_id=trace_id,
        meta={
            "mode": settings.execution_mode,
            "planner_mode": plan.planner_mode,
            "replan_count": plan.replan_count,
            "tool_planner_mode": tool_decision.planner_mode,
            "tool_planner_reason": tool_decision.reason,
            "tool_planner_fallback_reason": tool_decision.fallback_reason,
            "selected_skill": live_route.skill if live_route else "",
            "selected_tool": live_route.tool_name if live_route else "",
        },
    )


def run_agent_research(db: Session, request: TurnScopedRequest) -> RunResearchResponse:
    project = get_project(db, request.project_id)
    session = get_chat_session(db, request.project_id, request.session_id)
    todo = get_project_todo(db, project.id, request.todo_id) if request.todo_id else None
    ensure_next_sequence(session, request.sequence_id)
    assets = resolve_assets(db)
    context = build_context(db, request, project=project, session=session, assets=assets, todo=todo)
    plan, retrieval, answer = execute_agent_plan(db, request, context)
    memory = consolidate_memory(
        db,
        ConsolidateMemoryRequest(
            project_id=request.project_id,
            session_id=request.session_id,
            sequence_id=request.sequence_id,
            user_query=request.user_query,
            asset_ids=request.asset_ids,
            todo_id=request.todo_id,
            answer=answer.answer,
            citations=answer.citations,
        ),
        todo=todo,
    )
    trace_id = f"trace-{uuid.uuid4().hex[:12]}"
    persist_research_run(
        db,
        project=project,
        session=session,
        request=request,
        context=context,
        plan=plan,
        retrieval=retrieval,
        answer=answer,
        memory=memory,
        todo=todo,
        trace_id=trace_id,
    )
    return RunResearchResponse(
        project_id=project.id,
        session_id=session.id,
        sequence_id=request.sequence_id,
        context=context,
        plan=plan,
        retrieval=retrieval,
        answer=answer,
        memory=memory,
        trace_id=trace_id,
        meta={
            "mode": "agent_loop",
            "planner_mode": plan.planner_mode,
            "agent_max_steps": settings.agent_max_steps,
            "selected_tools": [
                step.title.replace("Observation: ", "")
                for step in plan.execution_trace
                if step.action == "agent_observation"
            ],
        },
    )


def stream_research_events(db: Session, request: TurnScopedRequest) -> Iterator[dict[str, object]]:
    project = get_project(db, request.project_id)
    session = get_chat_session(db, request.project_id, request.session_id)
    todo = get_project_todo(db, project.id, request.todo_id) if request.todo_id else None
    ensure_next_sequence(session, request.sequence_id)
    assets = resolve_assets(db)
    context = build_context(db, request, project=project, session=session, assets=assets, todo=todo)
    tool_decision = choose_live_tool(request, context)
    live_route = tool_decision.route
    if live_route is not None:
        initial_plan = plan_live_tool_route(request, live_route)
        yield {"type": "plan", "plan": initial_plan.model_dump(mode="json")}
        trace_events: list[PlanExecutionStep] = []
        plan, retrieval, answer = execute_live_tool_plan(
            request,
            live_route,
            on_step=trace_events.append,
        )
        for step in trace_events:
            yield {"type": "trace", "step": step.model_dump(mode="json")}
        yield {
            "type": "solver_summary",
            "solver_summary": plan.solver_summary,
            "replan_count": plan.replan_count,
            "replan_reason": plan.replan_reason,
        }
        yield {"type": "answer_delta", "delta": answer.answer, "answer": answer.answer}
        memory = consolidate_memory(
            db,
            ConsolidateMemoryRequest(
                project_id=request.project_id,
                session_id=request.session_id,
                sequence_id=request.sequence_id,
                user_query=request.user_query,
                asset_ids=request.asset_ids,
                todo_id=request.todo_id,
                answer=answer.answer,
                citations=answer.citations,
            ),
            todo=todo,
        )
        trace_id = f"trace-{uuid.uuid4().hex[:12]}"
        run = persist_research_run(
            db,
            project=project,
            session=session,
            request=request,
            context=context,
            plan=plan,
            retrieval=retrieval,
            answer=answer,
            memory=memory,
            todo=todo,
            trace_id=trace_id,
        )
        yield {
            "type": "complete",
            "run": run_to_detail(run).model_dump(mode="json"),
        }
        return
    plan = plan_tasks(request, todo=todo)
    yield {"type": "plan", "plan": plan.model_dump(mode="json")}
    plan, retrieval = execute_plan(request, todo=todo, plan=plan, assets=assets)
    for step in plan.execution_trace:
        yield {"type": "trace", "step": step.model_dump(mode="json")}
    yield {
        "type": "solver_summary",
        "solver_summary": plan.solver_summary,
        "replan_count": plan.replan_count,
        "replan_reason": plan.replan_reason,
    }
    answer_request = AnswerWithCitationsRequest(
        project_id=request.project_id,
        session_id=request.session_id,
        sequence_id=request.sequence_id,
        user_query=request.user_query,
        asset_ids=request.asset_ids,
        todo_id=request.todo_id,
        evidence_items=retrieval.evidence_items,
        plan_summary=plan.plan_summary,
        plan_tasks=plan.tasks,
        execution_trace=plan.execution_trace,
        packed_context=context.packed_context,
    )
    answer_text = ""
    if answer_request.evidence_items and settings.llm_provider == "deepseek" and settings.llm_api_key:
        try:
            chunks = []
            for delta in llm_answer_stream_chunks(answer_request):
                chunks.append(delta)
                answer_text = "".join(chunks)
                yield {"type": "answer_delta", "delta": delta, "answer": answer_text}
        except httpx.HTTPError:
            answer_text = ""
    if not answer_text.strip():
        answer = answer_with_citations(answer_request)
        answer_text = answer.answer
        yield {"type": "answer_delta", "delta": answer_text, "answer": answer_text}
    else:
        answer = AnswerResponse(
            project_id=request.project_id,
            session_id=request.session_id,
            sequence_id=request.sequence_id,
            answer=answer_text,
            citations=citations_from_evidence(retrieval.evidence_items),
        )
    memory = consolidate_memory(
        db,
        ConsolidateMemoryRequest(
            project_id=request.project_id,
            session_id=request.session_id,
            sequence_id=request.sequence_id,
            user_query=request.user_query,
            asset_ids=request.asset_ids,
            todo_id=request.todo_id,
            answer=answer.answer,
            citations=answer.citations,
        ),
        todo=todo,
    )
    trace_id = f"trace-{uuid.uuid4().hex[:12]}"
    run = persist_research_run(
        db,
        project=project,
        session=session,
        request=request,
        context=context,
        plan=plan,
        retrieval=retrieval,
        answer=answer,
        memory=memory,
        todo=todo,
        trace_id=trace_id,
    )
    yield {
        "type": "complete",
        "run": run_to_detail(run).model_dump(mode="json"),
    }


def stream_agent_events(db: Session, request: TurnScopedRequest) -> Iterator[dict[str, object]]:
    project = get_project(db, request.project_id)
    session = get_chat_session(db, request.project_id, request.session_id)
    todo = get_project_todo(db, project.id, request.todo_id) if request.todo_id else None
    ensure_next_sequence(session, request.sequence_id)
    assets = resolve_assets(db)
    context = build_context(db, request, project=project, session=session, assets=assets, todo=todo)
    initial_plan = agent_plan(request)
    yield {"type": "plan", "plan": initial_plan.model_dump(mode="json")}
    trace_events: list[PlanExecutionStep] = []
    plan, retrieval, answer = execute_agent_plan(db, request, context, on_step=trace_events.append)
    for step in trace_events:
        yield {"type": "trace", "step": step.model_dump(mode="json")}
    yield {
        "type": "solver_summary",
        "solver_summary": plan.solver_summary,
        "replan_count": plan.replan_count,
        "replan_reason": plan.replan_reason,
    }
    yield {"type": "answer_delta", "delta": answer.answer, "answer": answer.answer}
    memory = consolidate_memory(
        db,
        ConsolidateMemoryRequest(
            project_id=request.project_id,
            session_id=request.session_id,
            sequence_id=request.sequence_id,
            user_query=request.user_query,
            asset_ids=request.asset_ids,
            todo_id=request.todo_id,
            answer=answer.answer,
            citations=answer.citations,
        ),
        todo=todo,
    )
    trace_id = f"trace-{uuid.uuid4().hex[:12]}"
    run = persist_research_run(
        db,
        project=project,
        session=session,
        request=request,
        context=context,
        plan=plan,
        retrieval=retrieval,
        answer=answer,
        memory=memory,
        todo=todo,
        trace_id=trace_id,
    )
    yield {"type": "complete", "run": run_to_detail(run).model_dump(mode="json")}


def create_and_run(
    db: Session,
    project_id: str,
    session_id: str,
    payload: ResearchTurnRequest,
) -> ResearchRunDetailResponse:
    result = run_research(
        db,
        TurnScopedRequest(
            project_id=project_id,
            session_id=session_id,
            sequence_id=payload.sequence_id,
            user_query=payload.user_query,
            asset_ids=payload.asset_ids,
            todo_id=payload.todo_id,
        ),
    )
    run = db.scalar(select(ResearchRun).where(ResearchRun.trace_id == result.trace_id))
    return get_run(db, run.id)


def create_agent_and_run(
    db: Session,
    project_id: str,
    session_id: str,
    payload: ResearchTurnRequest,
) -> ResearchRunDetailResponse:
    result = run_agent_research(
        db,
        TurnScopedRequest(
            project_id=project_id,
            session_id=session_id,
            sequence_id=payload.sequence_id,
            user_query=payload.user_query,
            asset_ids=payload.asset_ids,
            todo_id=payload.todo_id,
        ),
    )
    run = db.scalar(select(ResearchRun).where(ResearchRun.trace_id == result.trace_id))
    return get_run(db, run.id)
