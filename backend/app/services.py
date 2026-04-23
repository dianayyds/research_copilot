from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db_models import Asset, Project, ResearchRun, Todo
from app.memory_manager import (
    build_memory_context_bundle,
    consolidate_layered_memories,
    list_layered_memories,
)
from app.models import (
    AnswerResponse,
    AnswerWithCitationsRequest,
    AssetCreate,
    AssetResponse,
    AssetUpdate,
    BuildContextResponse,
    Citation,
    ConsolidateMemoryRequest,
    ConsolidateMemoryResponse,
    DashboardResponse,
    EvidenceItem,
    MemoryRecordResponse,
    PlanTasksResponse,
    ProjectCreate,
    ProjectResponse,
    ProjectScopedRequest,
    ProjectUpdate,
    ProviderConfigResponse,
    ProviderInfo,
    ResearchRunDetailResponse,
    ResearchRunRequest,
    ResearchRunSummaryResponse,
    ResearchTask,
    RetrieveResponse,
    RunResearchResponse,
    TodoCreate,
    TodoResponse,
    TodoUpdate,
)
from app.semantic_store import get_semantic_memory_store
from app.vector_store import get_vector_store


@dataclass
class ChunkRecord:
    asset_id: str
    chunk_id: str
    title: str
    asset_type: str
    content: str
    source_path: str


def make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def ensure_session_id(session_id: str | None) -> str:
    return session_id or f"session-{uuid.uuid4().hex[:12]}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


def project_to_response(db: Session, project: Project) -> ProjectResponse:
    asset_count = db.query(Asset).filter(Asset.project_id == project.id).count()
    todo_count = db.query(Todo).filter(Todo.project_id == project.id).count()
    run_count = db.query(ResearchRun).filter(ResearchRun.project_id == project.id).count()
    return ProjectResponse(
        id=project.id,
        title=project.title,
        description=project.description,
        status=project.status,
        asset_count=asset_count,
        todo_count=todo_count,
        run_count=run_count,
        created_at=project.created_at,
        updated_at=project.updated_at,
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


def run_to_summary(run: ResearchRun) -> ResearchRunSummaryResponse:
    return ResearchRunSummaryResponse(
        id=run.id,
        project_id=run.project_id,
        todo_id=run.todo_id,
        session_id=run.session_id,
        query=run.query,
        status=run.status,
        answer_preview=run.answer_text[:180],
        trace_id=run.trace_id,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


def provider_config() -> ProviderConfigResponse:
    return ProviderConfigResponse(
        llm=ProviderInfo(
            provider=settings.llm_provider,
            model=settings.llm_model,
            mode="api_ready",
        ),
        embedding=ProviderInfo(
            provider=settings.embedding_provider,
            model=settings.embedding_model,
            mode="local_hybrid_rag",
        ),
        reranker=ProviderInfo(
            provider=settings.reranker_provider,
            model=settings.reranker_model,
            mode="local_cross_encoder",
        ),
        execution_mode=settings.execution_mode,
    )


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


def get_project(db: Session, project_id: str) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError("Project not found")
    return project


def touch_project(project: Project) -> None:
    project.updated_at = utc_now()


def get_project_todo(db: Session, project_id: str, todo_id: str) -> Todo:
    todo = db.scalar(select(Todo).where(Todo.id == todo_id, Todo.project_id == project_id))
    if todo is None:
        raise ValueError("Todo not found in project")
    return todo


def update_project(db: Session, project_id: str, payload: ProjectUpdate) -> ProjectResponse:
    project = get_project(db, project_id)
    updates = payload.model_dump(exclude_none=True)
    for key, value in updates.items():
        setattr(project, key, value)
    db.commit()
    db.refresh(project)
    return project_to_response(db, project)


def delete_project(db: Session, project_id: str) -> None:
    project = get_project(db, project_id)
    get_vector_store().delete_project(project_id)
    get_semantic_memory_store().delete_project(project_id)
    db.delete(project)
    db.commit()


def list_assets(db: Session, project_id: str) -> list[AssetResponse]:
    get_project(db, project_id)
    assets = db.scalars(select(Asset).where(Asset.project_id == project_id).order_by(desc(Asset.updated_at))).all()
    return [asset_to_response(asset) for asset in assets]


def create_asset(db: Session, project_id: str, payload: AssetCreate) -> AssetResponse:
    project = get_project(db, project_id)
    asset = Asset(
        id=make_id("asset"),
        project_id=project_id,
        title=payload.title,
        asset_type=payload.asset_type,
        content=payload.content,
    )
    db.add(asset)
    db.flush()
    sync_asset_chunks(project_id, [asset], [asset.id])
    touch_project(project)
    db.commit()
    db.refresh(asset)
    return asset_to_response(asset)


def delete_asset(db: Session, asset_id: str) -> None:
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise ValueError("Asset not found")
    project = get_project(db, asset.project_id)
    get_vector_store().delete_asset(asset.id)
    db.delete(asset)
    touch_project(project)
    db.commit()


def update_asset(db: Session, asset_id: str, payload: AssetUpdate) -> AssetResponse:
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise ValueError("Asset not found")
    updates = payload.model_dump(exclude_none=True)
    for key, value in updates.items():
        setattr(asset, key, value)
    db.flush()
    sync_asset_chunks(asset.project_id, [asset], [asset.id])
    touch_project(get_project(db, asset.project_id))
    db.commit()
    db.refresh(asset)
    return asset_to_response(asset)


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


def get_todo(db: Session, todo_id: str) -> Todo:
    todo = db.get(Todo, todo_id)
    if todo is None:
        raise ValueError("Todo not found")
    return todo


def update_todo(db: Session, todo_id: str, payload: TodoUpdate) -> TodoResponse:
    todo = get_todo(db, todo_id)
    updates = payload.model_dump(exclude_none=True)
    for key, value in updates.items():
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


def list_runs(db: Session, project_id: str) -> list[ResearchRunSummaryResponse]:
    get_project(db, project_id)
    runs = db.scalars(
        select(ResearchRun).where(ResearchRun.project_id == project_id).order_by(desc(ResearchRun.created_at))
    ).all()
    return [run_to_summary(run) for run in runs]


def get_run(db: Session, run_id: str) -> ResearchRunDetailResponse:
    run = db.get(ResearchRun, run_id)
    if run is None:
        raise ValueError("Run not found")
    return ResearchRunDetailResponse(
        id=run.id,
        project_id=run.project_id,
        todo_id=run.todo_id,
        session_id=run.session_id,
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


def build_context(db: Session, request: ProjectScopedRequest, *, project: Project, assets: list[Asset]) -> BuildContextResponse:
    session_id = ensure_session_id(request.session_id)
    memory_bundle = build_memory_context_bundle(db, project.id, session_id, request.user_query)
    asset_scope = ", ".join(asset.title for asset in assets[:6]) or "no assets yet"
    packed_context = "\n\n".join(
        [
            "## Instructions",
            f"Execution mode: {settings.execution_mode}. Use concise project-grounded reasoning.",
            "## Evidence Scope",
            f"Project '{project.title}' currently includes {len(assets)} assets: {asset_scope}.",
            "## Task State",
            f"Research request: {request.user_query}",
            "## Memory",
            memory_bundle.combined_text,
        ]
    )
    return BuildContextResponse(
        project_id=project.id,
        session_id=session_id,
        instruction_context=f"LLM provider target: {settings.llm_provider}/{settings.llm_model}",
        evidence_context=f"Assets available: {len(assets)}",
        task_state_context=request.user_query,
        memory_context=memory_bundle.combined_text,
        working_memory_context=memory_bundle.working_text,
        episodic_memory_context=memory_bundle.episodic_text,
        semantic_memory_context=memory_bundle.semantic_text,
        packed_context=packed_context,
    )


def plan_tasks(request: ProjectScopedRequest, *, todo: Todo | None = None) -> PlanTasksResponse:
    session_id = ensure_session_id(request.session_id)
    subject = todo.title if todo else request.user_query
    tasks = [
        ResearchTask(task_id="task-1", title="Clarify goal", goal=f"Clarify the objective of: {subject}"),
        ResearchTask(task_id="task-2", title="Collect evidence", goal="Retrieve the most relevant project assets."),
        ResearchTask(task_id="task-3", title="Cross-check", goal="Compare retrieved evidence for consistency and gaps."),
        ResearchTask(task_id="task-4", title="Synthesize", goal="Write the final answer and update project memory."),
    ]
    return PlanTasksResponse(project_id=request.project_id, session_id=session_id, tasks=tasks)


def build_chunks(assets: list[Asset], asset_ids: list[str]) -> list[ChunkRecord]:
    allowed_assets = {asset_id for asset_id in asset_ids if asset_id}
    selected_assets = [asset for asset in assets if not allowed_assets or asset.id in allowed_assets]
    chunks: list[ChunkRecord] = []
    for asset in selected_assets:
        for index, chunk_text in enumerate(split_chunks(asset.content), start=1):
            chunks.append(
                ChunkRecord(
                    asset_id=asset.id,
                    chunk_id=f"{asset.id}-chunk-{index:03d}",
                    title=asset.title,
                    asset_type=asset.asset_type,
                    content=chunk_text,
                    source_path=f"/projects/{asset.project_id}/assets/{asset.id}",
                )
            )
    return chunks


def chunk_payloads(project_id: str, assets: list[Asset], asset_ids: list[str]) -> list[dict[str, str]]:
    return [
        {
            "project_id": project_id,
            "asset_id": chunk.asset_id,
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "asset_type": chunk.asset_type,
            "content": chunk.content,
            "source_path": chunk.source_path,
        }
        for chunk in build_chunks(assets, asset_ids)
    ]


def sync_asset_chunks(project_id: str, assets: list[Asset], asset_ids: list[str]) -> None:
    payloads = chunk_payloads(project_id, assets, asset_ids)
    if not payloads:
        return
    get_vector_store().upsert_chunks(project_id, payloads)


def hybrid_retrieve(request: ProjectScopedRequest, *, assets: list[Asset]) -> RetrieveResponse:
    session_id = ensure_session_id(request.session_id)
    payloads = chunk_payloads(request.project_id, assets, request.asset_ids)
    if payloads:
        get_vector_store().upsert_chunks(request.project_id, payloads)
    ranked = get_vector_store().search(
        request.project_id,
        request.user_query,
        settings.retrieval_limit,
        request.asset_ids,
        payloads,
    )
    evidence_items = [
        EvidenceItem(
            asset_id=str(chunk["asset_id"]),
            chunk_id=str(chunk["chunk_id"]),
            label=f"C{index}",
            title=str(chunk["title"]),
            snippet=str(chunk["content"])[:320],
            source_path=str(chunk["source_path"]),
            score=round(float(chunk["score"]), 4),
            tags=[
                str(chunk["asset_type"]),
                settings.embedding_provider,
                settings.vector_store_provider,
                "hybrid",
                "reranked",
            ],
        )
        for index, chunk in enumerate(ranked, start=1)
    ]
    return RetrieveResponse(
        project_id=request.project_id,
        session_id=session_id,
        retrieval_mode=f"{settings.vector_store_provider}_hybrid_rerank",
        evidence_items=evidence_items,
    )


def llm_answer_markdown(request: AnswerWithCitationsRequest) -> str:
    evidence_block = "\n".join(
        f"[{item.label}] {item.title}\n{item.snippet}\nsource={item.source_path}"
        for item in request.evidence_items[: settings.retrieval_limit]
    )
    prompt = {
        "model": settings.llm_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a research copilot. Answer in concise Chinese markdown. "
                    "Use only the supplied project context and evidence. "
                    "Do not invent citations. Refer to evidence labels like [C1]."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"项目上下文：\n{request.packed_context}\n\n"
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
    response = httpx.post(
        f"{settings.llm_api_base.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
        },
        json=prompt,
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def answer_with_citations(request: AnswerWithCitationsRequest) -> AnswerResponse:
    session_id = ensure_session_id(request.session_id)
    evidence_items = request.evidence_items
    citations = [
        Citation(asset_id=item.asset_id, chunk_id=item.chunk_id, label=item.label, score=item.score)
        for item in evidence_items
    ]
    evidence_lines = [f"- [{item.label}] {item.title}: {item.snippet}" for item in evidence_items[:3]]
    fallback = "\n".join(
        [
            f"问题：{request.user_query}",
            "",
            "结论：",
            "当前已完成项目级检索，并整理出可用证据。",
            "",
            "关键依据：",
            "\n".join(evidence_lines) or "- 当前没有命中文档证据。",
            "",
            "下一步建议：",
            "- 继续补充论文、代码注释或实验笔记。",
            "- 对关键结论进行人工复核，再沉淀到长期记忆。",
        ]
    )
    answer = fallback
    if settings.llm_provider == "deepseek" and settings.llm_api_key:
        try:
            answer = llm_answer_markdown(request)
        except httpx.HTTPError:
            answer = fallback
    return AnswerResponse(project_id=request.project_id, session_id=session_id, answer=answer, citations=citations)


def consolidate_memory(db: Session, request: ConsolidateMemoryRequest, *, todo: Todo | None = None) -> ConsolidateMemoryResponse:
    session_id = ensure_session_id(request.session_id)
    memory_updates = consolidate_layered_memories(
        db,
        request.project_id,
        session_id,
        request.user_query,
        request.answer,
        [citation.label for citation in request.citations],
        todo_title=todo.title if todo else None,
    )
    return ConsolidateMemoryResponse(project_id=request.project_id, session_id=session_id, memory_updates=memory_updates)


def resolve_assets(db: Session, project_id: str) -> list[Asset]:
    return db.scalars(select(Asset).where(Asset.project_id == project_id).order_by(desc(Asset.updated_at))).all()


def run_research(db: Session, request: ProjectScopedRequest) -> RunResearchResponse:
    project = get_project(db, request.project_id)
    todo = get_project_todo(db, project.id, request.todo_id) if request.todo_id else None
    session_id = ensure_session_id(request.session_id)
    normalized_request = ProjectScopedRequest(
        project_id=request.project_id,
        user_query=request.user_query,
        asset_ids=request.asset_ids,
        session_id=session_id,
        todo_id=request.todo_id,
    )
    assets = resolve_assets(db, project.id)
    context = build_context(db, normalized_request, project=project, assets=assets)
    plan = plan_tasks(normalized_request, todo=todo)
    retrieval = hybrid_retrieve(normalized_request, assets=assets)
    answer = answer_with_citations(
        AnswerWithCitationsRequest(
            project_id=project.id,
            user_query=normalized_request.user_query,
            asset_ids=normalized_request.asset_ids,
            session_id=session_id,
            todo_id=request.todo_id,
            evidence_items=retrieval.evidence_items,
            packed_context=context.packed_context,
        )
    )
    memory = consolidate_memory(
        db,
        ConsolidateMemoryRequest(
            project_id=project.id,
            user_query=normalized_request.user_query,
            asset_ids=normalized_request.asset_ids,
            session_id=session_id,
            todo_id=request.todo_id,
            answer=answer.answer,
            citations=answer.citations,
        ),
        todo=todo,
    )
    trace_id = f"trace-{uuid.uuid4().hex[:12]}"
    run = ResearchRun(
        id=make_id("run"),
        project_id=project.id,
        todo_id=request.todo_id,
        session_id=session_id,
        query=normalized_request.user_query,
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
    if todo:
        todo.status = "done"
        todo.last_run_id = run.id
    touch_project(project)
    db.commit()
    return RunResearchResponse(
        project_id=project.id,
        session_id=session_id,
        context=context,
        plan=plan,
        retrieval=retrieval,
        answer=answer,
        memory=memory,
        trace_id=trace_id,
        meta={
            "mode": settings.execution_mode,
            "llm_provider": settings.llm_provider,
            "embedding_provider": settings.embedding_provider,
        },
    )


def create_and_run(db: Session, project_id: str, payload: ResearchRunRequest) -> ResearchRunDetailResponse:
    result = run_research(
        db,
        ProjectScopedRequest(
            project_id=project_id,
            user_query=payload.user_query,
            asset_ids=payload.asset_ids,
            session_id=payload.session_id,
            todo_id=payload.todo_id,
        ),
    )
    run = db.scalar(select(ResearchRun).where(ResearchRun.trace_id == result.trace_id))
    return get_run(db, run.id)


def dashboard(db: Session) -> DashboardResponse:
    latest_runs = db.scalars(select(ResearchRun).order_by(desc(ResearchRun.created_at)).limit(8)).all()
    return DashboardResponse(
        project_count=db.query(Project).count(),
        todo_count=db.query(Todo).count(),
        open_todo_count=db.query(Todo).filter(Todo.status != "done").count(),
        run_count=db.query(ResearchRun).count(),
        latest_runs=[run_to_summary(run) for run in latest_runs],
    )
