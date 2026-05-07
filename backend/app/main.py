from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from time import perf_counter

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from sqlalchemy.orm import Session

from app.asset_ingest import parse_uploaded_asset
from app.config import settings
from app.db import get_db, init_db
from app.models import (
    AssetCreate,
    AssetResponse,
    AssetUpdate,
    ChatSessionCreate,
    ChatSessionResponse,
    ChatSessionUpdate,
    MemoryRecordResponse,
    ProjectCreate,
    ProjectResponse,
    ProjectUpdate,
    ResumableUploadCompleteRequest,
    ResumableUploadInitRequest,
    ResumableUploadStatusResponse,
    ResearchRunDetailResponse,
    ResearchTurnRequest,
    TurnScopedRequest,
    TodoCreate,
    TodoResponse,
    TodoUpdate,
)
from app.semantic_store import ensure_semantic_memory_store
from app.services import (
    create_agent_and_run,
    create_and_run,
    create_asset,
    create_lats_and_run,
    create_project,
    create_session,
    create_todo,
    complete_resumable_asset_upload,
    delete_asset,
    delete_project,
    delete_session,
    delete_todo,
    get_project,
    get_run,
    list_assets,
    list_memory,
    list_projects,
    list_session_runs,
    list_sessions,
    list_todos,
    project_to_response,
    skill_tool_registry_payload,
    stream_agent_events,
    stream_lats_events,
    stream_research_events,
    init_resumable_asset_upload,
    resumable_upload_status,
    update_asset,
    update_project,
    update_session,
    update_todo,
    upload_resumable_asset_chunk,
)
from app.vector_store import ensure_vector_store


static_dir = Path(__file__).parent / "static"
logger = logging.getLogger("uvicorn.error")


def format_file_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.0f}MB"
    if size >= 1024:
        return f"{size / 1024:.0f}KB"
    return f"{size}B"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    ensure_vector_store()
    ensure_semantic_memory_store()
    yield


app = FastAPI(
    title="Research Copilot Runtime API",
    version=settings.app_version,
    description="Project-scoped chat workspace with layered memory and a global knowledge base.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def workspace() -> HTMLResponse:
    return HTMLResponse(
        (static_dir / "index.html").read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/static/{filename}", include_in_schema=False)
async def static_asset(filename: str) -> Response:
    path = static_dir / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Static file not found")
    media_type = {
        ".js": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
    }.get(path.suffix, "text/plain; charset=utf-8")
    return Response(
        path.read_text(encoding="utf-8"),
        media_type=media_type,
        headers={"Cache-Control": "no-store"},
    )


@app.get("/healthz")
async def healthz() -> dict[str, object]:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "dependencies": {
            "database": {
                "driver": "mysql+pymysql" if settings.resolved_database_url.startswith("mysql") else "sqlite",
                "host": settings.mysql_host,
                "port": settings.mysql_port,
                "name": settings.mysql_database,
            },
            "redis": f"{settings.redis_host}:{settings.redis_port}",
            "minio": f"{settings.minio_host}:{settings.minio_port}",
            "qdrant": f"{settings.qdrant_host}:{settings.qdrant_port}",
        },
    }


@app.get("/api/v1/skills")
async def list_skills_endpoint() -> dict[str, object]:
    return skill_tool_registry_payload()


@app.get("/api/v1/projects", response_model=list[ProjectResponse])
async def list_projects_endpoint(db: Session = Depends(get_db)) -> list[ProjectResponse]:
    return list_projects(db)


@app.post("/api/v1/projects", response_model=ProjectResponse)
async def create_project_endpoint(payload: ProjectCreate, db: Session = Depends(get_db)) -> ProjectResponse:
    return create_project(db, payload)


@app.get("/api/v1/projects/{project_id}", response_model=ProjectResponse)
async def get_project_endpoint(project_id: str, db: Session = Depends(get_db)) -> ProjectResponse:
    try:
        return project_to_response(db, get_project(db, project_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/api/v1/projects/{project_id}", response_model=ProjectResponse)
async def update_project_endpoint(project_id: str, payload: ProjectUpdate, db: Session = Depends(get_db)) -> ProjectResponse:
    try:
        return update_project(db, project_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/v1/projects/{project_id}")
async def delete_project_endpoint(project_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        delete_project(db, project_id)
        return {"status": "deleted"}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/projects/{project_id}/sessions", response_model=list[ChatSessionResponse])
async def list_sessions_endpoint(project_id: str, db: Session = Depends(get_db)) -> list[ChatSessionResponse]:
    try:
        return list_sessions(db, project_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/projects/{project_id}/sessions", response_model=ChatSessionResponse)
async def create_session_endpoint(
    project_id: str,
    payload: ChatSessionCreate,
    db: Session = Depends(get_db),
) -> ChatSessionResponse:
    try:
        return create_session(db, project_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/api/v1/sessions/{session_id}", response_model=ChatSessionResponse)
async def update_session_endpoint(
    session_id: str,
    payload: ChatSessionUpdate,
    db: Session = Depends(get_db),
) -> ChatSessionResponse:
    try:
        return update_session(db, session_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/v1/projects/{project_id}/sessions/{session_id}")
async def delete_session_endpoint(project_id: str, session_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        delete_session(db, project_id, session_id)
        return {"status": "deleted"}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get(
    "/api/v1/projects/{project_id}/sessions/{session_id}/runs",
    response_model=list[ResearchRunDetailResponse],
)
async def list_session_runs_endpoint(
    project_id: str,
    session_id: str,
    db: Session = Depends(get_db),
) -> list[ResearchRunDetailResponse]:
    try:
        return list_session_runs(db, project_id, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post(
    "/api/v1/projects/{project_id}/sessions/{session_id}/run",
    response_model=ResearchRunDetailResponse,
)
async def create_and_run_endpoint(
    project_id: str,
    session_id: str,
    payload: ResearchTurnRequest,
    db: Session = Depends(get_db),
) -> ResearchRunDetailResponse:
    try:
        return create_and_run(db, project_id, session_id, payload)
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc


@app.post("/api/v1/projects/{project_id}/sessions/{session_id}/run/stream")
async def create_and_run_stream_endpoint(
    project_id: str,
    session_id: str,
    payload: ResearchTurnRequest,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    request_payload = {
        "project_id": project_id,
        "session_id": session_id,
        "sequence_id": payload.sequence_id,
        "user_query": payload.user_query,
        "asset_ids": payload.asset_ids,
        "todo_id": payload.todo_id,
    }

    def event_stream():
        try:
            for event in stream_research_events(
                db,
                TurnScopedRequest.model_validate(request_payload),
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except ValueError as exc:
            yield json.dumps({"type": "error", "detail": str(exc)}, ensure_ascii=False) + "\n"
        except Exception as exc:  # pragma: no cover - runtime safeguard
            yield json.dumps({"type": "error", "detail": f"Streaming run failed: {exc}"}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store"},
    )


@app.post(
    "/api/v1/projects/{project_id}/sessions/{session_id}/agent/run",
    response_model=ResearchRunDetailResponse,
)
async def create_agent_and_run_endpoint(
    project_id: str,
    session_id: str,
    payload: ResearchTurnRequest,
    db: Session = Depends(get_db),
) -> ResearchRunDetailResponse:
    try:
        return create_agent_and_run(db, project_id, session_id, payload)
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc


@app.post("/api/v1/projects/{project_id}/sessions/{session_id}/agent/run/stream")
async def create_agent_and_run_stream_endpoint(
    project_id: str,
    session_id: str,
    payload: ResearchTurnRequest,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    request_payload = {
        "project_id": project_id,
        "session_id": session_id,
        "sequence_id": payload.sequence_id,
        "user_query": payload.user_query,
        "asset_ids": payload.asset_ids,
        "todo_id": payload.todo_id,
    }

    def event_stream():
        try:
            for event in stream_agent_events(
                db,
                TurnScopedRequest.model_validate(request_payload),
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except ValueError as exc:
            yield json.dumps({"type": "error", "detail": str(exc)}, ensure_ascii=False) + "\n"
        except Exception as exc:  # pragma: no cover - runtime safeguard
            yield json.dumps({"type": "error", "detail": f"Agent streaming run failed: {exc}"}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store"},
    )


@app.post(
    "/api/v1/projects/{project_id}/sessions/{session_id}/lats/run",
    response_model=ResearchRunDetailResponse,
)
async def create_lats_and_run_endpoint(
    project_id: str,
    session_id: str,
    payload: ResearchTurnRequest,
    db: Session = Depends(get_db),
) -> ResearchRunDetailResponse:
    try:
        return create_lats_and_run(db, project_id, session_id, payload)
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc


@app.post("/api/v1/projects/{project_id}/sessions/{session_id}/lats/run/stream")
async def create_lats_and_run_stream_endpoint(
    project_id: str,
    session_id: str,
    payload: ResearchTurnRequest,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    request_payload = {
        "project_id": project_id,
        "session_id": session_id,
        "sequence_id": payload.sequence_id,
        "user_query": payload.user_query,
        "asset_ids": payload.asset_ids,
        "todo_id": payload.todo_id,
    }

    def event_stream():
        try:
            for event in stream_lats_events(
                db,
                TurnScopedRequest.model_validate(request_payload),
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except ValueError as exc:
            yield json.dumps({"type": "error", "detail": str(exc)}, ensure_ascii=False) + "\n"
        except Exception as exc:  # pragma: no cover - runtime safeguard
            yield json.dumps({"type": "error", "detail": f"LATS streaming run failed: {exc}"}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/v1/runs/{run_id}", response_model=ResearchRunDetailResponse)
async def get_run_endpoint(run_id: str, db: Session = Depends(get_db)) -> ResearchRunDetailResponse:
    try:
        return get_run(db, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/assets", response_model=list[AssetResponse])
async def list_assets_endpoint(db: Session = Depends(get_db)) -> list[AssetResponse]:
    return list_assets(db)


@app.post("/api/v1/assets", response_model=AssetResponse)
async def create_asset_endpoint(payload: AssetCreate, db: Session = Depends(get_db)) -> AssetResponse:
    return create_asset(db, payload)


@app.post("/api/v1/assets/uploads/init", response_model=ResumableUploadStatusResponse)
async def init_resumable_asset_upload_endpoint(
    payload: ResumableUploadInitRequest,
    db: Session = Depends(get_db),
) -> ResumableUploadStatusResponse:
    try:
        return init_resumable_asset_upload(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/assets/uploads/{upload_id}/status", response_model=ResumableUploadStatusResponse)
async def resumable_asset_upload_status_endpoint(
    upload_id: str,
    db: Session = Depends(get_db),
) -> ResumableUploadStatusResponse:
    try:
        return resumable_upload_status(db, upload_id)
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc


@app.post("/api/v1/assets/uploads/{upload_id}/chunks", response_model=ResumableUploadStatusResponse)
async def upload_resumable_asset_chunk_endpoint(
    upload_id: str,
    chunk_index: int = Form(...),
    chunk: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> ResumableUploadStatusResponse:
    try:
        data = await chunk.read()
        if not data:
            raise ValueError("Chunk is empty")
        return upload_resumable_asset_chunk(
            db,
            upload_id,
            chunk_index=chunk_index,
            data=data,
            content_type=chunk.content_type or "application/octet-stream",
        )
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc


@app.post("/api/v1/assets/uploads/{upload_id}/complete", response_model=ResumableUploadStatusResponse)
async def complete_resumable_asset_upload_endpoint(
    upload_id: str,
    payload: ResumableUploadCompleteRequest,
    db: Session = Depends(get_db),
) -> ResumableUploadStatusResponse:
    try:
        return complete_resumable_asset_upload(db, upload_id, payload)
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc


@app.post("/api/v1/assets/upload-file", response_model=AssetResponse)
async def upload_asset_file_endpoint(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    asset_type: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> AssetResponse:
    filename = file.filename or "upload"
    started_at = perf_counter()
    logger.info(
        "asset_upload_started filename=%s size_hint=%s declared_type=%s title=%s",
        filename,
        getattr(file, "size", "unknown"),
        asset_type or "",
        title or "",
    )
    raw = await file.read()
    read_elapsed_ms = (perf_counter() - started_at) * 1000
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(raw) > settings.upload_max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file exceeds {format_file_size(settings.upload_max_bytes)}",
        )
    parse_started_at = perf_counter()
    try:
        parsed = parse_uploaded_asset(filename=filename, raw=raw, title=title, asset_type=asset_type)
    except ValueError as exc:
        logger.warning(
            "asset_upload_parse_failed filename=%s size_bytes=%s read_ms=%.1f parse_ms=%.1f detail=%s",
            filename,
            len(raw),
            read_elapsed_ms,
            (perf_counter() - parse_started_at) * 1000,
            str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    parse_elapsed_ms = (perf_counter() - parse_started_at) * 1000
    save_started_at = perf_counter()
    asset = create_asset(
        db,
        AssetCreate(
            title=parsed.title,
            asset_type=parsed.asset_type,
            content=parsed.content,
        ),
    )
    save_elapsed_ms = (perf_counter() - save_started_at) * 1000
    total_elapsed_ms = (perf_counter() - started_at) * 1000
    logger.info(
        "asset_upload_completed filename=%s asset_id=%s asset_type=%s size_bytes=%s content_chars=%s read_ms=%.1f parse_ms=%.1f save_ms=%.1f total_ms=%.1f",
        filename,
        asset.id,
        asset.asset_type,
        len(raw),
        len(parsed.content),
        read_elapsed_ms,
        parse_elapsed_ms,
        save_elapsed_ms,
        total_elapsed_ms,
    )
    return asset


@app.patch("/api/v1/assets/{asset_id}", response_model=AssetResponse)
async def update_asset_endpoint(asset_id: str, payload: AssetUpdate, db: Session = Depends(get_db)) -> AssetResponse:
    try:
        return update_asset(db, asset_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/v1/assets/{asset_id}")
async def delete_asset_endpoint(asset_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        delete_asset(db, asset_id)
        return {"status": "deleted"}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/projects/{project_id}/todos", response_model=list[TodoResponse])
async def list_todos_endpoint(project_id: str, db: Session = Depends(get_db)) -> list[TodoResponse]:
    try:
        return list_todos(db, project_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/projects/{project_id}/todos", response_model=TodoResponse)
async def create_todo_endpoint(project_id: str, payload: TodoCreate, db: Session = Depends(get_db)) -> TodoResponse:
    try:
        return create_todo(db, project_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/api/v1/todos/{todo_id}", response_model=TodoResponse)
async def update_todo_endpoint(todo_id: str, payload: TodoUpdate, db: Session = Depends(get_db)) -> TodoResponse:
    try:
        return update_todo(db, todo_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/v1/todos/{todo_id}")
async def delete_todo_endpoint(todo_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        delete_todo(db, todo_id)
        return {"status": "deleted"}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/projects/{project_id}/memory", response_model=list[MemoryRecordResponse])
async def list_memory_endpoint(project_id: str, db: Session = Depends(get_db)) -> list[MemoryRecordResponse]:
    try:
        return list_memory(db, project_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
