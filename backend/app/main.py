from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db, init_db
from app.models import (
    AnswerWithCitationsRequest,
    AssetCreate,
    AssetResponse,
    AssetUpdate,
    ConsolidateMemoryRequest,
    DashboardResponse,
    MemoryRecordResponse,
    ProjectCreate,
    ProjectResponse,
    ProjectScopedRequest,
    ProjectUpdate,
    ProviderConfigResponse,
    ResearchRunDetailResponse,
    ResearchRunRequest,
    ResearchRunSummaryResponse,
    TodoCreate,
    TodoResponse,
    TodoUpdate,
)
from app.services import (
    answer_with_citations,
    build_context,
    consolidate_memory,
    create_and_run,
    create_asset,
    create_project,
    create_todo,
    dashboard,
    decode_text_content,
    delete_asset,
    delete_project,
    delete_todo,
    get_project,
    get_run,
    hybrid_retrieve,
    list_assets,
    list_memory,
    list_projects,
    list_runs,
    list_todos,
    plan_tasks,
    project_to_response,
    provider_config,
    resolve_assets,
    run_research,
    update_asset,
    update_project,
    update_todo,
    uploaded_asset_title,
)
from app.semantic_store import ensure_semantic_memory_store
from app.vector_store import ensure_vector_store


static_dir = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    ensure_vector_store()
    ensure_semantic_memory_store()
    yield


app = FastAPI(
    title="Research Copilot Runtime API",
    version=settings.app_version,
    description="Plan-and-solve local research workspace with project TODOs, assets, runs, and memory.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def workspace() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/healthz")
def healthz() -> dict[str, object]:
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


@app.get("/api/v1/config/providers", response_model=ProviderConfigResponse)
def provider_config_endpoint() -> ProviderConfigResponse:
    return provider_config()


@app.get("/api/v1/dashboard", response_model=DashboardResponse)
def dashboard_endpoint(db: Session = Depends(get_db)) -> DashboardResponse:
    return dashboard(db)


@app.get("/api/v1/projects", response_model=list[ProjectResponse])
def list_projects_endpoint(db: Session = Depends(get_db)) -> list[ProjectResponse]:
    return list_projects(db)


@app.post("/api/v1/projects", response_model=ProjectResponse)
def create_project_endpoint(payload: ProjectCreate, db: Session = Depends(get_db)) -> ProjectResponse:
    return create_project(db, payload)


@app.get("/api/v1/projects/{project_id}", response_model=ProjectResponse)
def get_project_endpoint(project_id: str, db: Session = Depends(get_db)) -> ProjectResponse:
    try:
        return project_to_response(db, get_project(db, project_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/api/v1/projects/{project_id}", response_model=ProjectResponse)
def update_project_endpoint(project_id: str, payload: ProjectUpdate, db: Session = Depends(get_db)) -> ProjectResponse:
    try:
        return update_project(db, project_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/v1/projects/{project_id}")
def delete_project_endpoint(project_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        delete_project(db, project_id)
        return {"status": "deleted"}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/projects/{project_id}/assets", response_model=list[AssetResponse])
def list_assets_endpoint(project_id: str, db: Session = Depends(get_db)) -> list[AssetResponse]:
    try:
        return list_assets(db, project_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/projects/{project_id}/assets", response_model=AssetResponse)
def create_asset_endpoint(project_id: str, payload: AssetCreate, db: Session = Depends(get_db)) -> AssetResponse:
    try:
        return create_asset(db, project_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/projects/{project_id}/assets/upload-text", response_model=AssetResponse)
async def upload_text_asset_endpoint(
    project_id: str,
    file: UploadFile = File(...),
    asset_type: str = Form("note"),
    title: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> AssetResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(raw) > settings.upload_max_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file exceeds size limit")
    try:
        payload = AssetCreate(
            title=uploaded_asset_title(file.filename or "uploaded.txt", title),
            asset_type=asset_type,
            content=decode_text_content(raw),
        )
        return create_asset(db, project_id, payload)
    except ValueError as exc:
        status_code = 404 if "Project not found" in str(exc) else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@app.delete("/api/v1/assets/{asset_id}")
def delete_asset_endpoint(asset_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        delete_asset(db, asset_id)
        return {"status": "deleted"}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/api/v1/assets/{asset_id}", response_model=AssetResponse)
def update_asset_endpoint(asset_id: str, payload: AssetUpdate, db: Session = Depends(get_db)) -> AssetResponse:
    try:
        return update_asset(db, asset_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/projects/{project_id}/todos", response_model=list[TodoResponse])
def list_todos_endpoint(project_id: str, db: Session = Depends(get_db)) -> list[TodoResponse]:
    try:
        return list_todos(db, project_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/projects/{project_id}/todos", response_model=TodoResponse)
def create_todo_endpoint(project_id: str, payload: TodoCreate, db: Session = Depends(get_db)) -> TodoResponse:
    try:
        return create_todo(db, project_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/api/v1/todos/{todo_id}", response_model=TodoResponse)
def update_todo_endpoint(todo_id: str, payload: TodoUpdate, db: Session = Depends(get_db)) -> TodoResponse:
    try:
        return update_todo(db, todo_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/v1/todos/{todo_id}")
def delete_todo_endpoint(todo_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        delete_todo(db, todo_id)
        return {"status": "deleted"}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/projects/{project_id}/memory", response_model=list[MemoryRecordResponse])
def list_memory_endpoint(project_id: str, db: Session = Depends(get_db)) -> list[MemoryRecordResponse]:
    try:
        return list_memory(db, project_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/projects/{project_id}/runs", response_model=list[ResearchRunSummaryResponse])
def list_runs_endpoint(project_id: str, db: Session = Depends(get_db)) -> list[ResearchRunSummaryResponse]:
    try:
        return list_runs(db, project_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/runs/{run_id}", response_model=ResearchRunDetailResponse)
def get_run_endpoint(run_id: str, db: Session = Depends(get_db)) -> ResearchRunDetailResponse:
    try:
        return get_run(db, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/projects/{project_id}/run", response_model=ResearchRunDetailResponse)
def create_run_endpoint(project_id: str, payload: ResearchRunRequest, db: Session = Depends(get_db)) -> ResearchRunDetailResponse:
    try:
        return create_and_run(db, project_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/runtime/build_context")
def build_context_endpoint(payload: ProjectScopedRequest, db: Session = Depends(get_db)):
    try:
        project = get_project(db, payload.project_id)
        return build_context(
            db,
            payload,
            project=project,
            assets=resolve_assets(db, payload.project_id),
        ).model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/runtime/plan_tasks")
def plan_tasks_endpoint(payload: ProjectScopedRequest):
    return plan_tasks(payload).model_dump()


@app.post("/api/v1/runtime/hybrid_retrieve")
def hybrid_retrieve_endpoint(payload: ProjectScopedRequest, db: Session = Depends(get_db)):
    try:
        return hybrid_retrieve(payload, assets=resolve_assets(db, payload.project_id)).model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/runtime/answer_with_citations")
def answer_with_citations_endpoint(payload: AnswerWithCitationsRequest):
    return answer_with_citations(payload).model_dump()


@app.post("/api/v1/runtime/consolidate_memory")
def consolidate_memory_endpoint(payload: ConsolidateMemoryRequest, db: Session = Depends(get_db)):
    return consolidate_memory(db, payload).model_dump()


@app.post("/api/v1/runtime/research/run")
def run_research_endpoint(payload: ProjectScopedRequest, db: Session = Depends(get_db)):
    try:
        return run_research(db, payload).model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
