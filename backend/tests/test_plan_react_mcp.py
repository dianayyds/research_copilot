from __future__ import annotations

import json
import os
from types import SimpleNamespace

import httpx
import pytest

os.environ["DATABASE_URL"] = "sqlite:////tmp/research_copilot_test.db"
os.environ["VECTOR_STORE_PROVIDER"] = "stub"
os.environ["EMBEDDING_PROVIDER"] = "stub"
os.environ["LLM_PROVIDER"] = "stub"
os.environ["LLM_API_KEY"] = ""
os.environ["EXECUTION_MODE"] = "plan_react_mcp"

from app.db import Base, engine
from app.main import app
from app.models import PlanTasksResponse, ResearchTask
from app.services import PlanReactDecision
from app.semantic_store import reset_semantic_memory_store
from app.vector_store import reset_vector_store


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def reset_db() -> None:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    reset_vector_store()
    reset_semantic_memory_store()


def make_client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


async def create_project(client: httpx.AsyncClient, title: str) -> dict:
    response = await client.post("/api/v1/projects", json={"title": title, "description": "", "status": "active"})
    return response.json()


async def create_session(client: httpx.AsyncClient, project_id: str) -> dict:
    response = await client.post(f"/api/v1/projects/{project_id}/sessions", json={"title": "新会话"})
    return response.json()


class FakeMCPClient:
    def __init__(self) -> None:
        self.config = SimpleNamespace(name="github", transport="stdio")
        self.server_info = {"name": "fake-github"}
        self.call_count = 0

    def __enter__(self) -> "FakeMCPClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def list_tools(self) -> list[dict]:
        return [
            {"name": "list_issues", "description": "List issues", "annotations": {"readOnlyHint": True}},
            {"name": "create_issue", "description": "Create issue"},
        ]

    def list_resources(self) -> list[dict]:
        return [{"uri": "repo://README.md", "name": "README"}]

    def list_prompts(self) -> list[dict]:
        return [{"name": "summarize_repo", "description": "Summarize repo"}]

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        self.call_count += 1
        return {"content": [{"type": "text", "text": f"{name} returned issue #1"}]}

    def read_resource(self, uri: str) -> dict:
        return {"contents": [{"uri": uri, "text": "README content"}]}

    def get_prompt(self, name: str, arguments: dict | None = None) -> dict:
        return {"messages": [{"role": "user", "content": {"type": "text", "text": "prompt"}}]}


def fake_plan(request, context, catalog) -> PlanTasksResponse:
    return PlanTasksResponse(
        project_id=request.project_id,
        session_id=request.session_id,
        sequence_id=request.sequence_id,
        planner_mode="plan_react_mcp",
        plan_summary="先查 issue，再读 README。",
        tasks=[
            ResearchTask(
                task_id="task-1",
                title="查 GitHub issues",
                goal="读取 issue 列表",
                task_type="plan_react_node",
                output_key="issues",
                success_criteria="拿到 issue 列表",
                max_iterations=2,
            ),
            ResearchTask(
                task_id="task-2",
                title="读 README",
                goal="读取 README 资源",
                task_type="plan_react_node",
                depends_on=["task-1"],
                output_key="readme",
                success_criteria="拿到 README 内容",
                max_iterations=2,
            ),
        ],
    )


async def test_default_run_uses_plan_react_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_db()
    decisions = [
        PlanReactDecision("需要先查 issue", "mcp_call", "github", "tool", "list_issues", {"state": "open"}),
        PlanReactDecision("issue 已够", "final_answer", "github", "", "", {}, True, "issue 已读取"),
        PlanReactDecision("继续读 README", "mcp_call", "github", "resource", "repo://README.md", {}),
        PlanReactDecision("README 已够", "final_answer", "github", "", "", {}, True, "README 已读取"),
    ]

    monkeypatch.setattr("app.services.settings.execution_mode", "plan_react_mcp")
    monkeypatch.setattr("app.services.settings.llm_provider", "deepseek")
    monkeypatch.setattr("app.services.settings.llm_api_key", "test-key")
    monkeypatch.setattr("app.services.create_plan_react_mcp_client", lambda: FakeMCPClient())
    monkeypatch.setattr("app.services.llm_plan_react_plan", fake_plan)
    monkeypatch.setattr("app.services.llm_plan_react_next_step", lambda *args, **kwargs: decisions.pop(0))
    monkeypatch.setattr("app.services.llm_plan_react_final_answer", lambda *args, **kwargs: "已读取 GitHub issue 和 README。[C1] [C2]")

    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "MCP Project")
            session = await create_session(client, project["id"])
            response = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/run",
                json={"user_query": "总结 GitHub 项目状态", "sequence_id": 1, "asset_ids": []},
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["planner_mode"] == "plan_react_mcp"
    assert payload["retrieval"]["retrieval_mode"] == "mcp_plan_react"
    assert len(payload["retrieval"]["evidence_items"]) == 2
    actions = [step["action"] for step in payload["plan"]["execution_trace"]]
    assert "plan_node_start" in actions
    assert "react_action" in actions
    assert "mcp_observation" in actions
    assert "plan_node_complete" in actions
    assert payload["answer"]["citations"]


async def test_plan_react_blocks_side_effect_tool_without_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_db()
    decisions = [
        PlanReactDecision("尝试创建 issue", "mcp_call", "github", "tool", "create_issue", {"title": "x"}),
        PlanReactDecision("阻断后结束", "final_answer", "github", "", "", {}, True, "已阻断"),
    ]
    fake_client = FakeMCPClient()

    def single_task_plan(request, context, catalog) -> PlanTasksResponse:
        plan = fake_plan(request, context, catalog)
        return plan.model_copy(update={"tasks": plan.tasks[:1]})

    monkeypatch.setattr("app.services.settings.execution_mode", "plan_react_mcp")
    monkeypatch.setattr("app.services.settings.llm_provider", "deepseek")
    monkeypatch.setattr("app.services.settings.llm_api_key", "test-key")
    monkeypatch.setattr("app.services.settings.mcp_github_allowed_side_effect_tools", [])
    monkeypatch.setattr("app.services.create_plan_react_mcp_client", lambda: fake_client)
    monkeypatch.setattr("app.services.llm_plan_react_plan", single_task_plan)
    monkeypatch.setattr("app.services.llm_plan_react_next_step", lambda *args, **kwargs: decisions.pop(0))
    monkeypatch.setattr("app.services.llm_plan_react_final_answer", lambda *args, **kwargs: "未执行创建 issue。")

    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "MCP Block Project")
            session = await create_session(client, project["id"])
            response = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/run",
                json={"user_query": "创建一个 issue", "sequence_id": 1, "asset_ids": []},
            )

    assert response.status_code == 200
    payload = response.json()
    assert fake_client.call_count == 0
    blocked_steps = [step for step in payload["plan"]["execution_trace"] if step["action"] == "blocked_tool"]
    assert blocked_steps
    assert blocked_steps[0]["metadata"]["blocked"] is True


async def test_plan_react_requires_llm_key(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_db()
    monkeypatch.setattr("app.services.settings.execution_mode", "plan_react_mcp")
    monkeypatch.setattr("app.services.settings.llm_provider", "deepseek")
    monkeypatch.setattr("app.services.settings.llm_api_key", "")

    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "No Key Project")
            session = await create_session(client, project["id"])
            response = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/run",
                json={"user_query": "查 GitHub", "sequence_id": 1, "asset_ids": []},
            )

    assert response.status_code == 400
    assert "requires LLM_PROVIDER=deepseek" in response.json()["detail"]


async def test_plan_react_stream_emits_plan_trace_and_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_db()
    decisions = [
        PlanReactDecision("需要先查 issue", "mcp_call", "github", "tool", "list_issues", {"state": "open"}),
        PlanReactDecision("issue 已够", "final_answer", "github", "", "", {}, True, "issue 已读取"),
        PlanReactDecision("继续读 README", "mcp_call", "github", "resource", "repo://README.md", {}),
        PlanReactDecision("README 已够", "final_answer", "github", "", "", {}, True, "README 已读取"),
    ]

    monkeypatch.setattr("app.services.settings.execution_mode", "plan_react_mcp")
    monkeypatch.setattr("app.services.settings.llm_provider", "deepseek")
    monkeypatch.setattr("app.services.settings.llm_api_key", "test-key")
    monkeypatch.setattr("app.services.create_plan_react_mcp_client", lambda: FakeMCPClient())
    monkeypatch.setattr("app.services.llm_plan_react_plan", fake_plan)
    monkeypatch.setattr("app.services.llm_plan_react_next_step", lambda *args, **kwargs: decisions.pop(0))
    monkeypatch.setattr("app.services.llm_plan_react_final_answer", lambda *args, **kwargs: "已读取 GitHub issue 和 README。[C1] [C2]")

    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "MCP Stream Project")
            session = await create_session(client, project["id"])
            events: list[dict] = []
            async with client.stream(
                "POST",
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/run/stream",
                json={"user_query": "总结 GitHub 项目状态", "sequence_id": 1, "asset_ids": []},
            ) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.strip():
                        events.append(json.loads(line))

    event_types = [event["type"] for event in events]
    assert event_types[0] == "plan"
    assert "trace" in event_types
    assert "solver_summary" in event_types
    assert "answer_delta" in event_types
    assert "answer_quality" in event_types
    assert event_types[-1] == "complete"
    assert events[0]["plan"]["planner_mode"] == "plan_react_mcp"
    assert events[-1]["run"]["plan"]["planner_mode"] == "plan_react_mcp"
