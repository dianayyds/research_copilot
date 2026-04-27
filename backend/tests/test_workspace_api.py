from __future__ import annotations

from io import BytesIO
import json
import os

import httpx
import pytest

os.environ["DATABASE_URL"] = "sqlite:////tmp/research_copilot_test.db"
os.environ["VECTOR_STORE_PROVIDER"] = "stub"
os.environ["EMBEDDING_PROVIDER"] = "stub"
os.environ["LLM_PROVIDER"] = "stub"
os.environ["LLM_API_KEY"] = ""

from app.db import Base, engine  # noqa: E402
from app.live_tools import LiveToolEvidence, LiveToolResult, LiveToolRoute  # noqa: E402
from app.main import app  # noqa: E402
from app.services import AgentDecision  # noqa: E402
from app.semantic_store import get_semantic_memory_store, reset_semantic_memory_store  # noqa: E402
from app.vector_store import get_vector_store, reset_vector_store  # noqa: E402


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def reset_db() -> None:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    reset_vector_store()
    reset_semantic_memory_store()


async def create_project(client: httpx.AsyncClient, title: str) -> dict:
    response = await client.post(
        "/api/v1/projects",
        json={"title": title, "description": f"{title} description", "status": "active"},
    )
    return response.json()


async def create_session(client: httpx.AsyncClient, project_id: str, title: str = "新会话") -> dict:
    response = await client.post(f"/api/v1/projects/{project_id}/sessions", json={"title": title})
    return response.json()


def make_client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


def make_pdf_bytes(text: str) -> bytes:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )
    content = DecodedStreamObject()
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content.set_data(f"BT /F1 12 Tf 40 160 Td ({escaped}) Tj ET".encode("utf-8"))
    page[NameObject("/Contents")] = writer._add_object(content)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


async def test_root_page_serves_workspace() -> None:
    reset_db()
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            response = await client.get("/")
    assert response.status_code == 200
    assert "Research Copilot" in response.text
    assert "新对话" in response.text
    assert "资产" in response.text
    assert "LATS" in response.text


async def test_project_session_chat_flow_with_global_assets() -> None:
    reset_db()
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "Paper Studio")
            session = await create_session(client, project["id"])
            asset = (
                await client.post(
                    "/api/v1/assets",
                    json={
                        "title": "RAG Note",
                        "asset_type": "note",
                        "content": "当前检索设计采用 hybrid retrieval、rerank 和带引用输出。",
                    },
                )
            ).json()
            todo = (
                await client.post(
                    f"/api/v1/projects/{project['id']}/todos",
                    json={
                        "title": "整理 RAG 方案",
                        "description": "总结这个系统当前的检索设计",
                        "priority": "high",
                        "status": "todo",
                    },
                )
            ).json()

            first = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/run",
                json={
                    "user_query": "请总结系统当前的检索设计",
                    "todo_id": todo["id"],
                    "asset_ids": [asset["id"]],
                    "sequence_id": 1,
                },
            )
            second = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/run",
                json={
                    "user_query": "刚才你记录了什么检索设计？",
                    "asset_ids": [],
                    "sequence_id": 2,
                },
            )
            runs = await client.get(f"/api/v1/projects/{project['id']}/sessions/{session['id']}/runs")
            memory = await client.get(f"/api/v1/projects/{project['id']}/memory")
            sessions = await client.get(f"/api/v1/projects/{project['id']}/sessions")

    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["sequence_id"] == 1
    assert first_payload["plan"]["planner_mode"] == "two_stage"
    assert first_payload["plan"]["plan_summary"]
    assert first_payload["plan"]["search_queries"]
    assert first_payload["plan"]["tasks"][1]["depends_on"] == ["task-1"]
    assert first_payload["plan"]["tasks"][-1]["output_key"] == "final_answer"
    assert first_payload["plan"]["execution_trace"][0]["action"] == "scope"
    assert first_payload["plan"]["execution_trace"][-1]["action"] == "synthesize"
    assert first_payload["retrieval"]["evidence_items"][0]["asset_id"] == asset["id"]
    assert first_payload["retrieval"]["retrieval_mode"] == "stub_plan_hybrid_rerank"
    assert first_payload["answer"]["citations"]

    assert second.status_code == 200
    second_payload = second.json()
    assert "turn_1_answer" in second_payload["context"]["working_memory_context"]
    assert "Episodic Memory" in second_payload["context"]["memory_context"]
    assert "Semantic Memory" in second_payload["context"]["memory_context"]

    assert len(runs.json()) == 2
    assert sessions.json()[0]["last_sequence_id"] == 2
    assert any(item["memory_type"] == "working" for item in memory.json())
    assert any(item["memory_type"] == "episodic" for item in memory.json())
    assert any(item["memory_type"].startswith("semantic.") for item in memory.json())


async def test_second_session_sees_project_long_term_memory_but_not_working_memory() -> None:
    reset_db()
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "Memory Project")
            first_session = await create_session(client, project["id"], "Session A")
            await client.post(
                "/api/v1/assets",
                json={
                    "title": "Layered Memory",
                    "asset_type": "note",
                    "content": "项目采用 working、episodic、semantic 三层记忆。",
                },
            )
            first_run = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{first_session['id']}/run",
                json={
                    "user_query": "请记住本项目采用三层记忆",
                    "sequence_id": 1,
                    "asset_ids": [],
                },
            )
            second_session = await create_session(client, project["id"], "Session B")
            second_run = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{second_session['id']}/run",
                json={
                    "user_query": "这个项目记住了什么？",
                    "sequence_id": 1,
                    "asset_ids": [],
                },
            )

    assert first_run.status_code == 200
    assert second_run.status_code == 200
    payload = second_run.json()
    assert payload["context"]["working_memory_context"] == "- no working memory yet"
    assert payload["context"]["episodic_memory_context"] != "- no episodic memory yet"
    assert payload["context"]["semantic_memory_context"] != "- no semantic memory yet"


async def test_sequence_id_must_be_contiguous_within_session() -> None:
    reset_db()
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "Sequence Guard")
            session = await create_session(client, project["id"])
            invalid = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/run",
                json={
                    "user_query": "should fail",
                    "sequence_id": 2,
                    "asset_ids": [],
                },
            )

    assert invalid.status_code == 400
    assert invalid.json()["detail"] == "Sequence id must be 1 for this session"


async def test_cross_project_todo_is_rejected() -> None:
    reset_db()
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project_a = await create_project(client, "Project A")
            project_b = await create_project(client, "Project B")
            session_a = await create_session(client, project_a["id"])
            todo_b = (
                await client.post(
                    f"/api/v1/projects/{project_b['id']}/todos",
                    json={
                        "title": "B TODO",
                        "description": "owned by project B",
                        "priority": "medium",
                        "status": "todo",
                    },
                )
            ).json()
            response = await client.post(
                f"/api/v1/projects/{project_a['id']}/sessions/{session_a['id']}/run",
                json={
                    "user_query": "should fail",
                    "todo_id": todo_b["id"],
                    "sequence_id": 1,
                    "asset_ids": [],
                },
            )

    assert response.status_code == 404
    assert response.json()["detail"] == "Todo not found in project"


async def test_global_assets_are_visible_across_projects_but_memory_isolated() -> None:
    reset_db()
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            shared_asset = (
                await client.post(
                    "/api/v1/assets",
                    json={
                        "title": "Shared Asset",
                        "asset_type": "note",
                        "content": "全局知识库共享同一份文本资产。",
                    },
                )
            ).json()

            project_a = await create_project(client, "Alpha")
            session_a = await create_session(client, project_a["id"])
            await client.post(
                f"/api/v1/projects/{project_a['id']}/sessions/{session_a['id']}/run",
                json={
                    "user_query": "请记住 Alpha 项目关注 alpha topic",
                    "sequence_id": 1,
                    "asset_ids": [shared_asset["id"]],
                },
            )

            project_b = await create_project(client, "Beta")
            session_b = await create_session(client, project_b["id"])
            response = await client.post(
                f"/api/v1/projects/{project_b['id']}/sessions/{session_b['id']}/run",
                json={
                    "user_query": "这个项目能看到什么共享资产？",
                    "sequence_id": 1,
                    "asset_ids": [shared_asset["id"]],
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval"]["evidence_items"][0]["asset_id"] == shared_asset["id"]
    assert payload["context"]["working_memory_context"] == "- no working memory yet"
    assert "alpha topic" not in payload["context"]["episodic_memory_context"].lower()


async def test_delete_project_clears_project_data_but_keeps_global_assets() -> None:
    reset_db()
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "Cleanup")
            session = await create_session(client, project["id"])
            asset = (
                await client.post(
                    "/api/v1/assets",
                    json={
                        "title": "Shared Knowledge",
                        "asset_type": "note",
                        "content": "删除项目时不删除全局知识库。",
                    },
                )
            ).json()
            run = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/run",
                json={
                    "user_query": "请沉淀这个项目的记忆",
                    "sequence_id": 1,
                    "asset_ids": [asset["id"]],
                },
            )

            store = get_vector_store()
            semantic_store = get_semantic_memory_store()
            had_semantic_facts = bool(getattr(semantic_store, "facts"))
            response = await client.delete(f"/api/v1/projects/{project['id']}")
            projects = await client.get("/api/v1/projects")
            assets = await client.get("/api/v1/assets")

    assert run.status_code == 200
    assert getattr(store, "chunks")
    assert had_semantic_facts
    assert response.status_code == 200
    assert projects.json() == []
    assert assets.json()[0]["id"] == asset["id"]
    assert not any(fact["project_id"] == project["id"] for fact in getattr(semantic_store, "facts").values())


async def test_markdown_file_upload_creates_asset() -> None:
    reset_db()
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            response = await client.post(
                "/api/v1/assets/upload-file",
                data={"title": "检索说明", "asset_type": "markdown"},
                files={
                    "file": (
                        "retrieval.md",
                        b"# Hybrid Retrieval\n\nThis document explains rerank and citations.",
                        "text/markdown",
                    )
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["asset_type"] == "markdown"
    assert "Hybrid Retrieval" in payload["content"]
    assert "来源文件：retrieval.md" in payload["content"]


async def test_pdf_file_upload_extracts_page_text() -> None:
    reset_db()
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            response = await client.post(
                "/api/v1/assets/upload-file",
                data={"title": "PDF 说明", "asset_type": "pdf"},
                files={
                    "file": (
                        "intro.pdf",
                        make_pdf_bytes("Plan Solve PDF"),
                        "application/pdf",
                    )
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["asset_type"] == "pdf"
    assert "第1页" in payload["content"]
    assert "Plan Solve PDF" in payload["content"]


async def test_solver_replans_when_initial_evidence_is_weak() -> None:
    reset_db()
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "Replan Project")
            session = await create_session(client, project["id"])
            response = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/run",
                json={
                    "user_query": "《不存在的论文》这篇论文的第一作者是谁",
                    "sequence_id": 1,
                    "asset_ids": [],
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["replan_count"] == 1
    assert payload["plan"]["replan_reason"]
    assert any(step["action"] == "replan" for step in payload["plan"]["execution_trace"])


async def test_local_no_evidence_uses_direct_llm_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_db()

    def fake_direct_llm_answer(request, **kwargs) -> str:
        assert kwargs["reason"] == "local retrieval returned no evidence"
        assert "特朗普今年多大" in request.user_query
        return "截至 2026-04-27，特朗普 79 岁；他将在 2026-06-14 满 80 岁。"

    monkeypatch.setattr("app.services.direct_llm_answer", fake_direct_llm_answer)
    monkeypatch.setattr("app.services.settings.llm_provider", "deepseek")
    monkeypatch.setattr("app.services.settings.llm_api_key", "test-key")
    monkeypatch.setattr("app.services.settings.live_tools_enabled", False)
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "Fallback Project")
            session = await create_session(client, project["id"])
            response = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/run",
                json={
                    "user_query": "特朗普今年多大",
                    "sequence_id": 1,
                    "asset_ids": [],
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval"]["evidence_items"] == []
    assert "79 岁" in payload["answer"]["answer"]
    assert "当前知识库" not in payload["answer"]["answer"]
    assert "网络搜索" not in payload["answer"]["answer"]


async def test_restrictive_answer_with_weak_evidence_uses_direct_llm_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_db()

    def fake_llm_answer_markdown(request) -> str:
        assert request.evidence_items
        return "抱歉，我无法从当前的知识库和网络搜索中找到特朗普的年龄信息。"

    def fake_direct_llm_answer(request, **kwargs) -> str:
        assert kwargs["reason"] == "retrieved evidence did not support a usable answer"
        return "截至 2026-04-27，特朗普 79 岁。"

    monkeypatch.setattr("app.services.llm_answer_markdown", fake_llm_answer_markdown)
    monkeypatch.setattr("app.services.direct_llm_answer", fake_direct_llm_answer)
    monkeypatch.setattr("app.services.settings.llm_provider", "deepseek")
    monkeypatch.setattr("app.services.settings.llm_api_key", "test-key")
    monkeypatch.setattr("app.services.settings.live_tools_enabled", False)
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "Weak Evidence Fallback Project")
            session = await create_session(client, project["id"])
            await client.post(
                "/api/v1/assets",
                json={
                    "title": "特朗普简短资料",
                    "asset_type": "note",
                    "content": "特朗普曾担任美国总统，但这条笔记没有记录他的出生日期。",
                },
            )
            response = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/run",
                json={
                    "user_query": "特朗普今年多大",
                    "sequence_id": 1,
                    "asset_ids": [],
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval"]["evidence_items"]
    assert payload["answer"]["answer"] == "截至 2026-04-27，特朗普 79 岁。"
    assert payload["answer"]["citations"] == []


async def test_live_tool_failure_uses_direct_llm_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_db()

    def fake_execute_live_tool(route, query: str) -> LiveToolResult:
        assert route.tool_name == "public_web_search"
        raise ValueError("公开搜索没有返回可用摘要。")

    def fake_direct_llm_answer(request, **kwargs) -> str:
        assert kwargs["tool_payload"]["route"]["tool_name"] == "public_web_search"
        return "截至 2026-04-27，特朗普 79 岁；到 2026-06-14 会满 80 岁。"

    monkeypatch.setattr("app.services.execute_live_tool", fake_execute_live_tool)
    monkeypatch.setattr("app.services.direct_llm_answer", fake_direct_llm_answer)
    monkeypatch.setattr("app.services.settings.llm_provider", "deepseek")
    monkeypatch.setattr("app.services.settings.llm_api_key", "test-key")
    monkeypatch.setattr("app.services.settings.live_tools_enabled", True)
    monkeypatch.setattr("app.services.settings.public_web_search_enabled", True)
    monkeypatch.setattr("app.services.settings.llm_tool_planner_enabled", False)
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "Live Fallback Project")
            session = await create_session(client, project["id"])
            response = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/run",
                json={
                    "user_query": "特朗普几岁",
                    "sequence_id": 1,
                    "asset_ids": [],
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["planner_mode"] == "tool_router"
    assert payload["retrieval"]["evidence_items"] == []
    assert payload["plan"]["execution_trace"][2]["status"] == "failed"
    assert "79 岁" in payload["answer"]["answer"]
    assert "工具调用失败" not in payload["answer"]["answer"]
    assert payload["answer"]["citations"] == []


async def test_agent_restrictive_failure_answer_uses_direct_llm_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_db()

    def fake_llm_agent_next_step(request, context, history) -> AgentDecision:
        actions = [item for item in history if item.get("kind") == "action"]
        if not actions:
            return AgentDecision(
                thought="先查公开资料。",
                action="public_web_search",
                arguments={"query": "特朗普年龄"},
            )
        return AgentDecision(
            thought="搜索失败，结束。",
            action="final_answer",
            arguments={
                "answer": "抱歉，我无法从当前的知识库和网络搜索中找到特朗普的年龄信息。"
            },
        )

    def fake_execute_live_tool(route, query: str) -> LiveToolResult:
        raise ValueError("公开搜索没有返回可用摘要。")

    def fake_direct_llm_answer(request, **kwargs) -> str:
        assert kwargs["observations"]
        return "截至 2026-04-27，特朗普 79 岁。"

    monkeypatch.setattr("app.services.llm_agent_next_step", fake_llm_agent_next_step)
    monkeypatch.setattr("app.services.execute_live_tool", fake_execute_live_tool)
    monkeypatch.setattr("app.services.direct_llm_answer", fake_direct_llm_answer)
    monkeypatch.setattr("app.services.settings.llm_provider", "deepseek")
    monkeypatch.setattr("app.services.settings.llm_api_key", "test-key")
    monkeypatch.setattr("app.services.settings.agent_max_steps", 3)
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "Agent Fallback Project")
            session = await create_session(client, project["id"])
            response = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/agent/run",
                json={
                    "user_query": "特朗普今年多大",
                    "sequence_id": 1,
                    "asset_ids": [],
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval"]["evidence_items"] == []
    assert payload["answer"]["answer"] == "截至 2026-04-27，特朗普 79 岁。"
    assert "无法从当前" not in payload["answer"]["answer"]
    assert any(step["title"] == "Agent direct LLM fallback" for step in payload["plan"]["execution_trace"])


async def test_agent_restrictive_answer_with_evidence_uses_direct_llm_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_db()

    def fake_llm_agent_next_step(request, context, history) -> AgentDecision:
        actions = [item for item in history if item.get("kind") == "action"]
        if not actions:
            return AgentDecision(
                thought="先查本地资料。",
                action="local_rag_search",
                arguments={"query": "特朗普"},
            )
        return AgentDecision(
            thought="资料不够，结束。",
            action="final_answer",
            arguments={"answer": "抱歉，我无法从当前的知识库和网络搜索中找到特朗普的年龄信息。"},
        )

    def fake_direct_llm_answer(request, **kwargs) -> str:
        assert kwargs["observations"]
        return "截至 2026-04-27，特朗普 79 岁。"

    monkeypatch.setattr("app.services.llm_agent_next_step", fake_llm_agent_next_step)
    monkeypatch.setattr("app.services.direct_llm_answer", fake_direct_llm_answer)
    monkeypatch.setattr("app.services.settings.llm_provider", "deepseek")
    monkeypatch.setattr("app.services.settings.llm_api_key", "test-key")
    monkeypatch.setattr("app.services.settings.agent_max_steps", 3)
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "Agent Weak Evidence Fallback Project")
            session = await create_session(client, project["id"])
            await client.post(
                "/api/v1/assets",
                json={
                    "title": "特朗普简短资料",
                    "asset_type": "note",
                    "content": "特朗普曾担任美国总统，但这条笔记没有记录他的出生日期。",
                },
            )
            response = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/agent/run",
                json={
                    "user_query": "特朗普今年多大",
                    "sequence_id": 1,
                    "asset_ids": [],
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval"]["evidence_items"]
    assert payload["answer"]["answer"] == "截至 2026-04-27，特朗普 79 岁。"
    assert payload["answer"]["citations"] == []


async def test_weather_query_uses_live_tool_router(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_db()

    def fake_execute_live_tool(route, query: str) -> LiveToolResult:
        assert route.tool_name == "weather_lookup"
        assert "北京" in query
        return LiveToolResult(
            answer="北京当前天气：晴。\n气温 20°C，体感 19°C。\n数据来源：测试天气工具。",
            evidence=[
                LiveToolEvidence(
                    title="北京实时天气",
                    snippet="北京当前天气：晴；气温 20°C；体感 19°C",
                    source_path="https://open-meteo.com/",
                    tags=["live", "weather", "test"],
                )
            ],
            metadata={"provider": "test-weather"},
        )

    monkeypatch.setattr("app.services.execute_live_tool", fake_execute_live_tool)
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "Weather Project")
            session = await create_session(client, project["id"])
            response = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/run",
                json={
                    "user_query": "今天北京的天气如何？",
                    "sequence_id": 1,
                    "asset_ids": [],
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["planner_mode"] == "tool_router"
    assert "skill=weather_qa" in payload["plan"]["plan_summary"]
    assert "tool=weather_lookup" in payload["plan"]["plan_summary"]
    assert payload["retrieval"]["retrieval_mode"] == "live_tool"
    assert payload["retrieval"]["evidence_items"][0]["asset_id"] == "live-tool"
    assert payload["retrieval"]["evidence_items"][0]["source_path"] == "https://open-meteo.com/"
    assert "北京当前天气" in payload["answer"]["answer"]
    actions = [step["action"] for step in payload["plan"]["execution_trace"]]
    assert actions == ["route", "tool_call", "tool_result", "synthesize"]


async def test_live_tool_answer_uses_llm_for_complete_user_question(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_db()

    def fake_execute_live_tool(route, query: str) -> LiveToolResult:
        return LiveToolResult(
            answer=(
                "上海当前天气：晴。\n"
                "气温 13°C，体感 12°C，湿度 91%。\n"
                "今日气温约 12-24°C，最高降水概率 0%。\n"
                "当前降水量 0 mm，风速 7 km/h。\n"
                "数据来源：Open-Meteo forecast API。"
            ),
            evidence=[
                LiveToolEvidence(
                    title="上海实时天气",
                    snippet="上海当前天气：晴；气温 13°C；最高降水概率 0%；风速 7 km/h",
                    source_path="https://open-meteo.com/",
                    tags=["live", "weather", "test"],
                )
            ],
            metadata={"provider": "test-weather"},
        )

    def fake_llm_live_tool_answer(route, request, result, retrieval) -> str:
        assert route.skill == "weather_qa"
        assert "适合出游吗" in request.user_query
        assert "上海当前天气" in result.answer
        return "上海今天整体适合出游：天气晴、降水概率低、风速不大。建议带外套并注意早晚温差。[C1]"

    monkeypatch.setattr("app.services.execute_live_tool", fake_execute_live_tool)
    monkeypatch.setattr("app.services.llm_live_tool_answer", fake_llm_live_tool_answer)
    monkeypatch.setattr("app.services.settings.llm_provider", "deepseek")
    monkeypatch.setattr("app.services.settings.llm_api_key", "test-key")
    monkeypatch.setattr("app.services.settings.llm_tool_planner_enabled", False)
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "LLM Weather Project")
            session = await create_session(client, project["id"])
            response = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/run",
                json={
                    "user_query": "今天上海的天气怎么样？适合出游吗？",
                    "sequence_id": 1,
                    "asset_ids": [],
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["planner_mode"] == "tool_router"
    assert "整体适合出游" in payload["answer"]["answer"]
    assert payload["answer"]["citations"][0]["label"] == "C1"


async def test_llm_tool_planner_selects_weather_without_weather_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_db()

    def fake_llm_select_live_tool(request, context) -> LiveToolRoute:
        assert "适合出游吗" in request.user_query
        assert "Current Query" in context.packed_context
        return LiveToolRoute(
            intent="realtime_weather",
            skill="weather_qa",
            tool_name="weather_lookup",
            reason="用户询问今天上海是否适合出游，需要先查询天气事实。",
            confidence=0.88,
            planner_mode="llm_tool_planner",
        )

    def fake_execute_live_tool(route, query: str) -> LiveToolResult:
        assert route.planner_mode == "llm_tool_planner"
        assert route.tool_name == "weather_lookup"
        return LiveToolResult(
            answer="上海当前天气：晴。\n今日气温约 12-24°C，最高降水概率 0%。\n当前降水量 0 mm，风速 7 km/h。",
            evidence=[
                LiveToolEvidence(
                    title="上海实时天气",
                    snippet="上海当前天气：晴；最高降水概率 0%；风速 7 km/h",
                    source_path="https://open-meteo.com/",
                    tags=["live", "weather", "test"],
                )
            ],
            metadata={"provider": "test-weather"},
        )

    def fake_llm_live_tool_answer(route, request, result, retrieval) -> str:
        assert route.planner_mode == "llm_tool_planner"
        return "上海今天适合出游：天气晴、降水概率低、风速不大。[C1]"

    monkeypatch.setattr("app.services.llm_select_live_tool", fake_llm_select_live_tool)
    monkeypatch.setattr("app.services.execute_live_tool", fake_execute_live_tool)
    monkeypatch.setattr("app.services.llm_live_tool_answer", fake_llm_live_tool_answer)
    monkeypatch.setattr("app.services.settings.llm_provider", "deepseek")
    monkeypatch.setattr("app.services.settings.llm_api_key", "test-key")
    monkeypatch.setattr("app.services.settings.llm_tool_planner_enabled", True)
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "Planner Weather Project")
            session = await create_session(client, project["id"])
            response = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/run",
                json={
                    "user_query": "今天上海适合出游吗？",
                    "sequence_id": 1,
                    "asset_ids": [],
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["planner_mode"] == "llm_tool_planner"
    assert "LLM Tool Planner" in payload["plan"]["plan_summary"]
    assert "tool=weather_lookup" in payload["plan"]["plan_summary"]
    assert "适合出游" in payload["answer"]["answer"]


async def test_agent_loop_uses_local_rag_then_creates_todo(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_db()

    def fake_llm_agent_next_step(request, context, history) -> AgentDecision:
        actions = [item for item in history if item.get("kind") == "action"]
        if len(actions) == 0:
            return AgentDecision(
                thought="先检索本地资料确认项目事实。",
                action="local_rag_search",
                arguments={"query": "Agent 工具注册表"},
            )
        if len(actions) == 1:
            return AgentDecision(
                thought="已拿到资料，按用户要求创建 TODO。",
                action="todo_create",
                arguments={"title": "梳理 Agent 工具注册表", "description": "基于本地资料整理", "priority": "high"},
            )
        return AgentDecision(
            thought="本地资料和 TODO 创建都完成了。",
            action="final_answer",
            arguments={"answer": "已检索本地资料，并创建 TODO：梳理 Agent 工具注册表。[C1]"},
        )

    monkeypatch.setattr("app.services.llm_agent_next_step", fake_llm_agent_next_step)
    monkeypatch.setattr("app.services.settings.llm_provider", "deepseek")
    monkeypatch.setattr("app.services.settings.llm_api_key", "test-key")
    monkeypatch.setattr("app.services.settings.agent_max_steps", 4)
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "Agent Project")
            session = await create_session(client, project["id"])
            asset = (
                await client.post(
                    "/api/v1/assets",
                    json={
                        "title": "Agent Note",
                        "asset_type": "note",
                        "content": "Agent 工具注册表包含 local_rag_search、todo_create、memory_read 和 calculator。",
                    },
                )
            ).json()
            response = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/agent/run",
                json={
                    "user_query": "查一下 Agent 工具注册表，然后创建一个整理 TODO",
                    "sequence_id": 1,
                    "asset_ids": [asset["id"]],
                },
            )
            todos = await client.get(f"/api/v1/projects/{project['id']}/todos")

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["planner_mode"] == "agent_loop"
    assert payload["retrieval"]["retrieval_mode"] == "agent_tool_loop"
    assert payload["retrieval"]["evidence_items"][0]["asset_id"] == asset["id"]
    assert "创建 TODO" in payload["answer"]["answer"]
    actions = [step["action"] for step in payload["plan"]["execution_trace"]]
    assert actions.count("agent_action") == 3
    assert actions.count("agent_observation") == 2
    assert actions[-1] == "agent_final"
    assert any(todo["title"] == "梳理 Agent 工具注册表" and todo["priority"] == "high" for todo in todos.json())


async def test_agent_loop_uses_public_web_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_db()

    def fake_llm_agent_next_step(request, context, history) -> AgentDecision:
        actions = [item for item in history if item.get("kind") == "action"]
        if not actions:
            return AgentDecision(
                thought="用户要求公开资料，先搜索网络摘要。",
                action="public_web_search",
                arguments={"query": "DeepSeek function calling"},
            )
        return AgentDecision(
            thought="公开搜索已有结果，回答用户。",
            action="final_answer",
            arguments={"answer": "公开搜索显示 DeepSeek API 支持 function calling，用于让模型输出工具调用参数。[C1]"},
        )

    def fake_execute_live_tool(route, query: str) -> LiveToolResult:
        assert route.tool_name == "public_web_search"
        return LiveToolResult(
            answer="DeepSeek API 支持 function calling，可让模型按工具 schema 输出参数。",
            evidence=[
                LiveToolEvidence(
                    title="DeepSeek Function Calling",
                    snippet="Function calling lets the model return structured tool arguments.",
                    source_path="https://api-docs.deepseek.com/guides/function_calling/",
                    tags=["live", "web", "test"],
                )
            ],
            metadata={"provider": "test-web"},
        )

    monkeypatch.setattr("app.services.llm_agent_next_step", fake_llm_agent_next_step)
    monkeypatch.setattr("app.services.execute_live_tool", fake_execute_live_tool)
    monkeypatch.setattr("app.services.settings.llm_provider", "deepseek")
    monkeypatch.setattr("app.services.settings.llm_api_key", "test-key")
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "Web Agent Project")
            session = await create_session(client, project["id"])
            response = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/agent/run",
                json={
                    "user_query": "联网查一下 DeepSeek 是否支持 function calling",
                    "sequence_id": 1,
                    "asset_ids": [],
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["planner_mode"] == "agent_loop"
    assert payload["retrieval"]["evidence_items"][0]["source_path"].startswith("https://api-docs.deepseek.com")
    assert "function calling" in payload["answer"]["answer"]
    assert any("public_web_search" in step["title"] for step in payload["plan"]["execution_trace"])


async def test_agent_loop_uses_calculator_and_memory_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_db()

    def fake_llm_agent_next_step(request, context, history) -> AgentDecision:
        actions = [item for item in history if item.get("kind") == "action"]
        if len(actions) == 0:
            return AgentDecision(thought="先计算表达式。", action="calculator", arguments={"expression": "12 * (3 + 4)"})
        if len(actions) == 1:
            return AgentDecision(
                thought="用户要求记住结果，写入工作记忆。",
                action="memory_write",
                arguments={"key": "calculation_result", "content": "12 * (3 + 4) = 84", "importance": 0.9},
            )
        if len(actions) == 2:
            return AgentDecision(thought="读取记忆确认写入。", action="memory_read", arguments={"query": "calculation_result"})
        return AgentDecision(
            thought="计算和记忆确认完成。",
            action="final_answer",
            arguments={"answer": "计算结果是 84，并已写入工作记忆。"},
        )

    monkeypatch.setattr("app.services.llm_agent_next_step", fake_llm_agent_next_step)
    monkeypatch.setattr("app.services.settings.llm_provider", "deepseek")
    monkeypatch.setattr("app.services.settings.llm_api_key", "test-key")
    monkeypatch.setattr("app.services.settings.agent_max_steps", 5)
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "Memory Agent Project")
            session = await create_session(client, project["id"])
            response = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/agent/run",
                json={
                    "user_query": "计算 12 * (3 + 4)，然后记住结果",
                    "sequence_id": 1,
                    "asset_ids": [],
                },
            )
            memory = await client.get(f"/api/v1/projects/{project['id']}/memory")

    assert response.status_code == 200
    payload = response.json()
    titles = [step["title"] for step in payload["plan"]["execution_trace"]]
    assert any("calculator" in title for title in titles)
    assert any("memory_write" in title for title in titles)
    assert any("memory_read" in title for title in titles)
    assert "84" in payload["answer"]["answer"]
    assert any(item["memory_key"] == "calculation_result" for item in memory.json())


async def test_agent_stream_emits_agent_trace_events(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_db()

    def fake_llm_agent_next_step(request, context, history) -> AgentDecision:
        actions = [item for item in history if item.get("kind") == "action"]
        if not actions:
            return AgentDecision(thought="查看资产列表。", action="asset_list", arguments={})
        return AgentDecision(thought="资产列表已读取。", action="final_answer", arguments={"answer": "已读取资产列表。"})

    monkeypatch.setattr("app.services.llm_agent_next_step", fake_llm_agent_next_step)
    monkeypatch.setattr("app.services.settings.llm_provider", "deepseek")
    monkeypatch.setattr("app.services.settings.llm_api_key", "test-key")
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "Stream Agent Project")
            session = await create_session(client, project["id"])
            await client.post(
                "/api/v1/assets",
                json={"title": "Agent Asset", "asset_type": "note", "content": "agent stream asset"},
            )
            events: list[dict] = []
            async with client.stream(
                "POST",
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/agent/run/stream",
                json={
                    "user_query": "看看现在有哪些资产",
                    "sequence_id": 1,
                    "asset_ids": [],
                },
            ) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.strip():
                        events.append(json.loads(line))

    event_types = [event["type"] for event in events]
    assert "plan" in event_types
    assert "trace" in event_types
    assert "answer_delta" in event_types
    assert event_types[-1] == "complete"
    trace_actions = [event["step"]["action"] for event in events if event["type"] == "trace"]
    assert "agent_action" in trace_actions
    assert "agent_observation" in trace_actions
    assert "agent_final" in trace_actions
    assert events[-1]["run"]["plan"]["planner_mode"] == "agent_loop"


async def test_lats_agent_mcts_selects_calculator_path(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_db()
    monkeypatch.setattr("app.services.settings.lats_branching_factor", 3)
    monkeypatch.setattr("app.services.settings.lats_max_depth", 2)
    monkeypatch.setattr("app.services.settings.lats_iterations", 4)
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "LATS Project")
            session = await create_session(client, project["id"])
            response = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/lats/run",
                json={
                    "user_query": "请计算 12 * (3 + 4)",
                    "sequence_id": 1,
                    "asset_ids": [],
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["planner_mode"] == "lats_agent_mcts"
    assert payload["retrieval"]["retrieval_mode"] == "lats_agent_mcts"
    assert "84" in payload["answer"]["answer"]
    actions = [step["action"] for step in payload["plan"]["execution_trace"]]
    assert "lats_select" in actions
    assert "lats_expand" in actions
    assert "lats_action" in actions
    assert "lats_observation" in actions
    assert "lats_evaluate" in actions
    assert "lats_backprop" in actions
    assert "lats_final" in actions
    assert any("calculator" in step["summary"] for step in payload["plan"]["execution_trace"])


async def test_lats_agent_mcts_recovers_from_failed_public_search(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_db()
    monkeypatch.setattr("app.services.settings.lats_branching_factor", 3)
    monkeypatch.setattr("app.services.settings.lats_max_depth", 2)
    monkeypatch.setattr("app.services.settings.lats_iterations", 4)
    monkeypatch.setattr("app.services.settings.public_web_search_enabled", True)

    def fail_public_search(route: LiveToolRoute, query: str) -> LiveToolResult:
        if route.tool_name == "public_web_search":
            raise ValueError("public search unavailable")
        return LiveToolResult(answer="", evidence=[], metadata={})

    monkeypatch.setattr("app.services.execute_live_tool", fail_public_search)
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "LATS Recovery Project")
            session = await create_session(client, project["id"])
            asset = (
                await client.post(
                    "/api/v1/assets",
                    json={
                        "title": "DeepSeek Function Calling",
                        "asset_type": "note",
                        "content": (
                            "DeepSeek function calling lets a chat model choose structured tools, "
                            "return JSON arguments, and then answer from the tool observations."
                        ),
                    },
                )
            ).json()
            response = await client.post(
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/lats/run",
                json={
                    "user_query": "联网搜索一下 DeepSeek function calling 是什么",
                    "sequence_id": 1,
                    "asset_ids": [],
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["planner_mode"] == "lats_agent_mcts"
    assert payload["retrieval"]["evidence_items"][0]["asset_id"] == asset["id"]
    assert "function calling" in payload["answer"]["answer"].lower()
    trace = payload["plan"]["execution_trace"]
    assert any(step["action"] == "lats_observation" and step["status"] == "failed" for step in trace)
    assert any(step["action"] == "lats_action" and "local_rag_search" in step["summary"] for step in trace)
    assert any(step["action"] == "lats_final" for step in trace)


async def test_lats_stream_emits_mcts_trace_events(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_db()
    monkeypatch.setattr("app.services.settings.lats_branching_factor", 3)
    monkeypatch.setattr("app.services.settings.lats_max_depth", 2)
    monkeypatch.setattr("app.services.settings.lats_iterations", 3)
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "LATS Stream Project")
            session = await create_session(client, project["id"])
            events: list[dict] = []
            async with client.stream(
                "POST",
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/lats/run/stream",
                json={
                    "user_query": "请计算 8 * 7",
                    "sequence_id": 1,
                    "asset_ids": [],
                },
            ) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.strip():
                        events.append(json.loads(line))

    event_types = [event["type"] for event in events]
    assert "plan" in event_types
    assert "trace" in event_types
    assert "answer_delta" in event_types
    assert event_types[-1] == "complete"
    trace_actions = [event["step"]["action"] for event in events if event["type"] == "trace"]
    assert "lats_expand" in trace_actions
    assert "lats_select" in trace_actions
    assert "lats_evaluate" in trace_actions
    assert "lats_backprop" in trace_actions
    assert events[-1]["run"]["plan"]["planner_mode"] == "lats_agent_mcts"
    assert "56" in events[-1]["run"]["answer"]["answer"]


async def test_stream_no_evidence_uses_direct_fallback_without_constrained_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_db()

    def fail_llm_answer_stream_chunks(request):
        raise AssertionError("constrained stream should not run without evidence")

    def fake_direct_llm_answer(request, **kwargs) -> str:
        assert kwargs["reason"] == "local retrieval returned no evidence"
        return "截至 2026-04-27，特朗普 79 岁。"

    monkeypatch.setattr("app.services.llm_answer_stream_chunks", fail_llm_answer_stream_chunks)
    monkeypatch.setattr("app.services.direct_llm_answer", fake_direct_llm_answer)
    monkeypatch.setattr("app.services.settings.llm_provider", "deepseek")
    monkeypatch.setattr("app.services.settings.llm_api_key", "test-key")
    monkeypatch.setattr("app.services.settings.live_tools_enabled", False)
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "Stream Fallback Project")
            session = await create_session(client, project["id"])
            events: list[dict] = []
            async with client.stream(
                "POST",
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/run/stream",
                json={
                    "user_query": "特朗普今年多大",
                    "sequence_id": 1,
                    "asset_ids": [],
                },
            ) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.strip():
                        events.append(json.loads(line))

    answer_events = [event for event in events if event["type"] == "answer_delta"]
    assert answer_events[-1]["answer"] == "截至 2026-04-27，特朗普 79 岁。"
    assert events[-1]["run"]["answer"]["answer"] == "截至 2026-04-27，特朗普 79 岁。"


async def test_stream_run_emits_plan_trace_and_complete_events() -> None:
    reset_db()
    async with app.router.lifespan_context(app):
        async with make_client() as client:
            project = await create_project(client, "Stream Project")
            session = await create_session(client, project["id"])
            await client.post(
                "/api/v1/assets",
                json={
                    "title": "Author Note",
                    "asset_type": "note",
                    "content": "Adaptive Semantic Speech Transmission for High-Speed Scenarios 的第一作者是 Fangyu Liu。",
                },
            )
            events: list[dict] = []
            async with client.stream(
                "POST",
                f"/api/v1/projects/{project['id']}/sessions/{session['id']}/run/stream",
                json={
                    "user_query": "《Adaptive Semantic Speech Transmission for High-Speed Scenarios》这篇论文的第一作者是谁",
                    "sequence_id": 1,
                    "asset_ids": [],
                },
            ) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.strip():
                        events.append(json.loads(line))

    event_types = [event["type"] for event in events]
    assert "plan" in event_types
    assert "trace" in event_types
    assert "answer_delta" in event_types
    assert event_types[-1] == "complete"
    complete_event = events[-1]
    assert complete_event["run"]["plan"]["planner_mode"] == "two_stage"
