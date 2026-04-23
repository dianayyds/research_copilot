from __future__ import annotations

import os

from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:////tmp/research_copilot_test.db"
os.environ["VECTOR_STORE_PROVIDER"] = "stub"
os.environ["EMBEDDING_PROVIDER"] = "stub"

from app.db import Base, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.semantic_store import get_semantic_memory_store, reset_semantic_memory_store  # noqa: E402
from app.vector_store import get_vector_store, reset_vector_store  # noqa: E402


def reset_db() -> None:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    reset_vector_store()
    reset_semantic_memory_store()


def test_root_page_serves_workspace() -> None:
    reset_db()
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Research Copilot" in response.text


def test_plan_and_solve_workspace_flow() -> None:
    reset_db()
    client = TestClient(app)

    project = client.post(
        "/api/v1/projects",
        json={"title": "DeepSeek MVP", "description": "Plan-and-solve test project", "status": "active"},
    ).json()
    project_id = project["id"]

    asset = client.post(
        f"/api/v1/projects/{project_id}/assets",
        json={
            "title": "论文摘要",
            "asset_type": "paper",
            "content": "这篇论文讨论了检索增强生成、长任务分解和带引用答案的实现方法。",
        },
    ).json()
    client.post(
        f"/api/v1/projects/{project_id}/assets",
        json={
            "title": "代码仓说明",
            "asset_type": "code",
            "content": "代码仓包含任务编排、运行记录、项目记忆和 TODO 管理模块。",
        },
    )
    updated_asset = client.patch(
        f"/api/v1/assets/{asset['id']}",
        json={
            "content": "这篇论文讨论了检索增强生成、长任务分解、混合检索和带引用答案的实现方法。",
        },
    )
    assert updated_asset.status_code == 200
    assert "混合检索" in updated_asset.json()["content"]
    todo = client.post(
        f"/api/v1/projects/{project_id}/todos",
        json={
            "title": "梳理论文和代码关系",
            "description": "请总结论文与代码仓之间的映射关系",
            "priority": "high",
            "status": "todo",
        },
    ).json()

    run = client.post(
        f"/api/v1/projects/{project_id}/run",
        json={
            "user_query": "请总结论文与代码仓之间的映射关系",
            "todo_id": todo["id"],
            "asset_ids": [],
        },
    )
    assert run.status_code == 200
    payload = run.json()
    assert payload["answer"]["citations"]
    assert payload["memory"]["memory_updates"]
    assert payload["retrieval"]["evidence_items"]

    runs = client.get(f"/api/v1/projects/{project_id}/runs").json()
    memory = client.get(f"/api/v1/projects/{project_id}/memory").json()
    todos = client.get(f"/api/v1/projects/{project_id}/todos").json()

    assert len(runs) == 1
    assert len(memory) >= 3
    assert {item["memory_type"] for item in memory} >= {"working", "episodic"}
    assert any(item["memory_type"].startswith("semantic.") for item in memory)
    assert todos[0]["status"] == "done"


def test_cross_project_todo_is_rejected() -> None:
    reset_db()
    client = TestClient(app)

    project_a = client.post(
        "/api/v1/projects",
        json={"title": "Project A", "description": "A", "status": "active"},
    ).json()
    project_b = client.post(
        "/api/v1/projects",
        json={"title": "Project B", "description": "B", "status": "active"},
    ).json()
    todo = client.post(
        f"/api/v1/projects/{project_b['id']}/todos",
        json={
            "title": "B TODO",
            "description": "owned by project B",
            "priority": "medium",
            "status": "todo",
        },
    ).json()

    response = client.post(
        f"/api/v1/projects/{project_a['id']}/run",
        json={
            "user_query": "should fail",
            "todo_id": todo["id"],
            "asset_ids": [],
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Todo not found in project"


def test_project_activity_refreshes_project_order() -> None:
    reset_db()
    client = TestClient(app)

    first = client.post(
        "/api/v1/projects",
        json={"title": "First", "description": "old", "status": "active"},
    ).json()
    second = client.post(
        "/api/v1/projects",
        json={"title": "Second", "description": "newer", "status": "active"},
    ).json()

    before = client.get("/api/v1/projects").json()
    assert before[0]["id"] == second["id"]

    client.post(
        f"/api/v1/projects/{first['id']}/assets",
        json={
            "title": "Notebook",
            "asset_type": "note",
            "content": "touch project activity",
        },
    )

    after = client.get("/api/v1/projects").json()
    assert after[0]["id"] == first["id"]


def test_text_asset_upload_indexes_content() -> None:
    reset_db()
    client = TestClient(app)

    project = client.post(
        "/api/v1/projects",
        json={"title": "Upload Demo", "description": "upload", "status": "active"},
    ).json()

    upload = client.post(
        f"/api/v1/projects/{project['id']}/assets/upload-text",
        files={"file": ("notes.md", "BGE 与 Qdrant 将用于知识库向量检索。".encode("utf-8"), "text/markdown")},
        data={"asset_type": "note"},
    )
    assert upload.status_code == 200
    asset = upload.json()
    assert asset["title"] == "notes.md"

    run = client.post(
        f"/api/v1/projects/{project['id']}/run",
        json={
            "user_query": "知识库会使用什么做向量检索",
            "asset_ids": [asset["id"]],
        },
    )
    assert run.status_code == 200
    payload = run.json()
    assert payload["retrieval"]["evidence_items"]
    assert payload["retrieval"]["retrieval_mode"] == "stub_hybrid_rerank"
    assert payload["retrieval"]["evidence_items"][0]["asset_id"] == asset["id"]
    assert "hybrid" in payload["retrieval"]["evidence_items"][0]["tags"]
    assert "reranked" in payload["retrieval"]["evidence_items"][0]["tags"]


def test_followup_run_uses_layered_memory_context() -> None:
    reset_db()
    client = TestClient(app)

    project = client.post(
        "/api/v1/projects",
        json={"title": "Memory Demo", "description": "memory", "status": "active"},
    ).json()
    client.post(
        f"/api/v1/projects/{project['id']}/assets",
        json={
            "title": "设计说明",
            "asset_type": "note",
            "content": "系统采用 plan-and-solve、hybrid rag 和 layered memory。",
        },
    )

    first = client.post(
        f"/api/v1/projects/{project['id']}/run",
        json={
            "user_query": "请记录系统采用 layered memory",
            "session_id": "session-memory-demo",
            "asset_ids": [],
        },
    )
    assert first.status_code == 200

    second = client.post(
        f"/api/v1/projects/{project['id']}/run",
        json={
            "user_query": "刚才这个项目采用了什么记忆设计？",
            "session_id": "session-memory-demo",
            "asset_ids": [],
        },
    )
    assert second.status_code == 200
    payload = second.json()
    assert "Working Memory" in payload["context"]["memory_context"]
    assert "Episodic Memory" in payload["context"]["memory_context"]
    assert "Semantic Memory" in payload["context"]["memory_context"]
    assert "latest_answer_summary" in payload["context"]["working_memory_context"]
    assert "semantic." not in payload["context"]["semantic_memory_context"]
    assert "fact." in payload["context"]["semantic_memory_context"] or "decision." in payload["context"]["semantic_memory_context"]


def test_delete_project_clears_project_data() -> None:
    reset_db()
    client = TestClient(app)

    project = client.post(
        "/api/v1/projects",
        json={"title": "Cleanup Demo", "description": "cleanup", "status": "active"},
    ).json()
    client.post(
        f"/api/v1/projects/{project['id']}/assets",
        json={
            "title": "Cleanup Note",
            "asset_type": "note",
            "content": "项目删除时需要同步清理向量索引。",
        },
    )
    run = client.post(
        f"/api/v1/projects/{project['id']}/run",
        json={
            "user_query": "请沉淀这个项目的记忆",
            "session_id": "session-cleanup-demo",
        },
    )
    assert run.status_code == 200

    store = get_vector_store()
    semantic_store = get_semantic_memory_store()
    assert getattr(store, "chunks")
    assert getattr(semantic_store, "facts")

    response = client.delete(f"/api/v1/projects/{project['id']}")

    assert response.status_code == 200
    assert client.get("/api/v1/projects").json() == []
    assert not any(chunk["project_id"] == project["id"] for chunk in getattr(store, "chunks").values())
    assert not any(fact["project_id"] == project["id"] for fact in getattr(semantic_store, "facts").values())
