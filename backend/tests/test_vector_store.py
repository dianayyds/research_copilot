import os

os.environ["DATABASE_URL"] = "sqlite:////tmp/research_copilot_test.db"
os.environ["VECTOR_STORE_PROVIDER"] = "stub"
os.environ["EMBEDDING_PROVIDER"] = "stub"
os.environ["LLM_PROVIDER"] = "stub"
os.environ["LLM_API_KEY"] = ""

from app.vector_store import effective_model_max_length, searchable_chunks


def test_effective_model_max_length_caps_to_model_limit() -> None:
    assert effective_model_max_length(1024, 512, 514) == 512
    assert effective_model_max_length(512, 2048, 514) == 512
    assert effective_model_max_length(256, None, None) == 256


def test_searchable_chunks_filters_parent_chunks_but_keeps_legacy_payloads() -> None:
    chunks = [
        {"chunk_id": "legacy-1", "content": "legacy chunk without level"},
        {"chunk_id": "parent-1", "chunk_level": "parent", "content": "wide parent context"},
        {"chunk_id": "child-1", "chunk_level": "child", "content": "searchable child context"},
    ]

    assert [chunk["chunk_id"] for chunk in searchable_chunks(chunks)] == ["legacy-1", "child-1"]
