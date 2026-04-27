import os

os.environ["DATABASE_URL"] = "sqlite:////tmp/research_copilot_test.db"
os.environ["VECTOR_STORE_PROVIDER"] = "stub"
os.environ["EMBEDDING_PROVIDER"] = "stub"
os.environ["LLM_PROVIDER"] = "stub"
os.environ["LLM_API_KEY"] = ""

from app.vector_store import effective_model_max_length


def test_effective_model_max_length_caps_to_model_limit() -> None:
    assert effective_model_max_length(1024, 512, 514) == 512
    assert effective_model_max_length(512, 2048, 514) == 512
    assert effective_model_max_length(256, None, None) == 256
