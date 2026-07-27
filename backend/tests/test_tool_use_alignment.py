from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.tool_use_alignment.data import build_cases, generate_datasets, read_jsonl
from experiments.tool_use_alignment.metrics import evaluate_records, normalize_action, parse_decision, prediction_from_text
from experiments.tool_use_alignment.prompts import decision_json


def test_generate_datasets_writes_sft_and_dpo_jsonl(tmp_path: Path) -> None:
    summary = generate_datasets(tmp_path, seed=7, train_ratio=0.75)

    assert summary["train_cases"] > summary["eval_cases"] > 0
    sft_train = read_jsonl(tmp_path / "sft_train.jsonl")
    dpo_train = read_jsonl(tmp_path / "dpo_train.jsonl")

    assert sft_train[0]["prompt"]
    assert sft_train[0]["completion"].startswith("{")
    assert dpo_train[0]["chosen"] != dpo_train[0]["rejected"]
    assert {"prompt", "chosen", "rejected", "expected"} <= set(dpo_train[0])


def test_dataset_covers_all_tool_actions() -> None:
    actions = {case.action for case in build_cases()}
    assert {
        "local_rag_search",
        "public_web_search",
        "weather_lookup",
        "memory_read",
        "memory_write",
        "todo_list",
        "todo_create",
        "asset_list",
        "calculator",
        "final_answer",
    } <= actions


def test_parse_decision_handles_markdown_and_aliases() -> None:
    payload = parse_decision('```json\n{"thought":"x","action":"web_search","arguments":{"query":"q"}}\n```')

    assert payload is not None
    assert normalize_action(payload["action"]) == "public_web_search"


def test_prediction_from_text_rejects_invalid_json() -> None:
    action, arguments, valid = prediction_from_text("not json")

    assert action == "invalid_json"
    assert arguments == {}
    assert valid is False


def test_evaluate_records_computes_metrics() -> None:
    records = [
        {"id": "1", "category": "weather", "expected": {"action": "weather_lookup", "arguments": {"query": "北京天气"}}},
        {"id": "2", "category": "final", "expected": {"action": "final_answer", "arguments": {"answer": "你好"}}},
    ]
    predictions = [
        decision_json("需要天气工具", "weather_lookup", {"query": "北京天气"}),
        decision_json("直接回答", "public_web_search", {"query": "你好"}),
    ]

    metrics = evaluate_records(records, predictions)

    assert metrics["valid_json_rate"] == 1.0
    assert metrics["action_accuracy"] == 0.5
    assert metrics["tool_needed_f1"] > 0
    assert metrics["over_tool_rate"] == 1.0
