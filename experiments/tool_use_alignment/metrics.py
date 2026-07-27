from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from experiments.tool_use_alignment.prompts import ACTION_ALIASES, TOOL_ACTIONS, VALID_ACTIONS


INVALID_ACTION = "invalid_json"


def normalize_action(action: object) -> str:
    normalized = str(action or "").strip()
    if not normalized:
        return ""
    lower = normalized.lower()
    return ACTION_ALIASES.get(lower, normalized)


def parse_decision(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    decoder = json.JSONDecoder()
    candidates = [stripped]
    candidates.extend(stripped[index:] for index, char in enumerate(stripped) if char == "{")
    for candidate in candidates:
        try:
            payload, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _value_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) < 1e-6
        except (TypeError, ValueError):
            return False
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(actual) == expected
        except (TypeError, ValueError):
            return False
    return " ".join(str(expected).split()) == " ".join(str(actual).split())


def arguments_match(expected_action: str, expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    if expected_action == "final_answer":
        return bool(str(actual.get("answer") or "").strip())
    for key, expected_value in expected.items():
        if key not in actual or not _value_match(expected_value, actual[key]):
            return False
    return True


def expected_from_record(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    expected = record.get("expected") or {}
    if not isinstance(expected, dict):
        return "", {}
    action = normalize_action(expected.get("action"))
    arguments = expected.get("arguments") or {}
    return action, dict(arguments) if isinstance(arguments, dict) else {}


def prediction_from_text(text: str) -> tuple[str, dict[str, Any], bool]:
    payload = parse_decision(text)
    if payload is None:
        return INVALID_ACTION, {}, False
    action = normalize_action(payload.get("action"))
    arguments = payload.get("arguments") or {}
    if action not in VALID_ACTIONS:
        return INVALID_ACTION, {}, False
    return action, dict(arguments) if isinstance(arguments, dict) else {}, True


def evaluate_records(records: list[dict[str, Any]], predictions: list[str]) -> dict[str, Any]:
    if len(records) != len(predictions):
        raise ValueError("records and predictions must have the same length")

    valid_count = 0
    action_hits = 0
    argument_hits = 0
    expected_tool_count = 0
    expected_final_count = 0
    predicted_tool_count = 0
    true_tool_positive = 0
    over_tool_count = 0
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_case: list[dict[str, Any]] = []

    for record, prediction in zip(records, predictions, strict=True):
        expected_action, expected_args = expected_from_record(record)
        predicted_action, predicted_args, valid_json = prediction_from_text(prediction)
        valid_count += int(valid_json)
        action_match = predicted_action == expected_action
        action_hits += int(action_match)
        args_match = action_match and arguments_match(expected_action, expected_args, predicted_args)
        argument_hits += int(args_match)

        expected_tool = expected_action in TOOL_ACTIONS
        predicted_tool = predicted_action in TOOL_ACTIONS
        expected_tool_count += int(expected_tool)
        expected_final_count += int(not expected_tool)
        predicted_tool_count += int(predicted_tool)
        true_tool_positive += int(expected_tool and predicted_tool)
        over_tool_count += int((not expected_tool) and predicted_tool)
        confusion[expected_action][predicted_action] += 1
        per_case.append(
            {
                "id": record.get("id", ""),
                "category": record.get("category", ""),
                "expected_action": expected_action,
                "predicted_action": predicted_action,
                "valid_json": valid_json,
                "action_match": action_match,
                "arguments_match": bool(args_match),
            }
        )

    total = len(records) or 1
    precision = true_tool_positive / predicted_tool_count if predicted_tool_count else 0.0
    recall = true_tool_positive / expected_tool_count if expected_tool_count else 0.0
    tool_f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    over_tool_rate = over_tool_count / expected_final_count if expected_final_count else 0.0
    return {
        "total": len(records),
        "valid_json_rate": valid_count / total,
        "action_accuracy": action_hits / total,
        "argument_match_rate": argument_hits / total,
        "tool_precision": precision,
        "tool_recall": recall,
        "tool_needed_f1": tool_f1,
        "over_tool_rate": over_tool_rate,
        "confusion_matrix": {key: dict(value) for key, value in confusion.items()},
        "per_case": per_case,
    }


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_markdown_report(results: dict[str, dict[str, Any]], *, mode: str, source: str) -> str:
    lines = [
        "# SFT + DPO Tool-Use Alignment Report",
        "",
        f"- Evaluation mode: `{mode}`",
        f"- Evaluation source: `{source}`",
        "- Task: choose exactly one Research Copilot tool action, or `final_answer` when no tool is needed.",
    ]
    if mode == "dry-run":
        lines.append("- Note: dry-run metrics are simulated pipeline checks; use `--mode real` for model results.")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Stage | Cases | JSON valid | Action accuracy | Tool-needed F1 | Argument match | Over-tool rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for stage, metrics in results.items():
        lines.append(
            "| "
            f"{stage} | {metrics['total']} | {format_percent(metrics['valid_json_rate'])} | "
            f"{format_percent(metrics['action_accuracy'])} | {format_percent(metrics['tool_needed_f1'])} | "
            f"{format_percent(metrics['argument_match_rate'])} | {format_percent(metrics['over_tool_rate'])} |"
        )

    lines.extend(["", "## Confusion Matrices", ""])
    for stage, metrics in results.items():
        lines.append(f"### {stage}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(metrics["confusion_matrix"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    lines.extend(
        [
            "## Interview Notes",
            "",
            "- SFT teaches the model to emit the tool-call JSON schema from demonstrations.",
            "- DPO uses chosen/rejected decisions to reduce over-tooling and wrong-tool selection.",
            "- This is a lightweight RLHF-style preference alignment demo, not a full PPO RLHF reproduction.",
        ]
    )
    return "\n".join(lines) + "\n"
