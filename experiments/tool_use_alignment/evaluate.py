from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.tool_use_alignment.data import read_jsonl
from experiments.tool_use_alignment.metrics import evaluate_records, expected_from_record, format_markdown_report
from experiments.tool_use_alignment.prompts import TOOL_ACTIONS, decision_json
from experiments.tool_use_alignment.training_utils import generate_completion, load_model_for_inference, load_tokenizer


WRONG_ACTIONS = {
    "weather_lookup": ("final_answer", {"answer": "天气会变化，建议你自行确认。"}),
    "public_web_search": ("local_rag_search", {"query": "错误地只检索本地资料"}),
    "local_rag_search": ("public_web_search", {"query": "错误地搜索公网资料"}),
    "memory_read": ("local_rag_search", {"query": "错误地检索知识库"}),
    "memory_write": ("memory_read", {"query": "错误地读取记忆"}),
    "todo_create": ("final_answer", {"answer": "好的，我会记住这个待办。"}),
    "todo_list": ("todo_create", {"title": "错误地新建待办", "description": "", "priority": "medium"}),
    "asset_list": ("local_rag_search", {"query": "错误地检索资产内容"}),
    "calculator": ("final_answer", {"answer": "这个可以心算。"}),
    "final_answer": ("public_web_search", {"query": "错误地过度搜索"}),
}


def dry_run_prediction(record: dict[str, Any], *, stage: str, index: int) -> str:
    expected_action, expected_args = expected_from_record(record)
    if stage == "base" and index % 7 == 0:
        return "I think the answer is local_rag_search"
    if stage == "base" and index % 3 != 0:
        action, arguments = WRONG_ACTIONS[expected_action]
        return decision_json("base 模型尚未学会稳定工具路由。", action, arguments)
    if stage == "sft" and expected_action == "final_answer" and index % 2 == 0:
        action, arguments = WRONG_ACTIONS[expected_action]
        return decision_json("SFT 后格式稳定，但仍可能过度使用工具。", action, arguments)
    if stage == "sft" and index % 6 == 0:
        action, arguments = WRONG_ACTIONS[expected_action]
        return decision_json("SFT 后格式稳定，但仍会混淆部分工具。", action, arguments)
    if stage == "sft+dpo" and expected_action != "final_answer" and index % 17 == 0:
        action, arguments = WRONG_ACTIONS[expected_action]
        return decision_json("DPO 后仍保留少量错选工具案例。", action, arguments)
    thought = "根据用户意图选择正确工具。" if expected_action in TOOL_ACTIONS else "用户请求可直接回答，不需要工具。"
    return decision_json(thought, expected_action, expected_args)


def dry_run_predictions(records: list[dict[str, Any]], stage: str) -> list[str]:
    return [dry_run_prediction(record, stage=stage, index=index) for index, record in enumerate(records)]


def real_predictions(records: list[dict[str, Any]], *, model_name: str, adapter_dir: Path | None, max_new_tokens: int) -> list[str]:
    tokenizer = load_tokenizer(model_name)
    model = load_model_for_inference(model_name, adapter_dir)
    predictions = [generate_completion(model, tokenizer, str(record["prompt"]), max_new_tokens=max_new_tokens) for record in records]
    del model
    gc.collect()
    return predictions


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    records = read_jsonl(args.eval_file)
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise ValueError(f"No eval records found in {args.eval_file}")

    stage_predictions: dict[str, list[str]] = {}
    if args.mode == "dry-run":
        for stage in ("base", "sft", "sft+dpo"):
            stage_predictions[stage] = dry_run_predictions(records, stage)
    else:
        if not args.sft_adapter_dir.exists():
            raise FileNotFoundError(f"SFT adapter not found: {args.sft_adapter_dir}")
        if not args.dpo_adapter_dir.exists():
            raise FileNotFoundError(f"DPO adapter not found: {args.dpo_adapter_dir}")
        stage_predictions["base"] = real_predictions(
            records,
            model_name=args.model_name,
            adapter_dir=None,
            max_new_tokens=args.max_new_tokens,
        )
        stage_predictions["sft"] = real_predictions(
            records,
            model_name=args.model_name,
            adapter_dir=args.sft_adapter_dir,
            max_new_tokens=args.max_new_tokens,
        )
        stage_predictions["sft+dpo"] = real_predictions(
            records,
            model_name=args.model_name,
            adapter_dir=args.dpo_adapter_dir,
            max_new_tokens=args.max_new_tokens,
        )

    results = {stage: evaluate_records(records, predictions) for stage, predictions in stage_predictions.items()}
    report = format_markdown_report(results, mode=args.mode, source=str(args.eval_file))
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(report, encoding="utf-8")
    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"report_path": str(args.report_path), "mode": args.mode, "stages": list(results)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base/SFT/DPO tool-use decisions.")
    parser.add_argument("--eval-file", type=Path, default=Path("experiments/tool_use_alignment/data/eval_cases.jsonl"))
    parser.add_argument("--mode", choices=("dry-run", "real"), default="dry-run")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--sft-adapter-dir", type=Path, default=Path("experiments/tool_use_alignment/outputs/qwen3-0.6b-sft"))
    parser.add_argument("--dpo-adapter-dir", type=Path, default=Path("experiments/tool_use_alignment/outputs/qwen3-0.6b-sft-dpo"))
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("experiments/tool_use_alignment/reports/sft-dpo-tool-use-report.md"),
    )
    parser.add_argument("--metrics-json", type=Path, default=Path("experiments/tool_use_alignment/reports/metrics.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    summary = evaluate(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
