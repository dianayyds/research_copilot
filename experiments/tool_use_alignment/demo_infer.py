from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.tool_use_alignment.metrics import prediction_from_text
from experiments.tool_use_alignment.prompts import build_prompt
from experiments.tool_use_alignment.training_utils import generate_completion, load_model_for_inference, load_tokenizer


SAMPLE_QUERIES = [
    "明天北京会下雨吗？如果我要去清华附近调研，要不要带伞？",
    "联网查一下 Qwen3-0.6B 的模型卡，确认它的参数量和上下文长度。",
    "在本项目资料里检索 DeepSC 为什么低信噪比下还能保住句子意思。",
    "帮我计算 (87 + 93 + 91) / 3。",
    "把“我做过 SFT”这句话改得更专业一点。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run five fixed tool-use inference examples.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter-dir", type=Path, default=Path("experiments/tool_use_alignment/outputs/qwen3-0.6b-sft-dpo"))
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_dir = None if args.base_only else args.adapter_dir
    if adapter_dir and not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_dir}. Run train_sft.py and train_dpo.py first, or pass --base-only.")
    tokenizer = load_tokenizer(args.model_name)
    model = load_model_for_inference(args.model_name, adapter_dir)
    for index, query in enumerate(SAMPLE_QUERIES, start=1):
        output = generate_completion(model, tokenizer, build_prompt(query), max_new_tokens=args.max_new_tokens)
        action, arguments, valid_json = prediction_from_text(output)
        print(f"\n## Example {index}")
        print(f"query: {query}")
        print(f"valid_json: {valid_json}")
        print(f"action: {action}")
        print(f"arguments: {json.dumps(arguments, ensure_ascii=False)}")
        print(f"raw: {output}")


if __name__ == "__main__":
    main()
