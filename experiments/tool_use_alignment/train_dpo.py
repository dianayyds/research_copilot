from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.tool_use_alignment.training_utils import (
    config_instance,
    dataset_from_jsonl,
    load_base_model,
    load_tokenizer,
    trainer_tokenizer_kwargs,
)


def train(args: argparse.Namespace) -> dict[str, object]:
    try:
        from peft import PeftModel
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError("Missing dependency: trl/peft. Install with `pip install -e './backend[training]'`.") from exc

    tokenizer = load_tokenizer(args.model_name)
    base_model = load_base_model(args.model_name)
    model = PeftModel.from_pretrained(base_model, str(args.sft_adapter_dir), is_trainable=True)
    train_dataset = dataset_from_jsonl(args.train_file)
    eval_dataset = dataset_from_jsonl(args.eval_file) if args.eval_file and args.eval_file.exists() else None
    training_args = config_instance(
        DPOConfig,
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        report_to=[],
        bf16=args.bf16,
        fp16=args.fp16,
        beta=args.beta,
        loss_type="sigmoid",
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **trainer_tokenizer_kwargs(DPOTrainer, tokenizer),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    return {
        "output_dir": str(args.output_dir),
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset) if eval_dataset is not None else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DPO preference tuning from the SFT LoRA adapter.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--sft-adapter-dir", type=Path, default=Path("experiments/tool_use_alignment/outputs/qwen3-0.6b-sft"))
    parser.add_argument("--train-file", type=Path, default=Path("experiments/tool_use_alignment/data/dpo_train.jsonl"))
    parser.add_argument("--eval-file", type=Path, default=Path("experiments/tool_use_alignment/data/dpo_eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/tool_use_alignment/outputs/qwen3-0.6b-sft-dpo"))
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-prompt-length", type=int, default=1280)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=30)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = train(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
