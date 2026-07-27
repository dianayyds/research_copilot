# Qwen3 0.6B Tool-Use SFT + DPO Demo

This experiment is a self-contained demo for training an agent model to choose Research Copilot tools.
It does not change the FastAPI app. It creates synthetic tool-decision data, runs LoRA SFT, continues with DPO preference tuning, and writes a metrics report.

## Setup

Install the backend plus optional training dependencies:

```bash
python3 -m pip install --user -e "./backend[training]"
```

Generate the synthetic SFT/DPO datasets:

```bash
python3 experiments/tool_use_alignment/generate_data.py
```

## Quick Sanity Check

Before downloading Qwen3 or running GPU training, verify the evaluation/reporting pipeline:

```bash
python3 experiments/tool_use_alignment/evaluate.py --mode dry-run
```

This writes `experiments/tool_use_alignment/reports/sft-dpo-tool-use-report.md` with simulated metrics. Use it only to validate the pipeline; do not present dry-run numbers as real model results.

## Real Training

Run SFT:

```bash
python3 experiments/tool_use_alignment/train_sft.py \
  --bf16 \
  --max-steps 80
```

Run DPO from the SFT adapter:

```bash
python3 experiments/tool_use_alignment/train_dpo.py \
  --bf16 \
  --max-steps 60
```

Evaluate base, SFT, and SFT+DPO:

```bash
python3 experiments/tool_use_alignment/evaluate.py --mode real --limit 32
```

Run five fixed demo prompts:

```bash
python3 experiments/tool_use_alignment/demo_infer.py
```

Generated datasets, adapters, and metrics JSON are ignored by Git.

## What This Demonstrates

- SFT teaches the model to produce the expected JSON tool-call schema.
- DPO teaches a preference between correct and plausible-but-wrong tool decisions.
- The most useful interview metrics are JSON valid rate, action accuracy, tool-needed F1, argument match rate, and over-tool rate.
- This is a lightweight RLHF-style preference alignment demo based on DPO, not a full industrial PPO RLHF pipeline.

## Data Shape

SFT records use `prompt` and `completion`:

```json
{"prompt": "...用户问题...", "completion": "{\"thought\":\"...\",\"action\":\"weather_lookup\",\"arguments\":{\"query\":\"...\"}}"}
```

DPO records use `prompt`, `chosen`, and `rejected`:

```json
{"prompt": "...用户问题...", "chosen": "{\"action\":\"weather_lookup\",...}", "rejected": "{\"action\":\"final_answer\",...}"}
```

Actions mirror the current Research Copilot tool names:

- `local_rag_search`
- `public_web_search`
- `weather_lookup`
- `memory_read`
- `memory_write`
- `todo_list`
- `todo_create`
- `asset_list`
- `calculator`
- `final_answer`
