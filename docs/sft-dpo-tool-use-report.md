# SFT + DPO Tool-Use Alignment Report

- Evaluation mode: `real`
- Evaluation source: `experiments/tool_use_alignment/data/eval_cases.jsonl`
- Task: choose exactly one Research Copilot tool action, or `final_answer` when no tool is needed.

## Metrics

| Stage | Cases | JSON valid | Action accuracy | Tool-needed F1 | Argument match | Over-tool rate |
|---|---:|---:|---:|---:|---:|---:|
| base | 15 | 46.7% | 26.7% | 47.1% | 13.3% | 0.0% |
| sft | 15 | 100.0% | 86.7% | 100.0% | 66.7% | 0.0% |
| sft+dpo | 15 | 100.0% | 86.7% | 100.0% | 66.7% | 0.0% |

## Confusion Matrices

### base

```json
{
  "weather_lookup": {
    "invalid_json": 1,
    "public_web_search": 1
  },
  "memory_read": {
    "invalid_json": 1
  },
  "final_answer": {
    "final_answer": 2
  },
  "asset_list": {
    "local_rag_search": 1
  },
  "todo_create": {
    "invalid_json": 1
  },
  "local_rag_search": {
    "local_rag_search": 1,
    "invalid_json": 1
  },
  "public_web_search": {
    "invalid_json": 1,
    "public_web_search": 1
  },
  "calculator": {
    "invalid_json": 1,
    "final_answer": 1
  },
  "memory_write": {
    "invalid_json": 1
  },
  "todo_list": {
    "invalid_json": 1
  }
}
```

### sft

```json
{
  "weather_lookup": {
    "weather_lookup": 2
  },
  "memory_read": {
    "memory_read": 1
  },
  "final_answer": {
    "final_answer": 2
  },
  "asset_list": {
    "local_rag_search": 1
  },
  "todo_create": {
    "todo_create": 1
  },
  "local_rag_search": {
    "public_web_search": 1,
    "local_rag_search": 1
  },
  "public_web_search": {
    "public_web_search": 2
  },
  "calculator": {
    "calculator": 2
  },
  "memory_write": {
    "memory_write": 1
  },
  "todo_list": {
    "todo_list": 1
  }
}
```

### sft+dpo

```json
{
  "weather_lookup": {
    "weather_lookup": 2
  },
  "memory_read": {
    "memory_read": 1
  },
  "final_answer": {
    "final_answer": 2
  },
  "asset_list": {
    "local_rag_search": 1
  },
  "todo_create": {
    "todo_create": 1
  },
  "local_rag_search": {
    "public_web_search": 1,
    "local_rag_search": 1
  },
  "public_web_search": {
    "public_web_search": 2
  },
  "calculator": {
    "calculator": 2
  },
  "memory_write": {
    "memory_write": 1
  },
  "todo_list": {
    "todo_list": 1
  }
}
```

## Interview Notes

- SFT teaches the model to emit the tool-call JSON schema from demonstrations.
- DPO uses chosen/rejected decisions to reduce over-tooling and wrong-tool selection.
- This is a lightweight RLHF-style preference alignment demo, not a full PPO RLHF reproduction.

## Run Summary

- Base model: `Qwen/Qwen3-0.6B`.
- Hardware: NVIDIA GeForce RTX 4070 SUPER, 12GB VRAM.
- Dataset: 70 synthetic tool-decision examples; 55 train / 15 held-out eval, stratified across all 10 actions.
- SFT adapter: `experiments/tool_use_alignment/outputs/qwen3-0.6b-sft`, 80 steps, LoRA.
- Adopted DPO adapter: `experiments/tool_use_alignment/outputs/qwen3-0.6b-sft-dpo-mild10`, 10 steps, learning rate `2e-5`, beta `0.05`.
- A stronger 60-step DPO run reached eval reward accuracy `1.0` but reduced generation metrics, so the final reported SFT+DPO model uses the milder DPO adapter.
- Demo output: `experiments/tool_use_alignment/reports/demo_infer_mild10.txt`.
- Final metrics JSON: `experiments/tool_use_alignment/reports/final_real_metrics_rerun.json`.
