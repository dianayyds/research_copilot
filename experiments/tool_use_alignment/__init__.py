"""Local SFT + DPO demo for agent tool-use decisions."""

from experiments.tool_use_alignment.data import build_cases, generate_datasets
from experiments.tool_use_alignment.metrics import evaluate_records, parse_decision

__all__ = ["build_cases", "evaluate_records", "generate_datasets", "parse_decision"]
