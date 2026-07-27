from __future__ import annotations

import inspect
from dataclasses import fields, is_dataclass
from pathlib import Path
import json
from typing import Any

from experiments.tool_use_alignment.data import read_jsonl


def dataset_from_jsonl(path: Path):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install training extras with `pip install -e './backend[training]'`.") from exc
    return Dataset.from_list(read_jsonl(path))


def config_instance(config_cls: type, **kwargs: Any):
    accepted: set[str]
    if is_dataclass(config_cls):
        accepted = {field.name for field in fields(config_cls)}
    else:
        signature = inspect.signature(config_cls)
        if any(param.kind == param.VAR_KEYWORD for param in signature.parameters.values()):
            return config_cls(**kwargs)
        accepted = set(signature.parameters)
    return config_cls(**{key: value for key, value in kwargs.items() if key in accepted})


def trainer_tokenizer_kwargs(trainer_cls: type, tokenizer: Any) -> dict[str, Any]:
    parameters = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in parameters:
        return {"processing_class": tokenizer}
    if "tokenizer" in parameters:
        return {"tokenizer": tokenizer}
    return {}


def lora_config(target_modules: list[str] | None = None):
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise RuntimeError("Missing dependency: peft. Install training extras first.") from exc
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules
        or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def torch_dtype_and_device():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Missing dependency: torch. Install training extras first.") from exc
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return dtype, "cuda"
    return torch.float32, "cpu"


def load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Missing dependency: transformers. Install training extras first.") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(model_name: str):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("Missing dependency: transformers. Install training extras first.") from exc
    dtype, device = torch_dtype_and_device()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.config.use_cache = False
    return model


def load_model_for_inference(model_name: str, adapter_dir: Path | None = None):
    model = load_base_model(model_name)
    if adapter_dir:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Missing dependency: peft. Install training extras first.") from exc
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model


def generate_completion(model: Any, tokenizer: Any, prompt: str, *, max_new_tokens: int = 220) -> str:
    import torch

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return first_json_object_text(decoded) or decoded


def first_json_object_text(text: str) -> str:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return text[index : index + end]
    return ""
