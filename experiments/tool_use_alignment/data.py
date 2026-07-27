from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.tool_use_alignment.prompts import build_prompt, decision_json


@dataclass(frozen=True)
class ToolUseCase:
    case_id: str
    category: str
    user_query: str
    thought: str
    action: str
    arguments: dict[str, Any]
    rejected_action: str
    rejected_arguments: dict[str, Any]
    rejected_thought: str

    @property
    def completion(self) -> str:
        return decision_json(self.thought, self.action, self.arguments)

    @property
    def rejected_completion(self) -> str:
        return decision_json(self.rejected_thought, self.rejected_action, self.rejected_arguments)


def _case(
    case_id: str,
    category: str,
    query: str,
    action: str,
    arguments: dict[str, Any],
    *,
    thought: str,
    rejected_action: str,
    rejected_arguments: dict[str, Any],
    rejected_thought: str,
) -> ToolUseCase:
    return ToolUseCase(
        case_id=case_id,
        category=category,
        user_query=query,
        thought=thought,
        action=action,
        arguments=arguments,
        rejected_action=rejected_action,
        rejected_arguments=rejected_arguments,
        rejected_thought=rejected_thought,
    )


def build_cases() -> list[ToolUseCase]:
    cases: list[ToolUseCase] = []

    weather_items = [
        ("北京", "明天北京会下雨吗？如果我要去清华附近调研，要不要带伞？"),
        ("上海", "上海今天气温怎么样，晚上外出冷不冷？"),
        ("深圳", "深圳这两天适合安排户外访谈吗？看一下降雨和风速。"),
        ("杭州", "杭州明天会议结束后适合散步吗？查实时天气。"),
        ("成都", "成都现在天气如何，去电子科大附近要不要带外套？"),
        ("武汉", "武汉今天热不热？帮我看气温和体感温度。"),
        ("西安", "西安明天下午采样会不会下雨？"),
        ("南京", "南京今晚风大吗？适不适合骑车去学校？"),
    ]
    for index, (_, query) in enumerate(weather_items, start=1):
        cases.append(
            _case(
                f"weather-{index:02d}",
                "weather",
                query,
                "weather_lookup",
                {"query": query},
                thought="用户询问实时天气和出行环境，需要调用天气工具。",
                rejected_action="final_answer",
                rejected_arguments={"answer": "根据常识判断天气可能变化，请自行查看天气。"},
                rejected_thought="不调用工具直接猜测天气。",
            )
        )

    public_items = [
        "联网查一下 Qwen3-0.6B 的模型卡，确认它的参数量和上下文长度。",
        "帮我搜索最新的 DeepSeek API 文档入口。",
        "网上查一下 Hugging Face TRL DPOTrainer 现在推荐的数据格式。",
        "请搜公开资料：Open-Meteo forecast API 支持哪些 current 参数？",
        "帮我找一下 PEFT LoRA 官方文档，确认 LoRA 的作用。",
        "联网看看最近有没有 Qwen3 相关的 agent tool use 说明。",
        "请搜索 FunctionGemma 270M 的模型定位和限制。",
        "查一下 Hugging Face datasets JSONL 加载方式的公开文档。",
    ]
    for index, query in enumerate(public_items, start=1):
        cases.append(
            _case(
                f"public-{index:02d}",
                "public_web",
                query,
                "public_web_search",
                {"query": query},
                thought="用户明确要求联网或查询最新公开资料，需要公开搜索。",
                rejected_action="local_rag_search",
                rejected_arguments={"query": query},
                rejected_thought="错误地只查本地资料，可能拿不到最新公开信息。",
            )
        )

    local_items = [
        "在本项目资料里检索 DeepSC 为什么低信噪比下还能保住句子意思。",
        "根据我上传的论文笔记，找出 R-DeepSC 抵抗 semantic noise 的机制。",
        "检索当前知识库：DIB/DDIB 解决什么 rate-relevance tradeoff？",
        "从项目资产里找 SwitchAC-SIP 如何切换文本和音频语义。",
        "基于本项目资料说明 semantic communication 和 Shannon 通信的目标差异。",
        "查一下当前仓库文档里 Research Copilot 的技术亮点。",
        "在知识库里找 selective retransmission 的作用。",
        "用本地资料回答 DeepSC-ST 的发送端和接收端任务。",
        "项目代码里有哪些 agent 工具 schema？请先检索本项目。",
        "从我的论文资产中找未来 6G 语义原生网络的描述。",
    ]
    for index, query in enumerate(local_items, start=1):
        cases.append(
            _case(
                f"local-{index:02d}",
                "local_rag",
                query,
                "local_rag_search",
                {"query": query},
                thought="用户要求基于本项目资料或知识库回答，需要本地 RAG 检索。",
                rejected_action="public_web_search",
                rejected_arguments={"query": query},
                rejected_thought="错误地查询公网，忽略了项目内证据约束。",
            )
        )

    memory_write_items = [
        ("writing_style", "请记住：我的论文总结优先用中文，结构要按 motivation-method-result-limitations。"),
        ("interview_focus", "帮我记住，我面试 Agent 岗位时想重点讲工具路由和可观测 trace。"),
        ("metric_preference", "请记住：以后评价检索实验时优先报告 recall@5 和 MRR。"),
        ("demo_constraint", "记住这个约束：训练 demo 不能提交大模型文件到 Git。"),
        ("reading_scope", "请记住：语义通信综述先关注 DeepSC、R-DeepSC 和 task-oriented semantic communication。"),
        ("language_preference", "帮我记住：回答我项目相关问题时默认用中文。"),
    ]
    for index, (key, query) in enumerate(memory_write_items, start=1):
        cases.append(
            _case(
                f"memory-write-{index:02d}",
                "memory_write",
                query,
                "memory_write",
                {"key": key, "content": query, "importance": 0.8},
                thought="用户明确要求记住偏好或约束，需要写入记忆。",
                rejected_action="memory_read",
                rejected_arguments={"query": query},
                rejected_thought="错误地读取记忆，没有保存用户新偏好。",
            )
        )

    memory_read_items = [
        "我刚才让你记住的面试重点是什么？",
        "之前保存过的论文总结格式偏好是什么？",
        "项目记忆里有没有记录我对训练 demo 的约束？",
        "你还记得我说以后评估检索实验优先看什么指标吗？",
        "刚才我让你记住的语义通信阅读范围有哪些？",
        "查一下项目记忆，我默认希望你用什么语言回答项目问题？",
    ]
    for index, query in enumerate(memory_read_items, start=1):
        cases.append(
            _case(
                f"memory-read-{index:02d}",
                "memory_read",
                query,
                "memory_read",
                {"query": query},
                thought="用户询问之前保存的信息，需要读取项目记忆。",
                rejected_action="local_rag_search",
                rejected_arguments={"query": query},
                rejected_thought="错误地检索资料库，记忆问题应查 memory。",
            )
        )

    todo_create_items = [
        ("整理 SFT 和 DPO 的面试讲稿", "请给当前项目创建 TODO：整理 SFT 和 DPO 的面试讲稿。"),
        ("跑一次 Qwen3 0.6B SFT 小实验", "帮我添加待办：跑一次 Qwen3 0.6B SFT 小实验，优先级高。"),
        ("补充工具路由指标图", "新建 TODO，标题是补充工具路由指标图。"),
        ("检查 README 命令是否能复制运行", "把检查 README 命令是否能复制运行加入待办。"),
        ("比较 base、SFT、DPO 三组指标", "请创建一个待办：比较 base、SFT、DPO 三组指标。"),
        ("准备个人主页项目描述", "添加 TODO：准备个人主页项目描述。"),
    ]
    for index, (title, query) in enumerate(todo_create_items, start=1):
        cases.append(
            _case(
                f"todo-create-{index:02d}",
                "todo_create",
                query,
                "todo_create",
                {"title": title, "description": "", "priority": "medium"},
                thought="用户明确要求创建或添加待办，需要调用 todo_create。",
                rejected_action="final_answer",
                rejected_arguments={"answer": "好的，我会记得这个事项。"},
                rejected_thought="错误地口头答应，没有真正创建 TODO。",
            )
        )

    todo_list_items = [
        "列出当前项目还有哪些 TODO。",
        "查看一下未完成的待办列表。",
        "当前项目 TODO 里有哪些高优先级事项？",
        "帮我看看这个研究项目的待办清单。",
        "列出所有和训练 demo 有关的 TODO。",
    ]
    for index, query in enumerate(todo_list_items, start=1):
        cases.append(
            _case(
                f"todo-list-{index:02d}",
                "todo_list",
                query,
                "todo_list",
                {},
                thought="用户要查看待办列表，需要调用 todo_list。",
                rejected_action="todo_create",
                rejected_arguments={"title": query, "description": "", "priority": "medium"},
                rejected_thought="错误地新建待办，而不是查看列表。",
            )
        )

    asset_items = [
        "当前知识库里有哪些论文和资料？",
        "帮我列出已经导入的项目资产。",
        "这个 workspace 现在有哪些文档可用于 RAG？",
        "查一下我上传过哪些资料。",
        "请列出可检索的资产清单。",
    ]
    for index, query in enumerate(asset_items, start=1):
        cases.append(
            _case(
                f"asset-{index:02d}",
                "asset_list",
                query,
                "asset_list",
                {},
                thought="用户询问资料或资产清单，需要调用 asset_list。",
                rejected_action="local_rag_search",
                rejected_arguments={"query": query},
                rejected_thought="错误地检索内容，用户需要的是资产列表。",
            )
        )

    calc_items = [
        ("(87 + 93 + 91) / 3", "帮我计算 (87 + 93 + 91) / 3。"),
        ("128 * 0.75", "训练集 128 条，75% 是多少条？"),
        ("42 / 56", "如果 action 命中 42 条，总共 56 条，准确率是多少？"),
        ("5 * 8 + 12", "算一下 5 * 8 + 12。"),
        ("1 - 0.18", "over-tool rate 从 18% 降到多少才是 82% 正常率？"),
        ("2048 / 4", "2048 tokens 按 batch 4 平均是多少？"),
        ("0.92 * 50", "50 个样本 recall@5 为 0.92 时命中多少个？"),
        ("(16 + 24) / 80", "SFT 训练 80 条里有 16 条天气和 24 条本地检索，占比多少？"),
    ]
    for index, (expression, query) in enumerate(calc_items, start=1):
        cases.append(
            _case(
                f"calc-{index:02d}",
                "calculator",
                query,
                "calculator",
                {"expression": expression},
                thought="用户要求精确算术或指标核算，需要调用 calculator。",
                rejected_action="final_answer",
                rejected_arguments={"answer": "这个数值大约可以心算得到。"},
                rejected_thought="错误地跳过计算工具，容易算错。",
            )
        )

    final_items = [
        ("把“我做过 SFT”这句话改得更专业一点。", "我实现过面向 Agent 工具决策的监督微调实验。"),
        ("用一句话解释 SFT 和 DPO 的区别，不需要查资料。", "SFT 学习示范答案，DPO 学习偏好排序。"),
        ("把这句话翻译成英文：我做过本地 SFT 和 DPO 工具使用对齐 demo。", "I built a local SFT and DPO demo for aligning agent tool-use decisions."),
        ("给我的个人主页写一个 20 字以内的小标题：Agent 工具调用训练 demo。", "Agent 工具调用对齐实验"),
        ("不用联网，简单解释什么是 LoRA。", "LoRA 通过训练少量低秩适配参数来降低微调成本。"),
        ("帮我把“轻量级 RLHF-style preference alignment”翻译成中文。", "轻量级 RLHF 风格偏好对齐。"),
        ("请把这段话压缩成一句：SFT 学 schema，DPO 学偏好，指标看 action accuracy。", "该 demo 用 SFT 学习工具调用格式，用 DPO 优化工具使用偏好，并用 action accuracy 等指标评估。"),
        ("不需要工具，帮我列三个面试时可讲的关键词。", "SFT、DPO、tool-needed F1。"),
    ]
    for index, (query, answer) in enumerate(final_items, start=1):
        cases.append(
            _case(
                f"final-{index:02d}",
                "final_answer",
                query,
                "final_answer",
                {"answer": answer},
                thought="用户请求可直接完成，不需要外部工具。",
                rejected_action="public_web_search",
                rejected_arguments={"query": query},
                rejected_thought="错误地过度使用搜索工具。",
            )
        )

    return cases


def sft_record(case: ToolUseCase, split: str) -> dict[str, Any]:
    prompt = build_prompt(case.user_query)
    return {
        "id": case.case_id,
        "split": split,
        "category": case.category,
        "prompt": prompt,
        "completion": case.completion,
        "text": f"{prompt}{case.completion}",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": case.completion},
        ],
        "expected": {"action": case.action, "arguments": case.arguments},
    }


def dpo_record(case: ToolUseCase, split: str) -> dict[str, Any]:
    return {
        "id": case.case_id,
        "split": split,
        "category": case.category,
        "prompt": build_prompt(case.user_query),
        "chosen": case.completion,
        "rejected": case.rejected_completion,
        "expected": {"action": case.action, "arguments": case.arguments},
        "rejected_decision": {"action": case.rejected_action, "arguments": case.rejected_arguments},
    }


def split_cases(cases: list[ToolUseCase], *, seed: int, train_ratio: float) -> tuple[list[ToolUseCase], list[ToolUseCase]]:
    rng = random.Random(seed)
    grouped: dict[str, list[ToolUseCase]] = {}
    for case in cases:
        grouped.setdefault(case.action, []).append(case)

    train_cases: list[ToolUseCase] = []
    eval_cases: list[ToolUseCase] = []
    for action_cases in grouped.values():
        shuffled = list(action_cases)
        rng.shuffle(shuffled)
        eval_count = max(1, round(len(shuffled) * (1.0 - train_ratio)))
        eval_count = min(len(shuffled) - 1, eval_count)
        eval_cases.extend(shuffled[:eval_count])
        train_cases.extend(shuffled[eval_count:])

    rng.shuffle(train_cases)
    rng.shuffle(eval_cases)
    return train_cases, eval_cases


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def generate_datasets(output_dir: Path, *, seed: int = 42, train_ratio: float = 0.8) -> dict[str, Any]:
    train_cases, eval_cases = split_cases(build_cases(), seed=seed, train_ratio=train_ratio)
    paths = {
        "sft_train": output_dir / "sft_train.jsonl",
        "sft_eval": output_dir / "sft_eval.jsonl",
        "dpo_train": output_dir / "dpo_train.jsonl",
        "dpo_eval": output_dir / "dpo_eval.jsonl",
        "eval_cases": output_dir / "eval_cases.jsonl",
    }
    write_jsonl(paths["sft_train"], [sft_record(case, "train") for case in train_cases])
    write_jsonl(paths["sft_eval"], [sft_record(case, "eval") for case in eval_cases])
    write_jsonl(paths["dpo_train"], [dpo_record(case, "train") for case in train_cases])
    write_jsonl(paths["dpo_eval"], [dpo_record(case, "eval") for case in eval_cases])
    write_jsonl(paths["eval_cases"], [sft_record(case, "eval") for case in eval_cases])
    return {
        "output_dir": str(output_dir),
        "seed": seed,
        "train_ratio": train_ratio,
        "train_cases": len(train_cases),
        "eval_cases": len(eval_cases),
        "paths": {key: str(path) for key, path in paths.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic SFT/DPO data for tool-use alignment.")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/tool_use_alignment/data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()
    summary = generate_datasets(args.output_dir, seed=args.seed, train_ratio=args.train_ratio)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
