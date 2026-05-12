from __future__ import annotations

import os
from statistics import mean, pstdev

import pytest

os.environ["DATABASE_URL"] = "sqlite:////tmp/research_copilot_test.db"
os.environ["VECTOR_STORE_PROVIDER"] = "stub"
os.environ["EMBEDDING_PROVIDER"] = "stub"
os.environ["LLM_PROVIDER"] = "stub"
os.environ["LLM_API_KEY"] = ""

from app.db_models import Asset  # noqa: E402
from app.models import TurnScopedRequest  # noqa: E402
from app.config import settings  # noqa: E402
from app.services import (  # noqa: E402
    build_query_rewrite,
    hybrid_retrieve,
    hyde_document_query,
    infer_query_rewrite_intent,
    lexical_keyword_query,
    planned_search_queries,
    query_rewrite_weight_map,
    query_rewrite_search_queries,
    step_back_query,
)
from app.vector_store import reset_vector_store  # noqa: E402


def make_asset(asset_id: str, title: str, sections: list[tuple[str, str]]) -> Asset:
    content = "\n\n".join(f"## {heading}\n\n{body}" for heading, body in sections)
    return Asset(id=asset_id, title=title, asset_type="note", content=content)


def semantic_communication_assets() -> list[Asset]:
    return [
        make_asset(
            "asset-sc-overview",
            "语义通信总览",
            [
                (
                    "SC-DEF-01 语义通信目标",
                    "GOLD:SC-DEF-01 语义通信关注传输含义、意图和任务相关信息，而传统 Shannon 通信主要优化符号或比特级正确传输、信道容量和误码率。语义通信允许省略冗余比特，只要接收端能够恢复意义并完成任务。",
                ),
                (
                    "SC-METRIC-01 评价指标变化",
                    "GOLD:SC-METRIC-01 语义通信不能只用 BER、BLER 或吞吐量评价，因为比特正确不等于意义正确。评测需要加入语义相似度、任务成功率、BERTScore、BLEU、QoE、压缩率和频谱效率。",
                ),
            ],
        ),
        make_asset(
            "asset-deepsc",
            "DeepSC 文本语义通信",
            [
                (
                    "DSC-TEXT-01 Transformer 端到端链路",
                    "GOLD:DSC-TEXT-01 DeepSC 文本系统使用 Transformer 提取句子语义，把语义编码和信道编码联合优化，通过端到端训练直接面向句子意义恢复，而不是逐比特可靠传输。",
                ),
                (
                    "DSC-METRIC-01 Sentence similarity",
                    "GOLD:DSC-METRIC-01 DeepSC 使用 sentence similarity 衡量接收句子和原始句子的语义接近程度，用来解决传统 BLEU 或误码率无法衡量意义保持的问题。",
                ),
                (
                    "DSC-SNR-01 低信噪比鲁棒性",
                    "GOLD:DSC-SNR-01 在低 SNR 或 noisy channel 下，DeepSC 通过学习语义冗余和上下文表示，优先保住句子意思，因此比传统分离式信源信道编码更稳。",
                ),
            ],
        ),
        make_asset(
            "asset-rdeepsc",
            "R-DeepSC 鲁棒语义通信",
            [
                (
                    "RDSC-NOISE-01 Semantic noise 分类",
                    "GOLD:RDSC-NOISE-01 R-DeepSC 区分 literal semantic noise 和 adversarial semantic noise：前者来自语义层面的文字或表达扰动，后者来自对抗样本对语义表示的攻击。",
                ),
                (
                    "RDSC-ROBUST-01 抵抗语义噪声机制",
                    "GOLD:RDSC-ROBUST-01 R-DeepSC 通过 calibrated self-attention、语义置信度校准和 adversarial training 提升对语义噪声的鲁棒性，减少错误语义传播。",
                ),
            ],
        ),
        make_asset(
            "asset-deepsc-st",
            "DeepSC-ST 语音语义通信",
            [
                (
                    "ST-TASK-01 语音任务",
                    "GOLD:ST-TASK-01 DeepSC-ST 面向语音识别和语音合成双任务，发送端提取与识别相关的语义特征，接收端既可以恢复文本，也可以合成语音。",
                ),
                (
                    "ST-FEATURE-01 语音语义特征",
                    "GOLD:ST-FEATURE-01 DeepSC-ST 不必完整传输波形，而是传输任务相关语音语义特征，从而降低带宽占用并保持语音含义。",
                ),
            ],
        ),
        make_asset(
            "asset-task-oriented",
            "任务导向语义通信",
            [
                (
                    "TOC-EDGE-01 多设备边缘推理",
                    "GOLD:TOC-EDGE-01 任务导向语义通信适合多设备边缘推理，因为终端只上传与任务决策相关的语义特征，边缘侧融合多设备语义信息完成分类、检测或控制。",
                ),
                (
                    "TOC-IB-01 DIB DDIB Tradeoff",
                    "GOLD:TOC-IB-01 DIB 和 DDIB 用 information bottleneck 思想压缩无关信息，在传输率和任务相关性之间做 rate-relevance tradeoff，减少通信开销但保持推理准确率。",
                ),
                (
                    "TOC-SR-01 Selective retransmission",
                    "GOLD:TOC-SR-01 selective retransmission 在多设备语义通信中只重传影响任务判断的错误语义片段，而不是重发所有比特，因此能降低重传开销并提升边缘推理可靠性。",
                ),
            ],
        ),
        make_asset(
            "asset-high-speed",
            "高速语义语音传输",
            [
                (
                    "HS-SIP-01 Superimposed pilot",
                    "GOLD:HS-SIP-01 高速移动场景存在多普勒频移、快速衰落和信道时变，superimposed pilot 通过叠加导频帮助接收端估计动态信道，提升语义语音传输鲁棒性。",
                ),
                (
                    "HS-SWITCH-01 SwitchAC-SIP",
                    "GOLD:HS-SWITCH-01 SwitchAC-SIP 根据动态信道状态在文本语义和音频语义之间切换，决定发送更紧凑的文本语义还是保留更多音频细节。",
                ),
                (
                    "HS-BW-01 带宽动态分配",
                    "GOLD:HS-BW-01 高速语义语音系统会在文本和音频语义之间做带宽动态分配，信道差时优先保障文本含义，信道好时恢复更多语音自然度。",
                ),
            ],
        ),
        make_asset(
            "asset-future",
            "语义通信未来趋势",
            [
                (
                    "FUTURE-ARCH-01 语义原生网络",
                    "GOLD:FUTURE-ARCH-01 未来语义通信架构会走向语义原生 6G 网络，把知识库、语义编码器、语义解码器、反馈闭环、边缘智能和任务评价整合到网络协议栈中。",
                ),
                (
                    "FUTURE-TREND-01 多模态与标准化",
                    "GOLD:FUTURE-TREND-01 语义通信未来趋势包括多模态语义通信、跨模态对齐、边缘智能部署、轻量化模型、标准化接口、可信语义和安全隐私保护。",
                ),
            ],
        ),
        make_asset(
            "asset-distractors",
            "干扰资料",
            [
                (
                    "DIST-SHANNON-01 传统信道容量",
                    "DIST:DIST-SHANNON-01 Shannon 信道容量讨论在给定带宽和噪声下能够可靠传输的最大比特率，重点是符号可靠性而非语义任务效果。",
                ),
                (
                    "DIST-SEMANTIC-WEB-01 语义网",
                    "DIST:DIST-SEMANTIC-WEB-01 语义网 Semantic Web 关注 RDF、本体、知识图谱和网页数据互联，不等同于无线网络中的语义通信。",
                ),
                (
                    "DIST-ASR-01 普通 ASR",
                    "DIST:DIST-ASR-01 普通 ASR 自动语音识别把语音转成文本，但不一定联合优化无线信道、语义压缩和任务效果。",
                ),
            ],
        ),
    ]


BENCHMARK_CASES = [
    ("Q1", "语义通信和传统 Shannon 通信最大的目标差异是什么？", {"SC-DEF-01"}),
    ("Q2", "为什么语义通信不能只用 BER 或 BLER 评价？", {"SC-METRIC-01"}),
    ("Q3", "低信噪比时 DeepSC 为什么还能保住句子意思？", {"DSC-SNR-01", "DSC-TEXT-01"}),
    ("Q4", "DeepSC 论文里 sentence similarity 是为了解决什么评价问题？", {"DSC-METRIC-01"}),
    ("Q5", "R-DeepSC 说的两类 semantic noise 分别是什么？", {"RDSC-NOISE-01"}),
    ("Q6", "R-DeepSC 用什么机制抵抗语义噪声？", {"RDSC-ROBUST-01"}),
    ("Q7", "DeepSC-ST 的发送端和接收端各自承担什么语音任务？", {"ST-TASK-01", "ST-FEATURE-01"}),
    ("Q8", "任务导向语义通信为什么适合多设备边缘推理？", {"TOC-EDGE-01", "TOC-IB-01"}),
    ("Q9", "DIB/DDIB 在任务导向通信里解决什么 tradeoff？", {"TOC-IB-01"}),
    ("Q10", "selective retransmission 在多设备语义通信中起什么作用？", {"TOC-SR-01"}),
    ("Q11", "高速语义语音传输为什么要引入 superimposed pilot？", {"HS-SIP-01"}),
    ("Q12", "SwitchAC-SIP 如何根据动态信道调整文本和音频语义？", {"HS-SWITCH-01", "HS-BW-01"}),
    ("Q13", "如果用户问少传比特但不影响理解，应该召回哪些语义通信依据？", {"SC-DEF-01", "TOC-IB-01"}),
    ("Q14", "语义通信里的恢复意义与普通 ASR 或语义网有什么区别？", {"SC-DEF-01"}),
    ("Q15", "未来语义通信架构会怎么和 6G、边缘智能、知识库结合？", {"FUTURE-ARCH-01", "FUTURE-TREND-01"}),
]


PARAPHRASES = {
    "Q3": [
        "DeepSC 相比传统链路在 noisy channel 下的优势是什么？",
        "低 SNR 下文本语义通信为什么还能保持语义相似？",
    ],
    "Q8": [
        "task-oriented semantic communication 为什么适合边缘侧多设备协同推理？",
        "多个终端做边缘推理时为什么只上传任务相关语义就够了？",
    ],
    "Q13": [
        "想省带宽但保持意思，语义通信靠什么依据实现？",
        "少发冗余比特还能让接收端理解，这对应语义通信的哪些机制？",
    ],
}


def request_for(query: str) -> TurnScopedRequest:
    return TurnScopedRequest(project_id="proj-bench", session_id="sess-bench", sequence_id=1, user_query=query)


def baseline_queries(query: str) -> tuple[list[str], dict[str, float]]:
    queries = planned_search_queries(query, query)[:3]
    return queries, {item: 1.0 for item in queries}


def step_back_queries(query: str) -> tuple[list[str], dict[str, float]]:
    intent = infer_query_rewrite_intent(query)
    queries = [query, step_back_query(query, query, intent), lexical_keyword_query(query, query, intent)]
    return queries, {queries[0]: 1.0, queries[1]: 0.56, queries[2]: 0.68}


def hyde_queries(query: str) -> tuple[list[str], dict[str, float]]:
    intent = infer_query_rewrite_intent(query)
    queries = [query, lexical_keyword_query(query, query, intent), hyde_document_query(query, query, intent)]
    return queries, {queries[0]: 1.0, queries[1]: 0.68, queries[2]: 0.54}


def hybrid_three_queries(query: str) -> tuple[list[str], dict[str, float]]:
    intent = infer_query_rewrite_intent(query)
    queries = [query, step_back_query(query, query, intent), hyde_document_query(query, query, intent)]
    return queries, {queries[0]: 1.0, queries[1]: 0.56, queries[2]: 0.54}


def hybrid_full_queries(query: str) -> tuple[list[str], dict[str, float]]:
    rewrite = build_query_rewrite(query, query)
    return query_rewrite_search_queries(rewrite), query_rewrite_weight_map(rewrite)


def evidence_text(item) -> str:
    return f"{item.title}\n{item.snippet}"


def first_gold_rank(evidence_items, gold_markers: set[str]) -> int | None:
    for index, item in enumerate(evidence_items, start=1):
        text = evidence_text(item)
        if any(marker in text for marker in gold_markers):
            return index
    return None


def coverage(evidence_items, gold_markers: set[str]) -> float:
    if not gold_markers:
        return 0.0
    matched = {
        marker
        for marker in gold_markers
        if any(marker in evidence_text(item) for item in evidence_items)
    }
    return len(matched) / len(gold_markers)


def is_drift(evidence_items) -> bool:
    if not evidence_items:
        return False
    return "DIST:" in evidence_text(evidence_items[0])


def evaluate_strategy(
    assets: list[Asset],
    case_id: str,
    query: str,
    gold_markers: set[str],
    query_plan: tuple[list[str], dict[str, float]],
) -> dict:
    reset_vector_store()
    queries, query_weights = query_plan
    retrieval = hybrid_retrieve(request_for(query), assets=assets, search_queries=queries, query_weights=query_weights)
    evidence_items = retrieval.evidence_items
    rank = first_gold_rank(evidence_items, gold_markers)
    return {
        "case_id": case_id,
        "query": query,
        "queries": queries,
        "hit1": 1.0 if rank == 1 else 0.0,
        "hit3": 1.0 if rank is not None and rank <= 3 else 0.0,
        "hit5": 1.0 if rank is not None and rank <= 5 else 0.0,
        "mrr5": 1.0 / rank if rank is not None and rank <= 5 else 0.0,
        "coverage5": coverage(evidence_items[:5], gold_markers),
        "top_gold_rank": rank or 999,
        "top_title": evidence_items[0].title if evidence_items else "",
        "drift": 1.0 if is_drift(evidence_items) else 0.0,
        "matched_query": evidence_items[0].tags[-1] if evidence_items and evidence_items[0].tags else "",
    }


def summarize(rows: list[dict]) -> dict[str, float]:
    return {
        "hit1": mean(row["hit1"] for row in rows),
        "hit3": mean(row["hit3"] for row in rows),
        "hit5": mean(row["hit5"] for row in rows),
        "mrr5": mean(row["mrr5"] for row in rows),
        "coverage5": mean(row["coverage5"] for row in rows),
        "avg_rank": mean(row["top_gold_rank"] for row in rows),
        "rank_stdev": pstdev(row["top_gold_rank"] for row in rows),
        "drift_rate": mean(row["drift"] for row in rows),
    }


def run_semantic_communication_benchmark() -> dict[str, object]:
    assets = semantic_communication_assets()
    strategies = {
        "baseline": baseline_queries,
        "step_back": step_back_queries,
        "hyde": hyde_queries,
        "hybrid_3": hybrid_three_queries,
        "hybrid_full": hybrid_full_queries,
    }
    rows_by_strategy: dict[str, list[dict]] = {name: [] for name in strategies}
    for case_id, query, gold in BENCHMARK_CASES:
        for name, query_builder in strategies.items():
            rows_by_strategy[name].append(evaluate_strategy(assets, case_id, query, gold, query_builder(query)))
    paraphrase_rows: dict[str, list[dict]] = {"baseline": [], "hybrid_full": []}
    gold_by_case = {case_id: gold for case_id, _, gold in BENCHMARK_CASES}
    for case_id, queries in PARAPHRASES.items():
        for query in queries:
            paraphrase_rows["baseline"].append(
                evaluate_strategy(assets, case_id, query, gold_by_case[case_id], baseline_queries(query))
            )
            paraphrase_rows["hybrid_full"].append(
                evaluate_strategy(assets, case_id, query, gold_by_case[case_id], hybrid_full_queries(query))
            )
    return {
        "rows_by_strategy": rows_by_strategy,
        "summary": {name: summarize(rows) for name, rows in rows_by_strategy.items()},
        "paraphrase_summary": {name: summarize(rows) for name, rows in paraphrase_rows.items()},
    }


def test_query_rewrite_preserves_original_query_and_adds_hybrid_variants() -> None:
    rewrite = build_query_rewrite("低信噪比时 DeepSC 为什么还能保住句子意思？", "DeepSC 文本系统")

    assert rewrite.original_query == "低信噪比时 DeepSC 为什么还能保住句子意思？"
    assert query_rewrite_search_queries(rewrite)[0] == rewrite.original_query
    strategies = {variant.strategy for variant in rewrite.variants}
    assert {"baseline_expansion", "step_back", "lexical_domain_expansion", "hyde"} <= strategies
    assert any("低SNR" in variant.query or "sentence similarity" in variant.query for variant in rewrite.variants)


def test_lookup_query_rewrite_keeps_title_and_author_intent() -> None:
    query = "《Adaptive Semantic Speech Transmission for High-Speed Scenarios》这篇论文的第一作者是谁"
    rewrite = build_query_rewrite(query, query)
    queries = query_rewrite_search_queries(rewrite)

    assert queries[0] == query
    assert rewrite.intent == "lookup"
    assert any("作者" in item for item in queries)
    assert not any(variant.strategy == "hyde" for variant in rewrite.variants)


def test_query_rewrite_config_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    query = "它未来怎么发展？"
    subject = "语义通信"

    monkeypatch.setattr(settings, "query_rewrite_enabled", False)
    disabled = build_query_rewrite(query, subject)
    assert disabled.generated_by == "legacy_planned_queries"
    assert {variant.strategy for variant in disabled.variants} == {"baseline"}

    monkeypatch.setattr(settings, "query_rewrite_enabled", True)
    monkeypatch.setattr(settings, "query_rewrite_hyde_enabled", False)
    monkeypatch.setattr(settings, "query_rewrite_step_back_enabled", False)
    rewrite = build_query_rewrite(query, subject)
    strategies = {variant.strategy for variant in rewrite.variants}
    assert "hyde" not in strategies
    assert "step_back" not in strategies
    assert {"baseline_expansion", "standalone", "lexical_domain_expansion"} <= strategies


def test_query_rewrite_benchmark_outperforms_baseline_on_semantic_communication() -> None:
    benchmark = run_semantic_communication_benchmark()
    summary = benchmark["summary"]
    baseline = summary["baseline"]
    hybrid_full = summary["hybrid_full"]
    hybrid_three = summary["hybrid_3"]
    paraphrase = benchmark["paraphrase_summary"]

    assert hybrid_full["hit5"] >= baseline["hit5"]
    assert hybrid_full["mrr5"] > baseline["mrr5"]
    assert hybrid_full["coverage5"] > baseline["coverage5"]
    assert hybrid_three["mrr5"] >= baseline["mrr5"]
    assert paraphrase["hybrid_full"]["mrr5"] > paraphrase["baseline"]["mrr5"]
    assert hybrid_full["drift_rate"] <= baseline["drift_rate"]
