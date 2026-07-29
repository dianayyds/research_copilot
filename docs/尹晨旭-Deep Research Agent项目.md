# DeepResearch Copilot —— 单 Agent 深度研究智能体

**北京交通大学 · 个人科研项目** | 2026.02 - 2026.05 | [https://github.com/dianayyds/research_copilot](https://github.com/dianayyds/research_copilot)

面向复杂科研问题独立设计并实现的单 Agent 深度研究系统：Agent 自主拆解子问题、跑"检索—阅读评估—发现缺口—再检索"的迭代闭环，在长研究链路上做上下文管理，最终输出带可溯源引用的研究报告

- **Hybrid RAG 检索**：BGE-M3 向量检索 + BM25/jieba 关键词 + BGE Reranker 多路召回重排，向量负责语义、关键词精确命中术语、重排融合排序；自建语义通信 benchmark 上 Top-1 命中率由 80.0% 提升至 93.3%、MRR@5 由 0.869 提升至 0.967、Coverage@5 达 100%、Top-1 干扰漂移率由 13.3% 降至 6.7%
- **Agentic RAG**：基于 LangGraph StateGraph 带环图，由 Agent 自主决定检索轮数与终止时机；对每个子问题走"检索 → 相关性判断 → gap detection → 按需改写重检索"迭代闭环，仅在召回不足时触发 Query Rewrite / HyDE / Step-back 并支持关键数据的信源追溯递归，由 Reflection 结合最大轮数与 token 预算控制终止
- **自主任务拆解与重规划**：先将复杂问题拆成子问题、生成研究大纲，并在证据到手后识别盲区、动态补充子问题与改写研究计划
- **长链路上下文管理**：working / episodic / semantic 三层记忆，配合中间结果 compaction 摘要与 scratchpad 结构化记笔记；网页与文献只将摘要和来源 URL 写入全局 ResearchState、原文外置 MinIO，从源头抑制长链路上下文膨胀与失真
- **可溯源引用与矛盾处理**：每个关键结论绑定来源、可回指原文以防幻觉引用；来源冲突显式标注，交综合环节权衡而非擅自取舍
