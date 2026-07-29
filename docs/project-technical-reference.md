# 项目技术速查

最后更新：2026-06-27

本文档记录 Research Copilot 当前重要的运行时选型和实现细节。配置的事实来源以 `backend/app/config.py` 和 `.env.example` 为准；当模型、检索、记忆或工具路由行为变化时，需要同步更新本文档。

## 执行框架

前端当前默认使用 Plan-ReAct MCP 路径：

- 前端请求路径：`/api/v1/projects/{project_id}/sessions/{session_id}/run/stream`
- 后端流式函数：`stream_research_events`
- 后端非流式主函数：`run_research`
- Planner 模式：`plan_react_mcp`
- 默认执行模式环境变量：`EXECUTION_MODE=plan_react_mcp`

默认流程：

1. 从会话历史、资产、TODO 状态和分层记忆构建项目上下文。
2. 初始化 GitHub MCP client，执行 `initialize -> notifications/initialized` 握手。
3. 通过 MCP 拉取能力目录：`tools/list`、`resources/list`、`prompts/list`，并处理 cursor 分页。
4. 调用 DeepSeek 输出结构化 plan：`plan_summary` 和 `tasks[]`。
5. 每个 plan node 独立执行 ReAct loop，默认最多 `PLAN_REACT_DEFAULT_NODE_ITERATIONS=4` 轮。
6. 每轮 ReAct 由 DeepSeek 输出固定 JSON：`thought`、`action`、`server`、`capability_type`、`name`、`arguments`、`is_done`、`answer_fragment`。
7. 当 `action=mcp_call` 时执行 MCP tool/resource/prompt，并把结果追加为 observation。
8. 当 `is_done=true` 或 action 不是 MCP 调用时结束当前节点，进入下一个 plan node。
9. 所有节点完成后，DeepSeek 基于 observations 生成最终答案和 citation。
10. 沉淀 working、episodic、semantic 三层记忆。
11. 持久化本次 run，包括 context、plan、retrieval、answer、memory 和 trace metadata。

默认 plan 限制：

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| `PLAN_REACT_MAX_TASKS` | `6` | 单次运行最多计划节点数。 |
| `PLAN_REACT_DEFAULT_NODE_ITERATIONS` | `4` | 每个节点默认 ReAct 循环上限。 |
| `PLAN_REACT_LLM_TIMEOUT_SECONDS` | `60.0` | planner、ReAct decision、final answer 的单次 LLM 超时。 |

`/run/stream` 会在 Plan-ReAct MCP 后台执行时通过队列转发事件：planner 产出 plan 后先发送 `plan`，每个 ReAct action/observation 追加发送 `trace`，最后发送 summary、answer、quality 和 complete。流式事件仍兼容旧前端事件类型：

- `plan`
- `trace`
- `solver_summary`
- `answer_delta`
- `answer_quality`
- `complete`

其中 `trace` 增加以下 action：

- `plan_node_start`
- `react_action`
- `mcp_observation`
- `plan_node_complete`
- `blocked_tool`

旧的本地 Plan-and-Solve/RAG 逻辑仍保留在 `services.py` 内部，作为兼容路径存在，但不再是前端默认调用方式。项目中还存在一条单独的 Plan-Act-Observe Agent 路径：

- 接口路径：`/api/v1/projects/{project_id}/sessions/{session_id}/agent/run/stream`
- 后端流式函数：`stream_agent_events`
- 后端非流式主函数：`run_agent_research`
- Planner 模式：`agent_loop`
- 最大步数环境变量：`AGENT_MAX_STEPS=5`

Agent 路径允许模型或回退规则每一步选择一个工具，执行后写入观察结果，再继续下一步，直到输出 `final_answer` 或达到最大步数。

## LLM 配置

默认 LLM 配置：

| 配置项 | 默认值 |
|---|---|
| `LLM_PROVIDER` | `deepseek` |
| `LLM_MODEL` | `deepseek-chat` |
| `LLM_API_BASE` | `https://api.deepseek.com` |
| `LLM_API_KEY` | 默认空 |

默认 `plan_react_mcp` 路径强依赖 LLM。只有当 `LLM_PROVIDER=deepseek` 且配置了 `LLM_API_KEY` 时，默认 `/run` 和 `/run/stream` 才会继续执行；如果 key 为空，会返回明确配置错误，不再伪装成旧的确定性 Plan-and-Solve 流程。

依赖 LLM 的行为：

- Plan-ReAct MCP 的结构化 plan 生成。
- 每个 plan node 内的 ReAct decision 生成。
- 基于 MCP observations 的最终答案综合。
- 流式答案输出。
- weather/public-web 工具返回事实后的答案润色。
- `LLM_TOOL_PLANNER_ENABLED=true` 时的可选实时工具 planner。
- `/agent/run` 旧 Agent 模式下的可选下一步决策和最终综合。

## MCP Client 与 GitHub MCP Server

MCP client 实现在 `backend/app/mcp_client.py`，没有使用 MCP SDK，而是手写 JSON-RPC 2.0 通信层。

支持能力：

| 能力 | MCP method | 说明 |
|---|---|---|
| 初始化 | `initialize` + `notifications/initialized` | 启动后握手，并发送协议版本。 |
| 工具列表 | `tools/list` | 支持 cursor 分页。 |
| 工具调用 | `tools/call` | 用于执行 GitHub MCP tool。 |
| 资源列表 | `resources/list` | 支持 cursor 分页。 |
| 资源读取 | `resources/read` | 永远视为只读。 |
| Prompt 列表 | `prompts/list` | 支持 cursor 分页。 |
| Prompt 获取 | `prompts/get` | 永远视为只读。 |

Transport：

- `StdioMCPTransport` 是首版主路径。
- stdio server 通过 `subprocess.Popen` 启动。
- JSON-RPC request 逐行写入 stdin。
- JSON-RPC response 从 stdout 逐行读取。
- stderr 只进入服务端日志，不进入模型上下文。
- `StreamableHTTPMCPTransport` 保留接口骨架，支持 POST JSON-RPC、session id 保存和 JSON 响应解析；完整 SSE 消费不是首版主路径。

GitHub MCP 默认注册：

| 配置项 | 默认值 |
|---|---|
| `MCP_ENABLED` | `true` |
| `MCP_GITHUB_ENABLED` | `true` |
| `MCP_GITHUB_TRANSPORT` | `stdio` |
| `MCP_GITHUB_COMMAND` | `docker run --rm -i -e GITHUB_PERSONAL_ACCESS_TOKEN ghcr.io/github/github-mcp-server` |
| `MCP_GITHUB_ALLOWED_SIDE_EFFECT_TOOLS` | 空 |
| `MCP_GITHUB_URL` | 空 |
| `MCP_PROTOCOL_VERSION` | `2025-06-18` |
| `MCP_REQUEST_TIMEOUT_SECONDS` | `30.0` |
| `MCP_INITIALIZE_TIMEOUT_SECONDS` | `20.0` |
| `GITHUB_PERSONAL_ACCESS_TOKEN` | 默认空，默认路径必需 |

Docker 行为：

- `backend/Dockerfile` 安装 Docker CLI。
- `docker-compose.yml` 把 `/var/run/docker.sock` 挂载到 `runtime-api` 容器。
- runtime 容器通过 Docker socket 启动 `ghcr.io/github/github-mcp-server`。
- Docker socket 挂载是高权限本地开发配置，后续生产部署需要收敛权限边界。

工具风险策略：

| 能力 | 执行策略 |
|---|---|
| `resources/read` | 永远只读，自动执行。 |
| `prompts/get` | 永远只读，自动执行。 |
| tool 带 `readOnlyHint=true` | 视为只读，自动执行。 |
| tool 名称以 `get`、`list`、`search`、`read` 开头 | 启发式视为只读，自动执行。 |
| 其他 tool | 视为有副作用；只有出现在 `MCP_GITHUB_ALLOWED_SIDE_EFFECT_TOOLS` 中才执行。 |
| 未 allowlist 的有副作用 tool | 不执行真实 MCP 调用，返回 `blocked_tool` observation。 |

## Embedding 模型

默认 embedding 配置：

| 配置项 | 默认值 |
|---|---|
| `EMBEDDING_PROVIDER` | `local` |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` |
| `EMBEDDING_MODEL_DIR` | `/models` |
| `EMBEDDING_DEVICE` | `cpu` |
| `EMBEDDING_DIMENSION` | `1024` |
| `EMBEDDING_MAX_LENGTH` | `2048` |
| `EMBEDDING_BATCH_SIZE` | `4` |

实现细节：

- 实现在 `backend/app/vector_store.py` 的 `BgeM3Embedder` 中。
- 模型文件通过 `huggingface_hub.snapshot_download` 下载到 `EMBEDDING_MODEL_DIR`。
- Embedding 使用 Hugging Face `AutoTokenizer` 和 `AutoModel`。
- 向量取自 `outputs.last_hidden_state[:, 0]`。
- 向量在存储或查询前会做 L2 normalize。
- Qdrant collection 使用 cosine distance 和配置的 embedding dimension。

## 分词与稀疏检索

分词实现在 `backend/app/vector_store.py` 中。

中文友好分词：

- 主要方法：`jieba.cut_for_search(text)`。
- token 会去除空白并转小写。
- 额外使用正则 `[a-zA-Z0-9_./-]+` 抽取英文、数字和代码风格 token。
- 如果 `jieba` 不可用，则回退到正则 `[\u4e00-\u9fff]+` 抽取连续中文片段。
- token 会按原顺序去重。

稀疏检索：

- 依赖库：`rank-bm25`。
- 类：`BM25Okapi`。
- 函数：`bm25_search`。
- 如果 `rank_bm25` 不可用，则退回到 lexical overlap scoring。

## 重排模型

默认 reranker 配置：

| 配置项 | 默认值 |
|---|---|
| `RERANKER_PROVIDER` | `local` |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` |
| `RERANKER_MODEL_DIR` | `/models` |
| `RERANKER_DEVICE` | `cpu` |
| `RERANKER_BATCH_SIZE` | `2` |
| `RERANKER_MAX_LENGTH` | `.env.example`、Compose 和代码回退值均为 `1024` |

实现细节：

- 实现在 `backend/app/vector_store.py` 的 `BgeReranker` 中。
- 使用 Hugging Face `AutoTokenizer` 和 `AutoModelForSequenceClassification`。
- 对 query-passage pair 进行打分，输入形式为 `[query, passage]`。
- 使用 raw logits 作为 rerank score。
- 实际 max length 取配置值、tokenizer max length 和 model max positions 三者中的最小值。

## 向量库与 Hybrid RAG

默认向量库配置：

| 配置项 | 默认值 |
|---|---|
| `VECTOR_STORE_PROVIDER` | `qdrant` |
| `QDRANT_COLLECTION` | `knowledge_chunks` |
| `SEMANTIC_MEMORY_COLLECTION` | `semantic_memory_facts` |
| `RETRIEVAL_LIMIT` | `5` |
| `RETRIEVAL_CANDIDATE_LIMIT` | `16` |

主知识库检索由 `QdrantVectorStore.search` 实现：

1. 候选数量为 `max(RETRIEVAL_CANDIDATE_LIMIT, RETRIEVAL_LIMIT * 4)`。
2. 使用 BGE-M3 query embedding 在 Qdrant 中执行 dense search。
3. 在当前 asset chunks 上执行 sparse BM25 search。
4. 使用类似 Reciprocal Rank Fusion 的方式融合 dense/sparse 命中：`1 / (60 + rank)`。
5. 使用 BGE reranker 对融合后的候选进行重排。
6. 返回 top `RETRIEVAL_LIMIT` 条 evidence items。

测试或 stub vector store 会用 lexical overlap 作为 dense 侧，再执行 BM25、fusion 和确定性 score 排序。

## Query Rewrite

Query Rewrite 是确定性的，实现在 `backend/app/services.py` 中。它主要服务于旧本地 RAG 兼容路径、`local_rag_search` Agent 工具和需要项目资产检索的内部逻辑；默认 `plan_react_mcp` 主路径优先使用 MCP observation 作为外部工具证据。

默认 Query Rewrite 配置：

| 配置项 | 默认值 |
|---|---|
| `QUERY_REWRITE_ENABLED` | `true` |
| `QUERY_REWRITE_HYDE_ENABLED` | `true` |
| `QUERY_REWRITE_STEP_BACK_ENABLED` | `true` |
| `QUERY_REWRITE_MAX_QUERIES` | `10` |

意图识别可能把 query 分类为：

- `lookup`
- `taxonomy`
- `robustness`
- `compression`
- `comparison`
- `trend`
- `architecture`
- `technology_choice`
- `evaluation`
- `safety`
- `concept_overview`
- `general`

Rewrite 策略：

| 策略 | 权重 | 作用 |
|---|---:|---|
| `baseline_expansion` | `1.0` | 保留旧版 planned search queries，保证召回安全。 |
| `standalone` | `0.98` | 将短追问或代词较多的问题改写成独立检索 query。 |
| `lexical_domain_expansion` | `0.68` | 组合 tokenizer 输出和领域词，增强稀疏检索。 |
| `step_back` | `0.56` | 检索更高层的原理、架构和上下文。 |
| `hyde` | `0.54` | 生成 HyDE 风格的假想答案/文档 query，用于 dense retrieval。 |

重要行为：

- 变体会按 normalize 后的小写 query 文本去重。
- 变体先按权重排序，再按原始插入顺序排序。
- 只使用 top `QUERY_REWRITE_MAX_QUERIES` 个变体。
- 对 `lookup` 和 `taxonomy` 意图跳过 HyDE，以降低检索漂移。
- 多 query 结果融合时会应用 query weight。
- 如果关闭 Query Rewrite，则回退到旧版 `planned_search_queries`。

Replan 行为：

- 函数：`should_replan`。
- 没有证据时触发 replan。
- top evidence score 较弱时也会触发 replan：
  - lookup query 阈值：`0.28`。
  - 其他 query 阈值：`0.18`。
- Replan 会把检索扩展到标题页、摘要、第一页、全文和 introduction 风格查询。

## 记忆系统

记忆系统实现在 `backend/app/memory_manager.py` 和 `backend/app/semantic_store.py` 中。

默认记忆限制：

| 配置项 | 默认值 |
|---|---|
| `WORKING_MEMORY_TOKEN_THRESHOLD` | `1200` |
| `WORKING_MEMORY_COMPACTION_RATIO` | `0.75` |
| `EPISODIC_MEMORY_LIMIT` | `4` |
| `SEMANTIC_MEMORY_LIMIT` | `6` |

记忆层级：

- Working memory：当前 session 的 query 和 answer 摘要；低于 token 阈值时全部注入上下文。
- Episodic memory：项目级研究事件，按 token overlap、recency 和 importance 打分。
- Semantic memory：结构化事实，存储在 MySQL，并同步索引到 Qdrant。

Working memory 不再按每轮 research run 自动沉淀长期记忆。每轮结束后先同步写入短期记忆并返回回答；当当前 session 的短期记忆 token 总量超过 `WORKING_MEMORY_TOKEN_THRESHOLD` 时，系统会在响应后后台选择最旧的一批 working memory，累计到 `WORKING_MEMORY_TOKEN_THRESHOLD * WORKING_MEMORY_COMPACTION_RATIO` 的 token 规模，交给 LLM 记忆整理器一次性总结出 episodic memory 和 semantic memory。该流程不对每条候选执行 `ADD / UPDATE / DELETE / NOOP` 判断；LLM 返回结构化长期记忆后，后端负责校验、持久化和同步 semantic memory 索引，随后删除已整理的 working memory。LLM 不可用或没有返回可保存记忆时，系统跳过长期记忆整理并保留 working memory。

Semantic memory fact types：

- `fact`
- `open_question`
- `progress`
- `decision`
- `preference`

Semantic memory 检索复用核心检索栈：

1. 在 `semantic_memory_facts` Qdrant collection 中使用 BGE-M3 embedding 做 dense search。
2. 对 MySQL 中近期 semantic facts 做 sparse BM25。
3. 使用共享的 ranking merge 函数做 fusion。
4. 对 `fact_type + statement` 使用 BGE reranker 重排。

## 资产导入

支持上传的扩展名：

- `.txt`
- `.md`
- `.markdown`
- `.pdf`

解析行为：

- 文本解码按顺序尝试 `utf-8`、`utf-8-sig`、`gb18030`。
- Markdown front matter 存在时会被移除。
- PDF 默认按 `auto` 顺序尝试 GROBID、Docling、pypdf；未配置 GROBID 且未安装 Docling 时回退到 `pypdf.PdfReader`。
- pypdf 解析的 PDF 内容会保留 `<!-- page:N -->` 页面标记，供 chunk metadata 记录页码范围。
- 资产 chunk 使用 parent/child 两级切分，默认 parent `1800` tokens、child `480` tokens、child overlap `80` tokens。

相关存储服务：

- MySQL 存储 project、session、asset、run、TODO 和 memory 记录。
- Redis 存储断点续传元数据和 chunk bitmap。
- MinIO 存储原始上传分片和 compose 后的源文件。
- Qdrant 存储知识 chunk 向量和 semantic memory 向量。

## 实时工具与 Agent 工具

实时工具路由实现在 `backend/app/live_tools.py` 中，并由 `choose_live_tool` 包装调用。

实时工具：

| 工具 | 提供方 | 用途 |
|---|---|---|
| `weather_lookup` | Open-Meteo forecast API | 天气、气温、降雨、风速、出行/活动适宜性判断。 |
| `public_web_search` | DuckDuckGo Instant Answer API | 公开事实、官网/公开资料、最新/通用网络信息查询。 |
| `local_project_rag` | 内部路由 | 不调用 live tools，继续走项目 RAG。 |

Agent 工具注册表：

| 工具 | 是否只读 | 风险 | 用途 |
|---|---|---|---|
| `local_rag_search` | 是 | 低 | 检索本地项目/全局知识资产。 |
| `weather_lookup` | 是 | 低 | 查询实时天气。 |
| `public_web_search` | 是 | 中 | 查询公开网络摘要。 |
| `memory_read` | 是 | 低 | 读取项目分层记忆。 |
| `memory_write` | 否 | 中 | 用户明确要求时写入短期 working memory。 |
| `todo_create` | 否 | 中 | 创建项目 TODO。 |
| `todo_list` | 是 | 低 | 列出项目 TODO。 |
| `asset_list` | 是 | 低 | 列出知识资产。 |
| `calculator` | 是 | 低 | 计算简单算术表达式。 |

## 重要源码文件

| 路径 | 作用 |
|---|---|
| `backend/app/config.py` | 基于环境变量的运行时配置。 |
| `.env.example` | 本地配置模板。 |
| `backend/app/main.py` | FastAPI 路由和流式接口。 |
| `backend/app/services.py` | 默认 Plan-ReAct MCP 编排、旧本地 RAG/Plan-and-Solve 兼容路径、Query Rewrite、答案综合、Agent loop。 |
| `backend/app/mcp_client.py` | 手写 MCP JSON-RPC client、stdio transport、Streamable HTTP transport 骨架、GitHub MCP server 配置。 |
| `backend/app/vector_store.py` | 分词、BM25、embedding、Qdrant chunk search、reranking。 |
| `backend/app/semantic_store.py` | Semantic memory 向量索引和检索。 |
| `backend/app/memory_manager.py` | Working、episodic、semantic memory 的沉淀和召回。 |
| `backend/app/live_tools.py` | 天气和公开网络工具路由/执行。 |
| `backend/app/asset_ingest.py` | 文本、Markdown 和 PDF 解析。 |
| `backend/app/models.py` | Pydantic 请求/响应 schema。 |
| `specs/workflows/research-copilot.yaml` | 研究管线的 workflow-level contract。 |
