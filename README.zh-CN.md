# Research Copilot

[English](README.md) | 简体中文

Research Copilot 是一个面向项目范围知识工作的本地优先研究工作台。它把项目管理、文本资产导入、TODO 执行、带引用回答、运行历史和分层记忆整合到一个 FastAPI 服务和浏览器工作区里。

## 项目概览

当前 MVP 围绕固定的 `plan-and-solve` 流程展开：

1. 构建项目上下文
2. 规划研究任务
3. 从项目资产中检索证据
4. 生成带引用的回答
5. 持久化分层记忆，供后续追问复用

后端同时提供 API 和前端工作区，因此本地部署比较直接。

## 简历摘要

简介：传统科研工作面临资料碎片化与过程不可复盘问题。本项目构建面向私有项目知识库的 AI 研究工作台，集成资料导入、Hybrid RAG 检索、引用回答、运行轨迹和分层记忆，为用户提供可追溯、可复用的个性化研究辅助。

技术栈：FastAPI、Pydantic、SQLAlchemy、MySQL、Redis、MinIO、Qdrant、BGE-M3、BGE Reranker、BM25、jieba、DeepSeek API、Docker Compose、pytest。

- 实现基于 Qdrant + BGE-M3 + BM25/jieba + BGE Reranker 的 Hybrid RAG 检索链路，融合语义检索、关键词检索、RRF 排序和 rerank 精排，为回答生成提供可追溯 citation 证据。
- 引入 Query Rewrite 检索增强层，支持 standalone query、Step-back 抽象查询、HyDE 假设性文档和领域词扩展；在语义通信 benchmark 中将 hit@1 从 0.80 提升至 0.93，MRR@5 从 0.869 提升至 0.967。
- 设计 working / episodic / semantic 三层记忆体系：working 保存会话内近期问答，episodic 记录项目级研究事件，semantic 抽取事实、决策、开放问题和偏好，并结合向量检索支持长期语义召回。
- 实现 Plan-and-Solve 主控的 Agent 执行链路，在规划阶段拆解研究目标，在执行阶段引入 ReAct 式工具调用完成检索、观察、校验和答案综合，实现复杂研究问答过程可追踪、结果可复盘。

扩展规划：

- 计划基于 LazyRegistry 自动发现 skill manifest 并按需懒加载 handler；结合 Progressive Disclosure 渐进式暴露机制，先向模型提供工具摘要完成路由，再注入候选工具 schema 生成结构化参数，降低 token 开销和误调用风险。

## 主页面预览

![Research Copilot 主页面与 Agent 执行轨迹](docs/images/main-page-agent-trace.png)

主页面把对话、项目资产和运行历史放在同一个工作区里。左侧栏用于新建对话、查看资产和切换最近会话，中间区域展示当前问答。回答区域会显示逐步执行轨迹，方便查看系统如何检索证据、调用工具、整理观察结果并最终生成回答。

## 当前能力

- 项目 CRUD 和汇总计数
- 文本、Markdown 和 PDF 资产的创建、编辑与上传
- 支持浏览器 MD5、Redis bitmap 进度和 MinIO 分片合并的断点续传上传
- TODO 的创建、编辑、删除和直接执行
- 项目级混合检索，支持 dense + BM25 融合
- 本地 reranker 对证据做最终排序
- 带引用回答生成
- 实时/公开信息工具路由，天气类问题优先调用外部天气工具
- Plan-Act-Observe Agent 模式，可在 RAG、联网、天气、记忆、资产、TODO 和计算器之间进行结构化工具调用
- 面向研究场景的 skill/tool registry，包含只读/副作用风险元数据
- Agent 执行轨迹可视化，用于回放工具选择、观察结果和答案综合过程
- working、episodic、semantic 三层记忆
- 运行历史和详细执行结果查看
- 由同一个 FastAPI 服务直接提供浏览器工作区

## 运行行为

- 默认执行模式：`plan_and_solve`
- 默认 LLM 目标：`deepseek/deepseek-chat`
- 默认向量库：`qdrant`
- 默认向量模型：`BAAI/bge-m3`
- 默认重排模型：`BAAI/bge-reranker-base`
- 如果设置了 `LLM_API_KEY`，运行时会调用 DeepSeek Chat API 生成答案。
- 如果 `LLM_API_KEY` 为空，系统会退回到确定性的带引用摘要，保证整条流程仍可执行。

## 架构组成

主要组件如下：

- `FastAPI`：API 路由、生命周期启动和静态工作区页面
- `SQLAlchemy + MySQL`：项目、资产、TODO、运行记录和记忆持久化
- `Qdrant`：项目分块和语义记忆的向量存储
- `BGE-M3`：本地 dense embedding
- `BM25 + jieba`：词法检索和中文分词
- `BGE reranker`：本地证据重排
- `Redis`、`MinIO`、可选 `Ollama`：用于断点续传进度、对象分片和后续工作流扩展

## 仓库结构

```text
research_copilot/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── models.py
│   │   ├── services.py
│   │   ├── vector_store.py
│   │   ├── memory_manager.py
│   │   └── static/
│   ├── tests/
│   ├── Dockerfile
│   └── pyproject.toml
├── docs/
├── specs/
├── .env.example
├── docker-compose.yml
└── README.md
```

## 快速开始

### 环境要求

- Docker 和 Docker Compose
- 至少数 GB 的可用磁盘空间，用于本地 embedding 和 reranker 模型
- 如果你希望使用真实 LLM 生成而不是 fallback 摘要，需要准备 DeepSeek API Key

### 启动完整本地栈

```bash
cd /home/wsl/code/research_copilot
cp .env.example .env
docker compose up -d --build
```

启动后可访问：

- 工作区：`http://127.0.0.1:8001/`
- API 文档：`http://127.0.0.1:8001/docs`
- 健康检查：`http://127.0.0.1:8001/healthz`

### 可选 Ollama Profile

```bash
docker compose --profile llm up -d
```

## 首次运行说明

- `docker-compose.yml` 会把本地 `./models` 挂载到容器内的 `/models`。
- 如果 `./models` 下还没有所需的 BAAI 模型快照，运行时会在首次使用时从 Hugging Face 自动下载。
- 这些模型文件体积较大，且已经被排除在 Git 之外。
- 当前默认单个文件上传大小限制是 `100 MB`。

## 配置说明

把 `.env.example` 复制为 `.env` 后，只修改你需要的项即可。

关键环境变量：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `ENVIRONMENT` | `local` | 运行环境标识 |
| `LLM_PROVIDER` | `deepseek` | LLM 提供方目标 |
| `LLM_MODEL` | `deepseek-chat` | 聊天模型名 |
| `LLM_API_BASE` | `https://api.deepseek.com` | DeepSeek API 地址 |
| `LLM_API_KEY` | 空 | 设置后启用真实 LLM 回答生成 |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 本地向量模型 |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | 本地重排模型 |
| `VECTOR_STORE_PROVIDER` | `qdrant` | 向量存储后端 |
| `QDRANT_COLLECTION` | `knowledge_chunks` | 知识分块集合名 |
| `SEMANTIC_MEMORY_COLLECTION` | `semantic_memory_facts` | 语义记忆集合名 |
| `EXECUTION_MODE` | `plan_and_solve` | 执行模式 |
| `QUERY_REWRITE_ENABLED` | `true` | 是否启用混合 Query Rewrite 检索扩展 |
| `QUERY_REWRITE_HYDE_ENABLED` | `true` | 是否生成 HyDE 假想答案文档查询 |
| `QUERY_REWRITE_STEP_BACK_ENABLED` | `true` | 是否生成 Step-back 抽象查询 |
| `QUERY_REWRITE_MAX_QUERIES` | `10` | 单轮本地 RAG 最多使用的重写/扩展查询数 |
| `UPLOAD_MAX_BYTES` | `104857600` | 文件上传大小上限，单位字节 |
| `RESUMABLE_UPLOAD_MAX_BYTES` | `1073741824` | 断点续传上传大小上限，单位字节 |
| `RESUMABLE_UPLOAD_CHUNK_SIZE` | `8388608` | 前端分片上传默认分片大小 |
| `MINIO_BUCKET_RAW` | `agent-raw` | 本地栈创建的原始对象桶 |
| `MINIO_BUCKET_ARTIFACTS` | `agent-artifacts` | 分片和合并产物对象桶 |
| `LIVE_TOOLS_ENABLED` | `true` | 是否启用实时/公开信息工具路由 |
| `LLM_TOOL_PLANNER_ENABLED` | `true` | 是否让 LLM 先决定实时工具，再回退规则 |
| `LLM_TOOL_PLANNER_TIMEOUT_SECONDS` | `20.0` | 单次 LLM 工具规划调用超时时间 |
| `AGENT_MAX_STEPS` | `5` | `/agent/run` 的 Plan-Act-Observe 最大步数 |
| `PUBLIC_WEB_SEARCH_ENABLED` | `true` | 是否允许显式联网/搜索类问题调用公开搜索工具 |
| `LIVE_TOOL_TIMEOUT_SECONDS` | `8.0` | 单次外部工具调用超时时间 |
| `DATABASE_URL` | 空 | 可选，用于覆盖默认 MySQL 连接串 |

## API 概览

Swagger UI 地址是 `/docs`。

- `GET /api/v1/skills`：查看已注册研究 skill 和可执行工具 schema。

主要接口：

| 接口 | 方法 | 作用 |
| --- | --- | --- |
| `/healthz` | `GET` | 服务状态和依赖概览 |
| `/api/v1/projects` | `GET`, `POST` | 查询或创建项目 |
| `/api/v1/projects/{project_id}` | `GET`, `PATCH`, `DELETE` | 管理单个项目 |
| `/api/v1/projects/{project_id}/sessions` | `GET`, `POST` | 查询或创建项目会话 |
| `/api/v1/projects/{project_id}/sessions/{session_id}/run` | `POST` | 执行一次研究任务 |
| `/api/v1/projects/{project_id}/sessions/{session_id}/run/stream` | `POST` | 以 NDJSON 事件流执行研究任务 |
| `/api/v1/projects/{project_id}/sessions/{session_id}/agent/run` | `POST` | 执行一次 Plan-Act-Observe Agent 任务 |
| `/api/v1/projects/{project_id}/sessions/{session_id}/runs` | `GET` | 查看会话运行历史 |
| `/api/v1/assets` | `GET`, `POST` | 查询或创建全局资产 |
| `/api/v1/assets/upload-file` | `POST` | 旧版单请求上传 `.txt`、`.md`、`.markdown`、`.pdf` |
| `/api/v1/assets/uploads/init` | `POST` | 按 MD5 创建或恢复分片上传 |
| `/api/v1/assets/uploads/{upload_id}/chunks` | `POST` | 上传单个分片，并写入 Redis bitmap |
| `/api/v1/assets/uploads/{upload_id}/complete` | `POST` | 合并 MinIO 分片、解析文件并创建资产 |
| `/api/v1/assets/{asset_id}` | `PATCH`, `DELETE` | 更新或删除单个资产 |
| `/api/v1/projects/{project_id}/todos` | `GET`, `POST` | 查询或创建 TODO |
| `/api/v1/todos/{todo_id}` | `PATCH`, `DELETE` | 更新或删除单个 TODO |
| `/api/v1/projects/{project_id}/memory` | `GET` | 查看分层记忆记录 |

### 示例：上传文本资产

```bash
curl -X POST http://127.0.0.1:8001/api/v1/assets/upload-file \
  -F "asset_type=note" \
  -F "title=system-notes.txt" \
  -F "file=@./system-notes.txt"
```

### 示例：执行研究任务

```bash
curl -X POST http://127.0.0.1:8001/api/v1/projects/<project_id>/sessions/<session_id>/run \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "请总结论文与代码仓之间的关系",
    "sequence_id": 1,
    "asset_ids": []
  }'
```

## 本地开发

安装后端包：

```bash
python3 -m pip install --user -e ./backend
```

直接从 `backend/` 启动 API：

```bash
cd backend
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

基础检查：

```bash
python3 -m compileall backend/app
cd backend && pytest -q
curl http://127.0.0.1:8001/healthz
```

## 当前范围与限制

- 当前 MVP 主要面向文本类和 PDF 研究资产。
- 浏览器工作区由后端直接提供，更适合本地环境使用。
- Redis 用于保存断点续传元信息和分片 bitmap；MinIO 用于保存上传分片和合并后的源文件。
- OCR、报告导出和更完整的工作流集成仍然在路线图中。

## 文档

- [系统架构](docs/architecture.md)
- [用户手册](docs/user-manual.md)
- [工作流契约](specs/workflows/research-copilot.yaml)

## License

仓库当前还没有附带 license 文件。
