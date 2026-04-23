# Research Copilot

Research Copilot is a local-first knowledge workspace built around a
`plan-and-solve` runtime. It supports project management, text assets, TODO
execution, cited answers, run history, long-term memory, and real vector
retrieval.

## Current MVP

This repository now includes a complete backend and a browser-based workspace:

- project CRUD
- text asset ingestion and editing
- TODO CRUD and execution
- plan-and-solve run pipeline
- text file upload from the browser workspace
- local `bge-m3` embeddings with Qdrant vector search
- hybrid retrieval with dense + BM25 fusion
- local BGE reranker for final evidence ordering
- cited answer generation through DeepSeek chat
- layered memory with working / episodic / semantic recall
- long-term memory persistence
- run history and dashboard views

Model configuration in the current MVP:

- LLM provider: `deepseek`
- Embedding provider: `local`
- Reranker provider: `local`
- Vector store: `qdrant`
- Memory model: `working + episodic + semantic`
- Execution mode: `plan_and_solve`

## Repo Layout

```text
research_copilot/
├── backend/
│   ├── app/
│   │   ├── db.py
│   │   ├── db_models.py
│   │   ├── main.py
│   │   ├── models.py
│   │   ├── services.py
│   │   └── static/
│   ├── tests/
│   ├── Dockerfile
│   └── pyproject.toml
├── docs/
├── specs/
├── .env.example
└── docker-compose.yml
```

## Run Locally

```bash
cd /home/wsl/code/research_copilot
cp .env.example .env
docker compose up -d --build
```

After startup:

- Workspace: `http://127.0.0.1:8001/`
- API docs: `http://127.0.0.1:8001/docs`
- Health check: `http://127.0.0.1:8001/healthz`

You can now upload a `.txt` or `.md` file from the workspace or call:

```bash
curl -X POST http://127.0.0.1:8001/api/v1/projects/<project_id>/assets/upload-text \
  -F "asset_type=note" \
  -F "title=system-notes.txt" \
  -F "file=@./system-notes.txt"
```

## Test

```bash
cd backend
pytest -q
python3 -m compileall app tests
```

## Documentation

- Architecture: [docs/architecture.md](docs/architecture.md)
- User manual: [docs/user-manual.md](docs/user-manual.md)
- Technical highlights: [docs/technical-highlights.md](docs/technical-highlights.md)
- Source mapping: [docs/source-mapping.md](docs/source-mapping.md)
