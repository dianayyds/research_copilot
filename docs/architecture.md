# System Architecture

## Overview

The current system uses a single FastAPI service to host both the frontend
workspace and the backend runtime. The default runtime follows a
`plan_react_mcp` flow and persists project state in MySQL.

```mermaid
flowchart LR
    UI[Browser Workspace] --> API[FastAPI Runtime API]
    API --> PRM[Plan-ReAct MCP Runtime]
    PRM --> LLM[DeepSeek Planner and ReAct Decisions]
    PRM --> MCP[MCP JSON-RPC Client]
    MCP --> GH[GitHub MCP Server]
    PRM --> RET[Local Hybrid Retrieval / Legacy RAG]
    PRM --> MEM[Memory Consolidation]
    API --> DB[(MySQL)]
    API --> REDIS[(Redis)]
    API --> MINIO[(MinIO)]
    API --> QDRANT[(Qdrant)]
    CFG[DeepSeek Config<br/>Local Embedding Config] --> API
```

## Backend Layers

- `main.py`: HTTP routes, lifespan startup, static frontend mount
- `services.py`: project CRUD, TODO workflow, Plan-ReAct MCP orchestration, retrieval, memory
- `mcp_client.py`: hand-written JSON-RPC MCP client over stdio, plus a Streamable HTTP skeleton
- `db_models.py`: SQLAlchemy persistence model
- `models.py`: request and response schemas
- `static/`: browser workspace UI

## Runtime Flow

```mermaid
sequenceDiagram
    participant U as User
    participant W as Workspace
    participant R as Runtime API
    participant D as MySQL

    U->>W: Create project / asset / TODO
    W->>R: POST requests
    R->>D: Persist project data
    U->>W: Run TODO
    W->>R: POST /projects/{id}/run
    R->>R: Build context
    R->>R: Initialize GitHub MCP client
    R->>R: Ask DeepSeek for structured plan
    loop Per plan node
        R->>R: Ask DeepSeek for ReAct decision
        R->>R: Execute allowed MCP action
        R->>R: Record observation
    end
    R->>R: Synthesize cited answer
    R->>R: Update memory
    R->>D: Persist run + memory
    R-->>W: Return cited result
```

## Notes

- Default `/run` and `/run/stream` require DeepSeek credentials and GitHub MCP credentials.
- Local retrieval still uses open-source local components and project asset text, mainly for legacy RAG paths and local project tools.
- The current frontend is backend-served to keep the deployment simple and reliable.
- The local Docker Compose profile mounts `/var/run/docker.sock` so the runtime can launch the GitHub MCP server container; this is a high-privilege development setting.
