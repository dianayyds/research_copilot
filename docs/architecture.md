# System Architecture

## Overview

The current system uses a single FastAPI service to host both the frontend
workspace and the backend runtime. The runtime follows a `plan-and-solve`
pipeline and persists project state in MySQL.

```mermaid
flowchart LR
    UI[Browser Workspace] --> API[FastAPI Runtime API]
    API --> PS[Plan-and-Solve Service]
    PS --> RET[Local Hybrid Retrieval]
    PS --> MEM[Memory Consolidation]
    API --> DB[(MySQL)]
    API --> REDIS[(Redis)]
    API --> MINIO[(MinIO)]
    API --> QDRANT[(Qdrant)]
    CFG[DeepSeek Config<br/>Local Embedding Config] --> API
```

## Backend Layers

- `main.py`: HTTP routes, lifespan startup, static frontend mount
- `services.py`: project CRUD, TODO workflow, run orchestration, retrieval, memory
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
    R->>R: Plan tasks
    R->>R: Retrieve evidence
    R->>R: Synthesize answer
    R->>R: Update memory
    R->>D: Persist run + memory
    R-->>W: Return cited result
```

## Notes

- DeepSeek and local embedding providers are configuration targets, not yet active inference backends.
- Retrieval currently uses open-source local components and project asset text.
- The current frontend is backend-served to keep the deployment simple and reliable.
