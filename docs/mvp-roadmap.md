# MVP Roadmap

## Phase 0: Foundation

- Stand up MySQL, Redis, MinIO, Qdrant, and optional Ollama.
- Create one project-scoped data model.
- Define runtime API contracts before writing node logic.

Exit criteria:

- all local infra boots cleanly
- workflow shell can reach runtime service
- file upload path is stable

## Phase 1: Knowledge Ingestion

- upload PDF, markdown, and source code folders
- parse metadata and store raw artifacts
- chunk and index into hybrid retrieval store
- expose project asset list

Exit criteria:

- one project can ingest a paper and a repo
- chunks are retrievable by project id

## Phase 2: Research QA

- decompose question into tasks
- retrieve evidence bundle
- synthesize answer with citations
- stream intermediate steps to UI

Exit criteria:

- user can ask a research question and get a cited answer
- citations map to stored evidence items

## Phase 3: Memory

- extract project state from completed runs
- persist conclusions, TODOs, and preferences
- load memory in future turns

Exit criteria:

- follow-up question can reuse prior project conclusions
- memory is editable and inspectable

## Phase 4: Workflowization

- expose runtime nodes to the visual workflow shell
- build a reusable "Research Copilot" workflow template
- add retry, audit, and trace views

Exit criteria:

- one full workflow can run from upload to answer to memory update
- operators can inspect every step and every citation

## Phase 5: Extensions

- OCR
- notebook export
- report generation
- MCP or plugin tool gateway
- multi-agent specialization

Do not start here.
Ship the project-memory research loop first.
