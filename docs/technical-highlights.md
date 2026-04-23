# Technical Highlights

## 1. Plan-and-Solve First

The runtime is built around a fixed `plan-and-solve` pipeline instead of a
free-form ReAct loop. This keeps execution stable, easy to review, and easier
to connect with workflow orchestration later.

## 2. Project-Scoped Knowledge

All retrieval is scoped to a project. Assets, TODOs, runs, and memory belong to
the same workspace, so answers stay tied to concrete project state.

## 3. Open-Source Retrieval Path

The MVP uses open-source local components:

- `jieba` for Chinese-aware tokenization
- `rank-bm25` for lexical retrieval
- SQLAlchemy for persistence

This matches the rule of preferring open-source modules where they solve the
problem cleanly.

## 4. Replaceable Model Layer

The system already exposes provider configuration for:

- DeepSeek as the target LLM provider
- local embedding models as the target embedding provider

The current deterministic runtime is a clean placeholder that can be replaced by
real API and embedding calls without changing the UI or data model.

## 5. Full Closed Loop

The MVP already closes the core loop:

- create project
- add and refine assets
- create TODO
- execute research
- generate citations
- persist memory
- inspect run history
