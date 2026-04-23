# Source Mapping

This file maps each capability to the source that should influence it most.

| Capability | Primary Source | How To Use It | Notes |
|---|---|---|---|
| Visual workflow editor | PaiPai workflow agent | Reuse directly if the purchased code is clean enough | Outer shell |
| DAG execution | PaiPai workflow agent | Reuse concepts and execution shell | Keep Python runtime out of the Java process |
| Tool abstraction | hello-agents | Borrow the simple tool-centric mental model | Good for teaching and low-friction extension |
| Deep research pattern | hello-agents | Reuse TODO planning, summarization, reporting flow | Good starter workflow |
| Session runtime | DeepTutor | Borrow architecture and selected code patterns | Best fit for inner runtime |
| Bounded context | DeepTutor | Reuse the context-builder strategy | Important for long tasks |
| Knowledge base lifecycle | DeepTutor | Borrow indexing and KB management ideas | Replace storage shape as needed |
| Memory consolidation | DeepTutor | Reuse the extraction pattern | Persist to structured storage |
| Citation assembly | DeepTutor + custom | Keep evidence linked to answers | Make it first-class, not markdown-only |
| Project memory model | custom | Build specifically for your platform | None of the three sources is enough alone |
| Hybrid RAG | custom | Dense + sparse + metadata filters | Use Qdrant + MySQL + MinIO |
| Paper/code ingestion | custom + DeepTutor reference | Implement as dedicated workers | This is your platform differentiator |
| Workspace UI | custom | Build around project, evidence, memory, and runs | Not a generic chat page |

## Borrow vs Rewrite

### Borrow mostly as-is

- visual flow editor shell from the paid workflow project
- SSE and streaming interaction patterns
- selected runtime orchestration patterns from DeepTutor

### Borrow the idea, rewrite the implementation

- context packaging
- memory extraction and consolidation
- evidence ranking
- answer synthesis contracts
- project-oriented data model

### Build from scratch

- project memory schema
- citation persistence
- paper + code joint ingestion pipeline
- project workspace UX
- workflow node contracts between Java shell and Python runtime
