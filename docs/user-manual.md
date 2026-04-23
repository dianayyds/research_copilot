# User Manual

## 1. Start the System

```bash
cd /home/wsl/code/research_copilot
cp .env.example .env
docker compose up -d --build
```

Open `http://127.0.0.1:8001/`.

## 2. Create a Project

Use the left sidebar form to create a project. A project is the top-level
workspace for assets, TODOs, runs, and memory.

## 3. Add Knowledge Assets

In the `知识资产` panel, add text-based assets such as:

- paper summaries
- code repository notes
- lecture notes
- slide content

These texts are the retrieval source for cited answers.
You can also edit an existing asset before the next run.

## 4. Create and Manage TODOs

In the `TODO 列表` panel:

- create a TODO
- set priority and status
- edit existing TODOs
- delete TODOs
- run a TODO directly

## 5. Run Research

You have two execution paths:

1. Click `执行` on a TODO
2. Enter a custom question in the `执行面板`

The system will run:

- context packing
- task planning
- local hybrid retrieval
- answer synthesis
- memory consolidation

## 6. Review Results

Each run shows:

- final answer
- execution plan
- citations
- memory updates

Run history is available in the `运行记录` panel, and long-term memory is shown
in the `长期记忆` panel.
