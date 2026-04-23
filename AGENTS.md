# Repository Guidelines

## Project Structure & Module Organization

This repository is centered on `code/research_copilot/`.

- `backend/app/`: FastAPI runtime code.
  - `main.py`: API routes and app bootstrap.
  - `models.py`: Pydantic request/response models.
  - `services.py`: orchestration and business logic.
  - `config.py`: environment-backed settings.
- `backend/Dockerfile`, `backend/pyproject.toml`: backend packaging and container build.
- `docs/`: architecture and roadmap notes.
- `specs/workflows/`: workflow contracts such as `research-copilot.yaml`.
- `.env.example`: local configuration template.

Keep new runtime code inside `backend/app/` unless you are adding tests or documentation.

## Build, Test, and Development Commands

- `python3 -m pip install --user -e ./backend`: install the backend in editable mode.
- `python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8001`: run the API locally from `backend/`.
- `docker compose up -d --build runtime-api`: build and start the backend container.
- `docker compose up -d`: start the full local stack (`runtime-api`, MySQL, Redis, MinIO, Qdrant).
- `python3 -m compileall backend/app`: quick syntax check before committing.
- `curl http://127.0.0.1:8001/healthz`: verify the runtime is up.

## Coding Style & Naming Conventions

Use 4-space indentation and standard Python style. Prefer:

- `snake_case` for modules, functions, and variables
- `PascalCase` for Pydantic models
- explicit type hints on public functions
- small route handlers that delegate to `services.py`
- the simplest implementation that satisfies the requirement
- minimal branching; avoid unnecessary `if/elif/else` chains when a simpler structure will do
- prefer open-source modules when they solve the problem cleanly

Keep API schemas in `models.py` and avoid mixing HTTP concerns into service helpers.

## Testing Guidelines

There is no formal test suite yet. For new backend work, add `pytest` tests under `backend/tests/` using names like `test_healthz.py` or `test_services.py`.

At minimum, validate:

- model parsing and response shape
- pure service behavior
- endpoint health and happy paths

Until a suite exists, run `python3 -m compileall backend/app` and smoke-test changed endpoints with `curl`.

## Commit & Pull Request Guidelines

This directory does not currently include Git history, so use short imperative commit messages such as `Add memory persistence stub` or `Refine runtime health response`.

For pull requests, include:

- a brief summary of behavior changes
- affected paths, APIs, or env vars
- sample requests/responses for endpoint changes
- screenshots only when UI files are added later

## Security & Configuration Tips

Copy `.env.example` to `.env` for local development. Do not commit secrets, local credentials, or generated `.env` changes unless intentionally updating defaults.
