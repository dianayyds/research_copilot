#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed. Run scripts/install_server_environment.sh first." >&2
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "The Docker Compose plugin is not installed. Run scripts/install_server_environment.sh first." >&2
    exit 1
fi

if [[ ! -f .env ]]; then
    cp .env.example .env
    chmod 600 .env
    echo "Created .env from .env.example."
    echo "Set LLM_API_KEY and GITHUB_PERSONAL_ACCESS_TOKEN in .env for the default Plan-ReAct MCP mode."
fi

mkdir -p models
docker compose config --quiet
docker compose up -d --build
docker compose ps

echo "Research Copilot deployment started."
echo "Workspace: http://127.0.0.1:${BACKEND_PORT:-8001}/"
echo "Health:    http://127.0.0.1:${BACKEND_PORT:-8001}/healthz"
echo "The first embedding or reranking request may download several GB of model files into ./models."
