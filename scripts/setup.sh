#!/usr/bin/env bash
# Local dev bootstrap: brings up the infra containers (Postgres, Qdrant,
# Elasticsearch, Redis, Neo4j) and installs the Python environment, so
# `scripts/seed_data.py` and `langgraph dev` have something to run against.
#
#   ./scripts/setup.sh
#
# Idempotent — safe to re-run; `docker compose up -d` and `pip install -e .`
# are both no-ops on an already-current environment.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f .env ]; then
  echo "No .env found — copying .env.example. Fill in your API keys before running the app."
  cp .env.example .env
fi

echo "Starting infra containers (Postgres, Qdrant, Elasticsearch, Redis, Neo4j)..."
docker compose -f infra/docker/docker-compose.yml up -d

if [ ! -d .venv ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

echo
echo "Setup complete. Next steps:"
echo "  1. Edit .env with your GROQ_API_KEY (and optionally GITHUB_TOKEN, OPENAI_API_KEY)."
echo "  2. python -m scripts.seed_data   # load a small sample corpus"
echo "  3. langgraph dev                 # start the agent graphs"
echo "  4. uvicorn backend.app.api.main:app --reload --port 8000   # start the REST API"
