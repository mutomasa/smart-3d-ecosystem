#!/usr/bin/env bash
# Start FastAPI, Vite and Streamlit together. Ctrl+C stops all of them.
set -euo pipefail
cd "$(dirname "$0")/.."

trap 'kill 0' EXIT

uv run uvicorn backend.main:app --reload --port "${BACKEND_PORT:-8000}" &
(cd frontend && npm run dev -- --port 5173) &
uv run streamlit run dashboard/app.py --server.port 8501 --server.headless true &

wait
