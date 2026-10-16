#!/usr/bin/env bash
# Start FastAPI, Vite and Streamlit together. Ctrl+C stops all of them.
#   BACKEND_PORT=8010 ./scripts/dev.sh   # when 8000 is already in use
set -euo pipefail
cd "$(dirname "$0")/.."

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8501}"
export BACKEND_URL="${BACKEND_URL:-http://localhost:${BACKEND_PORT}}"
export FRONTEND_PORT
# The dashboard embeds the Vite dev server and talks to the API directly.
export SCENE_URL="${SCENE_URL:-http://localhost:${FRONTEND_PORT}/}"
export API_URL="${API_URL:-${BACKEND_URL}}"

if (exec 3<>"/dev/tcp/127.0.0.1/${BACKEND_PORT}") 2>/dev/null; then
  echo "port ${BACKEND_PORT} is already in use; set BACKEND_PORT to a free port" >&2
  exit 1
fi

trap 'kill 0' EXIT
echo "backend ${BACKEND_URL} | scene ${SCENE_URL} | dashboard http://localhost:${DASHBOARD_PORT}"

uv run uvicorn backend.main:app --reload --port "${BACKEND_PORT}" &
(cd frontend && npm run dev -- --port "${FRONTEND_PORT}" --strictPort) &
if [ -f dashboard/app.py ]; then
  uv run streamlit run dashboard/app.py --server.port "${DASHBOARD_PORT}" --server.headless true &
fi

wait
