#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
DEFAULT_API_PORT=8000

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: Python venv not found at $PYTHON_BIN"
  echo "Create it with: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

usage() {
  echo "Guard Product Launcher"
  echo "Usage:"
  echo "  ./run_guard.sh ui [--server.port 8510]"
  echo "  ./run_guard.sh analyze --input /path/to/video_or_folder [additional guard_analyze flags]"
  echo "  ./run_guard.sh analyze-v2 --input /path/to/video_or_folder [additional guard_v2_analyze flags]"
  echo "  ./run_guard.sh redis"
  echo "  ./run_guard.sh serve [uvicorn args]"
  echo "  ./run_guard.sh worker [celery args]"
  echo "  ./run_guard.sh train [training flags]"
  echo "  ./run_guard.sh doctor [--json] [--strict] [--strict-recommended]"
}

is_redis_ready() {
  if ! command -v redis-cli >/dev/null 2>&1; then
    return 1
  fi
  redis-cli ping >/dev/null 2>&1
}

is_guard_api_running() {
  pgrep -f "uvicorn service.api:app --host 0.0.0.0 --port ${DEFAULT_API_PORT}" >/dev/null 2>&1
}

is_guard_worker_running() {
  pgrep -f "celery -A service.celery_app worker" >/dev/null 2>&1
}

is_port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "sport = :${port}" | tail -n +2 | grep -q .
    return
  fi
  if command -v netstat >/dev/null 2>&1; then
    netstat -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${port}$"
    return
  fi
  return 1
}

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
  usage
  exit 1
fi
shift || true

cd "$ROOT_DIR"

case "$MODE" in
  ui)
    exec "$PYTHON_BIN" -m streamlit run "$ROOT_DIR/training/guard_app.py" "$@"
    ;;
  analyze)
    exec "$PYTHON_BIN" -m training.guard_analyze "$@"
    ;;
  analyze-v2)
    exec "$PYTHON_BIN" -m training.guard_v2_analyze "$@"
    ;;
  redis)
    if is_redis_ready; then
      echo "Redis is already running."
      exit 0
    fi
    exec redis-server --save "" --appendonly no
    ;;
  serve)
    if is_guard_api_running; then
      echo "Guard API already running on port ${DEFAULT_API_PORT}."
      exit 0
    fi
    if is_port_in_use "$DEFAULT_API_PORT"; then
      echo "Error: port ${DEFAULT_API_PORT} is already in use by another process."
      exit 1
    fi
    exec "$PYTHON_BIN" -m uvicorn service.api:app --host 0.0.0.0 --port "$DEFAULT_API_PORT" "$@"
    ;;
  worker)
    if is_guard_worker_running; then
      echo "Guard worker already running."
      exit 0
    fi
    if ! is_redis_ready; then
      echo "Error: Redis is not reachable. Start it with './run_guard.sh redis'."
      exit 1
    fi
    exec "$PYTHON_BIN" -m celery -A service.celery_app worker -Q text,image,video,audio -l info "$@"
    ;;
  train)
    exec "$PYTHON_BIN" -m training.train "$@"
    ;;
  doctor)
    exec "$PYTHON_BIN" -m training.guard_doctor "$@"
    ;;
  *)
    echo "Error: unknown mode '$MODE'"
    usage
    exit 1
    ;;
esac
