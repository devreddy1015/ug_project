#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${GUARD_VENV_DIR:-$ROOT_DIR/shield_product/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: Python venv not found at $VENV_DIR" >&2
  echo "Create it with:" >&2
  echo "  cd $ROOT_DIR/shield_product" >&2
  echo "  python -m venv .venv" >&2
  echo "  source .venv/bin/activate" >&2
  echo "  pip install -r requirements.txt" >&2
  exit 1
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" "$ROOT_DIR/run.py" "$@"
