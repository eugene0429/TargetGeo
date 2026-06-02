#!/usr/bin/env bash
# Launch the interactive viewer in the repo-local venv.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$REPO_DIR/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "venv missing - run ./setup_env.sh first" >&2
  exit 1
fi
exec "$PY" -m targetgeo.viewer "$@"
