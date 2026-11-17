#!/bin/sh
# Run Python inside this project: ./run.sh -m tinyfc_cjk.predict line.png
# Uses .venv (created by `uv sync --extra onnx`). To use another interpreter, put
#   TINYFC_PYTHON=/path/to/python
# in a .env.local file next to this script (git-ignored) or set it in the environment.
cd "$(dirname "$0")"
[ -f .env.local ] && . ./.env.local
PY="${TINYFC_PYTHON:-.venv/bin/python}"
PYTHONPATH="src:.pydeps${PYTHONPATH:+:$PYTHONPATH}" exec "$PY" "$@"
