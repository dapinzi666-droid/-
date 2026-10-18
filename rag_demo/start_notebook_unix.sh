#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
    echo '请先运行 bash setup_unix.sh。'
    exit 1
fi
export JUPYTER_CONFIG_DIR="$PWD/.cache/jupyter/config"
export JUPYTER_RUNTIME_DIR="$PWD/.cache/jupyter/runtime"
export JUPYTER_DATA_DIR="$PWD/.cache/jupyter/data"
export IPYTHONDIR="$PWD/.cache/ipython"
mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_RUNTIME_DIR" "$JUPYTER_DATA_DIR" "$IPYTHONDIR"
exec .venv/bin/python -m jupyterlab task2_rag_demo.ipynb --ip=127.0.0.1
