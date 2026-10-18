#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
RAG_PY="${RAG_PYTHON:-python3}"
"$RAG_PY" -c 'import sys; assert sys.version_info[:2] in ((3,11),(3,12)) and sys.maxsize > 2**32, "Use 64-bit Python 3.11 or 3.12; set RAG_PYTHON=python3.11 if needed"'
if [[ ! -x .venv/bin/python ]]; then
    "$RAG_PY" -m venv .venv
fi
if [[ "$(uname -s)" == Darwin ]]; then
    if [[ "$(uname -m)" == x86_64 ]]; then
        .venv/bin/python -m pip install --disable-pip-version-check torch==2.2.2
    else
        .venv/bin/python -m pip install --disable-pip-version-check torch==2.6.0
    fi
else
    .venv/bin/python -m pip install --disable-pip-version-check torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
.venv/bin/python -m ipykernel install --prefix .venv --name rag-demo --display-name 'RAG Demo (project)'
.venv/bin/python run_demo.py --doctor
echo '依赖安装完成。下一步：./.venv/bin/python run_demo.py --prepare，然后 --smoke 和 --experiments。'
