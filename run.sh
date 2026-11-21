#!/bin/bash
# Bir buyruq bilan o'rnatish va ishga tushirish
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null; then uv venv --python 3.12 .venv; uv pip install --python .venv/bin/python -r requirements.txt
  else python3 -m venv .venv; .venv/bin/pip install -r requirements.txt; fi
fi
.venv/bin/python download_models.py
.venv/bin/python server.py
