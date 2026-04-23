#!/usr/bin/env bash
# Разработка без лишних перезапусков: не следим за .venv (pip) — иначе обрывается скачивание модели HF.
set -euo pipefail
cd "$(dirname "$0")"
exec uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 \
  --reload-exclude '.venv' \
  --reload-exclude '**/.venv/**' \
  --reload-exclude '**/__pycache__/**' \
  "$@"
