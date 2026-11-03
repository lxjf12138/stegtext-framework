#!/usr/bin/env bash
set -euo pipefail
_stegatext_sync_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$_stegatext_sync_root"
source environment/activate.sh
uv sync --locked
python environment/link_project.py
uv pip check
