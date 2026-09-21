#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT"

export PYTHONPATH="$ROOT:$ROOT/pipeline:$ROOT/pipeline/revision_modules:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

exec python3 -u "$ROOT/pipeline/59A_genogramma_768d_component_ablation.py"
