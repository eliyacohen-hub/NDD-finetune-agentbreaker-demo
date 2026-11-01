#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/env.sh"

REMOVE_IMAGES=false
if [[ "${1:-}" == "--images" ]]; then
    REMOVE_IMAGES=true
elif [[ -n "${1:-}" ]]; then
    echo "Usage: scripts/clean.sh [--images]" >&2
    exit 2
fi

cd "${ROOT}"

if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
    docker rm --force \
        finetune-demo-base \
        finetune-demo-attacker \
        >/dev/null 2>&1 || true
    docker volume rm finetune-demo-hf-cache >/dev/null 2>&1 || true
    if "${REMOVE_IMAGES}"; then
        docker image rm "${GARAK_IMAGE}" >/dev/null 2>&1 || true
    fi
fi

remove_path() {
    local path="$1"
    [[ -e "${path}" ]] || return 0
    rm -rf "${path}" 2>/dev/null || true
    [[ ! -e "${path}" ]] && return 0

    if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
        docker run --rm \
            -e "TARGET=${path}" \
            -v "${ROOT}:${ROOT}" \
            -w "${ROOT}" \
            --entrypoint python \
            python:3.12-slim-bookworm \
            -c "import os, shutil; shutil.rmtree(os.environ['TARGET'], ignore_errors=False)"
    else
        echo "ERROR: could not remove ${path}; root-owned container output may require Docker." >&2
        return 1
    fi
}

remove_path "${ROOT}/.venv"
remove_path "${ROOT}/artifacts"
remove_path "${ROOT}/output/artifacts"
remove_path "${ROOT}/runs"

rm -f training/*.run.yaml training/train.log training/merge.log
rm -f agents/data/outbox.json

python3 - <<'PY'
from pathlib import Path
import shutil

root = Path.cwd()
for cache in root.rglob("__pycache__"):
    if ".git" not in cache.parts:
        shutil.rmtree(cache, ignore_errors=True)
for checkpoint in root.rglob(".ipynb_checkpoints"):
    if ".git" not in checkpoint.parts:
        shutil.rmtree(checkpoint, ignore_errors=True)
for path in Path("/tmp").glob("finetune-demo-garak-*"):
    shutil.rmtree(path, ignore_errors=True)
PY

echo "Removed project environments, generated artifacts, caches, and demo containers."
