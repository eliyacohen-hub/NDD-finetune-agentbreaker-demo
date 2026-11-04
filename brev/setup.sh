#!/usr/bin/env bash
# Brev launchable setup: runs once when the instance is deployed.
#
# Optional environment variables (set them in the launchable if available):
#   NGC_API_KEY     pulls the NeMo AutoModel image from nvcr.io (needed for training)
#   NVIDIA_API_KEY  hosted inference key; written to .env so the notebook needs no prompt
# Without NGC_API_KEY only the Python environment is prepared; finish later with
# `docker login nvcr.io -u '$oauthtoken' && ./setup.sh`.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
exec > >(tee -a "${HOME}/brev-setup.log") 2>&1
echo "=== Brev setup started $(date -u +%FT%TZ) ==="

# Use a uv-managed Python 3.11: setup.sh needs >= 3.11, and the system Python on
# Brev images often lacks ensurepip (python3-venv), which breaks `python -m venv`.
export PATH="${HOME}/.local/bin:${PATH}"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
export PYTHON="$(uv python find 3.11)"
[[ -x .venv/bin/pip ]] || rm -rf .venv   # discard a half-created venv from a failed run

# Docker may need the docker group on a fresh instance.
DOCKER_PREFIX=()
if command -v docker >/dev/null && ! docker info >/dev/null 2>&1; then
    sudo usermod -aG docker "${USER:-$(id -un)}" || true
    DOCKER_PREFIX=(sg docker -c)
fi

run_setup() {
    if ((${#DOCKER_PREFIX[@]})); then
        "${DOCKER_PREFIX[@]}" "PYTHON='${PYTHON}' NGC_API_KEY='${NGC_API_KEY:-}' ./setup.sh $*"
    else
        ./setup.sh "$@"
    fi
}

# 1. Python environment first, so the notebook works even if the image steps fail.
run_setup --skip-images

if [[ -n "${NVIDIA_API_KEY:-}" && ! -f .env ]]; then
    printf 'NVIDIA_API_KEY=%s\n' "${NVIDIA_API_KEY}" > .env
    chmod 600 .env
fi

# The notebooks request kernel "python3", which Jupyter resolves to Brev's own venv.
# A user-level kernel with the same name takes precedence and points at our .venv.
.venv/bin/python -m ipykernel install --user --name python3 --display-name ".venv (finetune-demo)"

# 2. Container images (training, serving, Garak). Needs registry access via NGC_API_KEY.
IMAGES_OK=true
if [[ -z "${NGC_API_KEY:-}" ]]; then
    IMAGES_OK=false
    echo "WARNING: NGC_API_KEY not set; skipping container images."
elif ! run_setup; then
    IMAGES_OK=false
    echo "WARNING: container image setup failed (see above). The notebook still opens;"
    echo "  fix with: docker login nvcr.io -u '\$oauthtoken' && ./setup.sh"
fi

cat <<MSG

=== Brev setup complete ===
Open JupyterLab from the Brev console and run:
  ndd_nemotron_garak_workshop.ipynb   (kernel: .venv (finetune-demo), selected automatically)
Container images ready: ${IMAGES_OK}
Log: ${HOME}/brev-setup.log
MSG
