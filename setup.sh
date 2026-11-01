#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT}/scripts/env.sh"

PYTHON="${PYTHON:-python3}"
RECREATE=false
SKIP_IMAGES=false
SKIP_GPU_CHECK=false

usage() {
    cat <<'EOF'
Usage: ./setup.sh [options]

Prepare the complete workshop environment on a Linux GPU server.

Options:
  --recreate        Delete and rebuild .venv.
  --skip-images     Install only the host Python environment.
  --skip-gpu-check  Do not run the containerized CUDA smoke test.
  -h, --help        Show this help.
EOF
}

while (($#)); do
    case "$1" in
        --recreate) RECREATE=true ;;
        --skip-images) SKIP_IMAGES=true ;;
        --skip-gpu-check) SKIP_GPU_CHECK=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

cd "${ROOT}"

command -v "${PYTHON}" >/dev/null || {
    echo "ERROR: ${PYTHON} is required." >&2
    exit 1
}

"${PYTHON}" - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit("ERROR: Python 3.11 or newer is required.")
print(f"Host Python: {sys.version.split()[0]}")
PY

if "${RECREATE}" && [[ -d .venv ]]; then
    echo "Removing existing .venv ..."
    rm -rf .venv
fi

if [[ ! -x .venv/bin/python ]]; then
    echo "Creating .venv ..."
    "${PYTHON}" -m venv .venv
fi

echo "Installing pinned host dependencies ..."
.venv/bin/python -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
.venv/bin/python -m pip install --disable-pip-version-check --requirement requirements.txt

if ! "${SKIP_IMAGES}"; then
    command -v docker >/dev/null || {
        echo "ERROR: Docker is required. Install Docker Engine and rerun setup." >&2
        exit 1
    }
    docker info >/dev/null || {
        echo "ERROR: Docker is unavailable to this user." >&2
        exit 1
    }

    if [[ -n "${NGC_API_KEY:-}" ]]; then
        echo "Authenticating to nvcr.io with NGC_API_KEY ..."
        printf '%s' "${NGC_API_KEY}" | docker login nvcr.io \
            --username '$oauthtoken' \
            --password-stdin >/dev/null
    fi

    echo "Pulling NeMo AutoModel image: ${AUTOMODEL_IMAGE}"
    if ! docker pull "${AUTOMODEL_IMAGE}"; then
        if docker image inspect "${AUTOMODEL_IMAGE}" >/dev/null 2>&1; then
            echo "WARNING: pull failed; using the existing local AutoModel image."
        else
        cat >&2 <<EOF
ERROR: Could not pull ${AUTOMODEL_IMAGE}.
Log in first with:
  docker login nvcr.io -u '\$oauthtoken'
Use an NGC API key as the password.
EOF
            exit 1
        fi
    fi

    echo "Pulling vLLM image: ${VLLM_IMAGE}"
    docker pull "${VLLM_IMAGE}"

    echo "Building pinned Garak image: ${GARAK_IMAGE}"
    export BUILDX_CONFIG="${BUILDX_CONFIG:-/tmp/finetune-demo-buildx-${UID}}"
    mkdir -p "${BUILDX_CONFIG}"
    docker build \
        --pull \
        --build-arg "GARAK_REVISION=${GARAK_REVISION}" \
        --file docker/garak/Dockerfile \
        --tag "${GARAK_IMAGE}" \
        .

    if ! "${SKIP_GPU_CHECK}"; then
        command -v nvidia-smi >/dev/null || {
            echo "ERROR: nvidia-smi is missing; install an NVIDIA driver." >&2
            exit 1
        }
        echo "Checking CUDA access from the training container ..."
        docker run --rm --gpus device=0 "${AUTOMODEL_IMAGE}" python -c \
            "import torch, nemo_automodel; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
    fi
fi

echo "Validating the clean-server installation ..."
VERIFY_ARGS=()
if "${SKIP_IMAGES}"; then
    VERIFY_ARGS+=(--skip-images)
elif "${SKIP_GPU_CHECK}"; then
    VERIFY_ARGS+=(--skip-gpu)
fi
.venv/bin/python scripts/verify_setup.py "${VERIFY_ARGS[@]}"

cat <<EOF

Setup complete.

Before running hosted-inference cells:
  export NVIDIA_API_KEY=nvapi-...

Start JupyterLab:
  source .venv/bin/activate
  jupyter lab ndd_nemotron_garak.ipynb

Images:
  training/merge: ${AUTOMODEL_IMAGE}
  serving:        ${VLLM_IMAGE}
  evaluation:     ${GARAK_IMAGE}
EOF
