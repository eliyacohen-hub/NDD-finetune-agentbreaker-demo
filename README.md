# NeMo Data Designer → fine-tune Nemotron Nano → evaluate with Garak

One notebook (`ndd_nemotron_garak.ipynb`) that runs the full model-customization loop:

1. **Generate** synthetic red-team training data with **NeMo Data Designer**.
2. **Filter** the attacks that worked and **split** train/eval with no leakage.
3. **LoRA fine-tune** a **Nemotron Nano** with **NeMo AutoModel**.
4. **Evaluate** base vs. fine-tuned with **Garak's `agent_breaker` probe** against two agents we build.

> Authorized, defensive security demo (like NVIDIA Agent Breaker / Garak). The bundled agents are
> intentionally vulnerable practice targets with obviously-fake planted data.

## What you need

| | Sections 0–2 (data generation) | Sections 3–4 (fine-tune + evaluate) |
|---|---|---|
| **Machine** | any (Mac/Linux/Windows), CPU only | **Linux + one ~80 GB NVIDIA GPU** (H100/A100 80GB) |
| **Also** | — | Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), and an [NGC](https://ngc.nvidia.com) API key |

- **`NVIDIA_API_KEY`** from [build.nvidia.com](https://build.nvidia.com) — required for every section (data generation + the eval judge).
- macOS has no NVIDIA GPU, so `docker run --gpus` won't work there — run sections 3–4 on a Linux GPU box.

## How to run

### Sections 0–2 — any machine

```bash
cd ndd-nemotron-garak-demo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export NVIDIA_API_KEY=nvapi-...
jupyter lab ndd_nemotron_garak.ipynb      # run cells through section 2
```

Produces the training/validation JSONL in `data/finetune/`.

### Sections 3–4 — Linux GPU box

**1. Log in to NGC** so Docker can pull the image (API key: [ngc.nvidia.com](https://ngc.nvidia.com) → **Setup → Generate API Key**):

```bash
docker login nvcr.io -u '$oauthtoken'     # password = your NGC API key
```

> A **TLS handshake timeout** here means the machine can't reach `nvcr.io` — check VPN/proxy
> (`curl -I https://nvcr.io/v2/` should return quickly).

**2. Start JupyterLab in the NeMo AutoModel container** (it ships the GPU training stack):

```bash
export NVIDIA_API_KEY=nvapi-...
docker run --gpus 1 --rm -it -p 8888:8888 \
    -e NVIDIA_API_KEY \
    -v "$PWD":/workspace -w /workspace \
    nvcr.io/nvidia/nemo-automodel:25.11.00 \
    bash -lc "pip install -r requirements.txt vllm && jupyter lab --ip 0.0.0.0 --allow-root --no-browser"
```

Open the printed `http://127.0.0.1:8888/...?token=...` URL and run the notebook top to bottom.

## Configuration (top of the notebook)

| Variable | Default | Meaning |
|---|---|---|
| `GEN_MODEL` | `nvidia/llama-3.3-nemotron-super-49b-v1` | the one build.nvidia.com model that generates data, role-plays victims, and judges |
| `BASE_MODEL_ID` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | the student model we fine-tune |
| `N_AGENTS` | 15 | synthetic agents to generate |
| `TOOLS_MIN` / `TOOLS_MAX` | 3 / 5 | tools per agent |
| `COMPARE_MODEL` | `None` | optional third red-team model for the comparison |

## What's in the folder

```
ndd_nemotron_garak.ipynb   the walkthrough
training/                  automodel_lora.yaml, finetune.py, merge_lora.py, chat template
agents/                    two LangGraph victim agents (Atlas/research, Pulse/devops) + mock data + server
garak_eval/                run_scan.py (drive Garak), compare.py (ASR table + chart)
```

The evaluation plugs the base and fine-tuned models into Garak's probe as the **red-team model**,
attacks each agent on `:8000`, and reports **Attack Success Rate** (landed attacks / attempts) —
a stronger attacker scores higher.
