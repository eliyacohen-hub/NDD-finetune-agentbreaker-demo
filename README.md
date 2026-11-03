# Learning demo: Garak probes → NDD → fine-tune → matched evaluation

Two notebooks cover the same model-customization loop:

- `ndd_nemotron_garak.ipynb` — full executable walkthrough.
- `ndd_nemotron_garak_workshop.ipynb` — live guided implementation lab.

1. Understand Garak's **attacker → victim → judge** probe loop.
2. **Generate** typed synthetic red-team data with **NeMo Data Designer**.
3. Inspect the quality funnel and split SFT records without agent leakage.
4. Configure and run one-GPU **LoRA fine-tuning** with **NeMo AutoModel**.
5. Compare base and fine-tuned attackers with a **matched Garak attempt budget**.

> Authorized, defensive security demo (like NVIDIA Agent Breaker / Garak). The bundled agents are
> intentionally vulnerable practice targets with obviously-fake planted data.

## Clean-server prerequisites

The full workshop path needs:

- Linux with one NVIDIA H100/A100 80 GB GPU and a working NVIDIA driver.
- Docker Engine plus the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Python 3.11 or newer.
- Network access to PyPI, GitHub, Hugging Face, `nvcr.io`, and build.nvidia.com.
- An [NGC API key](https://ngc.nvidia.com) for pulling the training image.
- `NVIDIA_API_KEY` from [build.nvidia.com](https://build.nvidia.com) for data generation, victim agents, and evaluation.

GPU drivers, Docker, and the NVIDIA Container Toolkit are host administration
prerequisites; the project setup deliberately does not modify those system components.

## One-command setup

Log in to NGC once, then run setup from the repository root:

```bash
docker login nvcr.io -u '$oauthtoken'   # use the NGC API key as password
./setup.sh
```

`setup.sh`:

1. Creates `.venv` and installs exact tested host dependencies.
2. Pulls the pinned NeMo AutoModel and vLLM images.
3. Builds Garak at the same pinned revision used by the training prompts.
4. Verifies Python imports, notebook schema, bundled data, container images, and GPU access.

It does not require `NVIDIA_API_KEY` and does not make paid inference calls.
To reproduce a clean machine locally:

```bash
./scripts/clean.sh --images
./setup.sh
```

## Run the guided exercise notebook

The exercise notebook is a direct copy of the live workflow with focused TODOs and
collapsible hints. Every participant needs the same API key, H100, Docker, model
access, and prepared environment as the main notebook:

```bash
source .venv/bin/activate
jupyter lab ndd_nemotron_garak_workshop.ipynb
```

Setup and infrastructure are provided. Learners implement profile-core generation,
profile scoring and selection, tool-analysis schemas and graph wiring, deterministic
branch sampling, the second adaptive attack loop, the SFT funnel, and important LoRA
parameter choices. Garak orchestration and report comparison are provided. Behavioral
checks stop at incomplete TODOs; there is no synthetic fallback path. The main
notebook is the complete reference solution.

## Run the notebook

```bash
cp .env.example .env
# Edit .env and replace NVIDIA_API_KEY, or export it in the shell.
source .venv/bin/activate
jupyter lab ndd_nemotron_garak.ipynb
```

On a remote GPU server, launch Jupyter with:

```bash
jupyter lab --no-browser --ip 0.0.0.0 --port 8888
```

Open the printed token URL from your laptop at
`http://<server-hostname-or-IP>:8888/lab?token=...`. If port 8888 is blocked,
create a tunnel with `ssh -N -L 8899:localhost:8888 <you>@<server>`, then open
`http://localhost:8899/lab?token=...`.

Run cells top to bottom. Every section starts with a validated configuration cell and
prints its resolved workload and output paths. CPU-side generation runs in `.venv`; training and merging
automatically use the pinned AutoModel container when the host lacks that GPU stack.
The evaluation section prints two pinned Docker commands for the base and fine-tuned
vLLM servers.

Each run is self-contained under
`output/artifacts/<UTC timestamp>_candidates-<N>_selected-<K>_tools-<min>-<max>_<model>/`.

For an unattended top-to-bottom run, including local evaluation servers, use:

```bash
export NDD_RUN_ID=my-audience-run
.venv/bin/python -u scripts/run_notebook.py --start-eval-servers
.venv/bin/python scripts/validate_run.py --run-id "$NDD_RUN_ID"
```

The validator checks every persisted DataFrame, tool and prompt coverage, all generated
attack branches, JSONL inputs, single-GPU training, loss, merged weights, all four
comparison reports, notebook outputs, and run isolation. It saves the result as
`output/artifacts/<run-id>/validation_report.json`.

### Workshop shortcut

To skip paid data generation, use the checked-in sample:

- `data/quick_finetune/agent_breaker_train.jsonl` — 512 records.
- `data/quick_finetune/agent_breaker_validation.jsonl` — 64 records.

The checked-in `training/automodel_lora.yaml` points to these files and writes to
`runs/attacker_lora/checkpoints/`. See `data/quick_finetune/README.md` for provenance.
The sample is for demonstration, not a production benchmark.

## Section-local configuration

The notebook deliberately avoids one giant configuration block:

- **Probe mechanics:** attacker, victim, judge, branch count, and baseline toggle.
- **NDD generation:** five profile candidates, two selected profiles, an NDD LLM
  quality score, uniform domain/tool/security samplers, generation model, simulated
  victim, branch limit, and seed.
- **SFT funnel:** accepted judge outcomes, analysis/exploit record types, validation
  fraction, and split seed.
- **Fine-tuning:** `TRAIN_DATASET_MODE = "bundled" | "generated"`, base model, GPU,
  backend, epochs, global/local batch, learning rate, AdamW, cosine schedule, weight
  decay, LoRA `dim`/`alpha`, seed, and merge toggle.
- **Matched evaluation:** victim agents, fixed attacker endpoints, judge, generations,
  attempts per tool, ports, and evaluation toggle.

The fine-tuning section explains how each control affects quality, runtime, and GPU
memory. It prints the resolved recipe, effective gradient accumulation, LoRA
`alpha / dim` scaling, exact one-GPU command, live logs, elapsed time, parsed losses,
best adapter, and merged model location.

NDD independently and uniformly samples the domain, configured tool count, and security
level for every profile candidate. Sampling is with replacement. The LLM generates
the operational profile, tools, and a security instruction; permissive profiles receive
no added instruction. A final NDD LLM judge scores each complete row from 1–5 for
realism, specificity, consistency, tool-purpose fit, direct attack surface, and
security-level fidelity. All five candidates are persisted, while only the
deterministically ranked top two continue to analysis. Every tool analysis must contain
exactly five unique candidate prompts.

The generated roles use healthcare scheduling, Git repository management, IT helpdesk,
coding assistance, retail order management, banking support, insurance claims, travel
booking, CRM sales, and personal-assistant domains. Research knowledge assistants and
on-call DevOps agents remain held out for evaluation.

Set `NDD_RUN_ID` before launching Jupyter if you want a stable run-directory
name instead of the generated timestamp/parameter name.

## What's in the folder

```
setup.sh                    clean-server installer and verifier
ndd_nemotron_garak.ipynb   the walkthrough
ndd_nemotron_garak_workshop.ipynb
                            live H100/API implementation lab with TODOs and hints
training/                  automodel_lora.yaml, finetune.py, merge_lora.py, chat template
agents/                    two LangGraph victim agents (Atlas/research, Pulse/devops) + mock data + server
garak_eval/                pinned agent configs, scan driver, ASR comparison
docker/garak/              reproducible pinned Garak image
scripts/                   cleanup, shared image pins, setup verification
output/artifacts/          self-contained run directories (created by the notebook)
```

The evaluation plugs the base and fine-tuned models into Garak's probe as the
**red-team model**, attacks each held-out agent on `:8000`, and reports **Attack
Success Rate** (landed attacks / configured attempt budget). Invalid attacker output
counts as a non-hit so denominators remain matched, while evaluated and skipped
attempt counts remain visible. Every attacker/victim pair starts a fresh victim
process so state-changing tools cannot influence the next comparison run.

## Run on NVIDIA Brev

Deploy the launchable with a single 80 GB GPU (H100/A100) and set its setup script to:

```bash
bash brev/setup.sh
```

Optional launchable environment variables: `NGC_API_KEY` (pull the training image) and
`NVIDIA_API_KEY` (pre-fill `.env`). Then open JupyterLab, select the `finetune-demo (.venv)`
kernel, and run `ndd_nemotron_garak_workshop.ipynb`. Setup output is logged to `~/brev-setup.log`.
