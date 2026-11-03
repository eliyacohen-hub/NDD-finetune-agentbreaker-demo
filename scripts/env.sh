#!/usr/bin/env bash

export AUTOMODEL_IMAGE="${AUTOMODEL_IMAGE:-nvcr.io/nvidia/nemo-automodel@sha256:c4f613005518d520c2ac3d9206d95617a2385f86cf8aa09582aad8d35957e2f2}"
export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.17.1}"
export GARAK_REVISION="${GARAK_REVISION:-4b7761c2108446043012549cbe8f7c942cabb26d}"
export GARAK_IMAGE="${GARAK_IMAGE:-finetune-demo-garak:${GARAK_REVISION:0:12}}"
