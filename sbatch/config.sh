#!/bin/bash

# Inserts relevant positional parameters: run_name run_dir

RUN_NAME=$1
RUN_DIR=$2

: "${RUN_NAME:?usage: sbatch <script> <run_name> <run_dir>}"
: "${RUN_DIR:?usage: sbatch <script> <run_name> <run_dir>}"

export PYTHONHASHSEED=42
export WANDB_MODE=offline
export WANDB_PROJECT="${WANDB_PROJECT:-federated-retriever-learning}"
export WANDB_NAME="$RUN_NAME"
export WANDB_RUN_GROUP="$RUN_NAME"
export WANDB_DIR="$RUN_DIR/wandb"
export TRITON_CACHE_DIR=/tmp/.triton
export BEIR_CACHE_DIR="${BEIR_CACHE_DIR:-$HOME/.cache/tevatron/beir_qrels}"

PY=.venv/bin/python
DEEPSPEED=.venv/bin/deepspeed
DS_CONFIG=./config/ds_zero0_config.json
