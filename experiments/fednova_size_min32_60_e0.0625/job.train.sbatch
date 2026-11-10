#!/bin/bash
#SBATCH --job-name=fednova-size-min32-60-e0.0625
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --exclude=dl-05

set -e

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source ./script/config.sh

# Isolate Flower's local SuperLink per job.
JOB_OFFSET=$(( ${SLURM_JOB_ID:-$$} % 1000 ))
export FLWR_HOME="/tmp/${USER}/flwr-${SLURM_JOB_ID:-$$}"
mkdir -p "$FLWR_HOME"
if [[ -f "$HOME/.flwr/config.toml" ]]; then
  cp "$HOME/.flwr/config.toml" "$FLWR_HOME/config.toml"
fi
export FLWR_LOCAL_SUPERLINK_HTTP_API_PORT=$((39091 + JOB_OFFSET))
export FLWR_LOCAL_CONTROL_API_PORT=$((40093 + JOB_OFFSET))

mkdir -p "$RUN_DIR/wandb" "$RUN_DIR/train" "$RUN_DIR/logs" "$RUN_DIR/validation"

flwr run . --stream \
  --federation-config "num_supernodes=7 client_resources_num_cpus=2 client_resources_num_gpus=0.14 init_args_num_gpus=1" \
  -c "output-dir=\"$RUN_DIR/train\"" \
  -c "run-name=\"$RUN_NAME\"" \
  -c "server-strategy='fednova'" \
  -c "weighting='size'" \
  -c "num-server-rounds=60" \
  -c "max_steps=0" \
  -c "num_train_epochs=0.0625" \
  -c "min_steps=32" \
  -c "eval-interval=12"
