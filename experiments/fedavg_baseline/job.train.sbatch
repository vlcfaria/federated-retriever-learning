#!/bin/bash
#SBATCH --job-name=fedavg_baseline
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=18:00:00

set -e

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source ./script/config.sh

mkdir -p "$RUN_DIR/wandb" "$RUN_DIR/train" "$RUN_DIR/logs" "$RUN_DIR/validation"

flwr run . --stream \
  --federation-config "num_supernodes=7 client_resources_num_cpus=2 client_resources_num_gpus=0.14 init_args_num_gpus=1" \
  -c "output-dir=\"$RUN_DIR/train\"" \
  -c "run-name=\"$RUN_NAME\""
