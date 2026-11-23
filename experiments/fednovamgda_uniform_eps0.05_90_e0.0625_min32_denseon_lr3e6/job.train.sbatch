#!/bin/bash
#SBATCH --job-name=fednovamgda-uniform-eps0.05-90-denseon-lr3e6
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00

set -e

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source ./script/config.sh
export WANDB_RUN_GROUP="fednovamgda-uniform-denseon-lr3e6"

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

export FLWR_DISABLE_UPDATE_CHECK=1
SUPERLINK_LOG="$RUN_DIR/logs/superlink-${SLURM_JOB_ID:-$$}.log"
mkdir -p "$FLWR_HOME/local-superlink"

flower-superlink \
  --insecure --simulation --isolation subprocess \
  --control-api-address "127.0.0.1:$FLWR_LOCAL_CONTROL_API_PORT" \
  --host 127.0.0.1 \
  --port "$FLWR_LOCAL_SUPERLINK_HTTP_API_PORT" \
  --log-file "$SUPERLINK_LOG" \
  --log-rotation-interval-hours 24 \
  --log-rotation-backup-count 7 \
  --database "$FLWR_HOME/local-superlink/state.db" &

SUPERLINK_PID=$!
trap 'kill "$SUPERLINK_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 300); do
  if ! kill -0 "$SUPERLINK_PID" 2>/dev/null; then
    echo "SuperLink exited before becoming healthy; log:"
    tail -100 "$SUPERLINK_LOG" 2>/dev/null || true
    exit 1
  fi
  if curl -sf "http://127.0.0.1:$FLWR_LOCAL_SUPERLINK_HTTP_API_PORT/health" >/dev/null; then
    break
  fi
  sleep 1
done

if ! curl -sf "http://127.0.0.1:$FLWR_LOCAL_SUPERLINK_HTTP_API_PORT/health" >/dev/null; then
  echo "SuperLink failed to become healthy; log:"
  tail -100 "$SUPERLINK_LOG" 2>/dev/null || true
  exit 1
fi

flwr run . --stream \
  --federation-config "num_supernodes=7 client_resources_num_cpus=4 client_resources_num_gpus=0.33 init_args_num_gpus=1" \
  -c "output-dir=\"$RUN_DIR/train\"" \
  -c "run-name=\"$RUN_NAME\"" \
  -c "model_name_or_path='lightonai/DenseOn-unsupervised'" \
  -c "pooling='cls'" \
  -c "query_prefix='query: '" \
  -c "passage_prefix='document: '" \
  -c "server-strategy='fednovamgda'" \
  -c "server-strategy-params='{\"epsilon\":0.05}'" \
  -c "weighting='uniform'" \
  -c "learning_rate=3e-6" \
  -c "num-server-rounds=90" \
  -c "max_steps=0" \
  -c "num_train_epochs=0.0625" \
  -c "min_steps=32" \
  -c "eval-interval=15"
