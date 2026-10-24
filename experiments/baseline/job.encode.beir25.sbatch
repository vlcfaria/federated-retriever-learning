#!/bin/bash
#SBATCH --job-name=encode-beir25
#SBATCH --partition=rtx8000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/home/vinicius.faria/federated-retriever-learning/experiments/baseline/logs/%x-%j.out
#SBATCH --error=/home/vinicius.faria/federated-retriever-learning/experiments/baseline/logs/%x-%j.err

set -e

ROOT=/home/vinicius.faria/federated-retriever-learning
RUN=e5-info-rlhn-baseline-centralized
RUN_DIR=$ROOT/experiments/baseline
PY=$ROOT/.venv/bin/python

export PYTHONHASHSEED=42
export WANDB_MODE=offline
export WANDB_PROJECT=${WANDB_PROJECT:-federated-retriever-learning}
export WANDB_NAME=$RUN
export WANDB_RUN_GROUP=$RUN
export WANDB_DIR=$RUN_DIR/wandb
export TRITON_CACHE_DIR=/tmp/.triton
export BEIR_CACHE_DIR="${BEIR_CACHE_DIR:-$HOME/.cache/tevatron/beir_qrels}"

mkdir -p "$RUN_DIR/wandb" "$RUN_DIR/encode" "$RUN_DIR/logs"

NSHARD=4
NGPU=${SLURM_GPUS_ON_NODE:-1}

# prevent orphaned shards
wave=()
cleanup() { for pid in "${wave[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT

for ds in arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact trec-covid webis-touche2020 cqadupstack-android cqadupstack-english cqadupstack-gaming cqadupstack-gis cqadupstack-mathematica cqadupstack-physics cqadupstack-programmers cqadupstack-stats cqadupstack-tex cqadupstack-unix cqadupstack-webmasters cqadupstack-wordpress; do
  out=$RUN_DIR/encode/$ds
  if [ -f "$out/.done" ]; then
    echo "encode: $ds already done, skipping"
    continue
  fi
  mkdir -p "$out"

  # materialize offline queries
  "$PY" -m tevatron.utils.prepare_queries --dataset "$ds" --output "$out/queries.jsonl"

  # encode corpus by giving one process per shard
  for ((s = 0; s < NSHARD; s += NGPU)); do
    wave=()
    for ((g = 0; g < NGPU && s + g < NSHARD; g++)); do
      idx=$((s + g))
      tag=$(printf '%02g' "$idx")
      CUDA_VISIBLE_DEVICES=$g "$PY" -m tevatron.retriever.driver.encode \
        --output_dir "$RUN_DIR/encode" \
        --model_name_or_path "$RUN_DIR/train" \
        --dataset_shard_index $idx \
        --dataset_config "$ds" \
        --encode_output_path "$out/corpus_$ds.${tag}.pkl" \
        --dataset_name=Tevatron/beir-corpus --passage_max_len=512 --per_device_eval_batch_size=1024 --dataset_number_of_shards=4 --attn_implementation=sdpa --bf16=false --pooling=mean --normalize=true '--query_prefix=query: ' '--passage_prefix=passage: ' --fp16=true \
        >"$RUN_DIR/logs/encode-$RUN-${ds}-shard${tag}-${SLURM_JOB_ID:-manual}.log" 2>&1 &
      wave+=($!)
    done
    for pid in "${wave[@]}"; do wait "$pid"; done
  done

  # encode queries in a single shard since there are so few
  CUDA_VISIBLE_DEVICES=0 "$PY" -m tevatron.retriever.driver.encode \
    --output_dir "$RUN_DIR/encode" \
    --model_name_or_path "$RUN_DIR/train" \
    --encode_is_query \
    --dataset_path "$out/queries.jsonl" \
    --encode_output_path "$out/query.pkl" \
    --dataset_name=json --query_max_len=512 --per_device_eval_batch_size=1024 --attn_implementation=sdpa --bf16=false --pooling=mean --normalize=true '--query_prefix=query: ' '--passage_prefix=passage: ' --fp16=true \
    >"$RUN_DIR/logs/encode-$RUN-${ds}-query-${SLURM_JOB_ID:-manual}.log" 2>&1

  touch "$out/.done"
done
