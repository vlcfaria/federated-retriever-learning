#!/bin/bash
#SBATCH --job-name=encode-msmarco
#SBATCH --partition=rtx8000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/home/vinicius.faria/federated-retriever-learning/sbatch/logs/%x-%j.out
#SBATCH --error=/home/vinicius.faria/federated-retriever-learning/sbatch/logs/%x-%j.err

set -e

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

source ./script/config.sh

MODEL_FAMILY=${MODEL_FAMILY:-e5}
case "$MODEL_FAMILY" in
  e5)
    ENCODER_POOLING=mean
    ENCODER_QUERY_PREFIX='query: '
    ENCODER_PASSAGE_PREFIX='passage: '
    ;;
  denseon)
    ENCODER_POOLING=cls
    ENCODER_QUERY_PREFIX='query: '
    ENCODER_PASSAGE_PREFIX='document: '
    ;;
  *)
    echo "Unknown MODEL_FAMILY: $MODEL_FAMILY" >&2
    exit 1
    ;;
esac

mkdir -p "$RUN_DIR/wandb" "$RUN_DIR/encode" "$RUN_DIR/logs"

NSHARD=4
NGPU=${SLURM_GPUS_ON_NODE:-1}

ENCODER_DTYPE="${ENCODER_DTYPE:-fp16}"
case "$ENCODER_DTYPE" in
  bf16) ENC_CMD=("$PY" script/encode_bf16.py); ENC_DTYPE_ARGS=(--bf16=true --fp16=false) ;;
  fp16) ENC_CMD=("$PY" -m tevatron.retriever.driver.encode); ENC_DTYPE_ARGS=(--bf16=false --fp16=true) ;;
  fp32) ENC_CMD=("$PY" -m tevatron.retriever.driver.encode); ENC_DTYPE_ARGS=(--bf16=false --fp16=false) ;;
  *) echo "Unknown ENCODER_DTYPE: $ENCODER_DTYPE (expected fp16|bf16|fp32)" >&2; exit 1 ;;
esac
echo "encode msmarco: dtype=$ENCODER_DTYPE"

# prevent orphaned shards
wave=()
cleanup() { for pid in "${wave[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT

for ds in msmarco; do
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
      tag=$(printf '%02d' "$idx")
      CUDA_VISIBLE_DEVICES=$g "${ENC_CMD[@]}" \
        --output_dir "$RUN_DIR/encode" \
        --model_name_or_path "$RUN_DIR/train" \
        --dataset_shard_index $idx \
        --encode_output_path "$out/corpus_$ds.${tag}.pkl" \
        --dataset_name=Tevatron/msmarco-passage-corpus --passage_max_len=512 --per_device_eval_batch_size=1024 --dataset_number_of_shards="$NSHARD" --attn_implementation=sdpa --pooling="$ENCODER_POOLING" --normalize=true "--query_prefix=$ENCODER_QUERY_PREFIX" "--passage_prefix=$ENCODER_PASSAGE_PREFIX" "${ENC_DTYPE_ARGS[@]}" \
        >"$RUN_DIR/logs/encode-$RUN_NAME-${ds}-shard${tag}-${SLURM_JOB_ID:-manual}.log" 2>&1 &
      wave+=($!)
    done
    for pid in "${wave[@]}"; do wait "$pid"; done
  done

  # encode queries in a single shard since there are so few
  CUDA_VISIBLE_DEVICES=0 "${ENC_CMD[@]}" \
    --output_dir "$RUN_DIR/encode" \
    --model_name_or_path "$RUN_DIR/train" \
    --encode_is_query \
    --dataset_path "$out/queries.jsonl" \
    --encode_output_path "$out/query.pkl" \
    --dataset_name=json --query_max_len=512 --per_device_eval_batch_size=512 --attn_implementation=sdpa --pooling="$ENCODER_POOLING" --normalize=true "--query_prefix=$ENCODER_QUERY_PREFIX" "--passage_prefix=$ENCODER_PASSAGE_PREFIX" "${ENC_DTYPE_ARGS[@]}" \
    >"$RUN_DIR/logs/encode-$RUN_NAME-${ds}-query-${SLURM_JOB_ID:-manual}.log" 2>&1

  touch "$out/.done"
done
