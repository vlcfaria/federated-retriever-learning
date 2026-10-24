#!/bin/bash
#SBATCH --job-name=search-beir25
#SBATCH --partition=rtx8000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=6:00:00
#SBATCH --output=/home/vinicius.faria/federated-retriever-learning/experiments/baseline/logs/%x-%j.out
#SBATCH --error=/home/vinicius.faria/federated-retriever-learning/experiments/baseline/logs/%x-%j.err

set -e

ROOT=/home/vinicius.faria/federated-retriever-learning
RUN=e5-info-rlhn-baseline-centralized
RUN_DIR=$ROOT/experiments/baseline
PY=$ROOT/.venv/bin/python

export PYTHONHASHSEED=42
export TRITON_CACHE_DIR=/tmp/.triton
export BEIR_CACHE_DIR="${BEIR_CACHE_DIR:-$HOME/.cache/tevatron/beir_qrels}"
# make FAISS aware of the number of our cpus
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

mkdir -p "$RUN_DIR/search" "$RUN_DIR/eval/beir25" "$RUN_DIR/logs"

for ds in arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact trec-covid webis-touche2020 cqadupstack-android cqadupstack-english cqadupstack-gaming cqadupstack-gis cqadupstack-mathematica cqadupstack-physics cqadupstack-programmers cqadupstack-stats cqadupstack-tex cqadupstack-unix cqadupstack-webmasters cqadupstack-wordpress; do
  out=$RUN_DIR/search/$ds
  if [ -f "$out/.done" ]; then
    echo "search: $ds already done, skipping"
    continue
  fi
  mkdir -p "$out"

  set -f
  "$PY" -m tevatron.retriever.driver.search \
    --query_reps "$RUN_DIR/encode/$ds/query.pkl" \
    --passage_reps "$RUN_DIR/encode/$ds/corpus_$ds.*.pkl" \
    --depth 1000 \
    --batch_size 64 \
    --shard_search \
    --gpu_float32 \
    --save_text \
    --save_ranking_to "$out/rank.text"
  set +f

  touch "$out/.done"
done

# start evaluation
export OPENAI_API_KEY=dummy

"$PY" -m tevatron.eval.run \
  --benchmark beir15 \
  --datasets arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact trec-covid webis-touche2020 cqadupstack-android cqadupstack-english cqadupstack-gaming cqadupstack-gis cqadupstack-mathematica cqadupstack-physics cqadupstack-programmers cqadupstack-stats cqadupstack-tex cqadupstack-unix cqadupstack-webmasters cqadupstack-wordpress \
  --ranklist_pattern "$RUN_DIR/search/{dataset}/rank.text" \
  --results_dir "$RUN_DIR/eval/beir25" \
  --name "$RUN" \
  >"$RUN_DIR/logs/eval-$RUN-${SLURM_JOB_ID:-manual}.log" 2>&1
