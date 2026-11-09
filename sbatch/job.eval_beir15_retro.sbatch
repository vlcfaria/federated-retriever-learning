#!/bin/bash
#SBATCH --job-name=eval-beir15-retro
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/vinicius.faria/federated-retriever-learning/sbatch/logs/%x-%j.out
#SBATCH --error=/home/vinicius.faria/federated-retriever-learning/sbatch/logs/%x-%j.err

set -e

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

export OPENAI_API_KEY=dummy
export BEIR_CACHE_DIR="${BEIR_CACHE_DIR:-$HOME/.cache/tevatron/beir_qrels}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Reuse the per-dataset JSONs from the old split eval dirs. Tevatron's eval.run
# reads them, skips re-scoring, and renders the true 15-unit BEIR-15 summary.
for run_dir in experiments/*; do
  [ -d "$run_dir/eval/beir25" ] || continue
  [ -f "$run_dir/eval/msmarco/msmarco.metrics.json" ] || continue

  out="$run_dir/eval/beir15"
  rm -rf "$out"
  mkdir -p "$out"
  cp "$run_dir"/eval/beir25/*.metrics.json "$out"/
  cp "$run_dir"/eval/msmarco/*.metrics.json "$out"/

  echo "=== $run_dir ==="
  .venv/bin/python -m tevatron.eval.run \
    --benchmark beir15 \
    --ranklist_pattern "$run_dir/search/{dataset}/rank.text" \
    --results_dir "$out" \
    --name "$(basename "$run_dir")"
done
