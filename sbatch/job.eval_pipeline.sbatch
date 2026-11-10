#!/bin/bash
#SBATCH --job-name=pipeline
#SBATCH --partition=rtx8000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --exclude=dl-05
#SBATCH --output=/home/vinicius.faria/federated-retriever-learning/sbatch/logs/%x-%j.out
#SBATCH --error=/home/vinicius.faria/federated-retriever-learning/sbatch/logs/%x-%j.err

set -e

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

: "${1:?usage: sbatch sbatch/job.pipeline.sbatch <run_name> <run_dir>}"
: "${2:?usage: sbatch sbatch/job.pipeline.sbatch <run_name> <run_dir>}"

for step in encode.beir25 encode.msmarco search.beir25 search.msmarco; do
  bash "./sbatch/job.$step.sbatch" "$1" "$2"
done

# Score the full BEIR-15 task set in one Tevatron eval call. This writes
# experiments/<run>/eval/beir15/summary.md with the true 15-unit aggregate:
# BEIR-13 + MS MARCO dev + cqadupstack.
export OPENAI_API_KEY=dummy
export BEIR_CACHE_DIR="${BEIR_CACHE_DIR:-$HOME/.cache/tevatron/beir_qrels}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

.venv/bin/python -m tevatron.eval.run \
  --benchmark beir15 \
  --ranklist_pattern "$2/search/{dataset}/rank.text" \
  --results_dir "$2/eval/beir15" \
  --name "$1"
