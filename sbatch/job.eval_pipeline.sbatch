#!/bin/bash
#SBATCH --job-name=pipeline
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

: "${1:?usage: sbatch sbatch/job.pipeline.sbatch <run_name> <run_dir>}"
: "${2:?usage: sbatch sbatch/job.pipeline.sbatch <run_name> <run_dir>}"

for step in encode.beir25 encode.msmarco search.beir25 search.msmarco; do
  bash "./sbatch/job.$step.sbatch" "$1" "$2"
done
