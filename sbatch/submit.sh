#!/bin/bash

#Submits a sbatch job, with the correct logs to the run name

set -e

cd "$(dirname "$0")/.."

run_name=${1:?usage: sbatch/submit.sh <run_name> <run_dir> [script]}
run_dir=${2:?usage: sbatch/submit.sh <run_name> <run_dir> [script]}
script=${3:-./sbatch/job.pipeline.sbatch}

mkdir -p "$run_dir/logs"
sbatch \
  --output="$run_dir/logs/%x-%j.out" \
  --error="$run_dir/logs/%x-%j.err" \
  "$script" "$run_name" "$run_dir"
