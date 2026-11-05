#!/bin/bash

#Submits a sbatch job, with the correct logs to the run name

set -e

cd "$(dirname "$0")/.."

run_name=${1:?usage: script/submit.sh <run_name> <run_dir> [script] [sbatch opts...] [-- script args...]}
run_dir=${2:?usage: script/submit.sh <run_name> <run_dir> [script] [sbatch opts...] [-- script args...]}
shift 2

# Optional script path; tolerate sbatch options in its place so they can be
# passed without naming the default script explicitly.
if [[ $# -gt 0 && "$1" != -* ]]; then
  script=$1
  shift
else
  script=./sbatch/job.eval_pipeline.sbatch
fi

# Arguments before `--` become sbatch options; arguments after go to the job script.
sbatch_opts=()
script_args=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then
    shift
    script_args=("$@")
    break
  fi
  sbatch_opts+=("$1")
  shift
done

mkdir -p "$run_dir/logs"
sbatch \
  --output="$run_dir/logs/%x-%j.out" \
  --error="$run_dir/logs/%x-%j.err" \
  "${sbatch_opts[@]}" \
  "$script" "$run_name" "$run_dir" "${script_args[@]}"
