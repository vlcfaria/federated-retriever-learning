#!/bin/bash
#SBATCH --job-name=train-baseline-centralized
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/vinicius.faria/federated-retriever-learning/experiments/baseline/logs/%x-%j.out
#SBATCH --error=/home/vinicius.faria/federated-retriever-learning/experiments/baseline/logs/%x-%j.err

set -e

RUN=e5-info-rlhn-baseline-centralized
RUN_DIR=./experiments/baseline
PY=.venv/bin/python
DEEPSPEED=.venv/bin/deepspeed
DS_CONFIG=./config/ds_zero0_config.json

export PYTHONHASHSEED=42
export WANDB_MODE=offline
export WANDB_PROJECT=${WANDB_PROJECT:-federated-retriever-learning}
export WANDB_NAME=$RUN
export WANDB_RUN_GROUP=$RUN
export WANDB_DIR=$RUN_DIR/wandb
export TRITON_CACHE_DIR=/tmp/.triton
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
# Single-node only: the irdma/RoCE IB transport segfaults NCCL on these nodes.
export NCCL_IB_DISABLE=1

mkdir -p "$RUN_DIR/wandb" "$RUN_DIR/train" "$RUN_DIR/logs" "$RUN_DIR/validation"

"$DEEPSPEED" --master_port=$((20000 + RANDOM % 20000)) \
  --module tevatron.retriever.driver.train \
  --deepspeed "$DS_CONFIG" \
  --output_dir "$RUN_DIR/train" \
  --model_name_or_path=intfloat/e5-small-unsupervised \
  --attn_implementation=sdpa --bf16=true --pooling=mean --normalize=true \
  '--query_prefix=query: ' '--passage_prefix=passage: ' \
  --dataset_name="../dataset/rlhn-680k/dev-split/train" \
  --query_max_len=350 --passage_max_len=350 --train_group_size=8 \
  --per_device_train_batch_size=256 --gradient_accumulation_steps=1 \
  --learning_rate=2e-5 --temperature=0.01 \
  --logging_steps=20 --save_strategy=steps --save_total_limit=10 \
  --report_to=wandb --seed=42 --weight_decay=0.01 --warmup_steps=0.1 \
  --gradient_checkpointing=true --dataloader_num_workers=8 \
  --max_steps=20000 --save_steps=2000 \
  --validate_mode=sync --eval_query_dir="../dataset/rlhn-680k/dev-split/dev" \
  --eval_corpus_dir=../dataset/rlhn-680k/dev-corpus-final --validate_query_max_len=512 \
  --validate_passage_max_len=512 --validate_output_dir="$RUN_DIR/validation"
