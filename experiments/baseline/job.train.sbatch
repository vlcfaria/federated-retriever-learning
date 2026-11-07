#!/bin/bash
#SBATCH --job-name=train-baseline-centralized
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=18:00:00

set -e

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source ./script/config.sh

mkdir -p "$RUN_DIR/wandb" "$RUN_DIR/train" "$RUN_DIR/logs" "$RUN_DIR/validation"

"$DEEPSPEED" --master_port=$((20000 + RANDOM % 20000)) \
  --module tevatron.retriever.driver.train \
  --deepspeed "$DS_CONFIG" \
  --output_dir "$RUN_DIR/train" \
  --model_name_or_path=intfloat/e5-small-unsupervised \
  --attn_implementation=sdpa --bf16=true --pooling=mean --normalize=true \
  '--query_prefix=query: ' '--passage_prefix=passage: ' \
  --dataset_name="../dataset/rlhn-680k/dev-split/train" \
  --query_max_len=192 --passage_max_len=192 --train_group_size=8 \
  --per_device_train_batch_size=256 --gradient_accumulation_steps=1 \
  --learning_rate=2e-5 --temperature=0.01 \
  --logging_steps=20 --save_strategy=steps --save_total_limit=10 \
  --report_to=wandb --seed=42 --weight_decay=0.01 --warmup_steps=0.1 \
  --gradient_checkpointing=true --dataloader_num_workers=8 \
  --max_steps=10000 --save_steps=2000 \
  --validate_mode=sync --eval_query_dir="../dataset/rlhn-680k/dev-split/dev" \
  --eval_corpus_dir=../dataset/rlhn-680k/dev-corpus-final --validate_query_max_len=512 \
  --validate_passage_max_len=512 --validate_output_dir="$RUN_DIR/validation"
