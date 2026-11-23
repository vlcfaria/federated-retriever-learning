#!/bin/bash
#SBATCH --job-name=central-denseon-b128
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=4-00:00:00
#SBATCH --exclude=dl-05
#SBATCH --output=/home/vinicius.faria/federated-retriever-learning/sbatch/logs/%x-%j.out
#SBATCH --error=/home/vinicius.faria/federated-retriever-learning/sbatch/logs/%x-%j.err

# Centralized DenseOn on the pooled RLHN pool, following experiments/baseline's
# DeepSpeed launcher (deepspeed --module ... --deepspeed $DS_CONFIG).
# Sized against the federated run: batch 128 -> 1024 passages/step
# (~1023 in-batch negatives, the same regime as LightOn's batch-128 recipe;
# federated batch 64 x group 8 had ~511), and 9141 steps = 90 rounds x 101.57.
# ds_zero0_config.json already uses WarmupDecayLR (warmup 0.1, linear decay to 0).
# Same recipe otherwise: 192/192, group 8, temp 0.01, LR 2e-5, wd 0.01, CLS.

set -e

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source ./script/config.sh

mkdir -p "$RUN_DIR/wandb" "$RUN_DIR/train" "$RUN_DIR/logs" "$RUN_DIR/validation"

"$DEEPSPEED" --master_port=$((20000 + RANDOM % 20000)) \
  --module tevatron.retriever.driver.train \
  --deepspeed "$DS_CONFIG" \
  --output_dir "$RUN_DIR/train" \
  --model_name_or_path=lightonai/DenseOn-unsupervised \
  --attn_implementation=sdpa --bf16=true --pooling=cls --normalize=true \
  '--query_prefix=query: ' '--passage_prefix=document: ' \
  --dataset_name="bigni/rlhn-680k-dev" --dataset_split=train \
  --query_max_len=192 --passage_max_len=192 --train_group_size=8 \
  --per_device_train_batch_size=128 --gradient_accumulation_steps=1 \
  --learning_rate=2e-5 --temperature=0.01 \
  --logging_steps=50 --save_strategy=steps --save_total_limit=6 \
  --report_to=wandb --seed=42 --weight_decay=0.01 --warmup_steps=0.1 \
  --gradient_checkpointing=true --dataloader_num_workers=12 \
  --max_steps=9141 --save_steps=1500 \
  --validate_mode=sync --eval_query_dir="../dataset/rlhn-680k/dev-split/dev" \
  --eval_corpus_dir=../dataset/rlhn-680k/dev-corpus-final --validate_query_max_len=512 \
  --validate_passage_max_len=512 --validate_output_dir="$RUN_DIR/validation" \
  --do_train
