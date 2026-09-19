#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DATA_ROOT:?Set DATA_ROOT to the processed BrightEx root}"
: "${CLOUD_DIR:?Set CLOUD_DIR to the authorised cloud-asset directory}"
: "${SAR_PRETRAINED:?Set SAR_PRETRAINED to the converted Sentinel-1 checkpoint}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-12}"

cd "${REPO_ROOT}/src"
python train.py \
  --dataset "BrightEx_Full_Seed${SEED}" \
  --train_batch_size 16 \
  --eval_batch_size 8 \
  --num_workers "${NUM_WORKERS}" \
  --max_iters 500000 \
  --warmup_iters 800 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --model_type "MBDD_Full_Seed${SEED}" \
  --model_param_path "${OUTPUT_ROOT}" \
  --train_dataset_path "${DATA_ROOT}" \
  --train_data_list_path "${REPO_ROOT}/splits/train.txt" \
  --val_dataset_path "${DATA_ROOT}" \
  --val_data_list_path "${REPO_ROOT}/splits/val.txt" \
  --cloud_path "${CLOUD_DIR}" \
  --sar_pretrained_path "${SAR_PRETRAINED}" \
  --seed "${SEED}"

