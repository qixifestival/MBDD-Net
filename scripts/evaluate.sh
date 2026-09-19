#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DATA_ROOT:?Set DATA_ROOT to the processed BrightEx root}"
: "${MODEL_PATH:?Set MODEL_PATH to a trained checkpoint}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/evaluation}"
NUM_WORKERS="${NUM_WORKERS:-12}"

EXTRA_ARGS=()
if [[ "${DISABLE_FAM:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable_fam)
fi
if [[ -n "${SAR_PRETRAINED:-}" ]]; then
  EXTRA_ARGS+=(--sar_pretrained_path "${SAR_PRETRAINED}")
fi

cd "${REPO_ROOT}/src"
python test.py \
  --model_path "${MODEL_PATH}" \
  --model_name MBDD \
  --test_batch_size 8 \
  --num_workers "${NUM_WORKERS}" \
  --test_dataset_path "${DATA_ROOT}/test" \
  --test_data_list_path "${REPO_ROOT}/splits/test.txt" \
  --output_dir "${OUTPUT_ROOT}/natural" \
  "${EXTRA_ARGS[@]}"

if [[ -n "${CLOUD_DIR:-}" ]]; then
  python test.py \
    --model_path "${MODEL_PATH}" \
    --model_name MBDD \
    --test_batch_size 8 \
    --num_workers "${NUM_WORKERS}" \
    --test_dataset_path "${DATA_ROOT}/test" \
    --test_data_list_path "${REPO_ROOT}/splits/test.txt" \
    --cloud_path "${CLOUD_DIR}" \
    --output_dir "${OUTPUT_ROOT}/cloud_p03" \
    "${EXTRA_ARGS[@]}"
fi

