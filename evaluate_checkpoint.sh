#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_FOLDER="${MODEL_FOLDER:-${1:-}}"
if [[ -z "$MODEL_FOLDER" || "$MODEL_FOLDER" == "-h" || "$MODEL_FOLDER" == "--help" ]]; then
  echo "Usage: MODEL_FOLDER=/path/to/diffusers/checkpoint ./evaluate_checkpoint.sh"
  echo "   or: ./evaluate_checkpoint.sh /path/to/diffusers/checkpoint"
  exit 1
fi

PROMPTS_CSV="${PROMPTS_CSV:-$SCRIPT_DIR/prompts/eval.csv}"
PROMPTS_EXTRA_ROOT="${PROMPTS_EXTRA_ROOT:-$SCRIPT_DIR/prompts}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/eval_outputs/$(basename "$MODEL_FOLDER")}"
GPU="${GPU:-0}"
NUM_IMAGES_PER_PROMPT="${NUM_IMAGES_PER_PROMPT:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VLM_BATCH_SIZE="${VLM_BATCH_SIZE:-32}"
GEN_PROMPT_BATCH="${GEN_PROMPT_BATCH:-16}"
SAVE_WORKERS="${SAVE_WORKERS:-6}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_MODE="${WANDB_MODE:-disabled}"

mkdir -p "$OUTPUT_ROOT"

cmd=(
  python "$SCRIPT_DIR/eval.py"
  --model_folder "$MODEL_FOLDER"
  --prompts_csv "$PROMPTS_CSV"
  --prompts_extra_root "$PROMPTS_EXTRA_ROOT"
  --output_root "$OUTPUT_ROOT"
  --gpu "$GPU"
  --use_confusables
  --log_file "$OUTPUT_ROOT/eval_log.txt"
  --batch_size "$BATCH_SIZE"
  --vlm_batch_size "$VLM_BATCH_SIZE"
  --gen_prompt_batch "$GEN_PROMPT_BATCH"
  --num_images_per_prompt "$NUM_IMAGES_PER_PROMPT"
  --save_workers "$SAVE_WORKERS"
  --wandb_mode "$WANDB_MODE"
)

if [[ -n "$WANDB_PROJECT" ]]; then
  cmd+=(--wandb_project "$WANDB_PROJECT")
fi

"${cmd[@]}"
