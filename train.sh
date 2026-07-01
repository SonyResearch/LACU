#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_MODEL="${BASE_MODEL:-runwayml/stable-diffusion-v1-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/lacu_sd15_10}"
PROMPTS_ROOT="${PROMPTS_ROOT:-$SCRIPT_DIR/prompts_new}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
ACCELERATE_NUM_PROCESSES="${ACCELERATE_NUM_PROCESSES:-1}"

MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-18}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LR_SCHEDULER="${LR_SCHEDULER:-constant_with_warmup}"
T_MAX="${T_MAX:-600}"
T_CUTOFF="${T_CUTOFF:-600}"
T_BIAS="${T_BIAS:-1.0}"
NUM_MAPPING_PROMPTS="${NUM_MAPPING_PROMPTS:-100}"
MAP_K="${MAP_K:-32}"
RETAIN_K="${RETAIN_K:-16}"

REPORT_TO="${REPORT_TO:-}"
WANDB_PROJECT="${WANDB_PROJECT:-LACU}"
WANDB_MODE="${WANDB_MODE:-disabled}"

CONCEPTS=(
  pikachu
  brad_pitt
  golf_ball
  van_gogh_style
  apple
  spiderman
  lionel_messi
  cartoon_style
  banana
  mickey_mouse
)

STEPS=(250 250 300 350 300 300 300 350 300 350)
LRS=(5e-5 5e-5 5e-5 5e-5 5e-5 5e-5 5e-5 5e-5 5e-5 5e-5)
WARMUPS=(30 30 30 30 30 30 30 30 30 30)
LAMBDA_UNLEARN=(5.0 5.0 5.0 7.0 5.5 5.5 6.0 7.0 6.0 6.0)
LAMBDA_PRESERVE=(25.0 25.0 25.0 25.0 25.0 25.0 25.0 25.0 25.0 25.0)
LAMBDA_REG=(2e-4 2e-4 2e-4 2e-4 2e-4 2e-4 2e-4 2e-4 2e-4 2e-4)

mkdir -p "$OUTPUT_ROOT" "$PROMPTS_ROOT"

display_name() {
  local concept="$1"
  printf '%s\n' "${concept//_/ }"
}

require_file() {
  local path="$1"
  local message="$2"
  if [[ ! -f "$path" ]]; then
    echo "[fatal] Missing $message: $path" >&2
    exit 1
  fi
}

ensure_forget_prompts() {
  local concept="$1"
  local concept_dir="$PROMPTS_ROOT/$concept"
  local train_csv="$concept_dir/train_${NUM_MAPPING_PROMPTS}.csv"

  if [[ -f "$train_csv" ]]; then
    return
  fi

  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[fatal] $train_csv is missing and OPENAI_API_KEY is not set." >&2
    exit 1
  fi

  python "$SCRIPT_DIR/prompt_gen.py" \
    --concept "$(display_name "$concept")" \
    --task mapped \
    --n "$NUM_MAPPING_PROMPTS" \
    --suffix "$NUM_MAPPING_PROMPTS" \
    --model "$OPENAI_MODEL" \
    --out_root "$PROMPTS_ROOT"
}

ensure_mapping() {
  local model_path="$1"
  local concept="$2"
  local concept_dir="$PROMPTS_ROOT/$concept"
  local map_csv="$concept_dir/map_model.csv"
  local candidates_json="$concept_dir/candidates_clip.json"

  ensure_forget_prompts "$concept"

  if [[ ! -f "$candidates_json" ]]; then
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      echo "[fatal] $candidates_json is missing and OPENAI_API_KEY is not set." >&2
      exit 1
    fi
    python "$SCRIPT_DIR/clip_map_gen.py" \
      --concept "$concept" \
      --prompts_root "$PROMPTS_ROOT" \
      --model "$OPENAI_MODEL" \
      --save_candidates
  fi

  if [[ ! -f "$map_csv" ]]; then
    python "$SCRIPT_DIR/model_score_map_gen.py" \
      --mode discrete \
      --K "$MAP_K" \
      --concept "$concept" \
      --model_path "$model_path" \
      --prompts_root "$PROMPTS_ROOT"
  fi
}

ensure_retain_prompts() {
  local model_path="$1"
  local concept="$2"
  local retain_txt="$PROMPTS_ROOT/$concept/related_score.txt"

  if [[ -f "$retain_txt" ]]; then
    return
  fi

  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[fatal] $retain_txt is missing and OPENAI_API_KEY is not set." >&2
    exit 1
  fi

  python "$SCRIPT_DIR/retain_prompt_gen.py" \
    --concept "$concept" \
    --model_path "$model_path" \
    --prompts_root "$PROMPTS_ROOT" \
    --llm_model "$OPENAI_MODEL" \
    --K "$RETAIN_K" \
    --skip_clip
}

train_one() {
  local model_path="$1"
  local concept="$2"
  local idx="$3"
  local steps="${STEPS[$idx]}"
  local lr="${LRS[$idx]}"
  local warmup="${WARMUPS[$idx]}"
  local lambda_unlearn="${LAMBDA_UNLEARN[$idx]}"
  local lambda_preserve="${LAMBDA_PRESERVE[$idx]}"
  local lambda_reg="${LAMBDA_REG[$idx]}"
  local order
  printf -v order "%02d" "$((idx + 1))"

  local concept_dir="$PROMPTS_ROOT/$concept"
  local out_dir="$OUTPUT_ROOT/${order}_${concept}"
  local final_ckpt="$out_dir/checkpoints/step_${steps}"

  if [[ -d "$final_ckpt" ]]; then
    echo "[resume] $concept already complete: $final_ckpt" >&2
    printf '%s\n' "$final_ckpt"
    return
  fi

  if [[ -d "$out_dir/checkpoints" ]]; then
    echo "[fatal] Partial output exists for $concept: $out_dir" >&2
    echo "        Move it aside or set OUTPUT_ROOT to a new directory before rerunning." >&2
    exit 1
  fi

  require_file "$concept_dir/train_${NUM_MAPPING_PROMPTS}.csv" "forget prompts"
  require_file "$concept_dir/map_model.csv" "score-selected mapping prompts"
  require_file "$concept_dir/related_score.txt" "score-selected retain prompts"

  local cmd=(
    accelerate launch
    --num_processes "$ACCELERATE_NUM_PROCESSES"
    --main_process_port "$MAIN_PROCESS_PORT"
    "$SCRIPT_DIR/train.py"
    --pretrained_model_name_or_path "$model_path"
    --output_dir "$out_dir"
    --forget_prompts_path "$concept_dir/train_${NUM_MAPPING_PROMPTS}.csv"
    --neutral_prompts_path "$concept_dir/map_model.csv"
    --retain_prompts_path "$concept_dir/related_score.txt"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --mixed_precision "$MIXED_PRECISION"
    --train_batch_size "$TRAIN_BATCH_SIZE"
    --learning_rate "$lr"
    --max_train_steps "$steps"
    --lambda_unlearn "$lambda_unlearn"
    --lambda_preserve "$lambda_preserve"
    --lambda_reg "$lambda_reg"
    --lr_scheduler "$LR_SCHEDULER"
    --lr_warmup_steps "$warmup"
    --unlearning_t_max "$T_MAX"
    --unlearning_cutoff "$T_CUTOFF"
    --unlearning_bias "$T_BIAS"
    --num_inference_steps_replay 20
    --wandb_project "$WANDB_PROJECT"
    --wandb_run_name "${order}_${concept}"
    --wandb_mode "$WANDB_MODE"
  )

  if [[ -n "$REPORT_TO" && "$REPORT_TO" != "none" ]]; then
    cmd+=(--report_to "$REPORT_TO")
  fi

  echo "[train] $concept -> $out_dir" >&2
  "${cmd[@]}" >&2
  printf '%s\n' "$final_ckpt"
}

current_model="$BASE_MODEL"

for idx in "${!CONCEPTS[@]}"; do
  concept="${CONCEPTS[$idx]}"
  echo "========================================================================"
  echo "[lacu] concept $((idx + 1))/${#CONCEPTS[@]}: $concept"
  echo "========================================================================"

  ensure_mapping "$current_model" "$concept"
  ensure_retain_prompts "$current_model" "$concept"
  current_model="$(train_one "$current_model" "$concept" "$idx")"
done

echo "[done] Final checkpoint: $current_model"
