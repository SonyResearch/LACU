#!/usr/bin/env bash
set -euo pipefail

### -------- Port handling (optional via sbatch arg) --------
PORT_ARG="${1:-0}"
if [[ "$PORT_ARG" != "0" ]]; then
  echo "[info] Using accelerate main port: $PORT_ARG"
  MAIN_PROCESS_PORT_FLAG=(--main_process_port "$PORT_ARG")
  export MASTER_PORT="$PORT_ARG"
else
  echo "[info] No port passed; using accelerate main port 0 (auto-pick)"
  MAIN_PROCESS_PORT_FLAG=(--main_process_port 0)
  export MASTER_PORT=0
fi

BASE_MODEL="runwayml/stable-diffusion-v1-5"
WPROJ="CUL_big"
MAX_STEPS=600
LR=1e-5
MIXED_PREC="fp16"
TRAIN_BS=13
GRAD_ACC=2
LAMBDA_UNLEARN=1.0
LAMBDA_PRESERVE=10.0
LR_SCHED="constant_with_warmup"
LR_WARMUP_STEPS=800
TMAX=700
NO_MAPPING=100  # "fixed" for fixed contexual mapping

# Output root
ROOT_OUT="/model"

# Concept list in the order you want
CONCEPTS=(
  pikachu
  brad_pitt
  dog
  golf_ball
  van_gogh_style
  apple
  spiderman
  lionel_messi
  cartoon_style
  banana
)


### -------- Helper: build prompt paths for a concept --------
prompt_paths() {
  local concept="$1"
  FORGET_PROMPTS="/prompts/${concept}/train_${NO_MAPPING}.csv"
  NEUTRAL_PROMPTS="/prompts/${concept}/map_${NO_MAPPING}.csv"
  RETAIN_PROMPTS="/prompts/${concept}/related.txt"
}

### -------- Training runner --------
run_one() {
  local model_name="$1" 
  local concept="$2"

  prompt_paths "$concept"

  local out_dir="${ROOT_OUT}/${concept}_${concept}"
  mkdir -p "$out_dir"

  {
    echo "[run] concept=${concept}  model=${model_name}"
    accelerate launch "${MAIN_PROCESS_PORT_FLAG[@]}" train.py \
      --previous_model_path="$model_name" \
      --output_dir="$out_dir" \
      --forget_prompts_path="$FORGET_PROMPTS" \
      --neutral_prompts_path="$NEUTRAL_PROMPTS" \
      --retain_prompts_path="$RETAIN_PROMPTS" \
      --gradient_accumulation_steps="$GRAD_ACC" \
      --wandb_project "$WPROJ" \
      --wandb_run_name "${concept}_${NO_MAPPING}" \
      --mixed_precision="$MIXED_PREC" \
      --train_batch_size="$TRAIN_BS" \
      --learning_rate "$LR" \
      --max_train_steps="$MAX_STEPS" \
      --lambda_unlearn "$LAMBDA_UNLEARN" \
      --lambda_preserve "$LAMBDA_PRESERVE" \
      --lr_scheduler "$LR_SCHED" \
      --lr_warmup_steps "$LR_WARMUP_STEPS" \
      --unlearning_t_max "$TMAX" \
      --report_to wandb

    echo "------------------------------------------------------------------------"
    echo "FINISHED: Concept=${concept}, NO_MAPPING=${NO_MAPPING}, λ_unlearn=${LAMBDA_UNLEARN}, λ_preserve=${LAMBDA_PRESERVE},"
    echo "          TMAX=${TMAX}, Tcutoff=${TCUTOFF}, TBias=${TBIAS}, OUT=${out_dir}"
    echo "------------------------------------------------------------------------"
  } >&2

  # Return path to the next model (the checkpoint of this run)
  printf '%s\n' "${out_dir}/checkpoints/step_${MAX_STEPS}"
}

### -------- Main loop --------
# First run starts from base model, subsequent runs chain from prior output.
CURRENT_MODEL="$BASE_MODEL"
for idx in "${!CONCEPTS[@]}"; do
  concept="${CONCEPTS[$idx]}"
  next_model_path="$(run_one "$CURRENT_MODEL" "$concept")"
  CURRENT_MODEL="$next_model_path"
done

echo "[done] All concepts processed in order: ${CONCEPTS[*]}"