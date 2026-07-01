<div align="center">

# [ECCV 2026] Locality-Aware Continual Unlearning for Diffusion Models

[ArXiv Preprint](https://arxiv.org/abs/2512.02657)

</div>

LACU is a continual unlearning framework for text-to-image diffusion models. It uses model-aware score-prediction distance to select local mapping targets and local replay prompts, then trains with teacher-student distillation and lightweight parameter regularization.

<div align="center">
  <img src="images/flow.png" alt="LACU training flow">
  <br>
  <sub>LACU builds a local unlearn path and a local preservation path at each continual step.</sub>
</div>

## Setup

Create the training environment:

```bash
conda env create -f ldm_environment.yml
conda activate lacu-train
```

Create the evaluation environment:

```bash
conda env create -f qwenvl_environment.yml
conda activate lacu-eval
```

The scripts download model weights from Hugging Face when a model id such as `runwayml/stable-diffusion-v1-5` is used. If you regenerate prompts or mappings, set:

```bash
export OPENAI_API_KEY=...
```

## Continual Unlearning

The release runner uses the first 10 concepts from the paper sequence:

```text
pikachu, brad_pitt, golf_ball, van_gogh_style, apple,
spiderman, lionel_messi, cartoon_style, banana, mickey_mouse
```

Cached prompt assets for these concepts are included under `prompts_new/`, so the default run does not need to call the OpenAI API unless a required prompt file is missing.

Run the 10-step SD v1.5 LACU sequence:

```bash
conda activate lacu-train
./train.sh
```

Useful overrides:

```bash
BASE_MODEL=runwayml/stable-diffusion-v1-5 \
OUTPUT_ROOT=outputs/lacu_sd15_10 \
ACCELERATE_NUM_PROCESSES=1 \
TRAIN_BATCH_SIZE=18 \
GRADIENT_ACCUMULATION_STEPS=2 \
./train.sh
```

The final checkpoint path is printed at the end of the run.

`BASE_MODEL` is only the initial checkpoint for the sequence. The runner feeds
each completed checkpoint back into `train.py` as `--pretrained_model_name_or_path`
for the next concept.

Our reported runs used 3 H100 GPUs. If you run on smaller GPUs, including 48 GB cards, reduce `TRAIN_BATCH_SIZE` and compensate with `GRADIENT_ACCUMULATION_STEPS`. Very small per-device batches can make it difficult to reproduce the same numbers, even when the effective batch size is similar.

## Method

<div align="center">
  <img src="images/visual_4.png" alt="Locality-aware target selection">
  <br>
  <sub>Locality-aware target selection chooses nearby safe prompts in the model score-prediction space.</sub>
</div>

## Evaluation

Evaluate a single Diffusers checkpoint folder:

```bash
conda activate lacu-eval
./evaluate_checkpoint.sh outputs/lacu_sd15_10/10_mickey_mouse/checkpoints/step_350
```

Useful overrides:

```bash
PROMPTS_CSV=prompts_new/eval.csv \
PROMPTS_EXTRA_ROOT=prompts_new \
OUTPUT_ROOT=eval_outputs/mickey_mouse_step_350 \
GPU=0 \
NUM_IMAGES_PER_PROMPT=8 \
./evaluate_checkpoint.sh /path/to/diffusers/checkpoint
```

The evaluator generates images, computes CLIP scores, and runs Qwen2.5-VL yes/no classification with confusables-aware prompts.

## Qualitative Results

<div align="center">
  <img src="images/results.png" alt="Qualitative continual unlearning and retention results">
</div>

## Key Files

- `train.sh`: portable first-10-concept LACU training runner.
- `train.py`: current LACU training implementation.
- `prompt_gen.py`: forget/validation prompt generation.
- `clip_map_gen.py`: candidate mapping prompt generation.
- `model_score_map_gen.py`: score-prediction mapping selection.
- `retain_prompt_gen.py`: score-prediction local replay prompt generation.
- `evaluate_checkpoint.sh`: single-checkpoint evaluation wrapper.
- `eval.py`: image generation, CLIP scoring, and Qwen2.5-VL evaluation.

## Citation

```bibtex
@inproceedings{george2026locality,
  title={Locality-Aware Continual Unlearning for Diffusion Models},
  author={George, Naveen and Murata, Naoki and Takida, Yuhta and Mopuri, Konda Reddy and Mitsufuji, Yuki},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026},
  organization={Springer}
}
```
