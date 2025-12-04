<div align="center">
      
# Distill, Forget, Repeat: A Framework for Continual Unlearning in Text-to-Image Diffusion Models

<div align="left">

<div align="center">

###  [Arxiv Preprint](https://arxiv.org/abs/2512.02657) <br>
Distill, Forget, Repeat proposes a distillation-based framework utilizing contextual trajectory re-steering, generative replay, and parameter regularization to enable stable sequential unlearning. This approach effectively removes targeted concepts while preventing catastrophic retention collapse and compounding ripple effects, ensuring the model's general generative quality remains intact.

</div>

<div align="center">
  <img
    src="Teaser.png"
    alt="Continual Unlearning performance"
  >
</div>

## Prepare

### Environment Setup
A suitable conda environment named ```ldm``` can be created and activated with:

```
conda env create -f ldm_environment.yml
conda activate ldm
```

### Prompt Preparation
Use the prompts listed in the paper’s Supplementary Section to generate the full prompt set.
A script for generating them ```prompt_gen.py``` is included. Running it will produce the prompts and save them into a folder named prompts/. Change the ```--task``` argument to change the mapping technique and related prompts for training.

Before running the script, ensure that your OpenAI API key is available in the environment (e.g., export OPENAI_API_KEY=...).

## Code Implementation

### Continual Unlearning
To start training, run the ```train.bash``` script.
All hyperparameters can be modified directly inside the script.

If you’re using different GPU hardware from our setup, adjust the batch size and gradient accumulation steps accordingly.
Our experiments used 3× H100 (80 GB) GPUs; smaller or lower-memory GPUs will require proportionally smaller batch sizes or higher accumulation.

## Evaluation

### Environment Setup
Due to version conflicts between previous diffusers and recent Qwen model, evaluation is run in a separate Conda environment.
Create and activate it with:

```
conda env create -f qwenvl_environment.yml
conda activate qwenvl
```

### Code implementation
To evaluate a single checkpoint, run:

```
python eval.py \
--model_folder "$model_dir"  \
--prompts_csv "$PROMPTS_CSV"  \
--output_root "$out_dir"  \
--gpu 0  \
--use_confusables  \
--log_file "$out_dir/eval_log.txt"  \
--wandb_project "$WANDP"  \
--wandb_run_name "$run_name"  \
--wandb_mode online  \
--batch_size 64    \
--vlm_batch_size 32   \
--gen_prompt_batch 16   \
--save_workers 6
```

where:
- ```model_dir``` is the path to the checkpoint folder.
- ```PROMPTS_CSV``` is the prompts CSV file.
- ```out_dir``` is the output directory.


## Cite Our Work
The preprint can be cited as follows:
```
@article{george2025distillforgetrepeatframework,
  title={Distill, Forget, Repeat: A Framework for Continual Unlearning in Text-to-Image Diffusion Models},
  author={George, Naveen and Murata, Naoki and Takida, Yuhta and Mopuri, Konda Reddy and Mitsufuji, Yuki},
  journal={arXiv preprint arXiv:2512.02657},
  year={2025}
}
```
