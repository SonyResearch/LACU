import argparse
from html import parser
import logging
import math
import os
import random
from pathlib import Path

import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from losses import get_unlearning_loss, get_preservation_loss, get_regularization_loss
from utils import load_prompts_from_file

import wandb

logger = get_logger(__name__) 

def parse_args():
    parser = argparse.ArgumentParser(description="Continual Unlearning")
    
    # --- Model Paths ---
    parser.add_argument("--previous_model_path", type=str, required=True, help="Path to the a model(Previous saved model).")
    parser.add_argument("--output_dir", type=str, default="models", help="The output directory where the unlearned model will be saved.")

    # --- Unlearning & Preservation Config ---
    parser.add_argument("--forget_prompts_path", type=str, required=True, help="Path to a text file containing prompts for the concept to unlearn.")
    parser.add_argument("--retain_prompts_path", type=str, required=True, help="Path to a text file containing prompts for preservation.")
    parser.add_argument("--neutral_prompts_path", type=str, default=None, help="Path to a text file with neutral prompts corresponding line-by-line to forget prompts.")

    # --- Loss Configuration ---
    parser.add_argument("--unlearning_loss_type", type=str, default="l2", choices=["l2", "cosine", "classifier_guidance"], help="Type of loss for the unlearning component.")
    parser.add_argument("--preservation_mode", type=str, default="distillation", choices=["distillation", "replay"], help="Strategy for the preservation component.")
    parser.add_argument("--lambda_unlearn", type=float, default=1.0, help="Weight for the unlearning loss.")
    parser.add_argument("--lambda_preserve", type=float, default=1.0, help="Weight for the preservation loss.")
    parser.add_argument("--lambda_reg", type=float, default=1e-4, help="Weight for the parameter regularization loss.")

    # --- Training Hyperparameters ---
    parser.add_argument("--resolution", type=int, default=512, help="The resolution for input images.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--max_train_steps", type=int, default=500, help="Total number of training steps to perform.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass."
    )

    # --- Sampler & Replay Config ---
    parser.add_argument("--num_inference_steps_replay", type=int, default=30, help="Number of DDIM steps for generating replay images.")

    # --- Misc ---
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Whether to use mixed precision.")
    parser.add_argument("--report_to", type=str, default=None, help='The integration to report the results and logs to.')

    parser.add_argument("--wandb_project", type=str, default="CUL")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default="", help='Comma-separated tags')
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online","offline","disabled"])

    # --- Unlearning timestep sampling ---
    parser.add_argument(
        "--unlearning_t_max", type=int, default=999,
        help=("Maximum timestep index (inclusive) to sample during UNLEARNING. "
            "Examples: 50, 100, 150, 300. If None, uses the scheduler's max (e.g., 999).")
    )
    
    args = parser.parse_args()
    
    if args.neutral_prompts_path is None:
        raise ValueError("You must provide --neutral_prompts_path.")

    return args

def make_replay_scheduler(base_model_path: str):
    return DDIMScheduler.from_pretrained(base_model_path, subfolder="scheduler")

def sample_timesteps(bs, T, device, t_max=999):
    t_max = int(T - 1 if t_max is None else min(t_max, T - 1))
    return torch.randint(0, t_max + 1, (bs,), device=device)

def clip_mean_similarity(clip_model, clip_processor, images, text, device):
    # images: list of PIL images; text: str
    inputs = clip_processor(
        images=images,
        text=[text] * len(images),
        return_tensors="pt",
        padding=True
    ).to(device)
    out = clip_model(**inputs)
    img = out.image_embeds
    txt = out.text_embeds
    img = img / img.norm(dim=-1, keepdim=True)
    txt = txt / txt.norm(dim=-1, keepdim=True)
    sim = (img * txt).sum(dim=-1)  # cosine similarity
    return sim.mean().item()

def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=os.path.join(args.output_dir, "logs")
    )

    # Everything you want pinned to the run config (add anything else you like)
    tracker_config = {
        "lambda_unlearn": args.lambda_unlearn,
        "lambda_preserve": args.lambda_preserve,
        "lambda_reg": args.lambda_reg,
        "learning_rate": args.learning_rate,
        "max_train_steps": args.max_train_steps,
        "unlearning_loss_type": args.unlearning_loss_type,
        "preservation_mode": args.preservation_mode,
        "num_inference_steps_replay": args.num_inference_steps_replay
    }

    init_kwargs = {}
    if args.report_to == "wandb":
        # These flow straight into wandb.init(**init_kwargs["wandb"])
        init_kwargs["wandb"] = {
            "entity": args.wandb_entity,
            "name": args.wandb_run_name,
            "group": args.wandb_group,
            "tags": [t for t in args.wandb_tags.split(",") if t],
            "mode": args.wandb_mode,  # online/offline/disabled
        }

    # This attaches W&B/TensorBoard trackers; for W&B the "project name" is wandb_project.
    accelerator.init_trackers(
        project_name=args.wandb_project if "wandb"==args.report_to else "CUL",
        config=tracker_config,
        init_kwargs=init_kwargs
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        logging_file = os.path.join(args.output_dir, "training.log")
        # Modify this part:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[ # Add this handlers list
                logging.StreamHandler(), # This keeps printing to the console
                logging.FileHandler(logging_file) # This saves to a file
            ]
        )

    # --- Load Models and Tokenizer ---
    tokenizer = CLIPTokenizer.from_pretrained(args.previous_model_path, subfolder="tokenizer", )
    text_encoder = CLIPTextModel.from_pretrained(args.previous_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.previous_model_path, subfolder="vae")
    
    # Student Model (the one we train)
    student_unet = UNet2DConditionModel.from_pretrained(args.previous_model_path, subfolder="unet")
    
    # Previous Task Teacher (frozen, for unlearning loss)
    previous_task_unet = UNet2DConditionModel.from_pretrained(args.previous_model_path, subfolder="unet")
    
    # Golden Teacher (frozen, for preservation loss)
    golden_teacher_unet = UNet2DConditionModel.from_pretrained(args.previous_model_path, subfolder="unet")

    # Freeze all teacher models and non-UNet components
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    previous_task_unet.requires_grad_(False)
    golden_teacher_unet.requires_grad_(False)

    # --- Setup Optimizer and Scheduler ---
    optimizer = torch.optim.AdamW(
        student_unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.previous_model_path, subfolder="scheduler")

    replay_scheduler = make_replay_scheduler(args.previous_model_path)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # --- Prepare for training with Accelerate ---
    student_unet, optimizer, lr_scheduler = accelerator.prepare(
        student_unet, optimizer, lr_scheduler
    )

    previous_task_unet.to(accelerator.device)
    golden_teacher_unet.to(accelerator.device)
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)

    # --- Load Prompts ---
    retain_prompts = load_prompts_from_file(args.retain_prompts_path)
    forget_prompts = load_prompts_from_file(args.forget_prompts_path)
    logger.info(f"Loaded {len(retain_prompts)} prompts for preservation.")
    logger.info(f"Loaded {len(forget_prompts)} prompts for the concept to unlearn.")

    neutral_prompts = load_prompts_from_file(args.neutral_prompts_path)
    logger.info(f"Loaded {len(neutral_prompts)} prompts for neutral.")
    if len(forget_prompts)!= len(neutral_prompts):
        raise ValueError("The forget prompts file and neutral prompts file must have the same number of lines.")

    # --- Setup Generative Replay Pipeline ---
    replay_pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=previous_task_unet,
        scheduler=replay_scheduler,
        safety_checker=None,
        feature_extractor=None,
    ).to(accelerator.device)
    replay_pipeline.set_progress_bar_config(disable=True)

    # --- Training Loop ---
    global_step = 0
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    while global_step < args.max_train_steps:
        student_unet.train()
        with accelerator.accumulate(student_unet):
            # 1. PRESERVATION STEP
            with torch.no_grad():
                current_retain_prompts = [random.choice(retain_prompts) for _ in range(args.train_batch_size)]
                replay_images = replay_pipeline(
                    prompt=current_retain_prompts, 
                    num_inference_steps=args.num_inference_steps_replay,
                    height=args.resolution,
                    width=args.resolution,
                    output_type="pt",
                    guidance_scale=1.0,
                    eta=0.0,  # 
                ).images
                replay_latents = vae.encode(replay_images).latent_dist.sample() * vae.config.scaling_factor
                noise_preserve = torch.randn_like(replay_latents)
                timesteps_preserve = torch.randint(0, noise_scheduler.config.num_train_timesteps, (args.train_batch_size,), device=accelerator.device).long()
                noisy_latents_preserve = noise_scheduler.add_noise(replay_latents, noise_preserve, timesteps_preserve)
                retain_input_ids = tokenizer(current_retain_prompts, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
                retain_embeds = text_encoder(retain_input_ids.to(accelerator.device)).last_hidden_state

            student_pred_preserve = student_unet(noisy_latents_preserve, timesteps_preserve, retain_embeds).sample
            golden_teacher_pred_preserve = golden_teacher_unet(noisy_latents_preserve, timesteps_preserve, retain_embeds).sample
        
            loss_preserve = get_preservation_loss(
                student_pred_preserve, 
                golden_teacher_pred_preserve, 
                noise_preserve, 
                mode=args.preservation_mode
            )

            # 2. UNLEARNING STEP
            with torch.no_grad():
                latents_unlearn = torch.randn_like(replay_latents)                
                # Sample paired forget/neutral prompts
                indices = [random.randint(0, len(forget_prompts) - 1) for _ in range(args.train_batch_size)]
                current_forget_prompts = [forget_prompts[i] for i in indices]

                gen_prompts = current_forget_prompts

                forget_images = replay_pipeline(
                    prompt=gen_prompts,
                    num_inference_steps=args.num_inference_steps_replay,
                    height=args.resolution, width=args.resolution,
                    output_type="pt",
                    guidance_scale=1.0,
                    eta=0.0,
                ).images
                generated_latents_unlearn = vae.encode(forget_images).latent_dist.sample() * vae.config.scaling_factor

                latents_unlearn = generated_latents_unlearn                
                current_neutral_prompts = [neutral_prompts[i] for i in indices]

                noise_unlearn = torch.randn_like(latents_unlearn)
                timesteps_unlearn = sample_timesteps(
                    bs=args.train_batch_size,
                    T=noise_scheduler.config.num_train_timesteps,
                    device=accelerator.device,
                    t_max=args.unlearning_t_max,
                )
                noisy_latents_unlearn = noise_scheduler.add_noise(latents_unlearn, noise_unlearn, timesteps_unlearn)

                forget_input_ids = tokenizer(current_forget_prompts, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
                neutral_input_ids = tokenizer(current_neutral_prompts, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").input_ids

                forget_embeds = text_encoder(forget_input_ids.to(accelerator.device)).last_hidden_state
                neutral_embeds = text_encoder(neutral_input_ids.to(accelerator.device)).last_hidden_state

            student_pred_unlearn = student_unet(noisy_latents_unlearn, timesteps_unlearn, forget_embeds).sample
            previous_task_pred_neutral = previous_task_unet(noisy_latents_unlearn, timesteps_unlearn, neutral_embeds).sample

            loss_unlearn = get_unlearning_loss(
                student_pred_unlearn, 
                previous_task_pred_neutral, 
                loss_type=args.unlearning_loss_type
            )

            # 3. REGULARIZATION STEP
            student_raw = accelerator.unwrap_model(student_unet)
            loss_reg = get_regularization_loss(student_raw, previous_task_unet)

            # 4. COMBINE LOSSES AND UPDATE
            total_loss = (
                args.lambda_preserve * loss_preserve + 
                args.lambda_unlearn * loss_unlearn + 
                args.lambda_reg * loss_reg
            ) 

            accelerator.backward(total_loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(student_unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                if accelerator.is_main_process:
                    progress_bar.update(1)

            if accelerator.is_main_process and accelerator.sync_gradients:
                lr_scalar = optimizer.param_groups[0]["lr"]  # scalar, not a list
                logs = {
                    "loss": float(total_loss.detach().item()),
                    "loss_preserve": float(loss_preserve.detach().item()),
                    "loss_unlearn": float(loss_unlearn.detach().item()),
                    "loss_reg": float(loss_reg.detach().item()),
                    "lr": float(lr_scalar),
                    "unlearning_timestep_sampling": (
                        f"[0, {args.unlearning_t_max}] inclusive; "                    ),
                }
                accelerator.log(logs, step=global_step)   # this feeds W&B/TB

                trigger_extra_steps = {500}
                if global_step > 0 and ((global_step+1) % 400 == 0 or (global_step + 1) in trigger_extra_steps):
                    # Saving checkpoints
                    ckpt_dir = os.path.join(args.output_dir, "checkpoints", f"step_{global_step+1}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    logger.info(f"Saving checkpoint to {ckpt_dir}")

                    # Build a full pipeline so the checkpoint is plug-and-play.
                    unwrapped_unet = accelerator.unwrap_model(student_unet)
                    ckpt_pipe = StableDiffusionPipeline.from_pretrained(
                        args.previous_model_path,
                        unet=unwrapped_unet,
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        scheduler=noise_scheduler,
                        safety_checker=None,
                        feature_extractor=None,
                    ).to(accelerator.device)

                    ckpt_pipe.save_pretrained(ckpt_dir)
                    del ckpt_pipe
                    torch.cuda.empty_cache()

        if global_step >= args.max_train_steps:
            break

    # --- Save the final model ---
    accelerator.wait_for_everyone()

    accelerator.end_training()

if __name__ == "__main__":
    main()
