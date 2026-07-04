import argparse
import logging
import os
import random

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from losses import get_unlearning_loss, get_preservation_loss, get_regularization_loss
from utils import load_prompts_from_file

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Locality-Aware Continual Unlearning for Stable Diffusion.")

    # --- Model Paths ---
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="HuggingFace model id or local Diffusers checkpoint to use as the current base model.",
    )
    parser.add_argument(
        "--latest_unlearned_checkpoint_path",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output_dir", type=str, default="models", help="The output directory where the unlearned model will be saved.")

    # --- Unlearning & Preservation Config ---
    parser.add_argument("--forget_prompts_path", type=str, required=True, help="Path to a text file containing prompts for the concept to unlearn.")
    parser.add_argument("--retain_prompts_path", type=str, required=True, help="Path to a text file containing prompts for preservation.")
    parser.add_argument("--map_prompts_path", type=str, required=True, help="Path to score-selected mapping prompts, aligned with forget prompts.")
    # --- Loss Configuration ---
    parser.add_argument("--lambda_unlearn", type=float, default=1.0, help="Weight for the unlearning loss.")
    parser.add_argument("--lambda_preserve", type=float, default=1.0, help="Weight for the preservation loss.")
    parser.add_argument("--lambda_reg", type=float, default=1e-4, help="Weight for the parameter regularization loss.")

    # --- Training Hyperparameters ---
    parser.add_argument("--resolution", type=int, default=512, help="The resolution for input images.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--max_train_steps", type=int, default=500, help="Total number of training steps to perform.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup", help="Learning-rate scheduler.")
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

    parser.add_argument("--num_inference_steps_replay", type=int, default=20, help="Number of DDIM steps for generating replay images.")

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
        "--unlearning_t_min", type=int, default=0,
        help="Minimum timestep index (inclusive) to sample during UNLEARNING."
    )
    parser.add_argument(
        "--unlearning_t_max", type=int, default=999,
        help=("Maximum timestep index (inclusive) to sample during UNLEARNING. "
            "Examples: 50, 100, 150, 300. If None, uses the scheduler's max (e.g., 999).")
    )

    return parser.parse_args()


def make_replay_scheduler(base_model_path: str):
    return DDIMScheduler.from_pretrained(base_model_path, subfolder="scheduler")

def sample_timesteps(bs, T, device, t_min=0, t_max=None):
    """
    Sample diffusion timesteps for training.
    - [t_min, t_max] are INCLUSIVE bounds (int indices).
    """
    t_min = max(0, int(t_min))
    t_max = int(T - 1 if t_max is None else min(t_max, T - 1))
    if t_min > t_max:
        raise ValueError(f"Invalid timestep range: t_min={t_min} > t_max={t_max} (T={T})")

    # Uniform in [t_min, t_max] (inclusive)
    return torch.randint(t_min, t_max + 1, (bs,), device=device)

def main():
    args = parse_args()
    if (
        args.latest_unlearned_checkpoint_path is not None
        and args.latest_unlearned_checkpoint_path != args.pretrained_model_name_or_path
    ):
        raise ValueError(
            "--latest_unlearned_checkpoint_path is deprecated. Pass the current base "
            "checkpoint via --pretrained_model_name_or_path instead."
        )
    model_path = args.pretrained_model_name_or_path

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=os.path.join(args.output_dir, "logs")
    )

    tracker_config = {
        "lambda_unlearn": args.lambda_unlearn,
        "lambda_preserve": args.lambda_preserve,
        "lambda_reg": args.lambda_reg,
        "learning_rate": args.learning_rate,
        "max_train_steps": args.max_train_steps,
        "num_inference_steps_replay": args.num_inference_steps_replay,
        "unlearning_latent_source": "generated",
        "preservation": "distillation",
    }

    init_kwargs = {}
    if args.report_to == "wandb":
        init_kwargs["wandb"] = {
            "entity": args.wandb_entity,
            "name": args.wandb_run_name,
            "group": args.wandb_group,
            "tags": [t for t in args.wandb_tags.split(",") if t],
            "mode": args.wandb_mode,
        }

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
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(logging_file)
            ]
        )

    # --- Load Models and Tokenizer ---
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")

    # Student M_t starts from the previous continual checkpoint M_{t-1}.
    # LACU only updates the UNet; text encoder and VAE stay fixed.
    student_unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")

    # Frozen teacher M_{t-1}. It provides both score-space targets:
    # 1) local unlearning target: teacher(for selected mapping prompt)
    # 2) local replay target: teacher(for nearby retain prompts)
    # A separate base_teacher_unet is unnecessary here because both roles use
    # the same previous-step checkpoint in the released LACU training flow.
    previous_task_unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")

    # Freeze all teacher/non-UNet components. Gradients should only update the
    # student UNet so the optimization matches the paper losses directly.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    previous_task_unet.requires_grad_(False)

    # --- Setup Optimizer and Scheduler ---
    optimizer = torch.optim.AdamW(
        student_unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    replay_scheduler = make_replay_scheduler(model_path)

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
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)

    # --- Load Prompts ---
    retain_prompts = load_prompts_from_file(args.retain_prompts_path)
    forget_prompts = load_prompts_from_file(args.forget_prompts_path)
    logger.info(f"Loaded {len(retain_prompts)} prompts for preservation.")
    logger.info(f"Loaded {len(forget_prompts)} prompts for the concept to unlearn.")

    map_prompts = load_prompts_from_file(args.map_prompts_path)
    logger.info(f"Loaded {len(map_prompts)} score-selected mapping prompts.")
    if len(forget_prompts) != len(map_prompts):
        raise ValueError("The forget prompts file and mapping prompts file must have the same number of lines.")

    # --- Setup Generative Replay Pipeline ---
    # Local replay samples latents from M_{t-1} for related retain prompts.
    # The preservation loss then asks the student to keep the same score
    # predictions on those nearby concepts, limiting locality damage.
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
            # 1. LOCAL PRESERVATION / REPLAY
            # Related retain prompts are selected before training by trajectory
            # similarity. Distilling M_{t-1} on these prompts protects concepts
            # close to the erased one, which is the locality term in LACU.
            with torch.no_grad():
                current_retain_prompts = [random.choice(retain_prompts) for _ in range(args.train_batch_size)]
                replay_latents = replay_pipeline(
                    prompt=current_retain_prompts,
                    num_inference_steps=args.num_inference_steps_replay,
                    height=args.resolution,
                    width=args.resolution,
                    output_type="latent",
                    guidance_scale=1.0,
                    eta=0.0,
                ).images
                noise_preserve = torch.randn_like(replay_latents)
                timesteps_preserve = torch.randint(0, noise_scheduler.config.num_train_timesteps, (args.train_batch_size,), device=accelerator.device).long()
                noisy_latents_preserve = noise_scheduler.add_noise(replay_latents, noise_preserve, timesteps_preserve)
                retain_input_ids = tokenizer(current_retain_prompts, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
                retain_embeds = text_encoder(retain_input_ids.to(accelerator.device)).last_hidden_state

            student_pred_preserve = student_unet(noisy_latents_preserve, timesteps_preserve, retain_embeds).sample
            teacher_pred_preserve = previous_task_unet(noisy_latents_preserve, timesteps_preserve, retain_embeds).sample

            loss_preserve = get_preservation_loss(student_pred_preserve, teacher_pred_preserve)

            # 2. LOCAL UNLEARNING
            # Each forget prompt is paired with a score-selected mapping prompt.
            # On the same noisy latent, the student conditioned on the forget
            # prompt is trained to match M_{t-1} conditioned on the mapping
            # prompt. This moves the erased concept locally instead of forcing
            # every prompt toward one global anchor.
            with torch.no_grad():
                # Sample paired forget/mapping prompts
                indices = [random.randint(0, len(forget_prompts) - 1) for _ in range(args.train_batch_size)]
                current_forget_prompts = [forget_prompts[i] for i in indices]
                current_map_prompts = [map_prompts[i] for i in indices]
                latents_unlearn = replay_pipeline(
                    prompt=current_forget_prompts,
                    num_inference_steps=args.num_inference_steps_replay,
                    height=args.resolution,
                    width=args.resolution,
                    output_type="latent",
                    guidance_scale=1.0,
                    eta=0.0,
                ).images

                noise_unlearn = torch.randn_like(latents_unlearn)
                timesteps_unlearn = sample_timesteps(
                    bs=args.train_batch_size,
                    T=noise_scheduler.config.num_train_timesteps,
                    device=accelerator.device,
                    t_min=args.unlearning_t_min,
                    t_max=args.unlearning_t_max,
                )
                noisy_latents_unlearn = noise_scheduler.add_noise(latents_unlearn, noise_unlearn, timesteps_unlearn)

                forget_input_ids = tokenizer(current_forget_prompts, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
                forget_embeds = text_encoder(forget_input_ids.to(accelerator.device)).last_hidden_state

                map_input_ids = tokenizer(current_map_prompts, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
                map_embeds = text_encoder(map_input_ids.to(accelerator.device)).last_hidden_state

            student_pred_unlearn = student_unet(noisy_latents_unlearn, timesteps_unlearn, forget_embeds).sample
            previous_task_pred_map = previous_task_unet(noisy_latents_unlearn, timesteps_unlearn, map_embeds).sample

            loss_unlearn = get_unlearning_loss(student_pred_unlearn, previous_task_pred_map)

            # 3. PARAMETER REGULARIZATION
            # This lightweight L2 penalty keeps M_t close to M_{t-1}, reducing
            # accumulated drift across the continual deletion sequence.
            student_raw = accelerator.unwrap_model(student_unet)
            loss_reg = get_regularization_loss(student_raw, previous_task_unet)

            # 4. COMBINE LACU OBJECTIVE TERMS AND UPDATE STUDENT
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
                        f"[{args.unlearning_t_min}, {args.unlearning_t_max}] inclusive"
                    ),
                }
                accelerator.log(logs, step=global_step)   # this feeds W&B/TB

                trigger_extra_steps = {100, 500}
                if global_step > 0 and (global_step % 200 == 0 or global_step in trigger_extra_steps):
                    ckpt_dir = os.path.join(args.output_dir, "checkpoints", f"step_{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    logger.info(f"Saving checkpoint to {ckpt_dir}")

                    # Build a full pipeline so the checkpoint is plug-and-play.
                    unwrapped_unet = accelerator.unwrap_model(student_unet)
                    ckpt_pipe = StableDiffusionPipeline.from_pretrained(
                        model_path,
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
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(student_unet)
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            unet=unwrapped_unet
        )
        final_checkpoint = os.path.join(args.output_dir, "checkpoints", f"step_{global_step}")
        pipeline.save_pretrained(final_checkpoint)
        logger.info(f"Unlearned model saved to {final_checkpoint}")

    accelerator.end_training()

if __name__ == "__main__":
    main()
