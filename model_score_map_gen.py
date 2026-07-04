#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-Output Score-based Candidate Selection for Concept Unlearning.

Instead of CLIP text-text similarity, this script uses the *target diffusion
model itself* to score candidate mapping prompts.  Two modes are provided:

MODE 1  – Discrete candidate scoring  (`--mode discrete`)
    For each (source_prompt, candidate_prompt) pair the script:
      1. Encodes both prompts into text embeddings.
      2. Samples K random (latent, timestep) pairs.
      3. Runs the UNet once per prompt to get noise predictions.
      4. Computes MSE between the two noise predictions.
    The candidate whose average MSE across K samples is *lowest* is
    selected – i.e. it induces the most similar denoising trajectory
    to the forget prompt while no longer containing the concept.

MODE 2  – Continuous embedding optimisation  (`--mode continuous`)
    Instead of picking from a fixed candidate set, this mode directly
    *optimises* a text embedding to minimise the noise-prediction MSE
    w.r.t. the forget prompt, while being regularised to stay close to
    the nearest discrete candidate (anchor).  The optimised embedding
    is then projected back to the closest token-sequence by greedy
    decoding, giving a readable prompt.

Candidates are loaded from the pre-generated `candidates_clip.json`.
Output is saved as `map_model.csv` (same format as `map_100.csv`).
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


# ────────────────────────────────────────────────────────────────
# Nudity-specific banned words (body parts flagged by safety classifiers)
# ────────────────────────────────────────────────────────────────
NUDITY_BANNED_WORDS = [
    "breast", "breasts", "nipple", "nipples", "chest",
    "genitalia", "genital", "genitals", "groin", "penis", "vagina", "vulva",
    "buttock", "buttocks", "butt", "anus", "anal",
    "feet", "foot", "toes", "barefoot",
    "belly", "midriff", "stomach", "navel", "abdomen",
    "armpit", "armpits", "underarm", "underarms",
    "nude", "naked", "bare", "topless", "shirtless",
    "nudity", "undressed", "unclothed", "exposed",
]


def _contains_banned(text: str, banned: list) -> bool:
    """Check if text contains any banned word (case-insensitive)."""
    low = text.lower()
    return any(b in low for b in banned)


# ────────────────────────────────────────────────────────────────
# Helpers (shared with prompt_gen.py / clip_map_gen.py)
# ────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_{2,}", "_", text).strip("_")
    return text or "concept"


def load_candidates_json(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ────────────────────────────────────────────────────────────────
# Diffusion model wrapper (lazy-loaded, shared across concepts)
# ────────────────────────────────────────────────────────────────

class DiffusionScorer:
    """Wraps a Stable Diffusion UNet + text encoder + scheduler for
    scoring candidate prompts via noise-prediction similarity."""

    def __init__(
        self,
        model_path: str,
        device: str = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._loaded = False

        # Placeholders
        self.unet = None
        self.transformer = None
        self.pipeline = None
        self.text_encoder = None
        self.tokenizer = None
        self.text_encoder_2 = None
        self.tokenizer_2 = None
        self.text_encoder_3 = None
        self.tokenizer_3 = None
        self.vae = None
        self.scheduler = None
        self.is_sdxl = False
        self.is_sd3 = False
        self.vae_scale_factor = 8

    def load(self):
        if self._loaded:
            return
        if self._looks_like_sd3_model():
            self._load_sd3()
            return

        from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
        from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

        print(f"[Model] Loading from {self.model_path} on {self.device} ({self.dtype}) …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, subfolder="tokenizer"
        )
        self.text_encoder = (
            CLIPTextModel.from_pretrained(self.model_path, subfolder="text_encoder")
            .to(self.device, dtype=self.dtype)
            .eval()
        )
        self.unet = (
            UNet2DConditionModel.from_pretrained(self.model_path, subfolder="unet")
            .to(self.device, dtype=self.dtype)
            .eval()
        )
        self.vae = AutoencoderKL.from_pretrained(self.model_path, subfolder="vae")
        if hasattr(self.vae.config, "block_out_channels"):
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.scheduler = DDPMScheduler.from_pretrained(
            self.model_path, subfolder="scheduler"
        )
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

        # SDXL adds a second tokenizer/text encoder and UNet extra conditions.
        addition_embed_type = getattr(self.unet.config, "addition_embed_type", None)
        self.is_sdxl = addition_embed_type == "text_time"
        if self.is_sdxl:
            self.tokenizer_2 = AutoTokenizer.from_pretrained(
                self.model_path, subfolder="tokenizer_2"
            )
            self.text_encoder_2 = (
                CLIPTextModelWithProjection.from_pretrained(
                    self.model_path, subfolder="text_encoder_2"
                )
                .to(self.device, dtype=self.dtype)
                .eval()
            )
            self.text_encoder_2.requires_grad_(False)

        self._loaded = True
        print(f"[Model] Loaded. mode={'SDXL' if self.is_sdxl else 'SD/SD2'}")

    def _looks_like_sd3_model(self) -> bool:
        path = Path(self.model_path)
        lower = str(self.model_path).lower()
        return (
            (path.exists() and (path / "transformer").exists())
            or "stable-diffusion-3" in lower
            or "sd3" in lower
            or "sdv3" in lower
        )

    def _load_sd3(self):
        from diffusers import StableDiffusion3Pipeline

        print(f"[Model] Loading SD3 from {self.model_path} on {self.device} ({self.dtype}) …")
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            self.model_path, torch_dtype=self.dtype
        ).to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)
        self.transformer = self.pipeline.transformer.eval()
        self.vae = self.pipeline.vae.eval()
        self.scheduler = self.pipeline.scheduler
        if hasattr(self.scheduler, "set_timesteps"):
            self.scheduler.set_timesteps(
                int(self.scheduler.config.num_train_timesteps), device=self.device
            )
        self.text_encoder = self.pipeline.text_encoder
        self.text_encoder_2 = self.pipeline.text_encoder_2
        self.text_encoder_3 = self.pipeline.text_encoder_3
        self.tokenizer = self.pipeline.tokenizer
        self.tokenizer_2 = self.pipeline.tokenizer_2
        self.tokenizer_3 = self.pipeline.tokenizer_3
        for module in (
            self.transformer,
            self.vae,
            self.text_encoder,
            self.text_encoder_2,
            self.text_encoder_3,
        ):
            if module is not None:
                module.requires_grad_(False)
        self.is_sd3 = True
        self._loaded = True
        print("[Model] Loaded. mode=SD3")

    # ── text encoding ──

    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """Encode text with the first text encoder → (B, seq_len, hidden_dim)."""
        self.load()
        toks = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return self.text_encoder(toks.input_ids.to(self.device))[0].to(self.dtype)

    @torch.no_grad()
    def encode_text_sdxl(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode SDXL prompts and return (prompt_embeds, pooled_prompt_embeds)."""
        self.load()
        if not self.is_sdxl:
            raise RuntimeError("encode_text_sdxl() called for a non-SDXL model.")

        prompt_embeds_list = []
        pooled_prompt_embeds = None
        for tokenizer, text_encoder in (
            (self.tokenizer, self.text_encoder),
            (self.tokenizer_2, self.text_encoder_2),
        ):
            toks = tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            out = text_encoder(
                toks.input_ids.to(self.device),
                output_hidden_states=True,
                return_dict=True,
            )
            prompt_embeds_list.append(out.hidden_states[-2])

            pooled_candidate = getattr(out, "text_embeds", None)
            if pooled_candidate is None:
                pooled_candidate = out[0]
            if pooled_candidate.ndim == 3:
                pooled_candidate = pooled_candidate[:, 0]
            pooled_prompt_embeds = pooled_candidate

        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1).to(self.dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(self.dtype)
        return prompt_embeds, pooled_prompt_embeds

    @torch.no_grad()
    def encode_text_sd3(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode SD3 prompts and return (prompt_embeds, pooled_prompt_embeds)."""
        self.load()
        if not self.is_sd3:
            raise RuntimeError("encode_text_sd3() called for a non-SD3 model.")
        prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt=prompts,
            prompt_2=prompts,
            prompt_3=prompts,
            device=torch.device(self.device),
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            max_sequence_length=256,
        )
        return prompt_embeds.to(self.dtype), pooled_prompt_embeds.to(self.dtype)

    def _default_latent_shape(self) -> Tuple[int, int, int, int]:
        model = self.transformer if self.is_sd3 else self.unet
        sample_size = getattr(model.config, "sample_size", 64)
        if isinstance(sample_size, (tuple, list)):
            h, w = int(sample_size[0]), int(sample_size[1])
        else:
            h = w = int(sample_size)
        in_channels = int(getattr(model.config, "in_channels", 4))
        return (1, in_channels, h, w)

    def _make_add_time_ids(self, latent: torch.Tensor) -> torch.Tensor:
        h = int(latent.shape[-2] * self.vae_scale_factor)
        w = int(latent.shape[-1] * self.vae_scale_factor)
        add_time_ids = [h, w, 0, 0, h, w]
        return torch.tensor([add_time_ids], device=self.device, dtype=self.dtype)

    def _sample_sd3_noisy(self, latent: torch.Tensor, noise: torch.Tensor):
        T = int(self.scheduler.config.num_train_timesteps)
        raw_t = torch.randint(0, T, (1,), device=self.device).long()
        step_index = (T - 1 - raw_t).clamp(0, T - 1).long()
        timestep = self.scheduler.timesteps.to(self.device)[step_index]
        sigma = self.scheduler.sigmas.to(device=self.device, dtype=self.dtype)[step_index]
        while sigma.ndim < latent.ndim:
            sigma = sigma.unsqueeze(-1)
        noisy = (1.0 - sigma) * latent + sigma * noise
        return noisy, timestep

    # ── forward through text encoder from raw token embeddings ──

    def _forward_from_token_embeds(
        self, token_embeds: torch.Tensor, attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Run raw token-level embeddings (shape (B, S, D) in the *input*
        embedding space, **before** position embeddings) through the CLIP
        text encoder's position-embedding layer, transformer encoder, and
        final layer-norm – bypassing CLIPTextTransformer.forward() which
        requires discrete input_ids.

        Returns the last-hidden-state (B, S, D) in the *output* hidden-state
        space, suitable for feeding into the UNet's cross-attention.

        NOTE: The causal mask is built with plain PyTorch so that the code
        works across all transformers versions (older versions lack the
        ``_create_4d_causal_attention_mask`` helper that was added in
        transformers ≥ 4.35).
        """
        text_model = self.text_encoder.text_model  # CLIPTextTransformer
        B, S, D = token_embeds.shape

        # 1. Add position embeddings (same as CLIPTextEmbeddings.forward
        #    when inputs_embeds is provided)
        hidden_states = text_model.embeddings(
            input_ids=None,
            position_ids=None,
            inputs_embeds=token_embeds,
        )

        # 2. Build causal attention mask with plain PyTorch
        #    Shape: (B, 1, S, S), upper-triangle = -inf (masked-out).
        #    This replicates what _create_4d_causal_attention_mask does.
        causal_attention_mask = torch.full(
            (S, S), fill_value=torch.finfo(hidden_states.dtype).min,
            device=hidden_states.device, dtype=hidden_states.dtype,
        )
        causal_attention_mask = torch.triu(causal_attention_mask, diagonal=1)
        causal_attention_mask = (
            causal_attention_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S)
        )

        # 3. Prepare explicit attention mask if given
        #    [bsz, seq_len] → [bsz, 1, 1, seq_len] with 0 / -inf
        if attention_mask is not None:
            inv_mask = 1.0 - attention_mask.to(hidden_states.dtype)
            inv_mask = inv_mask.unsqueeze(1).unsqueeze(1)  # (B,1,1,S)
            inv_mask = inv_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = inv_mask

        # 4. Encoder + final layer-norm
        encoder_outputs = text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        last_hidden = encoder_outputs[0]
        last_hidden = text_model.final_layer_norm(last_hidden)
        return last_hidden

    # ── noise-prediction scoring ──

    @torch.no_grad()
    def score_candidates_discrete(
        self,
        source_prompt: str,
        candidate_prompts: List[str],
        K: int = 8,
        seed: int = None,
        latent_shape: Optional[Tuple[int, ...]] = None,
    ) -> List[float]:
        """
        Score each candidate by average MSE between its noise prediction
        and the forget-prompt noise prediction, over K random
        (latent, timestep) draws.

        Lower MSE → more similar trajectory → better candidate.
        Returns list of *negative* MSEs (so higher = better, consistent
        with "select argmax").
        """
        self.load()
        device = self.device
        if latent_shape is None:
            latent_shape = self._default_latent_shape()

        if seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(seed)
        else:
            gen = None

        # Encode all prompts at once: [source, cand_1, …, cand_N]
        all_prompts = [source_prompt] + candidate_prompts
        if self.is_sd3:
            all_embeds, all_pooled = self.encode_text_sd3(all_prompts)
            src_embed = all_embeds[0:1]
            src_pooled = all_pooled[0:1]
            cand_embeds = all_embeds[1:]
            cand_pooled = all_pooled[1:]
            N = cand_embeds.shape[0]
        elif self.is_sdxl:
            all_embeds, all_pooled = self.encode_text_sdxl(all_prompts)
            src_embed = all_embeds[0:1]
            src_pooled = all_pooled[0:1]
            cand_embeds = all_embeds[1:]
            cand_pooled = all_pooled[1:]
            N = cand_embeds.shape[0]
        else:
            all_embeds = self.encode_text(all_prompts)  # (1+N, S, D)
            src_embed = all_embeds[0:1]       # (1, S, D)
            cand_embeds = all_embeds[1:]      # (N, S, D)
            N = cand_embeds.shape[0]

        accum_mse = torch.zeros(N, device=device)

        for _ in range(K):
            # Sample random latent & timestep
            latent = torch.randn(latent_shape, generator=gen, device="cpu").to(
                device, dtype=self.dtype
            )
            noise = torch.randn_like(latent)
            if self.is_sd3:
                noisy, t = self._sample_sd3_noisy(latent, noise)
            else:
                T = self.scheduler.config.num_train_timesteps
                t = torch.randint(0, T, (1,), device=device).long()
                noisy = self.scheduler.add_noise(latent, noise, t)

            # Source prediction
            if self.is_sd3:
                src_pred = self.transformer(
                    hidden_states=noisy,
                    timestep=t,
                    encoder_hidden_states=src_embed,
                    pooled_projections=src_pooled,
                    return_dict=True,
                ).sample
            elif self.is_sdxl:
                add_time_ids = self._make_add_time_ids(latent)
                src_pred = self.unet(
                    noisy,
                    t,
                    encoder_hidden_states=src_embed,
                    added_cond_kwargs={
                        "text_embeds": src_pooled,
                        "time_ids": add_time_ids,
                    },
                ).sample
            else:
                src_pred = self.unet(noisy, t, src_embed).sample  # (1,4,H,W)

            # Candidate predictions (loop to save VRAM vs. batching all)
            for j in range(N):
                c_embed = cand_embeds[j : j + 1]
                if self.is_sd3:
                    c_pooled = cand_pooled[j : j + 1]
                    c_pred = self.transformer(
                        hidden_states=noisy,
                        timestep=t,
                        encoder_hidden_states=c_embed,
                        pooled_projections=c_pooled,
                        return_dict=True,
                    ).sample
                elif self.is_sdxl:
                    c_pooled = cand_pooled[j : j + 1]
                    c_pred = self.unet(
                        noisy,
                        t,
                        encoder_hidden_states=c_embed,
                        added_cond_kwargs={
                            "text_embeds": c_pooled,
                            "time_ids": add_time_ids,
                        },
                    ).sample
                else:
                    c_pred = self.unet(noisy, t, c_embed).sample
                mse = F.mse_loss(c_pred, src_pred, reduction="mean")
                accum_mse[j] += mse

        avg_mse = accum_mse / K
        # Return negative MSE so that *higher* = better (argmax selection)
        return (-avg_mse).cpu().tolist()

    # ── helpers: tokenise → input-space embedding ──

    @torch.no_grad()
    def _get_input_embeds(self, prompts: List[str]) -> torch.Tensor:
        """Tokenise *prompts* and return the raw token embeddings from the
        CLIP input embedding table (i.e. **before** position embeddings or
        the transformer encoder).  Shape: (B, S, D) in input-embed space."""
        self.load()
        toks = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.device)
        embed_layer = self.text_encoder.get_input_embeddings()  # nn.Embedding
        return embed_layer(toks).to(self.dtype)  # (B, S, D)

    # ── continuous Gumbel-Softmax embedding optimisation ──

    def optimise_embedding(
        self,
        source_prompt: str,
        anchor_prompt: str,
        concept: str,
        K_per_step: int = 4,
        steps: int = 400,
        lr: float = 0.05,
        lambda_anchor: float = 0.05,
        lambda_concept: float = 5.0,
        latent_shape: Tuple[int, ...] = (1, 4, 64, 64),
        seed: int = 42,
    ) -> Tuple[str, float]:
        """
        Optimise a prompt in **discrete token space** using Gumbel-Softmax
        relaxation over the CLIP vocabulary.

        Instead of optimising a raw continuous embedding (which snaps back
        to the same tokens when projected), this method maintains a
        probability distribution over the vocabulary at each token position.
        Temperature-annealed Gumbel-Softmax gives differentiable sampling
        while gradually converging to hard token selections.

        Logits are initialised from cosine similarity between each anchor
        token's embedding and all vocab embeddings, so semantically similar
        tokens start with high probability and the optimiser can smoothly
        swap content words.

        The text encoder is temporarily cast to **float32** during
        optimisation to prevent gradient overflow through its 12
        transformer layers.

        Returns ``(decoded_text, final_loss)``.
        """
        self.load()
        if self.is_sdxl:
            raise NotImplementedError(
                "Continuous token optimization is not implemented for SDXL in this script. "
                "Use --mode discrete."
            )
        T = self.scheduler.config.num_train_timesteps
        device = self.device
        gen = torch.Generator(device="cpu").manual_seed(seed)

        # ── vocabulary & tokenisation setup ──
        max_len = self.tokenizer.model_max_length  # 77

        anchor_toks = self.tokenizer(
            [anchor_prompt], padding="max_length", truncation=True,
            max_length=max_len, return_tensors="pt",
        ).input_ids.to(device)  # (1, S)

        eos_id = self.tokenizer.eos_token_id  # 49407

        # Identify content token positions (skip BOS@0, EOS, PAD)
        anchor_ids = anchor_toks[0].tolist()
        eos_pos = max_len
        for i, tid in enumerate(anchor_ids):
            if i > 0 and tid == eos_id:
                eos_pos = i
                break
        content_mask = torch.zeros(max_len, dtype=torch.bool, device=device)
        content_mask[1:eos_pos] = True
        n_content = content_mask.sum().item()

        if n_content == 0:
            decoded = self.tokenizer.decode(anchor_ids, skip_special_tokens=True)
            return re.sub(r"\s+", " ", decoded).strip(), 0.0

        # ── frozen reference hidden states (computed in original fp16) ──
        src_hidden = self.encode_text([source_prompt])  # (1,S,D) fp16
        concept_hidden = self.encode_text([concept])    # (1,S,D) fp16

        # ── cast text encoder to float32 for gradient stability ──
        # Backprop through 12 CLIP transformer layers in fp16 causes
        # inf/NaN gradients; float32 prevents this.
        self.text_encoder.text_model.float()

        try:
            embed_table = self.text_encoder.get_input_embeddings()
            V = embed_table.num_embeddings   # ~49408

            # Get embedding weight AFTER float32 cast
            embed_weight = embed_table.weight  # (V, D) float32

            # ── initialise logits from cosine similarity ──
            with torch.no_grad():
                anchor_embed = embed_table(anchor_toks)  # (1, S, D) float32
                a_norm = F.normalize(anchor_embed, dim=-1)
                w_norm = F.normalize(embed_weight, dim=-1)
                init_logits = torch.matmul(a_norm, w_norm.T) * 20.0  # (1,S,V)

            logits = init_logits.clone().requires_grad_(True)

            # Hard one-hot for special positions – stays frozen
            special_onehot = F.one_hot(anchor_toks, V).float().to(device)

            optimiser = torch.optim.AdamW([logits], lr=lr)

            tau_start, tau_end = 1.5, 0.1
            best_loss = float("inf")
            best_logits = logits.data.clone()

            for step in range(steps):
                optimiser.zero_grad()
                frac = step / max(steps - 1, 1)
                tau = tau_start * (tau_end / tau_start) ** frac

                # ── Gumbel-Softmax → soft token distribution (1, S, V) ──
                soft = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)

                # Freeze special positions (BOS, EOS, PAD)
                mask_3d = content_mask.unsqueeze(0).unsqueeze(-1)
                effective = torch.where(mask_3d, soft, special_onehot)

                # Soft embedding in float32: (1,S,V) @ (V,D) → (1,S,D)
                soft_embed = torch.matmul(effective, embed_weight)

                # Forward through text encoder in float32
                opt_hidden = self._forward_from_token_embeds(soft_embed)

                # ── reconstruction loss (match source denoising) ──
                loss_recon = torch.tensor(0.0, device=device)
                for _ in range(K_per_step):
                    latent = torch.randn(
                        latent_shape, generator=gen, device="cpu"
                    ).to(device, dtype=self.dtype)
                    t = torch.randint(0, T, (1,), device=device).long()
                    noise = torch.randn_like(latent)
                    noisy = self.scheduler.add_noise(latent, noise, t)

                    with torch.no_grad():
                        src_pred = self.unet(noisy, t, src_hidden).sample

                    # Cast opt_hidden → fp16 for UNet; gradient flows back
                    # to float32 via the .half() autograd node.
                    opt_pred = self.unet(
                        noisy, t, opt_hidden.to(self.dtype)
                    ).sample
                    # Compute MSE in float32 for numerical stability
                    loss_recon = loss_recon + F.mse_loss(
                        opt_pred.float(), src_pred.float()
                    )

                loss_recon = loss_recon / K_per_step

                # ── anchor regularisation (cross-entropy in logit space) ──
                content_logits  = logits[0, content_mask]
                content_targets = anchor_toks[0, content_mask]
                loss_anchor = F.cross_entropy(content_logits, content_targets)

                # ── concept repulsion ──
                cos_sim = F.cosine_similarity(
                    opt_hidden.float().view(1, -1),
                    concept_hidden.float().view(1, -1),
                )
                loss_concept = cos_sim.mean()

                total_loss = (
                    loss_recon
                    + lambda_anchor * loss_anchor
                    + lambda_concept * loss_concept
                )

                total_loss.backward()

                # ── NaN guard: skip step if gradients are non-finite ──
                if logits.grad is not None and not torch.isfinite(logits.grad).all():
                    logits.grad.zero_()
                    continue

                # Zero gradient for non-content positions
                if logits.grad is not None:
                    logits.grad.data[0, ~content_mask] = 0.0

                torch.nn.utils.clip_grad_norm_([logits], max_norm=1.0)
                optimiser.step()

                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_logits = logits.data.clone()

        finally:
            # Always restore text encoder to original dtype
            self.text_encoder.text_model.to(self.dtype)

        # ── decode: argmax over logits → token IDs → text ──
        final_ids = best_logits[0].argmax(dim=-1)           # (S,)
        final_ids[0] = anchor_toks[0, 0]                    # restore BOS
        final_ids[eos_pos:] = anchor_toks[0, eos_pos:]      # restore EOS+PAD

        decoded = self.tokenizer.decode(
            final_ids.cpu().tolist(), skip_special_tokens=True,
        )
        decoded = re.sub(r"\s+", " ", decoded).strip()
        return decoded, best_loss

    # ── continuous mixture-of-candidates embedding optimisation ──

    def optimise_mixture_embedding(
        self,
        source_prompt: str,
        candidate_prompts: List[str],
        concept: str,
        K_per_step: int = 4,
        steps: int = 200,
        lr: float = 0.05,
        lambda_concept: float = 5.0,
        latent_shape: Tuple[int, ...] = (1, 4, 64, 64),
        seed: int = 42,
    ) -> Tuple[torch.Tensor, float, List[float]]:
        """
        Optimise a **learnable mixture** over K candidate text embeddings
        to find a target embedding whose UNet denoising trajectory best
        matches the forget-prompt trajectory.

        Instead of optimising raw vectors or discrete tokens, this method:
          1. Pre-computes output-space hidden states for all K candidates.
          2. Maintains a learnable weight vector ``w`` of size K.
          3. Computes ``e_target = sum_i softmax(w)_i * embed_i`` at each
             step.
          4. Minimises UNet noise-prediction MSE(e_target, e_source) plus
             a concept-repulsion term.

        The result is a continuous embedding that lives in the convex hull
        of the candidate embeddings – a "floating-point mixture" that
        cannot be expressed as a single text string but can be fed
        directly into the Teacher U-Net's cross-attention during training.

        Returns ``(e_target, final_loss, softmax_weights)``.
          - ``e_target``: Tensor of shape (1, S, D), dtype=self.dtype
          - ``final_loss``: float – best composite loss achieved
          - ``softmax_weights``: list of K floats summing to 1
        """
        self.load()
        if self.is_sdxl:
            raise NotImplementedError(
                "Continuous embedding-mixture optimization is not implemented for SDXL in this script. "
                "Use --mode discrete."
            )
        T = self.scheduler.config.num_train_timesteps
        device = self.device
        gen = torch.Generator(device="cpu").manual_seed(seed)
        K_cands = len(candidate_prompts)

        if K_cands == 0:
            raise ValueError("No candidate prompts provided.")

        # ── pre-compute all embeddings in the output hidden-state space ──
        with torch.no_grad():
            src_hidden = self.encode_text([source_prompt])        # (1,S,D) fp16
            concept_hidden = self.encode_text([concept])          # (1,S,D) fp16
            cand_hidden = self.encode_text(candidate_prompts)     # (K,S,D) fp16

        # Work in float32 for gradient stability
        src_hidden_f = src_hidden.float()           # (1,S,D)
        concept_hidden_f = concept_hidden.float()   # (1,S,D)
        cand_hidden_f = cand_hidden.float()         # (K,S,D)

        # ── learnable mixture weights (initialise uniform) ──
        w = torch.zeros(K_cands, device=device, requires_grad=True)

        optimiser = torch.optim.AdamW([w], lr=lr)

        best_loss = float("inf")
        best_w = w.data.clone()

        for step in range(steps):
            optimiser.zero_grad()

            # Softmax weights → mixture embedding
            alpha = F.softmax(w, dim=0)                          # (K,)
            # e_target = sum_i alpha_i * cand_hidden_i
            # alpha: (K,) → (K,1,1);  cand_hidden_f: (K,S,D)
            e_target_f = (alpha.unsqueeze(-1).unsqueeze(-1) * cand_hidden_f).sum(dim=0, keepdim=True)  # (1,S,D) float32

            # Cast to model dtype for UNet forward
            e_target_half = e_target_f.to(self.dtype)

            # ── reconstruction loss (match source denoising) ──
            loss_recon = torch.tensor(0.0, device=device)
            for _ in range(K_per_step):
                latent = torch.randn(
                    latent_shape, generator=gen, device="cpu"
                ).to(device, dtype=self.dtype)
                t_step = torch.randint(0, T, (1,), device=device).long()
                noise = torch.randn_like(latent)
                noisy = self.scheduler.add_noise(latent, noise, t_step)

                with torch.no_grad():
                    src_pred = self.unet(noisy, t_step, src_hidden).sample

                opt_pred = self.unet(noisy, t_step, e_target_half).sample
                loss_recon = loss_recon + F.mse_loss(
                    opt_pred.float(), src_pred.float()
                )

            loss_recon = loss_recon / K_per_step

            # ── concept repulsion ──
            cos_sim = F.cosine_similarity(
                e_target_f.view(1, -1),
                concept_hidden_f.view(1, -1),
            )
            loss_concept = cos_sim.mean()

            total_loss = loss_recon + lambda_concept * loss_concept

            total_loss.backward()

            # NaN guard
            if w.grad is not None and not torch.isfinite(w.grad).all():
                w.grad.zero_()
                continue

            torch.nn.utils.clip_grad_norm_([w], max_norm=1.0)
            optimiser.step()

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_w = w.data.clone()

        # ── compute final target embedding from best weights ──
        with torch.no_grad():
            final_alpha = F.softmax(best_w, dim=0)
            e_final = (final_alpha.unsqueeze(-1).unsqueeze(-1) * cand_hidden_f).sum(dim=0, keepdim=True)
            e_final = e_final.to(self.dtype)  # (1,S,D) back to model dtype

        return e_final, best_loss, final_alpha.cpu().tolist()

    # ── decode embedding back to a readable prompt ──

    @torch.no_grad()
    def decode_embedding_to_text(self, embed: torch.Tensor) -> str:
        """
        Greedy-decode an optimised embedding back to a readable prompt
        by finding the nearest token in the CLIP text encoder's embedding
        table for each position.

        This is an approximation – the result is a coarse projection of the
        continuous optimised embedding onto the discrete token space.
        """
        self.load()
        # Get the token embedding matrix from CLIP text encoder
        token_embeds = self.text_encoder.get_input_embeddings().weight  # (V, D)
        token_embeds = token_embeds.to(self.dtype)
        # embed: (1, S, D)
        e = embed.squeeze(0)  # (S, D)
        # Normalise for cosine similarity
        e_norm = e / (e.norm(dim=-1, keepdim=True) + 1e-8)
        t_norm = token_embeds / (token_embeds.norm(dim=-1, keepdim=True) + 1e-8)
        sims = e_norm @ t_norm.T  # (S, V)
        token_ids = sims.argmax(dim=-1).cpu().tolist()  # list of int

        # Decode, stripping special tokens and padding
        decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        # Clean up repeated spaces / artefacts
        decoded = re.sub(r"\s+", " ", decoded).strip()
        return decoded


# ────────────────────────────────────────────────────────────────
# Per-concept processing
# ────────────────────────────────────────────────────────────────

def _score_all_entries(
    data: List[dict],
    scorer: DiffusionScorer,
    K: int,
    seed: int,
    banned_words: List[str] = None,
) -> Tuple[List[dict], List[dict]]:
    """Score all candidate entries once. Returns (results, detailed)."""
    results = []
    detailed = []

    for entry in tqdm(data, desc="scoring"):
        prompt_id = entry["id"]
        source = entry["source"]
        cands = entry["candidates"]

        cand_texts = [c["prompt"] for c in cands]
        valid_idx = [i for i, t in enumerate(cand_texts)
                     if t.strip()
                     and not (banned_words and _contains_banned(t, banned_words))]
        valid_texts = [cand_texts[i] for i in valid_idx]

        if not cands or not valid_idx:
            results.append({
                "id": prompt_id, "mapped": "", "mapping_type": "",
                "note": "no valid candidates", "score": 0.0,
            })
            detailed.append({
                "id": prompt_id, "source": source,
                "candidates": [
                    {"prompt": c["prompt"], "mapping_type": c.get("mapping_type", "replace"),
                     "replacement": c.get("replacement", ""), "clip_score": c.get("clip_score"),
                     "model_score": None} for c in cands
                ],
                "selected_model": None, "selected_clip": entry.get("selected"),
            })
            continue

        scores = scorer.score_candidates_discrete(
            source_prompt=source,
            candidate_prompts=valid_texts,
            K=K,
            seed=seed,
        )

        # Build detailed per-candidate scores
        scored_cands = []
        score_map = dict(zip(valid_idx, scores))
        for j, c in enumerate(cands):
            sc = score_map.get(j)
            scored_cands.append({
                "prompt": c["prompt"],
                "mapping_type": c.get("mapping_type", "replace"),
                "replacement": c.get("replacement", ""),
                "clip_score": c.get("clip_score"),
                "model_score": round(sc, 6) if sc is not None else None,
            })

        best_local = max(range(len(scores)), key=lambda k: scores[k])
        best_global = valid_idx[best_local]
        best_cand = cands[best_global]
        best_score = scores[best_local]

        m_type = best_cand.get("mapping_type", "replace")
        repl = best_cand.get("replacement", "")
        note = f"{m_type}: {repl} (model_score={best_score:.6f})"

        results.append({
            "id": prompt_id,
            "mapped": best_cand["prompt"],
            "mapping_type": m_type,
            "note": note,
            "score": round(best_score, 6),
        })

        detailed.append({
            "id": prompt_id, "source": source,
            "candidates": scored_cands,
            "selected_model": best_global + 1,
            "selected_clip": entry.get("selected"),
        })

    return results, detailed


def process_concept_discrete(
    concept: str,
    prompts_root: Path,
    scorer: DiffusionScorer,
    K: int = 8,
    seed: int = 42,
    banned_words: List[str] = None,
    candidates_filename: str = "candidates_clip.json",
    output_csv_filename: str = "map_model.csv",
    detail_json_filename: str = "candidates_model_scores.json",
):
    """Discrete argmax scoring over pre-generated candidates."""
    slug = slugify(concept)
    concept_dir = prompts_root / slug
    cand_file = Path(candidates_filename)
    if not cand_file.is_absolute():
        cand_file = concept_dir / cand_file

    if not cand_file.exists():
        print(f"[SKIP] {concept} – {cand_file} not found.")
        return

    data = load_candidates_json(cand_file)
    print(f"\n{'='*60}")
    print(f"[CONCEPT] {concept}  ({len(data)} prompts, discrete mode)")
    print(f"{'='*60}")

    results, detailed = _score_all_entries(data, scorer, K, seed, banned_words=banned_words)

    # Save selected mapping CSV
    out_csv = Path(output_csv_filename)
    if not out_csv.is_absolute():
        out_csv = concept_dir / out_csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "prompt", "mapping_type", "note"])
        for r in results:
            w.writerow([r["id"], r["mapped"], r["mapping_type"], r["note"]])
    print(f"  [SAVED] {out_csv}")

    # Save detailed JSON with all scores
    detail_json = Path(detail_json_filename)
    if not detail_json.is_absolute():
        detail_json = concept_dir / detail_json
    with open(detail_json, "w", encoding="utf-8") as f:
        json.dump(detailed, f, ensure_ascii=False, indent=2)
    print(f"  [SAVED] {detail_json}")

    # Summary stats
    score_vals = [r["score"] for r in results if r["score"] != 0.0]
    if score_vals:
        print(f"  [STATS] Avg score: {sum(score_vals)/len(score_vals):.6f}  "
              f"Min: {min(score_vals):.6f}  Max: {max(score_vals):.6f}")


def process_concept_continuous(
    concept: str,
    prompts_root: Path,
    scorer: DiffusionScorer,
    K_per_step: int = 4,
    optim_steps: int = 80,
    lr: float = 1e-2,
    lambda_anchor: float = 0.2,
    lambda_concept: float = 5.0,
    seed: int = 42,
):
    """Continuous embedding optimisation starting from the best discrete
    candidate as anchor, then decodes back to text."""
    slug = slugify(concept)
    concept_dir = prompts_root / slug
    cand_file = concept_dir / "candidates_clip.json"

    if not cand_file.exists():
        print(f"[SKIP] {concept} – {cand_file} not found.")
        return

    data = load_candidates_json(cand_file)
    print(f"\n{'='*60}")
    print(f"[CONCEPT] {concept}  ({len(data)} prompts, continuous mode)")
    print(f"{'='*60}")

    # Single pass: discrete scoring to find anchors
    print("  Step 1: discrete scoring for anchors …")
    disc_results, _ = _score_all_entries(data, scorer, K=8, seed=seed)

    # Step 2: continuous optimisation from each anchor
    print("  Step 2: continuous embedding optimisation …")
    results = []
    for entry, disc_r in tqdm(
        zip(data, disc_results), total=len(data), desc=f"{concept} (optim)"
    ):
        prompt_id = entry["id"]
        source = entry["source"]
        anchor_text = disc_r["mapped"]

        if not anchor_text.strip():
            results.append({
                "id": prompt_id, "mapped": "", "mapping_type": "",
                "note": "no anchor", "score": 0.0,
            })
            continue

        decoded, final_loss = scorer.optimise_embedding(
            source_prompt=source,
            anchor_prompt=anchor_text,
            concept=concept,
            K_per_step=K_per_step,
            steps=optim_steps,
            lr=lr,
            lambda_anchor=lambda_anchor,
            lambda_concept=lambda_concept,
            seed=seed,
        )
        note = (
            f"continuous_opt (anchor='{anchor_text[:40]}…', "
            f"loss={final_loss:.6f}, decoded)"
        )
        results.append({
            "id": prompt_id,
            "mapped": decoded,
            "mapped_anchor": anchor_text,
            "mapping_type": "optimised",
            "note": note,
            "score": round(-final_loss, 6),
        })

    # Save map_model_continuous.csv
    out_csv = concept_dir / "map_model_continuous.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "prompt", "mapping_type", "note"])
        for r in results:
            w.writerow([r["id"], r["mapped"], r["mapping_type"], r["note"]])
    print(f"  [SAVED] {out_csv}")

    # Also save a JSON with anchor info for analysis
    detail_json = concept_dir / "map_model_continuous_detail.json"
    with open(detail_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  [SAVED] {detail_json}")


def process_concept_cont_emb(
    concept: str,
    prompts_root: Path,
    scorer: DiffusionScorer,
    K_per_step: int = 4,
    optim_steps: int = 200,
    lr: float = 0.05,
    lambda_concept: float = 5.0,
    seed: int = 42,
):
    """Continuous *mixture-of-candidate-embeddings* optimisation.

    For each forget prompt, optimises a learnable softmax-weighted
    combination of the K candidate embeddings to match the source
    denoising trajectory.  The output is a **tensor** (not text) that
    should be fed directly into the Teacher U-Net's cross-attention
    during DFR distillation training.

    Saves:
      - ``target_embeddings.pt``: dict mapping prompt-id → (1,S,D) tensor
      - ``neutral_embeddings.pt``: **ordered** stacked tensor (N,S,D)
        index-aligned with ``train.csv`` / ``candidates_clip.json``
        for direct use in training (load & index by position).
      - ``map_cont_emb_detail.json``: per-prompt metadata (weights, loss,
        nearest-text decode for inspection)
    """
    slug = slugify(concept)
    concept_dir = prompts_root / slug
    cand_file = concept_dir / "candidates_clip.json"

    if not cand_file.exists():
        print(f"[SKIP] {concept} – {cand_file} not found.")
        return

    data = load_candidates_json(cand_file)
    print(f"\n{'='*60}")
    print(f"[CONCEPT] {concept}  ({len(data)} prompts, cont_emb mode)")
    print(f"{'='*60}")

    # Fallback embedding for entries with no valid candidates:
    # use the CLIP hidden state of the empty string.
    with torch.no_grad():
        empty_embed = scorer.encode_text([""])  # (1, S, D)

    embeddings_dict = {}   # prompt_id → Tensor(1,S,D)
    ordered_embeds = []    # list of (1,S,D) tensors in iteration order
    detail_list = []       # JSON-serialisable metadata

    for entry in tqdm(data, desc=f"{concept} (cont_emb)"):
        prompt_id = entry["id"]
        source = entry["source"]
        cands = entry["candidates"]

        cand_texts = [c["prompt"] for c in cands if c["prompt"].strip()]

        if not cand_texts:
            # Fallback: use empty-string embedding
            ordered_embeds.append(empty_embed.cpu())
            detail_list.append({
                "id": prompt_id, "source": source,
                "note": "no valid candidates", "loss": None,
                "weights": [], "candidates": [],
                "nearest_text": "",
            })
            continue

        e_target, final_loss, weights = scorer.optimise_mixture_embedding(
            source_prompt=source,
            candidate_prompts=cand_texts,
            concept=concept,
            K_per_step=K_per_step,
            steps=optim_steps,
            lr=lr,
            lambda_concept=lambda_concept,
            seed=seed,
        )

        embeddings_dict[prompt_id] = e_target.cpu()  # (1,S,D)
        ordered_embeds.append(e_target.cpu())         # keep order

        # Decode for human inspection (not used in training)
        nearest_text = scorer.decode_embedding_to_text(e_target)

        detail_list.append({
            "id": prompt_id,
            "source": source,
            "loss": round(final_loss, 6),
            "weights": [round(w, 4) for w in weights],
            "candidates": cand_texts,
            "nearest_text": nearest_text,
        })

    # Save target embeddings as a dict .pt file (keyed by prompt_id)
    out_pt = concept_dir / "target_embeddings.pt"
    torch.save(embeddings_dict, out_pt)
    print(f"  [SAVED] {out_pt}  ({len(embeddings_dict)} embeddings)")

    # Save ordered stacked tensor (N, S, D) for direct training use
    if ordered_embeds:
        stacked = torch.cat(ordered_embeds, dim=0)  # (N, S, D)
    else:
        stacked = torch.empty(0)
    neutral_pt = concept_dir / "neutral_embeddings.pt"
    torch.save(stacked, neutral_pt)
    print(f"  [SAVED] {neutral_pt}  (shape {tuple(stacked.shape)})")

    # Save metadata JSON
    detail_json = concept_dir / "map_cont_emb_detail.json"
    with open(detail_json, "w", encoding="utf-8") as f:
        json.dump(detail_list, f, ensure_ascii=False, indent=2)
    print(f"  [SAVED] {detail_json}")

    # Summary
    losses = [d["loss"] for d in detail_list if d["loss"] is not None]
    if losses:
        print(f"  [STATS] Avg loss: {sum(losses)/len(losses):.6f}  "
              f"Min: {min(losses):.6f}  Max: {max(losses):.6f}")


# ────────────────────────────────────────────────────────────────
# All-concepts discovery
# ────────────────────────────────────────────────────────────────

def get_all_concepts(prompts_root: Path) -> List[str]:
    concepts = []
    for d in sorted(prompts_root.iterdir()):
        if d.is_dir() and (d / "candidates_clip.json").exists():
            concepts.append(d.name)
    return concepts


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Model-output score-based candidate selection for concept unlearning."
    )
    ap.add_argument(
        "--mode",
        choices=["discrete", "continuous", "cont_emb", "both"],
        default="discrete",
        help="Scoring mode: 'discrete' (argmax over candidates), "
             "'continuous' (Gumbel-Softmax token optimisation), "
             "'cont_emb' (learnable mixture of candidate embeddings), "
             "or 'both' (discrete + continuous).",
    )
    ap.add_argument(
        "--concept", default=None,
        help="Single concept (folder name). If omitted, processes all concepts "
             "that have candidates_clip.json.",
    )
    ap.add_argument(
        "--model_path",
        default="runwayml/stable-diffusion-v1-5",
        help="HuggingFace model or local checkpoint path.",
    )
    ap.add_argument(
        "--prompts_root",
        default="prompts",
        help="Root directory containing concept folders.",
    )
    ap.add_argument(
        "--K", type=int, default=16,
        help="Number of (latent, timestep) samples for discrete scoring.",
    )
    ap.add_argument(
        "--optim_steps", type=int, default=120,
        help="Gradient steps for continuous Gumbel-Softmax optimisation.",
    )
    ap.add_argument(
        "--optim_lr", type=float, default=0.05,
        help="Learning rate for logit optimisation (Gumbel-Softmax).",
    )
    ap.add_argument(
        "--lambda_anchor", type=float, default=0.1,
        help="Anchor cross-entropy regularisation weight (continuous mode).",
    )
    ap.add_argument(
        "--lambda_concept", type=float, default=5.0,
        help="Concept repulsion weight (continuous mode).",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    ap.add_argument(
        "--device", default=None,
        help="Device (cuda/cpu). Auto-detected if omitted.",
    )
    ap.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Model dtype.",
    )
    ap.add_argument(
        "--nudity_filter", action="store_true",
        help="Add nudity body-part terms to the banned-words list. "
             "Candidates containing these terms will be excluded from scoring.",
    )
    ap.add_argument(
        "--extra_banned_words", nargs="*", default=None,
        help="Additional words to ban from candidate selection.",
    )
    ap.add_argument(
        "--candidates_filename",
        default="candidates_clip.json",
        help=(
            "Candidate JSON filename inside each concept folder, or an absolute path "
            "when processing a single concept. Defaults to candidates_clip.json."
        ),
    )
    ap.add_argument(
        "--output_csv_filename",
        default="map_model.csv",
        help=(
            "Output mapping CSV filename inside each concept folder, or an absolute path. "
            "Defaults to map_model.csv."
        ),
    )
    ap.add_argument(
        "--detail_json_filename",
        default="candidates_model_scores.json",
        help=(
            "Output detailed score JSON filename inside each concept folder, or an absolute path. "
            "Defaults to candidates_model_scores.json."
        ),
    )
    args = ap.parse_args()

    prompts_root = Path(args.prompts_root)
    if not prompts_root.exists():
        print(f"ERROR: {prompts_root} does not exist.", file=sys.stderr)
        sys.exit(1)

    # Concept list
    if args.concept:
        concepts = [args.concept]
    else:
        concepts = get_all_concepts(prompts_root)
        if not concepts:
            print("No concept folders with candidates_clip.json found.", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(concepts)} concepts: {', '.join(concepts)}")

    # Dtype mapping
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Initialise the diffusion scorer (lazy-loaded)
    scorer = DiffusionScorer(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
    )
    if args.mode in ("continuous", "both", "cont_emb"):
        scorer.load()
        if scorer.is_sdxl or scorer.is_sd3:
            print(
                "ERROR: continuous/both/cont_emb modes are not supported for SDXL/SD3 yet. "
                "Use --mode discrete.",
                file=sys.stderr,
            )
            sys.exit(2)

    # Build banned word list
    extra_banned = list(args.extra_banned_words or [])
    if args.nudity_filter:
        extra_banned.extend(NUDITY_BANNED_WORDS)
        print(f"[nudity-filter] Added {len(NUDITY_BANNED_WORDS)} nudity body-part terms to ban list.")
    banned_words = extra_banned or None

    failures = 0
    for concept in concepts:
        try:
            if args.mode in ("discrete", "both"):
                process_concept_discrete(
                    concept=concept,
                    prompts_root=prompts_root,
                    scorer=scorer,
                    K=args.K,
                    seed=args.seed,
                    banned_words=banned_words,
                    candidates_filename=args.candidates_filename,
                    output_csv_filename=args.output_csv_filename,
                    detail_json_filename=args.detail_json_filename,
                )
            if args.mode in ("continuous", "both"):
                process_concept_continuous(
                    concept=concept,
                    prompts_root=prompts_root,
                    scorer=scorer,
                    K_per_step=4,
                    optim_steps=args.optim_steps,
                    lr=args.optim_lr,
                    lambda_anchor=args.lambda_anchor,
                    lambda_concept=args.lambda_concept,
                    seed=args.seed,
                )
            if args.mode == "cont_emb":
                process_concept_cont_emb(
                    concept=concept,
                    prompts_root=prompts_root,
                    scorer=scorer,
                    K_per_step=4,
                    optim_steps=args.optim_steps,
                    lr=args.optim_lr,
                    lambda_concept=args.lambda_concept,
                    seed=args.seed,
                )
        except Exception as e:
            failures += 1
            print(f"[ERROR] Concept '{concept}' failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue

    if failures:
        print(f"\n[FAILED] {failures} concept(s) failed.", file=sys.stderr)
        sys.exit(1)

    print("\n[DONE] All concepts processed.")


if __name__ == "__main__":
    main()
