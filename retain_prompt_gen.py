#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retain Prompt Generator for Continual Unlearning.

For each unlearning target concept, this script:
  1. Asks an LLM for the top 50 closest concepts (not synonyms/subtypes).
  2. Narrows to top 10 via two independent methods:
     a) CLIP text-text cosine similarity (→ related_clip.txt)
     b) Diffusion model trajectory similarity (→ related_score.txt)
  3. For each top-10 set, generates 10 prompts per concept (100 total)
     and saves them as retention prompts.

Output (saved to prompts/<concept>/):
  - related_concepts.txt             – LLM-generated related concepts
  - related_concepts_clip_top10.txt  – 10 CLIP-selected concepts + scores
  - related_concepts_score_top10.txt – 10 trajectory-selected concepts + scores
  - related_clip.txt                 – 100 retain prompts (CLIP method)
  - related_score.txt                – 100 retain prompts (trajectory method)
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Tuple

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


# ────────────────────────────────────────────────────────────────
# Helpers (shared patterns from prompt_gen.py / clip_map_gen.py)
# ────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_{2,}", "_", text).strip("_")
    return text or "concept"


def _strip_leading_markers(s: str) -> str:
    return re.sub(r'^\s*(?:-|\*|\d+[\).\s]|["""]+)\s*', '', s).strip()


def parse_lines(raw_text: str) -> List[str]:
    """Split raw LLM text into cleaned lines."""
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        ln = _strip_leading_markers(ln)
        if ln:
            cleaned.append(ln)
    return cleaned


def make_banlist(concept: str) -> List[str]:
    base = concept.strip().lower()
    forms = {base}
    if " " in base:
        forms.update(t.strip() for t in base.split())
    if base.endswith("y"):
        forms.add(base[:-1] + "ies")
    forms.add(base + "s")
    return sorted(forms)


def contains_banned(line: str, banned: List[str]) -> bool:
    low = line.lower()
    return any(b in low for b in banned)


# ────────────────────────────────────────────────────────────────
# LLM Prompts
# ────────────────────────────────────────────────────────────────

RELATED_50_PROMPT = """\
You are a helpful assistant tasked with finding concepts that are closely \
related to a given input concept, for the purpose of training a diffusion \
model to *retain* knowledge of nearby concepts during concept unlearning.

Given an input concept, provide a list of exactly **50** distinct concepts \
that are related to it. The concepts should be useful as retention targets \
— things the model should still be able to generate after unlearning the \
input concept.

REQUIREMENTS:
- Include a diverse mix of:
  1. **Close sibling items** in the same category (e.g., other dog breeds \
for "dog", other fruits for "apple") — around 15-20 items.
  2. **Broader category / hypernym terms** (e.g., "animal", "pet", "mammal" \
for "dog") — around 5-8 items.
  3. **Contextually co-occurring concepts** (things commonly seen together \
or in similar scenes, but NOT accessories/parts of the input concept) — \
around 10-15 items.
  4. **Analogous concepts from other domains** (e.g., for "dog": "horse" \
as another common pet/companion animal; for "Van Gogh style": other art \
styles) — around 5-10 items.

CRITICAL EXCLUSIONS (do NOT include any of these):
- **Synonyms** of the input concept (e.g., "canine" for "dog", "auto" for "car").
- **Subtypes / specific instances** of the input concept (e.g., "German \
Shepherd" for "dog", "Granny Smith" for "apple").
- **Parts / accessories / activities that directly imply the input** \
(e.g., "leash", "bone", "barking" for "dog"; "peeling" for "apple").
- Rare, obscure, technical, or brand-specific terms.
- Concepts that contain the input concept word itself.

STYLE:
- Use simple, common concepts that image generation models understand well.
- Each concept should be a short phrase (1-4 words).
- If the input is a specific person/celebrity, include similar famous \
individuals and generic person descriptors.
- If the input is an artistic style, include other popular art styles/mediums.

**Output format:** Return exactly 50 concepts, one per line, with no \
explanations, numbering, or extra text.
"""

RETAIN_PROMPTS_PROMPT = """\
You are a prompt generator for Stable Diffusion models.
Follow these rules to produce a list of prompts for model retention training.

**General Constraints (apply to all prompts):**
- Each prompt must be 5–10 words long, using simple, everyday vocabulary.
- Use present tense and natural grammar; use ≤1 comma and ≤1 adjective.
- No special tags or parameters (no ":" or "--ar 16:9" etc.).
- **Do not include the UNLEARN_CONCEPT or any obvious synonym/nickname \
for it in any prompt.**

**Task:**
You will be given:
- UNLEARN_CONCEPT: the concept being unlearned (must NOT appear in prompts).
- RELATED_CONCEPT: a single concept to generate prompts for.

Generate exactly **10 prompts** depicting that concept in diverse, everyday scenes.

**Content Guidelines:**
- Describe simple, plausible scenes involving the concept.
- Ensure variety: mix indoor/outdoor, day/night, weather, activities, \
solitary vs. group, near vs. far framing.
- Do NOT repeat sentence structures; avoid near-duplicates.
- Keep all prompts safe and neutral.

**Output Format:**
- Output exactly **10 prompts**, one per line (no blank lines).
- Do not include numbering, bullet points, section titles, or concept labels.
- Just the 10 prompts, nothing else.
"""


# ────────────────────────────────────────────────────────────────
# OpenAI API helpers
# ────────────────────────────────────────────────────────────────

def _openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        print(
            "[OpenAI] Python SDK not found; using stdlib HTTP fallback for Responses API.",
            file=sys.stderr,
        )
        return _OpenAIHTTPFallback()
    return OpenAI()


class _OpenAIHTTPFallback:
    def __init__(self):
        self.responses = _ResponsesHTTPFallback()


class _ResponsesHTTPFallback:
    def create(self, *, model, input, max_output_tokens):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI retain prompt generation.")
        payload = json.dumps(
            {
                "model": model,
                "input": input,
                "max_output_tokens": max_output_tokens,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API HTTP {exc.code}: {body}") from exc

def _collect_response_text(response) -> str:
    """Extract text from an OpenAI Responses API object."""
    collected = []
    if isinstance(response, dict):
        if response.get("output_text"):
            collected.append(response["output_text"])
        for item in response.get("output", []) or []:
            for c in item.get("content", []) or []:
                txt = c.get("text")
                if txt:
                    collected.append(txt)
            direct_txt = item.get("text")
            if direct_txt:
                collected.append(direct_txt)
        if collected:
            return "".join(collected).strip()
    output_list = getattr(response, "output", None)
    if output_list:
        for item in output_list:
            content = getattr(item, "content", None)
            if content:
                for c in content:
                    txt = getattr(c, "text", None)
                    if txt:
                        collected.append(txt)
            direct_txt = getattr(item, "text", None)
            if direct_txt:
                collected.append(direct_txt)
    if not collected:
        unified = getattr(response, "output_text", None)
        if unified:
            collected.append(unified)
    if not collected:
        raise ValueError("No textual output found in response object.")
    return "".join(collected).strip()


def get_50_related_concepts(
    concept: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 2000,
    retries: int = 3,
    backoff: float = 2.0,
) -> List[str]:
    """Ask an LLM for 50 related-but-not-synonym concepts."""
    client = _openai_client()
    last_err = None
    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "developer", "content": [{"type": "input_text", "text": RELATED_50_PROMPT}]},
                    {"role": "user", "content": [{"type": "input_text", "text": concept}]},
                ],
                max_output_tokens=max_tokens,
            )
            raw = _collect_response_text(response)
            concepts = parse_lines(raw)
            # Deduplicate while preserving order
            seen = set()
            unique = []
            for c in concepts:
                key = c.strip().lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(c)
            if len(unique) >= 30:
                # Good enough — take up to 50
                return unique[:50]
            raise ValueError(f"Only got {len(unique)} unique concepts, retrying…")
        except Exception as e:
            last_err = e
            time.sleep((backoff ** attempt) + 0.1 * (attempt + 1))
    raise RuntimeError(f"Failed to get 50 related concepts after {retries} attempts: {last_err}")


def _generate_prompts_for_one_concept(
    unlearn_concept: str,
    related_concept: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1500,
    retries: int = 3,
    backoff: float = 2.0,
) -> List[str]:
    """Generate 10 retain prompts for a single related concept via an LLM."""
    client = _openai_client()
    user_content = (
        f"UNLEARN_CONCEPT: {unlearn_concept}\n"
        f"RELATED_CONCEPT: {related_concept}"
    )
    last_err = None
    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "developer", "content": [{"type": "input_text", "text": RETAIN_PROMPTS_PROMPT}]},
                    {"role": "user", "content": [{"type": "input_text", "text": user_content}]},
                ],
                max_output_tokens=max_tokens,
            )
            raw = _collect_response_text(response)
            prompts = parse_lines(raw)
            if len(prompts) >= 7:
                return prompts[:10]
            raise ValueError(
                f"Only got {len(prompts)} prompts for '{related_concept}', expected 10. Retrying…"
            )
        except Exception as e:
            last_err = e
            time.sleep((backoff ** attempt) + 0.1 * (attempt + 1))
    raise RuntimeError(
        f"Failed to generate prompts for '{related_concept}' after {retries} attempts: {last_err}"
    )


def generate_retain_prompts(
    concept: str,
    top10: List[str],
    model: str = "gpt-4o-mini",
) -> List[str]:
    """Generate ~100 retain prompts (10 per concept) by calling the LLM once per concept."""
    all_prompts: List[str] = []
    for i, rc in enumerate(top10):
        print(f"    [{i+1}/{len(top10)}] Generating 10 prompts for '{rc}' …")
        batch = _generate_prompts_for_one_concept(concept, rc, model=model)
        all_prompts.extend(batch)
        print(f"    [{i+1}/{len(top10)}] Got {len(batch)} prompts for '{rc}'.")
    print(f"  Total retain prompts generated: {len(all_prompts)}")
    return all_prompts


# ────────────────────────────────────────────────────────────────
# CLIP-based concept ranking
# ────────────────────────────────────────────────────────────────

class CLIPConceptRanker:
    """Rank candidate concepts by CLIP text-text similarity to the target."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor
        print(f"[CLIP] Loading {self.model_name} on {self.device} …")
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        print("[CLIP] Model loaded.")

    @torch.no_grad()
    def rank(self, concept: str, candidates: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Return top_k candidates ranked by CLIP text-text cosine similarity
        to `concept`. Returns list of (concept_name, score) sorted descending.
        """
        self._load()
        all_texts = [concept] + candidates
        inputs = self._processor(
            text=all_texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        text_embeds = self._model.get_text_features(**inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        source_embed = text_embeds[0:1]         # (1, D)
        candidate_embeds = text_embeds[1:]       # (N, D)
        sims = (source_embed @ candidate_embeds.T).squeeze(0)  # (N,)
        sims_list = sims.cpu().tolist()

        # Sort by similarity descending
        scored = list(zip(candidates, sims_list))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ────────────────────────────────────────────────────────────────
# Trajectory-based concept ranking (reuses DiffusionScorer pattern)
# ────────────────────────────────────────────────────────────────

class TrajectoryConceptRanker:
    """Rank candidate concepts by UNet noise-prediction trajectory
    similarity to the unlearning target concept."""

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
        self.unet = None
        self.transformer = None
        self.pipeline = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.text_encoder_3 = None
        self.tokenizer = None
        self.tokenizer_2 = None
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

        print(f"[Trajectory] Loading from {self.model_path} on {self.device} ({self.dtype}) …")
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

        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self._loaded = True
        print(f"[Trajectory] Model loaded. mode={'SDXL' if self.is_sdxl else 'SD/SD2'}")

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

        print(f"[Trajectory] Loading SD3 from {self.model_path} on {self.device} ({self.dtype}) …")
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
        print("[Trajectory] Model loaded. mode=SD3")

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

    @torch.no_grad()
    def rank(
        self,
        concept: str,
        candidates: List[str],
        top_k: int = 10,
        K: int = 16,
        seed: int = 42,
        latent_shape: Tuple[int, ...] = None,
        batch_size: int = 8,
    ) -> List[Tuple[str, float]]:
        """
        Score each candidate concept by average MSE between its UNet noise
        prediction and the target concept's noise prediction over K random
        (latent, timestep) draws.

        Lower MSE → more similar trajectory → closer concept.
        Returns top_k candidates sorted by similarity (lowest MSE first).
        Each entry is (concept_name, negative_mse_score).
        """
        self.load()
        device = self.device
        gen = torch.Generator(device="cpu").manual_seed(seed)
        if latent_shape is None:
            latent_shape = self._default_latent_shape()

        N = len(candidates)
        accum_mse = torch.zeros(N, device=device)

        if self.is_sd3:
            src_embed, src_pooled = self.encode_text_sd3([concept])
            all_cand_embeds = []
            all_cand_pooled = []
            for i in range(0, N, batch_size):
                batch = candidates[i : i + batch_size]
                embeds, pooled = self.encode_text_sd3(batch)
                all_cand_embeds.append(embeds)
                all_cand_pooled.append(pooled)
            cand_embeds = torch.cat(all_cand_embeds, dim=0)
            cand_pooled = torch.cat(all_cand_pooled, dim=0)
        elif self.is_sdxl:
            src_embed, src_pooled = self.encode_text_sdxl([concept])
            all_cand_embeds = []
            all_cand_pooled = []
            for i in range(0, N, batch_size):
                batch = candidates[i : i + batch_size]
                embeds, pooled = self.encode_text_sdxl(batch)
                all_cand_embeds.append(embeds)
                all_cand_pooled.append(pooled)
            cand_embeds = torch.cat(all_cand_embeds, dim=0)
            cand_pooled = torch.cat(all_cand_pooled, dim=0)
        else:
            src_embed = self.encode_text([concept])  # (1, S, D)
            all_cand_embeds = []
            for i in range(0, N, batch_size):
                batch = candidates[i : i + batch_size]
                embeds = self.encode_text(batch)  # (B, S, D)
                all_cand_embeds.append(embeds)
            cand_embeds = torch.cat(all_cand_embeds, dim=0)  # (N, S, D)

        for k_idx in tqdm(range(K), desc="trajectory scoring", leave=False):
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

            # Source (target concept) prediction
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

            # Score each candidate
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
        neg_mse = (-avg_mse).cpu().tolist()

        # Sort by neg_mse descending (= lowest MSE first = most similar)
        scored = list(zip(candidates, neg_mse))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def unload(self):
        """Free GPU memory."""
        for attr in (
            "pipeline",
            "unet",
            "transformer",
            "text_encoder",
            "tokenizer",
            "scheduler",
        ):
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
                setattr(self, attr, None)
        for attr in ("text_encoder_2", "tokenizer_2", "text_encoder_3", "tokenizer_3", "vae"):
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
                setattr(self, attr, None)
        self.is_sdxl = False
        self.is_sd3 = False
        self._loaded = False
        torch.cuda.empty_cache()


# ────────────────────────────────────────────────────────────────
# File I/O helpers
# ────────────────────────────────────────────────────────────────

def save_concepts_list(path: Path, concepts: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        for c in concepts:
            f.write(c + "\n")
    print(f"  [SAVED] {path} ({len(concepts)} concepts)")


def save_scored_concepts(path: Path, scored: List[Tuple[str, float]]):
    with open(path, "w", encoding="utf-8") as f:
        for concept, score in scored:
            f.write(f"{concept}\t{score:.6f}\n")
    print(f"  [SAVED] {path} ({len(scored)} concepts with scores)")


def save_prompts(path: Path, prompts: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p + "\n")
    print(f"  [SAVED] {path} ({len(prompts)} prompts)")


# ────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────

def process_concept(
    concept: str,
    prompts_root: Path,
    llm_model: str,
    clip_model_name: str,
    model_path: str,
    K: int,
    seed: int,
    device: str,
    dtype: torch.dtype,
    skip_clip: bool = False,
    skip_score: bool = False,
    force: bool = False,
    extra_banned: List[str] = None,
):
    """Full pipeline for one concept: 50 LLM concepts → rank → generate prompts."""
    slug = slugify(concept)
    concept_dir = prompts_root / slug
    concept_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[CONCEPT] {concept}")
    print(f"{'='*60}")

    # ── Step 1: Get related concepts from LLM ──
    concepts_file = concept_dir / "related_concepts.txt"
    if concepts_file.exists() and not force:
        print(f"  [CACHE] Loading existing {concepts_file}")
        with open(concepts_file, encoding="utf-8") as f:
            all_concepts = [ln.strip() for ln in f if ln.strip()]
    else:
        print(f"  Step 1: Asking LLM for 50 related concepts …")
        all_concepts = get_50_related_concepts(concept, model=llm_model)
        save_concepts_list(concepts_file, all_concepts)

    print(f"  Got {len(all_concepts)} related concepts.")
    banned = make_banlist(concept)

    # Filter out any that accidentally contain the concept word
    filtered_concepts = [c for c in all_concepts if not contains_banned(c, banned)]
    if len(filtered_concepts) < len(all_concepts):
        removed = len(all_concepts) - len(filtered_concepts)
        print(f"  [FILTER] Removed {removed} concepts containing banned words.")
    if len(filtered_concepts) < 10:
        print(f"  [WARN] Only {len(filtered_concepts)} concepts left after filtering. "
              "Need at least 10. Proceeding with what we have.")

    # ── Step 2a: Rank by CLIP similarity ──
    clip_top10 = []
    if not skip_clip:
        clip_top10_file = concept_dir / "related_concepts_clip_top10.txt"
        clip_prompts_file = concept_dir / "related_clip.txt"

        if clip_prompts_file.exists() and not force:
            print(f"  [CACHE] {clip_prompts_file} already exists, skipping CLIP ranking.")
        else:
            print(f"  Step 2a: Ranking by CLIP text similarity …")
            clip_ranker = CLIPConceptRanker(model_name=clip_model_name, device=device)
            clip_top10 = clip_ranker.rank(concept, filtered_concepts, top_k=10)
            save_scored_concepts(clip_top10_file, clip_top10)

            top10_names = [name for name, _ in clip_top10]
            print(f"  CLIP top 10: {top10_names}")

            # Generate prompts for CLIP top 10
            print(f"  Step 3a: Generating 100 retain prompts (CLIP method) …")
            clip_prompts = generate_retain_prompts(concept, top10_names, model=llm_model)
            # Filter prompts that accidentally contain the unlearning concept
            clip_prompts = [p for p in clip_prompts if not contains_banned(p, banned)]
            save_prompts(clip_prompts_file, clip_prompts)

    # ── Step 2b: Rank by trajectory similarity ──
    if not skip_score:
        score_top10_file = concept_dir / "related_concepts_score_top10.txt"
        score_prompts_file = concept_dir / "related_score.txt"

        if score_prompts_file.exists() and not force:
            print(f"  [CACHE] {score_prompts_file} already exists, skipping trajectory ranking.")
        else:
            print(f"  Step 2b: Ranking by diffusion trajectory similarity …")
            traj_ranker = TrajectoryConceptRanker(
                model_path=model_path, device=device, dtype=dtype
            )
            # Filter concepts that contain banned words before scoring
            scoring_candidates = [c for c in filtered_concepts if not contains_banned(c, banned)]
            score_top10 = traj_ranker.rank(
                concept, scoring_candidates, top_k=10, K=K, seed=seed
            )
            save_scored_concepts(score_top10_file, score_top10)
            traj_ranker.unload()

            top10_names = [name for name, _ in score_top10]
            print(f"  Trajectory top 10: {top10_names}")

            # Generate prompts for trajectory top 10
            print(f"  Step 3b: Generating 100 retain prompts (trajectory method) …")
            score_prompts = generate_retain_prompts(concept, top10_names, model=llm_model)
            # Filter prompts that accidentally contain the unlearning concept
            score_prompts = [p for p in score_prompts if not contains_banned(p, banned)]
            save_prompts(score_prompts_file, score_prompts)

    print(f"\n  [DONE] Concept '{concept}' processed.")


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate retention prompts via related concept selection "
                    "(CLIP + trajectory scoring) for continual unlearning."
    )
    ap.add_argument(
        "--concept", required=True,
        help="The unlearning target concept (e.g., 'dog', 'pikachu').",
    )
    ap.add_argument(
        "--model_path",
        default="runwayml/stable-diffusion-v1-5",
        help="Diffusion model path for trajectory scoring. "
             "Use the current unlearned checkpoint in continual pipelines.",
    )
    ap.add_argument(
        "--llm_model", default="gpt-4o-mini",
        help="OpenAI model for LLM calls (concept discovery + prompt generation).",
    )
    ap.add_argument(
        "--clip_model",
        default="openai/clip-vit-large-patch14",
        help="CLIP model for text-text similarity ranking.",
    )
    ap.add_argument(
        "--prompts_root",
        default="prompts",
        help="Root directory containing concept folders.",
    )
    ap.add_argument(
        "--K", type=int, default=16,
        help="Number of (latent, timestep) samples for trajectory scoring.",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument(
        "--device", default=None,
        help="Device (cuda/cpu). Auto-detected if omitted.",
    )
    ap.add_argument(
        "--dtype", default="fp16", choices=["fp16", "bf16", "fp32"],
        help="Model dtype for trajectory scorer.",
    )
    ap.add_argument(
        "--skip_clip", action="store_true",
        help="Skip CLIP-based ranking (only do trajectory).",
    )
    ap.add_argument(
        "--skip_score", action="store_true",
        help="Skip trajectory-based ranking (only do CLIP).",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Overwrite existing files instead of using cached results.",
    )
    ap.add_argument(
        "--nudity_filter", action="store_true",
        help="Add nudity body-part terms (breast, genitalia, buttocks, feet, "
             "belly, armpits, etc.) to the banned-words list. Filters both "
             "the related concepts and the generated retain prompts.",
    )
    ap.add_argument(
        "--extra_banned_words", nargs="*", default=None,
        help="Additional words to ban from retain concepts and prompts.",
    )
    args = ap.parse_args()

    prompts_root = Path(args.prompts_root)
    if not prompts_root.exists():
        print(f"ERROR: {prompts_root} does not exist.", file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    extra_banned = list(args.extra_banned_words or [])
    if args.nudity_filter:
        extra_banned.extend(NUDITY_BANNED_WORDS)
        print(f"[nudity-filter] Added {len(NUDITY_BANNED_WORDS)} nudity body-part terms to ban list.")
    extra_banned = extra_banned or None

    process_concept(
        concept=args.concept,
        prompts_root=prompts_root,
        llm_model=args.llm_model,
        clip_model_name=args.clip_model,
        model_path=args.model_path,
        K=args.K,
        seed=args.seed,
        device=device,
        dtype=dtype,
        skip_clip=args.skip_clip,
        skip_score=args.skip_score,
        force=args.force,
        extra_banned=extra_banned,
    )

    print("\n[DONE] Retention prompt generation complete.")


if __name__ == "__main__":
    main()
