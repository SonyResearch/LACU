#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, gc, sys, re, time
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import torch
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from diffusers import StableDiffusionPipeline

# --- CLIP for scoring (unchanged) ---
from transformers import CLIPProcessor, CLIPModel
from transformers import CLIPTextModel

# --- Qwen2.5-VL for VLM binary classification ---
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Override modules if present in model_folder
from diffusers import UNet2DConditionModel

import wandb

from concurrent.futures import ThreadPoolExecutor, wait

import glob

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_grad_enabled(False)

YES_RE = re.compile(r"\b(yes|yep|yeah|y|true)\b", re.IGNORECASE)
NO_RE  = re.compile(r"\b(no|nope|n|false)\b", re.IGNORECASE)

CONFUSABLES: Dict[str, List[str]] = {
    "pikachu": ["Charmander", "Squirtle", "Jigglypuff"],
    "brad_pitt": ["Matt Damon", "George Clooney", "Leonardo DiCaprio"],
    "van_gogh_style": ["Impressionism Style", "Monet Style", "Post-Impressionist Style"],
    "golf_ball": ["Tennis Ball", "Baseball", "Ping Pong Ball"],
    "apple": ["Pear", "Tomato", "Orange"],
    "spiderman": ["Batman", "Superman", "Iron Man"],
    "lionel_messi": ["Cristiano Ronaldo", "Neymar", "Mbappe"],
    "cartoon_style": ["Anime Style", "Comic Book Style", "3D Style"],
    "banana": ["Mango", "Pineapple", "Cucumber"],
    "mickey_mouse": ["Donald Duck", "Minnie Mouse", "Goofy"],
    "dog": ["Wolf", "Fox", "Cat"]
}

def get_confusables(concept: str) -> List[str]:
    """Get list of confusable concepts for given concept."""
    key = normalize_concept_key(concept)
    return CONFUSABLES.get(key, [])

def parse_yes_no(text: Optional[str]) -> str:
    """Map model output to strict 'Yes'/'No'."""
    if not text:
        return "No"
    t = text.strip()
    tl = t.lower()
    if tl.startswith("yes"):
        return "Yes"
    if tl.startswith("no"):
        return "No"
    if YES_RE.search(t) and not NO_RE.search(t):
        return "Yes"
    if NO_RE.search(t) and not YES_RE.search(t):
        return "No"
    return "No"

def build_conversation(concept: str) -> list:
    """Chat template for Qwen: return Yes/No only."""
    system_msg = {
        "role": "system",
        "content": (
            "You are a precise visual binary classifier. "
            "Answer with exactly one word: Yes or No. Use English only."
        )
    }
    user_msg = {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": f"Does this image contain {concept}? Answer Yes or No only."}
        ]
    }
    return [system_msg, user_msg]

def build_conversation_soft(concept: str, negatives: List[str]) -> list:
    """Chat template with confusables awareness for stricter classification."""
    negative_clause = ""
    if negatives:
        formatted_negatives = [format_concept_for_vlm(neg) for neg in negatives]
        neg_list = ", ".join(formatted_negatives)
        negative_clause = (
            f" Do NOT count {neg_list} as {concept} if they appear instead. "
            f"If the object could be one of these, answer No."
        )
    return [
        {"role": "system",
         "content": (
             "You are a strict visual binary classifier. "
             "Say Yes only if the concept is clearly present as the intended class. "
             "If uncertain or ambiguous, answer No. "
             "Final answer must be one word: Yes or No."
         )},
        {"role": "user",
         "content": [
             {"type":"image"},
             {"type":"text",
              "text": f"Is there {concept} in this image?{negative_clause} Answer Yes or No only."}
         ]}
    ]

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

def format_concept_for_vlm(concept: str) -> str:
    """
    Format concept for VLM classification: remove underscores and capitalize properly.
    Examples:
    - "brad_pitt" -> "Brad Pitt"
    - "van_gogh_style" -> "Van Gogh Style"
    - "golf_ball" -> "Golf Ball"
    """
    # Replace underscores with spaces
    formatted = concept.replace('_', ' ')
    
    # Capitalize each word
    words = formatted.split()
    capitalized_words = []
    
    for word in words:
        # Handle special cases
        if word.lower() == 'van':
            capitalized_words.append('Van')
        elif word.lower() == 'gogh':
            capitalized_words.append('Gogh')
        else:
            capitalized_words.append(word.capitalize())
    
    return ' '.join(capitalized_words)

@torch.inference_mode()
def vlm_classify_folder_batch(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    images_dir: Path,
    concept: str,
    batch_size: int = 8,
    max_new_tokens: int = 3,
    use_confusables: bool = False,
) -> Tuple[pd.DataFrame, float]:
    """
    Run VLM Yes/No classification on all images in batches.
    Batches multiple images together for efficiency.
    """
    image_paths = [p for p in sorted(images_dir.rglob("*")) if is_image_file(p)]
    if not image_paths:
        return pd.DataFrame(columns=["image", "prediction", "raw_text"]), 0.0

    records = []
    n_yes = 0
    vlm_concept = format_concept_for_vlm(concept)
    is_style = is_style_concept(concept)
    negatives = get_confusables(concept) if use_confusables else []

    # Process in batches
    for batch_start in tqdm(range(0, len(image_paths), batch_size), desc=f"VLM classify (batch): {images_dir.name}"):
        batch_end = min(batch_start + batch_size, len(image_paths))
        batch_paths = image_paths[batch_start:batch_end]
        
        # Load images
        batch_images = []
        valid_indices = []
        
        for i, img_path in enumerate(batch_paths):
            try:
                pil_img = Image.open(str(img_path)).convert("RGB")
                batch_images.append(pil_img)
                valid_indices.append(i)
            except (UnidentifiedImageError, OSError):
                records.append({"image": str(img_path), "prediction": "No", "raw_text": "<invalid image>"})
        
        if not batch_images:
            continue
        
        # Build batch prompts
        batch_prompts = []
        for _ in batch_images:
            if use_confusables and negatives:
                messages = build_conversation_soft(vlm_concept, negatives)
            else:
                messages = build_conversation(vlm_concept)
            text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            batch_prompts.append(text_prompt)
        
        # Process batch
        try:
            inputs = processor(
                text=batch_prompts,
                images=batch_images,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )

            gen_ids = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
            out_texts = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            # Process results
            for batch_idx, img_idx in enumerate(valid_indices):
                img_path = batch_paths[img_idx]
                out_text = out_texts[batch_idx]
                pred = parse_yes_no(out_text)
                
                if pred == "Yes":
                    n_yes += 1
                records.append({"image": str(img_path), "prediction": pred, "raw_text": out_text.strip()})
        
        except Exception as e:
            print(f"[warn] Batch processing error: {e}")
            # Fallback: process individually
            for img_idx, img_path in enumerate(batch_paths):
                if img_idx not in valid_indices:
                    continue
                try:
                    vi = valid_indices.index(img_idx)
                    pil_img = batch_images[vi]
                    if use_confusables and negatives:
                        messages = build_conversation_soft(vlm_concept, negatives)
                    else:
                        messages = build_conversation(vlm_concept)
                    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    
                    inputs = processor(
                        text=[text_prompt],
                        images=[pil_img],
                        padding=True,
                        return_tensors="pt",
                    ).to(model.device)
                    
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=0.0,
                        top_p=1.0,
                    )
                    
                    gen_ids = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
                    out_text = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
                    pred = parse_yes_no(out_text)
                    
                    if pred == "Yes":
                        n_yes += 1
                    records.append({"image": str(img_path), "prediction": pred, "raw_text": out_text.strip()})
                except Exception:
                    records.append({"image": str(img_path), "prediction": "No", "raw_text": "<error>"})

    df = pd.DataFrame.from_records(records, columns=["image", "prediction", "raw_text"])
    acc = n_yes / len(image_paths) if image_paths else 0.0
    return df, acc


@torch.inference_mode()
def clip_score_batch(
    model: CLIPModel,
    processor: CLIPProcessor,
    prompts: List[str],
    images: List[Image.Image],
    device: torch.device,
    batch_size: int = 32,
) -> List[float]:
    """
    Compute CLIP scores in batches for efficiency.
    """
    scores = []
    
    for batch_start in tqdm(range(0, len(prompts), batch_size), desc="CLIP scoring"):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_images = images[batch_start:batch_end]
        
        try:
            inputs = processor(
                text=batch_prompts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(device)
            
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # [batch_size, batch_size]
            
            # Diagonal gives the score for each (prompt, image) pair
            batch_scores = [float(logits_per_image[i, i].item()) for i in range(len(batch_prompts))]
            scores.extend(batch_scores)
        except Exception as e:
            print(f"[warn] CLIP batch error: {e}, falling back to individual scoring")
            # Fallback
            for prompt, image in zip(batch_prompts, batch_images):
                try:
                    inputs = processor(
                        text=[prompt],
                        images=image,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=77
                    ).to(device)
                    outputs = model(**inputs)
                    scores.append(float(outputs.logits_per_image.item()))
                except:
                    scores.append(0.0)
    return scores

def normalize_concept_key(s: str) -> str:
    return s.strip().lower().replace(" ", "_")

def is_style_concept(concept: str) -> bool:
    k = normalize_concept_key(concept)
    return "style" in k

def _clean_eval_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    if not {'concept','prompt'}.issubset(df_raw.columns):
        raise AssertionError("eval.csv must have columns: concept,prompt")
    out = df_raw[['concept', 'prompt']].dropna().copy()
    out['concept'] = out['concept'].astype(str).str.strip()
    out['prompt']  = out['prompt'].astype(str).str.strip()
    return out

def _load_extra_prompts(root: str, prefer_name: str = 'val_100.csv') -> pd.DataFrame:
    rows = []
    root_path = Path(root)
    if not root_path.is_dir():
        return pd.DataFrame(columns=['concept','prompt'])

    # Try to read immediate subfolders as concepts
    concept_dirs = [p for p in root_path.iterdir() if p.is_dir()]
    for cdir in concept_dirs:
        concept = cdir.name
        # 1) preferred filename
        cand = cdir / prefer_name
        # 2) fallbacks
        fallback = cdir / 'val_fixed.csv'
        target = None
        if cand.exists():
            target = cand
        elif fallback.exists():
            target = fallback
        else:
            # last resort: rglob if users put the file deeper
            hits = list(cdir.rglob(prefer_name)) or list(cdir.rglob('val_fixed.csv'))
            if hits:
                target = hits[0]

        if target is None:
            # no CSV for this concept; skip silently
            continue

        try:
            dfx = pd.read_csv(target)
        except Exception as e:
            print(f"[warn] Could not read {target}: {e}")
            continue

        # choose a plausible text column
        prompt_col = None
        for col in ('prompt', 'text', 'caption'):
            if col in dfx.columns:
                prompt_col = col
                break
        if prompt_col is None:
            print(f"[warn] No 'prompt'/'text'/'caption' column in {target}, skipping")
            continue

        for ptxt in dfx[prompt_col].dropna().astype(str):
            ptxt = ptxt.strip()
            if ptxt:
                rows.append((concept, ptxt))

    if not rows:
        return pd.DataFrame(columns=['concept','prompt'])
    dfe = pd.DataFrame(rows, columns=['concept','prompt'])
    dfe['concept'] = dfe['concept'].astype(str).str.strip()
    dfe['prompt']  = dfe['prompt'].astype(str).str.strip()
    return dfe

def encode_prompts(pipe, prompts, device, do_guidance: bool):
    tok, te = pipe.tokenizer, pipe.text_encoder
    ti = tok(prompts, padding="max_length", max_length=tok.model_max_length,
             truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        pe = te(**ti).last_hidden_state

    ue = None
    if do_guidance:
        ui = tok([""] * len(prompts), padding="max_length",
                 max_length=tok.model_max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            ue = te(**ui).last_hidden_state
    return pe, ue

def _png_save_fast(img, path):
    # Still PNG, but much faster to write (lossless).
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, format='PNG', compress_level=1, optimize=False)

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate images (SD) and evaluate with VLM (Qwen2.5-VL yes/no) + CLIP scores."
    )
    parser.add_argument('--model_folder', type=str, required=True,
                        help='Path to SD weights (feature_extractor, unet, vae, etc.)')
    parser.add_argument('--prompts_csv', type=str, required=True,
                        help='CSV with columns: concept,prompt')
    parser.add_argument('--output_root', type=str, default='./outputs', help='Root folder to save generated images and results')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id for SD & CLIP (Qwen uses device_map=auto)')
    parser.add_argument('--log_file', type=str, default='eval_log.txt', help='Log file path')

        # --- Extra prompt source (per-concept folders) ---
    parser.add_argument('--prompts_extra_root', type=str,
                        default='/prompts/',
                        help='Root folder containing subfolders named by concept, each with val_100.csv (or val_fixed.csv).')
    parser.add_argument('--prompts_extra_filename', type=str,
                        default='val_100.csv',
                        help='Expected CSV filename inside each concept folder; will fall back to val_fixed.csv if missing.')
    parser.add_argument('--disable_prompts_extra', action='store_true',
                        help='Disable loading prompts from prompts_extra_root.')

    # --- Control which stages to run ---
    parser.add_argument('--skip_generation', action='store_true', 
                        help='Skip image generation (images already exist)')
    parser.add_argument('--skip_clip', action='store_true',
                        help='Skip CLIP scoring')
    parser.add_argument('--only_vlm', action='store_true',
                        help='Only run VLM classification (implies --skip_generation and --skip_clip)')

    # --- VLM classification mode ---
    parser.add_argument('--use_confusables', action='store_true',
                        help='Use confusables-aware classification for stricter evaluation')
    
    # === Batch size parameters ===
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for CLIP scoring and other batch operations')
    parser.add_argument('--vlm_batch_size', type=int, default=8,
                        help='Batch size for VLM classification (use smaller due to memory)')

    # --- Weights & Biases (optional) ---
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_mode', type=str, default=None, choices=['online','offline','disabled'])

    # --- VLM (Qwen) knobs ---
    parser.add_argument('--vlm_model_id', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--vlm_dtype', type=str, default='bfloat16', choices=['bfloat16', 'fp16', 'float32'])
    parser.add_argument('--vlm_attn_impl', type=str, default='sdpa', choices=['sdpa', 'flash_attention_2'])
    parser.add_argument('--vlm_min_pixels', type=int, default=None)
    parser.add_argument('--vlm_max_pixels', type=int, default=None)
    parser.add_argument('--vlm_device_map', type=str, default='auto')
    parser.add_argument('--vlm_max_new_tokens', type=int, default=3)

    # --- Generation control ---
    parser.add_argument('--num_images_per_prompt', type=int, default=8)

    parser.add_argument('--gen_prompt_batch', type=int, default=16,
                        help='#prompts to process together per SD forward pass')
    parser.add_argument('--preencode_text', action='store_true',
                        help='Pre-encode text embeddings to avoid repeated text-encoder work')
    parser.add_argument('--save_workers', type=int, default=4,
                        help='Threads for async image saving')

    args = parser.parse_args()

    # Handle --only_vlm flag
    if args.only_vlm:
        args.skip_generation = True
        args.skip_clip = True

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_root, exist_ok=True)

    wandb_run = None
    if args.wandb_project and args.wandb_mode != 'disabled':
        if not args.wandb_run_name:
            base_model = os.path.basename(os.path.normpath(args.model_folder))
            ts = time.strftime("%Y%m%d_%H%M%S")
            args.wandb_run_name = f"eval_{base_model}_{ts}"
        wandb_kwargs = dict(
            dir="/wandb/",
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name,
            config={
                "model_folder": args.model_folder,
                "prompts_csv": args.prompts_csv,
                "output_root": args.output_root,
                "gpu": args.gpu,
                "vlm_model_id": args.vlm_model_id,
                "vlm_dtype": args.vlm_dtype,
                "vlm_attn_impl": args.vlm_attn_impl,
                "vlm_min_pixels": args.vlm_min_pixels,
                "vlm_max_pixels": args.vlm_max_pixels,
            }
        )
        if args.wandb_mode:
            wandb_kwargs["mode"] = 'offline' if args.wandb_mode == 'offline' else 'online'
        wandb_run = wandb.init(**wandb_kwargs)

    # -------------------------------
    # Load prompts
    # -------------------------------
    df_base = _clean_eval_df(pd.read_csv(args.prompts_csv))

    if not args.disable_prompts_extra and os.path.isdir(args.prompts_extra_root):
        df_extra = _load_extra_prompts(args.prompts_extra_root, prefer_name=args.prompts_extra_filename)
    else:
        df_extra = pd.DataFrame(columns=['concept','prompt'])

    # Merge + de-dup (concept,prompt)
    df = pd.concat([df_base, df_extra], ignore_index=True)
    before = len(df)
    df.drop_duplicates(subset=['concept','prompt'], inplace=True)
    after = len(df)

    # Small diagnostics
    n_base_concepts  = df_base['concept'].nunique()
    n_extra_concepts = df_extra['concept'].nunique() if not df_extra.empty else 0
    print(f"[prompts] eval.csv concepts={n_base_concepts}, extra concepts={n_extra_concepts}, "
          f"rows merged={before} -> {after} unique")


    concepts = list(dict.fromkeys(df['concept'].tolist()))  # order-preserving unique

    all_prompt_image_map: Dict[str, List[Tuple[str, str]]] = {}

    # -------------------------------
    # STAGE 1: Generate images for all concepts (SD)
    # -------------------------------
    if not args.skip_generation:
        print(f"Loading Stable Diffusion pipeline from {args.model_folder} ...")

        def try_load_pipeline(path, device):
            try:
                pipe = StableDiffusionPipeline.from_pretrained(
                    path, torch_dtype=torch.float16
                ).to(device)
                pipe.safety_checker = None
                print("[INFO] Loaded pipeline directly from folder.")
                return pipe
            except Exception as e:
                print(f"[WARN] Direct load failed: {e}")
                return None

        # 1) Try to load your model folder
        pipe = try_load_pipeline(args.model_folder, device)

        # 2) Fallback: load base SD-1.5 and override UNet/text_encoder
        if pipe is None:
            print("[INFO] Falling back to SD 1.5 base weights...")
            base_id = "runwayml/stable-diffusion-v1-5"

            pipe = StableDiffusionPipeline.from_pretrained(
                base_id, torch_dtype=torch.float16
            ).to(device)
            pipe.safety_checker = None

            unet_path = os.path.join(args.model_folder, "unet")
            te_path   = os.path.join(args.model_folder, "text_encoder")

            # Override UNet
            if os.path.isdir(unet_path):
                print("[INFO] Overriding UNet from model folder...")
                pipe.unet = UNet2DConditionModel.from_pretrained(
                    unet_path, torch_dtype=torch.float16
                ).to(device)

            # Override text encoder
            if os.path.isdir(te_path):
                print("[INFO] Overriding text_encoder from model folder...")
                pipe.text_encoder = CLIPTextModel.from_pretrained(
                    te_path, torch_dtype=torch.float16
                ).to(device)

            pipe = pipe.to(torch.float16)

            print("[INFO] Fallback pipeline constructed successfully (forced fp16).")

        lora_root = os.path.join(args.model_folder, "lora")

        # Case 1: diffusers-format LoRA stored directly in model/lora/
        if os.path.isdir(lora_root) and os.path.isfile(os.path.join(lora_root, "pytorch_lora_weights.safetensors")):
            try:
                print(f"[LoRA] Found diffusers-format LoRA at: {lora_root}")
                pipe.load_lora_weights(lora_root)
                pipe.fuse_lora()
                print("[LoRA] Loaded + fused.")
            except Exception as e:
                print(f"[LoRA] Failed to load diffusers-format LoRA: {e}")

        else:
            # Case 2: recursive search for any .safetensors LoRA file (your case)
            lora_files = glob.glob(os.path.join(lora_root, "**", "*.safetensors"), recursive=True)
            if lora_files:
                for lf in lora_files:
                    try:
                        print(f"[LoRA] Found LoRA file: {lf}")
                        pipe.load_lora_weights(lf)
                        pipe.fuse_lora()
                        print("[LoRA] Loaded + fused.")
                    except Exception as e:
                        print(f"[LoRA] Failed to load {lf}: {e}")
            else:
                print("[LoRA] No LoRA found under model_folder.")

        save_pool = ThreadPoolExecutor(max_workers=args.save_workers)


        with open(args.log_file, "a") as logf:
            logf.write(f"\nUnlearned model - {args.output_root}\n")

            for concept in concepts:
                concept_key = normalize_concept_key(concept)

                futs = []

                concept_prompts = df[df['concept'] == concept]['prompt'].tolist()

                concept_dir = os.path.join(args.output_root, concept_key)
                os.makedirs(concept_dir, exist_ok=True)
                # os.makedirs(final_concept_dir, exist_ok=True)

                print(f"\n[GEN] Concept: {concept}  |  #prompts={len(concept_prompts)}  -> {concept_dir}")
                # logf.write(f"\nGenerating for concept: {concept} ({len(concept_prompts)} prompts)\n")

                prompt_image_map: List[Tuple[str, str]] = []

                # Process prompts in batches of args.gen_prompt_batch
                bsz = max(1, int(args.gen_prompt_batch))
                for start in tqdm(range(0, len(concept_prompts), bsz), desc=f"SD gen(batched): {concept_key}"):
                    batch_prompts = concept_prompts[start:start+bsz]

                    use_guidance = True  # you use default guidance in pipe; keep consistent

                    if args.preencode_text:
                        pe, ue = encode_prompts(pipe, batch_prompts, device, use_guidance=True)
                        call_kwargs = dict(prompt_embeds=pe, negative_prompt_embeds=ue)
                    else:
                        call_kwargs = dict(prompt=batch_prompts)

                    # Simple retry-on-OOM: shrink local batch by half and retry
                    local_M = args.num_images_per_prompt
                    local_prompts = batch_prompts
                    while True:
                        try:
                            out = pipe(
                                **call_kwargs,
                                num_images_per_prompt=local_M,
                                output_type="pil",
                            )
                            images = out.images  # length = len(local_prompts) * local_M
                            break
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            if len(local_prompts) > 1:
                                # halve the prompt batch if needed
                                local_prompts = local_prompts[: max(1, len(local_prompts)//2)]
                                if args.preencode_text:
                                    pe, pooled, ue = encode_prompts(pipe, local_prompts, device, use_guidance)
                                    call_kwargs = dict(prompt_embeds=pe,
                                                    negative_prompt_embeds=ue,
                                                    pooled_prompt_embeds=pooled)
                                else:
                                    call_kwargs = dict(prompt=local_prompts)
                                continue
                            elif local_M > 1:
                                local_M = max(1, local_M // 2)
                                continue
                            else:
                                raise  # even 1×1 OOM: surface the error

                    # schedule async saves; keep filenames identical pattern
                    for j, prompt in enumerate(local_prompts):
                        safe_prompt = re.sub(r'[\\/*?:"<>|]', "_", prompt)[:50]
                        for m in range(local_M):
                            idx = j*local_M + m
                            img = images[idx]
                            img_path = os.path.join(concept_dir, f"{safe_prompt}_{m}.png")
                            f = save_pool.submit(_png_save_fast, img, img_path)  # NEW
                            futs.append(f)                                       # NEW

                            prompt_image_map.append((prompt, img_path))

                wait(futs)
                print(f"[INFO] Updated {len(prompt_image_map)} image paths for {concept}")

                all_prompt_image_map[concept_key] = prompt_image_map

        # After the 'for concept in concepts:' loop ends:
        save_pool.shutdown(wait=True)  # NEW: cleanly close the thread pool
        
        # Free SD to reclaim VRAM before loading VLM/CLIP
        del pipe
        torch.cuda.empty_cache()
        gc.collect()

    else:
        # Build prompt_image_map from existing images
        print("Skipping generation, building image map from existing files...")
        for concept in concepts:
            concept_key = normalize_concept_key(concept)
            concept_dir = os.path.join(args.output_root, concept_key)
            if not os.path.exists(concept_dir):
                print(f"Warning: {concept_dir} does not exist, skipping concept {concept}")
                continue
            
            prompt_image_map: List[Tuple[str, str]] = []
            
            # Get all image files and extract prompts from filenames
            image_paths = [str(p) for p in Path(concept_dir).rglob("*") if is_image_file(p)]
            
            for img_path in image_paths:
                # Extract prompt from filename: "prompt_text_0.png" -> "prompt_text"
                img_filename = os.path.basename(img_path)
                # Remove extension first
                name_without_ext = os.path.splitext(img_filename)[0]
                # Split by underscore and remove the last part (which should be the number)
                parts = name_without_ext.split('_')
                if len(parts) > 1 and parts[-1].isdigit():
                    prompt = '_'.join(parts[:-1])
                else:
                    # Fallback: use the whole filename without extension
                    prompt = name_without_ext
                
                # Replace underscores with spaces to reconstruct the original prompt
                prompt = prompt.replace('_', ' ')
                prompt_image_map.append((prompt, img_path))
                
            all_prompt_image_map[concept_key] = prompt_image_map

    # -------------------------------
    # Load CLIP for scoring - OPTIONAL
    # -------------------------------
    clip_model = None
    clip_processor = None
    if not args.skip_clip:
        print("\nLoading CLIP (openai/clip-vit-base-patch32) for scoring...")
        clip_device = device  # same single GPU is fine
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(clip_device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # -------------------------------
    # Load VLM (Qwen2.5-VL)
    # -------------------------------
    print(f"Loading VLM: {args.vlm_model_id} ...")
    if args.vlm_dtype.lower() in ("bf16", "bfloat16"):
        vlm_dtype = torch.bfloat16
    elif args.vlm_dtype.lower() in ("fp16", "float16", "half"):
        vlm_dtype = torch.float16
    else:
        vlm_dtype = torch.float32

    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.vlm_model_id,
        torch_dtype=vlm_dtype,
        device_map=args.vlm_device_map,
        attn_implementation=args.vlm_attn_impl,
    )
    # Control visual token budget if requested
    if args.vlm_min_pixels is not None or args.vlm_max_pixels is not None:
        vlm_processor = AutoProcessor.from_pretrained(
            args.vlm_model_id,
            min_pixels=args.vlm_min_pixels if args.vlm_min_pixels is not None else 224 * 224,
            max_pixels=args.vlm_max_pixels if args.vlm_max_pixels is not None else 2048 * 2048,
        )
    else:
        vlm_processor = AutoProcessor.from_pretrained(args.vlm_model_id)

    # -------------------------------
    # STAGE 2: Evaluate each concept with VLM + CLIP
    # -------------------------------
    all_concept_metrics = []
    concept_step_eval = 0

    with open(args.log_file, "a") as logf:
        for concept in concepts:
            concept_key = normalize_concept_key(concept)
            concept_dir = os.path.join(args.output_root, concept_key)
            prompt_image_map = all_prompt_image_map.get(concept_key, [])

            print(f"\n[EVAL] Concept: {concept}  |  #images={len(prompt_image_map)}")
            logf.write(f"\nEvaluating concept: {concept} ({len(prompt_image_map)} images)\n")

            # --- VLM Yes/No classification ---
            df_vlm, vlm_accuracy = vlm_classify_folder_batch(
                model=vlm_model,
                processor=vlm_processor,
                images_dir=Path(concept_dir),
                concept=concept,  # use original concept text for the question
                batch_size=args.vlm_batch_size,
                max_new_tokens=args.vlm_max_new_tokens,
                use_confusables=args.use_confusables,
            )
            vlm_csv_path = os.path.join(concept_dir, "vlm_predictions.csv")
            try:
                df_vlm.to_csv(vlm_csv_path, index=False)
            except Exception:
                pass  # don't crash if CSV can't be written

            # --- CLIP scores per (prompt, image) ---
            avg_clip_score = 0.0
            clip_scores = []
            if not args.skip_clip and clip_model is not None:
                clip_prompts = [prompt for prompt, _ in prompt_image_map]
                clip_images = []
                for _, img_path in prompt_image_map:
                    try:
                        image = Image.open(img_path).convert("RGB")
                        clip_images.append(image)
                    except (UnidentifiedImageError, OSError):
                        clip_images.append(Image.new('RGB', (224, 224)))
                
                # Batch CLIP scoring
                clip_scores = clip_score_batch(
                    model=clip_model,
                    processor=clip_processor,
                    prompts=clip_prompts,
                    images=clip_images,
                    device=clip_device,
                    batch_size=args.batch_size,
                )

                avg_clip_score = (sum(clip_scores) / len(clip_scores)) if clip_scores else 0.0

            # --- Print + log ---
            if args.skip_clip:
                print(f"Concept: {concept} | VLM Acc: {vlm_accuracy:.4f}")
                logf.write(f"Concept: {concept}\nVLM Accuracy: {vlm_accuracy:.4f}\n")
            else:
                print(f"Concept: {concept} | VLM Acc: {vlm_accuracy:.4f} | Avg CLIP: {avg_clip_score:.4f}")
                logf.write(f"Concept: {concept}\nVLM Accuracy: {vlm_accuracy:.4f}\nAverage CLIP score: {avg_clip_score:.4f}\n")
            logf.write("-" * 50 + "\n")

            # Wandb logging (table + images + hist)
            if wandb_run is not None:
                table = wandb.Table(columns=["concept", "prompt", "image_path", "vlm_pred", "vlm_correct", "clip_score"])
                # Build a map image_path -> VLM pred quickly
                vlm_pred_map = {row["image"]: row["prediction"] for _, row in df_vlm.iterrows()} if not df_vlm.empty else {}
                for i, (prompt, img_path) in enumerate(prompt_image_map):
                    pred = vlm_pred_map.get(img_path, "")
                    corr = (pred == "Yes") if pred else None
                    cs = clip_scores[i] if i < len(clip_scores) else None
                    table.add_data(concept_key, prompt, img_path, pred, corr, cs)
                
                log_dict = {
                    f"{concept_key}/vlm_accuracy": vlm_accuracy,
                    f"{concept_key}/num_images": len(prompt_image_map),
                    f"{concept_key}/table": table,
                }
                if not args.skip_clip:
                    log_dict[f"{concept_key}/avg_clip"] = avg_clip_score
                    if clip_scores:
                        log_dict[f"{concept_key}/clip_hist"] = wandb.Histogram(clip_scores)
                
                wandb.log(log_dict, step=concept_step_eval)

            all_concept_metrics.append({
                "concept": concept_key,
                "vlm_accuracy": vlm_accuracy,
                "avg_clip": avg_clip_score,
                "num_images": len(prompt_image_map),
            })
            concept_step_eval += 1

    # -------------------------------
    # Summary metrics
    # -------------------------------
    if wandb_run is not None and all_concept_metrics:
        import statistics as stats
        accs = [m["vlm_accuracy"] for m in all_concept_metrics]
        clips = [m["avg_clip"] for m in all_concept_metrics]
        log_dict = {
            "summary/mean_vlm_accuracy": sum(accs)/len(accs),
            "summary/median_vlm_accuracy": stats.median(accs),
            "summary/num_concepts": len(all_concept_metrics)
        }
        if not args.skip_clip:
            log_dict["summary/mean_clip"] = sum(clips)/len(clips)
            log_dict["summary/median_clip"] = stats.median(clips)
        
        wandb.log(log_dict, step=concept_step_eval)
        
        columns = ["concept","vlm_accuracy","num_images"]
        if not args.skip_clip:
            columns.insert(2, "avg_clip")
        summary_table = wandb.Table(columns=columns)
        for m in all_concept_metrics:
            if args.skip_clip:
                summary_table.add_data(m["concept"], m["vlm_accuracy"], m["num_images"])
            else:
                summary_table.add_data(m["concept"], m["vlm_accuracy"], m["avg_clip"], m["num_images"])
        wandb.log({"summary/concepts_table": summary_table}, step=concept_step_eval)

# Cleanup
    try:
        if clip_model is not None:
            del clip_model
        del vlm_model, vlm_processor
    except Exception:
        pass
    torch.cuda.empty_cache()
    gc.collect()

    if wandb_run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
