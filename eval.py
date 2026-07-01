#!/usr/bin/env python3

import argparse
import csv
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel, CLIPProcessor, Qwen2_5_VLForConditionalGeneration

try:
    import wandb
except ImportError:
    wandb = None


torch.set_grad_enabled(False)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

YES_RE = re.compile(r"\b(yes|yep|yeah|y|true)\b", re.IGNORECASE)
NO_RE = re.compile(r"\b(no|nope|n|false)\b", re.IGNORECASE)

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
}


def normalize_concept_key(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def format_concept_for_vlm(text: str) -> str:
    cleaned = text.replace("_", " ").strip()
    special = {
        "brad pitt": "Brad Pitt",
        "van gogh style": "Van Gogh style",
        "lionel messi": "Lionel Messi",
        "mickey mouse": "Mickey Mouse",
    }
    return special.get(cleaned.lower(), cleaned)


def safe_filename(text: str, max_len: int = 90) -> str:
    text = re.sub(r'[\\/*?:"<>|]', "_", text).strip()
    text = re.sub(r"\s+", "_", text)
    return text[:max_len] or "prompt"


def parse_yes_no(text: str) -> str:
    if not text:
        return "No"
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered.startswith("yes"):
        return "Yes"
    if lowered.startswith("no"):
        return "No"
    if YES_RE.search(stripped) and not NO_RE.search(stripped):
        return "Yes"
    return "No"


def build_vlm_prompt(concept: str, use_confusables: bool) -> List[dict]:
    concept_text = format_concept_for_vlm(concept)
    negatives = CONFUSABLES.get(normalize_concept_key(concept), []) if use_confusables else []
    negative_clause = ""
    if negatives:
        negative_clause = (
            " Do not count "
            + ", ".join(format_concept_for_vlm(item) for item in negatives)
            + f" as {concept_text}."
        )
    return [
        {
            "role": "system",
            "content": "You are a strict visual binary classifier. Answer with exactly one word: Yes or No.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": f"Does this image clearly contain {concept_text}?{negative_clause} Answer Yes or No only.",
                },
            ],
        },
    ]


def clean_eval_df(df: pd.DataFrame) -> pd.DataFrame:
    if not {"concept", "prompt"}.issubset(df.columns):
        raise ValueError("Evaluation CSV must contain columns: concept,prompt")
    out = df[["concept", "prompt"]].dropna().copy()
    out["concept"] = out["concept"].astype(str).str.strip()
    out["prompt"] = out["prompt"].astype(str).str.strip()
    return out[(out["concept"] != "") & (out["prompt"] != "")]


def load_extra_prompts(root: Path, filename: str = "val_100.csv") -> pd.DataFrame:
    rows = []
    if not root.is_dir():
        return pd.DataFrame(columns=["concept", "prompt"])
    for concept_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        csv_path = concept_dir / filename
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        if "prompt" not in df.columns:
            continue
        for prompt in df["prompt"].dropna().astype(str):
            prompt = prompt.strip()
            if prompt:
                rows.append((concept_dir.name, prompt))
    return pd.DataFrame(rows, columns=["concept", "prompt"])


def load_prompts(prompts_csv: Path, extra_root: Path, disable_extra: bool) -> pd.DataFrame:
    base = clean_eval_df(pd.read_csv(prompts_csv))
    extra = pd.DataFrame(columns=["concept", "prompt"])
    if not disable_extra:
        extra = load_extra_prompts(extra_root)
    merged = pd.concat([base, extra], ignore_index=True)
    merged.drop_duplicates(subset=["concept", "prompt"], inplace=True)
    return merged.reset_index(drop=True)


def save_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def generate_images(args, df: pd.DataFrame, device: torch.device) -> Dict[str, List[Tuple[str, str]]]:
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_folder,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        safety_checker=None,
        feature_extractor=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    output_root = Path(args.output_root)
    prompt_image_map: Dict[str, List[Tuple[str, str]]] = {}
    save_pool = ThreadPoolExecutor(max_workers=args.save_workers)
    futures = []

    for concept in dict.fromkeys(df["concept"].tolist()):
        concept_key = normalize_concept_key(concept)
        concept_dir = output_root / concept_key
        concept_prompts = df[df["concept"] == concept]["prompt"].tolist()
        prompt_image_map[concept_key] = []
        manifest_rows = []

        for start in tqdm(range(0, len(concept_prompts), args.gen_prompt_batch), desc=f"generate {concept_key}"):
            batch_prompts = concept_prompts[start : start + args.gen_prompt_batch]
            images = pipe(batch_prompts, num_images_per_prompt=args.num_images_per_prompt).images
            for prompt_index, prompt in enumerate(batch_prompts):
                for image_index in range(args.num_images_per_prompt):
                    image = images[prompt_index * args.num_images_per_prompt + image_index]
                    filename = f"{start + prompt_index:04d}_{safe_filename(prompt)}_{image_index:02d}.png"
                    image_path = concept_dir / filename
                    futures.append(save_pool.submit(save_image, image_path, image))
                    prompt_image_map[concept_key].append((prompt, str(image_path)))
                    manifest_rows.append({"concept": concept, "prompt": prompt, "image": str(image_path)})

        for future in futures:
            future.result()
        futures.clear()
        pd.DataFrame(manifest_rows).to_csv(concept_dir / "manifest.csv", index=False)

    save_pool.shutdown(wait=True)
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prompt_image_map


def load_existing_images(output_root: Path) -> Dict[str, List[Tuple[str, str]]]:
    prompt_image_map: Dict[str, List[Tuple[str, str]]] = {}
    for manifest in sorted(output_root.glob("*/manifest.csv")):
        concept_key = manifest.parent.name
        df = pd.read_csv(manifest)
        prompt_image_map[concept_key] = [(row["prompt"], row["image"]) for _, row in df.iterrows()]
    if not prompt_image_map:
        raise FileNotFoundError("No generated manifests found. Run without --skip_generation first.")
    return prompt_image_map


def load_vlm(args):
    dtype_map = {"bfloat16": torch.bfloat16, "fp16": torch.float16, "float32": torch.float32}
    processor_kwargs = {}
    if args.vlm_min_pixels is not None:
        processor_kwargs["min_pixels"] = args.vlm_min_pixels
    if args.vlm_max_pixels is not None:
        processor_kwargs["max_pixels"] = args.vlm_max_pixels

    processor = AutoProcessor.from_pretrained(args.vlm_model_id, **processor_kwargs)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.vlm_model_id,
        torch_dtype=dtype_map[args.vlm_dtype],
        device_map=args.vlm_device_map,
        attn_implementation=args.vlm_attn_impl,
    ).eval()
    return model, processor


def vlm_classify(model, processor, rows: List[Tuple[str, str]], concept_key: str, args) -> pd.DataFrame:
    records = []
    concept_text = format_concept_for_vlm(concept_key)
    for start in tqdm(range(0, len(rows), args.vlm_batch_size), desc=f"vlm {concept_key}"):
        batch = rows[start : start + args.vlm_batch_size]
        images = []
        valid = []
        batch_records = [None] * len(batch)
        for batch_index, (prompt, image_path) in enumerate(batch):
            try:
                images.append(Image.open(image_path).convert("RGB"))
                valid.append((batch_index, prompt, image_path))
            except (OSError, UnidentifiedImageError):
                batch_records[batch_index] = {
                    "prompt": prompt,
                    "image": image_path,
                    "prediction": "No",
                    "raw_text": "",
                }
        if not valid:
            records.extend(batch_records)
            continue

        messages = [build_vlm_prompt(concept_text, args.use_confusables) for _ in valid]
        text_prompts = [
            processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
            for message in messages
        ]
        inputs = processor(text=text_prompts, images=images, padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.vlm_max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        generated = [out[len(inp) :] for inp, out in zip(inputs.input_ids, outputs)]
        texts = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for (batch_index, prompt, image_path), raw in zip(valid, texts):
            batch_records[batch_index] = {
                "prompt": prompt,
                "image": image_path,
                "prediction": parse_yes_no(raw),
                "raw_text": raw.strip(),
            }
        records.extend(batch_records)
    return pd.DataFrame(records)


def clip_scores(rows: List[Tuple[str, str]], model, processor, device: torch.device, batch_size: int) -> List[float]:
    scores = []
    for start in tqdm(range(0, len(rows), batch_size), desc="clip"):
        batch = rows[start : start + batch_size]
        prompts = []
        images = []
        batch_scores = [0.0] * len(batch)
        valid_indices = []
        for batch_index, (prompt, image_path) in enumerate(batch):
            try:
                images.append(Image.open(image_path).convert("RGB"))
                prompts.append(prompt)
                valid_indices.append(batch_index)
            except (OSError, UnidentifiedImageError):
                pass
        if not images:
            scores.extend(batch_scores)
            continue
        inputs = processor(text=prompts, images=images, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = model(**inputs)
        diagonal = outputs.logits_per_image.diag().detach().float().cpu().tolist()
        for batch_index, score in zip(valid_indices, diagonal):
            batch_scores[batch_index] = score
        scores.extend(batch_scores)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate one Diffusers checkpoint with CLIP and Qwen2.5-VL.")
    parser.add_argument("--model_folder", required=True, help="Diffusers checkpoint folder.")
    parser.add_argument("--prompts_csv", required=True, help="CSV with concept,prompt columns.")
    parser.add_argument("--output_root", default="eval_outputs", help="Directory for images and metrics.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id for SD and CLIP.")
    parser.add_argument("--log_file", default=None, help="Optional evaluation log path.")
    parser.add_argument("--prompts_extra_root", default="prompts_new", help="Folder with per-concept val_100.csv files.")
    parser.add_argument("--disable_prompts_extra", action="store_true", help="Use only --prompts_csv.")
    parser.add_argument("--skip_generation", action="store_true", help="Reuse existing images and manifests.")
    parser.add_argument("--skip_clip", action="store_true", help="Skip CLIP scoring.")
    parser.add_argument("--only_vlm", action="store_true", help="Reuse images and run VLM only.")
    parser.add_argument("--use_confusables", action="store_true", help="Use stricter confusable-aware VLM prompts.")
    parser.add_argument("--batch_size", type=int, default=64, help="CLIP batch size.")
    parser.add_argument("--vlm_batch_size", type=int, default=32, help="VLM batch size.")
    parser.add_argument("--gen_prompt_batch", type=int, default=16, help="Generation prompt batch size.")
    parser.add_argument("--num_images_per_prompt", type=int, default=8)
    parser.add_argument("--save_workers", type=int, default=6)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="disabled")
    parser.add_argument("--vlm_model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--vlm_dtype", choices=["bfloat16", "fp16", "float32"], default="bfloat16")
    parser.add_argument("--vlm_attn_impl", choices=["sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--vlm_min_pixels", type=int, default=None)
    parser.add_argument("--vlm_max_pixels", type=int, default=None)
    parser.add_argument("--vlm_device_map", default="auto")
    parser.add_argument("--vlm_max_new_tokens", type=int, default=3)
    args = parser.parse_args()

    if args.only_vlm:
        args.skip_generation = True
        args.skip_clip = True

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file) if args.log_file else output_root / "eval_log.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    df = load_prompts(Path(args.prompts_csv), Path(args.prompts_extra_root), args.disable_prompts_extra)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.skip_generation:
        prompt_image_map = load_existing_images(output_root)
    else:
        prompt_image_map = generate_images(args, df, device)

    wandb_run = None
    if args.wandb_project and args.wandb_mode != "disabled":
        if wandb is None:
            raise ImportError("wandb is not installed but --wandb_project was provided.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            dir=str(output_root / "wandb"),
            config=vars(args),
        )

    clip_model = clip_processor = None
    if not args.skip_clip:
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

    vlm_model, vlm_processor = load_vlm(args)
    summary_rows = []

    with log_file.open("w", encoding="utf-8") as log:
        for concept_key, rows in prompt_image_map.items():
            vlm_df = vlm_classify(vlm_model, vlm_processor, rows, concept_key, args)
            if not args.skip_clip:
                scores = clip_scores(rows, clip_model, clip_processor, device, args.batch_size)
                vlm_df["clip_score"] = scores[: len(vlm_df)]
            else:
                vlm_df["clip_score"] = None

            out_csv = output_root / concept_key / "metrics.csv"
            vlm_df.to_csv(out_csv, index=False)
            vlm_yes_rate = float((vlm_df["prediction"] == "Yes").mean()) if len(vlm_df) else 0.0
            clip_mean = float(pd.to_numeric(vlm_df["clip_score"], errors="coerce").mean()) if not args.skip_clip else None
            summary = {
                "concept": concept_key,
                "num_images": len(vlm_df),
                "vlm_yes_rate": vlm_yes_rate,
                "clip_mean": clip_mean,
            }
            summary_rows.append(summary)
            line = f"{concept_key}: images={len(vlm_df)} vlm_yes_rate={vlm_yes_rate:.4f}"
            if clip_mean is not None:
                line += f" clip_mean={clip_mean:.4f}"
            print(line)
            log.write(line + "\n")
            if wandb_run is not None:
                wandb.log({f"{concept_key}/vlm_yes_rate": vlm_yes_rate, f"{concept_key}/clip_mean": clip_mean})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "summary.csv", index=False)
    if wandb_run is not None:
        wandb.log({"summary": wandb.Table(dataframe=summary_df)})
        wandb.finish()


if __name__ == "__main__":
    main()
