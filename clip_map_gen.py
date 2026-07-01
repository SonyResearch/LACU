#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP-based mapping prompt generator.

For each concept in prompts_new/:
  1. Reads train_100.csv (source prompts for the concept to unlearn).
  2. For each source prompt, calls OpenAI to generate 10 candidate mapping
     prompts (concept replaced or removed), including one "removal-only" variant.
  3. Computes CLIP text-similarity between the source prompt and each candidate.
  4. Selects the candidate with the highest CLIP score.
  5. Saves the result to map_clip.csv (same format as map_100.csv).
"""

import argparse, csv, json, os, re, sys, time
from pathlib import Path
from typing import List, Dict, Any

import torch
from openai import OpenAI

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
# System prompt for candidate generation
# ────────────────────────────────────────────────────────────────

CANDIDATE_MAP_SYS_PROMPT = """You are a mapping-prompt generator for diffusion-model concept unlearning.

TASK
Given a SOURCE prompt that depicts a specific CONCEPT, produce exactly 10 candidate mapped prompts.
Each candidate replaces or removes the CONCEPT from the source prompt to create an alternative scene.

RULES
1. Candidates 1-8: REPLACE the concept with a context-appropriate, visually plausible alternative.
   - Use diverse replacement categories across the 8 candidates (animals, objects, people, vehicles, foods, plants, etc.).
   - Preserve the sentence structure, determiners, plurality, and tense.
   - The replacement must NOT be a synonym, subtype, or closely related variant of the CONCEPT.
   - Keep the same style: 4-12 words, simple natural language, present tense, everyday vocabulary.
   - Each candidate should use a DIFFERENT replacement concept for variety.

2. Candidate 9: REPLACE with a broad generic category word (e.g., "object", "animal", "person", "thing", "item") that fits the context.

3. Candidate 10: REMOVE the concept entirely. Keep the rest of the scene intact and grammatically correct. 
   If the concept is the main subject, rephrase to describe just the scene/setting without it.

====================================================
BANNED WORDS — CRITICAL (read carefully)
====================================================
- NEVER include the original CONCEPT or ANY word from the BANNED_WORDS list in any candidate prompt.
- This includes morphological variants: e.g., if "cartoon" is banned, then "cartoonish", "cartoony", "cartoons", "cartoon-like" are ALL banned.
- However, distinct but related concepts (e.g., "anime", "manga", "comic") are NOT banned — only the exact concept and its direct morphological forms.
  Examples for "cartoon style": ban "cartoon", "cartoonish", "cartoony" — but "anime", "manga", "comic" are allowed.
  Examples for "dog": ban "dog", "dogs", "doggy" — but "puppy", "canine", "hound" are allowed.
- The BANNED_WORDS list will be provided in the user message. Only those exact words and their morphological variants are banned.

IMPORTANT
- All candidates must be visually depictable, safe, and non-graphic.
- Keep the overall scene structure (location, action, framing) as close to the source as possible.
- Double-check every candidate against the BANNED_WORDS list before outputting.

OUTPUT FORMAT (STRICT JSON)
{
  "candidates": [
    {"id": 1, "prompt": "<mapped prompt>", "mapping_type": "replace", "replacement": "<what replaced the concept>"},
    {"id": 2, "prompt": "<mapped prompt>", "mapping_type": "replace", "replacement": "<what replaced the concept>"},
    ...
    {"id": 9, "prompt": "<mapped prompt>", "mapping_type": "replace", "replacement": "<generic category word>"},
    {"id": 10, "prompt": "<mapped prompt>", "mapping_type": "remove", "replacement": "none"}
  ]
}
"""


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_{2,}", "_", text).strip("_")
    return text or "concept"


def extract_json_block(s: str) -> str:
    """Robustly extract a JSON object from LLM output."""
    fence = re.search(r"```json\s*(\{.*?\})\s*```", s, re.S | re.I)
    if fence:
        return fence.group(1)
    first = s.find("{")
    last = s.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("No JSON object detected in model output.")
    block = s[first : last + 1]
    json.loads(block)  # validate
    return block


def read_train_csv(path: Path) -> List[Dict[str, str]]:
    """Read train_100.csv → list of {id, prompt}."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"id": row["id"].strip(), "prompt": row["prompt"].strip()})
    return rows


def make_banlist(concept: str) -> List[str]:
    """Generate a basic morphological ban-list from the concept string."""
    base = concept.strip().lower()
    forms = {base}
    # Split multi-word concepts into individual tokens
    if " " in base:
        forms.update(t.strip() for t in base.split() if len(t.strip()) > 2)
    # Common suffixes
    for word in list(forms):
        forms.add(word + "s")
        forms.add(word + "ish")
        forms.add(word + "y")
        forms.add(word + "like")
        forms.add(word + "-like")
        if word.endswith("e"):
            forms.add(word[:-1] + "ing")
        else:
            forms.add(word + "ing")
    return sorted(f for f in forms if f)


def get_banned_words(concept: str, extra_banned: List[str] = None) -> List[str]:
    """Get the ban-list from morphological forms of the concept,
    plus any extra banned words (e.g. nudity body-part terms)."""
    words = make_banlist(concept)
    if extra_banned:
        words = sorted(set(words) | set(w.strip().lower() for w in extra_banned))
    return words


def contains_banned(text: str, banned: List[str]) -> bool:
    """Check if text contains any banned word."""
    low = text.lower()
    return any(b in low for b in banned)


def filter_candidates(candidates: List[Dict], banned: List[str]) -> List[Dict]:
    """Flag candidates that leak banned words (for logging/debugging)."""
    for c in candidates:
        prompt = c.get("prompt", "")
        c["_has_banned"] = contains_banned(prompt, banned)
    return candidates


# ────────────────────────────────────────────────────────────────
# OpenAI candidate generation
# ────────────────────────────────────────────────────────────────

def generate_candidates(
    client: OpenAI,
    concept: str,
    source_prompt: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 2000,
    retries: int = 3,
    backoff: float = 2.0,
    banned_words: List[str] = None,
) -> List[Dict[str, str]]:
    """Generate 10 candidate mapping prompts for a single source prompt."""
    ban_str = ", ".join(banned_words) if banned_words else "(none)"
    user_msg = (
        f'CONCEPT = "{concept}"\n'
        f'SOURCE PROMPT = "{source_prompt}"\n'
        f'BANNED_WORDS (do NOT use any of these or their variants in any candidate): [{ban_str}]\n'
        f"Produce exactly 10 candidates per the JSON schema."
    )
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CANDIDATE_MAP_SYS_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                temperature=0.9,
            )
            raw = resp.choices[0].message.content.strip()
            json_str = extract_json_block(raw)
            payload = json.loads(json_str)
            candidates = payload.get("candidates", [])
            if len(candidates) < 1:
                raise ValueError("No candidates returned.")
            return candidates
        except Exception as e:
            last_err = e
            time.sleep((backoff**attempt) + 0.1 * (attempt + 1))
    raise RuntimeError(
        f"Candidate generation failed after {retries} attempts: {last_err}"
    )


def generate_candidates_batch(
    client: OpenAI,
    concept: str,
    prompts: List[str],
    model: str = "gpt-4o-mini",
    max_tokens: int = 16000,
    retries: int = 3,
    backoff: float = 2.0,
    batch_size: int = 10,
    banned_words: List[str] = None,
) -> List[List[Dict[str, str]]]:
    """Generate candidates for a batch of source prompts in a single API call."""
    numbered = "\n".join(
        f"{i+1}. \"{p}\"" for i, p in enumerate(prompts)
    )
    ban_str = ", ".join(banned_words) if banned_words else "(none)"
    user_msg = (
        f'CONCEPT = "{concept}"\n'
        f'BANNED_WORDS (do NOT use any of these or their variants in any candidate): [{ban_str}]\n\n'
        f"SOURCE PROMPTS (generate 10 candidates for EACH):\n{numbered}\n\n"
        f"Return a JSON object with key \"results\" containing a list of objects, "
        f"one per source prompt (in order). Each object has keys \"source_id\" (1-based) "
        f"and \"candidates\" (list of 10 candidate objects as specified)."
    )
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CANDIDATE_MAP_SYS_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                temperature=0.9,
            )
            raw = resp.choices[0].message.content.strip()
            json_str = extract_json_block(raw)
            payload = json.loads(json_str)
            results = payload.get("results", [])
            if len(results) != len(prompts):
                raise ValueError(
                    f"Expected {len(prompts)} result groups, got {len(results)}"
                )
            return [r["candidates"] for r in results]
        except Exception as e:
            last_err = e
            time.sleep((backoff**attempt) + 0.1 * (attempt + 1))
    raise RuntimeError(
        f"Batch candidate generation failed after {retries} attempts: {last_err}"
    )


# ────────────────────────────────────────────────────────────────
# CLIP scoring
# ────────────────────────────────────────────────────────────────

class CLIPScorer:
    """Lazy-loaded CLIP model for text-text similarity scoring."""

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
    def score_pairs(
        self, source_prompt: str, candidates: List[str]
    ) -> List[float]:
        """
        Compute CLIP text-text cosine similarity between the source prompt
        and each candidate prompt.
        Returns a list of similarity scores (higher = more similar).
        """
        self._load()
        all_texts = [source_prompt] + candidates
        inputs = self._processor(
            text=all_texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        text_embeds = self._model.get_text_features(**inputs)
        # Normalize
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        source_embed = text_embeds[0:1]  # (1, D)
        candidate_embeds = text_embeds[1:]  # (N, D)
        sims = (source_embed @ candidate_embeds.T).squeeze(0)  # (N,)
        return sims.cpu().tolist()


# ────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────

def process_concept(
    concept: str,
    prompts_root: Path,
    model: str,
    batch_size: int,
    clip_scorer: CLIPScorer,
    save_candidates: bool = False,
    extra_banned: List[str] = None,
):
    """Process a single concept: generate candidates, score, save best."""
    slug = slugify(concept)
    concept_dir = prompts_root / slug
    train_csv = concept_dir / "train_100.csv"

    if not train_csv.exists():
        print(f"[SKIP] {concept} – {train_csv} not found.")
        return

    rows = read_train_csv(train_csv)
    print(f"\n{'='*60}")
    print(f"[CONCEPT] {concept}  ({len(rows)} prompts)")
    print(f"{'='*60}")

    # Build ban-list for this concept (with optional extra banned words)
    banned = get_banned_words(concept, extra_banned=extra_banned)
    print(f"  Banned words ({len(banned)}): {', '.join(banned[:15])}{'…' if len(banned)>15 else ''}")

    client = OpenAI()  # uses OPENAI_API_KEY env var

    all_results = []  # list of dicts for final CSV

    # Optionally store all candidates for debugging
    all_candidates_data = []

    # Process in batches
    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        batch_prompts = [r["prompt"] for r in batch_rows]
        batch_ids = [r["id"] for r in batch_rows]

        print(f"  Generating candidates for prompts {batch_start+1}-{batch_start+len(batch_rows)} …")

        try:
            batch_candidates = generate_candidates_batch(
                client=client,
                concept=concept,
                prompts=batch_prompts,
                model=model,
                batch_size=len(batch_prompts),
                banned_words=banned,
            )
        except Exception as e:
            # Fallback: generate one-by-one
            print(f"  [WARN] Batch failed ({e}), falling back to one-by-one …")
            batch_candidates = []
            for p in batch_prompts:
                try:
                    cands = generate_candidates(
                        client=client, concept=concept, source_prompt=p, model=model,
                        banned_words=banned,
                    )
                    batch_candidates.append(cands)
                except Exception as e2:
                    print(f"    [ERROR] Prompt '{p[:40]}…': {e2}")
                    batch_candidates.append([])

        # Score each prompt's candidates with CLIP
        for i, (row, cands) in enumerate(zip(batch_rows, batch_candidates)):
            prompt_id = row["id"]
            source_prompt = row["prompt"]

            if not cands:
                print(f"    [WARN] No candidates for prompt {prompt_id}, skipping.")
                all_results.append({
                    "id": prompt_id,
                    "prompt": source_prompt,
                    "mapped": "",
                    "mapping_type": "",
                    "note": "no candidates generated",
                    "clip_score": 0.0,
                })
                continue

            # Post-filter: flag candidates that leak banned words
            cands = filter_candidates(cands, banned)
            n_leaked = sum(1 for c in cands if c.get("_has_banned", False))
            if n_leaked:
                print(f"    [WARN] {n_leaked}/{len(cands)} candidates for prompt {prompt_id} contain banned words, excluding them.")

            candidate_texts = [c.get("prompt", "") for c in cands]
            # Filter out empty candidates AND those with banned words
            valid_indices = [
                j for j, t in enumerate(candidate_texts)
                if t.strip() and not cands[j].get("_has_banned", False)
            ]
            if not valid_indices:
                print(f"    [WARN] All candidates empty for prompt {prompt_id}.")
                all_results.append({
                    "id": prompt_id,
                    "prompt": source_prompt,
                    "mapped": "",
                    "mapping_type": "",
                    "note": "all candidates empty",
                    "clip_score": 0.0,
                })
                continue

            valid_texts = [candidate_texts[j] for j in valid_indices]
            scores = clip_scorer.score_pairs(source_prompt, valid_texts)

            # Find best
            best_local_idx = max(range(len(scores)), key=lambda k: scores[k])
            best_global_idx = valid_indices[best_local_idx]
            best_cand = cands[best_global_idx]
            best_score = scores[best_local_idx]

            mapping_type = best_cand.get("mapping_type", "replace")
            replacement = best_cand.get("replacement", "")
            note = f"{mapping_type}: {replacement} (clip={best_score:.4f})"

            all_results.append({
                "id": prompt_id,
                "prompt": source_prompt,
                "mapped": best_cand["prompt"],
                "mapping_type": mapping_type,
                "note": note,
                "clip_score": round(best_score, 4),
            })

            if save_candidates:
                all_candidates_data.append({
                    "id": prompt_id,
                    "source": source_prompt,
                    "candidates": [
                        {
                            "prompt": candidate_texts[j],
                            "mapping_type": cands[j].get("mapping_type", "replace"),
                            "replacement": cands[j].get("replacement", ""),
                            "clip_score": round(scores[valid_indices.index(j)], 4) if j in valid_indices else None,
                        }
                        for j in range(len(cands))
                    ],
                    "selected": best_global_idx + 1,
                })

        print(f"  Scored and selected best for batch {batch_start+1}-{batch_start+len(batch_rows)}.")

    # ── Save map_clip.csv (same format as map_100.csv) ──
    out_csv = concept_dir / "map_clip.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prompt", "mapping_type", "note"])
        for r in all_results:
            writer.writerow([r["id"], r["mapped"], r["mapping_type"], r["note"]])

    print(f"  [SAVED] {out_csv}")

    # ── Optionally save all candidates JSON for debugging ──
    if save_candidates and all_candidates_data:
        cand_json = concept_dir / "candidates_clip.json"
        with open(cand_json, "w", encoding="utf-8") as f:
            json.dump(all_candidates_data, f, ensure_ascii=False, indent=2)
        print(f"  [SAVED] {cand_json}")

    # Print summary stats
    scores_list = [r["clip_score"] for r in all_results if r["clip_score"] > 0]
    if scores_list:
        avg = sum(scores_list) / len(scores_list)
        print(f"  [STATS] Avg CLIP score: {avg:.4f}  "
              f"Min: {min(scores_list):.4f}  Max: {max(scores_list):.4f}")


def get_all_concepts(prompts_root: Path) -> List[str]:
    """Discover all concept folders under prompts_root that have train_100.csv."""
    concepts = []
    for d in sorted(prompts_root.iterdir()):
        if d.is_dir() and (d / "train_100.csv").exists():
            concepts.append(d.name)
    return concepts


def main():
    ap = argparse.ArgumentParser(
        description="Generate CLIP-scored mapping prompts for concept unlearning."
    )
    ap.add_argument(
        "--concept",
        default=None,
        help="Single concept to process (folder name or display name). "
             "If omitted, processes ALL concepts in prompts_new/.",
    )
    ap.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model for candidate generation (default: gpt-4o-mini).",
    )
    ap.add_argument(
        "--clip_model",
        default="openai/clip-vit-large-patch14",
        help="CLIP model for scoring (default: openai/clip-vit-large-patch14).",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of prompts to batch per API call (default: 10).",
    )
    ap.add_argument(
        "--prompts_root",
        default="prompts_new",
        help="Root directory containing concept folders.",
    )
    ap.add_argument(
        "--save_candidates",
        action="store_true",
        help="Save all candidates + scores to candidates_clip.json for debugging.",
    )
    ap.add_argument(
        "--nudity_filter",
        action="store_true",
        help="Add nudity body-part terms (breast, genitalia, buttocks, feet, "
             "belly, armpits, etc.) to the banned-words list. Use when "
             "unlearning nudity to ensure candidates stay clothed/safe.",
    )
    ap.add_argument(
        "--extra_banned_words",
        nargs="*",
        default=None,
        help="Additional words to ban from candidates (space-separated).",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="Device for CLIP (cuda/cpu). Auto-detected if omitted.",
    )
    args = ap.parse_args()

    prompts_root = Path(args.prompts_root)
    if not prompts_root.exists():
        print(f"ERROR: {prompts_root} does not exist.", file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)

    # Determine concept list
    if args.concept:
        concepts = [args.concept]
    else:
        concepts = get_all_concepts(prompts_root)
        if not concepts:
            print("No concept folders with train_100.csv found.", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(concepts)} concepts: {', '.join(concepts)}")

    # Build extra banned word list
    extra_banned = list(args.extra_banned_words or [])
    if args.nudity_filter:
        extra_banned.extend(NUDITY_BANNED_WORDS)
        print(f"[nudity-filter] Added {len(NUDITY_BANNED_WORDS)} nudity body-part terms to ban list.")
    extra_banned = extra_banned or None

    # Initialize CLIP scorer once (lazy-loaded on first use)
    clip_scorer = CLIPScorer(model_name=args.clip_model, device=args.device)

    for concept in concepts:
        try:
            process_concept(
                concept=concept,
                prompts_root=prompts_root,
                model=args.model,
                batch_size=args.batch_size,
                clip_scorer=clip_scorer,
                save_candidates=args.save_candidates,
                extra_banned=extra_banned,
            )
        except Exception as e:
            print(f"[ERROR] Concept '{concept}' failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue

    print("\n[DONE] All concepts processed.")


if __name__ == "__main__":
    main()
