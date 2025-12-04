import argparse, csv, json, os, re, sys, time
from pathlib import Path
from typing import Any, Dict, Tuple, List

# --- OpenAI SDK (official) ---
from openai import OpenAI

# ========== Meta-prompt (system) ==========

FIXED_MAP_SYS_PROMPT = """
You are a dataset prompt writer for diffusion models (e.g., Stable Diffusion).
Your job is to produce short, simple, natural-language prompts that are easy for diffusion models to parse and that depict a clearly visible instance of the specified concept in-frame.

GOAL
Produce a JSON dataset where each TRAIN pair has:
- "src": a short natural prompt that visibly depicts the INPUT_CONCEPT.
- "mapped": the SAME scene but with INPUT_CONCEPT replaced by MAPPED_CONCEPT.
- "mapping_type": "replace" in almost all cases; use "remove" ONLY if a literal replacement would be grammatically impossible, visually implausible, or unsafe.

GENERAL STYLE
- 5–12 words, present tense, everyday vocabulary, ≤1 comma, ≤2 adjectives.
- No tags/parameters (no aspect ratios, seeds, CFG, hashtags).
- Minimal punctuation; keep sentences simple and independent.

VISUAL GROUNDING
- INPUT_CONCEPT must be a visible, concrete subject in "src".
- The "mapped" sentence must remain depictable and keep the same scene framing and action.

FIXED MAPPING RULES (MANDATORY)
- INPUT_CONCEPT is given as CONCEPT.
- MAPPED_CONCEPT is given as MAPPED_CONCEPT.
- Replace INPUT_CONCEPT tokens with MAPPED_CONCEPT while preserving:
  * Number/plurality (“a dog” → “a cat”, “two dogs” → “two cats”).
  * Determiners and articles (“a/an/one/two/many”); fix a/an as needed.
  * Core syntax and structure of the sentence.
- The INPUT_CONCEPT might have different names and types also try to replace them with MAPPED_CONCEPT.(For example, if INPUT_CONCEPT is "dog", then "puppy","canine" or "German Shepherd" in the prompts. All these variations should be replaced with the corresponding form of MAPPED_CONCEPT, such as "a kitten", "cat", or types of cat.)
- Do NOT invent a different replacement; always use exactly MAPPED_CONCEPT if sensible.
- If replacement would be nonsensical/unsafe (rare), set mapping_type="remove" and drop the concept phrase; keep the rest intact.

CONCEPT TYPES
1) OBJECT/ANIMAL: replace with MAPPED_CONCEPT as the new subject. Keep count and determiners aligned.
2) PERSON (generic or named): keep neutral content. If MAPPED_CONCEPT is a named person, use the name neutrally; if generic (e.g., “a person”, “a runner”), use it verbatim.
3) ARTISTIC STYLE: if INPUT_CONCEPT is a style phrase (e.g., “in Van Gogh style”), swap just the style to “in MAPPED_CONCEPT style” and keep the scene identical.
4) SAFETY-SENSITIVE: only neutral, non-graphic content. If literal replacement is unsafe, use mapping_type="remove".

DIVERSITY
- Across the N training pairs: vary indoor/outdoor, day/night, weather, actions, distance (near/far), solo vs small group.
- No near-duplicates; avoid trivial rewording.

VALIDATION (20 prompts)
- New scenes not used in training.
- Must visibly depict the INPUT_CONCEPT (not the mapped one).
- Same brevity and style constraints.

OUTPUT FORMAT (STRICT JSON)
{
  "concept": "<string, as given>",
  "train_count": <int N>,
  "train_pairs": [
    {
      "id": <int, 1-based>,
      "src": "<short prompt including the INPUT_CONCEPT>",
      "mapped": "<short prompt where INPUT_CONCEPT is replaced by MAPPED_CONCEPT or removed>",
      "mapping_type": "<'replace' | 'remove'>",
      "note": "<very short rationale (e.g., 'dog→cat; same scene')>"
    },
    ...
  ],
  "validation_count": 20,
  "validation_prompts": ["<short prompt>", "..."]
}
"""

SYS_PROMPT = """
You are a dataset prompt writer for diffusion models (e.g., Stable Diffusion).
Your job is to produce short, simple, natural-language prompts that are easy for diffusion models to parse and that depict a clearly visible instance of the specified concept in-frame.

====================================================
GENERAL STYLE
====================================================
- 4–10 words per prompt.
- One simple main clause; present tense preferred.
- Everyday vocabulary only; 0–1 adjectives.
- No tags/parameters (no colons, aspect ratios, seeds, CFG, hashtags).
- Minimal punctuation (commas/periods only when needed).

====================================================
VISUAL GROUNDING (MANDATORY for both TRAIN and VALIDATION)
====================================================
- The concept MUST be a **visible, concrete subject** in the image.
- If the concept is polysemous, use the **physical object** sense (e.g., “apple” = fruit, **not** the company).
- The scene must make the concept **clearly depictable and in-frame**.
  Prefer contexts where the subject is visible (e.g., not hidden, microscopic, or distant).
- Match number and determiners consistently (a/an/one/two/many).

====================================================
MAPPED COUNTERPARTS (CONTEXT-PRESERVING REPLACEMENT OR REMOVAL)
====================================================
You must produce a “mapped” counterpart for each training prompt by **replacing the specified concept** with a **context-appropriate, visually plausible subject**, while preserving the sentence’s structure and scene.

- If a **natural replacement** makes the prompt more realistic or coherent (even if it changes category, e.g., object → person), prefer that.
- If no sensible replacement exists, **REMOVE** the concept tokens and keep the rest intact.
- The mapped prompt must describe a **visually depictable** subject or scene (no abstractions like “happiness” or “freedom”).
- Preserve number/determiners and the grammatical shape of the sentence.
- The mapped prompt must remain **realistic and visually grounded** — not surreal or symbolic.

====================================================
CONCEPT TYPES & MAPPING RULES
====================================================
1) FLEXIBLE REPLACEMENT RULE (Default)
   - Choose a replacement that fits the **scene context**, even if it’s from a different semantic category.
     Example:
       "An apple sitting on a beach" → "A person sitting on a beach" ✅
       (cross-category but visually coherent)

2) SAME-CATEGORY REPLACEMENT (When Naturally Fits)
   - Replace with a category sibling when it makes contextual sense.
     Example:
       "A dog chasing a ball" → "A cat chasing a ball"

3) PERSON OR NAMED ENTITY
   - Replace with a neutral, visually grounded role or human activity.
     Example:
       "Barack Obama giving a speech" → "A teacher giving a speech"

4) ARTISTIC STYLE OR ARTIST NAME
   - If the concept refers to a style or artist, either:
       a) DROP the style phrase but keep the rest of the scene, or
       b) SWAP to a generic style label like “cartoon style” or “realistic photo”.
     Example:
       "A landscape in Van Gogh style" → "A landscape in cartoon style"

5) SAFETY-SENSITIVE CONTENT (weapons, nudity, harm)
   - All training prompts must be **non-graphic, non-violent, and legal**.
   - Mapped version should replace with a **benign analogue** (e.g., “knife” → “spoon”) or remove the phrase entirely.

====================================================
DIVERSITY REQUIREMENTS
====================================================
- Use **at least 5 distinct replacement concepts** across the entire mapped dataset.
- Replacement concepts should vary in category (person, animal, object, etc.).
- Ensure contextual appropriateness — avoid trivial visual similarity as the only reason for mapping.
- Across prompts, balance:
  - Indoor vs. outdoor
  - Day vs. night
  - Weather conditions
  - Simple everyday activities (walking, cooking, reading, sitting, playing, riding, shopping, etc.)
  - Solitary vs. small-group scenes
  - Near vs. far framing

====================================================
VALIDATION SET (20 PROMPTS)
====================================================
- Create 20 **distinct** simple prompts containing the concept.
- Each must satisfy:
  - 4–8 words
  - Natural, realistic, everyday phrasing
  - The concept must be **visible in-frame**
- No overlaps or trivial paraphrases of training prompts.

====================================================
OUTPUT FORMAT (STRICT JSON)
====================================================
Return a single JSON object with the following structure:

{
  "concept": "<string, as given>",
  "train_count": <int N>,
  "train_pairs": [
    {
      "id": <int, 1-based>,
      "src": "<short prompt including the concept>",
      "mapped": "<short prompt with concept replaced or removed>",
      "mapping_type": "<'replace' | 'remove'>",
      "note": "<very short rationale for the chosen mapped concept>"
    },
    ...
  ],
  "validation_count": 20,
  "validation_prompts": ["<short prompt>", "..."]
}
"""

RELATED_CONCEPTS_PROMPT = """You are a helpful assistant tasked with finding related concepts.
Given an input concept, provide a list of exactly 10 distinct concepts that are related to it, with good diversity:
- Include a mix of:
  1. **Close or sibling items** (e.g., similar type or category) (Around 4-6 items),
  2. **Broader category terms** (e.g., the class or group it belongs to) (2-4 items),
  3. **Contextual or analogical items** (things that commonly co-occur, appear together, or share thematic context) (2-4 items)
- (Very Important) Are **not synonyms, subtypes, accessories, or activities that directly imply the INPUT_CONCEPT**.(for example, if INPUT is "dog", do not include "leash" or "bone" or "puppy").
- Avoid concepts which contains or strongly imply the INPUT_CONCEPT.
- Use simple, common concepts that are easy for image generation models to understand.
- Avoid rare, obscure, technical, or brand-specific terms. 
- Include broad generic concepts if relevant (e.g., "fruit", "vehicle", "person", "animal").
- If the input is a specific person or celebrity, include a few other similar famous individuals or colleagues, and also include a generic person descriptor (e.g., "a man", "a woman", or a related role).
- If the input is an artistic style or medium, include other popular art styles or mediums.
- If the input is a weapon or harmful concept, list benign alternatives in a similar context (e.g., replace weapons with cameras, replace fighting with sports).
- **Output format:** Return exactly 10 concepts, one per line, with no explanations or numbering.
"""

PROMPT_GENERATION_PROMPT = """You are a prompt generator for Stable Diffusion models.
Follow these rules and tasks to produce a list of prompts:

**General Constraints (apply to all prompts):**
- Each prompt must be 5–10 words long, using simple, everyday vocabulary.
- Use present tense and natural grammar; use ≤1 comma and ≤1 adjective per prompt.
- No special tags or parameters (no ":" or "--ar 16:9" etc.).
- **Do not include the INPUT concept or any obvious synonym/nickname for it in any prompt.**

**Task Distribution:**
You will be given a list of 10 RELATED_CONCEPTS (numbered 1–10) for the INPUT concept.
Create prompts as follows:
- For each of the 10 concepts: generate **10 prompts** (total **100** prompts).
- Then generate **50 additional random prompts** unrelated to the INPUT concept or its direct domain.

**Content Guidelines:**
- For prompts using the related concepts: describe simple, plausible scenes or subjects involving that concept (e.g., common actions like walking, reading, playing; or typical settings).
- Ensure variety in scenes and wording: mix indoor/outdoor settings, day/night times, different weather, and a range of activities.
- Do **not** repeat the same sentence structure across prompts or only swap the subject; avoid near-duplicates or trivial paraphrasing across all prompts.
- For the 50 unrelated prompts: choose topics completely outside the INPUT concept’s category. (They should not mention or strongly imply the INPUT concept or the related concepts.)
- Keep all prompts **safe and neutral** in tone (no sexualization of real people, no graphic violence, etc.).

**Output Format:**
- Output a total of **150 prompts**, one prompt per line (no blank lines).
- Do not include any numbering, bullet points, or section titles. Just provide the prompts as a continuous list.
"""

# ========== Helper functions ==========

def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_{2,}", "_", text).strip("_")
    return text or "concept"

def extract_json_block(s: str) -> str:
    """
    Robustly extract a JSON object from LLM output.
    Handles ```json ...``` fences or plain text with a single top-level object.
    """
    fence = re.search(r"```json\s*(\{.*?\})\s*```", s, re.S | re.I)
    if fence:
        return fence.group(1)
    # Fallback: first {...} that parses
    # Try progressive trimming from the start and end
    first = s.find("{")
    last = s.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("No JSON object detected in model output.")
    # attempt the widest range first, then shrink if needed
    for i in range(first, first + 10):
        for j in range(last, max(last - 200, i + 1), -1):
            block = s[i:j+1]
            try:
                json.loads(block)
                return block
            except Exception:
                continue
    # Final attempt: greedy capture
    block = s[first:last+1]
    json.loads(block)  # will raise if invalid
    return block

def validate_payload(payload: Dict[str, Any], concept: str, n: int) -> Tuple[int, int]:
    if not isinstance(payload, dict):
        raise ValueError("Top-level JSON must be an object.")
    required_top = ["concept", "train_count", "train_pairs", "validation_count", "validation_prompts"]
    for k in required_top:
        if k not in payload:
            raise ValueError(f"Missing key: {k}")
    if payload["concept"] is None or not str(payload["concept"]).strip():
        raise ValueError("Field 'concept' is empty.")
    if int(payload["train_count"]) != n:
        raise ValueError(f"train_count != N (got {payload['train_count']} vs {n})")
    if not isinstance(payload["train_pairs"], list) or len(payload["train_pairs"]) != n:
        raise ValueError(f"train_pairs length != N (got {len(payload['train_pairs'])} vs {n})")
    if int(payload["validation_count"]) != 20 or len(payload["validation_prompts"]) != 20:
        raise ValueError("validation_count/prompts must be exactly 20.")

    # quick field sanity
    for p in payload["train_pairs"]:
        for key in ["id", "src", "mapped", "mapping_type", "note"]:
            if key not in p:
                raise ValueError(f"train_pairs item missing '{key}'.")
    
    return len(payload["train_pairs"]), len(payload["validation_prompts"])

def save_outputs(base_dir: Path, data: Dict[str, Any], suffix: str = None) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    # Full JSON
    with open(base_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Determine filenames with optional suffix
    train_filename = f"train_{suffix}.csv" if suffix else "train.csv"
    map_filename = f"map_{suffix}.csv" if suffix else "map.csv"
    val_filename = f"val_{suffix}.csv" if suffix else "val.csv"

    # Train (source)
    with open(base_dir / train_filename, "w", newline="", encoding="utf-8") as f1:
        w = csv.writer(f1)
        w.writerow(["id", "prompt"])
        for p in data["train_pairs"]:
            w.writerow([p["id"], p["src"]])

    # Train (mapped)
    with open(base_dir / map_filename, "w", newline="", encoding="utf-8") as f2:
        w = csv.writer(f2)
        w.writerow(["id", "prompt", "mapping_type", "note"])
        for p in data["train_pairs"]:
            w.writerow([p["id"], p["mapped"], p["mapping_type"], p["note"]])

    # Validation
    with open(base_dir / val_filename, "w", newline="", encoding="utf-8") as fv:
        w = csv.writer(fv)
        w.writerow(["id", "prompt"])
        for i, s in enumerate(data["validation_prompts"], 1):
            w.writerow([i, s])

def _strip_leading_markers(s: str) -> str:
    # Remove common numbering/bullets/quotes if the model slips
    return re.sub(r'^\s*(?:-|\*|\d+[\).\s]|["“”]+)\s*', '', s).strip()

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

def parse_prompts_lines(raw_text: str) -> List[str]:
    # Split raw text into lines and clean formatting
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        # Remove any leading bullet or number if present
        ln = _strip_leading_markers(ln)
        if ln:
            cleaned.append(ln)
    return cleaned

def call_openai(concept: str, n: int, model: str, mapped_concept: str = None, sys_prompt: str = None, max_tokens: int = 25000, retries: int = 1, backoff: float = 2.0, verbosity: str="low", reasoning_effort: str="medium") -> str:
    client = OpenAI()  # uses OPENAI_API_KEY env var
    user_prompt = f'CONCEPT = "{concept}"\nN = {n}.'
    if mapped_concept:
        user_prompt += f'\nMAPPED_CONCEPT = "{mapped_concept}"'
    user_prompt += "\nProduce output exactly per the JSON schema."
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "developer", "content": [{"type": "input_text", "text": (sys_prompt or SYS_PROMPT)}]},
                    {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
                ],
                max_output_tokens=max_tokens,
                text={"format": {"type": "json_object"}, "verbosity": verbosity},
                reasoning={"effort": reasoning_effort},
            )

            collected: list[str] = []

            # Preferred: walk resp.output if present
            output_list = getattr(resp, "output", None)
            if output_list:
                for item in output_list:
                    # Some SDK versions wrap textual content in a 'content' list
                    content = getattr(item, "content", None)
                    if content:
                        for c in content:
                            txt = getattr(c, "text", None)
                            if txt:
                                collected.append(txt)
                    # Direct text node possibility
                    direct_txt = getattr(item, "text", None)
                    if direct_txt:
                        collected.append(direct_txt)

            # Fallback: unified text aggregation property
            if not collected:
                unified = getattr(resp, "output_text", None)
                if unified:
                    collected.append(unified)

            if not collected:
                raise ValueError("No textual output found in response object (resp.output and resp.output_text empty).")

            return "".join(collected).strip()
        except Exception as e:
            last_err = e
            # exponential backoff with jitter
            time.sleep((backoff ** attempt) + 0.1 * (attempt + 1))
    raise RuntimeError(f"OpenAI API call failed after {retries} attempts: {last_err}")

def get_related_concepts(concept: str, model: str, max_tokens: int = 1000) -> List[str]:
    """Call the API to get a list of 10 related concepts for the input concept."""
    client = OpenAI()
    # Prepare the prompt with the system (developer) instructions and user message as the concept.
    response = client.responses.create(
        model=model,
        input=[
            {"role": "developer", "content": [{"type": "input_text", "text": RELATED_CONCEPTS_PROMPT}]},
            {"role": "user", "content": [{"type": "input_text", "text": concept}]},
        ],
        max_output_tokens=max_tokens,
        text={"format": {"type": "text"}, "verbosity": "low"},
        reasoning={"effort": "minimal"},
    )
    # Collect and clean the output text
    raw_output = ""
    if hasattr(response, "output"):
        for part in response.output:
            if hasattr(part, "content") and part.content:
                for c in part.content:
                    if hasattr(c, "text"):
                        raw_output += c.text
            elif hasattr(part, "text"):
                raw_output += part.text
    else:
        raw_output = getattr(response, "output_text", "")
    raw_output = raw_output.strip()
    # Split into lines and take up to 10 (in case model returns extra or fewer)
    concepts = [ln.strip() for ln in raw_output.splitlines() if ln.strip()]
    if len(concepts) < 10:
        combined = ",".join(concepts)
        parts = [x.strip() for x in combined.split(",") if x.strip()]
        if len(parts) >= 10:
            concepts = parts[:10]
    concepts = concepts[:10]
    if len(concepts) < 10:
        concepts += ["placeholder"] * (10 - len(concepts))  # filler if needed (should not usually happen)
    return concepts

def generate_prompts_from_concepts(concept: str, related_list: List[str], model: str, max_tokens: int = 15000) -> str:
    """Call the API to generate prompts given a related concepts list and input concept."""
    client = OpenAI()
    # Format the user content with the concept and the numbered related concepts list.
    related_lines = "\n".join(f"{i+1}. {rc}" for i, rc in enumerate(related_list))
    user_content = f"INPUT concept: {concept}\nRELATED_CONCEPTS:\n{related_lines}"
    # Make the API call with the prompt generation instructions
    response = client.responses.create(
        model=model,
        input=[
            {"role": "developer", "content": [{"type": "input_text", "text": PROMPT_GENERATION_PROMPT}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_content}]},
        ],
        max_output_tokens=max_tokens,
        text={"format": {"type": "text"}, "verbosity": "low"},
        reasoning={"effort": "medium"},
    )
    # Collect the output text from the response
    output_text = ""
    if hasattr(response, "output"):
        for part in response.output:
            if hasattr(part, "content") and part.content:
                for c in part.content:
                    if hasattr(c, "text"):
                        output_text += c.text
            elif hasattr(part, "text"):
                output_text += part.text
    else:
        output_text = getattr(response, "output_text", "")
    return output_text.strip()

def main():
    ap = argparse.ArgumentParser(description="Generate SD prompts + mapped counterparts via OpenAI.")
    ap.add_argument("--concept", required=True, help="Concept string (object/animal/person/style/safety term).")
    ap.add_argument("--task", choices=["mapped", "related","fixed_map"], required=True, help="Task type: 'mapped' produces JSON dataset; 'related' produces prompts-only.")
    ap.add_argument("--n", type=int, help="Number of TRAIN prompt pairs.")
    ap.add_argument("--model", default="gpt-5", help="OpenAI model (e.g., gpt-4o-mini, gpt-4o).")
    ap.add_argument("--max_tokens", type=int, default=25000, help="Max tokens for completion.")
    ap.add_argument("--out_root", default="/prompts", help="Root output directory.")
    ap.add_argument("--map_to", help="Fixed target concept for mapped prompts (e.g., 'cat'). Required for --task mapped.")
    ap.add_argument("--suffix", help="Optional suffix to append to output filenames (e.g., train_50.csv).")
    args = ap.parse_args()

    slug = slugify(args.concept)
    out_dir = Path(args.out_root) / slug
    concept = args.concept
    model = args.model

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)

    if args.task == "mapped":
        raw = call_openai(
            concept=args.concept, n=args.n, model=args.model
        )

        try:
            json_str = extract_json_block(raw)
            payload = json.loads(json_str)
        except Exception as e:
            print("---- RAW MODEL OUTPUT ----")
            print(raw)
            print("---- END RAW OUTPUT ----")
            raise RuntimeError(f"Failed to parse JSON from model output: {e}") from e
            
        # Validate
        train_len, val_len = validate_payload(payload, args.concept, args.n)

        # Save
        save_outputs(out_dir, payload, args.suffix)

        print(f"[OK] concept={args.concept!r}  train={train_len}  val={val_len}")
        print(f"Saved to: {out_dir.resolve()}")
    
    elif args.task == "fixed_map":
        if not args.map_to:
            print("ERROR: --map_to is required for --task fixed_map.", file=sys.stderr)
            sys.exit(2)

        raw = call_openai(
            concept=args.concept,
            n=args.n,
            model=args.model,
            mapped_concept=args.map_to,
            sys_prompt=FIXED_MAP_SYS_PROMPT,  # <-- use the fixed mapping prompt
            max_tokens=args.max_tokens,
        )

        try:
            json_str = extract_json_block(raw)
            payload = json.loads(json_str)
        except Exception as e:
            print("---- RAW MODEL OUTPUT ----")
            print(raw)
            print("---- END RAW OUTPUT ----")
            raise RuntimeError(f"Failed to parse JSON from model output: {e}") from e

        train_len, val_len = validate_payload(payload, args.concept, args.n)
        save_outputs(out_dir, payload, args.suffix)
        print(f"[OK] concept={args.concept!r}  map_to={args.map_to!r}  train={train_len}  val={val_len}")
        print(f"Saved to: {out_dir.resolve()}")

    else:
        # Step 1: Get related concepts list
        related_concepts = get_related_concepts(concept, model=model)
        # Print the related concepts to terminal for verification
        print(f"Related concepts for '{concept}': {', '.join(related_concepts)}")
        # Step 2: Generate prompts using the related concepts
        raw_prompts = generate_prompts_from_concepts(concept, related_concepts, model=model, max_tokens=args.max_tokens)
        prompts = parse_prompts_lines(raw_prompts)
        # Save all prompts to a single output file
        out_file = out_dir / "related.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            for line in prompts:
                f.write(line + "\n")
        # Log success message
        print(f"[OK] concept='{concept}' – generated {len(prompts)} prompts.", file=sys.stderr)
        print(f"Saved prompts to: {out_file.resolve()}", file=sys.stderr)

if __name__ == "__main__":
    main()
