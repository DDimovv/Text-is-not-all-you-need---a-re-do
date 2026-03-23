import os
import json
import random
import torch
from tqdm import tqdm

from transformers import (
    AutoProcessor,
)
from transformers import Qwen2AudioForConditionalGeneration

# ---------------- CONFIG -------------------

HF_DATASET = "frostymelonade/SemEval2017-task7-pun-detection"
HF_SPLIT = "test"

OUT_BASE = "cache/phase1_text_only_raw"
OUT_ALL = OUT_BASE + ".jsonl"
OUT_HET = OUT_BASE + ".heterographic.jsonl"
OUT_HOM = OUT_BASE + ".homographic.jsonl"

TYPES = {"heterographic", "homographic"}
PER_TYPE = 250
SEED = 42

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
MAX_NEW_TOKENS = 120

# ---------------- PROMPT ----------------

def build_messages(text):
    return [
        {
            "role": "system",
            "content": "You are an expert linguist."
        },
        {
            "role": "user",
            "content": f"""Explain whether the following text contains a pun.

Instructions:
- Do NOT explain your analysis process.
- Do NOT define what a pun is.
- Focus ONLY on the linguistic mechanism.
- If the text is a pun, clearly state:
  • the word or phrase involved
  • the two meanings or sound-based ambiguity
- If it is not a pun, clearly state that no wordplay or ambiguity is present.

Write a concise paragraph (3–6 sentences).

Text:
{text}
"""
        }
    ]

# ---------------- HELPERS ----------------

def normalize_id(x):
    return str(x).strip() if x else None

def load_done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(l).get("id") for l in f}

# ---------------- MAIN ----------------

def main():
    device = "cuda"
    dtype = torch.float16
    torch.set_grad_enabled(False)


    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=dtype,
    ).eval()


    from datasets import load_dataset
    ds = load_dataset(HF_DATASET, split=HF_SPLIT)

    # ---- filter by type ----
    items = []
    for row in ds:
        if row.get("type") in TYPES:
            items.append({
                "id": normalize_id(row.get("id")),
                "text": row.get("text", ""),
                "type": row.get("type"),
                "label": row.get("label"),
            })

    print(f"After type filter: {len(items)}")

    # ---- per-type sampling ----
    grouped = {}
    for x in items:
        grouped.setdefault(x["type"], []).append(x)

    rng = random.Random(SEED)
    sampled = []
    for t in sorted(grouped.keys()):
        rng.shuffle(grouped[t])
        sampled.extend(grouped[t][:PER_TYPE])

    items = sampled


    # ---- output setup ----
    os.makedirs(os.path.dirname(OUT_ALL), exist_ok=True)
    done_all = load_done_ids(OUT_ALL)

    def generate(text: str) -> str:
        messages = build_messages(text)

        prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=prompt,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=40,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        gen_tokens = out[0][prompt_len:]

        return processor.tokenizer.decode(
            gen_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

    with open(OUT_ALL, "a", encoding="utf-8") as f_all, \
         open(OUT_HET, "a", encoding="utf-8") as f_het, \
         open(OUT_HOM, "a", encoding="utf-8") as f_hom:

        for item in tqdm(items, desc="Phase 1 (Qwen2-Audio, text-only)"):
            i = item["id"]
            if not i or i in done_all:
                continue

            text = item["text"]
            if not text:
                continue

            reason = generate(text)

            out_obj = {
                "id": i,
                "Text": text,
                "RawReason": reason,
                "Label": item["label"],
                "Type": item["type"],
            }

            line = json.dumps(out_obj, ensure_ascii=False) + "\n"

            f_all.write(line)
            if item["type"] == "heterographic":
                f_het.write(line)
            elif item["type"] == "homographic":
                f_hom.write(line)

            f_all.flush()
            f_het.flush()
            f_hom.flush()

            torch.cuda.empty_cache()

    print("Done.")
    print(f"ALL -> {OUT_ALL}")
    print(f"HET -> {OUT_HET}")
    print(f"HOM -> {OUT_HOM}")

# ---------------- RUN ----------------

if __name__ == "__main__":
    main()
