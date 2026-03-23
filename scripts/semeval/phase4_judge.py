import json
import torch
from tqdm import tqdm
from collections import Counter

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

#--------------- CONFIG ---------------

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_NEW_TOKENS = 80

TEXT_HET = "cache/phase3_text.heterographic.jsonl"
TEXT_HOM = "cache/phase3_text.homographic.jsonl"

AUDIO_HET = "cache/phase3_audio.heterographic.jsonl"
AUDIO_HOM = "cache/phase3_audio.homographic.jsonl"

OUT_HET = "cache/phase4_judge_text_vs_audio.heterographic.jsonl"
OUT_HOM = "cache/phase4_judge_text_vs_audio.homographic.jsonl"

# ----------------JUDGE PROMPT -----------------

JUDGE_PROMPT = """You are a strict evaluator of linguistic explanations.

Your task:
Given a text and two explanations, decide which explanation better identifies
whether the text is a pun AND explains the linguistic mechanism correctly.

Rules:
- Do NOT prefer an explanation because it appears first.
- Do NOT reward verbosity.
- Prefer correctness, clarity, and accurate identification of wordplay.
- If both explanations are equally good or equally weak, choose a tie.

Return ONLY valid JSON in exactly this format:
{{"Choice": "Explanation 1 is much better" | "Explanation 2 is much better" | "Explanation 1 and 2 are of similar quality",
 "Reason": "<short justification>"}}

Text:
{text}

Explanation 1:
{exp1}

Explanation 2:
{exp2}
"""

# ----------------- MODEL (FP16, NO QUANT) -----------------

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()

# ---------------- HELPERS -----------------

def load_map(path):
    with open(path, encoding="utf-8") as f:
        return {x["id"]: x for x in map(json.loads, f)}

def generate_judge(prompt: str):
    messages = [
        {"role": "system", "content": "You are a judge that outputs ONLY valid JSON."},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]

    decoded = tokenizer.decode(
        out[0][prompt_len:],
        skip_special_tokens=True,
    )
    print(decoded)
    # robust JSON extraction
    start, end = decoded.find("{"), decoded.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(decoded[start:end + 1])
        except Exception:
            pass

    return {"Choice": "INVALID", "Reason": "Parse failure"}

# ---------------- RUN -----------------

def run_judge(text_path, audio_path, out_path, label):
    print(f"\n=== Judging {label} ===")

    text_items = load_map(text_path)
    audio_items = load_map(audio_path)

    ids = sorted(set(text_items) & set(audio_items))
    votes = Counter()

    with open(out_path, "w", encoding="utf-8") as f:
        for i in tqdm(ids):
            t = text_items[i]
            a = audio_items[i]

            prompt = JUDGE_PROMPT.format(
                text=t["Text"],
                exp1=t["Reason"],
                exp2=a["Reason"],
            )

            judge = generate_judge(prompt)
            choice = judge.get("Choice", "INVALID")
            votes[choice] += 1

            out = {
                "id": i,
                "type": label,
                "judge": judge,
                "text_reason": t["Reason"],
                "audio_reason": a["Reason"],
            }

            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # ------- PRINT STATS ---------
    total = sum(votes.values())
    print(f"\nResults for {label} (n={total})")
    for k, v in votes.items():
        pct = (v / total * 100) if total else 0.0
        print(f"  {k}: {v} ({pct:.1f}%)")

    print("Wrote:", out_path)

# ----------------- MAIN -----------------

if __name__ == "__main__":
    run_judge(TEXT_HET, AUDIO_HET, OUT_HET, "heterographic")
    run_judge(TEXT_HOM, AUDIO_HOM, OUT_HOM, "homographic")
