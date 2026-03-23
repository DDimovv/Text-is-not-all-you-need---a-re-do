import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------- CONFIG ----------------

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
MAX_NEW_TOKENS = 200

TEXT_HET_IN = "cache/phase1_text_only_raw.heterographic.jsonl"
TEXT_HOM_IN = "cache/phase1_text_only_raw.homographic.jsonl"

AUDIO_HET_IN = "cache/phase2_audio_raw.heterographic.jsonl"
AUDIO_HOM_IN = "cache/phase2_audio_raw.homographic.jsonl"

TEXT_ALL_OUT = "cache/phase3_text.jsonl"
TEXT_HET_OUT = "cache/phase3_text.heterographic.jsonl"
TEXT_HOM_OUT = "cache/phase3_text.homographic.jsonl"

AUDIO_ALL_OUT = "cache/phase3_audio.jsonl"
AUDIO_HET_OUT = "cache/phase3_audio.heterographic.jsonl"
AUDIO_HOM_OUT = "cache/phase3_audio.homographic.jsonl"

JSONIFY_PROMPT = """You are a classification system.

STRICT RULES:
- Output ONLY a single JSON object
- No markdown
- No extra text
- Use EXACT strings for Choice

Task:
You are given an explanation about whether a text is a pun.

1. Decide if the explanation concludes the text IS or IS NOT a pun
2. Rewrite the explanation into a clean, concise reason

Output format (EXACT):
{{
  "Reason": "<clean explanation>",
  "Choice": "The text is a pun" | "The text is not a pun"
}}

Explanation:
{reason}
"""

# ------------------MODEL----------------

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).eval()

#  ---------------HELPERS--------------- 

def generate_json(reason_text: str):
    prompt = JSONIFY_PROMPT.format(reason=reason_text)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    start, end = decoded.find("{"), decoded.rfind("}")
    if start == -1 or end == -1:
        return None

    try:
        obj = json.loads(decoded[start:end+1])
        if obj.get("Choice") in {"The text is a pun", "The text is not a pun"}:
            return obj
    except Exception:
        pass

    return None


def process_split(in_path, out_all, out_split, reason_key):
    with open(in_path, encoding="utf-8") as f:
        items = [json.loads(l) for l in f]

    with open(out_all, "a", encoding="utf-8") as f_all, \
         open(out_split, "w", encoding="utf-8") as f_split:

        for item in tqdm(items, desc=f"JSONifying {in_path}"):
            raw_reason = item.get(reason_key)
            if not raw_reason:
                continue

            parsed = generate_json(raw_reason)
            if not parsed:
                parsed = {"Reason": None, "Choice": None}

            out_obj = {
                "id": item["id"],
                "Text": item["Text"],
                "Reason": parsed["Reason"],
                "Choice": parsed["Choice"],
            }

            line = json.dumps(out_obj, ensure_ascii=False) + "\n"
            f_all.write(line)
            f_split.write(line)

# ------------------RUN -----------------
def main():
    # TEXT
    process_split(TEXT_HET_IN, TEXT_ALL_OUT, TEXT_HET_OUT, "RawReason")
    process_split(TEXT_HOM_IN, TEXT_ALL_OUT, TEXT_HOM_OUT, "RawReason")

    # AUDIO
    process_split(AUDIO_HET_IN, AUDIO_ALL_OUT, AUDIO_HET_OUT, "RawReasonAudio")
    process_split(AUDIO_HOM_IN, AUDIO_ALL_OUT, AUDIO_HOM_OUT, "RawReasonAudio")

    print("Done.")
    print("TEXT  ->", TEXT_ALL_OUT)
    print("AUDIO ->", AUDIO_ALL_OUT)

if __name__ == "__main__":
    main()
