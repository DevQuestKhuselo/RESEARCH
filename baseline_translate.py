"""
Baseline translation script for:
  Evaluating the Impact of Quantisation and Pruning on Multilingual LLM
  Translation for Low-Resource African Languages

Loads LLaMAX/LLaMAX3-8B-Alpaca and generates translations for a given
FLORES-200 direction CSV (built earlier by build_flores_csvs.py), saving
source / reference / hypothesis triples for scoring with BLEU, chrF++,
COMET, and AfriCOMET later.

Requirements:
    pip install torch transformers accelerate

Usage:
    python baseline_translate.py \
        --input data/flores200_en-sw_devtest.csv \
        --src_lang English --tgt_lang Swahili \
        --output results/baseline_en-sw_devtest.csv

Run once per translation direction. Language names must match the
model's "Supported Languages" naming (e.g. 'Swahili', 'Zulu', 'English',
'French') -- see the LLaMAX3-8B-Alpaca model card.
"""

import argparse
import csv
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "LLaMAX/LLaMAX3-8B-Alpaca"


def prompt_template(query, src_language, trg_language):
    """Official LLaMAX prompt format -- must match this exactly for
    the model to behave as evaluated in the LLaMAX paper."""
    instruction = f"Translate the following sentences from {src_language} to {trg_language}."
    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n"
        f"### Instruction:\n{instruction}\n"
        f"### Input:\n{query}\n### Response:"
    )
    return prompt


def load_model(model_path=MODEL_ID, dtype=torch.float16, device_map="auto"):
    """Load the base model (default) or a compressed checkpoint directory.

    GPTQ checkpoints saved by quantise_gptq.py carry a `quantization_config`
    in config.json; with `gptqmodel` installed, Transformers loads them
    through the same from_pretrained call, so the generation code is
    identical for baseline and quantised runs.
    """
    print(f"Loading {model_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device_map
    )
    model.eval()
    return tokenizer, model


def translate_one(tokenizer, model, sentence, src_lang, tgt_lang, max_new_tokens=256):
    prompt = prompt_template(sentence, src_lang, tgt_lang)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # greedy decoding for reproducible baseline scores
        )

    # Only decode the newly generated tokens (strip the prompt back off)
    new_tokens = generate_ids[0][inputs["input_ids"].shape[-1]:]
    output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return output.strip()


def run(input_csv, src_lang, tgt_lang, output_csv, limit=None):
    with open(input_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if limit:
        rows = rows[:limit]

    print(f"Translating {len(rows)} sentences: {src_lang} -> {tgt_lang}")

    tokenizer, model = load_model()

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "reference", "hypothesis"])

        for i, row in enumerate(rows):
            src_sentence = row["source"]
            reference = row["target"]
            hypothesis = translate_one(tokenizer, model, src_sentence, src_lang, tgt_lang)
            writer.writerow([src_sentence, reference, hypothesis])

            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(rows)} done")

    print(f"Done. Wrote {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to FLORES-200 direction CSV")
    parser.add_argument("--src_lang", required=True, help="e.g. English, Swahili, Zulu, French")
    parser.add_argument("--tgt_lang", required=True, help="e.g. English, Swahili, Zulu, French")
    parser.add_argument("--output", required=True, help="Path to write results CSV")
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional: only translate first N sentences (useful for a quick smoke test)")
    args = parser.parse_args()

    run(args.input, args.src_lang, args.tgt_lang, args.output, args.limit)
