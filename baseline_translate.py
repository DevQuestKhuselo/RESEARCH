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
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "LLaMAX/LLaMAX3-8B-Alpaca"

# GPT-QModel kernel used when loading GPTQ checkpoints. "auto" picked Marlin on
# mscluster106 (Quadro RTX 8000), which must be JIT-compiled and failed there
# (no CUDA_HOME). gptq_torch needs no compilation and runs on any GPU;
# override with e.g. GPTQ_BACKEND=gptq_triton.
GPTQ_BACKEND_DEFAULT = "gptq_torch"


def gptq_loading_override(model_path):
    """GPTQConfig that only overrides the kernel (backend) of a saved GPTQ
    checkpoint; bits, group size etc. still come from its config.json."""
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    with open(cfg_path) as f:
        qc = json.load(f).get("quantization_config") or {}
    if str(qc.get("quant_method", "")).lower() != "gptq":
        return None
    from transformers import GPTQConfig
    backend = os.environ.get("GPTQ_BACKEND", GPTQ_BACKEND_DEFAULT)
    print(f"GPTQ checkpoint ({qc.get('bits')}-bit): using GPT-QModel backend '{backend}'")
    return GPTQConfig(bits=qc["bits"], backend=backend)


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


def load_model(model_path=MODEL_ID, dtype=torch.float16, device_map=None):
    """Load the base model (default) or a compressed checkpoint directory.

    GPTQ checkpoints saved by quantise_gptq.py carry a `quantization_config`
    in config.json; with `gptqmodel` installed, Transformers loads them
    through the same from_pretrained call, so the generation code is
    identical for baseline and quantised runs.
    """
    # Place the whole model on the GPU explicitly. With device_map="auto",
    # accelerate silently falls back to CPU when a GPU fails its probe
    # (seen on mscluster111, 20 Oct 2026), and GPTQ kernels then crash with
    # "Expected a cuda device, but got: cpu". An explicit "cuda" map makes a
    # broken GPU fail loudly at load time instead.
    if device_map is None:
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    extra = {}
    override = gptq_loading_override(model_path)
    if override is not None:
        extra["quantization_config"] = override
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device_map, **extra
    )
    model.eval()
    # GPTQ layers keep their quantised weights in buffers, so check both
    devices = {t.device.type for t in [*model.parameters(), *model.buffers()]}
    print(f"Model tensors on: {sorted(devices)}")
    if device_map == "cuda" and devices != {"cuda"}:
        raise RuntimeError(f"Expected the whole model on the GPU, got {sorted(devices)}")
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
