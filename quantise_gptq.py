"""
GPTQ post-training quantisation of LLaMAX3-8B-Alpaca at 8-bit or 4-bit,
using GPT-QModel (`gptqmodel`), the maintained successor to AutoGPTQ.

Calibration data is a BALANCED mix of the four study languages
(English, French, Swahili, isiZulu), drawn from the FLORES-200 *dev* split.
English-only calibration (the tutorial default) would bias GPTQ's error
correction toward English activations -- a confound this study must avoid.

Methodological notes
--------------------
* Each language gets a DIFFERENT set of FLORES sentence IDs, so the
  calibration set is not just one sentence seen four times.
* Because calibration uses `dev`, the `dev` translation scores of the
  quantised models are not fully independent of calibration. Report
  `devtest` as the headline split; `dev` is a secondary check.
* The exact sentence IDs used are written to
  `{out_dir}/calibration_ids.json` for reproducibility.

Requirements (in the `research` conda env, on a GPU node):
    pip install -v gptqmodel

Usage:
    python quantise_gptq.py --bits 4 --out_dir models/llamax3-8b-gptq-4bit
    python quantise_gptq.py --bits 8 --out_dir models/llamax3-8b-gptq-8bit
"""

import argparse
import json
import os
import random
import time

MODEL_ID = "LLaMAX/LLaMAX3-8B-Alpaca"
FLORES_CODES = ["eng_Latn", "fra_Latn", "swh_Latn", "zul_Latn"]


def load_flores_split(flores_root, split, code):
    path = os.path.join(flores_root, split, f"{code}.{split}")
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def build_calibration(flores_root, n_per_lang, seed):
    """Return (texts, ids_by_lang). Disjoint sentence IDs per language."""
    per_lang = {c: load_flores_split(flores_root, "dev", c) for c in FLORES_CODES}
    n_sentences = len(per_lang[FLORES_CODES[0]])
    needed = n_per_lang * len(FLORES_CODES)
    if needed > n_sentences:
        raise ValueError(f"Need {needed} distinct sentences, dev has {n_sentences}")

    rng = random.Random(seed)
    ids = rng.sample(range(n_sentences), needed)

    texts, ids_by_lang = [], {}
    for i, code in enumerate(FLORES_CODES):
        chunk = sorted(ids[i * n_per_lang:(i + 1) * n_per_lang])
        ids_by_lang[code] = chunk
        texts.extend(per_lang[code][j] for j in chunk)

    rng.shuffle(texts)  # interleave languages
    return texts, ids_by_lang


def dir_size_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total / 1024 ** 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, choices=[4, 8], required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--flores_root", default="flores200_dataset")
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument("--n_per_lang", type=int, default=32,
                   help="Calibration sentences per language (total = 4x this)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=1)
    args = p.parse_args()

    # Imported here so --help works without a GPU environment
    import torch
    from gptqmodel import GPTQModel, QuantizeConfig

    texts, ids_by_lang = build_calibration(args.flores_root, args.n_per_lang, args.seed)
    print(f"Calibration: {len(texts)} sentences "
          f"({args.n_per_lang} x {len(FLORES_CODES)} languages, seed={args.seed})")

    quant_config = QuantizeConfig(bits=args.bits, group_size=args.group_size)
    print(f"Loading {MODEL_ID} for {args.bits}-bit GPTQ (group_size={args.group_size}) ...")
    model = GPTQModel.load(MODEL_ID, quant_config)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model.quantize(texts, batch_size=args.batch_size)
    quant_seconds = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None

    os.makedirs(args.out_dir, exist_ok=True)
    model.save(args.out_dir)

    meta = {
        "base_model": MODEL_ID,
        "bits": args.bits,
        "group_size": args.group_size,
        "calibration_split": "dev",
        "calibration_n_per_lang": args.n_per_lang,
        "seed": args.seed,
        "calibration_ids": ids_by_lang,
        "quantisation_seconds": round(quant_seconds, 1),
        "peak_gpu_mem_gb_during_quant": round(peak_gb, 2) if peak_gb else None,
        "checkpoint_size_gb": round(dir_size_gb(args.out_dir), 3),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gptqmodel_version": __import__("gptqmodel").__version__,
    }
    with open(os.path.join(args.out_dir, "calibration_ids.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {args.bits}-bit model to {args.out_dir} "
          f"({meta['checkpoint_size_gb']} GB, {quant_seconds / 60:.1f} min)")


if __name__ == "__main__":
    main()
