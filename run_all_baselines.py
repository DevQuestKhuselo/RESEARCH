"""
Runs baseline translation across ALL required directions and splits,
loading the LLaMAX3-8B-Alpaca model only ONCE (not once per direction).

Usage:
    python run_all_baselines.py
"""

import os
import csv
from baseline_translate import load_model, translate_one

LANG_NAMES = {"en": "English", "fr": "French", "sw": "Swahili", "zu": "Zulu"}

DIRECTIONS = ["en-fr", "en-sw", "en-zu", "sw-en", "zu-en"]
SPLITS = ["dev", "devtest"]

DATA_DIR = "data"
RESULTS_DIR = "results"


def run_direction(tokenizer, model, direction, split):
    src, tgt = direction.split("-")
    src_lang, tgt_lang = LANG_NAMES[src], LANG_NAMES[tgt]

    input_csv = os.path.join(DATA_DIR, f"flores200_{direction}_{split}.csv")
    output_csv = os.path.join(RESULTS_DIR, f"baseline_{direction}_{split}.csv")

    if os.path.exists(output_csv):
        print(f"  Skipping {direction} ({split}) -- already done: {output_csv}")
        return

    with open(input_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"=== {direction} ({split}): {src_lang} -> {tgt_lang}, {len(rows)} sentences ===")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "reference", "hypothesis"])
        for i, row in enumerate(rows):
            hypothesis = translate_one(tokenizer, model, row["source"], src_lang, tgt_lang)
            writer.writerow([row["source"], row["target"], hypothesis])
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(rows)} done")

    print(f"  Wrote {output_csv}")


if __name__ == "__main__":
    tokenizer, model = load_model()  # loaded ONCE, reused for every direction/split

    for direction in DIRECTIONS:
        for split in SPLITS:
            run_direction(tokenizer, model, direction, split)

    print("\nAll baseline translations complete.")
