"""
Runs translation across ALL required directions and splits, loading the
model only ONCE. Works for the baseline and for compressed checkpoints.

Usage:
    python run_all_baselines.py                                   # baseline
    python run_all_baselines.py --model_path models/llamax3-8b-gptq-4bit --prefix gptq4
    python run_all_baselines.py --model_path models/llamax3-8b-gptq-8bit --prefix gptq8

Timing benchmark (short, meant to run with sbatch --exclusive on one node):
    python run_all_baselines.py --model_path ... --prefix timing_gptq4 --splits devtest --limit 100

Outputs:
    results/{prefix}_{direction}_{split}.csv      source / reference / hypothesis
    results/efficiency_{prefix}.csv               latency, tokens/s, GPU memory

Resume safety: a direction/split is skipped only if its output file exists
AND has one row per input sentence. A partial file (e.g. a leftover smoke
test, see PROGRESS.md section 7) is re-translated instead of silently kept.
"""

import argparse
import csv
import os
import time

import torch

from baseline_translate import MODEL_ID, load_model, translate_one

LANG_NAMES = {"en": "English", "fr": "French", "sw": "Swahili", "zu": "Zulu"}

DIRECTIONS = ["en-fr", "en-sw", "en-zu", "sw-en", "zu-en"]
SPLITS = ["dev", "devtest"]

DATA_DIR = "data"
RESULTS_DIR = "results"


def count_rows(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def is_complete(output_csv, expected_rows):
    if not os.path.exists(output_csv):
        return False
    got = count_rows(output_csv)
    if got != expected_rows:
        print(f"  {output_csv} has {got}/{expected_rows} rows -- re-translating")
        return False
    return True


def run_direction(tokenizer, model, prefix, direction, split, limit=None):
    src, tgt = direction.split("-")
    src_lang, tgt_lang = LANG_NAMES[src], LANG_NAMES[tgt]

    input_csv = os.path.join(DATA_DIR, f"flores200_{direction}_{split}.csv")
    output_csv = os.path.join(RESULTS_DIR, f"{prefix}_{direction}_{split}.csv")

    with open(input_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]

    if is_complete(output_csv, len(rows)):
        print(f"  Skipping {direction} ({split}) -- complete: {output_csv}")
        return None

    print(f"=== [{prefix}] {direction} ({split}): {src_lang} -> {tgt_lang}, {len(rows)} sentences ===")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    n_new_tokens = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "reference", "hypothesis"])
        for i, row in enumerate(rows):
            hypothesis = translate_one(tokenizer, model, row["source"], src_lang, tgt_lang)
            n_new_tokens += len(tokenizer(hypothesis, add_special_tokens=False)["input_ids"])
            writer.writerow([row["source"], row["target"], hypothesis])
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(rows)} done")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    print(f"  Wrote {output_csv} ({elapsed / 60:.1f} min)")
    return {
        "prefix": prefix,
        "direction": direction,
        "split": split,
        "sentences": len(rows),
        "seconds": round(elapsed, 1),
        "sec_per_sentence": round(elapsed / len(rows), 4),
        "output_tokens_per_sec": round(n_new_tokens / elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_ID,
                        help="HF model id or local checkpoint dir (default: base model)")
    parser.add_argument("--prefix", default="baseline",
                        help="Output file prefix, e.g. baseline, gptq8, gptq4")
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only the first N sentences per direction (timing benchmarks)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tokenizer, model = load_model(args.model_path)  # loaded ONCE
    load_mem_gb = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"Model resident GPU memory: {load_mem_gb:.2f} GB on {gpu}" if load_mem_gb else "No GPU")

    stats = []
    for direction in DIRECTIONS:
        for split in args.splits:
            s = run_direction(tokenizer, model, args.prefix, direction, split, args.limit)
            if s:
                stats.append(s)

    if stats:
        peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None
        eff_path = os.path.join(RESULTS_DIR, f"efficiency_{args.prefix}.csv")
        new_file = not os.path.exists(eff_path)
        with open(eff_path, "a", newline="", encoding="utf-8") as f:
            fields = list(stats[0].keys()) + ["gpu", "model_mem_gb", "peak_mem_gb"]
            w = csv.DictWriter(f, fieldnames=fields)
            if new_file:
                w.writeheader()
            for s in stats:
                s.update(gpu=gpu,
                         model_mem_gb=round(load_mem_gb, 2) if load_mem_gb else None,
                         peak_mem_gb=round(peak_gb, 2) if peak_gb else None)
                w.writerow(s)
        print(f"Efficiency stats appended to {eff_path}")

    print(f"\nAll [{args.prefix}] translations complete.")


if __name__ == "__main__":
    main()
