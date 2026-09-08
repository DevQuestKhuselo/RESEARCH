"""
Scores translation results (produced by baseline_translate.py / run_all_baselines.py)
using BLEU, chrF++, COMET, and AfriCOMET.

Requirements:
    pip install sacrebleu unbabel-comet

Usage:
    python evaluate.py --results_dir results --output scores_summary.csv

Reads every results/baseline_{direction}_{split}.csv (or quantised_*/pruned_*
variants later) and computes all four metrics per file, writing one row per
file into a summary CSV.
"""

import argparse
import csv
import glob
import os

import sacrebleu
from comet import download_model, load_from_checkpoint

COMET_MODEL_ID = "Unbabel/wmt22-comet-da"
AFRICOMET_MODEL_ID = "masakhane/africomet-stl-1.1"


def load_comet_models():
    print(f"Loading COMET model ({COMET_MODEL_ID}) ...")
    comet_path = download_model(COMET_MODEL_ID)
    comet_model = load_from_checkpoint(comet_path)

    print(f"Loading AfriCOMET model ({AFRICOMET_MODEL_ID}) ...")
    africomet_path = download_model(AFRICOMET_MODEL_ID)
    africomet_model = load_from_checkpoint(africomet_path)

    return comet_model, africomet_model


def read_triples(path):
    sources, references, hypotheses = [], [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sources.append(row["source"])
            references.append(row["reference"])
            hypotheses.append(row["hypothesis"])
    return sources, references, hypotheses


def score_file(path, comet_model, africomet_model):
    sources, references, hypotheses = read_triples(path)

    # BLEU (sacrebleu expects references as list-of-lists: one list per ref set)
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])

    # chrF++ (word_order=2 is the "chrF++" variant, vs plain chrF at word_order=0)
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)

    # COMET / AfriCOMET expect a list of dicts with src/mt/ref
    comet_data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]

    comet_result = comet_model.predict(comet_data, batch_size=8, gpus=1)
    africomet_result = africomet_model.predict(comet_data, batch_size=8, gpus=1)

    return {
        "bleu": bleu.score,
        "chrf++": chrf.score,
        "comet": comet_result.system_score,
        "africomet": africomet_result.system_score,
    }


def run(results_dir, output_csv, pattern="baseline_*.csv"):
    files = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not files:
        print(f"No files matching {pattern} found in {results_dir}")
        return

    comet_model, africomet_model = load_comet_models()

    rows = []
    for path in files:
        filename = os.path.basename(path)
        print(f"\nScoring {filename} ...")
        scores = score_file(path, comet_model, africomet_model)
        scores["file"] = filename
        rows.append(scores)
        print(f"  BLEU={scores['bleu']:.2f}  chrF++={scores['chrf++']:.2f}  "
              f"COMET={scores['comet']:.4f}  AfriCOMET={scores['africomet']:.4f}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "bleu", "chrf++", "comet", "africomet"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote summary to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output", default="scores_summary.csv")
    parser.add_argument("--pattern", default="baseline_*.csv",
                         help="Glob pattern to select which result files to score "
                              "(e.g. 'quantised_8bit_*.csv' later on)")
    args = parser.parse_args()

    run(args.results_dir, args.output, args.pattern)
