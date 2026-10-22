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
import time

import sacrebleu
from comet import download_model, load_from_checkpoint

COMET_MODEL_ID = "Unbabel/wmt22-comet-da"
AFRICOMET_MODEL_ID = "masakhane/africomet-stl-1.1"


def load_one(model_id):
    """With HF_HUB_OFFLINE=1 (set by run_evaluate.sh) everything must already
    be cached by run_prefetch_eval.sh; a scoring job then never waits on the
    network (an uncached download hung for 8 h on 20-22 Oct 2026)."""
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    print(f"Loading {model_id} ({'offline, from cache' if offline else 'may download'}) ...", flush=True)
    try:
        path = download_model(model_id, local_files_only=offline)
    except KeyError as e:
        if offline:
            raise RuntimeError(f"{model_id} is not cached. Run `sbatch run_prefetch_eval.sh` first.") from e
        raise
    return load_from_checkpoint(path, local_files_only=offline)


def load_comet_models():
    return load_one(COMET_MODEL_ID), load_one(AFRICOMET_MODEL_ID)


def read_triples(path):
    sources, references, hypotheses = [], [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sources.append(row["source"])
            references.append(row["reference"])
            hypotheses.append(row["hypothesis"])
    return sources, references, hypotheses


def score_file(path, comet_model, africomet_model, batch_size=32):
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

    comet_result = comet_model.predict(comet_data, batch_size=batch_size, gpus=1, progress_bar=False)
    africomet_result = africomet_model.predict(comet_data, batch_size=batch_size, gpus=1, progress_bar=False)

    return {
        "bleu": bleu.score,
        "chrf++": chrf.score,
        "comet": comet_result.system_score,
        "africomet": africomet_result.system_score,
    }


FIELDS = ["file", "bleu", "chrf++", "comet", "africomet"]


def read_partial(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {row["file"]: row for row in csv.DictReader(f)}


def run(results_dir, output_csv, pattern="baseline_*.csv", batch_size=32):
    """Scores each file and appends it to {output_csv}.partial immediately, so a
    job that hits its time limit loses at most one file. Resubmitting the same
    command skips files already in the .partial file. The final summary is
    written to output_csv only once every file is scored (so an interrupted
    re-score never half-overwrites an existing summary)."""
    files = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not files:
        print(f"No files matching {pattern} found in {results_dir}")
        return

    partial_csv = output_csv + ".partial"
    done = read_partial(partial_csv)
    todo = [p for p in files if os.path.basename(p) not in done]
    print(f"{len(files)} files match {pattern}: {len(done)} already scored, {len(todo)} to do")

    if todo:
        comet_model, africomet_model = load_comet_models()
        for path in todo:
            filename = os.path.basename(path)
            print(f"\nScoring {filename} ...", flush=True)
            t0 = time.time()
            scores = score_file(path, comet_model, africomet_model, batch_size)
            scores["file"] = filename
            new_file = not os.path.exists(partial_csv)
            with open(partial_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                if new_file:
                    writer.writeheader()
                writer.writerow(scores)
            done[filename] = scores
            print(f"  BLEU={scores['bleu']:.2f}  chrF++={scores['chrf++']:.2f}  "
                  f"COMET={scores['comet']:.4f}  AfriCOMET={scores['africomet']:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for path in files:
            writer.writerow({k: done[os.path.basename(path)][k] for k in FIELDS})
    os.remove(partial_csv)
    print(f"\nWrote summary to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output", default="scores_summary.csv")
    parser.add_argument("--pattern", default="baseline_*.csv",
                         help="Glob pattern to select which result files to score "
                              "(e.g. 'quantised_8bit_*.csv' later on)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="COMET batch size (was 8; larger is faster on a big GPU)")
    args = parser.parse_args()

    run(args.results_dir, args.output, args.pattern, args.batch_size)
