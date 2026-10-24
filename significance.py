"""
Paired bootstrap significance for compression-induced degradation
(BLEU, chrF++, and COMET / AfriCOMET when segment scores exist).

For each compressed config and direction, sentences are resampled with
replacement (the same indices for the uncompressed and compressed outputs, so
the test is paired) and the relative change

    rel_delta = 100 * (compressed - base) / base

is recomputed on every resample. Reported per config x direction x metric:
  * the point estimate and its 95% percentile interval,
  * p_degrad: share of resamples with rel_delta >= 0 (one-sided p for
    "compression lowers the score").

The study's hypothesis is tested directly as a *disparity*: for each
low-resource direction d,

    disparity_d = rel_delta_d - rel_delta_en-fr     (negative = d degrades more)

with directions resampled independently. Its 95% interval excluding 0 means
direction d degrades significantly more than the high-resource control.

COMET and AfriCOMET use the segment-level scores evaluate.py writes to
results/segscores/ (system score = mean of segment scores); a metric is skipped
for any file without them.

CPU only, no GPU or network. Usage:
    python significance.py --configs gptq8 gptq4 mag30 --output significance_devtest.csv
"""

import argparse
import csv
import os
import zlib

import numpy as np
from sacrebleu.metrics import BLEU, CHRF

DIRECTIONS = ["en-fr", "en-sw", "en-zu", "sw-en", "zu-en"]
CONTROL = "en-fr"
SACRE = {"bleu": lambda: BLEU(), "chrf++": lambda: CHRF(word_order=2)}
NEURAL = ["comet", "africomet"]
METRICS = list(SACRE) + NEURAL


class MeanScorer:
    """COMET-style metric: corpus score = mean of segment scores."""
    def __init__(self, n):
        self.n = n


def read_segscores(results_dir, prefix, direction, split, column):
    path = os.path.join(results_dir, "segscores", f"{prefix}_{direction}_{split}.csv")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return np.array([float(r[column]) for r in csv.DictReader(f)])


def read(results_dir, prefix, direction, split):
    path = os.path.join(results_dir, f"{prefix}_{direction}_{split}.csv")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["hypothesis"] for r in rows], [r["reference"] for r in rows]


def sentence_stats(metric, hyps, refs):
    """Per-sentence sufficient statistics, so a corpus score for any resample
    is just a sum of rows (exactly matches sacrebleu's corpus score)."""
    return np.array(metric._extract_corpus_statistics(hyps, [refs]), dtype=np.float64)


def corpus_score(metric, stats):
    if isinstance(metric, MeanScorer):
        return float(stats.mean())
    return metric._compute_score_from_stats(list(stats.sum(axis=0))).score


def boot_scores(metric, stats, idx):
    """Corpus scores for each resample (rows of idx)."""
    if isinstance(metric, MeanScorer):
        return stats[idx].mean(axis=1)
    return np.array([metric._compute_score_from_stats(list(stats[i].sum(axis=0))).score for i in idx])


def stream(seed, *key):
    """Independent, reproducible random stream per (metric, config, direction):
    adding a model or direction never changes the other results."""
    return np.random.default_rng([seed, zlib.crc32("|".join(key).encode())])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="fp16", help="Uncompressed reference outputs")
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--split", default="devtest")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--n", type=int, default=1000, help="Bootstrap resamples")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--output", default="significance_devtest.csv")
    args = p.parse_args()

    rows = []
    for metric_name in METRICS:
        for cfg in args.configs:
            boot_rel, point_rel = {}, {}
            for d in DIRECTIONS:
                base = read(args.results_dir, args.base, d, args.split)
                comp = read(args.results_dir, cfg, d, args.split)
                if base is None or comp is None:
                    continue
                if len(comp[1]) != len(base[1]):
                    print(f"  skipping {cfg} {d}: {len(comp[1])}/{len(base[1])} sentences (incomplete run)")
                    continue
                if base[1] != comp[1]:
                    raise ValueError(f"{cfg} {d}: references differ from {args.base}; outputs not aligned")
                if metric_name in SACRE:
                    metric = SACRE[metric_name]()
                    sb = sentence_stats(metric, *base)
                    sc = sentence_stats(metric, *comp)
                else:
                    sb = read_segscores(args.results_dir, args.base, d, args.split, metric_name)
                    sc = read_segscores(args.results_dir, cfg, d, args.split, metric_name)
                    if sb is None or sc is None:
                        continue
                    if len(sb) != len(base[1]) or len(sc) != len(comp[1]):
                        raise ValueError(f"{cfg} {d}: segment scores do not match the translations")
                    metric = MeanScorer(len(sb))
                n = len(sb)
                idx = stream(args.seed, metric_name, cfg, d).integers(0, n, size=(args.n, n))
                b_base, b_comp = boot_scores(metric, sb, idx), boot_scores(metric, sc, idx)
                s_base, s_comp = corpus_score(metric, sb), corpus_score(metric, sc)
                rel = 100 * (b_comp - b_base) / np.where(b_base == 0, np.nan, b_base)
                boot_rel[d] = rel
                point_rel[d] = 100 * (s_comp - s_base) / s_base
                lo, hi = np.nanpercentile(rel, [2.5, 97.5])
                rows.append({
                    "metric": metric_name, "config": cfg, "direction": d, "split": args.split,
                    "base": round(s_base, 4), "compressed": round(s_comp, 4),
                    "abs_delta": round(s_comp - s_base, 4),
                    "rel_delta_pct": round(point_rel[d], 2),
                    "rel_ci95_lo": round(lo, 2), "rel_ci95_hi": round(hi, 2),
                    "p_degrad": round(float(np.nanmean(rel >= 0)), 4),
                    "disparity_vs_en-fr": "", "disp_ci95_lo": "", "disp_ci95_hi": "",
                    "p_disparity": "",
                })
            if CONTROL in boot_rel:
                for r in rows:
                    if r["metric"] == metric_name and r["config"] == cfg and r["direction"] != CONTROL \
                            and r["direction"] in boot_rel:
                        d = r["direction"]
                        # independent resamples per direction -> difference distribution
                        perm = stream(args.seed, metric_name, cfg, d, "disparity").permutation(args.n)
                        diff = boot_rel[d] - boot_rel[CONTROL][perm]
                        lo, hi = np.nanpercentile(diff, [2.5, 97.5])
                        r["disparity_vs_en-fr"] = round(point_rel[d] - point_rel[CONTROL], 2)
                        r["disp_ci95_lo"], r["disp_ci95_hi"] = round(lo, 2), round(hi, 2)
                        r["p_disparity"] = round(float(np.nanmean(diff >= 0)), 4)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print(f"Paired bootstrap, n={args.n}, seed={args.seed}, split={args.split}, base={args.base}")
    for metric_name in METRICS:
        if not any(r["metric"] == metric_name for r in rows):
            print(f"\n[{metric_name}] no segment scores found -- run run_evaluate.sh to create results/segscores/")
            continue
        print(f"\n[{metric_name}] rel. change % [95% CI]   | disparity vs en-fr, pp [95% CI]")
        for r in rows:
            if r["metric"] != metric_name:
                continue
            disp = (f"{r['disparity_vs_en-fr']:>7} [{r['disp_ci95_lo']}, {r['disp_ci95_hi']}]"
                    if r["disparity_vs_en-fr"] != "" else "   (control)")
            print(f"  {r['config']:<6} {r['direction']:<6} {r['rel_delta_pct']:>7} "
                  f"[{r['rel_ci95_lo']}, {r['rel_ci95_hi']}]  p={r['p_degrad']:<6} | {disp}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
