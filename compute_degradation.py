"""
Relative degradation (Delta) of compressed models vs the uncompressed baseline,
per direction, split and metric, plus averages by language-resource category.

    abs_delta = compressed - baseline            (negative = worse)
    rel_delta = 100 * (compressed - baseline) / baseline   (%)

Usage:
    python compute_degradation.py \
        --baseline scores_baseline_summary.csv \
        --compressed scores_gptq8_summary.csv scores_gptq4_summary.csv \
        --output degradation_gptq.csv

Score files are the output of evaluate.py: columns file,bleu,chrf++,comet,africomet,
with file names like {prefix}_{direction}_{split}.csv.
"""

import argparse
import csv
import re
from collections import defaultdict

METRICS = ["bleu", "chrf++", "comet", "africomet"]

CATEGORY = {
    "en-fr": "high->high",
    "en-sw": "high->low",
    "en-zu": "high->low",
    "sw-en": "low->high",
    "zu-en": "low->high",
}

NAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<direction>[a-z]{2}-[a-z]{2})_(?P<split>dev|devtest)\.csv$")


def read_scores(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = NAME_RE.match(row["file"])
            if not m:
                continue
            key = (m["prefix"], m["direction"], m["split"])
            out[key] = {k: float(row[k]) for k in METRICS}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="scores_baseline_summary.csv")
    p.add_argument("--compressed", nargs="+", required=True)
    p.add_argument("--output", default="degradation_summary.csv")
    p.add_argument("--headline_split", default="devtest",
                   help="Split used for the printed category table (dev was used for GPTQ calibration)")
    args = p.parse_args()

    base = {(d, s): v for (_, d, s), v in read_scores(args.baseline).items()}
    comp = {}
    for path in args.compressed:
        comp.update(read_scores(path))

    rows = []
    for (prefix, direction, split), scores in sorted(comp.items()):
        if (direction, split) not in base:
            print(f"WARNING: no baseline for {direction} {split}, skipping")
            continue
        for metric in METRICS:
            b, c = base[(direction, split)][metric], scores[metric]
            rows.append({
                "config": prefix,
                "direction": direction,
                "category": CATEGORY.get(direction, "?"),
                "split": split,
                "metric": metric,
                "baseline": round(b, 4),
                "compressed": round(c, 4),
                "abs_delta": round(c - b, 4),
                "rel_delta_pct": round(100 * (c - b) / b, 2) if b else None,
            })

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}\n")

    # Printed summary: mean relative Delta (%) by config x category, headline split
    agg = defaultdict(list)
    for r in rows:
        if r["split"] == args.headline_split:
            agg[(r["config"], r["category"], r["metric"])].append(r["rel_delta_pct"])
    configs = sorted({k[0] for k in agg})
    cats = ["high->high", "high->low", "low->high"]
    print(f"Mean relative Delta (%) on {args.headline_split} (negative = degradation)")
    print(f"{'config':<10}{'category':<12}" + "".join(f"{m:>11}" for m in METRICS))
    for cfg in configs:
        for cat in cats:
            vals = [agg.get((cfg, cat, m)) for m in METRICS]
            if not any(vals):
                continue
            cells = "".join(f"{sum(v) / len(v):>11.2f}" if v else f"{'-':>11}" for v in vals)
            print(f"{cfg:<10}{cat:<12}{cells}")


if __name__ == "__main__":
    main()
