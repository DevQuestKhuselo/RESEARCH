# Evaluating the Impact of Quantisation and Pruning on Multilingual LLM Translation for Low-Resource African Languages

**Author:** Khuselo Sofohlo
**Supervisors:** Dr Steven James, Prof Benjamin Rosman
**Institution:** School of Computer Science and Applied Mathematics, University of the Witwatersrand
**Course:** COMS4057A Research Project

## Overview

This research investigates whether post-training quantisation (GPTQ) and magnitude-based pruning disproportionately degrade multilingual translation quality for low-resource African language pairs compared to high-resource language pairs in large language models.

Model compression is essential for efficient LLM deployment, but most compression research evaluates degradation on English and high-resource benchmarks, implicitly assuming uniform degradation across languages. This assumption may not hold for low-resource languages with weaker cross-lingual representations. This project systematically evaluates that assumption using empirical translation-quality measurements before and after compression.

## Research Question

To what extent do quantisation and pruning disproportionately affect translation quality for low-resource African language pairs compared to high-resource language pairs in multilingual large language models?

## Hypothesis

Low-resource translation directions will experience significantly larger degradation in translation quality after quantisation and pruning than high-resource translation directions, due to weaker and less robust multilingual representations.

## Model

**[LLaMAX3-8B-Alpaca](https://huggingface.co/LLaMAX/LLaMAX3-8B-Alpaca)** — an instruction-tuned, decoder-only LLM built on continued multilingual pretraining of Llama-3-8B, supporting 100+ languages including Swahili and isiZulu.

> **Note on model selection:** the original proposal specified Aya-8B (Aya Expanse). During setup it was confirmed that Aya Expanse 8B only supports 23 languages, none of which are isiZulu or Swahili, making it incompatible with this study's translation directions. LLaMAX3-8B-Alpaca was selected as a replacement because (a) it explicitly supports both required low-resource languages, and (b) as a decoder-only Llama-family architecture, it has strong existing tooling support for GPTQ quantisation, unlike encoder-decoder alternatives (e.g. Aya-101).

## Translation Directions

| Category | Direction |
|---|---|
| High-resource → high-resource | English → French |
| High-resource → low-resource | English → Swahili, English → isiZulu |
| Low-resource → high-resource | Swahili → English, isiZulu → English |

## Datasets

- **[FLORES-200](https://github.com/facebookresearch/flores)** — standardised multilingual evaluation benchmark (dev/devtest splits), professionally translated, sentence-aligned across all required languages.
- **[Masakhane LAFAND-MT](https://github.com/masakhane-io/lafand-mt)** — community-built, news-domain parallel corpora for African languages, used as a supplementary in-domain evaluation set.

## Compression Techniques

- **Quantisation:** GPTQ post-training quantisation at 8-bit and 4-bit precision.
- **Pruning:** Unstructured global magnitude-based pruning at 30% and 50% sparsity.

## Evaluation Metrics

- **Translation quality:** BLEU, chrF++, COMET, AfriCOMET
- **Efficiency:** model size reduction, compression ratio, GPU memory usage, inference latency
- **Analysis:** relative degradation (Δ) computed per metric and compression configuration, compared across language-resource categories

## Repository Structure

```
.
├── data/                      # Parallel translation-direction CSVs (generated, not committed)
├── results/                   # Baseline and compressed-model translation outputs + scores
├── build_flores_csvs.py       # Builds per-direction parallel CSVs from a local FLORES-200 copy
├── baseline_translate.py      # Runs baseline (uncompressed) translation with LLaMAX3-8B-Alpaca
├── quantise_gptq.py           # GPTQ quantisation at 8-bit / 4-bit (gptqmodel, multilingual calibration)
├── run_all_baselines.py       # Translate all directions with any model (--model_path, --prefix)
├── compute_degradation.py     # Relative degradation (Δ) vs baseline, by resource category
├── significance.py            # Paired bootstrap CIs and low- vs high-resource disparity test
├── run_quantise.sh / run_prune.sh / run_translate.sh / run_timing.sh / run_evaluate.sh / run_prefetch_eval.sh   # SLURM jobs (see HPC_RUNBOOK.md)
├── prune_magnitude.py         # Global unstructured magnitude pruning (30% / 50%), run_prune.sh
├── evaluate.py                # Scores translations with BLEU / chrF++ / COMET / AfriCOMET
└── README.md
```

## Setup

```bash
pip install torch transformers accelerate sacrebleu gptqmodel   # translation + quantisation env
pip install unbabel-comet                                          # separate evaluation env
```

### 1. Build the dataset

Download FLORES-200 manually (see [flores repo](https://github.com/facebookresearch/flores/blob/main/flores200/README.md)), then:

```bash
python build_flores_csvs.py <path_to_flores200_dataset_folder>
```

### 2. Run the baseline translation pipeline

```bash
python baseline_translate.py \
    --input data/flores200_en-sw_devtest.csv \
    --src_lang English --tgt_lang Swahili \
    --output results/baseline_en-sw_devtest.csv
```

Repeat for each translation direction (en-fr, en-sw, en-zu, sw-en, zu-en).

### 3. Apply compression and re-evaluate

See **[HPC_RUNBOOK.md](HPC_RUNBOOK.md)** for the full cluster workflow (setup, GPTQ quantisation, translation, scoring, Δ analysis).

Quantisation and pruning scripts, along with the evaluation/scoring pipeline, are being added as the project progresses through the phases below.

## Research Plan

| Phase | Period | Focus |
|---|---|---|
| 1 | June | Environment setup, baseline evaluation |
| 2 | July | GPTQ quantisation experiments (8-bit, 4-bit) |
| 3 | Aug–Sep | Magnitude pruning experiments (30%, 50%), comparative analysis |
| 4 | October | Write-up and submission |

## Limitations

- Evaluates a limited number of models and language pairs; findings may not generalise to all multilingual architectures or the full diversity of African languages.
- BLEU/chrF++ do not fully capture semantic or cultural nuance.
- Focuses on post-training compression only (not compression-aware retraining or distillation).

## Citation

If referencing this work, please cite the accompanying research proposal/report (details to be added on completion).
