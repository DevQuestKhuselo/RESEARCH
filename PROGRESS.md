# Progress Report: Evaluating the Impact of Quantisation and Pruning on Multilingual LLM Translation for Low-Resource African Languages

**Author:** Khuselo Sofohlo
**Supervisors:** Dr Steven James, Prof Benjamin Rosman
**Report covers:** Proposal validation → environment setup → dataset acquisition → baseline translation pipeline → baseline evaluation results
**Status at time of writing:** Baseline (uncompressed) results complete for all five translation directions. GPTQ quantisation code complete and own cluster account set up (§11–12); quantisation jobs not yet run.

---

## 1. Overview

This report documents the work completed in setting up the research pipeline for the project, covering every significant decision, failure, and fix from initial proposal review through to obtaining baseline translation quality scores. The intent is to give a complete, honest account — including the mistakes made and how they were diagnosed and corrected — so the process is reproducible and the reasoning behind each deviation from the original written proposal is traceable.

---

## 2. Model Selection: A Correction to the Original Proposal

### 2.1 The problem

The original research proposal specified **Aya-8B** as the model under study. Before starting implementation, the exact model was investigated and it was discovered that **Aya Expanse 8B** — the model normally referred to as "Aya-8B" — supports only 23 languages (Arabic, Chinese, Czech, Dutch, English, French, German, Greek, Hebrew, Hindi, Indonesian, Italian, Japanese, Korean, Persian, Polish, Portuguese, Romanian, Russian, Spanish, Turkish, Ukrainian, Vietnamese). **Neither Swahili nor isiZulu is supported.**

This is a direct conflict with the proposal's own translation-direction design (§4.8), which requires en↔sw and en↔zu directions. Continuing with Aya Expanse 8B would have made roughly 60% of the planned experiments impossible to run.

### 2.2 Investigation

Two alternative Cohere models were considered:

- **Aya-101**: the original Aya model, 13B parameters, confirmed via academic literature (Lugha-Llama paper) to support both Swahili and Zulu. However, it uses an **mT5 encoder-decoder architecture**, not a decoder-only transformer.
- **Tiny Aya**: a newer, smaller (3.35B) Cohere Labs model explicitly listing Swahili, Xhosa, and Zulu among 70+ supported languages, but with less-established tooling support at the time of investigation.

The architecture question mattered because GPTQ — the quantisation method central to this research — was designed for and is predominantly tooled around **decoder-only** transformer architectures. An mT5-based model risked poor or unsupported GPTQ compatibility, which would undermine the core methodology.

### 2.3 Resolution

**LLaMAX3-8B-Alpaca** (`LLaMAX/LLaMAX3-8B-Alpaca` on HuggingFace) was selected as the replacement model. It is:

- Decoder-only, built via continued multilingual pretraining of **Llama-3-8B**, giving it strong existing GPTQ tooling compatibility.
- Confirmed via its own model card to explicitly support both Swahili and Zulu among 100+ languages.
- Instruction-tuned (the "-Alpaca" variant), consistent with the proposal's original framing of using an instruction-tuned multilingual LLM.

A directly relevant prior-work paper was also found during this investigation (arXiv:2601.18306, Jan 2026), which performs GPTQ/AWQ quantisation specifically on Llama-3.1-8B and Qwen2.5-7B with Swahili and isiXhosa calibration data — useful as related work to cite and position against.

**Outcome:** the model substitution preserves the original translation-direction design and hypothesis with no change to the research question, while resolving both the language-coverage and architecture-compatibility problems.

---

## 3. Dataset Acquisition: FLORES-200

### 3.1 What worked

FLORES-200 was identified as the correct dataset (professionally translated, sentence-aligned, standard multilingual MT benchmark, covers all four required languages: `eng_Latn`, `fra_Latn`, `swh_Latn`, `zul_Latn`).

### 3.2 What didn't work, and how it was fixed

Programmatic download went through several failed attempts before landing on a working approach:

| Attempt | Problem | Fix |
|---|---|---|
| `datasets.load_dataset("facebook/flores", ...)` | Gated repo requiring HF authentication, and uses a deprecated script-based loader format newer `datasets` versions refuse to run | Abandoned the HF `datasets` library route entirely |
| Ungated mirror (`Muennighoff/flores200`) | Same script-loader incompatibility | Also abandoned |
| Direct download via `https://tinyurl.com/flores200dataset` | `requests` received an HTML "preview" interstitial page from TinyURL instead of a redirect, saved as if it were the tarball | Diagnosed via `file` command showing "HTML document" instead of gzip; resolved the real underlying URL via `curl -sIL` and by scraping the preview page's HTML for the embedded `dl.fbaipublicfiles.com` link |
| Direct download from resolved URL `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz` | Still intermittently returned non-gzip content in one run | Added a gzip-magic-byte validation check to the download script so bad downloads fail loudly and re-trigger, rather than silently producing an empty extracted folder |
| **Manual browser download** | N/A — this is what ultimately worked | User downloaded `flores200_dataset.tar.gz` directly via browser (bypassing whatever was blocking automated `requests`/`curl` downloads), extracted it locally |

**Root cause never fully diagnosed**, but manual browser download reliably worked where automated approaches did not — likely some form of bot/User-Agent detection on the server or CDN side.

### 3.3 Dataset build script

`build_flores_csvs.py` was written to read the locally-extracted FLORES-200 folder (`{root}/dev/{lang}.dev`, `{root}/devtest/{lang}.devtest`, per the dataset's own README) and produce five parallel CSVs per split (`data/flores200_{direction}_{split}.csv`) for the required directions: en-fr, en-sw, en-zu, sw-en, zu-en.

This ran successfully once pointed at the correct local folder path (accounting for a nested-folder quirk from the zip extraction on Windows).

### 3.4 Masakhane / LAFAND-MT

Cloned via `git clone https://github.com/masakhane-io/lafand-mt.git` as a planned supplementary in-domain (news-text) evaluation set, alongside FLORES-200. Not yet integrated into the evaluation pipeline as of this report.

---

## 4. Compute Environment: Wits HPC (MS Cluster)

### 4.1 Why the switch to HPC

Initial attempts to run the 8B model locally caused the user's machine to freeze — expected, since fp16 inference of an 8B-parameter model requires roughly 16GB of memory, which most consumer laptops don't have available as free RAM/VRAM. The Wits HPC cluster (SLURM-based, referenced in the original proposal) was used instead.

**Caveat noted during setup:** the cluster account in use belongs to a friend, since the user's own login credentials were still being provisioned. This was flagged as a risk (storage quotas, job history, and fair-use all tie back to the friend's account) and the user was advised to migrate to their own account once available, and to avoid heavy/long jobs in the meantime.

### 4.2 Cluster discovery

- Login node: `mscluster-login`. SLURM partitions available: `batch`, `stampede`, `bigbatch`, `biggpu`, `gpuexpress`.
- `sinfo -o "%P %G %N"` reported GRES as `(null)` for all partitions — this cluster does not track GPUs through SLURM's generic-resource (`--gres`) system, unlike the standard tutorial documentation assumed. Attempting `srun --gres=gpu:1` failed with `Invalid generic resource (gres) specification`.
- `nvidia-smi` run directly on the login node showed only an RTX 2060 — this is the **login node's own GPU** and must not be used for jobs (per the "never run on the head node" rule).
- The `biggpu` partition was identified as the correct target, requested via `srun --partition=biggpu` **without** `--gres`. This successfully allocated real GPU compute nodes.
- GPU hardware varied by allocation across the session: an **RTX PRO 6000 Blackwell (96GB VRAM)** on one allocation, a **Quadro RTX 8000 (48GB VRAM)** on another. Both are more than sufficient for an 8B model in fp16 (~16GB), so this variability did not block progress, but was flagged as a concern for later efficiency-metric comparisons (inference latency / GPU memory measurements should be pinned to a specific node for consistency once that phase begins).

### 4.3 PyTorch / CUDA compatibility (Blackwell-specific issue)

The RTX PRO 6000 Blackwell GPU (compute capability sm_120) is new enough that older PyTorch builds do not ship kernels compiled for it. Research confirmed PyTorch only gained proper Blackwell support from **version 2.7.0 with CUDA 12.8 (cu128) wheels** onward. The generic HPC tutorial's suggested install command (`conda install pytorch pytorch-cuda=12.1`) would very likely have produced a build lacking Blackwell kernels, causing a `CUDA error: no kernel image is available for execution on the device` failure.

**Fix:** installed PyTorch explicitly from the `cu128` wheel index:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
This correctly resolved to `torch-2.11.0+cu128`, confirmed working on both GPU types encountered.

### 4.4 Other environment setup issues and fixes

| Issue | Fix |
|---|---|
| `conda create` failed with a Terms-of-Service error for the `defaults` channel | Ran `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main` (and `/pkgs/r`) |
| GitHub push failed with "Invalid username or token. Password authentication is not supported" | GitHub disabled password auth for git operations in 2021; generated a Personal Access Token (classic, `repo` scope) and used it as the password |
| Files created directly on the cluster via `nano` (e.g. `run_all_baselines.py`) were not present in the local git clone | Pulled them back to the local machine via `scp` before committing, to keep the GitHub repo as the single source of truth |
| No `.gitignore` | Added one excluding `data/`, `results/`, `output/`, and Python cache files, to avoid committing large regenerable data to the repository |

---

## 5. Baseline Translation Pipeline

### 5.1 `baseline_translate.py`

Implements the official LLaMAX prompt template (Alpaca-style instruction format) and greedy-decoding translation for a single direction/split CSV. Initial smoke test (5 sentences, en→sw) produced fluent, largely accurate Swahili, confirming the pipeline and prompt format were correct. One semantic inversion was noted in the smoke test output ("we now have" mistranslated as "we don't have") — flagged as expected baseline translation-quality noise, not a pipeline defect.

### 5.2 `run_all_baselines.py`

An efficiency fix: the original design would have reloaded the 8B model from scratch for each of 10 direction/split combinations (~4-5 minutes of pure loading overhead each, ~45+ minutes wasted in total). Rewritten to load the model **once** and loop over all directions/splits within a single process. Includes a resume-safety feature: skips any direction/split whose output file already exists, so an interrupted job can be resubmitted without re-doing completed work.

**This resume-safety feature caused one real bug** (see §7 below).

### 5.3 SLURM batch job

`run_baseline_all.sh` submits `run_all_baselines.py` via `sbatch` rather than an interactive session, since the full run (5 directions × ~1000 devtest sentences each) takes hours and the `biggpu` partition proved to have inconsistent availability for interactive `srun` sessions (queued and timed out multiple times).

**Operational mistake made and corrected:** at one point, `python run_all_baselines.py` was run directly after an `srun` allocation had already been cancelled (via `Ctrl+C`), meaning the command executed on the **login node** rather than a compute node — exactly the situation the cluster's usage policy warns against. This happened twice before being caught. Resolution: switched fully to `sbatch`-based submission for anything long-running, avoiding interactive `srun` reliance given the partition's queue congestion.

---

## 6. Evaluation Pipeline

### 6.1 Metrics and tooling

`evaluate.py` computes BLEU and chrF++ (via `sacrebleu`) and COMET and AfriCOMET (via `unbabel-comet`) for each results file. Model checkpoints used:
- COMET: `Unbabel/wmt22-comet-da`
- AfriCOMET: `masakhane/africomet-stl-1.1` (confirmed via its model card to cover Swahili and Zulu; reference-based variant, matching the fact that FLORES-200 provides gold references)

### 6.2 Dependency conflicts, and how they were resolved

Installing `unbabel-comet` directly into the same environment as the translation pipeline (`research`, running `torch 2.11.0+cu128`) risked two problems: `unbabel-comet` requires `numpy<2.0` (conflicting with `numpy 2.2.6` already installed for `transformers`), and pulls in its own PyTorch-Lightning-compatible torch version that could silently override the carefully-matched Blackwell-compatible torch build.

**Fix:** created a separate, isolated conda environment (`comet_eval`) specifically for evaluation, avoiding any cross-contamination with the translation environment. This is a deliberate architectural choice: translation and evaluation now run in fully separate environments, connected only via the CSV files each stage reads/writes.

A secondary issue arose even within the isolated environment: `pip install unbabel-comet` initially appeared to hang for 10+ minutes attempting to build `numpy==1.26.4` from source, despite prebuilt wheels existing. Resolved by force-installing the wheel directly first (`pip install --only-binary=:all: numpy==1.26.4`) before installing `unbabel-comet`, which then resolved normally.

### 6.3 `ModuleNotFoundError: No module named 'pkg_resources'`

The first full evaluation batch job failed immediately (before any scoring occurred) with this error. Root cause: newer `setuptools` versions (≥81) removed the `pkg_resources` module, which `torchmetrics` (a dependency of `pytorch-lightning`, a dependency of `unbabel-comet`) still imports internally. This is an upstream compatibility gap in the `unbabel-comet` dependency chain, not a mistake in this project's code.

**Fix:** `pip install "setuptools<81"` within the `comet_eval` environment.

---

## 7. Bug: Incomplete Data Silently Accepted by the Resume-Safety Logic

**What happened:** the 5-sentence smoke-test output file (`results/baseline_en-sw_devtest.csv`, produced during initial pipeline validation) was never deleted before the full batch job ran. Because `run_all_baselines.py`'s skip-logic only checks whether an output file *exists*, not whether it's *complete*, the full run silently skipped translating that entire direction, leaving only 5 sentences where ~1012 were expected — while reporting no error and appearing to complete successfully.

**How it was caught:** a routine `ls -la results/` file-size sanity check, comparing all ten output files, showed `baseline_en-sw_devtest.csv` at 3KB versus ~400-460KB for every other file.

**Fix:** deleted the incomplete file and re-ran `run_all_baselines.py`; the resume logic then correctly regenerated only the missing direction (skipping the 9 already-complete ones), avoiding a multi-hour full re-run.

**Process lesson:** always verify output completeness (row counts or file sizes), not just presence, before trusting a "skip if exists" resume mechanism — and before feeding results into an expensive downstream step (evaluation, in this case, was correctly *not* run against the incomplete data because this was caught first).

---

## 8. GPTQ Library Selection

Before beginning quantisation work, `auto-gptq` (the library referenced in the original proposal and most existing tutorials) was checked against current tooling status. HuggingFace's own documentation now states plainly that **AutoGPTQ is no longer supported in Transformers**, replaced by **GPT-QModel** (`gptqmodel` package) — an actively maintained fork with continued releases as recently as the month of writing, including explicit Blackwell (sm100+) kernel support directly relevant to the GPU hardware available on this cluster.

**Decision:** use `gptqmodel` via `transformers`' native `GPTQConfig` integration, not `auto-gptq`.

`quantise_gptq.py` was written using this approach, with one deliberate methodological choice: the GPTQ calibration dataset is built from a **balanced mix of all four target languages** (English, French, Swahili, Zulu; 32 sentences each, drawn from the FLORES-200 dev split), rather than the English-only calibration set most GPTQ tutorials default to. Using English-only calibration would bias quantisation toward preserving English-relevant weights — precisely the kind of confound this research is designed to test for, so it was avoided from the outset.

`baseline_translate.py` and `run_all_baselines.py` were refactored to accept a `--model_path` (and `--prefix`) argument, so the same, already-validated translation and evaluation pipeline can be reused for the 8-bit and 4-bit quantised models without duplicating code.

**Status:** quantisation has not yet been executed as of this report — `gptqmodel` installation and the quantisation batch job are the immediate next steps.

---

## 9. Baseline Results

Full baseline (uncompressed) results obtained across all five translation directions, both `dev` and `devtest` splits:

| Direction | Split | BLEU | chrF++ | COMET | AfriCOMET |
|---|---|---|---|---|---|
| en→fr | dev | 41.07 | 62.81 | 0.8637 | 0.7618 |
| en→fr | devtest | 40.02 | 62.64 | 0.8580 | 0.7545 |
| en→sw | dev | 25.12 | 51.88 | 0.8036 | 0.6814 |
| en→sw | devtest | 24.14 | 50.69 | 0.8007 | 0.6772 |
| en→zu | dev | 6.55 | 34.77 | 0.7137 | 0.5685 |
| en→zu | devtest | 6.05 | 33.85 | 0.6986 | 0.5512 |
| sw→en | dev | 37.56 | 59.55 | 0.8457 | 0.7585 |
| sw→en | devtest | 36.80 | 58.78 | 0.8439 | 0.7561 |
| zu→en | dev | 27.04 | 50.15 | 0.7709 | 0.6832 |
| zu→en | devtest | 26.56 | 49.64 | 0.7671 | 0.6778 |

### 9.1 Observations

- **en→zu is a dramatic outlier**, scoring far below every other direction (BLEU 6.0 on devtest, versus 24-40 elsewhere). This is consistent with the research hypothesis (weaker low-resource representations), but the size of the gap warranted a qualitative check before trusting it at face value.
- **Generation into isiZulu is far more fragile than generation into Swahili**, even though isiZulu→English comprehension (BLEU 26.6) is reasonably close to Swahili→English (BLEU 36.8). Manual inspection of `results/baseline_en-zu_devtest.csv` shows the model attempting the task and producing grammatically coherent sentence structures, but with heavy **code-mixing of untranslated English vocabulary** (e.g. `kwi-diabetes`, `zi-diabetic` — Bantu noun-class prefixes glued directly onto English medical terms) and morphology that appears to bleed in from isiXhosa (e.g. repeated use of the `kwi-` locative prefix, more characteristic of Xhosa than Zulu's typical `ku-`).
- This suggests the finding is genuine signal rather than a pipeline defect: the model's *comprehension* of isiZulu is reasonably competent, but its *generation* capability for isiZulu specifically is markedly weaker and linguistically confused — an important baseline characteristic to keep in mind when later interpreting what GPTQ/pruning compression does to this direction, since it is starting from an already low and qualitatively different floor than the other four directions.

---

## 10. Current Repository State

```
.
├── data/                      # FLORES-200 parallel CSVs (gitignored, regenerable)
├── results/                   # Baseline translation outputs (10 files, complete)
├── models/                    # (to be created) quantised model checkpoints
├── output/                    # SLURM job logs (gitignored)
├── build_flores_csvs.py       # Builds per-direction parallel CSVs from local FLORES-200
├── baseline_translate.py      # Single-direction translation (baseline or quantised model)
├── run_all_baselines.py       # All-direction translation, single model load, resume-safe
├── run_baseline_all.sh        # SLURM batch job for the above
├── evaluate.py                # BLEU / chrF++ / COMET / AfriCOMET scoring
├── run_evaluate.sh            # SLURM batch job for evaluation
├── quantise_gptq.py           # GPTQ quantisation (8-bit / 4-bit), multilingual calibration
├── run_quantise.sh            # SLURM batch job for quantisation
├── scores_baseline_summary.csv # Baseline metric results (Section 9 above)
└── README.md
```

`prune_magnitude.py` (magnitude-based pruning, per proposal §4.6) was added later; see §13.

---

## 11. Update (10 Oct 2026): quantisation code committed

`quantise_gptq.py`, described in §8, had not been pushed to GitHub. It has been rewritten against the current GPT-QModel API (v7.x, `GPTQModel.load` → `quantize` → `save`) and committed, with:

- Balanced calibration: 32 FLORES `dev` sentences per language, with **disjoint sentence IDs across languages**, IDs saved to `calibration_ids.json`. Since calibration uses `dev`, **`devtest` is the headline split** for compressed models.
- `run_all_baselines.py` now takes `--model_path` / `--prefix`, skips a direction only if its output has the full row count (fixes the §7 bug), and logs latency, tokens/s and GPU memory to `results/efficiency_{prefix}.csv`.
- `compute_degradation.py` computes absolute and relative Δ per direction, split and metric, averaged by resource category.
- SLURM scripts no longer hardcode another user's home path; they run from the submit directory under any account. `run_baseline_all.sh` is replaced by the general `run_translate.sh`.
- `HPC_RUNBOOK.md` documents moving to the author's own cluster account (§4.1 risk) and the full quantisation workflow.

## 12. Own cluster account set up (10 Oct 2026)

The §4.1 risk is resolved: work now runs under the author's own MS cluster account (`ksofohlo`).

**Storage.** `/gluster` does not exist on the MS cluster, despite the handbook mentioning it. `/datasets/ksofohlo` (owner-only, 377 TB NFS volume) holds the HuggingFace cache and model checkpoints via `cluster_env.sh`, keeping multi-GB files out of the home directory.

**Hardware.** biggpu has 7 nodes (mscluster106–112) with ~122 GB RAM. GPUs are mixed: mscluster111 has an NVIDIA RTX PRO 6000 Blackwell 96 GB (driver 595.71.05, CUDA 13.2), but mscluster106, where all quantisation and timed translation ran, has a **Quadro RTX 8000 48 GB** (Turing, sm_75), as recorded in the quantisation metadata. SLURM GRES is `(null)`, consistent with §4.2.

**Faulty-node incident.** At 22:48 the only idle GPU nodes were broken: on mscluster110 `nvidia-smi` found no device (job 66708), and on mscluster111 every GPU field read `ERR!` and torch saw no GPU (job 66709). mscluster108 was down. These were reported to the cluster admins. The job scripts now exclude these nodes (temporary) and run `gpu_preflight.sh`, which allocates a tensor on the GPU and fails the job in seconds rather than hours into a run.

**Environments** (versions recorded for reproducibility):

| Env | Key versions |
|---|---|
| `research` (translate, quantise) | Python 3.11, torch 2.11.0+cu128, transformers 5.19.0, accelerate 1.15.0, gptqmodel 7.5.0, sacrebleu 2.6.0, numpy 2.4.6 |
| `comet_eval` (score) | Python 3.11, torch 2.14.1+cu130, unbabel-comet 2.2.7, transformers 4.57.6, pytorch-lightning 2.6.6, numpy 1.26.4, setuptools 80.10.2 |

**Setup issues and fixes.**

| Issue | Fix |
|---|---|
| pip printed a `BadMetadata ... invalid metadata entry 'version'` traceback after downgrading conda's setuptools 84 | Harmless; packages installed and import correctly. Runbook now pins `setuptools<81` at `conda create` |
| `import gptqmodel` failed: `No module named 'requests'` | `pip install requests`; added to the runbook |
| gptqmodel installed with `--no-build-isolation` (to build against the existing torch) on a node with a broken GPU | Kernels will compile on first real GPU use; watch the first quantise log |
| `set -euo pipefail` before `conda activate` could kill jobs if an activate hook reads an unset variable | `set -u` now enabled only after activation |

**Reproducibility.** The committed baseline scores were produced on the previous account, possibly with different COMET/torch versions. The baseline will be re-scored in the new `comet_eval` environment so baseline and compressed scores come from identical scoring software. The original baseline run recorded no timing, so the fp16 model is re-run on mscluster106 (prefix `fp16`) as the timing reference; all timed runs use `--exclusive` on that one node because GPUs are not SLURM-tracked and could otherwise be shared.

## 13. GPTQ quantisation run and pruning code (11–17 Oct 2026)

**Further node failures.** The first quantisation jobs (66719, 66720) landed on mscluster112, whose GPU also reported "No devices were found"; `gpu_preflight.sh` stopped both within 10 seconds. mscluster107 was then down or drained. Jobs were resubmitted excluding 107, 108 and 110–112.

**Quantisation results** (jobs 67056/67057, mscluster106, Quadro RTX 8000, gptqmodel 7.5.0, group size 128, seed 42, identical calibration set for both):

| Model | Checkpoint size | Compression vs fp16 (~15.0 GiB) | Quantisation time | Peak GPU memory |
|---|---|---|---|---|
| GPTQ 8-bit | 8.63 GiB | ~1.7× | 11.6 min | 3.76 GB |
| GPTQ 4-bit | 5.36 GiB | ~2.8× | 8.4 min | 3.76 GB |

Compression is below the nominal 2×/4× because only the transformer blocks are quantised: the input embeddings and LM head stay fp16 and, with Llama-3's 128,256-token vocabulary, total ~2 GiB. Metadata is in `results/gptq{8,4}_quantisation_meta.json`.

**Hardware caveat for latency.** On Turing, GPT-QModel falls back from its fastest (Ampere+) kernels, so quantised-model latency measured here is hardware-specific and may not improve on fp16; memory and size reductions are hardware-independent.

**Pruning code.** `prune_magnitude.py` implements the proposal's global unstructured magnitude pruning, run as its own phase (`run_prune.sh 30|50`). Design choices: only the linear weights inside the transformer blocks are pruned (embeddings, LM head and norms are kept); one global threshold, found by bisection rather than sorting ~7B values; per-module and per-layer achieved sparsity saved in `pruning_meta.json`, because a global threshold prunes layers unevenly; an optional per-matrix (`layer`) scope as a sensitivity check. Verified on a small Llama model: target sparsity reached, surviving weights unchanged, embeddings/LM head/norms untouched, checkpoint reloads and runs through the translation pipeline. Pruned checkpoints are dense fp16, so pruning's efficiency is reported as non-zero parameters rather than size or latency.

**Scheduling changes (17 Oct 2026).** After 22 hours, the three translation jobs pinned to mscluster106 with `--exclusive` (69601–69603) had still not started, because an exclusive job waits for a completely empty node. Quality and timing were therefore separated. Full translations run on any working GPU; latency and memory come from a short exclusive benchmark (`run_timing.sh`, the first 100 devtest sentences per direction, every model on one node). The baseline re-score (69604, mscluster109, confirming that node's GPU works) hit its 2-hour limit and, because `evaluate.py` wrote only at the end, saved nothing. `evaluate.py` now saves each file as it is scored, resumes after a timeout, uses COMET batch size 32 (was 8), and the job limit is 8 hours. Recomputing BLEU and chrF++ for the baseline files in a clean environment reproduced the committed values exactly.

**GPTQ translation failure (17 Oct 2026).** The first gptq8/gptq4 translation jobs (69601, 69602) failed within two minutes, while the fp16 re-run (69603) completed all ten files in 4 h 50 min on mscluster106 with `--exclusive`. Transformers 5.19 requires the `optimum` package (≥1.24) to load GPTQ checkpoints, and it was not part of the `research` environment; GPT-QModel itself only needs it for loading through Transformers. Fix: `pip install "optimum>=1.24" --no-deps` (optimum 2.3.0 verified compatible with transformers 5.19 and leaves torch unchanged). The pruning jobs (69605, 69606) sat pending for three days because they were restricted to mscluster109, which was then drained.

**Silent CPU fallback (20 Oct 2026).** After installing `optimum`, both GPTQ checkpoints loaded on mscluster111 and GPT-QModel selected its `TritonV2Linear` kernel, but generation failed with `Expected a cuda device, but got: cpu`. `nvidia-smi -L` listed the GPU (RTX PRO 6000 Blackwell), but accelerate's probe could not use it ("Device 0 seems unavailable"), so `device_map="auto"` silently placed the model on CPU. mscluster111's GPU is therefore still faulty. `load_model` now places the model explicitly on `cuda` and raises an error if any tensor is not on the GPU, so a broken GPU fails loudly. The fp16 results are unaffected: on a working GPU, `"auto"` and `"cuda"` both put the whole 8B model on GPU 0.

## 14. Immediate Next Steps

1. ~~Set up own cluster account and install `gptqmodel`~~ (done, §12). Confirm `torch.cuda.is_available()` on a healthy biggpu node.
2. ~~`sbatch run_quantise.sh 8` and `sbatch run_quantise.sh 4`~~ (done, §13).
3. `sbatch run_translate.sh ...` for gptq8 and gptq4, `run_timing.sh` (exclusive, mscluster106) for fp16/gptq8/gptq4, then `sbatch run_evaluate.sh baseline` (re-score, §12) and `sbatch run_evaluate.sh gptq{8,4}`.
4. `python compute_degradation.py` for relative degradation (Δ) per the proposal's methodology (§4.11), comparing quantised-vs-baseline score drops across the five directions to test the core hypothesis.
5. ~~Implement magnitude pruning~~ (done, §13). Run `sbatch run_prune.sh 30` and `50`, then translate (`mag30`, `mag50`) and evaluate (HPC_RUNBOOK.md §3b).
6. Integrate Masakhane/LAFAND-MT as the supplementary in-domain evaluation set (currently downloaded but unused).
