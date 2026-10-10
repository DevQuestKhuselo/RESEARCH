# HPC Runbook: own-account setup → GPTQ quantisation phase

All commands run on the Wits MS cluster **login node** unless they are inside an `sbatch` job.
Never run Python model code directly on the login node (see PROGRESS.md §5.3).

## 1. One-time setup on your own account

### 1a. GitHub access from the cluster (SSH key, no token in shell history)

```bash
ssh-keygen -t ed25519 -C "sofohlo.k@gmail.com"     # accept defaults
cat ~/.ssh/id_ed25519.pub                           # copy this
```
Add the key at GitHub → Settings → SSH and GPG keys → New SSH key, then:
```bash
ssh -T git@github.com                               # should greet DevQuestKhuselo
git clone git@github.com:DevQuestKhuselo/RESEARCH.git ~/RESEARCH
cd ~/RESEARCH
git config user.name  "Khuselo Sofohlo"
git config user.email "sofohlo.k@gmail.com"
mkdir -p output models
```

### 1b. Conda environments

```bash
# Miniconda, if not already installed on your account
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Translation + quantisation env (Blackwell-safe torch, see PROGRESS.md §4.3)
conda create -y -n research python=3.11
conda activate research
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate sacrebleu
pip install -v gptqmodel           # first GPU use JIT-compiles kernels (needs ninja, pulled in)
conda deactivate

# Evaluation env (kept separate, see PROGRESS.md §6.2)
conda create -y -n comet_eval python=3.11
conda activate comet_eval
pip install --only-binary=:all: numpy==1.26.4
pip install unbabel-comet sacrebleu "setuptools<81"
conda deactivate
```

### 1c. Data

`data/` is gitignored, so rebuild it from the FLORES copy that *is* in the repo:
```bash
conda activate research && python build_flores_csvs.py flores200_dataset && conda deactivate
```
Baseline translations and scores are already in the repo (`results/`, `scores_baseline_summary.csv`),
so the baseline does **not** need re-running.

### 1d. HuggingFace

LLaMAX3-8B-Alpaca is ungated, so no login is needed. The ~16 GB download caches to
`~/.cache/huggingface`. If your home quota is small, point it at scratch first:
`export HF_HOME=/path/to/your/scratch/hf` (add to `~/.bashrc`).

## 2. Quantise (one job per bit-width)

```bash
cd ~/RESEARCH
sbatch run_quantise.sh 8
sbatch run_quantise.sh 4
squeue -u $USER
tail -f output/gptq-quantise.*.out
```
Each job writes `models/llamax3-8b-gptq-{8,4}bit/` plus `calibration_ids.json` (calibration sentence
IDs, quantisation time, checkpoint size). GPTQ loads the fp16 model to CPU RAM and quantises layer by
layer on the GPU; if a job is killed for memory, add `#SBATCH --mem=64G` to `run_quantise.sh`.

Optional sanity check that a checkpoint loads and translates, before the multi-hour run
(inside an interactive GPU allocation, not the login node):
```bash
python -c "from baseline_translate import *; t,m=load_model('models/llamax3-8b-gptq-4bit'); \
print(translate_one(t,m,'Good morning, how are you?','English','Zulu'))"
```

## 3. Translate with the quantised models

```bash
sbatch run_translate.sh models/llamax3-8b-gptq-8bit gptq8
sbatch run_translate.sh models/llamax3-8b-gptq-4bit gptq4
```
Outputs `results/gptq{8,4}_{direction}_{split}.csv` and `results/efficiency_gptq{8,4}.csv`.
Partial files are now detected by row count and re-translated (fix for PROGRESS.md §7).

**Efficiency comparability:** latency/memory depend on the GPU. `nvidia-smi` output is at the top of
each job log and the GPU name is recorded per row in `efficiency_*.csv`. Only compare efficiency rows
from the same GPU model; re-run the baseline translation for timing on that GPU if needed
(`sbatch run_translate.sh LLaMAX/LLaMAX3-8B-Alpaca baseline_timing`).

## 4. Score and compute degradation

```bash
sbatch run_evaluate.sh gptq8
sbatch run_evaluate.sh gptq4
# after both finish (CPU-only, fine on the login node):
python compute_degradation.py --compressed scores_gptq8_summary.csv scores_gptq4_summary.csv \
  --output degradation_gptq.csv
```
The printed table gives mean relative Δ (%) per config × resource category on `devtest`.
Use `devtest` as the headline split: GPTQ calibration sentences come from `dev`.

## 5. Commit results

```bash
git add results/ scores_gptq*_summary.csv degradation_gptq.csv
git commit -m "GPTQ 8-bit and 4-bit translation results and degradation analysis"
git push
```
