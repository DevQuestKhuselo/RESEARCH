# HPC Runbook: own-account setup → GPTQ quantisation phase

Follows *A Tutorial on Using the Mathematical Sciences HPC Cluster* (H. Wang, Apr 2025).
All commands run on the MS cluster **login node** unless they are inside an `sbatch` job.
Never run Python model code directly on the login node (handbook §1.5; PROGRESS.md §5.3).

## 0. Logging in with your own account

From a terminal on your laptop (Windows: PowerShell or WSL):
```bash
ssh <your-username>@146.141.21.100
```
Type `yes` the first time to trust the host, then your own password. Type `logout` when done.
Files move between laptop and cluster with `scp` / `rsync` (handbook §1.3), but for this project
git is the single source of truth, so prefer `git push` / `git pull`.

**Storage (handbook §3.2):** large files belong in `/gluster/<username>/` or `/datasets/<username>/`,
not in your home directory. This project downloads a ~16 GB base model and writes ~6 GB (4-bit) and
~9 GB (8-bit) checkpoints. The job scripts source `cluster_env.sh`, which detects whichever of those
folders you can write to, puts the HuggingFace cache there, and makes `models/` a symlink into it.
Check they exist for you:
```bash
ls -ld /gluster/$USER /datasets/$USER
```
If neither exists, ask the cluster admins to create one before running the quantisation jobs.

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
mkdir -p output       # SLURM does not create the log folder; jobs fail silently without it
git config user.name  "Khuselo Sofohlo"
git config user.email "sofohlo.k@gmail.com"
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

LLaMAX3-8B-Alpaca is ungated, so no login is needed. The job scripts set `HF_HOME` to your
`/gluster` (or `/datasets`) folder automatically via `cluster_env.sh`.

### 1e. Moving off your friend's account

Everything that matters is already on GitHub (code, baseline translations, baseline scores), so
nothing needs re-running. Before you stop using their account:

1. **Revoke the GitHub token used there.** PROGRESS.md §4.4 says a Personal Access Token was used to
   push from the cluster; git may have saved it in their home directory. On GitHub go to
   Settings → Developer settings → Personal access tokens and delete it. On their account, remove any
   saved copy: `rm -f ~/.git-credentials` (only if it holds *your* token) and
   `git config --global --unset credential.helper`.
2. **Copy anything not on GitHub.** In their `~/RESEARCH`, run `git status`. Untracked files such as an
   earlier `quantise_gptq.py` or job logs in `output/` can be compared or copied with
   `scp`, but the versions in this repo supersede them.
3. **Delete your copies of the large files** there (`~/.cache/huggingface`, `~/RESEARCH/models` if
   any) so they don't use your friend's quota.
4. In your write-up, note that baseline runs used a different account; results do not depend on it.

## 2. Quantise (one job per bit-width)

```bash
cd ~/RESEARCH
sbatch run_quantise.sh 8
sbatch run_quantise.sh 4
squeue --me                          # PD = pending, R = running
tail -f output/gptq-quantise.*.out
```
Each job writes `models/llamax3-8b-gptq-{8,4}bit/` plus `calibration_ids.json` (calibration sentence
IDs, quantisation time, checkpoint size). GPTQ loads the fp16 model to CPU RAM and quantises layer by
layer on the GPU; if a job is killed for memory, add `#SBATCH --mem=64G` to `run_quantise.sh`.

Optional sanity check that a checkpoint loads and translates, before the multi-hour run.
Open an interactive GPU session first (handbook §2.2), and cancel it when finished:
```bash
srun --partition=biggpu --nodes=1 --ntasks=1 --time=00:30:00 --pty bash -i
conda activate research && cd ~/RESEARCH && source cluster_env.sh
python -c "from baseline_translate import *; t,m=load_model('models/llamax3-8b-gptq-4bit'); \
print(translate_one(t,m,'Good morning, how are you?','English','Zulu'))"
exit                                  # then check `squeue --me` and `scancel <jobid>` if still listed
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
# after both finish: lightweight, but per the handbook not on the login node.
# Run it on your laptop after `git pull`, or inside a short srun session:
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
