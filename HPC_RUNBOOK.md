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
On the MS cluster **`/gluster` does not exist** (checked 10 Oct 2026); `/datasets/<username>/`
does (owner-only, 377 TB volume), so the scripts use `/datasets/$USER/hf_cache` and
`/datasets/$USER/RESEARCH-models`. Check yours with `ls -ld /datasets/$USER`.

**Hardware (biggpu, Oct 2026):** 7 nodes (mscluster106–112), ~122 GB RAM each. GPUs are **mixed**:
mscluster111 has an NVIDIA RTX PRO 6000 Blackwell 96 GB (sm_120, driver 595.71.05 / CUDA 13.2), while
mscluster106 has a Quadro RTX 8000 48 GB (Turing, sm_75). Check `nvidia-smi` in each job log.
mscluster106 is the node used for all timed runs in this project. GPUs are **not** requested with `--gres`
(GRES is `(null)`); asking for the partition is enough. By 11 Oct 2026 nodes 110, 111 and 112 had faulty GPUs and
107 and 108 were down or drained. The job scripts exclude these (`--exclude` on the `sbatch` command line
overrides the list) and run `gpu_preflight.sh`, which fails a job within seconds
if the GPU is unusable. Edit the `--exclude` line once the admins fix those nodes.

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

Miniconda may already be installed: check `ls ~/miniconda3` first and skip the install if so.
Accept the channel terms **before** `conda create`. Setuptools is pinned at creation so pip
does not later downgrade it mid-install (that caused a harmless but alarming
`BadMetadata ... invalid metadata entry 'version'` traceback during the first setup).
```bash
# Miniconda, only if ~/miniconda3 does not exist
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Translation + quantisation env (Blackwell-safe torch, see PROGRESS.md §4.3)
conda create -y -n research python=3.11 "setuptools<81"
conda activate research
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate sacrebleu requests
conda deactivate

# gptqmodel: install INSIDE a GPU session so it builds against the torch above.
# Pick a healthy node (see Hardware above); kernels may still JIT-compile on first use.
srun --partition=biggpu --exclude=mscluster[107-108,110-112] --time=00:45:00 --pty bash -i
source ~/miniconda3/etc/profile.d/conda.sh && conda activate research
nvidia-smi                          # real values, not ERR!, before continuing
pip install -v gptqmodel --no-build-isolation
python -c "import torch, gptqmodel; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
exit

# Evaluation env (kept separate, see PROGRESS.md §6.2)
conda create -y -n comet_eval python=3.11 "setuptools<81"
conda activate comet_eval
# Plain pip pulls a CUDA 13 (cu130) torch here. The driver supports it, and it only affects
# scoring, which is why section 4 re-scores the baseline in this same environment.
pip install --only-binary=:all: numpy==1.26.4
pip install unbabel-comet sacrebleu "setuptools<81"
conda deactivate
```

### 1c. Data

The ten `data/flores200_*.csv` files are tracked in the repo and complete (997 dev / 1012 devtest
rows each), so nothing needs building. Only if they are missing, rebuild them inside an `srun`
session (not on the login node): `python build_flores_csvs.py flores200_dataset`.
Baseline translations and scores are already in the repo (`results/`, `scores_baseline_summary.csv`),
so the baseline does **not** need re-running.

### 1d. HuggingFace

LLaMAX3-8B-Alpaca is ungated, so no login is needed. The job scripts set `HF_HOME` to your
`/datasets/$USER` folder automatically via `cluster_env.sh`.

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

Before the first submission, check the job scripts' conda handling once (login node, no Python run):
```bash
bash -c 'set -euo pipefail; source ~/miniconda3/etc/profile.d/conda.sh; conda activate research; echo ok'
```
```bash
cd ~/RESEARCH
sbatch run_quantise.sh 8
sbatch run_quantise.sh 4
squeue --me                          # PD = pending, R = running
squeue -p biggpu -o "%.8i %.10u %.12L %N"   # time left on other users' jobs
tail -f output/gptq-quantise.*.out
```
Each log should start with `Large files -> /datasets/<you>`, the node name, a real `nvidia-smi`
line, then `GPU OK: NVIDIA RTX PRO 6000 Blackwell ...`. Watch the first run for kernel-build messages:
gptqmodel was installed on a node whose GPU was broken, so its kernels compile on first real use.
Each job writes `models/llamax3-8b-gptq-{8,4}bit/` plus `calibration_ids.json` (calibration sentence
IDs, quantisation time, checkpoint size). GPTQ loads the fp16 model to CPU RAM and quantises layer by
layer on the GPU; if a job is killed for memory, add `#SBATCH --mem=64G` to `run_quantise.sh`.

Optional sanity check that a checkpoint loads and translates, before the multi-hour run.
Open an interactive GPU session first (handbook §2.2), and cancel it when finished:
```bash
srun --partition=biggpu --exclude=mscluster[107-108,110-112] --nodes=1 --ntasks=1 --time=00:30:00 --pty bash -i
nvidia-smi                            # must show real values; ERR! or "No devices" = broken node
source ~/miniconda3/etc/profile.d/conda.sh && conda activate research && cd ~/RESEARCH && source cluster_env.sh
python -c "from baseline_translate import *; t,m=load_model('models/llamax3-8b-gptq-4bit'); \
print(translate_one(t,m,'Good morning, how are you?','English','Zulu'))"
exit                                  # then check `squeue --me` and `scancel <jobid>` if still listed
```

## 3. Translate with the quantised models

Use `--exclusive` and pin one node: GPUs are not SLURM-tracked here, so without `--exclusive` other
jobs can share the GPU and distort latency. Run the fp16 model on the same node as the timing reference.
```bash
sbatch --exclusive -w mscluster106 run_translate.sh models/llamax3-8b-gptq-8bit gptq8
sbatch --exclusive -w mscluster106 run_translate.sh models/llamax3-8b-gptq-4bit gptq4
sbatch --exclusive -w mscluster106 run_translate.sh LLaMAX/LLaMAX3-8B-Alpaca fp16
```
If a job hits its 12-hour limit, resubmit the same command: complete directions are skipped.
Use the prefix `fp16`, never `baseline_*`: `run_evaluate.sh baseline` scores every `baseline_*.csv`.
Outputs `results/gptq{8,4}_{direction}_{split}.csv` and `results/efficiency_gptq{8,4}.csv`.
Partial files are now detected by row count and re-translated (fix for PROGRESS.md §7).

**Efficiency comparability:** latency/memory depend on the GPU. `nvidia-smi` output is at the top of
each job log and the GPU name is recorded per row in `efficiency_*.csv`. Only compare efficiency rows
from the same GPU model (the `fp16` run above is the reference). On the Turing RTX 8000, GPT-QModel
cannot use its fastest (Ampere+) kernels, so quantised latency there is hardware-specific.

## 3b. Magnitude pruning (separate phase)

Pruning runs separately from quantisation, on the fp16 base model:
```bash
sbatch run_prune.sh 30          # -> models/llamax3-8b-mag30 (~15 GB, dense fp16 with zeros)
sbatch run_prune.sh 50          # -> models/llamax3-8b-mag50
# after both finish:
sbatch --exclusive -w mscluster106 run_translate.sh models/llamax3-8b-mag30 mag30
sbatch --exclusive -w mscluster106 run_translate.sh models/llamax3-8b-mag50 mag50
sbatch run_evaluate.sh mag30
sbatch run_evaluate.sh mag50
cp models/llamax3-8b-mag30/pruning_meta.json results/mag30_pruning_meta.json
cp models/llamax3-8b-mag50/pruning_meta.json results/mag50_pruning_meta.json
```
Each pruning job takes minutes (model load + save dominate). Pruning is **global** unstructured magnitude
pruning over the transformer blocks' linear weights, as in the proposal; embeddings and the LM head are
kept. `pruning_meta.json` records achieved sparsity per module type and per layer, since a global
threshold prunes some layers far more than others. If 50% global pruning collapses output (empty or
repeated text), run the per-matrix variant as a sensitivity check: `sbatch run_prune.sh 50 layer`
(-> `models/llamax3-8b-mag50-layer`, translate with prefix `mag50layer`).

Pruned checkpoints are stored dense, so size, memory and latency should be ~unchanged from fp16;
report pruning's compression as non-zero parameter count, not file size.

## 4. Score and compute degradation

Re-score the baseline in your `comet_eval` environment first, so baseline and compressed scores come
from identical scoring software (the committed baseline scores were produced on the old account,
possibly with different COMET/torch versions):
```bash
sbatch run_evaluate.sh baseline      # overwrites scores_baseline_summary.csv; compare with git diff
sbatch run_evaluate.sh gptq8
sbatch run_evaluate.sh gptq4
# after both finish: lightweight, but per the handbook not on the login node.
# Run it on your laptop after `git pull`, or inside a short srun session:
python compute_degradation.py --compressed scores_gptq8_summary.csv scores_gptq4_summary.csv \
  --output degradation_gptq.csv
python compute_degradation.py --compressed scores_mag30_summary.csv scores_mag50_summary.csv \
  --output degradation_pruning.csv
```
The printed table gives mean relative Δ (%) per config × resource category on `devtest`.
Use `devtest` as the headline split: GPTQ calibration sentences come from `dev`.

## 5. Commit results

```bash
git add results/ scores_gptq*_summary.csv degradation_gptq.csv
git commit -m "GPTQ 8-bit and 4-bit translation results and degradation analysis"
git push
```
