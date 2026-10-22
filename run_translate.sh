#!/bin/bash
# Translation job for any model. Usage (from the repo root):
#   sbatch run_translate.sh                                         # baseline
#   sbatch run_translate.sh models/llamax3-8b-gptq-8bit gptq8
#   sbatch run_translate.sh models/llamax3-8b-gptq-4bit gptq4
#   sbatch run_translate.sh models/llamax3-8b-mag50 mag50 devtest   # one split only
# GPTQ checkpoints load with the GPT-QModel kernel in $GPTQ_BACKEND
# (default gptq_torch, see baseline_translate.py), e.g.
#   sbatch --export=ALL,GPTQ_BACKEND=gptq_triton run_translate.sh ...
#SBATCH --partition=biggpu
#SBATCH --time=12:00:00
#SBATCH --job-name=translate
# TEMPORARY: nodes with faulty GPUs (110, 111, 112) or down/drained (107, 108)
# as of 11 Oct 2026.
# Remove nodes from this list once the admins fix them; excluding healthy
# nodes only lengthens queue times.
#SBATCH --exclude=mscluster[107-108,110-112]
#SBATCH -o output/%x.%N.%j.out
#SBATCH -e output/%x.%N.%j.err

# -u is enabled only after conda activate: some activate hooks read unset
# variables and would kill the job with "unbound variable".
set -eo pipefail
MODEL_PATH=${1:-LLaMAX/LLaMAX3-8B-Alpaca}
PREFIX=${2:-baseline}
SPLITS=${3:-dev devtest}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate research
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cluster_env.sh
source gpu_preflight.sh
mkdir -p results

# shellcheck disable=SC2086  # SPLITS is a deliberate word list
python run_all_baselines.py --model_path "$MODEL_PATH" --prefix "$PREFIX" --splits $SPLITS

echo "Translation job [$PREFIX] finished."
