#!/bin/bash
# Translation job for any model. Usage (from the repo root):
#   sbatch run_translate.sh                                         # baseline
#   sbatch run_translate.sh models/llamax3-8b-gptq-8bit gptq8
#   sbatch run_translate.sh models/llamax3-8b-gptq-4bit gptq4
#SBATCH --partition=biggpu
#SBATCH --time=12:00:00
#SBATCH --job-name=translate
# TEMPORARY: nodes with faulty GPUs (110, 111) or down (108) on 10 Oct 2026.
# Remove nodes from this list once the admins fix them; excluding healthy
# nodes only lengthens queue times.
#SBATCH --exclude=mscluster[108,110-111]
#SBATCH -o output/%x.%N.%j.out
#SBATCH -e output/%x.%N.%j.err

# -u is enabled only after conda activate: some activate hooks read unset
# variables and would kill the job with "unbound variable".
set -eo pipefail
MODEL_PATH=${1:-LLaMAX/LLaMAX3-8B-Alpaca}
PREFIX=${2:-baseline}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate research
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cluster_env.sh
source gpu_preflight.sh
mkdir -p results

python run_all_baselines.py --model_path "$MODEL_PATH" --prefix "$PREFIX"

echo "Translation job [$PREFIX] finished."
