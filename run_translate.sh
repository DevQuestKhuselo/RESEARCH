#!/bin/bash
# Translation job for any model. Usage (from the repo root):
#   sbatch run_translate.sh                                         # baseline
#   sbatch run_translate.sh models/llamax3-8b-gptq-8bit gptq8
#   sbatch run_translate.sh models/llamax3-8b-gptq-4bit gptq4
#SBATCH --partition=biggpu
#SBATCH --time=12:00:00
#SBATCH --job-name=translate
#SBATCH -o output/%x.%N.%j.out
#SBATCH -e output/%x.%N.%j.err

set -euo pipefail
MODEL_PATH=${1:-LLaMAX/LLaMAX3-8B-Alpaca}
PREFIX=${2:-baseline}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate research
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p output results

nvidia-smi --query-gpu=name,memory.total --format=csv
python run_all_baselines.py --model_path "$MODEL_PATH" --prefix "$PREFIX"

echo "Translation job [$PREFIX] finished."
