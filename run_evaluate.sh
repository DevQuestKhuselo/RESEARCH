#!/bin/bash
# Scoring job. Usage (from the repo root):
#   sbatch run_evaluate.sh            # baseline
#   sbatch run_evaluate.sh gptq8
#   sbatch run_evaluate.sh gptq4
#SBATCH --partition=biggpu
#SBATCH --time=02:00:00
#SBATCH --job-name=evaluate
#SBATCH -o output/%x.%N.%j.out
#SBATCH -e output/%x.%N.%j.err

set -euo pipefail
PREFIX=${1:-baseline}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate comet_eval
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p output

python evaluate.py --results_dir results --output "scores_${PREFIX}_summary.csv" --pattern "${PREFIX}_*.csv"

echo "Evaluation job [$PREFIX] finished."
