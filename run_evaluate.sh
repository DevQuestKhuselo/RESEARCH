#!/bin/bash
# Scoring job. Usage (from the repo root):
#   sbatch run_evaluate.sh            # baseline
#   sbatch run_evaluate.sh gptq8
#   sbatch run_evaluate.sh gptq4
#SBATCH --partition=biggpu
#SBATCH --time=02:00:00
#SBATCH --job-name=evaluate
# TEMPORARY: nodes with faulty GPUs (110, 111) or down (108) on 10 Oct 2026.
# Remove nodes from this list once the admins fix them; excluding healthy
# nodes only lengthens queue times.
#SBATCH --exclude=mscluster[108,110-111]
#SBATCH -o output/%x.%N.%j.out
#SBATCH -e output/%x.%N.%j.err

# -u is enabled only after conda activate: some activate hooks read unset
# variables and would kill the job with "unbound variable".
set -eo pipefail
PREFIX=${1:-baseline}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate comet_eval
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cluster_env.sh
source gpu_preflight.sh
python -c "from importlib.metadata import version as v; print('unbabel-comet', v('unbabel-comet'), '| sacrebleu', v('sacrebleu'), '| torch', v('torch'))"

python evaluate.py --results_dir results --output "scores_${PREFIX}_summary.csv" --pattern "${PREFIX}_*.csv"

echo "Evaluation job [$PREFIX] finished."
