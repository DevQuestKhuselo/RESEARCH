#!/bin/bash
# Scoring job. Usage (from the repo root), after `sbatch run_prefetch_eval.sh`
# has cached the COMET models once:
#   sbatch run_evaluate.sh baseline
#   sbatch run_evaluate.sh baseline fp16 gptq8 gptq4     # several in one job
# Runs offline (HF_HUB_OFFLINE=1): an uncached model fails at once.
# If it hits the time limit, resubmit the same command: finished files are kept.
#SBATCH --partition=biggpu
#SBATCH --time=08:00:00
#SBATCH --job-name=evaluate
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
[ "$#" -gt 0 ] || set -- baseline

source ~/miniconda3/etc/profile.d/conda.sh
conda activate comet_eval
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cluster_env.sh
source gpu_preflight.sh
export HF_HUB_OFFLINE=1
python -c "from importlib.metadata import version as v; print('unbabel-comet', v('unbabel-comet'), '| sacrebleu', v('sacrebleu'), '| torch', v('torch'))"

for PREFIX in "$@"; do
  echo "##### Scoring [$PREFIX]"
  python evaluate.py --results_dir results --output "scores_${PREFIX}_summary.csv" --pattern "${PREFIX}_*.csv"
done

echo "Evaluation job [$*] finished."
