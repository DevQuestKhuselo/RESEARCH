#!/bin/bash
# GPTQ quantisation job. Usage (from the repo root):
#   sbatch run_quantise.sh 8
#   sbatch run_quantise.sh 4
#SBATCH --partition=biggpu
#SBATCH --time=08:00:00
#SBATCH --job-name=gptq-quantise
# TEMPORARY: nodes with faulty GPUs (110, 111) or down (108) on 10 Oct 2026.
# Remove nodes from this list once the admins fix them; excluding healthy
# nodes only lengthens queue times.
#SBATCH --exclude=mscluster[108,110-111]
#SBATCH -o output/%x.%N.%j.out
#SBATCH -e output/%x.%N.%j.err

# -u is enabled only after conda activate: some activate hooks read unset
# variables and would kill the job with "unbound variable".
set -eo pipefail
BITS=${1:?usage: sbatch run_quantise.sh <4|8>}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate research
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cluster_env.sh
source gpu_preflight.sh

python quantise_gptq.py --bits "$BITS" --out_dir "models/llamax3-8b-gptq-${BITS}bit"

echo "Quantisation (${BITS}-bit) finished."
