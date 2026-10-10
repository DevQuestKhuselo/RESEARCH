#!/bin/bash
# GPTQ quantisation job. Usage (from the repo root):
#   sbatch run_quantise.sh 8
#   sbatch run_quantise.sh 4
#SBATCH --partition=biggpu
#SBATCH --time=08:00:00
#SBATCH --job-name=gptq-quantise
#SBATCH -o output/%x.%N.%j.out
#SBATCH -e output/%x.%N.%j.err

set -euo pipefail
BITS=${1:?usage: sbatch run_quantise.sh <4|8>}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate research
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p output models

nvidia-smi --query-gpu=name,memory.total --format=csv
python quantise_gptq.py --bits "$BITS" --out_dir "models/llamax3-8b-gptq-${BITS}bit"

echo "Quantisation (${BITS}-bit) finished."
