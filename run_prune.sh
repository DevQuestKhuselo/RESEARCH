#!/bin/bash
# Magnitude pruning job. Usage (from the repo root):
#   sbatch run_prune.sh 30            # 30% global magnitude pruning -> models/llamax3-8b-mag30
#   sbatch run_prune.sh 50            # 50%                          -> models/llamax3-8b-mag50
#   sbatch run_prune.sh 50 layer      # per-matrix variant           -> models/llamax3-8b-mag50-layer
#SBATCH --partition=biggpu
#SBATCH --time=02:00:00
#SBATCH --job-name=prune
# TEMPORARY: nodes with faulty GPUs (110, 111, 112) or down/drained (107, 108)
# as of 11 Oct 2026. Edit once the admins fix them.
#SBATCH --exclude=mscluster[107-108,110-112]
#SBATCH -o output/%x.%N.%j.out
#SBATCH -e output/%x.%N.%j.err

set -eo pipefail
PCT=${1:?usage: sbatch run_prune.sh <30|50> [global|layer]}
SCOPE=${2:-global}
case "$PCT" in ''|*[!0-9]*) echo "sparsity must be a whole percentage 1-99, e.g. 30" >&2; exit 1;; esac
if [ "$PCT" -lt 1 ] || [ "$PCT" -gt 99 ]; then echo "sparsity must be a whole percentage 1-99, e.g. 30" >&2; exit 1; fi
SUFFIX=""; [ "$SCOPE" = "global" ] || SUFFIX="-$SCOPE"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate research
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cluster_env.sh
source gpu_preflight.sh

SPARSITY=$(printf "0.%02d" "$PCT")
python prune_magnitude.py --sparsity "$SPARSITY" --scope "$SCOPE" \
    --out_dir "models/llamax3-8b-mag${PCT}${SUFFIX}"

echo "Pruning (${PCT}%, ${SCOPE}) finished."
