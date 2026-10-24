#!/bin/bash
# GPTQ quantisation job. Usage (from the repo root):
#   sbatch run_quantise.sh 8
#   sbatch run_quantise.sh 4
# Variants, built one after another in ONE job (saves job slots):
#   sbatch run_quantise.sh 4 s43 s44 en
#     s<N> = calibration seed N (balanced 4 languages) -> models/llamax3-8b-gptq-4bit-s<N>
#     en   = English-only calibration, seed 42         -> models/llamax3-8b-gptq-4bit-en
#SBATCH --partition=biggpu
#SBATCH --time=08:00:00
#SBATCH --job-name=gptq-quantise
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
BITS=${1:?usage: sbatch run_quantise.sh <4|8> [s<seed> | en ...]}
shift

source ~/miniconda3/etc/profile.d/conda.sh
conda activate research
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cluster_env.sh
source gpu_preflight.sh

if [ "$#" -eq 0 ]; then
  python quantise_gptq.py --bits "$BITS" --out_dir "models/llamax3-8b-gptq-${BITS}bit"
fi
for V in "$@"; do
  case "$V" in
    s[0-9]*) EXTRA=(--seed "${V#s}") ;;
    en)      EXTRA=(--calib_langs eng_Latn) ;;
    *) echo "unknown variant '$V' (use s<seed> or en)" >&2; exit 1 ;;
  esac
  echo "##### ${BITS}-bit variant $V"
  python quantise_gptq.py --bits "$BITS" --out_dir "models/llamax3-8b-gptq-${BITS}bit-$V" "${EXTRA[@]}"
done

echo "Quantisation (${BITS}-bit ${*:-default}) finished."
