#!/bin/bash
# Short latency/memory benchmark: first 100 devtest sentences of every
# direction, for several models back-to-back in ONE exclusive job (the
# cluster allows at most 6 queued+running jobs per user, and one job also
# guarantees identical conditions for every model).
#
#   sbatch --exclusive -w mscluster106 run_timing.sh
#       -> fp16, gptq8 and gptq4 (default)
#   sbatch --exclusive -w mscluster106 run_timing.sh models/llamax3-8b-mag50:mag50
#       -> any list of <model_path>:<name> pairs
#
# Writes results/timing_<name>_*.csv and results/efficiency_timing_<name>.csv.
# Quality scores come from the full run_translate.sh runs, not these files.
# Each model is a separate Python process, so GPU memory is freed in between.
# Resubmitting skips models already benchmarked.
#SBATCH --partition=biggpu
#SBATCH --time=02:00:00
#SBATCH --job-name=timing
#SBATCH --exclude=mscluster[107-108,110-112]
#SBATCH -o output/%x.%N.%j.out
#SBATCH -e output/%x.%N.%j.err

set -eo pipefail
if [ "$#" -eq 0 ]; then
  set -- "LLaMAX/LLaMAX3-8B-Alpaca:fp16" \
         "models/llamax3-8b-gptq-8bit:gptq8" \
         "models/llamax3-8b-gptq-4bit:gptq4"
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate research
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cluster_env.sh
source gpu_preflight.sh
mkdir -p results

for pair in "$@"; do
  MODEL_PATH=${pair%:*}
  NAME=${pair##*:}
  if [ -z "$MODEL_PATH" ] || [ -z "$NAME" ] || [ "$MODEL_PATH" = "$pair" ]; then
    echo "bad argument '$pair', expected <model_path>:<name>" >&2; exit 1
  fi
  echo "##### Timing benchmark: $NAME ($MODEL_PATH)"
  python run_all_baselines.py --model_path "$MODEL_PATH" --prefix "timing_${NAME}" \
      --splits devtest --limit 100
done

echo "Timing benchmarks finished: $*"
