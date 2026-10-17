#!/bin/bash
# Short latency/memory benchmark: first 100 devtest sentences of every
# direction. Run every model on the SAME node with --exclusive, e.g.
#   sbatch --exclusive -w mscluster106 run_timing.sh LLaMAX/LLaMAX3-8B-Alpaca fp16
#   sbatch --exclusive -w mscluster106 run_timing.sh models/llamax3-8b-gptq-8bit gptq8
#   sbatch --exclusive -w mscluster106 run_timing.sh models/llamax3-8b-gptq-4bit gptq4
# Writes results/timing_<name>_*.csv and results/efficiency_timing_<name>.csv.
# Quality scores come from the full run_translate.sh runs, not these files.
#SBATCH --partition=biggpu
#SBATCH --time=01:00:00
#SBATCH --job-name=timing
#SBATCH --exclude=mscluster[107-108,110-112]
#SBATCH -o output/%x.%N.%j.out
#SBATCH -e output/%x.%N.%j.err

set -eo pipefail
MODEL_PATH=${1:?usage: sbatch --exclusive -w <node> run_timing.sh <model_path> <name>}
NAME=${2:?usage: sbatch --exclusive -w <node> run_timing.sh <model_path> <name>}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate research
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cluster_env.sh
source gpu_preflight.sh
mkdir -p results

python run_all_baselines.py --model_path "$MODEL_PATH" --prefix "timing_${NAME}" \
    --splits devtest --limit 100

echo "Timing benchmark [$NAME] finished."
