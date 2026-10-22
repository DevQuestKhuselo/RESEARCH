#!/bin/bash
# One-off: download and cache the COMET and AfriCOMET scoring models (and the
# XLM-R encoders COMET fetches when loading them) into $HF_HOME, so scoring
# jobs can run offline. Runs on a CPU partition: no GPU is needed, and the
# GPU queue is not touched.
#   sbatch run_prefetch_eval.sh
#SBATCH --partition=batch
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --job-name=prefetch-eval
#SBATCH -o output/%x.%N.%j.out
#SBATCH -e output/%x.%N.%j.err

set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate comet_eval
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cluster_env.sh

for MODEL in Unbabel/wmt22-comet-da masakhane/africomet-stl-1.1; do
  echo "##### $MODEL"
  # 25 min cap per model: a stalled download fails instead of hanging
  timeout 1500 python -c "
import sys
from comet import download_model, load_from_checkpoint
path = download_model(sys.argv[1])
load_from_checkpoint(path)          # also fetches the encoder config/tokenizer
print('cached:', sys.argv[1], '->', path)
" "$MODEL"
done

# Prove the cache is complete: load both again with the network disabled
HF_HUB_OFFLINE=1 python -c "
import evaluate
evaluate.load_comet_models()
print('Offline load OK: scoring jobs can run.')
"
