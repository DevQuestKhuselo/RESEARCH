#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --time=02:00:00
#SBATCH --job-name=evaluate-baseline
#SBATCH -o /home-mscluster/tmhizha/RESEARCH/output/eval.%N.%j.out
#SBATCH -e /home-mscluster/tmhizha/RESEARCH/output/eval.%N.%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate comet_eval

cd ~/RESEARCH
mkdir -p output

python evaluate.py --results_dir results --output scores_baseline_summary.csv --pattern "baseline_*.csv"

echo "Evaluation job finished."
