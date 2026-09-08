#!/bin/bash                                                                     
#SBATCH --partition=biggpu                                                      
#SBATCH --time=12:00:00                                                         
#SBATCH --job-name=baseline-translate                                           
#SBATCH -o /home-mscluster/tmhizha/RESEARCH/output/slurm.%N.%j.out              
#SBATCH -e /home-mscluster/tmhizha/RESEARCH/output/slurm.%N.%j.err              

source ~/miniconda3/etc/profile.d/conda.sh
conda activate research

cd ~/RESEARCH
mkdir -p output results

python run_all_baselines.py

echo "Job finished."
