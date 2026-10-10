# Sourced by the SLURM job scripts after `conda activate`. Fails the job in
# seconds if the node's GPU is unusable, instead of hours in. On 10 Oct 2026
# mscluster110 ("No devices were found") and mscluster111 (all fields ERR!)
# accepted jobs but had no working GPU.
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; torch.zeros(1).cuda(); print('GPU OK:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"
