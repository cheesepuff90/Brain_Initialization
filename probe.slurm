#!/bin/bash
#SBATCH --mail-user=grace.s.ko@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=bi
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=10-00:00:00
#SBATCH --mem=400G
#SBATCH --partition=interactive_gpu
#SBATCH --account=dsi_dgx_iacc
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:8
#SBATCH --reservation=dgx98306

#SBATCH --output=logs_probe/imgnet_probe.log
#SBATCH --error=logs_probe/imgnet_probe.log

set -euo pipefail

echo "Host: $(hostname)"
date
nvidia-smi

echo "Activating environment..."
source /home/kogs/bi/bin/activate

export HF_HUB_TOKEN=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
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=/tmp/huggingface
export TRANSFORMERS_CACHE=/tmp/huggingface/models
export HF_HUB_CACHE=/tmp/huggingface/hub

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE"

echo "Starting ImageNet linear probe..."
python tests/probe_2.py

echo "Job finished."
date