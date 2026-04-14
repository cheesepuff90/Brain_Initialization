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

#SBATCH --output=logs/bi_10.log
#SBATCH --error=logs/bi_10.log

set -euo pipefail

echo "Host: $(hostname)"
nvidia-smi

echo "Starting Slurm job..."
source /home/kogs/bi/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Launching launch_ddp.sh..."
./launch_ddp.sh --config_preset perceptual_init_yfcc3m_vitb32 --log_dir /home/kogs/checkpoints --num_workers 8 --resume_ckpt_path /home/kogs/checkpoints/perceptual_init_yfcc3m_vitb32/checkpoints/epoch_epoch=003.ckpt
echo "Slurm job finished."