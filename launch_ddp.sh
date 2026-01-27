#!/bin/bash

# This script launches distributed training with torchrun on 1 node with N GPUs
# Updated to support the full 3-step experiment across all model variants with unified naming

# --- Configuration ---
# Set the number of GPUs to use on this node
NUM_GPUS_PER_NODE=1 # Adjust as needed

# Training script entry point
TRAIN_SCRIPT="train_clip.py"

# --- Default Training Parameters (can be overridden by passing args to this script) ---
# These are passed directly to the train_clip.py script
# Example: ./launch_ddp.sh --compile --save_every_n_train_steps 2000 --batch_size 512
# Arguments set here will be overridden if the same argument is passed when calling the script.

# Config preset - Updated with unified naming conventions for 3-step experiment
# Step 1: NIGHTS initialization
#   - 'nights_init_vitb32', 'nights_init_vitb16', 'nights_init_vitl14', 'nights_init_vith14'
#   - 'nights_init_resnet50' (ResNet-50 variant)
# Step 2: Perceptual-initialized YFCC15M pretraining  
#   - 'perceptual_init_yfcc15m_vitb32', 'perceptual_init_yfcc15m_vitb16', 
#     'perceptual_init_yfcc15m_vitl14', 'perceptual_init_yfcc15m_vith14'
#   - 'perceptual_init_yfcc15m_resnet50' (ResNet-50 variant)
# Step 3: Baseline YFCC15M pretraining or Fine-tuning
#   - 'baseline_yfcc15m_vitb32', 'baseline_yfcc15m_vitb16', 'baseline_yfcc15m_vitl14', 'baseline_yfcc15m_vith14'
#   - 'declip_yfcc15m_litdata_resnet50' (ResNet-50 baseline)
# Step 4: Fine-tuning (TODO)
#   - 'nights_ft_from_yfcc_vitb32' (fine-tuning from baseline)
CONFIG_PRESET_FLAG="--config_preset perceptual_init_yfcc15m_vitb32" # e.g., set to "--config_preset nights_init_vitb32" to enable by default

# Training strategy (ddp, deepspeed, etc.)
STRATEGY="ddp"
# Enable torch.compile (requires PyTorch >= 2.0) - pass --compile to enable
COMPILE_FLAG="" # e.g., set to "--compile" to enable by default
# Number of top checkpoints to keep based on monitor metric - pass --save_top_k K
SAVE_TOP_K_FLAG="" # e.g., set to "--save_top_k 3" (preset default is 3)
# Batch size per device - pass --batch_size B
BATCH_SIZE_FLAG="" # e.g., set to "--batch_size 768" (preset default)
# Number of workers - pass --num_workers W
NUM_WORKERS_FLAG="" # e.g., set to "--num_workers 16" (preset default)
# Log directory - pass --log_dir PATH
LOG_DIR_FLAG="" # e.g., set to "--log_dir training_logs_litdata"
# Parquet data directory - pass --parquet_data_dir PATH
PARQUET_DATA_DIR_FLAG=""
# wandb - pass --wandb to enable
WANDB_FLAG="" # e.g., set to "--wandb" to enable by default
# Test mode - pass --test to enable
TEST_FLAG="" # e.g., set to "--test" to enable test mode by default

# --- End Configuration ---

# Example usage for 3-step experiment with unified naming:
# 
# Step 1: NIGHTS initialization (choose model variant)
# ./launch_ddp.sh --config_preset nights_init_vitb32        # ViT-B/32
# ./launch_ddp.sh --config_preset nights_init_vitb16        # ViT-B/16  
# ./launch_ddp.sh --config_preset nights_init_vitl14        # ViT-L/14
# ./launch_ddp.sh --config_preset nights_init_vith14        # ViT-H/14
# ./launch_ddp.sh --config_preset nights_init_resnet50      # ResNet-50
#
# Step 1 Alternative: THINGS or ImageNet-1k initialization (ablation studies)
# ./launch_ddp.sh --config_preset things_init_vitb32        # THINGS dataset
# ./launch_ddp.sh --config_preset imagenet1k_init_vitb32    # ImageNet-1k dataset
#
# Step 2: Perceptual-initialized YFCC15M pretraining
# ./launch_ddp.sh --config_preset perceptual_init_yfcc15m_vitb32   # ViT-B/32
# ./launch_ddp.sh --config_preset perceptual_init_yfcc15m_vitb16   # ViT-B/16
# ./launch_ddp.sh --config_preset perceptual_init_yfcc15m_vitl14   # ViT-L/14  
# ./launch_ddp.sh --config_preset perceptual_init_yfcc15m_vith14   # ViT-H/14
# ./launch_ddp.sh --config_preset perceptual_init_yfcc15m_resnet50 # ResNet-50
#
# Step 3: Baseline YFCC15M pretraining (for comparison)
# ./launch_ddp.sh --config_preset declip_yfcc15m_litdata_vitb32          # ViT-B/32
# ./launch_ddp.sh --config_preset declip_yfcc15m_litdata_vitb16          # ViT-B/16
# ./launch_ddp.sh --config_preset declip_yfcc15m_litdata_vitl14          # ViT-L/14
# ./launch_ddp.sh --config_preset declip_yfcc15m_litdata_vith14          # ViT-H/14
# ./launch_ddp.sh --config_preset declip_yfcc15m_litdata_resnet50        # ResNet-50
#
# Step 3: Fine-tuning (alternative)
# ./launch_ddp.sh --config_preset nights_ft_from_yfcc_vitb32      # ViT-B/32
#
# Test mode (quick pipeline verification):
# ./launch_ddp.sh --config_preset nights_init_vitb32 --test

# Parse any additional arguments to forward to the training script
# All arguments passed to this script (e.g., ./launch_ddp.sh --epochs 10)
# will be appended to the torchrun command and will OVERRIDE defaults set above.
CLI_ARGS="$@"

# Environment setup for torchrun
NUM_NODES=1 # Assuming single-node training
NODE_RANK=0
MASTER_ADDR=${MASTER_ADDR:-"localhost"} # Use environment variable if set, otherwise default
MASTER_PORT=${MASTER_PORT:-12355} # Use environment variable if set, otherwise default

# Ensure the package is installed (optional, if you prefer explicit install)
# if [ ! -f ".package_installed" ]; then
#     echo "Installing package in development mode..."
#     pip install -e .
#     touch .package_installed
# else
#     echo "Package assumed to be installed."
# fi

# --- Launch Training ---
echo "-----------------------------------------------------"
echo " Launching Distributed Training via torchrun"
echo "-----------------------------------------------------"
echo " Script:           $TRAIN_SCRIPT"
echo " Nodes:            $NUM_NODES"
echo " GPUs per node:    $NUM_GPUS_PER_NODE"
echo " Master Address:   $MASTER_ADDR"
echo " Master Port:      $MASTER_PORT"
echo "-----------------------------------------------------"
echo " Default Config Preset:      '$CONFIG_PRESET_FLAG'"
echo " Default Strategy:           $STRATEGY"
echo " Default Compile Flag:       '$COMPILE_FLAG'"
echo " Default Save Top K:         '$SAVE_TOP_K_FLAG'"
echo " Default Batch Size Flag:    '$BATCH_SIZE_FLAG'"
echo " Default Num Workers Flag:   '$NUM_WORKERS_FLAG'"
echo " Default Log Dir Flag:       '$LOG_DIR_FLAG'"
echo " Default Parquet Dir Flag:   '$PARQUET_DATA_DIR_FLAG'"
echo " Default Test Flag:          '$TEST_FLAG'"
echo " Default Wandb Flag:         '$WANDB_FLAG'"
echo "-----------------------------------------------------"
echo " CLI Overrides/Additions:    '$CLI_ARGS'"
echo "-----------------------------------------------------"

export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1} # Often helps prevent hangs
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}  # Disable Infiniband if causing issues
# Potentially increase shared memory if workers crash
# export NCCL_SHM_DISABLE=1

# Use torchrun for launching
# Arguments are ordered such that CLI_ARGS come last and can override defaults passed earlier
torchrun \
    --nnodes=$NUM_NODES \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $TRAIN_SCRIPT \
    --strategy=$STRATEGY \
    --devices=$NUM_GPUS_PER_NODE \
    --num_nodes=$NUM_NODES \
    $CONFIG_PRESET_FLAG \
    $PARQUET_DATA_DIR_FLAG \
    $COMPILE_FLAG \
    $SAVE_TOP_K_FLAG \
    $BATCH_SIZE_FLAG \
    $NUM_WORKERS_FLAG \
    $LOG_DIR_FLAG \
    $WANDB_FLAG \
    $TEST_FLAG \
    $CLI_ARGS # Append any extra arguments passed to this script

EXIT_CODE=$?
echo "-----------------------------------------------------"
if [ $EXIT_CODE -eq 0 ]; then
    echo " Training script finished successfully!"
else
    echo " Training script finished with errors (Exit Code: $EXIT_CODE)"
fi
echo "-----------------------------------------------------"

exit $EXIT_CODE 