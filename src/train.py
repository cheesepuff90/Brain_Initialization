import os
import argparse
import random
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
import math # Needed for scheduler estimation
from collections import OrderedDict

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint, LearningRateMonitor, EarlyStopping
)
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_info # For logging on rank 0

# Import configs
from config import (
    get_config,
    

    # Unified naming conventions
    NOD_INIT_VITB32_CONFIG,
    NOD_INIT_VITB16_CONFIG,
    NOD_INIT_VITL14_CONFIG,
    NOD_INIT_VITH14_CONFIG,
    NOD_INIT_RESNET50_CONFIG,
    
    # THINGS initialization configs
    THINGS_INIT_VITB32_CONFIG,
    # IMAGENET1K_INIT_VITB32_CONFIG,
    
    BASELINE_YFCC15M_VITB32_CONFIG,
    BASELINE_YFCC15M_VITB16_CONFIG,
    BASELINE_YFCC15M_VITL14_CONFIG,
    BASELINE_YFCC15M_VITH14_CONFIG,
    BASELINE_YFCC15M_RESNET50_CONFIG,
    
    BRAIN_INIT_YFCC15M_VITB32_CONFIG,
    BRAIN_INIT_YFCC15M_VITB16_CONFIG,
    BRAIN_INIT_YFCC15M_VITL14_CONFIG,
    BRAIN_INIT_YFCC15M_VITH14_CONFIG,
    BRAIN_INIT_YFCC15M_RESNET50_CONFIG,
    
    NIGHTS_FT_FROM_YFCC_VITB32_CONFIG,
    
    TrainingConfig, DeCLIPLitDataTrainingConfig, NodTrainingConfig, ThingsTrainingConfig,# Import base types
)
# Import dataset and module
from src.dataset import create_dataloaders # Updated function
from src.lightning_module import CLIPLightningModule

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Basic logging configuration

import torch

# 1) Wrap get_device_capability to catch the RuntimeError and return a dummy value
_orig_get_cap = torch.cuda.get_device_capability
def _safe_get_device_capability(device: int = 0):
    try:
        return _orig_get_cap(device)
    except RuntimeError:
        # pretend we have an Ampere card (sm_80)
        return (8, 0)
torch.cuda.get_device_capability = _safe_get_device_capability

# 2) Then force Lightning's check to just return False (or True if you prefer)
import lightning.fabric.accelerators.cuda as _cuda_acc
_cuda_acc._is_ampere_or_later = lambda *args, **kwargs: False

# Helper function to load vision weights (adapted from tests/test_init_with_nights.py)
def load_vision_weights_from_checkpoint(
    target_module: CLIPLightningModule,
    checkpoint_path: str,
    vision_prefix: str = "model.visual.",
    text_model_prefix: str = "model.text."
):
    """
    Loads only the vision weights from a checkpoint into the target_module.
    The text model weights in target_module will remain as initialized.
    """
    if not os.path.exists(checkpoint_path):
        rank_zero_info(f"Checkpoint file not found: {checkpoint_path}. Vision weights not loaded.")
        return

    rank_zero_info(f"Loading vision weights from checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as e:
        rank_zero_info(f"Failed to load checkpoint with weights_only=True: {e}")
        rank_zero_info("Attempting to load checkpoint with weights_only=False (ensure checkpoint is trusted).")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as e_fallback:
            rank_zero_info(f"Failed to load checkpoint with weights_only=False: {e_fallback}. Vision weights not loaded.")
            return

    state_dict = checkpoint.get("state_dict", checkpoint)
    if not state_dict:
        rank_zero_info("No state_dict found in checkpoint. Vision weights not loaded.")
        return

    vision_state_dict = OrderedDict()
    loaded_vision_keys = 0
    skipped_text_keys = 0
    skipped_other_keys = 0

    rank_zero_info(f"\nInspecting checkpoint keys for vision model (prefix: '{vision_prefix}')...")

    for key, value in state_dict.items():
        if key.startswith(vision_prefix):
            new_key = key[len(vision_prefix):]
            vision_state_dict[new_key] = value
            loaded_vision_keys += 1
        elif key.startswith(text_model_prefix):
            skipped_text_keys += 1
        else:
            # This key does not match the vision or text prefix.
            # It could be an optimizer state, scheduler state, or other metadata.
            skipped_other_keys += 1


    if loaded_vision_keys > 0:
        rank_zero_info(f"Found {loaded_vision_keys} keys matching vision prefix '{vision_prefix}'.")
        rank_zero_info(f"Skipped {skipped_text_keys} keys matching text prefix '{text_model_prefix}'.")
        rank_zero_info(f"Skipped {skipped_other_keys} other keys.")

        try:
            # Ensure the vision model component exists
            if hasattr(target_module, 'model') and hasattr(target_module.model, 'visual'):
                missing_keys, unexpected_keys = target_module.model.visual.load_state_dict(vision_state_dict, strict=False)
                rank_zero_info("\nAttempted to load vision weights into the target model's vision branch.")
                if missing_keys:
                    rank_zero_info(f"Missing keys in target vision model: {missing_keys}")
                if unexpected_keys:
                    rank_zero_info(f"Unexpected keys in checkpoint's vision state_dict: {unexpected_keys}")
                if not missing_keys and not unexpected_keys:
                    rank_zero_info("Successfully loaded vision weights with no missing or unexpected keys.")
                else:
                    rank_zero_info("Vision weights loaded with some discrepancies (check missing/unexpected keys above).")
            else:
                rank_zero_info("Target module does not have 'model.visual' attribute. Vision weights not loaded.")

        except RuntimeError as e:
            rank_zero_info(f"\nError loading vision state_dict: {e}")
            rank_zero_info("This might be due to model architecture mismatch or incorrect key mapping.")
            rank_zero_info("Ensure the vision model architecture in the checkpoint is compatible.")
            if hasattr(target_module, 'model') and hasattr(target_module.model, 'visual'):
                rank_zero_info("Target vision model keys:")
                for k_idx, k_val in enumerate(target_module.model.visual.state_dict().keys()):
                    if k_idx < 10: rank_zero_info(f"  {k_val}") # Print first 10
                    elif k_idx == 10: rank_zero_info("  ...")
                rank_zero_info("Source vision model keys (from checkpoint, after stripping prefix):")
                for k_idx, k_val in enumerate(vision_state_dict.keys()):
                    if k_idx < 10: rank_zero_info(f"  {k_val}") # Print first 10
                    elif k_idx == 10: rank_zero_info("  ...")

    else:
        rank_zero_info(f"No keys found with the vision prefix '{vision_prefix}'. Vision weights not loaded.")
        rank_zero_info("Please check the checkpoint structure and the prefix.")
        rank_zero_info("Available top-level keys in checkpoint's state_dict (first 20):")
        for k_idx, k_val in enumerate(list(state_dict.keys())):
            if k_idx < 20: rank_zero_info(f"  {k_val}")
            elif k_idx == 20: rank_zero_info("  ..."); break
    
    # The text branch remains as initialized by CLIPLightningModule
    rank_zero_info("Text branch of the target model remains as per its initial initialization.")

def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP model")

    # --- Arguments to select Training Mode ---
    parser.add_argument(
        "--config_preset", type=str, default="nod_init_vitb32",
        choices=[
            # NOD initialization (Step 1)
            "nod_init_vitb32", "nod_init_vitb16", "nod_init_vitl14", "nod_init_vith14",
            "nod_init_resnet50",  # ResNet-50 variant
            # THINGS initialization (Alternative to NOD)
            "things_init_vitb32",
            # ImageNet-1k initialization (Ablation study)
            "imagenet1k_init_vitb32",
            # Baseline YFCC15M pretraining (Step 3 baseline)
            "declip_yfcc15m_litdata_vitb32", "declip_yfcc15m_litdata_vitb16", "declip_yfcc15m_litdata_vitl14", "declip_yfcc15m_litdata_vith14",
            "declip_yfcc15m_litdata_resnet50",  # ResNet-50 baseline
            # brain-initialized YFCC15M pretraining (Step 2)
            "brain_init_yfcc15m_vitb32", "brain_init_yfcc15m_vitb16", 
            "brain_init_yfcc15m_vitl14", "brain_init_yfcc15m_vith14",
            "brain_init_yfcc15m_resnet50",  # ResNet-50 brain init
            # Fine-tuning (Step 3)
            "nights_ft_from_yfcc_vitb32"
        ],
        help="Choose a configuration preset for the 3-step experiment."
    )

    # --- Test Mode ---
    parser.add_argument(
        "--test", action="store_true", default=False,
        help="Test mode: run only 5 training steps and 1 epoch to verify the training pipeline works."
    )

    # --- Dataset Arguments (Overrides presets if provided) ---
    parser.add_argument(
        "--dataset_type", type=str, choices=["nod", "things", "imagenet1k", "yfcc15m", "yfcc15m_litdata"], # Added things and imagenet1k
        help="Type of dataset ('nod', 'things', 'imagenet1k', 'yfcc15m' (HF streaming), 'yfcc15m_litdata'). Overrides preset."
    )
    parser.add_argument(
        "--dataset_root", type=str, help="Path to the NOD dataset root."
    )
    parser.add_argument(
        "--hf_dataset_name", type=str, help="HF dataset name (if dataset_type='yfcc15m')."
    )
    parser.add_argument(
        "--hf_cache_dir", type=str, help="HF cache directory."
    )
    parser.add_argument(
        "--hf_streaming", action="store_true", default=False, help="Enable HF streaming (if dataset_type='yfcc15m')."
    )
    parser.add_argument(
        "--use_litdata", action="store_true", default=None, help="Force use of LitData (if dataset_type='yfcc15m_litdata')."
    )
    parser.add_argument(
        "--parquet_data_dir", type=str, help="Path to local parquet files directory (for litdata). Overrides preset."
    )
    parser.add_argument(
        "--image_col", type=str, help="Column name for image data. Overrides preset."
    )
    parser.add_argument(
        "--text_col", type=str, help="Column name for text data. Overrides preset."
    )
    parser.add_argument(
        "--tokenizer_name", type=str, help="Tokenizer model name. Overrides preset."
    )
    parser.add_argument(
        "--val_dataset_root", type=str, help="Root directory for validation dataset (e.g., CIFAR10, CIFAR100, etc.). Overrides preset."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, help="Specific batch size for evaluation dataloader. Overrides preset."
    )
    parser.add_argument(
        "--multi_val", action="store_true", default=False, help="Enable multiple validation datasets (CIFAR10 and CIFAR100)"
    )

    # --- Training Arguments (Overrides presets if provided) ---
    parser.add_argument("--batch_size", type=int, help="Batch size per device. Overrides preset.")
    parser.add_argument("--num_workers", type=int, help="Dataloader workers. Overrides preset.")
    parser.add_argument("--epochs", type=int, help="Number of epochs. Overrides preset.")
    parser.add_argument("--lr", type=float, help="Learning rate. Overrides preset.")
    parser.add_argument("--weight_decay", type=float, help="Weight decay. Overrides preset.")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "sgd"], help="Optimizer. Overrides preset.")
    parser.add_argument("--scheduler", type=str, choices=["cosine", "const", "const-cooldown"], help="LR scheduler. Overrides preset.")
    parser.add_argument("--warmup_steps", type=int, help="LR warmup steps. Overrides preset.")
    parser.add_argument("--gradient_clip_val", type=float, help="Gradient clipping value. Overrides preset.")
    parser.add_argument("--accumulate_grad_batches", type=int, help="Gradient accumulation steps. Overrides preset.")
    parser.add_argument(
        "--validate_every_n_epoch", 
        type=int, 
        help="Run validation every N epochs (default=1, None to disable). Overrides preset. Takes priority over --validate_every_n_steps."
    )
    parser.add_argument(
        "--validate_every_n_steps", 
        type=int, 
        help="Run validation every N steps or fraction of epoch (float 0-1 = fraction of epoch, int > 1 = steps). Overrides preset."
    )

    # --- Loss Arguments (Overrides presets if provided) ---
    parser.add_argument("--margin", type=float, help="Triplet loss margin (for nights). Overrides preset.")
    parser.add_argument("--distance_type", type=str, choices=["cosine", "euclidean"], help="Triplet loss distance type (for nights). Overrides preset.")
    # CLIP loss params are usually fixed in the config

    # --- Device Arguments ---
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator type (auto, gpu, cpu).")
    parser.add_argument("--devices", type=str, default="auto", help="Device indices (comma-separated), number of devices, or 'auto'.")

    # --- Distributed Training Arguments ---
    parser.add_argument("--strategy", type=str, default=None, choices=["ddp", "ddp_spawn", "ddp_sharded", "deepspeed", None], help="Distributed training strategy.")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes for distributed training.")
    parser.add_argument("--sync_batchnorm", action="store_true", default=None, help="Enable sync batchnorm.")
    parser.add_argument("--no_sync_batchnorm", action="store_false", dest="sync_batchnorm", help="Disable sync batchnorm.")
    parser.add_argument("--find_unused_parameters", action="store_true", default=None, help="Find unused parameters in DDP.")
    parser.add_argument("--no_find_unused_parameters", action="store_false", dest="find_unused_parameters", help="Disable find unused parameters in DDP.")
    parser.add_argument("--deterministic", action="store_true", default=None, help="Use deterministic algorithms.")
    parser.add_argument("--no_deterministic", action="store_false", dest="deterministic", help="Disable deterministic algorithms.")


    # --- Logging and Checkpointing ---
    parser.add_argument("--log_dir", type=str, default="/tmp/checkpoints/pretrain_checkpoints", help="Directory for saving logs and checkpoints.")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name (default: auto-generated).")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="Log training metrics every N steps.")
    parser.add_argument("--save_top_k", type=int, help="Save top-k checkpoints based on monitor metric. Overrides preset.")
    parser.add_argument("--monitor_metric", type=str, default=None, help="Metric to monitor for checkpoints/early stopping. Overrides preset.")
    parser.add_argument("--monitor_mode", type=str, choices=["min", "max"], help="Mode for monitored metric ('min' or 'max'). Overrides preset.")
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")

    # --- Weights & Biases Arguments ---
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, help="W&B project name. Overrides preset.")
    parser.add_argument("--wandb_entity", type=str, help="W&B entity name. Overrides preset.")
    parser.add_argument("--wandb_tags", type=str, help="W&B tags (comma-separated). Overrides preset.")
    parser.add_argument("--wandb_log_model", type=str, help="Log model checkpoints to W&B ('all', 'best', 'False'). Overrides preset.")
    parser.add_argument("--wandb_save_code", action="store_true", default=None, help="Save code to W&B.")
    parser.add_argument("--no_wandb_save_code", action="store_false", dest="wandb_save_code", help="Disable code saving to W&B.")

    # --- Performance Arguments ---
    parser.add_argument("--compile", action="store_true", default=False, help="Enable torch.compile (requires PyTorch >= 2.0).")
    parser.add_argument("--no_compile", action="store_false", dest="compile", help="Disable torch.compile.")

    # --- Reproducibility ---
    parser.add_argument("--seed", type=int, help="Random seed. Overrides preset.")

    # --- Initialization Arguments ---
    parser.add_argument(
        "--init_ckpt_path", type=str, default=None,
        help="Path to initialize vision model from (e.g., NOD checkpoint). Vision branch only."
    )
    parser.add_argument(
        "--freeze_vision_blocks", type=int, default=None,
        help="Number of ViT blocks to freeze in the vision model (0 = none, >0 freezes from the start)."
    )
    
    return parser.parse_args()

# --- Helper Functions (count_available_gpus, detect_torchrun_environment, set_seed) ---
def count_available_gpus():
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()

def detect_torchrun_environment():
    env_keys = ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'NODE_RANK', 'MASTER_ADDR', 'MASTER_PORT']
    env_info = {}
    torchrun_detected = all(key in os.environ for key in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK'])
    if torchrun_detected:
        for key in env_keys:
            env_info[key] = os.environ.get(key)
    return torchrun_detected, env_info

def set_seed(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
             logger.info("Setting deterministic CUDA options")
             torch.backends.cudnn.deterministic = True
             torch.backends.cudnn.benchmark = False
        else:
             logger.info("Setting non-deterministic CUDA options (benchmark=True)")
             torch.backends.cudnn.deterministic = False
             torch.backends.cudnn.benchmark = True


# --- Configuration Setup ---
def setup_training_config(args) -> Union[TrainingConfig, DeCLIPLitDataTrainingConfig, NodTrainingConfig]:
    """Create training configuration from preset and command-line args"""
    default_monitor_metric = None
    default_monitor_mode = "min"
    current_config_class = TrainingConfig # Default base
    using_litdata = False 

    # Start with the selected preset config
    if args.config_preset == "nod_init_vitb32":
        config = NOD_INIT_VITB32_CONFIG.model_copy(deep=True)
        current_config_class = NodTrainingConfig
        default_monitor_metric = config.monitor_metric
        default_monitor_mode = config.monitor_mode
    elif args.config_preset == "nod_init_vitb16":
        config = NOD_INIT_VITB16_CONFIG.model_copy(deep=True)
        current_config_class = NodTrainingConfig
        default_monitor_metric = config.monitor_metric
        default_monitor_mode = config.monitor_mode
    elif args.config_preset == "nod_init_vitl14":
        config = NOD_INIT_VITL14_CONFIG.model_copy(deep=True)
        current_config_class = NodTrainingConfig
        default_monitor_metric = config.monitor_metric
        default_monitor_mode = config.monitor_mode
    elif args.config_preset == "nod_init_vith14":
        config = NOD_INIT_VITH14_CONFIG.model_copy(deep=True)
        current_config_class = NodTrainingConfig
        default_monitor_metric = config.monitor_metric
        default_monitor_mode = config.monitor_mode
    elif args.config_preset == "nod_init_resnet50":
        config = NOD_INIT_RESNET50_CONFIG.model_copy(deep=True)
        current_config_class = NodTrainingConfig
        default_monitor_metric = config.monitor_metric
        default_monitor_mode = config.monitor_mode
        # THINGS initialization configs  
    elif args.config_preset == "things_init_vitb32":
        config = THINGS_INIT_VITB32_CONFIG.model_copy(deep=True)
        current_config_class = ThingsTrainingConfig  
        default_monitor_metric = config.monitor_metric
        default_monitor_mode = config.monitor_mode
    # ImageNet-1k initialization config (ablation study)
    # elif args.config_preset == "imagenet1k_init_vitb32":
    #     config = IMAGENET1K_INIT_VITB32_CONFIG.model_copy(deep=True)
    #     current_config_class = TrainingConfig  
    #     default_monitor_metric = config.monitor_metric
    #     default_monitor_mode = config.monitor_mode
    # Baseline YFCC15M pretraining (Step 3 baseline)
    elif args.config_preset == "declip_yfcc15m_litdata_vitb32":
        config = BASELINE_YFCC15M_VITB32_CONFIG.model_copy(deep=True)
        current_config_class = DeCLIPLitDataTrainingConfig
        default_monitor_metric = config.monitor_metric
        default_monitor_mode = config.monitor_mode
        using_litdata = True
    elif args.config_preset == "declip_yfcc15m_litdata_vitb16":
        config = BASELINE_YFCC15M_VITB16_CONFIG.model_copy(deep=True)
        current_config_class = DeCLIPLitDataTrainingConfig
        default_monitor_metric = config.monitor_metric
        default_monitor_mode = config.monitor_mode
        using_litdata = True
    elif args.config_preset == "declip_yfcc15m_litdata_vitl14":
        config = BASELINE_YFCC15M_VITL14_CONFIG.model_copy(deep=True)
        current_config_class = DeCLIPLitDataTrainingConfig
        default_monitor_metric = config.monitor_metric
        default_monitor_mode = config.monitor_mode
        using_litdata = True
    elif args.config_preset == "declip_yfcc15m_litdata_vith14":
        config = BASELINE_YFCC15M_VITH14_CONFIG.model_copy(deep=True)
        current_config_class = DeCLIPLitDataTrainingConfig
        default_monitor_metric = config.monitor_metric
        default_monitor_mode = config.monitor_mode
        using_litdata = True
    elif args.config_preset == "declip_yfcc15m_litdata_resnet50":
        config = BASELINE_YFCC15M_RESNET50_CONFIG.model_copy(deep=True)
        current_config_class = DeCLIPLitDataTrainingConfig
        default_monitor_metric = config.monitor_metric
        default_monitor_mode = config.monitor_mode
        using_litdata = True
    # brain-initialized YFCC15M pretraining (Step 2)
    elif args.config_preset == "brain_init_yfcc15m_vitb32":
        config = BRAIN_INIT_YFCC15M_VITB32_CONFIG.model_copy(deep=True)
        current_config_class = DeCLIPLitDataTrainingConfig
        default_monitor_metric = config.monitor_metric
        default_monitor_mode = config.monitor_mode
        using_litdata = True
    elif args.config_preset == "brain_init_yfcc15m_vitb16":
        config = BRAIN_INIT_YFCC15M_VITB16_CONFIG.model_copy(deep=True)
        current_config_class = DeCLIPLitDataTrainingConfig
        default_monitor_metric = config.monitor_metric
        default_monitor_mode = config.monitor_mode
        using_litdata = True
    elif args.config_preset == "brain_init_yfcc15m_vitl14":
        config = BRAIN_INIT_YFCC15M_VITL14_CONFIG.model_copy(deep=True)
        current_config_class = DeCLIPLitDataTrainingConfig
        default_monitor_metric = config.monitor_metric
        default_monitor_mode = config.monitor_mode
        using_litdata = True
    elif args.config_preset == "brain_init_yfcc15m_vith14":
        config = BRAIN_INIT_YFCC15M_VITH14_CONFIG.model_copy(deep=True)
        current_config_class = DeCLIPLitDataTrainingConfig
        default_monitor_metric = config.monitor_metric
        default_monitor_mode = config.monitor_mode
        using_litdata = True
    elif args.config_preset == "brain_init_yfcc15m_resnet50":
        config = BRAIN_INIT_YFCC15M_RESNET50_CONFIG.model_copy(deep=True)
        current_config_class = DeCLIPLitDataTrainingConfig
        default_monitor_metric = config.monitor_metric
        default_monitor_mode = config.monitor_mode
        using_litdata = True
    # Fine-tuning (Step 3)
    elif args.config_preset == "nights_ft_from_yfcc_vitb32":
        config = NIGHTS_FT_FROM_YFCC_VITB32_CONFIG.model_copy(deep=True)
        current_config_class = NodTrainingConfig
        default_monitor_metric = config.monitor_metric
        default_monitor_mode = config.monitor_mode
        # This mode does not use LitData for its primary NIGHTS dataset
        using_litdata = False
    else:
        raise ValueError(f"Invalid config_preset: {args.config_preset}.")

    # Handle wandb enabled/disabled logic first - this takes precedence over everything
    if args.wandb:
        # Explicitly enable wandb
        config.wandb.enabled = True
        rank_zero_info("W&B explicitly enabled via --wandb flag.")
    else:
        # Use the config default if --wandb is not specified
        # If --wandb is False (default), we still respect the preset config's wandb setting
        # Only override if explicitly set to False via command line
        if hasattr(args, 'wandb') and args.wandb is False:
            # This means --wandb was not passed, so use config default
            pass  # Keep config.wandb.enabled as is from preset
        rank_zero_info(f"W&B setting from config: {config.wandb.enabled}")

    # Override with command-line args if they are provided
    # Data settings
    if args.dataset_type is not None:
        config.data.dataset_type = args.dataset_type
        # Update using_litdata flag based on final dataset_type
        using_litdata = (config.data.dataset_type == "yfcc15m_litdata")
    if args.dataset_root is not None: config.data.dataset_root = args.dataset_root # For nights
    # HF settings
    if args.hf_dataset_name is not None: config.data.hf_dataset_name = args.hf_dataset_name
    if args.hf_cache_dir is not None: config.data.hf_cache_dir = args.hf_cache_dir
    if args.hf_streaming is not None: config.data.hf_streaming = args.hf_streaming
    # LitData settings
    if args.use_litdata is not None: # Explicit override
        config.data.use_litdata = args.use_litdata
        using_litdata = config.data.use_litdata
    else: # Infer from type if not explicitly set
         config.data.use_litdata = using_litdata

    if args.parquet_data_dir is not None: config.data.parquet_data_dir = args.parquet_data_dir

    if args.image_col is not None: config.data.image_col = args.image_col
    if args.text_col is not None: config.data.text_col = args.text_col
    if args.tokenizer_name is not None: config.data.tokenizer_name = args.tokenizer_name
    if args.batch_size is not None: config.data.batch_size = args.batch_size
    if args.num_workers is not None: config.data.num_workers = args.num_workers
    if args.val_dataset_root is not None: config.data.val_dataset_root = args.val_dataset_root
    if args.eval_batch_size is not None: config.data.eval_batch_size = args.eval_batch_size


    # Training settings
    if args.epochs is not None: config.epochs = args.epochs
    if args.seed is not None: config.seed = args.seed
    if args.accelerator is not None: config.accelerator = args.accelerator
    if args.devices is not None: config.devices = args.devices # Parsed later
    if args.gradient_clip_val is not None: config.gradient_clip_val = args.gradient_clip_val
    if args.accumulate_grad_batches is not None: config.accumulate_grad_batches = args.accumulate_grad_batches
    if args.log_every_n_steps is not None: config.log_every_n_steps = args.log_every_n_steps
    # if args.log_dir is not None: config.save_dir = args.log_dir
    if args.save_top_k is not None: config.save_top_k = args.save_top_k
    # if args.validate_every_n_epoch is not None:
    #     config.validate_every_n_epoch = args.validate_every_n_epoch
    #     config.validate_every_n_steps = None # Explicitly disable step-based validation
    # elif args.validate_every_n_steps is not None:
    #     config.validate_every_n_steps = args.validate_every_n_steps
    #     config.validate_every_n_epoch = None # Explicitly disable epoch-based validation
    # --- With this updated logic ---
    if args.validate_every_n_steps is not None:
        config.validate_every_n_steps = args.validate_every_n_steps
        config.validate_every_n_epoch = None  # Ensure epoch-based validation is disabled
    elif args.validate_every_n_epoch is not None:
        config.validate_every_n_epoch = args.validate_every_n_epoch
        config.validate_every_n_steps = None
    
    config.save_every_n_train_steps = None

    # Optimizer settings
    if args.optimizer is not None: config.optimizer.name = args.optimizer
    if args.lr is not None: config.optimizer.lr = args.lr
    if args.weight_decay is not None: config.optimizer.weight_decay = args.weight_decay

    # Scheduler settings
    if args.scheduler is not None: config.scheduler.name = args.scheduler
    if args.warmup_steps is not None: config.scheduler.warmup_steps = args.warmup_steps
    config.scheduler.epochs = None # Not directly used by scheduler config
    # Ensure epochs are set
    if not config.epochs or config.epochs <= 0:
         rank_zero_info(f"Warning: Invalid epochs value ({config.epochs}). Defaulting to 1 epoch.")
         config.epochs = 1


    # Loss settings
    if hasattr(config.loss, 'margin') and args.margin is not None: config.loss.margin = args.margin
    if hasattr(config.loss, 'distance_type') and args.distance_type is not None: config.loss.distance_type = args.distance_type

    # Distributed training settings
    if args.strategy is not None: config.distributed.strategy = args.strategy
    if args.sync_batchnorm is not None: config.distributed.sync_batchnorm = args.sync_batchnorm # Handled later based on distribution status
    if args.find_unused_parameters is not None: config.distributed.find_unused_parameters = args.find_unused_parameters
    if args.num_nodes is not None: config.distributed.num_nodes = args.num_nodes

    # Wandb settings - only override if wandb is enabled
    if config.wandb.enabled:
        if args.wandb_project is not None: 
            config.wandb.project = args.wandb_project
        if args.experiment_name is not None: 
            config.wandb.name = args.experiment_name # Set later if None
        if args.wandb_entity is not None: 
            config.wandb.entity = args.wandb_entity
        if args.wandb_tags is not None: 
            config.wandb.tags = args.wandb_tags.split(",") if args.wandb_tags else []
        # Handle log_model boolean/string logic
        if args.wandb_log_model is not None:
            log_model_arg = args.wandb_log_model.lower()
            if log_model_arg == 'false':
                config.wandb.log_model = False
            elif log_model_arg in ['true', 'all', 'best']:
                config.wandb.log_model = log_model_arg # Keep 'all' or 'best' as string
            else:
                logger.warning(f"Invalid --wandb_log_model value: {args.wandb_log_model}. Using preset default: {config.wandb.log_model}.")
        if args.wandb_save_code is not None: 
            config.wandb.save_code = args.wandb_save_code
    else:
        rank_zero_info("W&B is disabled - ignoring all wandb-related CLI arguments.")

    # --- Initialization settings ---
    if args.init_ckpt_path is not None:
        config.init_ckpt_path = args.init_ckpt_path
    if args.freeze_vision_blocks is not None:
        config.freeze_vision_blocks = args.freeze_vision_blocks


    # --- Set Monitor Metric/Mode AFTER potential overrides ---
    config.monitor_metric = args.monitor_metric if args.monitor_metric is not None else default_monitor_metric
    config.monitor_mode = args.monitor_mode if args.monitor_mode is not None else default_monitor_mode

    # --- Test Mode Override ---
    if args.test:
        rank_zero_info("TEST MODE ENABLED: Overriding configuration for quick pipeline verification.")
        config.epochs = 1
        config.log_every_n_steps = 1
        config.validate_every_n_epoch = 1
        config.validate_every_n_steps = None
        config.save_top_k = 1
        # Add test mode tag to wandb
        if hasattr(config, 'wandb') and config.wandb.tags:
            config.wandb.tags.append("test-mode")
        rank_zero_info("Test mode: Training for 1 epoch with 5 training steps and frequent logging.")

    # Make sure the final config object is of the correct Pydantic type
    # This is important if args heavily modified a base TrainingConfig
    if not isinstance(config, current_config_class) and current_config_class != TrainingConfig:
        try:
            config_dict = config.dict()
            # Filter dict for fields present in target class to avoid errors
            valid_fields = {f for f in current_config_class.model_fields.keys()}
            filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
            # Special handling for nested Pydantic models if direct casting fails
            # For now, assume direct casting from filtered dict works for major structures
            config = current_config_class(**filtered_dict)
            rank_zero_info(f"Config re-casted to {current_config_class.__name__} after applying CLI overrides.")
        except Exception as e:
            rank_zero_info(f"Warning: Could not re-cast config to {current_config_class.__name__} after overrides: {e}. Using original type {type(config).__name__}.")

    rank_zero_info(f"Final Configuration Type: {type(config).__name__}")
    rank_zero_info(f"Training for {config.epochs} epochs.")
    rank_zero_info(f"Using LitData: {config.data.use_litdata}")
    rank_zero_info(f"Monitoring metric: '{config.monitor_metric}' (mode: {config.monitor_mode})")
    if config.monitor_metric is None and (args.save_top_k is None or config.save_top_k > 0):
        rank_zero_info("No monitor metric set. Checkpointing will only save the 'last' checkpoint.")

    return config

# --- Callback and Logger Setup ---
def setup_callbacks(config: TrainingConfig, experiment_name: str, args):
    callbacks = []
    ckpt_path = os.path.join(config.save_dir, experiment_name, "checkpoints")
    os.makedirs(ckpt_path, exist_ok=True)
    rank_zero_info(f"Checkpoints will be saved to: {ckpt_path}")

    monitor_enabled = config.monitor_metric is not None

    # Callback: Save based on monitor metric (top-k) AND save the last checkpoint
    if monitor_enabled:
        monitor_key_slug = config.monitor_metric.replace('/', '_')
        monitor_filename_format = f"monitor_{monitor_key_slug}" + "-epoch_{epoch:03d}"
        top_k_checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_path,
            filename=monitor_filename_format,
            monitor=config.monitor_metric,
            mode=config.monitor_mode,
            save_top_k=config.save_top_k,
            save_last=True, # Ensure the last checkpoint is always saved
        )
        callbacks.append(top_k_checkpoint_callback)
        rank_zero_info(f"Checkpointing Strategy: Monitor='{config.monitor_metric}', Mode='{config.monitor_mode}', Save Top K={config.save_top_k}, Save Last=True")
    else:
         # Fallback: If no monitoring is enabled, just save the last checkpoint
         last_checkpoint_callback = ModelCheckpoint(
             dirpath=ckpt_path,
             filename="last-epoch_{epoch:03d}",
             save_last=True,
             save_top_k=0 # Explicitly disable top-k
         )
         callbacks.append(last_checkpoint_callback)
         rank_zero_info("Checkpointing Strategy: No monitor metric set. Saving only the last checkpoint.")

    # --- LR Monitor ---
    lr_monitor = LearningRateMonitor(logging_interval="step") # Log LR every step
    callbacks.append(lr_monitor)

    # --- Early Stopping ---
    if args.early_stopping and monitor_enabled:
        early_stop_callback = EarlyStopping(
            monitor=config.monitor_metric,
            min_delta=1e-5,
            patience=args.patience,
            verbose=True,
            mode=config.monitor_mode,
            check_on_train_epoch_end=False # Check after validation epoch
        )
        callbacks.append(early_stop_callback)
        rank_zero_info(f"Early Stopping Enabled: Monitor='{config.monitor_metric}', Patience={args.patience}")
    elif args.early_stopping and not monitor_enabled:
         rank_zero_info("Early stopping requested but no monitor_metric set. Disabling early stopping.")

    return callbacks

def setup_loggers(config: TrainingConfig, experiment_name: str, args):
    loggers = []
    tb_logger = TensorBoardLogger(
        save_dir=config.save_dir,
        name=experiment_name,
    )
    loggers.append(tb_logger)

    # Only setup wandb if it's enabled in config
    if config.wandb.enabled:
        is_torchrun, env_info = detect_torchrun_environment()
        # Only init wandb on global rank 0
        rank = int(env_info.get('RANK', 0)) if is_torchrun else 0
        if rank == 0:
            try:
                # Use experiment name for wandb run name if not overridden
                wandb_name = config.wandb.name if config.wandb.name else experiment_name
                wandb_logger = WandbLogger(
                    project=config.wandb.project,
                    name=wandb_name,
                    entity=config.wandb.entity,
                    tags=config.wandb.tags,
                    log_model=config.wandb.log_model, # Pass the processed value
                    save_code=config.wandb.save_code,
                    # Group runs by node if using torchrun across nodes
                    group=f"exp_{experiment_name}_node{env_info.get('NODE_RANK', 0)}" if is_torchrun and config.distributed.num_nodes > 1 else f"exp_{experiment_name}",
                    job_type="training",
                    # Pass config dict to wandb
                    config=config.dict() # Log the final merged config
                )
                loggers.append(wandb_logger)
                logger.info(f"Weights & Biases logging enabled for project '{config.wandb.project}', run '{wandb_name}'.")
                logger.info(f"W&B tags: {config.wandb.tags}")
            except Exception as e:
                logger.error(f"Failed to initialize WandbLogger: {e}. Disabling wandb.")
                config.wandb.enabled = False # Disable it in the config object too
        else:
            logger.info(f"Skipping wandb initialization on rank {rank}.")
            # Ensure wandb is marked disabled for non-zero ranks if logic depends on it later
            config.wandb.enabled = False
    else:
        logger.info("W&B logging is disabled.")

    return loggers

# --- Main Training Function ---
def main():
    args = parse_args()
    is_torchrun, env_info = detect_torchrun_environment()
    rank = int(env_info.get('RANK', 0)) if is_torchrun else 0

    # Setup configuration based on preset and overrides
    config = setup_training_config(args)
    
    # Load the appropriate model config for the selected training config
    _, model_config = get_config(args.config_preset)

    # Generate experiment name if needed
    experiment_name = args.experiment_name
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Make name more descriptive
        preset_name = args.config_preset or config.data.dataset_type
        experiment_name = f"{preset_name}_{timestamp}"
        if is_torchrun:
            experiment_name += f"_ddp_node{env_info.get('NODE_RANK', 0)}" # Add node rank

    # Set seed *before* initializing models or dataloaders
    # Use the final seed value from the config object
    # set_seed(config.seed, config.deterministic)
    # rank_zero_info(f"Set seed to {config.seed} (Deterministic: {config.deterministic})")
    set_seed(config.seed, getattr(config, "deterministic", True))
    rank_zero_info(f"Set seed to {config.seed} (Deterministic: {getattr(config, 'deterministic', True)})")

    # --- Set torch matmul precision for Tensor Cores optimization ---
    if torch.cuda.is_available():
        try:
            # Set to 'medium' for good balance of performance and precision on A100 GPUs
            torch.set_float32_matmul_precision('medium')
            rank_zero_info("Set torch.set_float32_matmul_precision to 'medium' for Tensor Cores optimization")
        except Exception as e:
            rank_zero_info(f"Failed to set torch matmul precision: {e}")

    # --- Determine Number of Devices and World Size ---
    num_avail_gpus = count_available_gpus()
    is_distributed = False
    world_size = 1
    local_batch_size = config.data.batch_size # Batch size per device

    if isinstance(config.devices, str) and config.devices != "auto":
        try:
            if "," in config.devices:
                devices_list = [int(x.strip()) for x in config.devices.split(",")]
                config.devices = devices_list
                num_devices_per_node = len(devices_list)
                world_size = num_devices_per_node * config.distributed.num_nodes
                is_distributed = world_size > 1
                rank_zero_info(f"Using specified devices: {devices_list} (Count: {num_devices_per_node})")
            else:
                num_devices_per_node = int(config.devices)
                config.devices = num_devices_per_node # Store the count
                world_size = num_devices_per_node * config.distributed.num_nodes
                is_distributed = world_size > 1
                rank_zero_info(f"Using specified device count: {num_devices_per_node}")
        except ValueError:
            rank_zero_info(f"Invalid devices format: {config.devices}. Using 'auto'.")
            config.devices = "auto"

    if config.devices == "auto":
        if is_torchrun:
             # Read from torchrun env vars
             local_world_size = int(env_info.get('LOCAL_WORLD_SIZE', 1))
             num_nodes = int(env_info.get('NNODES', config.distributed.num_nodes)) # Use NNODES from torchrun if available
             world_size = int(env_info.get('WORLD_SIZE', 1))

             if num_nodes != config.distributed.num_nodes:
                  rank_zero_info(f"Overriding config.distributed.num_nodes ({config.distributed.num_nodes}) with torchrun NNODES ({num_nodes})")
                  config.distributed.num_nodes = num_nodes

             config.devices = local_world_size # Let Lightning handle device IDs per node based on count
             is_distributed = world_size > 1
             rank_zero_info(f"torchrun detected. world_size={world_size}, devices_per_node={local_world_size}, num_nodes={num_nodes}")
        elif num_avail_gpus > 0:
             # Use all available GPUs on the node if not using torchrun
             config.devices = num_avail_gpus # Use the count
             world_size = num_avail_gpus * config.distributed.num_nodes # Assumes same num GPUs per node
             is_distributed = world_size > 1
             rank_zero_info(f"Auto-detected {num_avail_gpus} GPUs per node. world_size={world_size}")
        else:
              # CPU training
             config.devices = 1
             world_size = 1 * config.distributed.num_nodes
             is_distributed = world_size > 1
             rank_zero_info("Using CPU.")

    # Final check on distribution status
    if world_size <= 1:
        is_distributed = False
        if is_torchrun:
             rank_zero_info("torchrun detected but world_size <= 1. Running in non-distributed mode.")


    # Set default sync_batchnorm if distributed
    if is_distributed and config.distributed.sync_batchnorm is None:
         config.distributed.sync_batchnorm = True
         rank_zero_info("Distributed training detected. Enabling sync_batchnorm by default.")
    elif config.distributed.sync_batchnorm is None:
         config.distributed.sync_batchnorm = False # Default to False if not distributed

    # --- Calculate max_steps/max_epochs for Trainer (Simplified) ---
    max_epochs_trainer = config.epochs if config.epochs and config.epochs > 0 else -1
    max_steps_trainer = -1 # Always epoch-based now
    if max_epochs_trainer == -1:
        rank_zero_info("Warning: Trainer max_epochs is not set. Training will rely on dataset length or manual stop.")
    else:
        rank_zero_info(f"Trainer will run for max_epochs = {max_epochs_trainer}")


    # --- Determine Validation Check Interval (Minimalist) ---
    val_check_interval_trainer = None
    check_val_every_n_epoch_trainer = None

    # If validate_every_n_steps was provided (not None), use it and disable epoch checking.
    if config.validate_every_n_steps is not None:
        val_check_interval_trainer = config.validate_every_n_steps
        check_val_every_n_epoch_trainer = None  # Disable epoch check
        if isinstance(val_check_interval_trainer, float):
             rank_zero_info(f"Validation check mode: Every {val_check_interval_trainer:.2f} fraction of epoch.")
        else: # Should be int
             rank_zero_info(f"Validation check mode: Every {val_check_interval_trainer} steps.")
    # Otherwise, use epoch-based checking (defaulting to 1 if config value is None)
    else:
        check_val_every_n_epoch_trainer = config.validate_every_n_epoch if config.validate_every_n_epoch else 1
        val_check_interval_trainer = None  # Disable step check
        rank_zero_info(f"Validation check mode: Every {check_val_every_n_epoch_trainer} epoch(s).")


    # Setup loggers and callbacks
    loggers = setup_loggers(config, experiment_name, args)
    if config.wandb.enabled and config.wandb.name is None: config.wandb.name = experiment_name
    callbacks = setup_callbacks(config, experiment_name, args)

    # Create dataloaders with multi_val parameter
    rank_zero_info(f"Creating dataloaders for dataset type: {config.data.dataset_type}")
    rank_zero_info(f"Multi-validation enabled: {args.multi_val}")
    data_loaders = create_dataloaders(
        config=config.data,
        seed=config.seed,
        distributed=is_distributed,
        multi_val=args.multi_val  # Pass the multi_val parameter
    )
    val_dataloader = data_loaders.get("val")
    if val_dataloader is None and config.monitor_metric and 'val/' in config.monitor_metric:
         rank_zero_info(f"Monitoring validation metric '{config.monitor_metric}' but no validation dataloader found. Checkpointing/EarlyStopping might fail.")

    # Determine train_dataset_size for the Lightning Module scheduler (only for epoch-based NIGHTS)
    # We don't need train_dataset_size for LitData as the scheduler relies on trainer's max_epochs
    train_dataset_size_for_module = 0
    if config.data.dataset_type in ['nod', 'things', 'imagenet1k'] and max_epochs_trainer > 0: # Epoch-based triplet datasets
        try:
            if data_loaders.get("train") and hasattr(data_loaders["train"], 'dataset') and hasattr(data_loaders["train"].dataset, '__len__'):
                train_dataset_size_for_module = len(data_loaders["train"].dataset)
                rank_zero_info(f"{config.data.dataset_type.upper()} training dataset size (for scheduler): {train_dataset_size_for_module}")
            else:
                 rank_zero_info(f"Could not determine length of {config.data.dataset_type.upper()} dataset for epoch-based scheduler. LR schedule might be inaccurate.")
        except TypeError:
            rank_zero_info(f"Could not determine length of {config.data.dataset_type.upper()} dataset (TypeError).")


    # Create model
    rank_zero_info(f"Initializing CLIPLightningModule for training mode: {config.data.dataset_type}")
    rank_zero_info(f"Using model config: {model_config.vision_cfg.layers} layers, {model_config.embed_dim} embed_dim, patch_size {model_config.vision_cfg.patch_size}")
    model = CLIPLightningModule(
        model_config=model_config,
        training_config=config,
        train_dataset_size=train_dataset_size_for_module,
    )

    # --- Custom Weight Loading Logic ---
    if hasattr(config, 'init_ckpt_path') and config.init_ckpt_path:
            rank_zero_info(f"Attempting to load initial vision weights from: {config.init_ckpt_path}")
            load_vision_weights_from_checkpoint(
                target_module=model,
                checkpoint_path=config.init_ckpt_path
            )
            rank_zero_info("Vision weight loading attempt finished (if applicable). Text branch remains as per its initialization.")
    else:
        rank_zero_info("No init_ckpt_path specified in config, or attribute not found. Skipping weight initialization from checkpoint.")

    # --- Freezing Logic for QKV Fine-tuning ---
    if args.config_preset == "nights_ft_from_yfcc_vitb32":
        rank_zero_info("Configuring model for fine-tuning: Freezing all weights except Vision QKV layers.")
        
        # Freeze all parameters in the CLIP model instance first
        for param in model.model.parameters():
            param.requires_grad = False
        rank_zero_info("All parameters in model.model initially frozen.")

        # Unfreeze QKV layers in the vision transformer
        unfrozen_qkv_count = 0
        if hasattr(model.model, 'visual') and hasattr(model.model.visual, 'transformer'):
            for i, block in enumerate(model.model.visual.transformer):
                if hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
                    for name, param in block.attn.qkv.named_parameters():
                        param.requires_grad = True
                        # rank_zero_info(f"  Unfrozen: visual.transformer[{i}].attn.qkv.{name}")
                        unfrozen_qkv_count +=1
        if unfrozen_qkv_count > 0:
             rank_zero_info(f"Unfroze {unfrozen_qkv_count} parameters in Vision QKV layers.")
        else:
             rank_zero_info("Warning: No Vision QKV parameters found or unfrozen. Check model structure.")


        # Log trainable parameters to verify
        rank_zero_info("Trainable parameters after QKV unfreezing:")
        for name, param in model.model.named_parameters():
            if param.requires_grad:
                rank_zero_info(f"  - {name} (shape: {param.shape})")
    
    if args.config_preset == "nod_init_vitb32":
        rank_zero_info("Configuring model for NOD training: Freezing text branch and logit scale.")
        
        # Freeze text branch
        if hasattr(model.model, 'text'):
            for param in model.model.text.parameters():
                param.requires_grad = False
            rank_zero_info("Text branch frozen.")
        
        # Freeze logit scale
        if hasattr(model.model, 'logit_scale'):
            model.model.logit_scale.requires_grad = False
            rank_zero_info("Logit scale frozen.")
        
    # After the existing nights_init_vitb32 freeze logic:
    if args.config_preset =="things_init_vitb32":
        rank_zero_info("Configuring model for THINGS training: Freezing text branch and logit scale.")
        
        # Freeze text branch
        if hasattr(model.model, 'text'):
            for param in model.model.text.parameters():
                param.requires_grad = False
            rank_zero_info("Text branch frozen.")
        
        # Freeze logit scale
        if hasattr(model.model, 'logit_scale'):
            model.model.logit_scale.requires_grad = False
            rank_zero_info("Logit scale frozen.")

    # --- Apply torch.compile ---
    if args.compile:
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version >= (2, 0):
            rank_zero_info("Attempting to compile the model with torch.compile...")
            try:
                model = torch.compile(model, mode="reduce-overhead")
                rank_zero_info("Model compiled successfully.")
            except Exception as e:
                rank_zero_info(f"torch.compile failed: {e}. Proceeding without compilation.")
        else:
            rank_zero_info(f"torch.compile requires PyTorch >= 2.0, but found {torch.__version__}. Skipping compilation.")


    # Configure training strategy for Lightning Trainer
    trainer_strategy = "auto"
    if is_distributed:
        # Use strategy from config if specified
        if config.distributed.strategy and config.distributed.strategy.lower() != "ddp":
             trainer_strategy = config.distributed.strategy
             rank_zero_info(f"Using specified strategy: {trainer_strategy}")
        else:
             trainer_strategy = DDPStrategy(
                find_unused_parameters=config.distributed.find_unused_parameters,
                gradient_as_bucket_view=True # Often beneficial
             )
             rank_zero_info(f"Using DDP strategy (gradient_as_bucket_view=True, find_unused_parameters={config.distributed.find_unused_parameters})")

    # calculate accumulate_grad_batches
    if not config.accumulate_grad_batches:
        accumulate_grad_batches = config.target_batch_size // config.data.batch_size // world_size
        rank_zero_info(f"Calculated accumulate_grad_batches: {accumulate_grad_batches}")
    else:
        accumulate_grad_batches = config.accumulate_grad_batches
        rank_zero_info(f"Using specified accumulate_grad_batches: {accumulate_grad_batches}")
    
    # Create trainer
    rank_zero_info(f"Configuring Trainer with max_epochs={max_epochs_trainer}, max_steps={max_steps_trainer}")
    
    # Test mode configuration
    limit_train_batches = 100 if args.test else 1.0
    if args.test:
        rank_zero_info(f"Test mode: Limiting training to {limit_train_batches} batches per epoch.")
    
    trainer = pl.Trainer(
        max_epochs=max_epochs_trainer,
        max_steps=max_steps_trainer, # Should be -1
        limit_train_batches=limit_train_batches,  # Limit to 5 batches in test mode
        enable_progress_bar=(rank == 0), # Only show progress on rank 0
        accelerator=config.accelerator if not args.test else "cpu",
        devices=config.devices if not args.test else 1,
        num_nodes=config.distributed.num_nodes if not args.test else 1,
        strategy=trainer_strategy,
        precision=config.precision,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=1, #config.log_every_n_steps,
        val_check_interval=val_check_interval_trainer,
        check_val_every_n_epoch=check_val_every_n_epoch_trainer,
        num_sanity_val_steps=-1 if val_dataloader else 0, # Full sanity check if val exists, else 0
        #gradient_clip_val=config.gradient_clip_val, # Not needed for manual
        accumulate_grad_batches=accumulate_grad_batches,
        deterministic=config.deterministic,
        benchmark=not config.deterministic,
        sync_batchnorm=config.distributed.sync_batchnorm,
    )

    # Add hook for setting epoch in distributed samplers/datasets
    # This now needs to handle both DistributedSampler (NIGHTS) and StreamingDataset (LitData)
    class SetEpochHook(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            current_epoch = trainer.current_epoch
            train_loader = trainer.train_dataloader
            if train_loader is None: return

            # Handle standard DataLoader with DistributedSampler
            sampler = getattr(train_loader, 'sampler', None)
            # Handle nested sampler if DataLoader wraps another
            if hasattr(sampler, 'sampler') and isinstance(sampler.sampler, torch.utils.data.DistributedSampler):
                 sampler.sampler.set_epoch(current_epoch)
            elif isinstance(sampler, torch.utils.data.DistributedSampler):
                 sampler.set_epoch(current_epoch)

            # Handle LitData StreamingDataLoader/StreamingDataset
            dataset = getattr(train_loader, 'dataset', None)
            if hasattr(dataset, 'set_epoch') and callable(dataset.set_epoch):
                 try:
                     dataset.set_epoch(current_epoch)
                     # rank_zero_info(f"Set epoch {current_epoch} on LitData dataset.")
                 except Exception as e:
                      rank_zero_info(f"Warning: Failed to set epoch on LitData dataset: {e}")


    trainer.callbacks.append(SetEpochHook())


    # --- Train ---
    rank_zero_info("Starting training...")
    # Check if validation is configured but no validation dataloader exists
    if val_dataloader is None and (val_check_interval_trainer is not None or check_val_every_n_epoch_trainer is not None):
        rank_zero_info("Validation is configured to run, but no validation dataloader was created. Validation steps will be skipped.")

    trainer.fit(
        model=model,
        train_dataloaders=data_loaders["train"],
        val_dataloaders=val_dataloader # Pass potentially None val_dataloader
    )

    # --- Test ---
    rank_zero_info("Starting testing...")
    test_dataloader = data_loaders.get("test")
    if test_dataloader:
        # Determine best checkpoint path
        best_ckpt_path = None
        if config.monitor_metric and hasattr(trainer.checkpoint_callback, 'best_model_path') and trainer.checkpoint_callback.best_model_path:
            best_ckpt_path = trainer.checkpoint_callback.best_model_path
            rank_zero_info(f"Testing using best checkpoint: {best_ckpt_path}")
        elif hasattr(trainer.checkpoint_callback, 'last_model_path') and trainer.checkpoint_callback.last_model_path:
            best_ckpt_path = trainer.checkpoint_callback.last_model_path # Fallback to last
            rank_zero_info(f"Testing using last checkpoint (monitor metric '{config.monitor_metric}' not found or no best path): {best_ckpt_path}")
        else:
            rank_zero_info("Could not determine best/last checkpoint path. Skipping testing.")


        if best_ckpt_path:
            trainer.test(
                # model=model, # Trainer loads the checkpoint
                dataloaders=test_dataloader,
                ckpt_path=best_ckpt_path # Use the determined path
            )
    else:
        rank_zero_info("No test dataloader found. Skipping testing.")


    # --- Clean up ---
    rank_zero_info("Training finished.")
    # PTL handles process group cleanup

    return 0


if __name__ == "__main__":
    main() 
