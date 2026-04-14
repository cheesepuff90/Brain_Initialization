import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union

try:
    # When imported as a module
    from .model_config import (
        CLIPConfig,
        CLIPVisionConfig,
        CLIPTextConfig
    )
    from .training_config import (
        DataConfig,
        OptimizerGroupConfig,
        DualOptimizerConfig,
        SchedulerConfig,
        ContrastiveLossConfig,
        WandbConfig,
        DistributedConfig,
        TrainingConfig,
        YFCC15MLitDataConfig,
        DeCLIPSchedulerConfig,
        CLIPLossConfig,
        DeCLIPLitDataTrainingConfig,
        NodTrainingConfig,
        ThingsTrainingConfig,
        # ImageNet1kTrainingConfig
    )
except ImportError:
    # When run directly
    from model_config import (
    CLIPConfig,
    CLIPVisionConfig,
    CLIPTextConfig
)
    from training_config import (
    DataConfig,
    OptimizerGroupConfig,
    DualOptimizerConfig,
    SchedulerConfig,
    ContrastiveLossConfig,
    WandbConfig,
    DistributedConfig,
    TrainingConfig,
    YFCC15MLitDataConfig,
    DeCLIPSchedulerConfig,
    CLIPLossConfig,
    DeCLIPLitDataTrainingConfig,
    NodTrainingConfig,
    ThingsTrainingConfig,
    # ImageNet1kTrainingConfig
)

# Get the directory where this config.py file is located
CONFIG_DIR = Path(__file__).parent

# Paths to configuration directories
MODEL_CONFIG_DIR = CONFIG_DIR / "modelconfig"
TRAINING_CONFIG_DIR = CONFIG_DIR / "trainingconfig"


def load_json_config(file_path: Path) -> Dict[str, Any]:
    """Load a JSON configuration file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)


def load_model_config(model_name: str) -> CLIPConfig:
    """Load a model configuration from JSON and convert to CLIPConfig."""
    config_file = MODEL_CONFIG_DIR / f"{model_name}.json"
    config_data = load_json_config(config_file)
    
    # Convert JSON data to Pydantic models
    vision_cfg = CLIPVisionConfig(**config_data["vision_cfg"])
    text_cfg = CLIPTextConfig(**config_data["text_cfg"])
    
    # Remove nested configs from main config data
    config_data_copy = config_data.copy()
    config_data_copy.pop("vision_cfg")
    config_data_copy.pop("text_cfg")
    config_data_copy.pop("name", None)  # Remove name field if present
    
    return CLIPConfig(
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        **config_data_copy
    )


def load_training_config(config_name: str) -> TrainingConfig:
    """Load a training configuration from JSON and convert to appropriate TrainingConfig subclass."""
    config_file = TRAINING_CONFIG_DIR / f"{config_name}.json"
    config_data = load_json_config(config_file)
    
    # Determine the appropriate config class based on the config name or type
    config_type = config_data.get("name", config_name)
    
    # Convert nested dictionaries to appropriate Pydantic models
    data_config = None
    if config_data.get("data"):
        data_dict = config_data["data"]
        if data_dict.get("dataset_type") == "yfcc15m_litdata":
            data_config = YFCC15MLitDataConfig(**data_dict)
        else:
            data_config = DataConfig(**data_dict)
    
    optimizer_config = None
    if config_data.get("optimizer"):
        opt_dict = config_data["optimizer"]
        # Check if it's a dual optimizer config (has vision/text keys)
        if "vision" in opt_dict and "text" in opt_dict:
            optimizer_config = DualOptimizerConfig(
                vision=OptimizerGroupConfig(**opt_dict["vision"]),
                text=OptimizerGroupConfig(**opt_dict["text"])
            )
        else:
            optimizer_config = OptimizerGroupConfig(**opt_dict)
    
    scheduler_config = None
    if config_data.get("scheduler"):
        sched_dict = config_data["scheduler"]
        if "warmup_lr_vision" in sched_dict or "warmup_lr_text" in sched_dict:
            scheduler_config = DeCLIPSchedulerConfig(**sched_dict)
        else:
            scheduler_config = SchedulerConfig(**sched_dict)
    
    loss_config = None
    if config_data.get("loss"):
        loss_dict = config_data["loss"]
        if loss_dict.get("margin") is not None:
            loss_config = ContrastiveLossConfig(**loss_dict)
        else:
            loss_config = CLIPLossConfig(**loss_dict)
    
    distributed_config = None
    if config_data.get("distributed"):
        distributed_config = DistributedConfig(**config_data["distributed"])
    
    wandb_config = None
    if config_data.get("wandb"):
        wandb_config = WandbConfig(**config_data["wandb"])
    
    # Remove nested configs from main config data
    config_data_copy = config_data.copy()
    for key in ["data", "optimizer", "scheduler", "loss", "distributed", "wandb", "name", "model_config"]:
        config_data_copy.pop(key, None)
    
    # Determine which TrainingConfig subclass to use
    if "nod" in config_type.lower():
        return NodTrainingConfig(
            data=data_config,
            optimizer=optimizer_config,
            scheduler=scheduler_config,
            loss=loss_config,
            distributed=distributed_config,
            wandb=wandb_config,
            **config_data_copy
        )
    elif "things" in config_type.lower():
        return ThingsTrainingConfig(
            data=data_config,
            optimizer=optimizer_config,
            scheduler=scheduler_config,
            loss=loss_config,
            distributed=distributed_config,
            wandb=wandb_config,
            **config_data_copy
        )
    elif "declip" in config_type.lower() or "yfcc" in config_type.lower():
        return DeCLIPLitDataTrainingConfig(
            data=data_config,
            optimizer=optimizer_config,
            scheduler=scheduler_config,
            loss=loss_config,
            distributed=distributed_config,
            wandb=wandb_config,
            **config_data_copy
        )
    else:
        return TrainingConfig(
            data=data_config,
            optimizer=optimizer_config,
            scheduler=scheduler_config,
            loss=loss_config,
            distributed=distributed_config,
            wandb=wandb_config,
            **config_data_copy
        )


def get_config(training_config_name: str, model_config_name: Optional[str] = None) -> tuple[TrainingConfig, CLIPConfig]:
    """
    Load both training and model configurations.
    
    Args:
        training_config_name: Name of the training config JSON file (without .json extension)
        model_config_name: Name of the model config JSON file (without .json extension).
                          If None, will try to extract from training config.
    
    Returns:
        Tuple of (TrainingConfig, CLIPConfig)
    """
    # Load training config
    training_config = load_training_config(training_config_name)
    
    # Determine model config name
    if model_config_name is None:
        # Try to get model config from training config JSON
        config_file = TRAINING_CONFIG_DIR / f"{training_config_name}.json"
        config_data = load_json_config(config_file)
        model_config_name = config_data.get("model_config")
        
        if model_config_name is None:
            raise ValueError(f"No model_config specified in training config {training_config_name} and no model_config_name provided")
    
    # Load model config
    model_config = load_model_config(model_config_name)
    
    return training_config, model_config


def list_available_configs() -> Dict[str, list]:
    """List all available configuration files."""
    model_configs = []
    training_configs = []
    
    if MODEL_CONFIG_DIR.exists():
        model_configs = [f.stem for f in MODEL_CONFIG_DIR.glob("*.json")]
    
    if TRAINING_CONFIG_DIR.exists():
        training_configs = [f.stem for f in TRAINING_CONFIG_DIR.glob("*.json")]
    
    return {
        "model_configs": sorted(model_configs),
        "training_configs": sorted(training_configs)
    }


# --- Backward Compatibility: Pre-defined configs ---
# These maintain the same names as before for easy migration

def get_vit_b_32_config() -> CLIPConfig:
    """Get ViT-B-32 model configuration."""
    return load_model_config("vit_b_32")

def get_vit_b_16_config() -> CLIPConfig:
    """Get ViT-B-16 model configuration."""
    return load_model_config("vit_b_16")

def get_vit_l_14_config() -> CLIPConfig:
    """Get ViT-L-14 model configuration."""
    return load_model_config("vit_l_14")

def get_vit_h_14_config() -> CLIPConfig:
    """Get ViT-H-14 model configuration."""
    return load_model_config("vit_h_14")

def get_resnet_50_config() -> CLIPConfig:
    """Get ResNet-50 model configuration."""
    return load_model_config("resnet_50")

# Backward compatibility for training configs
VIT_B_32_CONFIG = get_vit_b_32_config()
RESNET_50_CONFIG = get_resnet_50_config()

# NIGHTS initialization configurations (Step 1)
def get_nod_init_vitb32_config() -> tuple[NodTrainingConfig, CLIPConfig]:
    """Get NOD init ViT-B/32 configuration."""
    training_config, model_config = get_config("nod_init_vitb32")
    return training_config, model_config

def get_nod_init_vitb16_config() -> tuple[NodTrainingConfig, CLIPConfig]:
    """Get NOD init ViT-B/16 configuration."""
    training_config, model_config = get_config("nod_init_vitb16")
    return training_config, model_config

def get_nod_init_vitl14_config() -> tuple[NodTrainingConfig, CLIPConfig]:
    """Get NOD init ViT-L/14 configuration."""
    training_config, model_config = get_config("nod_init_vitl14")
    return training_config, model_config

def get_nod_init_vith14_config() -> tuple[NodTrainingConfig, CLIPConfig]:
    """Get NOD init ViT-H/14 configuration."""
    training_config, model_config = get_config("nod_init_vith14")
    return training_config, model_config

# THINGS initialization configurations (Alternative to NIGHTS)
def get_things_init_vitb32_config() -> tuple[ThingsTrainingConfig, CLIPConfig]:
    """Get THINGS init ViT-B/32 configuration."""
    training_config, model_config = get_config("things_init_vitb32")
    return training_config, model_config

# # ImageNet-1k initialization configurations (Ablation study)
# def get_imagenet1k_init_vitb32_config() -> tuple[TrainingConfig, CLIPConfig]:
#     """Get ImageNet-1k init ViT-B/32 configuration."""
#     training_config, model_config = get_config("imagenet1k_init_vitb32")
#     return training_config, model_config

# Baseline YFCC15M pretraining configurations (Step 3 baseline)
def get_baseline_yfcc15m_vitb32_config() -> tuple[DeCLIPLitDataTrainingConfig, CLIPConfig]:
    """Get baseline YFCC15M ViT-B/32 configuration."""
    training_config, model_config = get_config("declip_yfcc15m_litdata_vitb32")
    return training_config, model_config

def get_baseline_yfcc15m_vitb16_config() -> tuple[DeCLIPLitDataTrainingConfig, CLIPConfig]:
    """Get baseline YFCC15M ViT-B/16 configuration."""
    training_config, model_config = get_config("declip_yfcc15m_litdata_vitb16")
    return training_config, model_config

def get_baseline_yfcc15m_vitl14_config() -> tuple[DeCLIPLitDataTrainingConfig, CLIPConfig]:
    """Get baseline YFCC15M ViT-L/14 configuration."""
    training_config, model_config = get_config("declip_yfcc15m_litdata_vitl14")
    return training_config, model_config

def get_baseline_yfcc15m_vith14_config() -> tuple[DeCLIPLitDataTrainingConfig, CLIPConfig]:
    """Get baseline YFCC15M ViT-H/14 configuration."""
    training_config, model_config = get_config("declip_yfcc15m_litdata_vith14")
    return training_config, model_config

# brain-initialized YFCC15M pretraining configurations (Step 2)
def get_brain_init_yfcc15m_vitb32_config() -> tuple[DeCLIPLitDataTrainingConfig, CLIPConfig]:
    """Get brain-initialized YFCC15M ViT-B/32 configuration."""
    training_config, model_config = get_config("brain_init_yfcc15m_vitb32")
    return training_config, model_config

def get_brain_init_yfcc15m_vitb16_config() -> tuple[DeCLIPLitDataTrainingConfig, CLIPConfig]:
    """Get brain-initialized YFCC15M ViT-B/16 configuration."""
    training_config, model_config = get_config("brain_init_yfcc15m_vitb16")
    return training_config, model_config

def get_brain_init_yfcc15m_vitl14_config() -> tuple[DeCLIPLitDataTrainingConfig, CLIPConfig]:
    """Get brain-initialized YFCC15M ViT-L/14 configuration."""
    training_config, model_config = get_config("brain_init_yfcc15m_vitl14")
    return training_config, model_config

def get_brain_init_yfcc15m_vith14_config() -> tuple[DeCLIPLitDataTrainingConfig, CLIPConfig]:
    """Get brain-initialized YFCC15M ViT-H/14 configuration."""
    training_config, model_config = get_config("brain_init_yfcc15m_vith14")
    return training_config, model_config

# Fine-tuning configurations (Step 3 alternative)
def get_nights_ft_from_yfcc_vitb32_config() -> tuple[NodTrainingConfig, CLIPConfig]:
    """Get NIGHTS fine-tuning from YFCC ViT-B/32 configuration."""
    training_config, model_config = get_config("nights_ft_from_yfcc")
    return training_config, model_config

# ResNet-50 experiment configurations
def get_nod_init_resnet50_config() -> tuple[NodTrainingConfig, CLIPConfig]:
    """Get NOD init ResNet-50 configuration."""
    training_config, model_config = get_config("nod_init_resnet50")
    return training_config, model_config

def get_baseline_yfcc15m_resnet50_config() -> tuple[DeCLIPLitDataTrainingConfig, CLIPConfig]:
    """Get baseline YFCC15M ResNet-50 configuration."""
    training_config, model_config = get_config("declip_yfcc15m_litdata_resnet50")
    return training_config, model_config

def get_brain_init_yfcc15m_resnet50_config() -> tuple[DeCLIPLitDataTrainingConfig, CLIPConfig]:
    """Get brain-initialized YFCC15M ResNet-50 configuration."""
    training_config, model_config = get_config("brain_init_yfcc15m_resnet50")
    return training_config, model_config

# Unified configuration instances with new naming
NOD_INIT_VITB32_CONFIG, _ = get_nod_init_vitb32_config()
NOD_INIT_VITB16_CONFIG, _ = get_nod_init_vitb16_config()
NOD_INIT_VITL14_CONFIG, _ = get_nod_init_vitl14_config()
NOD_INIT_VITH14_CONFIG, _ = get_nod_init_vith14_config()
THINGS_INIT_VITB32_CONFIG, _ = get_things_init_vitb32_config()

BASELINE_YFCC15M_VITB32_CONFIG, _ = get_baseline_yfcc15m_vitb32_config()
BASELINE_YFCC15M_VITB16_CONFIG, _ = get_baseline_yfcc15m_vitb16_config()
BASELINE_YFCC15M_VITL14_CONFIG, _ = get_baseline_yfcc15m_vitl14_config()
BASELINE_YFCC15M_VITH14_CONFIG, _ = get_baseline_yfcc15m_vith14_config()

BRAIN_INIT_YFCC15M_VITB32_CONFIG, _ = get_brain_init_yfcc15m_vitb32_config()
BRAIN_INIT_YFCC15M_VITB16_CONFIG, _ = get_brain_init_yfcc15m_vitb16_config()
BRAIN_INIT_YFCC15M_VITL14_CONFIG, _ = get_brain_init_yfcc15m_vitl14_config()
BRAIN_INIT_YFCC15M_VITH14_CONFIG, _ = get_brain_init_yfcc15m_vith14_config()

NIGHTS_FT_FROM_YFCC_VITB32_CONFIG, _ = get_nights_ft_from_yfcc_vitb32_config()
# IMAGENET1K_INIT_VITB32_CONFIG, _ = get_imagenet1k_init_vitb32_config()

# ResNet-50 experiment configurations
NOD_INIT_RESNET50_CONFIG, _ = get_nod_init_resnet50_config()
BASELINE_YFCC15M_RESNET50_CONFIG, _ = get_baseline_yfcc15m_resnet50_config()
BRAIN_INIT_YFCC15M_RESNET50_CONFIG, _ = get_brain_init_yfcc15m_resnet50_config()

# Legacy naming for backward compatibility (deprecated - use new naming above)
NOD_INIT_CONFIG = NOD_INIT_VITB32_CONFIG  # Default to B/32
DECLIP_YFCC15M_LITDATA_VITB32_CONFIG = BASELINE_YFCC15M_VITB32_CONFIG
DECLIP_YFCC15M_LITDATA_VITH14_CONFIG = BASELINE_YFCC15M_VITH14_CONFIG
NIGHTS_INIT_DECLIP_YFCC15M_LITDATA_VITB32_CONFIG = BRAIN_INIT_YFCC15M_VITB32_CONFIG
NIGHTS_FT_FROM_YFCC_CONFIG = NIGHTS_FT_FROM_YFCC_VITB32_CONFIG


if __name__ == "__main__":
    # Example usage and testing
    print("Available configurations:")
    configs = list_available_configs()
    print(f"Model configs: {configs['model_configs']}")
    print(f"Training configs: {configs['training_configs']}")
    
    # Test loading a configuration
    try:
        training_config, model_config = get_config("declip_yfcc15m_litdata_vitb32")
        print(f"\nSuccessfully loaded training config: {training_config.wandb.tags}")
        print(f"Successfully loaded model config: {model_config.vision_cfg.patch_size}")
    except Exception as e:
        print(f"Error loading config: {e}")
