from config.model_config import (
    CLIPConfig,
    CLIPVisionConfig,
    CLIPTextConfig
)
from config.training_config import (
    TrainingConfig,
    DataConfig,
    OptimizerGroupConfig,
    DualOptimizerConfig,
    SchedulerConfig,
    ContrastiveLossConfig,
    WandbConfig,
    DistributedConfig,
    YFCC15MLitDataConfig,
    DeCLIPSchedulerConfig,
    CLIPLossConfig,
    DeCLIPLitDataTrainingConfig,
    NodTrainingConfig,
    ThingsTrainingConfig
)
from config.config import (
    # New JSON-based configuration functions
    get_config,
    load_model_config,
    load_training_config,
    list_available_configs,
    
    # Backward compatibility - individual model configs
    get_vit_b_32_config,
    get_vit_b_16_config,
    get_vit_l_14_config,
    get_vit_h_14_config,
    
    # Unified naming conventions - NOD initialization (Step 1)
    get_nod_init_vitb32_config,
    get_nod_init_vitb16_config,
    get_nod_init_vitl14_config,
    get_nod_init_vith14_config,
    get_nod_init_resnet50_config,

    # Unified naming conventions - THINGS initialization (Step 1)
    get_things_init_vitb32_config,
    # Unified naming conventions - ImageNet-1k initialization (Ablation study)
    # get_imagenet1k_init_vitb32_config,
    # Unified naming conventions - Baseline YFCC15M pretraining (Step 3 baseline)
    get_baseline_yfcc15m_vitb32_config,
    get_baseline_yfcc15m_vitb16_config,
    get_baseline_yfcc15m_vitl14_config,
    get_baseline_yfcc15m_vith14_config,
    get_baseline_yfcc15m_resnet50_config,
    
    # Unified naming conventions - Perceptual-initialized YFCC15M pretraining (Step 2)
    get_perceptual_init_yfcc15m_vitb32_config,
    get_perceptual_init_yfcc15m_vitb16_config,
    get_perceptual_init_yfcc15m_vitl14_config,
    get_perceptual_init_yfcc15m_vith14_config,
    get_perceptual_init_yfcc15m_resnet50_config,
    
    # Unified naming conventions - Fine-tuning (Step 3 alternative)
    get_nights_ft_from_yfcc_vitb32_config,
    
    # Unified configuration instances with new naming
    VIT_B_32_CONFIG,
    RESNET_50_CONFIG,
    NOD_INIT_VITB32_CONFIG,
    NOD_INIT_VITB16_CONFIG,
    NOD_INIT_VITL14_CONFIG,
    NOD_INIT_VITH14_CONFIG,
    NOD_INIT_RESNET50_CONFIG,
    THINGS_INIT_VITB32_CONFIG,
    # IMAGENET1K_INIT_VITB32_CONFIG,

    BASELINE_YFCC15M_VITB32_CONFIG,
    BASELINE_YFCC15M_VITB16_CONFIG,
    BASELINE_YFCC15M_VITL14_CONFIG,
    BASELINE_YFCC15M_VITH14_CONFIG,
    BASELINE_YFCC15M_RESNET50_CONFIG,
    
    PERCEPTUAL_INIT_YFCC15M_VITB32_CONFIG,
    PERCEPTUAL_INIT_YFCC15M_VITB16_CONFIG,
    PERCEPTUAL_INIT_YFCC15M_VITL14_CONFIG,
    PERCEPTUAL_INIT_YFCC15M_VITH14_CONFIG,
    PERCEPTUAL_INIT_YFCC15M_RESNET50_CONFIG,
    
    NIGHTS_FT_FROM_YFCC_VITB32_CONFIG,
    
    # Legacy naming for backward compatibility (deprecated - use new naming above)
    NOD_INIT_CONFIG,
    DECLIP_YFCC15M_LITDATA_VITB32_CONFIG,
    DECLIP_YFCC15M_LITDATA_VITH14_CONFIG,
    NIGHTS_INIT_DECLIP_YFCC15M_LITDATA_VITB32_CONFIG,
    NIGHTS_FT_FROM_YFCC_CONFIG,
)


__all__ = [
    # Pydantic Models from model_config.py
    "CLIPConfig",
    "CLIPVisionConfig",
    "CLIPTextConfig",

    # Pydantic Models from training_config.py
    "TrainingConfig",
    "DataConfig",
    "OptimizerGroupConfig",
    "DualOptimizerConfig",
    "SchedulerConfig",
    "ContrastiveLossConfig",
    "WandbConfig",
    "DistributedConfig",
    "YFCC15MLitDataConfig",
    "DeCLIPSchedulerConfig",
    "CLIPLossConfig",
    "DeCLIPLitDataTrainingConfig",
    "NodTrainingConfig",
    "ThingsTrainingConfig",

    # New JSON-based configuration functions
    "get_config",
    "load_model_config",
    "load_training_config",
    "list_available_configs",
    
    # Individual model config functions
    "get_vit_b_32_config",
    "get_vit_b_16_config",
    "get_vit_l_14_config",
    "get_vit_h_14_config",
    "get_resnet_50_config",
    
    # Unified naming conventions - NOD initialization (Step 1)
    "get_nod_init_vitb32_config",
    "get_nod_init_vitb16_config",
    "get_nod_init_vitl14_config",
    "get_nod_init_vith14_config",
    "get_nod_init_resnet50_config",

    # Unified naming conventions - THINGS initialization (Step 1)
    "get_things_init_vitb32_config",
    # Unified naming conventions - ImageNet-1k initialization (Ablation study)
    # "get_imagenet1k_init_vitb32_config",

    # Unified naming conventions - Baseline YFCC15M pretraining (Step 3 baseline)
    "get_baseline_yfcc15m_vitb32_config",
    "get_baseline_yfcc15m_vitb16_config",
    "get_baseline_yfcc15m_vitl14_config",
    "get_baseline_yfcc15m_vith14_config",
    "get_baseline_yfcc15m_resnet50_config",
    
    # Unified naming conventions - Perceptual-initialized YFCC15M pretraining (Step 2)
    "get_perceptual_init_yfcc15m_vitb32_config",
    "get_perceptual_init_yfcc15m_vitb16_config",
    "get_perceptual_init_yfcc15m_vitl14_config",
    "get_perceptual_init_yfcc15m_vith14_config",
    "get_perceptual_init_yfcc15m_resnet50_config",
    
    # Unified naming conventions - Fine-tuning (Step 3 alternative)
    "get_nights_ft_from_yfcc_vitb32_config",

    # Unified configuration instances with new naming
    "VIT_B_32_CONFIG",
    "RESNET_50_CONFIG",
    "NOD_INIT_VITB32_CONFIG",
    "NOD_INIT_VITB16_CONFIG",
    "NOD_INIT_VITL14_CONFIG",
    "NOD_INIT_VITH14_CONFIG",
    "NOD_INIT_RESNET50_CONFIG",
    "THINGS_INIT_VITB32_CONFIG",
    # "IMAGENET1K_INIT_VITB32_CONFIG",
    
    "BASELINE_YFCC15M_VITB32_CONFIG",
    "BASELINE_YFCC15M_VITB16_CONFIG",
    "BASELINE_YFCC15M_VITL14_CONFIG",
    "BASELINE_YFCC15M_VITH14_CONFIG",
    "BASELINE_YFCC15M_RESNET50_CONFIG",
    
    "PERCEPTUAL_INIT_YFCC15M_VITB32_CONFIG",
    "PERCEPTUAL_INIT_YFCC15M_VITB16_CONFIG",
    "PERCEPTUAL_INIT_YFCC15M_VITL14_CONFIG",
    "PERCEPTUAL_INIT_YFCC15M_VITH14_CONFIG",
    "PERCEPTUAL_INIT_YFCC15M_RESNET50_CONFIG",
    
    "NIGHTS_FT_FROM_YFCC_VITB32_CONFIG",
    
    # Legacy naming for backward compatibility (deprecated - use new naming above)
    "NOD_INIT_CONFIG",
    "DECLIP_YFCC15M_LITDATA_VITB32_CONFIG",
    "DECLIP_YFCC15M_LITDATA_VITH14_CONFIG",
    "NIGHTS_INIT_DECLIP_YFCC15M_LITDATA_VITB32_CONFIG",
    "NIGHTS_FT_FROM_YFCC_CONFIG",
]