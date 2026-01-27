# Brain Initialization and Pre-training with CLIP

This project implements the Brain Initialization (BI) paradigm for training CLIP models on ViT-B/32. BI adapts the Perceptual-Initialization (PI) training scaffold to use brain-derived representational structure as the step-0 prior for the vision encoder. PI is introduced in the paper "Beginning with You: Perceptual-Initialization Improves Vision-Language Representation and Alignment" on ArXiV.

The core idea is to initialize the vision encoder using similarity constraints derived from the Natural Object Dataset (NOD) fMRI representational geometry. After this initialization stage, we run standard large-scale self-supervised image–text contrastive pretraining on YFCC15M. We compare against a matched baseline trained from random initialization.

## Features

-   **Brain Initialization:** Uses NOD fMRI representational structure (RDM-derived similarity orderings) to seed the vision encoder before web-scale pretraining.
-   **Multiple ViT Architectures:** Supports ViT-B/32, ViT-B/16, ViT-L/14, and ViT-H/14 model variants for comprehensive experimentation.
-   **YFCC15M Pretraining:** Supports large-scale contrastive pretraining on the YFCC15M dataset across all model variants.
-   **JSON-based Configuration System:** Employs a flexible JSON-based configuration system with backward compatibility for easy experiment management.
-   **Distributed Training:** Supports Distributed Data Parallel (DDP) training using `torchrun`, facilitated by the `launch_ddp.sh` script.
-   **Comprehensive 3-Step Experiment:** Includes configurations for the complete experimental pipeline across all model variants.

## Configuration System

The project uses a flexible JSON-based configuration system located in the `config/` directory:

### Model Configurations (`config/modelconfig/`)
- `vit_b_32.json` - ViT-B/32 model configuration

### Training Configurations (`config/trainingconfig/`)

**Step 1: NOD Initialization**
- `nod_init_vitb32.json` - ViT-B/32 NIGHTS initialization

**Step 2: Perceptual-Initialized YFCC15M Pretraining**
- `perceptual_init_yfcc15m_vitb32.json` - ViT-B/32 perceptual-initialized pretraining

**Step 3: Baseline YFCC15M Pretraining**
- `declip_yfcc15m_litdata_vitb32.json` - ViT-B/32 baseline pretraining

## Setup and Installation

### Requirements

-   Python 3.7+
-   PyTorch 2.0+
-   CUDA-capable GPU (recommended for training)

### Setting up Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

This project uses two main datasets:

1.  **NIGHTS Dataset:** Used for brain initialization (Stage 1).
    *   You will need: 1) the NOD stimulus images used to instantiate image triplets, and 2) the fMRI-derived representational structure (e.g., RDMs) used to define similarity constraints
    *   Configure the path in the training configurations under `data.dataset_root`.
2.  **YFCC15M Dataset:** Used for large-scale contrastive pretraining (For Stage 2 of PI, and Baseline).
    *   The project can use a pre-downloaded version (e.g., Parquet files) or stream from Hugging Face Hub.
    *   To prefetch (recommended for multiple runs), you can use the provided script:
        ```bash
        python download.py
        ```
        This script will output the storage location. Configure this path in the training configurations under `data.parquet_data_dir`.
    *   For litData streaming, ensure `data.hf_dataset_name` is correctly set (e.g., `'hf://datasets/Kaichengalex/YFCC15M/data'`).

## Training

All training routines can be launched using the `launch_ddp.sh` script, which handles distributed training setup with `torchrun`.

### General Usage of `launch_ddp.sh`

```bash
# Make the launch script executable
chmod +x launch_ddp.sh

# Run with a specific preset
./launch_ddp.sh --config_preset="<your_chosen_preset>" 
```

-   `NUM_GPUS_PER_NODE` in `launch_ddp.sh` can be modified to change the number of GPUs used.
-   Any arguments passed to `launch_ddp.sh` after the script name (like `--batch_size` above) are forwarded to the underlying `train_clip.py` script, overriding defaults set in the JSON configurations.

### 3-Step Experimental Workflows

The following presets correspond to the complete experimental pipeline across all model variants:

#### Step 1: NIGHTS Initialization (Perceptual Initialization)

Train the vision encoder on the NIGHTS dataset using triplet contrastive loss:

```bash
# ViT-B/32
./launch_ddp.sh --config_preset="nights"
```

Note the path to the saved checkpoint for use in Step 2.

#### Step 2: Perceptual-Initialized YFCC15M Pretraining

Train the full CLIP model with the perceptually initialized vision encoder:

```bash
# ViT-B/32
./launch_ddp.sh --config_preset="perceptual_init_yfcc15m_vitb32"
```

The configurations automatically load the appropriate NIGHTS checkpoint via `init_ckpt_path`.

#### Step 3: Baseline YFCC15M Pretraining (for comparison)

Train CLIP models from scratch on YFCC15M:

```bash
# ViT-B/32
./launch_ddp.sh --config_preset="yfcc15m_litdata"
```

### Configuration Customization

You can override any configuration parameter via command line:

```bash
# Override batch size and learning rate
./launch_ddp.sh --config_preset="nod" --batch_size=256 --lr=0.001

# Override checkpoint path for perceptual initialization
./launch_ddp.sh --config_preset="perceptual_init_yfcc15m_vitl14" \
                --init_ckpt_path="/custom/path/to/nod_checkpoint.ckpt"
```

## Development

To contribute to this project, please make sure to:
1. Set up the virtual environment as described above.
2. Install the package in development mode: `pip install -e .`
3. Follow the existing code style and patterns.
4. When adding new configurations, place model configs in `config/modelconfig/` and training configs in `config/trainingconfig/`.

## License

This project is licensed under the MIT License.

## Acknowledgements

This work builds upon and utilizes several key resources and prior research:
-   [OpenCLIP](https://github.com/mlfoundations/open_clip) for the CLIP model implementations.
-   [DreamSim](https://github.com/dreamsim-nights/dreamsim) for the NIGHTS dataset and triplet training insights.
-   [YFCC15M Dataset](https://huggingface.co/datasets/Kaichengalex/YFCC15M) and related processing for training data.
-   [DeCLIP](https://github.com/Sense-GVT/DeCLIP) for training recipe insights.
-   [DataComp](https://github.com/mlfoundations/datacomp) for evaluation dataset and recipe insights.
-   [PyTorch Lightning](https://www.pytorchlightning.ai/) for the training framework.
-   [litData](https://github.com/Lightning-AI/litData) for the data ingestion framework.
