import os
import sys
import torch
import numpy as np
from PIL import Image, TiffImagePlugin, ExifTags
from torchvision import transforms
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings
import io

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torchmetrics import Accuracy

# NEW imports for LitData
import litdata as ld
from litdata import StreamingDataset, StreamingDataLoader

# Suppress specific PIL warnings more aggressively
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
warnings.filterwarnings("ignore", category=UserWarning, message=".*EXIF data.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Corrupt EXIF.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Truncated File Read.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Metadata Warning.*")

# Monkey patch PIL's getexif method to prevent UnicodeDecodeError
original_getexif = Image.Image.getexif

def safe_getexif(self):
    try:
        return original_getexif(self)
    except Exception:
        # Return an empty EXIF
        return Image.Exif()

# Apply the monkey patch
Image.Image.getexif = safe_getexif

# Adjust Python path to import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Hugging Face Token and Cache Setup
token_path = os.path.expanduser("~/.cache/huggingface/token")
if os.path.exists(token_path):
    with open(token_path) as f:
        hf_token = f.read().strip()
else:
    hf_token = os.environ.get("HF_HUB_TOKEN")  # Fallback to environment variable
    if not hf_token:
        print("Hugging Face token not found. Please set it in ~/.cache/huggingface/token or as HF_HUB_TOKEN env var.")

os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface/models"
os.environ["HF_HUB_CACHE"] = "/tmp/huggingface/hub"
os.environ["HF_HUB_TOKEN"] = hf_token if hf_token else ""

os.makedirs(os.environ['HF_HOME'], exist_ok=True)
os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)
os.makedirs(os.environ['HF_HUB_CACHE'], exist_ok=True)

try:
    from src.lightning_module import CLIPLightningModule
    from config import VIT_B_32_CONFIG  # Assuming this config is for the base CLIP model
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure 'src' and 'config.py' are in the project root and accessible.")
    sys.exit(1)

# --- Configuration ---
CHECKPOINT_PATHS = [
    #"/tmp/checkpoints/pretrain_checkpoints/nights_ft_from_yfcc_20250512_231704_ddp_nodeNone/checkpoints/last.ckpt",
    "/dev/shm/checkpoints/yfcc15m_litdata_20250506_013343_ddp_nodeNone/checkpoints/monitor_val_cifar100_epoch_acc1-epoch_epoch=031.ckpt",
    "/dev/shm/checkpoints/nights_init_yfcc15m_litdata_20250507_202423_ddp_nodeNone/checkpoints/monitor_val_cifar100_epoch_acc1-epoch_epoch=031.ckpt"
]

# URIs for streaming ImageNet from Hugging Face with LitData
HF_URI_TRAIN = "hf://datasets/benjamin-paine/imagenet-1k/train"
HF_URI_VAL = "hf://datasets/benjamin-paine/imagenet-1k/validation"

NUM_CLASSES = 1000
IMAGE_KEY = "image"
LABEL_KEY = "label"

# CLIP Preprocessing
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

BATCH_SIZE_PER_GPU = 4096  # Reduced batch size to avoid memory issues
NUM_GPUS_TO_USE = 6
NUM_WORKERS = 8  # Reduced worker count to avoid issues

# Training parameters for the linear probe
EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01

# --- Helper Functions ---
def get_image_transform(image_size):
    """Returns the image transformation pipeline for CLIP."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        lambda image: image.convert("RGB") if image.mode != "RGB" else image,
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_IMAGE_MEAN, std=CLIP_IMAGE_STD),
    ])

def load_clip_model(ckpt_path, model_config):
    """Loads the OpenCLIP model from a checkpoint, typically to CPU first for Lightning."""
    print(f"Loading CLIP model from checkpoint: {ckpt_path} to CPU")
    try:
        # Load to CPU, Lightning will move it to the correct device
        lightning_model = CLIPLightningModule.load_from_checkpoint(
            ckpt_path,
            map_location="cpu", 
            model_config=model_config,
            strict=False
        )
        # No need to call .to(device) or .eval() here, Lightning handles it.
        return lightning_model.model # Return the core OpenCLIP model
    except Exception as e:
        print(f"Error loading model from {ckpt_path}: {e}")
        raise

# ----------------------------------------------------------------------
# 1. Dataset wrapper that applies the CLIP transform
# ----------------------------------------------------------------------
class LitImageNetDataset(StreamingDataset):
    def __init__(self, hf_uri: str, transform, shuffle=False, drop_last=False):
        super().__init__(hf_uri, shuffle=shuffle, drop_last=drop_last)
        self.transform = transform            # CLIP pipeline

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        img = sample["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img), torch.tensor(sample["label"], dtype=torch.long)

# --- Model with Linear Probe ---
class CLIPWithLinearProbe(nn.Module):
    def __init__(self, clip_model, embedding_dim, num_classes):
        super().__init__()
        self.clip_model = clip_model
        
        # Freeze CLIP model parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        self.probe = nn.Linear(embedding_dim, num_classes)

    def forward(self, images):
        with torch.no_grad():
            # Use normalize=True as features are typically normalized for linear probes
            image_embeddings = self.clip_model.encode_image(images, normalize=True)
        return self.probe(image_embeddings)

# ----------------------------------------------------------------------
# 2. Lightning DataModule using LitData loaders
# ----------------------------------------------------------------------
class ImageNetLitDataModule(pl.LightningDataModule):
    def __init__(self, transform, batch_size, num_workers):
        super().__init__()
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Index datasets once (optional but faster)
        try:
            ld.index_hf_dataset(HF_URI_TRAIN)
            ld.index_hf_dataset(HF_URI_VAL)
        except Exception as e:
            print(f"Warning: Could not pre-index datasets: {e}")
            print("Will still attempt to stream without pre-indexing")
            
        if stage == "fit" or stage is None:
            self.train_ds = LitImageNetDataset(
                HF_URI_TRAIN, self.transform, shuffle=True, drop_last=True)
            
        if stage == "fit" or stage == "validate" or stage is None:
            self.val_ds = LitImageNetDataset(
                HF_URI_VAL, self.transform, shuffle=False, drop_last=False)

    def train_dataloader(self):
        return StreamingDataLoader(
            self.train_ds, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return StreamingDataLoader(
            self.val_ds, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True)

# --- PyTorch Lightning Module ---
class ImageNetLightningModule(pl.LightningModule):
    def __init__(self, clip_checkpoint_path, model_config, embedding_dim, num_classes, learning_rate, weight_decay):
        super().__init__()
        self.save_hyperparameters() # Saves init args to hparams

        self.base_clip_model = load_clip_model(clip_checkpoint_path, model_config)
        self.model = CLIPWithLinearProbe(self.base_clip_model, embedding_dim, num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        self.train_accuracy(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        self.val_accuracy(outputs, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
        
    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        # Optimize only the probe's parameters
        optimizer = torch.optim.AdamW(self.model.probe.parameters(), 
                                      lr=self.hparams.learning_rate, 
                                      weight_decay=self.hparams.weight_decay,
                                      fused=torch.cuda.is_available()) # Fused AdamW if on GPU
        return optimizer

# --- Main Script ---
def main():
    # Determine number of GPUs to use for DDP
    available_gpus = torch.cuda.device_count()
    gpus_for_trainer = min(NUM_GPUS_TO_USE, available_gpus) if available_gpus > 0 else 0
    
    if gpus_for_trainer == 0 :
        print("No GPUs detected by PyTorch. Running on CPU (if accelerator='cpu').")
        accelerator = "cpu"
        devices_trainer = 1 # For CPU
        strategy_trainer = None
    else:
        print(f"Using {gpus_for_trainer} GPUs for DDP training.")
        accelerator = "gpu"
        devices_trainer = gpus_for_trainer
        strategy_trainer = "ddp" 

    try:
        image_size = VIT_B_32_CONFIG.vision_cfg.image_size
        embedding_dim = VIT_B_32_CONFIG.embed_dim
    except AttributeError as e:
        print(f"Error accessing model config: {e}. Defaulting image_size and embedding_dim.")
        image_size = 224
        embedding_dim = 512

    image_transform = get_image_transform(image_size)
    print(f"Using image size: {image_size}x{image_size}, Embedding dim for probe: {embedding_dim}")

    results_summary = []

    for ckpt_path in CHECKPOINT_PATHS:
        print(f"\n--- Processing checkpoint: {os.path.basename(ckpt_path)} ---")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}. Skipping.")
            continue

        # Instantiate DataModule with LitData
        data_module = ImageNetLitDataModule(
            transform=image_transform,
            batch_size=BATCH_SIZE_PER_GPU,
            num_workers=NUM_WORKERS,
        )

        # Instantiate LightningModule
        lightning_model = ImageNetLightningModule(
            clip_checkpoint_path=ckpt_path,
            model_config=VIT_B_32_CONFIG,
            embedding_dim=embedding_dim,
            num_classes=NUM_CLASSES,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # Setup logger and callbacks
        csv_logger = CSVLogger("logs", name=f"clip_linear_probe_imagenet_litdata_{os.path.basename(ckpt_path).replace('.ckpt','')}")
        progress_bar_callback = TQDMProgressBar(refresh_rate=10) # Refresh rate for tqdm
        
        # Checkpoint callback to save the best model based on validation accuracy
        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",      # Metric to monitor
            mode="max",             # Maximize the metric
            save_top_k=1,           # Save only the best model
            dirpath=os.path.join(csv_logger.log_dir, "checkpoints"), # Save checkpoints in logger's dir
            filename="{epoch}-{val_acc:.2f}" # Filename format
        )

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices_trainer,
            strategy=strategy_trainer if gpus_for_trainer > 0 else None,
            max_epochs=EPOCHS,
            logger=csv_logger,
            callbacks=[progress_bar_callback, checkpoint_callback],
            precision='bf16-mixed' if accelerator == "gpu" else 32 # Optional: for mixed precision
        )

        print(f"Starting training for checkpoint: {os.path.basename(ckpt_path)}")
        trainer.fit(lightning_model, datamodule=data_module)
        
        # After fit, the best val_acc is available in checkpoint_callback
        best_val_acc = checkpoint_callback.best_model_score
        if best_val_acc is not None:
             print(f"Finished training for {os.path.basename(ckpt_path)}. Best Validation Accuracy: {best_val_acc.item():.2f}%")
             results_summary.append({
                "checkpoint": os.path.basename(ckpt_path),
                "dataset": "imagenet-1k (LitData streaming)",
                "accuracy": best_val_acc.item() # Storing the numeric value
            })
        else:
            print(f"Finished training for {os.path.basename(ckpt_path)}. No validation accuracy recorded by checkpoint callback.")

    print("\n--- Summary of Results ---")
    if results_summary:
        for res in results_summary:
            print(f"Checkpoint: {res['checkpoint']}, Dataset: {res['dataset']}, Accuracy: {res['accuracy']:.2f}%")
    else:
        print("No results to display. Check for errors during processing.")

if __name__ == "__main__":
    main()
