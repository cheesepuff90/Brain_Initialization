import os
import sys
import io
import csv
import warnings
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from typing import Optional

# -----------------------------
# PIL / warning handling
# -----------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
warnings.filterwarnings("ignore", category=UserWarning, message=".*EXIF data.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Corrupt EXIF.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Truncated File Read.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Metadata Warning.*")

_original_getexif = Image.Image.getexif
def safe_getexif(self):
    try:
        return _original_getexif(self)
    except Exception:
        return Image.Exif()
Image.Image.getexif = safe_getexif

# -----------------------------
# Project imports
# -----------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

token_path = os.path.expanduser("~/.cache/huggingface/token")
if os.path.exists(token_path):
    with open(token_path) as f:
        hf_token = f.read().strip()
else:
    hf_token = os.environ.get("HF_HUB_TOKEN")
    if not hf_token:
        print("Hugging Face token not found. Please set ~/.cache/huggingface/token or HF_HUB_TOKEN.")

os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface/models"
os.environ["HF_HUB_CACHE"] = "/tmp/huggingface/hub"
os.environ["HF_HUB_TOKEN"] = hf_token if hf_token else ""

os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)

from datasets import load_dataset
from src.lightning_module import CLIPLightningModule
from config import VIT_B_32_CONFIG

# -----------------------------
# Configuration
# -----------------------------
CHECKPOINTS = [
    {
        "label": "0pct",
        "path": "/home/kogs/checkpoints/perceptual_init_yfcc3m_vitb32_0pct/checkpoints/last.ckpt",
    },
    {
        "label": "25pct",
        "path": "/home/kogs/checkpoints/perceptual_init_yfcc3m_vitb32_25pct/checkpoints/last.ckpt",
    },
    {
        "label": "100pct",
        "path": "/home/kogs/checkpoints/perceptual_init_yfcc3m_vitb32_100pct/checkpoints/last.ckpt",
    },
]

DATASET_NAME = "imagenet-1k"
NUM_CLASSES = 1000
IMAGE_KEY = "image"
LABEL_KEY = "label"
SPLIT_TRAIN = "train"
SPLIT_VAL = "validation"

CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

BATCH_SIZE_IMAGES = 512
BATCH_SIZE_HEAD = 512
NUM_WORKERS = 4
HEAD_EPOCHS = 32
HEAD_LR = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOG_ROOT = Path("logs")
LOG_ROOT.mkdir(exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def get_image_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        lambda image: image.convert("RGB") if image.mode != "RGB" else image,
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_IMAGE_MEAN, std=CLIP_IMAGE_STD),
    ])


def load_clip_model(ckpt_path: str, model_config):
    print(f"Loading CLIP model from checkpoint: {ckpt_path}")
    lightning_model = CLIPLightningModule.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        model_config=model_config,
        strict=False,
    )
    model = lightning_model.model
    model.eval()
    model.to(DEVICE)
    return model


class ImageNet1kDataset(Dataset):
    def __init__(self, split: str, transform, hf_token_val: Optional[str]):
        self.transform = transform
        print(f"Loading ImageNet-1k '{split}' split...")
        self.dataset = load_dataset(
            DATASET_NAME,
            split=split,
            token=hf_token_val,
            trust_remote_code=True,
            num_proc=4,
        )
        print(f"Successfully loaded {len(self.dataset)} samples from '{split}' split.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[IMAGE_KEY]
        label = item[LABEL_KEY]

        if image.mode != "RGB":
            image = image.convert("RGB")

        image_data = io.BytesIO()
        image.save(image_data, format=image.format or "JPEG")
        image_data.seek(0)
        image = Image.open(image_data)

        transformed = self.transform(image)
        return transformed, torch.tensor(label, dtype=torch.long)


class LinearHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=1e-6)
        self.fc = nn.Linear(input_dim, num_classes)
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()

    def forward(self, x):
        return self.fc(self.bn(x))


def extract_or_load_features(
    clip_model,
    dataloader,
    split_name: str,
    cache_path: Path,
):
    if cache_path.exists():
        print(f"Loading cached {split_name} features from {cache_path}...")
        data = torch.load(cache_path, map_location="cpu")
        return data["features"], data["labels"]

    print(f"Extracting features for {split_name}...")
    feature_list = []
    label_list = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=split_name):
            images = images.to(DEVICE, non_blocking=True)
            feats = clip_model.encode_image(images, normalize=True)
            feature_list.append(feats.cpu())
            label_list.append(labels.cpu())

    features = torch.cat(feature_list, dim=0)
    labels = torch.cat(label_list, dim=0)

    print(f"Saving {split_name} features to {cache_path}...")
    torch.save({"features": features, "labels": labels}, cache_path)

    return features, labels


def train_linear_head(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    input_dim: int,
    num_classes: int,
    output_csv: Path,
):
    print(f"\nTraining linear head (dim={input_dim}, lr={HEAD_LR}, epochs={HEAD_EPOCHS})...")

    dataset = TensorDataset(train_feats, train_labels)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_HEAD,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    head = LinearHead(input_dim, num_classes).to(DEVICE)
    optimizer = optim.SGD(head.parameters(), lr=HEAD_LR, momentum=0.9, weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HEAD_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    records = []
    head.train()

    for epoch in range(HEAD_EPOCHS):
        total_loss = 0.0
        total_correct1 = 0
        total_correct5 = 0
        total = 0

        for b_feats, b_labels in loader:
            b_feats = b_feats.to(DEVICE, non_blocking=True)
            b_labels = b_labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            outputs = head(b_feats)
            loss = criterion(outputs, b_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, pred1 = outputs.topk(1, 1, True, True)
            total_correct1 += pred1.squeeze(1).eq(b_labels).sum().item()

            _, pred5 = outputs.topk(5, 1, True, True)
            total_correct5 += pred5.eq(b_labels.view(-1, 1)).sum().item()
            total += b_labels.size(0)

        scheduler.step()

        avg_loss = total_loss / len(loader)
        train_acc1 = total_correct1 / total
        train_acc5 = total_correct5 / total

        record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_acc1": train_acc1,
            "train_acc5": train_acc5,
            "lr": scheduler.get_last_lr()[0],
        }
        records.append(record)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1}/{HEAD_EPOCHS}] "
                f"loss={avg_loss:.4f} train_acc1={train_acc1:.4f} train_acc5={train_acc5:.4f} "
                f"lr={record['lr']:.6f}"
            )

        pd.DataFrame(records).to_csv(output_csv, index=False)

    return head, pd.DataFrame(records)


def evaluate_head(head, val_feats: torch.Tensor, val_labels: torch.Tensor):
    head.eval()
    dataset = TensorDataset(val_feats, val_labels)
    loader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    correct1 = 0
    correct5 = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for b_feats, b_labels in loader:
            b_feats = b_feats.to(DEVICE, non_blocking=True)
            b_labels = b_labels.to(DEVICE, non_blocking=True)

            outputs = head(b_feats)
            loss = criterion(outputs, b_labels)
            total_loss += loss.item()

            _, pred1 = outputs.topk(1, 1, True, True)
            correct1 += pred1.squeeze(1).eq(b_labels).sum().item()

            _, pred5 = outputs.topk(5, 1, True, True)
            correct5 += pred5.eq(b_labels.view(-1, 1)).sum().item()
            total += b_labels.size(0)

    return {
        "val_loss": total_loss / len(loader),
        "val_acc1": correct1 / total,
        "val_acc5": correct5 / total,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    image_size = VIT_B_32_CONFIG.vision_cfg.image_size
    embedding_dim = VIT_B_32_CONFIG.embed_dim
    image_transform = get_image_transform(image_size)

    results_summary = []

    for ckpt_info in CHECKPOINTS:
        ckpt_label = ckpt_info["label"]
        ckpt_path = ckpt_info["path"]

        print(f"\n--- Processing checkpoint: {ckpt_label} ---")
        print(f"Checkpoint path: {ckpt_path}")

        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found for {ckpt_label}: {ckpt_path}. Skipping.")
            continue

        run_dir = LOG_ROOT / f"clip_linear_probe_imagenet_{ckpt_label}"
        run_dir.mkdir(parents=True, exist_ok=True)

        train_cache = run_dir / "cached_features_train.pt"
        val_cache = run_dir / "cached_features_val.pt"
        epoch_log_csv = run_dir / "linear_probe_epoch_log.csv"

        # dataset / loaders
        train_ds = ImageNet1kDataset(SPLIT_TRAIN, image_transform, hf_token)
        val_ds = ImageNet1kDataset(SPLIT_VAL, image_transform, hf_token)

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE_IMAGES,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE_IMAGES,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        # backbone
        clip_model = load_clip_model(ckpt_path, VIT_B_32_CONFIG)

        # feature extraction
        train_feats, train_labels = extract_or_load_features(
            clip_model, train_loader, f"{ckpt_label}_train", train_cache
        )
        val_feats, val_labels = extract_or_load_features(
            clip_model, val_loader, f"{ckpt_label}_val", val_cache
        )

        # free VRAM after extraction
        del clip_model, train_loader, val_loader, train_ds, val_ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # train head
        head, epoch_df = train_linear_head(
            train_feats=train_feats,
            train_labels=train_labels,
            input_dim=embedding_dim,
            num_classes=NUM_CLASSES,
            output_csv=epoch_log_csv,
        )

        # evaluate
        metrics = evaluate_head(head, val_feats, val_labels)

        # append val metrics to epoch log and resave
        epoch_df["val_loss"] = None
        epoch_df["val_acc1"] = None
        epoch_df["val_acc5"] = None
        epoch_df.loc[epoch_df.index[-1], "val_loss"] = metrics["val_loss"]
        epoch_df.loc[epoch_df.index[-1], "val_acc1"] = metrics["val_acc1"]
        epoch_df.loc[epoch_df.index[-1], "val_acc5"] = metrics["val_acc5"]
        epoch_df.to_csv(epoch_log_csv, index=False)

        print(f"Finished training for {ckpt_label}.")
        print(f"Top-1 Validation Accuracy: {metrics['val_acc1'] * 100:.4f}%")
        print(f"Top-5 Validation Accuracy: {metrics['val_acc5'] * 100:.4f}%")

        results_summary.append({
            "checkpoint": ckpt_label,
            "checkpoint_path": ckpt_path,
            "dataset": DATASET_NAME,
            "accuracy_top1": metrics["val_acc1"] * 100,
            "accuracy_top5": metrics["val_acc5"] * 100,
        })

        pd.DataFrame(results_summary).to_csv("logs/imagenet_linear_probe_results.csv", index=False)
        print(f"[{ckpt_label}] Saved intermediate CSV")

        # save head checkpoint
        torch.save(
            {
                "checkpoint_label": ckpt_label,
                "checkpoint_path": ckpt_path,
                "head_state_dict": head.state_dict(),
                "metrics": metrics,
            },
            run_dir / "linear_probe_head.pt",
        )

    print("\n--- Summary of Results ---")
    print(f"{'Checkpoint':<20} | {'Top-1 Accuracy':<15} | {'Top-5 Accuracy':<15}")
    print("-" * 60)

    if results_summary:
        df = pd.DataFrame(results_summary)
        save_path = "logs/imagenet_linear_probe_results.csv"
        df.to_csv(save_path, index=False)
        print(f"\nSaved CSV to {save_path}")

        for res in results_summary:
            print(f"{res['checkpoint']:<20} | {res['accuracy_top1']:>13.4f}% | {res['accuracy_top5']:>13.4f}%")
    else:
        print("No results to display.")


if __name__ == "__main__":
    main()