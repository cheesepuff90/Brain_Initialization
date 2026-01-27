#!/usr/bin/env python3
"""
test_checkpoint_loading.py

Load a saved CLIPLightningModule checkpoint and run
basic forward passes with dummy image and text data.
"""
import os
import torch
from src.lightning_module import CLIPLightningModule
from config import VIT_B_32_CONFIG

def main():
    # ------------------------------------------------------------------
    # 1) Point to your checkpoint (or override via CKPT_PATH env var)
    # ------------------------------------------------------------------
    ckpt_path = os.environ.get(
        "CKPT_PATH",
        "training_logs/yfcc15m_litdata_20250503_145332_ddp_nodeNone"
        "/checkpoints/monitor_val_yfcc15m_clip_loss-epoch_epoch=013.ckpt"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {ckpt_path} onto {device}")

    # ------------------------------------------------------------------
    # 2) Load the Lightning module (must supply model_config manually)
    #    and allow unexpected keys (e.g. HF text-model weights)
    # ------------------------------------------------------------------
    lightning_model = CLIPLightningModule.load_from_checkpoint(
        ckpt_path,
        map_location=device,           # ensure loaded onto correct device
        model_config=VIT_B_32_CONFIG,  # restore model architecture
        strict=False                   # ignore unexpected HF text-model keys
    )
    lightning_model = lightning_model.to(device)
    lightning_model.eval()

    # Extract the underlying CLIP model
    clip_model = lightning_model.model

    # ------------------------------------------------------------------
    # 3) Dummy Image‐only forward
    # ------------------------------------------------------------------
    batch_size = 2
    img_size   = VIT_B_32_CONFIG.vision_cfg.image_size
    dummy_images = torch.randn(batch_size, 3, img_size, img_size, device=device)
    img_feats = clip_model.encode_image(dummy_images, normalize=True)
    print(f"[Image only] features shape: {img_feats.shape}")

    # ------------------------------------------------------------------
    # 4) Dummy Text‐only forward
    # ------------------------------------------------------------------
    dummy_texts = ["a photo of a cat", "a photo of a dog"]
    tokenizer = clip_model.get_tokenizer()
    tokenized = tokenizer(
        dummy_texts,
        padding="max_length",
        truncation=True,
        max_length=VIT_B_32_CONFIG.text_cfg.context_length,
        return_tensors="pt"
    ).to(device)
    txt_feats = clip_model.encode_text(tokenized['input_ids'], normalize=True)
    print(f"[Text only] features shape: {txt_feats.shape}")

    # ------------------------------------------------------------------
    # 5) Combined image–text logits forward
    # ------------------------------------------------------------------
    logits_per_image, logits_per_text, logit_scale = clip_model(
        dummy_images,
        tokenized['input_ids']
    )
    print(f"[Image→Text] logits_per_image shape: {logits_per_image.shape}")
    print(f"[Text→Image] logits_per_text shape: {logits_per_text.shape}")
    print(f"Learned logit_scale: {logit_scale.item():.4f}")

if __name__ == "__main__":
    main()
