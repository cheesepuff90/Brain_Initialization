import os
import math
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path # Add pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.serialization import add_safe_globals
# Add imports for evaluation
import torchvision.datasets as tv_datasets
from torch.utils.data import Dataset
# Removed unused zsc import
# Import BatchEncoding for CustomTokenizer type hinting
try:
    from transformers.tokenization_utils_base import BatchEncoding
except ImportError:
    BatchEncoding = dict # Fallback
# Add torchmetrics import
import torchmetrics

from config import (
    TrainingConfig, CLIPConfig, 
    ContrastiveLossConfig, CLIPLossConfig  # Add these imports to fix the error
)
from src.model import CLIP
from src.loss import TripletContrastiveLoss, ClipLoss

class CLIPLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training CLIP model.
    Supports both Triplet contrastive learning (nod) and
    Image-Text contrastive learning (CLIP style).
    Includes CIFAR10 zero-shot evaluation for CLIP style.
    """
    def __init__(
        self,
        model_config: CLIPConfig,
        training_config: TrainingConfig,
        train_dataset_size: int = 0,  # Used for epoch-based LR scheduling (nod)
    ):
        """
        Args:
            model_config: Configuration for the CLIP model
            training_config: Configuration for training
            train_dataset_size: Size of the training dataset for LR scheduling (nod)
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model_config'])

        self.model_config = model_config
        self.training_config = training_config
        self.train_dataset_size = train_dataset_size

        self.train_mode = training_config.data.dataset_type

        if self.train_mode == "nod_imagenet_triplets":
            self.train_mode = "nod"

        if self.train_mode not in ['nod', 'things', 'imagenet1k', 'yfcc15m', 'yfcc15m_litdata']:
            raise ValueError(f"Unsupported dataset_type in training_config: {self.train_mode}")

        # THINGS and ImageNet1k use triplet learning like nod
        use_random_text_encoder = (self.train_mode in ['nod', 'things', 'imagenet1k', 'yfcc15m_litdata'])
        self.model = CLIP(
             config=model_config,
             random_text_encoder=use_random_text_encoder
        )
        """for name, param in self.model.named_parameters():
            param.requires_grad = True

        # === Load weights from nod checkpoint if provided ===
        init_ckpt_path = getattr(self.training_config, "init_ckpt_path", None)
        if init_ckpt_path:
            print(f"[INIT] Loading model weights from: {init_ckpt_path}")
            try:
                # Attempt weights-only load (PyTorch 2.6+ default)
                try:
                    ckpt = torch.load(init_ckpt_path, map_location="cpu", weights_only=True)
                except Exception as e:
                    print(f"[INIT] weights_only=True failed: {e}")
                    # Retry without weights_only (but only safe if checkpoint is trusted!)
                    add_safe_globals({'TrainingConfig': TrainingConfig})
                    ckpt = torch.load(init_ckpt_path, map_location="cpu",weights_only=False)

                state_dict = ckpt.get("state_dict", ckpt)
                # Compare keys manually
                model_keys = set(self.model.state_dict().keys())
                ckpt_keys = set(state_dict.keys())
                matched_keys = model_keys & ckpt_keys
                missing_keys = model_keys - ckpt_keys
                unexpected_keys = ckpt_keys - model_keys

                print(f"[INIT] Matched keys: {len(matched_keys)}")
                print(f"[INIT] Missing keys: {len(missing_keys)}")
                print(f"[INIT] Unexpected keys: {len(unexpected_keys)}")
            except Exception as e:
                print(f"[INIT] Failed to load checkpoint: {e}")

        # === Freeze first N transformer blocks in visual encoder if requested ===
        freeze_blocks = getattr(self.training_config, "freeze_vision_blocks", 0)
        if self.train_mode.startswith("yfcc") and freeze_blocks > 0:
            print(f"[INIT] Freezing first {freeze_blocks} vision transformer blocks")
            for name, param in self.model.named_parameters():
                if any(f"resblocks.{i}." in name for i in range(freeze_blocks)):
                    param.requires_grad = False"""


        if self.train_mode in ['nod', 'imagenet1k','things']:
            self.loss_fn = TripletContrastiveLoss(
                margin=training_config.loss.margin,
                distance_type=training_config.loss.distance_type,
                scale=training_config.loss.scale
            )
        
        elif self.train_mode in ['yfcc15m', 'yfcc15m_litdata']:
            rank = self.global_rank if hasattr(self, 'global_rank') else int(os.environ.get("RANK", 0))
            world_size = self.world_size if hasattr(self, 'world_size') else int(os.environ.get("WORLD_SIZE", 1))
            if isinstance(training_config.loss, ContrastiveLossConfig) and not isinstance(training_config.loss, CLIPLossConfig):
                 loss_cfg = training_config.loss
                 clip_loss_params = {
                     "local_loss": getattr(loss_cfg, 'local_loss', False),
                     "gather_with_grad": getattr(loss_cfg, 'gather_with_grad', False),
                     "cache_labels": getattr(loss_cfg, 'cache_labels', True),
                 }
            else:
                 loss_cfg = training_config.loss
                 clip_loss_params = {
                     "local_loss": loss_cfg.local_loss,
                    #  "gather_with_grad": loss_cfg.gather_with_grad,
                     "gather_with_grad": True,
                     "cache_labels": loss_cfg.cache_labels,
                 }
            self.loss_fn = ClipLoss(
                **clip_loss_params,
                rank=rank,
                world_size=world_size
            )
            self.logit_scale_value = self.model.logit_scale.item()
            self.classifier_weights = {}
            self._classifier_epoch = -1
            self.cifar10_classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            self.cifar10_templates = ["a photo of a {c}."]
            self.cifar100_classes = (
                'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
                'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
                'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
                'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
                'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
                'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                'worm'
            )
            self.cifar100_templates = ["a photo of a {c}."]
            num_classes_cifar10 = len(self.cifar10_classes)
            top_k_val_cifar10 = min(5, num_classes_cifar10) if num_classes_cifar10 > 1 else 1
            self.val_accuracy_top1_cifar10 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes_cifar10, top_k=1)
            if top_k_val_cifar10 >= 5:
                self.val_accuracy_top5_cifar10 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes_cifar10, top_k=top_k_val_cifar10)
            else:
                self.val_accuracy_top5_cifar10 = None
            num_classes_cifar100 = len(self.cifar100_classes)
            top_k_val_cifar100 = min(5, num_classes_cifar100) if num_classes_cifar100 > 1 else 1
            self.val_accuracy_top1_cifar100 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes_cifar100, top_k=1)
            if top_k_val_cifar100 >= 5:
                self.val_accuracy_top5_cifar100 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes_cifar100, top_k=top_k_val_cifar100)
            else:
                self.val_accuracy_top5_cifar100 = None
            self.val_accuracy_top1 = self.val_accuracy_top1_cifar10
            self.val_accuracy_top5 = self.val_accuracy_top5_cifar10

        self.train_correct = 0
        self.train_total = 0
        self.val_correct = 0
        self.val_total = 0
    
    def setup(self, stage: str):
        if stage == 'fit' and self.train_mode.startswith("yfcc15m"):
             # Classifier weights are needed for CIFAR evaluation
             pl.utilities.rank_zero_info("CIFAR10/100 classifier weights will be built at the start of each validation epoch.")

    def forward(self, image, text=None):
        if text is None:
            return self.model.encode_image(image, normalize=True)
        else:
            return self.model(image, text)

    def on_train_start(self):
        if not self.trainer.loggers:
             return
        def log_hyperparams_on_rank_zero():
            if hasattr(self.logger, "log_hyperparams"):
                hp_dict = {
                    "model_type": "ViT-B/32",
                    "embed_dim": self.model_config.embed_dim,
                    "vision_layers": self.model_config.vision_cfg.layers,
                    "vision_width": self.model_config.vision_cfg.width,
                    "vision_patch_size": self.model_config.vision_cfg.patch_size,
                    "lr": self.training_config.optimizer.lr,
                    "weight_decay": self.training_config.optimizer.weight_decay,
                    "batch_size": self.training_config.data.batch_size,
                    "epochs": self.training_config.epochs,
                    "optimizer": self.training_config.optimizer.name,
                    "scheduler": self.training_config.scheduler.name,
                    "warmup_steps": self.training_config.scheduler.warmup_steps,
                    "train_mode": self.train_mode,
                    "model/text/layers": getattr(self.model_config.text_cfg, 'layers', "N/A"),
                    "model/text/width": getattr(self.model_config.text_cfg, 'width', "N/A"),
                    "loss/type": self.loss_fn.__class__.__name__,
                }
                if self.train_mode in ['nod', 'imagenet1k', 'things']:  # Add 'things' here
                    hp_dict["loss/margin"] = self.training_config.loss.margin
                    hp_dict["loss/distance_type"] = self.training_config.loss.distance_type
                    hp_dict["loss/scale"] = self.training_config.loss.scale
                elif self.train_mode == 'yfcc15m':
                     hp_dict["loss/logit_scale_init"] = self.model_config.init_logit_scale
                     hp_dict["loss/local_loss"] = getattr(self.training_config.loss, 'local_loss', False)
                     hp_dict["loss/gather_with_grad"] = getattr(self.training_config.loss, 'gather_with_grad', False)
                try:
                    self.logger.log_hyperparams(hp_dict)
                except Exception:
                    pass
        log_hyperparams_on_rank_zero()

    # def training_step(self, batch, batch_idx):
    #     if self.train_mode == 'nod':
    #         return self._triplet_step(batch, batch_idx, step_type="train")
    #     elif self.train_mode in ['yfcc15m', 'yfcc15m_litdata']:
    #         return self._clip_step(batch, batch_idx, step_type="train")
    #     else:
    #         raise ValueError(f"Invalid train_mode: {self.train_mode}")

    def training_step(self, batch, batch_idx):
        if self.train_mode in ['nod', 'imagenet1k', 'things']:  # Add 'things' here
            result = self._triplet_step(batch, batch_idx, step_type="train")
        elif self.train_mode in ['yfcc15m', 'yfcc15m_litdata']:
            result = self._clip_step(batch, batch_idx, step_type="train")
        else:
            raise ValueError(f"Invalid train_mode: {self.train_mode}")

        return result["loss"]  # This ensures loss is used for backprop


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Determine current dataloader type based on train_mode and dataloader_idx
        if self.train_mode in ['nod', 'imagenet1k', 'things']:  # Add 'things' here
            return self._triplet_step(batch, batch_idx, step_type="val")
        elif self.train_mode in ['yfcc15m', 'yfcc15m_litdata']:
            # YFCC15M Training Mode - Validation can be YFCC itself or CIFAR
            # Dataloader Indices convention (assuming YFCC val loader exists and is first):
            # 0: YFCC15M Validation Split
            # 1: CIFAR10
            # 2: CIFAR100 (if multi_val)
            # Need to adjust if YFCC val loader doesn't exist or multi_val=False

            # Check if the first dataloader is expected to be YFCC val
            # This is a bit indirect, could be improved if config explicitly tracked loaders
            # A simple check: if the batch has 2 elements (img, txt) it's likely YFCC
            is_likely_yfcc_val = (isinstance(batch, (list, tuple)) and len(batch) == 2 and torch.is_tensor(batch[0]) and torch.is_tensor(batch[1]))

            if dataloader_idx == 0 and is_likely_yfcc_val:
                 # --- YFCC15M Validation ---
                 try:
                     loss_dict = self._clip_step(batch, batch_idx, step_type="val")
                     self.log("val/yfcc15m/clip_loss", loss_dict["loss"].detach(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
                     return loss_dict # Return loss for potential averaging
                 except Exception as e:
                     # Log failure for this batch
                     pl.utilities.rank_zero_warn(f"YFCC validation step failed: {e}")
                     self.log("val/yfcc15m/clip_loss", 0.0, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
                     return None # Indicate failure for this step

            else:
                 # --- CIFAR Validation ---
                 # Adjust dataloader_idx mapping if YFCC val loader wasn't present (idx 0 becomes CIFAR10)
                 # For simplicity, assume YFCC val exists if we reach here and idx != 0,
                 # otherwise, assume idx 0 = CIFAR10, idx 1 = CIFAR100
                 cifar_idx_offset = 1 if is_likely_yfcc_val else 0 # If YFCC was idx 0, CIFAR starts at 1

                 if dataloader_idx == cifar_idx_offset: # CIFAR10
                     dataset_name = "cifar10"
                     accuracy_top1 = self.val_accuracy_top1_cifar10
                     accuracy_top5 = self.val_accuracy_top5_cifar10
                 elif dataloader_idx == cifar_idx_offset + 1: # CIFAR100
                     dataset_name = "cifar100"
                     accuracy_top1 = self.val_accuracy_top1_cifar100
                     accuracy_top5 = self.val_accuracy_top5_cifar100
                 else:
                      # Should not happen with current logic, but log a warning
                      pl.utilities.rank_zero_warn(f"Unexpected dataloader_idx {dataloader_idx} in validation_step for yfcc15m mode.")
                      return None

                 # Check/Build classifier weights
                 if dataset_name not in self.classifier_weights or self._classifier_epoch != self.current_epoch:
                     try:
                         self._build_zero_shot_classifier(dataset=dataset_name)
                         self._classifier_epoch = self.current_epoch
                     except Exception as e:
                         pl.utilities.rank_zero_warn(f"Failed to build {dataset_name} classifier: {e}")
                         # Log zero accuracy to avoid downstream errors
                         self.log(f"val/{dataset_name}/epoch_acc1", 0.0, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
                         if accuracy_top5 is not None:
                             self.log(f"val/{dataset_name}/epoch_acc5", 0.0, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
                         return None # Skip this batch

                 # Proceed with CIFAR evaluation
                 images, target = batch
                 try:
                     images = images.to(self.device)
                     target = target.to(self.device)
                     image_features = self.model.encode_image(images, normalize=True)
                     # Ensure classifier weights are available and on the correct device
                     if dataset_name not in self.classifier_weights:
                         raise RuntimeError(f"Classifier weights for {dataset_name} not found after build attempt.")
                     classifier_weights = self.classifier_weights[dataset_name].to(image_features.device)

                     logits = self.model.logit_scale.exp() * image_features @ classifier_weights
                     accuracy_top1.update(logits, target)
                     if accuracy_top5 is not None:
                         accuracy_top5.update(logits, target)
                     # Log step accuracy? Usually not needed, epoch is better.
                     # self.log(f"val/{dataset_name}/step_acc1", accuracy_top1(logits, target), on_step=True, on_epoch=False)
                     return None # No loss to return for accuracy calculation
                 except Exception as e:
                      pl.utilities.rank_zero_warn(f"CIFAR validation step failed for {dataset_name}: {e}")
                      # Log zero accuracy for this batch to indicate failure
                      self.log(f"val/{dataset_name}/epoch_acc1", 0.0, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
                      if accuracy_top5 is not None:
                         self.log(f"val/{dataset_name}/epoch_acc5", 0.0, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
                      return None
        else:
             raise ValueError(f"Invalid train_mode: {self.train_mode}")

    def test_step(self, batch, batch_idx):
        if self.train_mode in ['nod', 'imagenet1k','things']:
            result = self._triplet_step(batch, batch_idx, step_type="test")
            return result
        elif self.train_mode in ['yfcc15m', 'yfcc15m_litdata']:
            self.log("test/loss", 0.0, sync_dist=True)
            return None
        else:
            raise ValueError(f"Invalid train_mode: {self.train_mode}")

    def _triplet_step(self, batch, batch_idx, step_type="train"):
        ref_img, dist0_img, dist1_img, target, _ = batch
        ref_features = self.model.encode_image(ref_img, normalize=False)
        dist0_features = self.model.encode_image(dist0_img, normalize=False)
        dist1_features = self.model.encode_image(dist1_img, normalize=False)
        loss_dict = self.loss_fn(ref_features, dist0_features, dist1_features, target)
        loss = loss_dict["loss"]
        decisions = (loss_dict["logits"] > 0).float()
        correct = (decisions == target).sum().item()
        total = target.size(0)
        log_on_step = (step_type == "train")
        log_on_epoch = True
        prog_bar = (step_type != "test")
        self.log(f"{step_type}/triplet_loss", loss, on_step=log_on_step, on_epoch=log_on_epoch, prog_bar=prog_bar, sync_dist=True)
        self.log(f"{step_type}/dist0_mean", loss_dict["dist0"].mean().detach(), on_step=False, on_epoch=log_on_epoch, sync_dist=True)
        self.log(f"{step_type}/dist1_mean", loss_dict["dist1"].mean().detach(), on_step=False, on_epoch=log_on_epoch, sync_dist=True)
        if step_type == "train":
            self.train_correct += correct
            self.train_total += total
        if step_type == "val":
            self.val_correct += correct
            self.val_total += total
        return {"loss": loss, "correct": correct, "total": total}

    def _clip_step(self, batch, batch_idx, step_type="train"):
        images, texts = batch
        image_features = self.model.encode_image(images, normalize=True)
        text_features = self.model.encode_text(texts, normalize=True)
        loss_dict = self.loss_fn(
            image_features,
            text_features,
            self.model.logit_scale,
            output_dict=True
        )
        loss = loss_dict["contrastive_loss"]
        log_on_step = (step_type == "train")
        log_on_epoch = True
        prog_bar = (step_type == "train")
        self.log(f"{step_type}/clip_loss", loss, on_step=log_on_step, on_epoch=log_on_epoch, prog_bar=prog_bar, sync_dist=True)
        self.log(f"{step_type}/loss_img", loss_dict["loss_img"].detach(), on_step=False, on_epoch=log_on_epoch, sync_dist=True)
        self.log(f"{step_type}/loss_txt", loss_dict["loss_txt"].detach(), on_step=False, on_epoch=log_on_epoch, sync_dist=True)
        self.log(f"{step_type}/logit_scale", self.model.logit_scale.detach(), on_step=log_on_step, on_epoch=log_on_epoch, sync_dist=True)
        return {"loss": loss}

    def on_train_epoch_end(self):
        if self.train_mode in ['nod', 'things', 'imagenet1k'] and self.train_total > 0:
            epoch_accuracy = self.train_correct / self.train_total
            self.log("train/epoch_accuracy", epoch_accuracy, prog_bar=True, sync_dist=True)
            self.train_correct = 0
            self.train_total = 0

    def on_validation_epoch_end(self):
        if self.train_mode in ['nod', 'things', 'imagenet1k']:
            epoch_accuracy = self.val_correct / self.val_total
            self.log("val/epoch_accuracy", epoch_accuracy, prog_bar=True, sync_dist=True)
            self.val_correct = 0
            self.val_total = 0
        
        elif self.train_mode.startswith("yfcc15m"):
             # --- YFCC15M Loss Summary ---
             # Lightning automatically averages metrics logged with on_epoch=True
             # We retrieve it from trainer.callback_metrics for the console log
             if "val/yfcc15m/clip_loss" in self.trainer.callback_metrics:
                 yfcc_loss_epoch = self.trainer.callback_metrics["val/yfcc15m/clip_loss"]
                 pl.utilities.rank_zero_info(f"Epoch {self.current_epoch}: YFCC15M Val Loss = {yfcc_loss_epoch:.4f}")
             else:
                 # This might happen if the first validation step failed consistently
                 pl.utilities.rank_zero_info(f"Epoch {self.current_epoch}: YFCC15M Val Loss metric not found.")


             # --- CIFAR10 Accuracy Summary ---
             try:
                 # Check if the metric object exists and has computed values
                 if hasattr(self.val_accuracy_top1_cifar10, 'compute'):
                     final_acc1_cifar10 = self.val_accuracy_top1_cifar10.compute()
                     self.log("val/cifar10/epoch_acc1", final_acc1_cifar10, on_epoch=True, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
                     pl.utilities.rank_zero_info(f"Epoch {self.current_epoch}: CIFAR10 Val Acc@1 = {final_acc1_cifar10.item():.4f}")
                     if self.val_accuracy_top5_cifar10 and hasattr(self.val_accuracy_top5_cifar10, 'compute'):
                         final_acc5_cifar10 = self.val_accuracy_top5_cifar10.compute()
                         self.log("val/cifar10/epoch_acc5", final_acc5_cifar10, on_epoch=True, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
                         pl.utilities.rank_zero_info(f"Epoch {self.current_epoch}: CIFAR10 Val Acc@5 = {final_acc5_cifar10.item():.4f}")
                 else:
                     pl.utilities.rank_zero_info(f"Epoch {self.current_epoch}: CIFAR10 metric object not ready for compute.")
             except Exception as e:
                 pl.utilities.rank_zero_warn(f"Error computing/logging CIFAR10 validation metrics: {e}")
             finally:
                 # Always reset metrics
                 if hasattr(self.val_accuracy_top1_cifar10, 'reset'): self.val_accuracy_top1_cifar10.reset()
                 if self.val_accuracy_top5_cifar10 and hasattr(self.val_accuracy_top5_cifar10, 'reset'): self.val_accuracy_top5_cifar10.reset()


             # --- CIFAR100 Accuracy Summary ---
             try:
                 if hasattr(self.val_accuracy_top1_cifar100, 'compute'):
                     final_acc1_cifar100 = self.val_accuracy_top1_cifar100.compute()
                     self.log("val/cifar100/epoch_acc1", final_acc1_cifar100, on_epoch=True, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
                     pl.utilities.rank_zero_info(f"Epoch {self.current_epoch}: CIFAR100 Val Acc@1 = {final_acc1_cifar100.item():.4f}")
                     if self.val_accuracy_top5_cifar100 and hasattr(self.val_accuracy_top5_cifar100, 'compute'):
                         final_acc5_cifar100 = self.val_accuracy_top5_cifar100.compute()
                         self.log("val/cifar100/epoch_acc5", final_acc5_cifar100, on_epoch=True, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
                         pl.utilities.rank_zero_info(f"Epoch {self.current_epoch}: CIFAR100 Val Acc@5 = {final_acc5_cifar100.item():.4f}")
                 else:
                      pl.utilities.rank_zero_info(f"Epoch {self.current_epoch}: CIFAR100 metric object not ready for compute.")
             except Exception as e:
                 pl.utilities.rank_zero_warn(f"Error computing/logging CIFAR100 validation metrics: {e}")
             finally:
                 # Always reset metrics
                 if hasattr(self.val_accuracy_top1_cifar100, 'reset'): self.val_accuracy_top1_cifar100.reset()
                 if self.val_accuracy_top5_cifar100 and hasattr(self.val_accuracy_top5_cifar100, 'reset'): self.val_accuracy_top5_cifar100.reset()

    def _get_optimizer(self) -> Optimizer:
        cfg = self.training_config.optimizer
        params_with_wd = []
        params_without_wd = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or '.bias' in name or 'logit_scale' in name or 'norm' in name or '.ln' in name or 'positional_embedding' in name:
                 params_without_wd.append(param)
            else:
                 params_with_wd.append(param)
        param_groups = [
            {"params": params_with_wd, "weight_decay": cfg.weight_decay, "name": "params_with_wd"},
            {"params": params_without_wd, "weight_decay": 0.0, "name": "params_without_wd"},
        ]
        if cfg.name == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=cfg.lr,
                betas=(cfg.beta1, cfg.beta2),
                eps=cfg.eps,
            )
        elif cfg.name == "adam":
             optimizer = torch.optim.Adam(
                param_groups,
                lr=cfg.lr,
                betas=(cfg.beta1, cfg.beta2),
                eps=cfg.eps,
            )
        elif cfg.name == "sgd":
             optimizer = torch.optim.SGD(
                param_groups,
                lr=cfg.lr,
                momentum=cfg.momentum if hasattr(cfg, 'momentum') else 0.9,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.name}")
        return optimizer

    def _get_lr_scheduler(self, optimizer: Optimizer) -> Dict[str, Any]:
        cfg_scheduler = self.training_config.scheduler
        cfg_opt = self.training_config.optimizer
        cfg_data = self.training_config.data
        schedule_config = {"interval": "step", "frequency": 1}
        if self.trainer is None:
            if self.training_config.epochs and self.train_dataset_size > 0:
                effective_batch_size_per_step = cfg_data.batch_size * self.trainer.world_size * self.trainer.accumulate_grad_batches
                steps_per_epoch = math.ceil(self.train_dataset_size / max(1, effective_batch_size_per_step))
                total_steps = steps_per_epoch * self.training_config.epochs
            else:
                total_steps = 1_000_000
        else:
            if self.trainer.max_steps and self.trainer.max_steps != -1:
                total_steps = self.trainer.max_steps
            elif self.trainer.max_epochs and self.trainer.max_epochs != -1:
                limit_batches = self.trainer.limit_train_batches or 1.0
                if isinstance(limit_batches, int):
                     steps_per_epoch = limit_batches
                elif self.trainer.num_training_batches != float('inf'):
                     steps_per_epoch = int(self.trainer.num_training_batches * limit_batches)
                else:
                     steps_per_epoch = max(10000, cfg_scheduler.warmup_steps * 5)
                total_steps = steps_per_epoch * self.trainer.max_epochs
            else:
                total_steps = 1_000_000
        warmup_steps = cfg_scheduler.warmup_steps
        if warmup_steps >= total_steps:
            warmup_steps = max(1, total_steps // 2)
        if cfg_scheduler.name == "cosine":
            min_lr_ratio = getattr(cfg_scheduler, 'min_lr', 0.0) / cfg_opt.lr if cfg_opt.lr > 0 else 0.0
            def lr_lambda(current_step):
                if warmup_steps > 0 and current_step < warmup_steps:
                    return float(current_step) / float(warmup_steps)
                decay_steps = max(1, total_steps - warmup_steps)
                progress = float(current_step - warmup_steps) / float(decay_steps)
                progress = max(0.0, min(1.0, progress))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                scale_factor = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
                return scale_factor
        elif cfg_scheduler.name == "const":
             def lr_lambda(current_step):
                 if warmup_steps > 0 and current_step < warmup_steps:
                    return float(current_step) / float(warmup_steps)
                 return 1.0
        else:
             raise ValueError(f"Unsupported scheduler: {cfg_scheduler.name}")
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        schedule_config["scheduler"] = scheduler
        return schedule_config

    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        scheduler_config = self._get_lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config
        }

    def _build_zero_shot_classifier(self, dataset="cifar10"):
        # Check if already built for this epoch to avoid redundant computation/broadcast
        if dataset in self.classifier_weights and self._classifier_epoch == self.current_epoch:
             # pl.utilities.rank_zero_info(f"Using cached {dataset} classifier weights for epoch {self.current_epoch}.")
             return

        pl.utilities.rank_zero_info(f"Building zero-shot classifier for {dataset} (Epoch {self.current_epoch})...")
        original_mode = self.model.training
        self.model.eval()
        if dataset == "cifar10":
            classes = self.cifar10_classes
            templates = self.cifar10_templates
        elif dataset == "cifar100":
            classes = self.cifar100_classes
            templates = self.cifar100_templates
        else:
            self.model.train(original_mode)
            raise ValueError(f"Unsupported dataset: {dataset}")
        with torch.no_grad():
            all_texts = []
            for class_name in classes:
                class_texts = [template.format(c=class_name) for template in templates]
                all_texts.extend(class_texts)
            try:
                # Tokenize the texts first
                tokenized_output = self.model.tokenize(all_texts)
                # Extract the input_ids tensor
                text_input_ids = tokenized_output['input_ids'].to(self.device)
            except Exception as e:
                self.model.train(original_mode)
                raise e # Re-raise tokenization error
            try:
                # Pass the input_ids tensor to encode_text
                text_features = self.model.encode_text(text_input_ids, normalize=True)
            except Exception as e:
                 self.model.train(original_mode)
                 raise e # Re-raise encoding error
            embed_dim = text_features.shape[-1]
            n_classes = len(classes)
            n_templates = len(templates)
            if text_features.shape[0] == n_classes * n_templates:
                 text_features = text_features.view(n_classes, n_templates, embed_dim)
                 text_features = text_features.mean(dim=1)
            else:
                 if text_features.shape[0] == n_classes:
                     pass
                 else:
                     self.model.train(original_mode)
                     raise ValueError(f"Cannot reconcile text feature shape {text_features.shape} with {n_classes} classes and {n_templates} templates.")
            text_features = F.normalize(text_features, dim=-1)
            classifier_weights = text_features.t().contiguous()
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if is_distributed:
            classifier_weights = classifier_weights.to(self.device)
            torch.distributed.broadcast(classifier_weights, src=0)
        self.classifier_weights[dataset] = classifier_weights
        self.model.train(original_mode)
 
    def on_validation_epoch_start(self):
        # This function is now primarily for building the classifiers if needed
        # The actual check happens within validation_step before using the weights
        if self.train_mode.startswith("yfcc15m"):
             self.model.to(self.device) # Ensure model is on correct device
             # Attempt to build both classifiers if they haven't been built this epoch
             # Errors during build are handled within validation_step where they are used
             if "cifar10" not in self.classifier_weights or self._classifier_epoch != self.current_epoch:
                  try:
                      self._build_zero_shot_classifier(dataset="cifar10")
                      # Set epoch marker after successful build
                      # self._classifier_epoch = self.current_epoch # Set marker here or in validation_step? Let's do it in validation_step after first successful use.
                  except Exception as e:
                       pl.utilities.rank_zero_warn(f"Failed to build CIFAR10 classifier on epoch start: {e}")
             if "cifar100" not in self.classifier_weights or self._classifier_epoch != self.current_epoch:
                 try:
                     self._build_zero_shot_classifier(dataset="cifar100")
                 except Exception as e:
                      pl.utilities.rank_zero_warn(f"Failed to build CIFAR100 classifier on epoch start: {e}")
            # Reset epoch marker at the START of validation epoch
             self._classifier_epoch = self.current_epoch
    
    #for debugging
    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     if self.global_step % 100 == 0 and self.global_rank == 0:
    #         self.print("\n🔍 Gradient check:")
    #         for name, param in self.model.named_parameters():
    #             if param.requires_grad:
    #                 grad_mean = param.grad.mean().item() if param.grad is not None else "None"
    #                 val_mean = param.data.mean().item()
    #                 self.print(f"{name:<50} | mean={val_mean:.5f} | grad_mean={grad_mean}")
