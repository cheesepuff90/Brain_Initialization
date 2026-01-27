import math
import os
from typing import Optional, Tuple, List, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class HingeLoss(nn.Module):
    """
    Hinge loss for contrastive learning with triplets.
    
    Uses a margin-based approach to enforce that the distance between the reference and the
    more similar image is smaller than the distance between the reference and less similar image.
    """
    def __init__(self, margin: float = 0.1):
        """
        Args:
            margin: The margin between positive and negative distances
        """
        super().__init__()
        self.margin = margin
        self.step_count = 0

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Difference in distance (dist0 - dist1)
            targets: Binary targets (1 if dist1 is more similar to reference, 0 otherwise)
            
        Returns:
            loss: Hinge loss
        """
        # Convert targets to -1, 1 for easier hinge loss calculation
        # 1 -> 1, 0 -> -1
        targets = 2 * targets - 1
        
        # Calculate hinge loss
        # If target is 1, dist0 should be larger than dist1 by at least margin
        # If target is -1, dist1 should be larger than dist0 by at least margin
        individual_losses = F.relu(self.margin - targets * logits)
        loss = torch.mean(individual_losses)
        
        
        
        # Print training info every 50 steps
        if self.step_count % 50 == 0:
            print(f"\n=== HingeLoss (Step {self.step_count}) ===")
            n_samples = min(3, logits.shape[0])
            for i in range(n_samples):
                print(f"Sample {i}: logits={logits[i].item():.4f}, target={targets[i].item()}, loss={individual_losses[i].item():.4f}")
            print(f"Average Loss: {loss.item():.4f}")
        
        self.step_count += 1
        
        return loss


class TripletContrastiveLoss(nn.Module):
    """
    Contrastive loss for triplets in the NIGHTS dataset.

    Measures the difference in similarity between (ref, dist0) and (ref, dist1)
    using the specified distance metric and enforces margins based on human preferences.
    """
    def __init__(
        self,
        margin: float = 0.1,
        distance_type: str = "cosine",
        scale: float = 1.0
    ):
        """
        Args:
            margin: Margin for contrastive learning
            distance_type: Type of distance to use (cosine or euclidean)
            scale: Scale factor for cosine similarity
        """
        super().__init__()
        self.margin = margin
        self.distance_type = distance_type
        self.scale = scale
        
        # Use the hinge loss implementation
        self.hinge_loss = HingeLoss(margin=margin)
        self.step_count = 0

    def compute_distance(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Compute the distance between two feature vectors
        
        Args:
            feat1: First feature vector
            feat2: Second feature vector
            
        Returns:
            distance: Distance between features
        """
        if self.distance_type == "cosine":
            # Normalize features for cosine similarity
            feat1 = F.normalize(feat1, dim=-1)
            feat2 = F.normalize(feat2, dim=-1)
            
            # 1 - cosine similarity = 2 * (1 - (feat1 · feat2)) / 2
            # Higher value means more distance
            similarity = torch.sum(feat1 * feat2, dim=-1)
            distance = 1.0 - similarity
            
            # Scale the distance
            distance = distance * self.scale
        elif self.distance_type == "euclidean":
            # Euclidean distance
            distance = torch.sqrt(torch.sum((feat1 - feat2) ** 2, dim=-1) + 1e-8)
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")
        
        return distance

    def forward(
        self,
        ref_features: torch.Tensor,
        dist0_features: torch.Tensor,
        dist1_features: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive loss for triplets.
        
        Args:
            ref_features: Features for reference images
            dist0_features: Features for first distorted images
            dist1_features: Features for second distorted images
            targets: Binary targets (1 if dist1 is more similar to reference, 0 otherwise)
            
        Returns:
            Dictionary with loss and distances
        """
        # Compute distances
        dist0 = self.compute_distance(ref_features, dist0_features)
        dist1 = self.compute_distance(ref_features, dist1_features)
        
        # Compute logits (difference between distances)
        # Positive logit means dist0 > dist1, which suggests dist1 is more similar to reference
        logits = dist0 - dist1
        
        # Compute loss
        loss = self.hinge_loss(logits, targets)
        
        
         # Print training info every 50 steps
        if self.step_count % 50 == 0:
            print(f"\n=== TripletContrastiveLoss (Step {self.step_count}) ===")
            n_samples = min(3, logits.shape[0])
            for i in range(n_samples):
                target_desc = 'dist1 more similar' if targets[i].item() == 1 else 'dist0 more similar'
                pred_desc = 'dist1 more similar' if logits[i].item() > 0 else 'dist0 more similar'
                correct = "✓" if (logits[i].item() > 0) == (targets[i].item() == 1) else "✗"
                print(f"Sample {i}: dist0={dist0[i].item():.4f}, dist1={dist1[i].item():.4f}")
                print(f"         logits={logits[i].item():.4f}, target={target_desc}, pred={pred_desc} {correct}")
            print(f"Average Loss: {loss.item():.4f}")
        
        self.step_count += 1
        
        
        # Return loss and distances
        return {
            "loss": loss,
            "dist0": dist0,
            "dist1": dist1,
            "logits": logits
        }


class GatherFeatures(torch.autograd.Function):
    """
    Gather tensors from all processes, supporting differentiation.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.batch_size = x.shape[0]
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        # Reduce grads to the correct device/rank
        # Clip gradient based on batch size
        grad_size = grads[0].shape[0]
        # Ensure gradients are scaled correctly based on original batch size
        # Original code uses torch.distributed.get_rank() but PTL manages ranks
        local_rank = dist.get_rank()
        return all_gradients[local_rank] * ctx.batch_size / grad_size


gather_features = GatherFeatures.apply


class ClipLoss(nn.Module):
    """
    CLIP loss (InfoNCE) implementation.
    Handles local and distributed training.
    """
    def __init__(
        self,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        rank: int = 0, # Set by Lightning automatically
        world_size: int = 1, # Set by Lightning automatically
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # Cache for labels
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # Calculated labels
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor, # This is the learned parameter from the model
        output_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute CLIP InfoNCE loss.

        Args:
            image_features: Features for images [local_batch_size, feat_dim]
            text_features: Features for text [local_batch_size, feat_dim]
            logit_scale: Learned temperature parameter (scalar tensor)
            output_dict: Whether to return loss components as a dict

        Returns:
            Loss tensor or dict containing loss components.
        """
        device = image_features.device

        # --- Feature Gathering for Distributed Training ---
        if self.world_size > 1:
            # Gather features from all GPUs
            if self.gather_with_grad:
                # Allows backprop through gathered features
                all_image_features = gather_features(image_features)
                all_text_features = gather_features(text_features)
                if not self.local_loss:
                    # Concatenate features if not using local loss calculation
                    image_features = torch.cat(all_image_features, dim=0)
                    text_features = torch.cat(all_text_features, dim=0)
            else:
                 # Gather features without backprop support (more efficient)
                all_image_features = [torch.zeros_like(image_features) for _ in range(self.world_size)]
                all_text_features = [torch.zeros_like(text_features) for _ in range(self.world_size)]
                dist.all_gather(all_image_features, image_features)
                dist.all_gather(all_text_features, text_features)
                if not self.local_loss:
                    image_features = torch.cat(all_image_features, dim=0)
                    text_features = torch.cat(all_text_features, dim=0)

        # --- Logits Calculation ---
        # Features are assumed to be normalized already by the model encode methods
        # logit_scale is learnable parameter, apply exp()
        # Clamp logit scale to prevent instability (from OpenCLIP)
        logit_scale = torch.clamp(logit_scale.exp(), max=100)

        if self.local_loss:
            # Calculate local logits only
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        else:
             # Calculate global logits (all features vs all features)
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logits_per_image.T # More efficient

        # --- Loss Calculation ---
        num_logits = logits_per_image.shape[0]
        labels = self.get_ground_truth(device, num_logits)

        loss_img = F.cross_entropy(logits_per_image, labels, reduction='mean')
        loss_txt = F.cross_entropy(logits_per_text, labels, reduction='mean')

        # Average the image and text loss
        loss = (loss_img + loss_txt) / 2
        
        if output_dict:
            return {
                "contrastive_loss": loss,
                "loss_img": loss_img,
                "loss_txt": loss_txt,
                "logits_per_image": logits_per_image,
                "logits_per_text": logits_per_text,
                "logit_scale": logit_scale # Return the processed scale
            }
        else:
            return loss 
