# TODO: move other helper functions here and ensure logging consistency

from src.lightning_module import CLIPLightningModule
import os
import torch
from collections import OrderedDict


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
        print(f"Checkpoint file not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint from: {checkpoint_path}")
    # Load checkpoint on CPU to avoid GPU memory issues if the model is large
    # and ensure compatibility if the checkpoint was saved on a different device.
    try:
        # Try with weights_only=True first for security and efficiency
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"Failed to load checkpoint with weights_only=True: {e}")
        print("Attempting to load checkpoint with weights_only=False (ensure checkpoint is trusted).")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as e_fallback:
            print(f"Failed to load checkpoint with weights_only=False: {e_fallback}")
            return

    state_dict = checkpoint.get("state_dict", checkpoint)

    vision_state_dict = OrderedDict()
    loaded_vision_keys = 0
    skipped_text_keys = 0
    skipped_other_keys = 0

    print(f"\nInspecting checkpoint keys for vision model (prefix: '{vision_prefix}')...")

    for key, value in state_dict.items():
        if key.startswith(vision_prefix):
            new_key = key[len(vision_prefix):]
            vision_state_dict[new_key] = value
            loaded_vision_keys += 1
        elif key.startswith(text_model_prefix):
            skipped_text_keys += 1
        else:
            skipped_other_keys += 1


    if loaded_vision_keys > 0:
        print(f"Found {loaded_vision_keys} keys matching vision prefix '{vision_prefix}'.")
        print(f"Skipped {skipped_text_keys} keys matching text prefix '{text_model_prefix}'.")
        print(f"Skipped {skipped_other_keys} other keys (e.g., optimizer, text model parts if prefix is different).")

        try:
            target_module.model.visual.load_state_dict(vision_state_dict, strict=True)
            print("\nSuccessfully loaded vision weights into the YFCC model's vision branch.")


        except RuntimeError as e:
            print(f"\nError loading vision state_dict: {e}")
            print("This might be due to model architecture mismatch or incorrect key mapping.")
            print("Ensure the vision model architecture in the checkpoint is compatible.")
            print("Target vision model keys:")
            for k in target_module.model.visual.state_dict().keys():
                print(f"  {k}")
            print("Source vision model keys (from checkpoint, after stripping prefix):")
            for k in vision_state_dict.keys():
                print(f"  {k}")

    else:
        print(f"No keys found with the vision prefix '{vision_prefix}'. Vision weights not loaded.")
        print("Please check the checkpoint structure and the prefix.")
        print("Available top-level keys in checkpoint's state_dict:")
        for k in list(state_dict.keys())[:20]: # Print some example keys
            print(f"  {k}")

    print("\nThe text branch of the YFCC model remains randomly initialized.")
