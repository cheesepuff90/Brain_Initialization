import pytest
import torch
from PIL import Image
import numpy as np
import time # Import time for potential delays if needed

# Assume your dataset module is in src
from src.dataset import YFCC15MDataset, get_transform
from transformers import AutoTokenizer

# --- Tokenizer Setup ---
# Using a real tokenizer for realistic testing
TOKENIZER_NAME = "bert-base-uncased" # Or use the one specified in your main config
CONTEXT_LENGTH = 77 # Standard CLIP context length

try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
except Exception as e:
    print(f"Warning: Failed to load tokenizer {TOKENIZER_NAME}. Tests might fail. {e}")
    tokenizer = None # Allow tests to proceed but likely fail downstream

def tokenize_text(text):
    if tokenizer is None:
        raise RuntimeError("Tokenizer failed to load.")
    # Ensure text is a string before tokenizing
    text_to_tokenize = text if isinstance(text, str) else ""
    result = tokenizer(
        text_to_tokenize,
        max_length=CONTEXT_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    return result['input_ids'][0] # Shape: [context_length]

# --- Test Function ---

# Removed @patch decorator
def test_yfcc15m_dataset_real_iteration():
    """
    Tests the YFCC15MDataset iterator with the actual Kaichengalex/YFCC15M
    dataset stream, fetching a small number of items.
    """
    # Removed mock configuration

    # Use standard transform
    image_size = 224
    transform = get_transform(image_size=image_size, is_train=False) # Use eval transform for simplicity

    # Instantiate the dataset with the actual HF dataset name
    dataset = YFCC15MDataset(
        hf_dataset_name="Kaichengalex/YFCC15M", # Using the actual dataset name
        split="train", # Assuming 'train' split exists for this dataset
        transform=transform,
        tokenizer=tokenize_text,
        image_col="image", # Keep suspected image column name
        text_col="text", # *** IMPORTANT: Update text column name if different for this dataset ***
                      # Common names: 'text', 'caption'. Check the dataset viewer on Hugging Face.
        image_size=image_size,
        interpolation="bicubic", # Added interpolation explicitly
        # cache_dir= # Optional: Specify a cache dir if needed
    )

    # Iterate through a small number of items from the real dataset stream
    outputs = []
    items_to_fetch = 5 # Fetch only a few items to keep the test quick
    items_processed = 0
    timeout_seconds = 120 # Add a timeout to prevent hanging indefinitely
    start_time = time.time()

    print(f"\nAttempting to fetch {items_to_fetch} items from Kaichengalex/YFCC15M stream...")
    try:
        for item in dataset:
            if time.time() - start_time > timeout_seconds:
                print("Warning: Test timed out while fetching items.")
                break
            if items_processed >= items_to_fetch:
                print(f"Successfully fetched {items_to_fetch} items.")
                break

            # Check if the item is valid (basic check, might need more robust checks)
            if item is not None and len(item) == 2:
                 outputs.append(item)
                 items_processed += 1
                 print(f"Fetched item {items_processed}/{items_to_fetch}")
            else:
                 print("Skipped an invalid item from the stream.")
                 # Continue fetching to reach the target number of *valid* items

    except Exception as e:
        pytest.fail(f"Error occurred during dataset iteration: {e}")

    # --- Assertions ---
    # 1. Check that *at least one* valid item was yielded (or the exact number if reliable)
    # Using >= 1 is safer for real-world streams where initial items might be invalid
    assert len(outputs) > 0, \
        f"Expected to fetch at least 1 valid item, but got {len(outputs)}. Check dataset stream/availability/column names."
    assert len(outputs) <= items_to_fetch, \
        f"Fetched more items ({len(outputs)}) than requested ({items_to_fetch})."


    print(f"Successfully yielded {len(outputs)} valid items.")

    # 2. Check the structure and type/shape of the yielded items
    for i, (img_tensor, text_tokens) in enumerate(outputs):
        print(f"Checking structure of yielded item {i+1}...")
        assert isinstance(img_tensor, torch.Tensor), f"Item {i+1}: Image output should be a Tensor, got {type(img_tensor)}"
        assert img_tensor.shape == (3, image_size, image_size), \
            f"Item {i+1}: Image tensor shape mismatch: expected (3, {image_size}, {image_size}), got {img_tensor.shape}"

        assert isinstance(text_tokens, torch.Tensor), f"Item {i+1}: Text output should be a Tensor, got {type(text_tokens)}"
        assert text_tokens.shape == (CONTEXT_LENGTH,), \
            f"Item {i+1}: Text tokens shape mismatch: expected ({CONTEXT_LENGTH},), got {text_tokens.shape}"
        assert text_tokens.dtype == torch.long, f"Item {i+1}: Text tokens should be LongTensor, got {text_tokens.dtype}"

    print("All yielded items have the correct structure and types.")

if __name__ == "__main__":
    test_yfcc15m_dataset_real_iteration()