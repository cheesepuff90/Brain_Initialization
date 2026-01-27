import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from PIL import Image
import torch # Needed for tensor conversion in dataset if any
from io import BytesIO  # Add this import

# --- Configuration ---
DATASET_NAME = "local/YFCC15M"
TOKENIZER_NAME = "openai/clip-vit-base-patch32" # Standard CLIP tokenizer
NUM_EXAMPLES_TO_SHOW = 5
# ---------------------

# Load the dataset in streaming mode
try:
    ds = load_dataset('parquet', split="train", streaming=True, data_files="/tmp/hub/datasets--Kaichengalex--YFCC15M/snapshots/fc79df6bae9cbceac054a5d2598c168dfd8fe39b/data/train-*.parquet")
    print(f"Successfully loaded streaming dataset '{DATASET_NAME}'.")
except Exception as e:
    print(f"Error loading dataset '{DATASET_NAME}': {e}")
    exit(1)

# Load the tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Successfully loaded tokenizer '{TOKENIZER_NAME}'.")
except Exception as e:
    print(f"Error loading tokenizer '{TOKENIZER_NAME}': {e}")
    exit(1)

print("\n--- First {} Examples ---".format(NUM_EXAMPLES_TO_SHOW))

# Process and display the first N examples
for i, example in enumerate(ds):
    if i >= NUM_EXAMPLES_TO_SHOW:
        break
    print(f"\n--- Example {i+1} ---")

    try:
        # --- Image ---
        image_data = example.get('images')
        if image_data is None:
            print("Image: Not available")
            continue # Skip if image is missing
        
        # Check if image is in bytes format
        if isinstance(image_data, dict) and 'bytes' in image_data:
            image = Image.open(BytesIO(image_data['bytes']))
        elif isinstance(image_data, bytes):
            image = Image.open(BytesIO(image_data))
        else:
            image = image_data  # Already a PIL image
            
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # --- Text ---
        token_ids_raw = example.get('texts')
        if token_ids_raw is None:
            print("Text: Not available")
            decoded_text = "N/A"
        else:
            # Convert to list of integers if necessary (handles list of floats/tensors)
            try:
                # Filter out potential non-numeric if needed, convert valid ones to int
                token_ids = [int(t) for t in token_ids_raw if isinstance(t, (int, float)) and not torch.isnan(torch.tensor(t) if isinstance(t, float) else t)]
                if not token_ids:
                     print("Text: Empty or invalid token list after filtering.")
                     decoded_text = "[Empty/Invalid Tokens]"
                else:
                    # Decode the token IDs back to text
                    # skip_special_tokens=True removes padding, BOS, EOS etc.
                    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
                    print(f"Decoded Text: {decoded_text}")
            except Exception as decode_err:
                 print(f"Error decoding text tokens: {decode_err}")
                 print(f"Raw tokens: {token_ids_raw}")
                 decoded_text = "[Decoding Error]"


        # --- Display ---
        fig, ax = plt.subplots(1, 1, figsize=(6, 7)) # Adjust size as needed
        ax.imshow(image)
        ax.set_title(f"Example {i+1}\n{decoded_text}", wrap=True, fontsize=10)
        ax.axis('off') # Hide axes ticks
        plt.tight_layout()
        plt.savefig(f"example_{i+1}.png")
        plt.close()

    except Exception as e:
        print(f"Error processing or displaying example {i+1}: {e}")
        # You might want to inspect the 'example' structure here if errors occur
        # print(f"Problematic example structure: {example}")


print(f"\nDisplayed first {NUM_EXAMPLES_TO_SHOW} examples.")
