import os
import sys
import torch
from torchvision import transforms
from datasets import load_dataset, Features, ClassLabel, Image as HFImage, Dataset
from PIL import Image

# Adjust Python path to import project modules
# Assuming 'tests/' is a subdirectory of the project root, and 'config.py' is in the root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from config import VIT_B_32_CONFIG
except ImportError as e:
    print(f"Error importing VIT_B_32_CONFIG from config.py: {e}")
    print("Please ensure 'config.py' is in the project root and VIT_B_32_CONFIG is defined there.")
    sys.exit(1)

# --- Configuration (some parts might be duplicated from test_linearprob.py or a shared config) ---
DATASET_CONFIG = {
    "cifar10": {
        "hf_name": "cifar10",
        "split_train": "train",
        "split_test": "test",
        "image_key": "img",
        "label_key": "label",
        "num_classes": 10
    },
    "cifar100": {
        "hf_name": "cifar100",
        "split_train": "train",
        "split_test": "test",
        "image_key": "img",
        "label_key": "fine_label", # Use fine_label for 100 classes
        "num_classes": 100
    }
}

CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

PROCESSED_DATA_DIR = "./processed_datasets" # Directory to save processed datasets

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

def get_hf_dataset_raw(dataset_config_entry, split_name: str) -> Dataset:
    """Loads a raw Hugging Face dataset."""
    hf_name = dataset_config_entry["hf_name"]
    ds_features = None
    if hf_name == "cifar10":
        ds_features = Features({'img': HFImage(), 'label': ClassLabel(num_classes=dataset_config_entry["num_classes"])})
    elif hf_name == "cifar100":
        ds_features = Features({
            'img': HFImage(),
            'fine_label': ClassLabel(num_classes=dataset_config_entry["num_classes"]), # This should be 'fine_label'
            'coarse_label': ClassLabel(num_classes=20)
        })
    
    dataset = load_dataset(hf_name, split=split_name, features=ds_features, trust_remote_code=True)
    return dataset

def main_preprocess():
    print("Starting dataset preprocessing script.")

    try:
        image_size = VIT_B_32_CONFIG.vision_cfg.image_size
    except AttributeError:
        print("Error: VIT_B_32_CONFIG.vision_cfg.image_size not found.")
        print("Defaulting image_size to 224. This might be incorrect for your model.")
        image_size = 224
    
    image_transform_global = get_image_transform(image_size)
    print(f"Using global image size {image_size}x{image_size} for dataset transformations.")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    print(f"Processed datasets will be saved to: {PROCESSED_DATA_DIR}")

    for ds_name, ds_config_entry in DATASET_CONFIG.items():
        print(f"\nProcessing dataset: {ds_name}")
        current_image_key = ds_config_entry["image_key"]
        current_label_key = ds_config_entry["label_key"]

        def map_transform_batch_fn(examples_batch: dict):
            pixel_values_batch = [image_transform_global(img) for img in examples_batch[current_image_key]]
            # The map function will carry over other columns (like labels) by default
            return {'pixel_values': pixel_values_batch}

        for split_type in ["train", "test"]:
            split_name_hf = ds_config_entry[f"split_{split_type}"]
            print(f"  Loading raw '{split_name_hf}' split for {ds_name}...")
            try:
                raw_ds = get_hf_dataset_raw(ds_config_entry, split_name_hf)
                
                print(f"  Applying map transform to '{split_name_hf}' split for {ds_name}...")
                processed_ds = raw_ds.map(
                    map_transform_batch_fn,
                    batched=True,
                    num_proc=16,
                    remove_columns=[current_image_key] # Remove original image column
                )
                
                # Ensure the output columns include 'pixel_values' and the correct label key
                processed_ds.set_format(type='torch', columns=['pixel_values', current_label_key])
                
                save_path = os.path.join(PROCESSED_DATA_DIR, ds_name, split_type)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print(f"  Saving processed {split_type} split of {ds_name} to {save_path}...")
                processed_ds.save_to_disk(save_path)
                print(f"  Successfully saved to {save_path}.")

            except Exception as e:
                print(f"  FAILED to process and save '{split_name_hf}' split for {ds_name}. Error: {e}")
                import traceback
                traceback.print_exc()
                continue 

    print("\nDataset preprocessing script finished.")

if __name__ == "__main__":
    main_preprocess()