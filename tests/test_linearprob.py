import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Adjust Python path to import project modules
# Assuming 'tests/' is a subdirectory of the project root, and 'src/' and 'config.py' are in the root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os, sys
# Put your real token string in an env var or read it from file
token_path = os.path.expanduser("~/.cache/huggingface/token")
with open(token_path) as f:
    hf_token = f.read().strip()

os.environ["HF_HOME"] = "/tmp/huggingface"          # keep fast cache
os.environ["HF_HUB_TOKEN"] = hf_token  

# Set Hugging Face cache directory
os.environ['HF_HOME'] = '/tmp/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface/models'
os.environ['HF_HUB_CACHE'] = '/tmp/huggingface/hub'

# Ensure the cache directories exist
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)
os.makedirs(os.environ['HF_HUB_CACHE'], exist_ok=True)


from datasets import load_dataset, Features, ClassLabel, Image as HFImage
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm


try:
    from src.lightning_module import CLIPLightningModule
    from config import VIT_B_32_CONFIG
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure 'src' and 'config.py' are in the project root and accessible.")
    print("PYTHONPATH might need to be set, or the script run from the project root directory.")
    sys.exit(1)


# --- Configuration ---
CHECKPOINT_PATHS = [
    #"/tmp/checkpoints/pretrain_checkpoints/nights_ft_from_yfcc_20250512_231704_ddp_nodeNone/checkpoints/last.ckpt",
    #"/dev/shm/checkpoints/yfcc15m_litdata_20250506_013343_ddp_nodeNone/checkpoints/monitor_val_cifar100_epoch_acc1-epoch_epoch=031.ckpt",
    "/dev/shm/checkpoints/nights_init_yfcc15m_litdata_20250507_202423_ddp_nodeNone/checkpoints/monitor_val_cifar100_epoch_acc1-epoch_epoch=031.ckpt"
]

DATASET_CONFIG = {
    "cifar10": {
        "hf_name": "cifar10",
        "num_classes": 10,
        "split_train": "train",
        "split_test": "test",
        "image_key": "img",
        "label_key": "label"
    },
    "cifar100": {
        "hf_name": "cifar100",
        "num_classes": 100,
        "split_train": "train",
        "split_test": "test",
        "image_key": "img",
        "label_key": "fine_label" # Use fine_label for 100 classes
    },
    #"imagenet1k": {
    #    "hf_name": "imagenet-1k",
    #    "num_classes": 1000,
    #    "split_train": "train",
    #    "split_test": "validation",
    #    "image_key": "image",
    #    "label_key": "label"
    #}
}
# CLIP Preprocessing (standard values)
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

BATCH_SIZE = 4096  # Batch size for feature extraction
# LOGREG_C = 0.316   # Regularization strength for Logistic Regression (scikit-learn)
# LOGREG_MAX_ITER = 1000 # Max iterations for Logistic Regression solver (scikit-learn)
NUM_WORKERS = 16 # Number of workers for DataLoader

# PyTorch Logistic Regression parameters
PT_LR_EPOCHS = 1000       # Number of epochs for PyTorch LR (similar to max_iter)
PT_LR_LEARNING_RATE = 0.01 # Learning rate for PyTorch LR (Adam optimizer)
PT_LR_WEIGHT_DECAY = 0.0  # Weight decay (L2 penalty) for PyTorch LR.
                          # Scikit-learn's C=0.316 implies a lambda of ~3.16.
                          # Start with 0.0 and tune if necessary.

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

def load_clip_model(ckpt_path, model_config, device):
    """Loads the CLIP model from a checkpoint."""
    print(f"Loading CLIP model from checkpoint: {ckpt_path}")
    lightning_model = CLIPLightningModule.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        model_config=model_config,
        strict=False  # Allow unexpected keys (e.g., HF text-model weights)
    )
    lightning_model = lightning_model.to(device)
    lightning_model.eval()
    return lightning_model.model # Return the core OpenCLIP model

def get_hf_dataset(dataset_config_entry, image_transform, split):
    """Loads and preprocesses a Hugging Face dataset."""
    hf_name = dataset_config_entry["hf_name"]
    image_key = dataset_config_entry["image_key"]
    label_key = dataset_config_entry["label_key"]

    # Define features for datasets like CIFAR to ensure correct loading
    if hf_name == "cifar10":
        ds_features = Features({'img': HFImage(), 'label': ClassLabel(num_classes=10)})
    elif hf_name == "cifar100":
        ds_features = Features({'img': HFImage(), 'fine_label': ClassLabel(num_classes=100), 'coarse_label': ClassLabel(num_classes=20)})
    elif hf_name == "imagenet-1k":
        ds_features = Features({'image': HFImage(), 'label': ClassLabel(num_classes=1000)})
    else:
        ds_features = None # Let datasets library infer for others

    # Add use_auth_token=True for gated datasets like ImageNet-1k
    # Also reduce or remove num_proc to avoid parallel download issues
    if hf_name == "imagenet-1k":
        dataset = load_dataset(hf_name, split=split,
                             trust_remote_code=True, token=hf_token, 
                             num_proc=4)  # Reduce workers from 16 to 4
    else:
        dataset = load_dataset(hf_name, split=split, features=ds_features, 
                             token=hf_token, num_proc=16)

    def transform_batch(batch):
        batch['pixel_values'] = [image_transform(image.convert("RGB")) for image in batch[image_key]]
        return batch
    if not hf_name == "imagenet-1k":
        dataset = dataset.map(transform_batch, batched=True, remove_columns=[image_key])
        dataset.set_format(type='torch', columns=['pixel_values', label_key])
    else:
        dataset.set_format(type='torch', columns=['image', label_key])
    return dataset

@torch.no_grad()
def extract_features(clip_model, dataloader, device, label_key_name, hf_name, image_transform):
    """Extracts features and labels using the CLIP model, returns CPU tensors."""
    all_features_cpu = []
    all_labels_cpu = []
    for batch in tqdm(dataloader, desc="Extracting features"):
        if not hf_name == "imagenet-1k":
            images = batch['pixel_values'].to(device)
        else:
            images = image_transform(batch['image']).to(device)
        labels = batch[label_key_name] # This will be a tensor of labels

        # Use normalize=True for L2 normalization of features, standard in CLIP
        features = clip_model.encode_image(images, normalize=True)
        all_features_cpu.append(features.cpu())
        all_labels_cpu.append(labels.cpu()) # labels are already tensors, ensure they are on CPU

    return torch.cat(all_features_cpu), torch.cat(all_labels_cpu)

def evaluate_linear_probe(train_features, train_labels, test_features, test_labels,
                          C=0.316, max_iter=1000, random_state=42): # Default C and max_iter kept for reference
    """Trains a scikit-learn logistic regression model and evaluates it."""
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver='liblinear', # Good for this scale of data
        tol=1e-6
    )
    print(f"  Training scikit-learn Logistic Regression (C={C}, max_iter={max_iter})...")
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy * 100 # Return as percentage

def evaluate_linear_probe_pytorch(train_features_tensor_cpu, train_labels_tensor_cpu,
                                 test_features_tensor_cpu, test_labels_tensor_cpu,
                                 num_classes, device,
                                 epochs=100, lr=0.01, weight_decay=0.0):
    """Trains a logistic regression model using PyTorch and evaluates it.
    Assumes input tensors are on the CPU.
    """
    print(f"  Training PyTorch Logistic Regression (epochs={epochs}, lr={lr}, weight_decay={weight_decay})...")

    # Move tensors to the target device
    train_features = train_features_tensor_cpu.float().to(device)
    # Ensure train_labels are long type for CrossEntropyLoss
    train_labels = train_labels_tensor_cpu.long().to(device)
    test_features = test_features_tensor_cpu.float().to(device)
    # test_labels_tensor_cpu will be converted to numpy only for accuracy_score

    # Define the model (a simple linear layer)
    input_dim = train_features.shape[1]
    model = torch.nn.Linear(input_dim, num_classes).to(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_features)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        # Print loss periodically (e.g., 10 times during training or every epoch if few epochs)
        if (epoch + 1) % (max(1, epochs // 10)) == 0 or epoch == epochs -1:
            print(f"    Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_features)
        _, predicted_labels_tensor = torch.max(test_outputs, 1)
        predicted_labels_np = predicted_labels_tensor.cpu().numpy()

    # Convert test_labels_tensor_cpu to numpy for accuracy_score
    test_labels_np = test_labels_tensor_cpu.numpy()
    accuracy = accuracy_score(test_labels_np, predicted_labels_np)
    return accuracy * 100 # Return as percentage

# --- Main Script ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        image_size = VIT_B_32_CONFIG.vision_cfg.image_size
    except AttributeError:
        print("Error: VIT_B_32_CONFIG.vision_cfg.image_size not found.")
        print("Please ensure 'config.py' and VIT_B_32_CONFIG are correctly defined.")
        print("Defaulting image_size to 224. This might be incorrect for your model.")
        image_size = 224 # Fallback

    image_transform = get_image_transform(image_size)
    print(f"Using image size: {image_size}x{image_size}")

    results_summary = []

    for ckpt_path in CHECKPOINT_PATHS:
        print(f"\n--- Processing checkpoint: {os.path.basename(ckpt_path)} ---")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}. Skipping.")
            continue

        try:
            # Assuming all checkpoints use VIT_B_32_CONFIG for model architecture
            clip_model = load_clip_model(ckpt_path, VIT_B_32_CONFIG, device)
        except Exception as e:
            print(f"Failed to load checkpoint {ckpt_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

        for ds_name, ds_info in DATASET_CONFIG.items():
            print(f"\n  -- Testing on dataset: {ds_name} --")
            current_ds_config = DATASET_CONFIG[ds_name]

            # Load data
            print(f"    Loading {ds_name} data...")
            try:
                train_dataset = get_hf_dataset(current_ds_config, image_transform, current_ds_config["split_train"])
                test_dataset = get_hf_dataset(current_ds_config, image_transform, current_ds_config["split_test"])

                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True if device.type == 'cuda' else False)
                test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True if device.type == 'cuda' else False)
            except Exception as e:
                print(f"    Failed to load data for {ds_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Extract features
            print(f"    Extracting features for {ds_name} train set...")
            train_features, train_labels = extract_features(clip_model, train_loader, device, current_ds_config["label_key"], current_ds_config["hf_name"], image_transform)
            print(f"    Extracted {train_features.shape[0]} train features of dimension {train_features.shape[1]}.")

            print(f"    Extracting features for {ds_name} test set...")
            test_features, test_labels = extract_features(clip_model, test_loader, device, current_ds_config["label_key"], current_ds_config["hf_name"], image_transform)
            print(f"    Extracted {test_features.shape[0]} test features of dimension {test_features.shape[1]}.")

            # Perform linear probe
            print(f"    Performing linear probe for {ds_name}...")
            # accuracy = evaluate_linear_probe(
            #     train_features, train_labels,
            #     test_features, test_labels,
            #     C=LOGREG_C, max_iter=LOGREG_MAX_ITER
            # )
            accuracy = evaluate_linear_probe_pytorch(
                train_features, train_labels,
                test_features, test_labels,
                num_classes=current_ds_config["num_classes"],
                device=device,
                epochs=PT_LR_EPOCHS,
                lr=PT_LR_LEARNING_RATE,
                weight_decay=PT_LR_WEIGHT_DECAY
            )
            print(f"    Accuracy on {ds_name}: {accuracy:3f}%")
            results_summary.append({
                "checkpoint": os.path.basename(ckpt_path),
                "dataset": ds_name,
                "accuracy": accuracy
            })

    print("\n--- Summary of Results ---")
    if results_summary:
        for res in results_summary:
            print(f"Checkpoint: {res['checkpoint']}, Dataset: {res['dataset']}, Accuracy: {res['accuracy']:3f}%")
    else:
        print("No results to display. Check for errors during processing.")

if __name__ == "__main__":
    main()
