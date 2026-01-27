import torchvision.datasets as tv_datasets
import os
import torch 

# --- Define classes from your lightning_module.py ---
# Copied directly from src/lightning_module.py
implemented_cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
implemented_cifar10_templates = ["a photo of a {c}."] # Included for context, but not needed for class name check

implemented_cifar100_classes = (
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
implemented_cifar100_templates = ["a photo of a {c}."]

def compare_classes(dataset_name, implemented_classes, torchvision_classes):
    """Helper function to compare and print class list differences."""
    print(f"\n--- Comparison for {dataset_name} ---")
    print(f"Implementation classes (first 10): {list(implemented_classes[:10])}...")
    print(f"Torchvision classes    (first 10): {list(torchvision_classes[:10])}...")
    
    # Check if they are identical (same elements in the same order)
    are_identical = (list(implemented_classes) == list(torchvision_classes))

    if are_identical:
        print(f"\n✅ SUCCESS: The {dataset_name} class names and order match exactly.")
        return True
    else:
        print(f"\n❌ MISMATCH: The {dataset_name} class names or order do not match.")
        if len(implemented_classes) != len(torchvision_classes):
            print(f"  - Length differs: Implementation ({len(implemented_classes)}) vs Torchvision ({len(torchvision_classes)})")
        
        # Find first mismatch index
        first_mismatch_idx = -1
        for i, (impl_cls, tv_cls) in enumerate(zip(list(implemented_classes), list(torchvision_classes))):
            if impl_cls != tv_cls:
                first_mismatch_idx = i
                break
        
        if first_mismatch_idx != -1:
             print(f"  - First difference at index {first_mismatch_idx}: Implementation='{implemented_classes[first_mismatch_idx]}', Torchvision='{torchvision_classes[first_mismatch_idx]}'")

        # Check if sets are the same (ignoring order)
        if set(implemented_classes) == set(torchvision_classes):
             print("  - Note: The sets of class names are the same, but the order is different.")
        else:
             print("  - Note: The sets of class names are also different.")
        return False

# --- Setup Dummy Root ---
dummy_root_base = os.path.join(os.getcwd(), 'temp_cifar_check')
os.makedirs(dummy_root_base, exist_ok=True)

# --- Check CIFAR10 ---
cifar10_success = False
print("\n====================")
print("Checking CIFAR10...")
print("====================")
dummy_root_cifar10 = '/tmp/dataset_eval'
try:
    print("Attempting to load CIFAR10 dataset info from torchvision...")
    cifar10_dataset_info = tv_datasets.CIFAR10(root=dummy_root_cifar10, train=False, download=True) 
    torchvision_cifar10_classes = cifar10_dataset_info.classes
    print("Successfully retrieved CIFAR10 classes from torchvision.")
    cifar10_success = compare_classes("CIFAR10", implemented_cifar10_classes, torchvision_cifar10_classes)
except Exception as e:
    print(f"\n❌ ERROR: Failed to instantiate or access CIFAR10 dataset info: {e}")

# --- Check CIFAR100 ---
cifar100_success = False
print("\n=====================")
print("Checking CIFAR100...")
print("=====================")
dummy_root_cifar100 = '/tmp/dataset_eval'
try:
    print("Attempting to load CIFAR100 dataset info from torchvision...")
    # Use CIFAR100 class
    cifar100_dataset_info = tv_datasets.CIFAR100(root=dummy_root_cifar100, train=False, download=True) 
    torchvision_cifar100_classes = cifar100_dataset_info.classes
    print("Successfully retrieved CIFAR100 classes from torchvision.")
    cifar100_success = compare_classes("CIFAR100", implemented_cifar100_classes, torchvision_cifar100_classes)
except Exception as e:
    print(f"\n❌ ERROR: Failed to instantiate or access CIFAR100 dataset info: {e}")

# --- Final Summary ---
print("\n====================")
print("      SUMMARY")
print("====================")
print(f"CIFAR10 Check:  {'✅ MATCH' if cifar10_success else '❌ MISMATCH'}")
print(f"CIFAR100 Check: {'✅ MATCH' if cifar100_success else '❌ MISMATCH'}")

# Optional Cleanup: Uncomment if you want to remove the temp directories
# finally:
#     import shutil
#     if os.path.exists(dummy_root_base):
#         print(f"\nCleaning up temporary directory: {dummy_root_base}")
#         shutil.rmtree(dummy_root_base)
