import os
import random
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Callable, Tuple, Dict, List, Union
from io import BytesIO
import time # For indexing check
from functools import partial
# Set TOKENIZERS_PARALLELISM to avoid deadlocks with forked processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler, IterableDataset
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageFile, UnidentifiedImageError
import hashlib
from transformers import AutoTokenizer # Hugging Face tokenizer
import torchvision.datasets as tv_datasets # Add torchvision import
import warnings

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- LitData Imports ---
try:
    import litdata as ld
    from litdata.streaming.item_loader import ParquetLoader
    LITDATA_AVAILABLE = True
except ImportError:
    LITDATA_AVAILABLE = False
    ParquetLoader = None # Define dummy for type hints

# --- Text Augmentation Import ---
# Removed EDA import block

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Suppress specific PIL warnings about large images
warnings.filterwarnings("ignore", "(Possibly )?DecompressionBombWarning")

logger = logging.getLogger(__name__)

def seed_worker(worker_id):
    """
    Worker seed function for DataLoader to ensure reproducibility in distributed training
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TripletTransform:
    """
    Apply same transform to all three images in a triplet
    """
    def __init__(self, transform: Callable):
        self.transform = transform

    def __call__(self, imgs: List[Image.Image]) -> List[torch.Tensor]:
        return [self.transform(img) for img in imgs]


# --- Gaussian Blur Implementation (from DeCLIP/MoCo) ---
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
# --- End Gaussian Blur ---

def get_transform(image_size: int = 224, is_train: bool = True, interpolation: str = "bicubic"):
    """
    Get the image transform pipeline.
    Uses MOCOv2 augmentations for training, standard CLIP eval transforms otherwise.
    
    Args:
        image_size: Size to resize image to
        is_train: Whether to include data augmentation for training
        interpolation: Interpolation method for resizing (used for eval only)
    """
    interpolation_map = {
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "bilinear": transforms.InterpolationMode.BILINEAR,
        "nearest": transforms.InterpolationMode.NEAREST
    }
    interpolation_mode = interpolation_map.get(interpolation, transforms.InterpolationMode.BICUBIC)
    
    # Normalize with OpenAI CLIP mean and std (as used in the rest of the project)
    normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
    
    if is_train:
        # MOCO V2 style augmentations (closer to DeCLIP)
        # Reference: https://github.com/facebookresearch/moco/blob/main/main_moco.py
        # DeCLIP config: prototype/data/imagenet_dataloader.py build_common_augmentation('MOCOV2')
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)), # DeCLIP uses 0.2-1.0
            transforms.RandomApply([ # Applied with p=0.8
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) # DeCLIP uses 0.4, 0.4, 0.4, 0.1
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2), # DeCLIP uses p=0.2
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5), # DeCLIP uses p=0.5
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # Standard CLIP evaluation transform
        transform = transforms.Compose([
            transforms.Resize(
                image_size, # Use the integer size directly
                interpolation=interpolation_mode
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
    
    return transform

def nights_transform(image_size: int = 224, interpolation: str = "bicubic"):
    interpolation_map = {
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "bilinear": transforms.InterpolationMode.BILINEAR,
        "nearest": transforms.InterpolationMode.NEAREST
    }
    interpolation_mode = interpolation_map.get(interpolation, transforms.InterpolationMode.BICUBIC)
    
    # Normalize with OpenAI CLIP mean and std (as used in the rest of the project)
    normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
    t = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=interpolation_mode),
        transforms.ToTensor(),
        normalize
    ])
    return t


class NIGHTSDataset(Dataset):
    """
    NIGHTS (Novel Image Generations with Human-Tested Similarities) dataset 
    for contrastive learning from DreamSim triplets.
    
    Each triplet consists of a reference image and two distortions.
    """
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: int = 224,
        interpolation: str = "bicubic",
        min_votes: int = 6,
    ):
        """
        Args:
            root_dir: Path to the NIGHTS dataset containing data.csv
            split: Dataset split to use (train, val, test)
            transform: Optional transform to apply to images
            image_size: Size of images to resize to
            interpolation: Interpolation method for resize
            min_votes: Minimum number of unanimous votes required for inclusion
        """
        self.root_dir = root_dir
        self.split = split
        
        # Load CSV with triplet data
        self.csv_path = os.path.join(self.root_dir, "data.csv")
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"NIGHTS data.csv not found at {self.csv_path}. Please check --dataset_root.")
        self.csv = pd.read_csv(self.csv_path)
        
        # Filter triplets with sufficient unanimous votes
        self.csv = self.csv[self.csv['votes'] >= min_votes]
        
        # Filter by split
        if self.split in ["train", "val", "test"]:
            self.csv = self.csv[self.csv["split"] == split]
        elif split == 'test_imagenet':
            self.csv = self.csv[self.csv['split'] == 'test']
            self.csv = self.csv[self.csv['is_imagenet'] == True]
        elif split == 'test_no_imagenet':
            self.csv = self.csv[self.csv['split'] == 'test']
            self.csv = self.csv[self.csv['is_imagenet'] == False]
        else:
            raise ValueError(f'Invalid split: {split}')
        
        # Set up transform for images
        self.transform = transform
        if self.transform is None:
            is_train = (split == "train")
            self.transform = get_transform(
                image_size=image_size,
                is_train=is_train,
                interpolation=interpolation
            )
        
        self.triplet_transform = TripletTransform(self.transform)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        """Get a triplet with reference and two distorted images"""
        # Get data for this triplet
        row = self.csv.iloc[idx]
        triplet_id = row['id']
        
        # Calculate p based on left_vote and right_vote
        # p indicates how many users preferred right image to left image
        # (higher p means right image is more similar to reference)
        left_vote = row['left_vote']
        right_vote = row['right_vote']
        p = right_vote / (left_vote + right_vote) if (left_vote + right_vote) > 0 else 0.5
        
        # Load the images from paths
        ref_path = row['ref_path']  # reference path
        left_path = row['left_path']  # left distortion path
        right_path = row['right_path']  # right distortion path
        
        try:
            ref_path = os.path.join(self.root_dir, ref_path)
            left_path = os.path.join(self.root_dir, left_path)
            right_path = os.path.join(self.root_dir, right_path)
            
            # suppress decompression warnings when opening images
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                ref_img = Image.open(ref_path).convert('RGB')
                left_img = Image.open(left_path).convert('RGB')
                right_img = Image.open(right_path).convert('RGB')
            
            # Apply transforms to all three images
            ref_tensor, left_tensor, right_tensor = self.triplet_transform([ref_img, left_img, right_img])
            
            # Convert the preference p to a target
            # 1 means right image is closer to reference than left image
            # 0 means left image is closer to reference than right image
            target = 1.0 if p > 0.5 else 0.0
            
            return ref_tensor, left_tensor, right_tensor, target, triplet_id

        except Exception as e:
            logger.error(f"Error loading triplet {triplet_id} (idx {idx}): {e}")
            logger.error(f"Paths: Ref={ref_path}, Left={left_path}, Right={right_path}")
            # Return a dummy triplet or skip? For now, let's return the previous item if possible
            if idx > 0:
                logger.warning(f"Returning item at index {idx-1} instead.")
                return self.__getitem__(idx - 1)
            else:
                # If the first item fails, we have a bigger problem
                raise RuntimeError(f"Failed to load the first item (idx 0) of the dataset.") from e


# --- LitData Preprocessing Function ---
def preprocess_yfcc_litdata(
    item: Dict,
    transform: Callable,
    tokenizer: AutoTokenizer, # Add tokenizer argument
    # Removed eda_augmenter argument
    image_col: str = "images",
    text_col: str = "texts",
    context_length: int = 77 # Standard CLIP context length
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Processes a single item yielded by LitData's ParquetLoader for YFCC15M.
    Removed EDA text augmentation.

    Args:
        item: Dictionary containing raw data (image bytes, token list).
        transform: Image transform function.
        tokenizer: Tokenizer instance for decoding/encoding text.
        image_col: Key for image data in the item dict.
        text_col: Key for text token data in the item dict.
        context_length: Target sequence length for token padding/truncation.

    Returns:
        A tuple (image_tensor, text_tensor) or None if processing fails.
    """
    try:
        # ---- Image Processing (Existing Logic) ----
        img_data = item.get(image_col)
        if img_data is None:
            return None
        
        # Handle different image data formats
        if isinstance(img_data, bytes):
            img = Image.open(BytesIO(img_data))
        elif isinstance(img_data, dict) and "bytes" in img_data:
             img = Image.open(BytesIO(img_data["bytes"]))
        elif isinstance(img_data, Image.Image):
             img = img_data
        else:
            # Log the unexpected format for debugging
            # if random.random() < 0.001: logger.warning(f"Unexpected img_data format: {type(img_data)}")
            return None
        
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Ensure we have a valid PIL Image before applying transform
        if not isinstance(img, Image.Image):
            # if random.random() < 0.001: logger.warning(f"Expected PIL Image, got {type(img)}")
            return None
            
        image_tensor = transform(img)

        tokens_raw = item.get(text_col)
        if tokens_raw is None or (isinstance(tokens_raw, list) and len(tokens_raw) == 0):
            return None

        # Convert raw tokens (potentially list of floats/ints) to clean list of ints
        try:
             token_ids = [int(t) for t in tokens_raw if isinstance(t, (int, float)) and not np.isnan(t)]
             if not token_ids: return None # Skip if no valid tokens
        except: # Catch potential conversion errors
             return None # Skip if conversion fails

        # Process token_ids directly without decode/re-tokenize step
        try:
            # Handle padding and truncation directly on token_ids
            if len(token_ids) > context_length:
                # Truncate if too long
                token_ids = token_ids[:context_length]
            elif len(token_ids) < context_length:
                # Pad if too short - use tokenizer's pad_token_id
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                token_ids = token_ids + [pad_token_id] * (context_length - len(token_ids))
            
            # Convert to PyTorch tensor
            tokens_tensor = torch.tensor(token_ids, dtype=torch.long)
            
            # Verify final tensor shape
            if tokens_tensor.shape[0] != context_length:
                # if random.random() < 0.001: logger.warning(f"Token tensor shape mismatch: {tokens_tensor.shape}. Expected {context_length}.")
                return None # Skip if shape is wrong

        except Exception as process_e:
            # Log infrequently if token processing fails
            # if random.random() < 0.001: logger.warning(f"Token processing failed: {process_e}. Tokens: {token_ids[:10]}...")
            return None # Skip item

        return image_tensor, tokens_tensor

    except Exception as e:
        # Log generic processing errors less frequently
        # if random.random() < 0.001: logger.error(f"Failed to process LitData item: {e}", exc_info=True)
        return None


# --- Custom LitData Dataset Class ---
class YFCC15MLitDataDataset(ld.StreamingDataset):
    """
    Custom StreamingDataset for YFCC15M using LitData.
    Applies preprocessing including image transforms and text tokenization.
    """
    def __init__(
        self,
        *args,
        transform: Callable, # Image transform
        tokenizer: AutoTokenizer, # Tokenizer instance
        # Removed eda_augmenter
        image_col: str = "images",
        text_col: str = "texts",
        context_length: int = 77,
        **kwargs
    ):
        # IMPORTANT: Don't pass transform to base class to avoid litdata 0.2.51 issue
        # where base class applies transform to raw dict instead of PIL image
        super().__init__(*args, **kwargs)
        
        # Store our custom transform separately to avoid base class interference
        self._custom_transform = transform
        self.tokenizer = tokenizer
        # Removed self.eda_augmenter
        self.image_col = image_col
        self.text_col = text_col
        self.context_length = context_length
        
        # Disable base class transform to prevent it from being applied to raw dict
        # Use identity function instead of None to avoid TypeError
        self.transform = lambda x: x

        if self._custom_transform is None:
             raise ValueError("An image transform function must be provided")
        if self.tokenizer is None:
             raise ValueError("A tokenizer instance must be provided")
        # Removed check for TEXTAUGMENT_AVAILABLE
    
    def set_transform(self, transform: Callable):
        """Method to update transform after dataset creation (e.g., after splitting)"""
        self._custom_transform = transform
    
    def set_preprocessing_params(self, transform: Callable, tokenizer: AutoTokenizer, 
                                image_col: str = None, text_col: str = None, 
                                context_length: int = None):
        """Method to update all preprocessing parameters after dataset creation"""
        self._custom_transform = transform
        self.tokenizer = tokenizer
        if image_col is not None:
            self.image_col = image_col
        if text_col is not None:
            self.text_col = text_col
        if context_length is not None:
            self.context_length = context_length

    def __getitem__(self, index) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        # 1. Get the raw item dictionary from the base class
        # Note: base class transform is disabled to prevent it from being applied to raw dict
        raw_item = super().__getitem__(index)

        # 2. Apply the preprocessing function using our custom transform
        processed_item = preprocess_yfcc_litdata(
            item=raw_item,
            transform=self._custom_transform,
            tokenizer=self.tokenizer, # Pass stored tokenizer
            # Removed eda_augmenter
            image_col=self.image_col,
            text_col=self.text_col,
            context_length=self.context_length
        )

        # 3. Handle potential None return from preprocessing
        if processed_item is None:
             # Raise IndexError, DataLoader's collate_fn should handle skipping it
             raise IndexError(f"Preprocessing failed for index {index}")

        return processed_item


# --- First add helper functions for cifar100 validation dataset ---
def create_cifar100_validation_dataloader(
    transform, 
    batch_size: int, 
    num_workers: int, 
    pin_memory: bool, 
    distributed: bool = False, 
    val_dataset_root: str = None, 
    seed: int = 42
):
    """Create a CIFAR100 validation dataloader"""
    logger.info("Creating CIFAR100 validation dataloader")
    
    # Default cache path or use config
    cifar100_cache_dir = val_dataset_root or os.path.join(os.getcwd(), 'cifar100_eval_cache')
    os.makedirs(cifar100_cache_dir, exist_ok=True)
    logger.info(f"CIFAR100 cache/download directory: {cifar100_cache_dir}")
    
    try:
        val_dataset = tv_datasets.CIFAR100(
            root=cifar100_cache_dir,
            train=False,  
            download=True,
            transform=transform
        )
        logger.info(f"CIFAR100 test set loaded successfully ({len(val_dataset)} samples).")
        
        val_sampler = None
        if distributed:
            try:
                # Check if process group is initialized before creating DistributedSampler
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=seed, drop_last=False)
                    logger.info("Using DistributedSampler for CIFAR100 validation")
                else:
                    logger.warning("torch.distributed not initialized when creating val_sampler. Using sequential sampler.")
            except Exception as e:
                logger.warning(f"Error creating DistributedSampler for validation: {e}. Using sequential sampler.")
        
        dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,  # No shuffle for validation
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            persistent_workers=False,
            prefetch_factor=2 if num_workers > 0 else None
        )
        logger.info(f"Created CIFAR100 validation DataLoader with batch size {batch_size}.")
        return dataloader
    except Exception as e:
        logger.error(f"Failed to create CIFAR100 validation dataset/dataloader: {e}", exc_info=True)
        logger.warning("Proceeding without CIFAR100 validation.")
        return None

def create_cifar10_validation_dataloader(
    transform, 
    batch_size: int, 
    num_workers: int, 
    pin_memory: bool, 
    distributed: bool = False, 
    val_dataset_root: str = None, 
    seed: int = 42
):
    """Create a CIFAR10 validation dataloader"""
    logger.info("Creating CIFAR10 validation dataloader")
    
    # Default cache path or use config
    cifar10_cache_dir = val_dataset_root or os.path.join(os.getcwd(), 'cifar10_eval_cache')
    os.makedirs(cifar10_cache_dir, exist_ok=True)
    logger.info(f"CIFAR10 cache/download directory: {cifar10_cache_dir}")
    
    try:
        val_dataset = tv_datasets.CIFAR10(
            root=cifar10_cache_dir,
            train=False,  # Use the test set for validation
            download=True,
            transform=transform
        )
        logger.info(f"CIFAR10 test set loaded successfully ({len(val_dataset)} samples).")
        
        val_sampler = None
        if distributed:
            try:
                # Check if process group is initialized before creating DistributedSampler
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=seed, drop_last=False)
                    logger.info("Using DistributedSampler for CIFAR10 validation")
                else:
                    logger.warning("torch.distributed not initialized when creating val_sampler. Using sequential sampler.")
            except Exception as e:
                logger.warning(f"Error creating DistributedSampler for validation: {e}. Using sequential sampler.")
        
        dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,  # No shuffle for validation
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            persistent_workers=False,
            prefetch_factor=2 if num_workers > 0 else None
        )
        logger.info(f"Created CIFAR10 validation DataLoader with batch size {batch_size}.")
        return dataloader
    except Exception as e:
        logger.error(f"Failed to create CIFAR10 validation dataset/dataloader: {e}", exc_info=True)
        logger.warning("Proceeding without CIFAR10 validation.")
        return None

# --- Now modify the create_dataloaders function ---
def create_dataloaders(
    config: 'DataConfig', # Use the config object
    seed: int = 42,
    distributed: bool = True,
    multi_val: bool = False, # New parameter to enable multiple validation datasets
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for train, validation, and test sets based on config.

    Args:
        config: DataConfig object containing dataset and dataloader settings.
        seed: Random seed for reproducibility.
        distributed: Whether to use DistributedSampler for training.
        multi_val: Whether to return multiple validation dataloaders

    Returns:
        Dictionary of dataloaders for each split.
    """
    dataloaders = {}
    g = torch.Generator()
    g.manual_seed(seed)
    
    # --- Initialize Tokenizer ---
    # Moved outside the dataset loop as it's needed for YFCC text processing
    tokenizer = None
    if config.dataset_type == "yfcc15m_litdata":
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
            logger.info(f"Tokenizer '{config.tokenizer_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load tokenizer '{config.tokenizer_name}': {e}")
            raise RuntimeError(f"Tokenizer '{config.tokenizer_name}' is required for YFCC15M.") from e
        
    use_persistent_workers = config.num_workers > 0 and getattr(config, 'use_persistent_workers', False)
    
    # Common DataLoader kwargs (ensure pin_memory uses the config value)
    dataloader_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory, # Use config value here
        "worker_init_fn": seed_worker,
        "persistent_workers": True, # Usually False for streaming
        "prefetch_factor": 4 if config.num_workers > 0 else None # Common default
    }

    if config.dataset_type == "nights":
        logger.info(f"Creating dataloaders for NIGHTS dataset from: {config.dataset_root}")
        train_transform = nights_transform(config.image_size, config.interpolation)
        eval_transform = nights_transform(config.image_size, config.interpolation)

        datasets = {
            "train": NIGHTSDataset(
                root_dir=config.dataset_root, split="train", transform=train_transform,
                image_size=config.image_size, interpolation=config.interpolation
            ),
            "val": NIGHTSDataset(
                root_dir=config.dataset_root, split="val", transform=eval_transform,
                image_size=config.image_size, interpolation=config.interpolation
            ),
            "test": NIGHTSDataset(
                root_dir=config.dataset_root, split="test", transform=eval_transform,
                image_size=config.image_size, interpolation=config.interpolation
            )
        }

        for split, dataset in datasets.items():
             is_train = (split == "train")
             sampler = None
             shuffle = False
             drop_last = False

             if distributed and is_train: # Only use DDP sampler for training
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=True,
                        seed=seed,
                        drop_last=True
                    )
                    logger.info("Using DistributedSampler for training")
                    shuffle = False # Sampler handles shuffle
                else:
                    logger.warning("torch.distributed not initialized. Skipping DistributedSampler for training")
                    shuffle = True
                    drop_last = True
                 
             elif is_train:
                 shuffle = True # Shuffle training if not distributed
                 drop_last = True
             else: # Validation/Test
                if distributed:
                     # Use non-shuffling DDP sampler for validation/test
                    #  sampler = DistributedSampler(dataset, shuffle=False, seed=seed, drop_last=False)
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        sampler = DistributedSampler(dataset, shuffle=False, seed=seed, drop_last=False)
                        logger.info(f"Using DistributedSampler for {split}")
                    else:
                        logger.warning("torch.distributed not initialized for validation/test sampler.")  
                 
                shuffle = False
                drop_last = False


             dataloaders[split] = DataLoader(
                 dataset,
                 batch_size=config.batch_size,
                 sampler=sampler,
                 shuffle=shuffle,
                 drop_last=drop_last,
                 generator=g if shuffle and sampler is None else None, # Generator only for non-distrib shuffle
                 **dataloader_kwargs
             )

    elif config.dataset_type == "yfcc15m_litdata":
        if not LITDATA_AVAILABLE:
            raise ImportError("LitData is required for 'yfcc15m_litdata' dataset type. Please install litdata.")
        if tokenizer is None: # Should not happen due to check above, but for safety
             raise RuntimeError("Tokenizer could not be initialized for YFCC15M LitData.")


        logger.info(f"Creating LitData dataloaders for YFCC15M dataset.")
        logger.info(f"  Parquet data directory: {config.parquet_data_dir}")
        # Removed logging about text augmentation

        # --- Define Transforms ---
        train_transform = get_transform(config.image_size, True, config.interpolation)
        eval_transform = get_transform(config.image_size, False, config.interpolation)
        context_length = 77 # Standard CLIP

        # --- Create Custom LitData StreamingDataset for TRAINING ---
        try:
            # We first instantiate the full dataset intended for training
            full_train_dataset = YFCC15MLitDataDataset(
                input_dir=config.hf_dataset_name if not config.parquet_data_dir else config.parquet_data_dir,
                cache_dir=config.hf_cache_dir,
                index_path="index.json" if not config.dataset_index_path else config.dataset_index_path,
                item_loader=ParquetLoader(),
                shuffle=True,
                # Don't pass transform to base class - it will be stored as _custom_transform
                transform=train_transform,
                tokenizer=tokenizer,
                image_col=config.image_col,
                text_col=config.text_col,
                context_length=context_length
            )
            logger.info("Initial YFCC15MLitDataDataset created for splitting.")

            # --- Split the dataset into train and validation ---
            val_split_size = 10000
            len_full_train_dataset = len(full_train_dataset)
            if len_full_train_dataset > val_split_size:
                 logger.info(f"Splitting {val_split_size} samples for validation from YFCC15M stream len {len_full_train_dataset}...")
                 train_dataset, val_dataset_yfcc = ld.train_test_split(
                     full_train_dataset,
                     splits=[1 - val_split_size / len_full_train_dataset, val_split_size / len_full_train_dataset],
                     seed=seed
                 )
                 logger.info(f"YFCC15M split complete: Train size ~{len(train_dataset)}, Val size {len(val_dataset_yfcc)}")
                 
                 # Fix for litdata 0.2.51: Properly set preprocessing parameters for validation dataset
                 if hasattr(val_dataset_yfcc, 'set_preprocessing_params'):
                     val_dataset_yfcc.set_preprocessing_params(
                         transform=eval_transform,
                         tokenizer=tokenizer,
                         image_col=config.image_col,
                         text_col=config.text_col,
                         context_length=context_length
                     )
                     logger.info("Applied evaluation transform and preprocessing params to YFCC15M validation split using set_preprocessing_params.")
                 elif hasattr(val_dataset_yfcc, 'set_transform'):
                     val_dataset_yfcc.set_transform(eval_transform)
                     logger.info("Applied evaluation transform to YFCC15M validation split using set_transform.")
                 elif hasattr(val_dataset_yfcc, 'item_loader') and hasattr(val_dataset_yfcc.item_loader, 'transform'):
                      val_dataset_yfcc.item_loader.transform = eval_transform
                      logger.info("Applied evaluation transform to YFCC15M validation split via item_loader.")
                 else:
                      logger.warning("Could not modify transform for YFCC15M validation split. It might use training augmentations.")

            else:
                 logger.warning(f"Full dataset size ({len(full_train_dataset)}) is not larger than requested validation size ({val_split_size}). Using full dataset for training and skipping YFCC validation split.")
                 train_dataset = full_train_dataset
                 val_dataset_yfcc = None

        except FileNotFoundError as e:
             # Catch specific error if index is still missing
             logger.error(f"LitData index file ('index.json') likely missing inside the input_dir: {config.parquet_data_dir}")
             logger.error(f"Original error: {e}")
             logger.error("Please run the indexing script (e.g., index_dataset.py) first.")
             raise
        except Exception as e:
            logger.error(f"Failed to create YFCC15MLitDataDataset: {e}", exc_info=True)
            raise

        # --- Create LitData StreamingDataLoader for TRAINING ---
        # Use collate function to handle potential IndexErrors from __getitem__
        def skip_indexing_error_collate(batch):
             # Filter out None values that signify an IndexError occurred
             filtered_batch = [item for item in batch if item is not None]
             if not filtered_batch:
                 return None # Skip batch if all items failed
             # Use default collate on the filtered batch
             try:
                 return torch.utils.data.dataloader.default_collate(filtered_batch)
             except Exception as e:
                  # Log infrequent collation errors
                  # if random.random() < 0.001: logger.error(f"Error during collate_fn: {e}. Batch content: {filtered_batch}")
                  return None # Skip batch if collate fails

        if train_dataset: # Only create loader if train_dataset exists
            dataloaders["train"] = ld.StreamingDataLoader(
                train_dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                persistent_workers=use_persistent_workers,
                prefetch_factor=dataloader_kwargs.get("prefetch_factor", 8),
                collate_fn=skip_indexing_error_collate # Use the custom collate
            )
            logger.info(f"LitData StreamingDataLoader created for training.")
        else:
             dataloaders["train"] = None
             logger.error("Failed to create training dataset/dataloader for YFCC15M.")


        # --- Create validation dataloaders ---
        eval_batch_size = config.eval_batch_size if config.eval_batch_size else config.batch_size // 2
        val_dataloaders = [] # Initialize empty list

        # --- Create YFCC15M Validation DataLoader (if split was successful) ---
        if val_dataset_yfcc:
             val_loader_yfcc = ld.StreamingDataLoader(
                 val_dataset_yfcc,
                 batch_size=eval_batch_size,
                 num_workers=config.num_workers,
                 pin_memory=config.pin_memory,
                 persistent_workers=use_persistent_workers, # May need False if transform change doesn't persist workers
                 prefetch_factor=dataloader_kwargs.get("prefetch_factor", 8),
                 collate_fn=skip_indexing_error_collate # Use the custom collate
             )
             val_dataloaders.append(val_loader_yfcc) # Add YFCC val loader first
             logger.info(f"Created YFCC15M validation DataLoader (index 0)")

        # --- Create CIFAR Validation Loaders (conditionally) ---
        cifar10_dataloader = None
        cifar100_dataloader = None
        if multi_val:
            logger.info("Creating multiple validation dataloaders (CIFAR10 and CIFAR100)")
            cifar10_dataloader = create_cifar10_validation_dataloader(
                transform=eval_transform,
                batch_size=eval_batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                distributed=distributed,
                val_dataset_root=config.val_dataset_root,
                seed=seed
            )
            cifar100_dataloader = create_cifar100_validation_dataloader(
                transform=eval_transform,
                batch_size=eval_batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                distributed=distributed,
                val_dataset_root=config.val_dataset_root,
                seed=seed
            )
            if cifar10_dataloader:
                val_dataloaders.append(cifar10_dataloader) # Add CIFAR10 (index 1 or potentially 0)
                logger.info(f"Added CIFAR10 validation DataLoader (index {len(val_dataloaders)-1})")
            if cifar100_dataloader:
                val_dataloaders.append(cifar100_dataloader) # Add CIFAR100 (index 2 or potentially 1)
                logger.info(f"Added CIFAR100 validation DataLoader (index {len(val_dataloaders)-1})")
        else:
            # Original logic: Create just CIFAR10
            logger.info("Creating CIFAR10 validation dataloader.")
            cifar10_dataloader = create_cifar10_validation_dataloader(
                transform=eval_transform,
                batch_size=eval_batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                distributed=distributed,
                val_dataset_root=config.val_dataset_root,
                seed=seed
            )
            if cifar10_dataloader:
                 val_dataloaders.append(cifar10_dataloader) # Add CIFAR10 (index 1 or potentially 0)
                 logger.info(f"Added CIFAR10 validation DataLoader (index {len(val_dataloaders)-1})")


        # --- Assign the list of dataloaders ---
        if val_dataloaders:
             dataloaders["val"] = val_dataloaders
             logger.info(f"Created {len(val_dataloaders)} validation dataloader(s) in total.")
        else:
             dataloaders["val"] = None
             logger.warning("No validation dataloaders could be created.")
        
        # --- Test dataloader ---
        dataloaders["test"] = None # Test set not handled in this setup
        logger.info("Created dataloaders for YFCC15M using LitData.")


    elif config.dataset_type == "things":
        logger.info(f"Creating dataloaders for THINGS dataset (CSV format only)")
        logger.info(f"Images from: {config.dataset_root}")
        
        # Get CSV triplet directory - this is REQUIRED
        csv_triplet_dir = getattr(config, 'csv_triplet_dir', None)
        if not csv_triplet_dir:
            raise ValueError("csv_triplet_dir must be specified in config for THINGS dataset")
        
        logger.info(f"Triplets from: {csv_triplet_dir}")
        
        train_transform = get_transform(config.image_size, True, config.interpolation)
        eval_transform = get_transform(config.image_size, False, config.interpolation)

        # Subset sizes should be None to use all your generated triplets
        train_subset = getattr(config, 'subset_size_train', None)
        val_subset = getattr(config, 'subset_size_val', None)
        test_subset = getattr(config, 'subset_size_test', None)
        subset_seed = getattr(config, 'subset_seed', 42)

        datasets = {
            "train": THINGSDataset(
                root_dir=config.dataset_root,
                split="train", 
                transform=train_transform,
                image_size=config.image_size, 
                interpolation=config.interpolation,
                subset_size=train_subset,  # Should be None
                subset_seed=subset_seed,
                csv_triplet_dir=csv_triplet_dir  # REQUIRED
            ),
            "val": THINGSDataset(
                root_dir=config.dataset_root,
                split="val", 
                transform=eval_transform,
                image_size=config.image_size, 
                interpolation=config.interpolation,
                subset_size=val_subset,  # Should be None
                subset_seed=subset_seed,
                csv_triplet_dir=csv_triplet_dir  # REQUIRED
            ),
            "test": THINGSDataset(
                root_dir=config.dataset_root,
                split="test", 
                transform=eval_transform,
                image_size=config.image_size, 
                interpolation=config.interpolation,
                subset_size=test_subset,  # Should be None
                subset_seed=subset_seed,
                csv_triplet_dir=csv_triplet_dir  # REQUIRED
            )
        }

        for split, dataset in datasets.items():
             is_train = (split == "train")
             sampler = None
             shuffle = False
             drop_last = False

             if distributed and is_train: # Only use DDP sampler for training
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=True,
                        seed=seed,
                        drop_last=True
                    )
                    logger.info("Using DistributedSampler for training")
                    shuffle = False # Sampler handles shuffle
                else:
                    logger.warning("torch.distributed not initialized. Skipping DistributedSampler for training")
                    shuffle = True
                    drop_last = True
                 
             elif is_train:
                 shuffle = True # Shuffle training if not distributed
                 drop_last = True
             else: # Validation/Test
                if distributed:
                     # Use non-shuffling DDP sampler for validation/test
                    #  sampler = DistributedSampler(dataset, shuffle=False, seed=seed, drop_last=False)
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        sampler = DistributedSampler(dataset, shuffle=False, seed=seed, drop_last=False)
                        logger.info(f"Using DistributedSampler for {split}")
                    else:
                        logger.warning("torch.distributed not initialized for validation/test sampler.")  
                 
                shuffle = False
                drop_last = False


             dataloaders[split] = DataLoader(
                 dataset,
                 batch_size=config.batch_size,
                 sampler=sampler,
                 shuffle=shuffle,
                 drop_last=drop_last,
                 generator=g if shuffle and sampler is None else None, # Generator only for non-distrib shuffle
                 **dataloader_kwargs
             )
    elif config.dataset_type.startswith("nod_imagenet_triplets"):
        logger.info("Creating dataloaders for NOD (group-RDM) category triplets using ImageNet folders")
        logger.info(f"ImageNet root (synset folders): {config.dataset_root}")

        # Parent dir that contains subfolders: all/, front/, mid/, late/
        csv_base_dir = getattr(config, "csv_triplet_dir", None)
        if not csv_base_dir:
            raise ValueError("csv_triplet_dir must be specified (parent dir containing all/front/mid/late)")

        # 1) Prefer explicit config field if you add it
        group = getattr(config, "triplet_group", None)

        # Choose subfolder if it exists; otherwise assume csv_base_dir already points to the group folder
        csv_base_dir = Path(csv_base_dir)
        csv_group_dir = csv_base_dir / "csv" / group
        if csv_group_dir.exists() and csv_group_dir.is_dir():
            csv_triplet_dir = str(csv_group_dir)
        else:
            csv_triplet_dir = str(csv_base_dir)

        logger.info(f"Triplet group: {group}")
        logger.info(f"Triplet CSV directory used: {csv_triplet_dir}")

        train_transform = get_transform(config.image_size, True, config.interpolation)
        eval_transform  = get_transform(config.image_size, False, config.interpolation)

        datasets = {
            "train": ImageNetCategoryTripletDataset(
                imagenet_root=config.dataset_root,
                csv_triplet_dir=csv_triplet_dir,
                split="train",
                transform=train_transform,
                image_size=config.image_size,
                interpolation=config.interpolation,
                seed=seed,
                preload_file_lists=True,
            ),
            "val": ImageNetCategoryTripletDataset(
                imagenet_root=config.dataset_root,
                csv_triplet_dir=csv_triplet_dir,
                split="val",
                transform=eval_transform,
                image_size=config.image_size,
                interpolation=config.interpolation,
                seed=seed,
                preload_file_lists=True,
            ),
            "test": ImageNetCategoryTripletDataset(
                imagenet_root=config.dataset_root,
                csv_triplet_dir=csv_triplet_dir,
                split="test",
                transform=eval_transform,
                image_size=config.image_size,
                interpolation=config.interpolation,
                seed=seed,
                preload_file_lists=True,
            ),
        }

        for split, dataset in datasets.items():
            is_train = (split == "train")
            sampler = None
            shuffle = False
            drop_last = False

            if distributed and is_train:
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    sampler = DistributedSampler(dataset, shuffle=True, seed=seed, drop_last=True)
                    logger.info("Using DistributedSampler for training")
                    shuffle = False
                    drop_last = True
                else:
                    logger.warning("torch.distributed not initialized. Skipping DistributedSampler for training")
                    shuffle = True
                    drop_last = True
            elif is_train:
                shuffle = True
                drop_last = True
            else:
                if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
                    sampler = DistributedSampler(dataset, shuffle=False, seed=seed, drop_last=False)
                    logger.info(f"Using DistributedSampler for {split}")
                shuffle = False
                drop_last = False

            dataloaders[split] = DataLoader(
                dataset,
                batch_size=config.batch_size,
                sampler=sampler,
                shuffle=shuffle,
                drop_last=drop_last,
                generator=g if shuffle and sampler is None else None,
                **dataloader_kwargs
            )

    else:
        raise ValueError(f"Unsupported dataset_type: {config.dataset_type}")

    return dataloaders

class THINGSDataset(Dataset):
    def __init__(
        self,
        root_dir: str,  # "/tmp/things" - for images
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: int = 224,
        interpolation: str = "bicubic",
        subset_size: Optional[int] = None,
        subset_seed: int = 42,
        cache_images: bool = False,  # DISABLED: No caching
        csv_triplet_dir: Optional[str] = None
    ):
        self.root_dir = root_dir
        self.split = split
        self.cache_images = False  # Force disable caching
        
        # ONLY use CSV format - no fallback to text format
        if not csv_triplet_dir or not os.path.exists(csv_triplet_dir):
            raise ValueError(f"csv_triplet_dir must be provided and exist: {csv_triplet_dir}")
        
        self.csv_triplet_dir = csv_triplet_dir
        logger.info(f"Using CSV triplet format from: {self.csv_triplet_dir}")
        self._init_csv_format()
        
        # Set up transforms
        self.transform = transform
        if self.transform is None:
            is_train = (split == "train")
            self.transform = get_transform(
                image_size=image_size,
                is_train=is_train,
                interpolation=interpolation
            )
        
        self.triplet_transform = TripletTransform(self.transform)
        
        # Your datasets are pre-made - ignore any subsetting parameters
        if subset_size is not None:
            logger.info(f"⚠️  Ignoring subset_size={subset_size}, using pre-made dataset with {len(self.triplets)} triplets")
        
        logger.info(f" Using pre-made THINGS dataset with {len(self.triplets)} triplets (NO CACHING)")
    
    def _init_csv_format(self):
        """Initialize dataset using your pre-made CSV format"""
        # Load index to image mapping from your CSV
        mapping_file = os.path.join(self.csv_triplet_dir, "index_to_image_mapping.csv")
        if not os.path.exists(mapping_file):
            raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
        
        self.mapping_df = pd.read_csv(mapping_file)
        self.index_to_image = dict(zip(self.mapping_df['index'], self.mapping_df['image_name']))
        
        # Images are in root_dir/Things1854 
        self.image_dir = os.path.join(self.root_dir, "Things1854")
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # Load your pre-made triplet CSV
        split_files = {
            "train": "train_triplets.csv",    # Your 13,900 triplets
            "val": "val_triplets.csv",        # Your 1,720 triplets  
            "test": "test_triplets.csv"       # Your 1,824 triplets
        }
        
        triplet_file = os.path.join(self.csv_triplet_dir, split_files[self.split])
        if not os.path.exists(triplet_file):
            raise FileNotFoundError(f"Triplet file not found: {triplet_file}")
        
        triplets_df = pd.read_csv(triplet_file)
        
        # Validate required columns
        required_cols = ['anchor_idx', 'positive_idx', 'negative_idx']
        missing_cols = [col for col in required_cols if col not in triplets_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {triplet_file}: {missing_cols}")
        
        # Convert to list of tuples - YOUR EXACT DATASET
        self.triplets = [
            (row['anchor_idx'], row['positive_idx'], row['negative_idx']) 
            for _, row in triplets_df.iterrows()
        ]
        
        logger.info(f"✅ Loaded {len(self.triplets)} pre-made triplets from: {triplet_file}")
        logger.info(f"✅ Loaded mapping for {len(self.index_to_image)} images")
    
    def _get_image_path(self, img_idx: int) -> str:
        """Get image path from index using your mapping CSV"""
        if img_idx not in self.index_to_image:
            raise ValueError(f"Image index {img_idx} not found in mapping")
        
        image_name = self.index_to_image[img_idx]  # e.g., "aardvark_01b.jpg"
        img_path = os.path.join(self.image_dir, image_name)  # /tmp/things/Things1854/aardvark_01b.jpg
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        return img_path
    
    def _load_image(self, img_idx: int) -> Image.Image:
        """Load image directly without caching"""
        img_path = self._get_image_path(img_idx)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            img = Image.open(img_path).convert('RGB')
        return img
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        """Load triplet of images using your index-to-image mapping with NIGHTS-style targets."""
        indices = self.triplets[idx]  # (anchor_idx, positive_idx, negative_idx)
        
        try:
            # Load images directly without caching
            ref_img = self._load_image(indices[0])    # Anchor = Reference
            pos_img = self._load_image(indices[1])    # Positive = More similar to ref
            neg_img = self._load_image(indices[2])    # Negative = Less similar to ref
            
            # Apply transforms
            ref_tensor, pos_tensor, neg_tensor = self.triplet_transform([ref_img, pos_img, neg_img])
            
             
            target = 0
            triplet_id = f"things_nights_{idx}_{indices[0]}_{indices[1]}_{indices[2]}"
            
            return ref_tensor, pos_tensor, neg_tensor, target, triplet_id
            
        except Exception as e:
            logger.error(f"Error loading THINGS triplet {idx} with indices {indices}: {e}")
            if idx > 0:
                return self.__getitem__(idx - 1)
            else:
                raise RuntimeError(f"Failed to load first THINGS triplet") from e
            

def _list_images_in_dir(cat_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".JPEG", ".JPG"}
    files = []
    if not cat_dir.exists() or not cat_dir.is_dir():
        return files
    for p in cat_dir.iterdir():
        if p.is_file() and (p.suffix in exts):
            files.append(p)
    return files


class ImageNetCategoryTripletDataset(Dataset):
    """
    Reads category-triplet CSVs with columns:
      anchor_cat, positive_cat, negative_cat
    where each category is an ImageNet synset folder under imagenet_root.

    For each row, samples one random image from each category folder and returns:
      (ref_tensor, pos_tensor, neg_tensor, target, triplet_id)

    This matches the THINGSDataset/NIGHTS-style output expected by your training loop.
    """

    def __init__(
        self,
        imagenet_root: str,          # directory containing synset folders, e.g. .../ILSVRC/train/
        csv_triplet_dir: str,        # directory containing train_triplets.csv, val_triplets.csv, test_triplets.csv
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: int = 224,
        interpolation: str = "bicubic",
        seed: int = 42,
        preload_file_lists: bool = True,
    ):
        super().__init__()
        self.imagenet_root = Path(imagenet_root)
        self.csv_triplet_dir = Path(csv_triplet_dir)
        self.split = split
        self.seed = seed

        # Transform setup (reuse your existing augmentations)
        self.transform = transform
        if self.transform is None:
            is_train = (split == "train")
            self.transform = get_transform(
                image_size=image_size,
                is_train=is_train,
                interpolation=interpolation,
            )
        self.triplet_transform = TripletTransform(self.transform)

        # Load triplets CSV
        split_file = {
            "train": "train_triplets.csv",
            "val": "val_triplets.csv",
            "test": "test_triplets.csv",
        }.get(split)
        if split_file is None:
            raise ValueError(f"Invalid split: {split}")

        csv_path = self.csv_triplet_dir / split_file
        if not csv_path.exists():
            raise FileNotFoundError(f"Triplet CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Support either category CSV OR (fallback) index CSV
        if {"anchor_cat", "positive_cat", "negative_cat"}.issubset(df.columns):
            self.triplets = list(
                zip(df["anchor_cat"].astype(str), df["positive_cat"].astype(str), df["negative_cat"].astype(str))
            )
            self.mode = "cat"
        else:
            raise ValueError(
                f"{csv_path} must have columns: anchor_cat, positive_cat, negative_cat "
                f"(found: {list(df.columns)})"
            )

        # Build per-category file lists (only for categories used in the CSV)
        self.cat_to_files = {}
        self.bad_files = {}
        if preload_file_lists:
            used_cats = set()
            for a, p, n in self.triplets:
                used_cats.add(a); used_cats.add(p); used_cats.add(n)

            missing = []
            for cat in sorted(used_cats):
                cat_dir = self.imagenet_root / cat
                files = _list_images_in_dir(cat_dir)
                if len(files) == 0:
                    missing.append(cat)
                self.cat_to_files[cat] = files

            if missing:
                # Don't hard-fail immediately; but this will break at __getitem__ if sampled
                print(f"[WARN] {len(missing)} categories had no images under {self.imagenet_root}. "
                      f"Examples: {missing[:10]}")

    def __len__(self) -> int:
        return len(self.triplets)

    # def _sample_image_path(self, cat: str, idx: int) -> Path:
    #     # Lazy-load if not preloaded
    #     if cat not in self.cat_to_files:
    #         cat_dir = self.imagenet_root / cat
    #         self.cat_to_files[cat] = _list_images_in_dir(cat_dir)

    #     files = self.cat_to_files[cat]
    #     if not files:
    #         raise FileNotFoundError(f"No images found for category '{cat}' under {self.imagenet_root / cat}")

    #     # Deterministic-ish sampling per index (helps reproducibility across workers)
    #     rng = random.Random(self.seed + 1000003 * idx + hash(cat) % 1000003)
    #     return files[rng.randrange(len(files))]

    # def _load_image(self, path: Path) -> Image.Image:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore", Image.DecompressionBombWarning)
    #         return Image.open(str(path)).convert("RGB")

    def _stable_int(self, s: str) -> int:
        return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)

    def _sample_image_path(self, cat: str, idx: int, attempt: int = 0) -> Path:
        if cat not in self.cat_to_files:
            cat_dir = self.imagenet_root / cat
            self.cat_to_files[cat] = _list_images_in_dir(cat_dir)

        files = self.cat_to_files[cat]
        if not files:
            raise FileNotFoundError(f"No images found for category '{cat}' under {self.imagenet_root / cat}")

        bad = self.bad_files.get(cat, set())
        candidates = [p for p in files if p.name not in bad]
        if not candidates:
            candidates = files  # fallback if everything marked bad

        rng = random.Random(self.seed + 1000003 * idx + 10007 * attempt + self._stable_int(cat))
        return candidates[rng.randrange(len(candidates))]
    
    def _load_image(self, path: Path, cat: str) -> Image.Image:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                img = Image.open(str(path)).convert("RGB")
                img.load()  # IMPORTANT: forces decode now (catches many corrupt files)
            return img
        except (OSError, UnidentifiedImageError, ValueError) as e:
            self.bad_files.setdefault(cat, set()).add(path.name)
            raise e

    # def __getitem__(self, idx: int):
    #     a_cat, p_cat, n_cat = self.triplets[idx]

    #     try:
    #         a_path = self._sample_image_path(a_cat, idx)
    #         p_path = self._sample_image_path(p_cat, idx)
    #         n_path = self._sample_image_path(n_cat, idx)

    #         ref_img = self._load_image(a_path)
    #         pos_img = self._load_image(p_path)
    #         neg_img = self._load_image(n_path)

    #         ref_tensor, pos_tensor, neg_tensor = self.triplet_transform([ref_img, pos_img, neg_img])

    #         target = 0  # same as THINGSDataset; your loss uses ordering implicitly
    #         triplet_id = f"nod_imagenet_{self.split}_{idx}_{a_cat}_{p_cat}_{n_cat}"

    #         return ref_tensor, pos_tensor, neg_tensor, target, triplet_id

    #     except Exception as e:
    #         logger.error(f"Error loading ImageNet category triplet idx={idx} cats={a_cat,p_cat,n_cat}: {e}")
    #         if idx > 0:
    #             return self.__getitem__(idx - 1)
    #         raise
    def __getitem__(self, idx: int):
        a_cat, p_cat, n_cat = self.triplets[idx]

        max_tries = 30
        last_err = None

        for attempt in range(max_tries):
            try:
                a_path = self._sample_image_path(a_cat, idx, attempt)
                p_path = self._sample_image_path(p_cat, idx, attempt)
                n_path = self._sample_image_path(n_cat, idx, attempt)

                ref_img = self._load_image(a_path, a_cat)
                pos_img = self._load_image(p_path, p_cat)
                neg_img = self._load_image(n_path, n_cat)

                ref_tensor, pos_tensor, neg_tensor = self.triplet_transform([ref_img, pos_img, neg_img])

                target = 0
                triplet_id = f"nod_imagenet_{self.split}_{idx}_{a_cat}_{p_cat}_{n_cat}"
                return ref_tensor, pos_tensor, neg_tensor, target, triplet_id

            except Exception as e:
                last_err = e
                continue

        logger.error(f"Failed to fetch a valid triplet after {max_tries} tries for idx={idx} cats={a_cat,p_cat,n_cat}. Last err={last_err}")
        raise last_err
