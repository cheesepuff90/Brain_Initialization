import os
import logging
from transformers import AutoTokenizer
import torch # Needed for get_transform

# --- Mock LitData components if not installed ---
# In a real environment, you'd have litdata installed.
# For this test script, we mock the necessary parts if unavailable.
try:
    import litdata as ld
    from litdata.streaming.item_loader import ParquetLoader
    LITDATA_AVAILABLE = True
except ImportError:
    print("Warning: litdata not found. Using mock objects.")
    LITDATA_AVAILABLE = False
    # Mock base class and loader
    class MockStreamingDataset:
        def __init__(self, *args, **kwargs):
            print("MockStreamingDataset initialized.")
            # Simulate loading index to get a length
            self._len = self._load_mock_index(kwargs.get('input_dir'))

        def _load_mock_index(self, input_dir):
            # Simulate checking for an index file and returning a dummy length
            mock_index_path = os.path.join(input_dir, "index.json") if input_dir else None
            if mock_index_path and os.path.exists(mock_index_path):
                 print(f"Found mock index at {mock_index_path}. Returning simulated length.")
                 # In a real scenario, litdata reads this file.
                 # Let's return a dummy length for testing.
                 return 1000000 # Example length
            elif input_dir:
                 print(f"Mock index not found at {mock_index_path}. Returning length 0.")
                 return 0
            else:
                 print("No input_dir provided to MockStreamingDataset. Returning length 0.")
                 return 0

        def __len__(self):
            return self._len

        def __getitem__(self, index):
            # Not needed for len() check
            raise NotImplementedError

    class MockParquetLoader:
        def __init__(self, *args, **kwargs):
             print("MockParquetLoader initialized.")
             pass # No state needed for this mock

    ld = type('MockLitDataModule', (object,), {'StreamingDataset': MockStreamingDataset})()
    ParquetLoader = MockParquetLoader
# --- End Mock LitData ---

# Need to import necessary components from your project files
# Assuming they are accessible in the Python path
try:
    from src.dataset import get_transform, YFCC15MLitDataDataset as RealYFCC15MLitDataDataset
    # Decide whether to use the real class or the mock based on litdata availability
    if LITDATA_AVAILABLE:
        YFCC15MLitDataDataset = RealYFCC15MLitDataDataset
    else:
        # If litdata isn't available, use the mock StreamingDataset as the base
        # for the structure expected by the code, even though YFCC15MLitDataDataset
        # itself might be importable.
        print("Using Mock base for YFCC15MLitDataDataset structure due to missing litdata.")
        class YFCC15MLitDataDataset(ld.StreamingDataset):
             # Re-define __init__ to match the structure but use the mock base
             def __init__(
                 self,
                 *args,
                 transform: callable,
                 tokenizer,
                 image_col: str = "images",
                 text_col: str = "texts",
                 context_length: int = 77,
                 **kwargs
             ):
                 super().__init__(*args, **kwargs)
                 print("Mock YFCC15MLitDataDataset __init__ called.")
                 self.transform = transform
                 self.tokenizer = tokenizer
                 self.image_col = image_col
                 self.text_col = text_col
                 self.context_length = context_length
                 # The length is handled by the mock base class

             def __getitem__(self, index):
                  # Not needed for len() check
                  raise NotImplementedError

except ImportError as e:
    print(f"Error importing from src.dataset: {e}")
    print("Please ensure the script is run from the project root or src is in PYTHONPATH.")
    exit(1)

# Configure logging minimally
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration based on defaults ---
# From launch_ddp.sh default
PARQUET_DATA_DIR = "/tmp/hub/datasets--Kaichengalex--YFCC15M/snapshots/fc79df6bae9cbceac054a5d2598c168dfd8fe39b/data/"
# From train.py -> config -> DECLIP_YFCC15M_LITDATA_VITB32_CONFIG (likely defaults)
IMAGE_SIZE = 224 # Common default, check config if different
INTERPOLATION = "bicubic" # Common default
TOKENIZER_NAME = "openai/clip-vit-base-patch32" # Common default for CLIP ViT-B/32
IMAGE_COL = "images" # Default in dataset.py
TEXT_COL = "texts" # Default in dataset.py
CONTEXT_LENGTH = 77 # Default in dataset.py

# --- Setup components ---
logger.info("Setting up transform and tokenizer...")
# Use is_train=True as in create_dataloaders for the initial dataset instance
transform = get_transform(image_size=IMAGE_SIZE, is_train=True, interpolation=INTERPOLATION)

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    logger.info(f"Tokenizer '{TOKENIZER_NAME}' loaded.")
except Exception as e:
    logger.error(f"Failed to load tokenizer '{TOKENIZER_NAME}': {e}")
    exit(1)

# --- Instantiate Dataset ---
logger.info(f"Attempting to instantiate YFCC15MLitDataDataset...")
logger.info(f"  Input Directory: {PARQUET_DATA_DIR}")
logger.info(f"  Using {'Real' if LITDATA_AVAILABLE else 'Mock'} LitData components.")

# Check if the directory and index file exist (useful for debugging)
if not os.path.isdir(PARQUET_DATA_DIR):
    logger.warning(f"Parquet data directory not found: {PARQUET_DATA_DIR}")
elif not os.path.exists(os.path.join(PARQUET_DATA_DIR, "index.json")):
     logger.warning(f"LitData index file ('index.json') not found in {PARQUET_DATA_DIR}")
     logger.warning("The dataset length might be 0 or incorrect if using real LitData.")
     if not LITDATA_AVAILABLE:
          logger.warning("Using mock LitData, which might simulate a non-zero length even without the index.")


try:
    # Instantiate the dataset class (either real or mock structure)
    # We mimic the instantiation within create_dataloaders before splitting
    dataset = YFCC15MLitDataDataset(
        input_dir=PARQUET_DATA_DIR,
        item_loader=ParquetLoader(),
        shuffle=True, # As done in create_dataloaders before splitting
        # Pass necessary arguments for preprocessing
        transform=transform,
        tokenizer=tokenizer,
        image_col=IMAGE_COL,
        text_col=TEXT_COL,
        context_length=CONTEXT_LENGTH
    )
    logger.info("Dataset instantiated successfully.")

    # --- Get Length ---
    dataset_length = len(dataset)
    logger.info(f"Length of the dataset: {dataset_length}")

except FileNotFoundError as e:
    logger.error(f"Caught FileNotFoundError: {e}")
    logger.error("This likely means the LitData index ('index.json') is missing or the input directory is incorrect.")
    logger.error(f"Checked directory: {PARQUET_DATA_DIR}")
except Exception as e:
    logger.error(f"An unexpected error occurred during dataset instantiation or length check: {e}", exc_info=True)
