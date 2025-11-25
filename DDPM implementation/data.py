import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CelebA
from torchvision import transforms


class DiffSet(Dataset):
    """
    Custom Dataset wrapper for diffusion models.
    
    This class prepares standard vision datasets (MNIST, FashionMNIST, CIFAR10)
    for use in diffusion model training by:
    1. Loading the dataset
    2. Resizing images to a consistent size (32x32)
    3. Normalizing pixel values to [-1, 1] range
    
    The [-1, 1] normalization is crucial for diffusion models because:
    - It centers the data around 0
    - It matches the output range of tanh activation (often used in diffusion)
    - It provides better numerical stability during training
    """
    
    def __init__(self, train, dataset_name):
        """
        Initialize the diffusion dataset.
        
        Args:
            train (bool): If True, load training set. If False, load test set.
            dataset_name (str): Name of dataset - "MNIST", "FashionMNIST", or "CIFAR10"
        """

        # ============================================================
        # DATASET CONFIGURATION MAPPING
        # ============================================================
        # Maps dataset name to: (Dataset Class, Image Size, Channels)
        ds_mapping = {
            "MNIST": (MNIST, 32, 1),           # Grayscale handwritten digits
            "FashionMNIST": (FashionMNIST, 32, 1),  # Grayscale clothing items
            "CIFAR10": (CIFAR10, 32, 3),       # RGB natural images
        }

        # ============================================================
        # LOAD AND TRANSFORM DATASET
        # ============================================================
        # Basic transform: convert PIL Image to PyTorch tensor
        # This converts pixel values from [0, 255] to [0, 1]
        t = transforms.Compose([transforms.ToTensor()])
        
        # Unpack dataset configuration
        ds, img_size, channels = ds_mapping[dataset_name]
        
        # Download and load the dataset
        # - "./data" is where datasets will be stored
        # - download=True will download if not present
        # - train determines train/test split
        ds = ds("./data", download=True, train=train, transform=t)

        # Store dataset and metadata as instance variables
        self.ds = ds
        self.dataset_name = dataset_name
        self.size = img_size      # 32 (all datasets resized to 32x32)
        self.depth = channels     # 1 for grayscale, 3 for RGB

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.ds)

    def __getitem__(self, item):
        """
        Get a single preprocessed sample from the dataset.
        
        Args:
            item (int): Index of the sample to retrieve
            
        Returns:
            torch.Tensor: Preprocessed image tensor with values in [-1, 1]
        
        Processing pipeline:
        
        For MNIST/FashionMNIST (28x28 → 32x32):
        ┌────────────────┐      ┌────────────────────┐
        │   28x28 img    │      │     32x32 img      │
        │  ┌──────────┐  │      │  ┌──────────────┐  │
        │  │          │  │ PAD  │  │              │  │
        │  │  MNIST   │  │ ───► │  │    MNIST     │  │
        │  │          │  │      │  │              │  │
        │  └──────────┘  │      │  └──────────────┘  │
        └────────────────┘      └────────────────────┘
           (add 2px border around all sides)
        
        For CIFAR10 (already 32x32):
        No padding needed!
        
        Normalization (all datasets):
        [0, 1] range ───────────────► [-1, 1] range
        
        Original:  0.0 ═══════ 0.5 ═══════ 1.0
                   (black)   (gray)   (white)
                              ↓
        After:    -1.0 ═══════ 0.0 ═══════ 1.0
                  (black)   (gray)   (white)
        
        Formula: new_value = (old_value * 2.0) - 1.0
        """
        
        # Get the image from the underlying dataset
        # ds[item] returns (image, label), we only need the image [0]
        ds_item = self.ds[item][0]

        # ============================================================
        # STEP 1: ENSURE CONSISTENT IMAGE SIZE (32x32)
        # ============================================================
        if self.dataset_name == "MNIST" or self.dataset_name == "FashionMNIST":
            # MNIST/FashionMNIST are 28x28, need to pad to 32x32
            # Add 2 pixels of padding on each side (top, bottom, left, right)
            pad = transforms.Pad(2)
            data = pad(ds_item)  # 28x28 → 32x32
        else:
            # CIFAR10 is already 32x32, no padding needed
            data = ds_item
        
        # ============================================================
        # STEP 2: NORMALIZE TO [-1, 1] RANGE
        # ============================================================
        # At this point, data is in [0, 1] range (from ToTensor())
        # Transform to [-1, 1] range for diffusion model training
        # 
        # Mathematical transformation:
        #   Input range:  [0, 1]
        #   Multiply by 2: [0, 2]
        #   Subtract 1:   [-1, 1]
        #
        # Example values:
        #   0.0 (black)  → (0.0 * 2) - 1 = -1.0
        #   0.5 (gray)   → (0.5 * 2) - 1 =  0.0
        #   1.0 (white)  → (1.0 * 2) - 1 =  1.0
        data = (data * 2.0) - 1.0
        
        return data


# ============================================================
# USAGE EXAMPLE
# ============================================================
# 
# # Create training dataset
# train_dataset = DiffSet(train=True, dataset_name="MNIST")
# 
# # Create test dataset
# test_dataset = DiffSet(train=False, dataset_name="CIFAR10")
# 
# # Use with DataLoader
# from torch.utils.data import DataLoader
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# 
# # Get a batch
# for batch in train_loader:
#     # batch shape: (batch_size, channels, height, width)
#     # batch values: in range [-1, 1]
#     print(batch.shape)  # e.g., torch.Size([64, 1, 32, 32]) for MNIST
#     break
# ============================================================