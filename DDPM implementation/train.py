"""
DIFFUSION MODEL TRAINING SCRIPT                           

This script handles:                                                        
1. Training the diffusion model                                          
2. Generating sample images during inference                             
3. Creating a GIF showing the denoising process                          

The GIF visualization shows how the model gradually removes noise          
from pure random noise to generate coherent images.                        
"""

import torch
from data import DiffSet
import pytorch_lightning as pl
from model import DiffusionModel
from torch.utils.data import DataLoader
import imageio
import glob
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def sample_gif(model, train_dataset, output_dir) -> None:
    """
    Generate a GIF showing the denoising process.
    
    This function creates a visual demonstration of how the diffusion model
    works by capturing each denoising step and assembling them into an animated GIF.
    
    Visualization Process:
    ┌──────────────────────────────────────────────────────────┐
    │  Frame 1: Pure noise (t=1000)         [random pixels]    │
    │  Frame 2: Slightly denoised (t=999)   [slight shapes]    │
    │  Frame 3: More denoised (t=998)       [vague objects]    │
    │  ...                                                     │
    │  Frame 999: Almost clean (t=2)        [clear image]      │
    │  Frame 1000: Clean image (t=1)        [final result]     │
    │  (Final frame held for 100 frames to show result)        │
    └──────────────────────────────────────────────────────────┘
    
    The output is a 3x3 grid of images, showing 9 different generated
    samples all denoising simultaneously.
    
    Args:
        model: Trained DiffusionModel
        train_dataset: Dataset (used to get image dimensions and channels)
        output_dir: Directory to save the output GIF
    """
    
    # ============================================================
    # CONFIGURATION
    # ============================================================
    # Shape of the grid of images in the GIF
    # [3, 3] means a 3x3 grid = 9 images total
    gif_shape = [3, 3]
    
    # Total number of images to generate
    sample_batch_size = gif_shape[0] * gif_shape[1]  # 3 * 3 = 9
    
    # How many frames to hold the final image at the end
    # This makes the final result visible for longer in the GIF
    n_hold_final = 100
    
    print(f"Generating {sample_batch_size} samples in a {gif_shape[0]}x{gif_shape[1]} grid...")

    # ============================================================
    # GENERATE SAMPLES (REVERSE DIFFUSION PROCESS)
    # ============================================================
    """
    Sampling Process Visualization:
    
    t=1000: [▓▓▓▓▓▓▓▓] ← pure noise
    t=900:  [▓▓▓▓▓▒▒▒] ← still very noisy
    t=700:  [▓▓▒▒░░░░] ← some structure emerging
    t=500:  [▒▒░░░   ] ← clear shapes forming
    t=300:  [░░      ] ← details appearing
    t=100:  [        ] ← nearly clean
    t=1:    [  CLEAN ] ← final image!
    
    We capture every single step to show the full transformation.
    """
    
    # Lists to store generated samples and their corresponding timesteps
    gen_samples = []
    sampled_steps = []
    
    # ============================================================
    # STEP 1: Initialize with pure random noise
    # ============================================================
    # Generate random noise following N(0, I) distribution
    # Shape: (batch_size, channels, height, width)
    x = torch.randn(
        (sample_batch_size, train_dataset.depth, train_dataset.size, train_dataset.size)
    )
    
    print(f"Starting with noise shape: {x.shape}")
    
    # ============================================================
    # STEP 2: Iteratively denoise from t=T-1 down to t=1
    # ============================================================
    # Create timesteps: [999, 998, 997, ..., 2, 1]
    # We go from t_range-1 (999) down to 1, stepping by -1
    sample_steps = torch.arange(model.t_range - 1, 0, -1)
    
    sampled_t = 0  # Track current timestep
    
    # Denoise step by step
    for t in tqdm(sample_steps, desc="Sampling"):
        # Apply one denoising step: x_t → x_(t-1)
        x = model.denoise_sample(x, t)
        sampled_t = t
        
        # Save this intermediate result
        gen_samples.append(x)
        sampled_steps.append(sampled_t)
    
    # ============================================================
    # STEP 3: Hold the final image for multiple frames
    # ============================================================
    # Add the final clean image many times so viewers can see it clearly
    for _ in range(n_hold_final):
        gen_samples.append(x)
        sampled_steps.append(sampled_t)
    
    print(f"Generated {len(gen_samples)} frames (including {n_hold_final} hold frames)")

    # ============================================================
    # IMAGE PROCESSING FOR GIF CREATION
    # ============================================================
    
    # Stack all samples into a single tensor
    # Shape: (num_frames, batch_size, channels, height, width)
    gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)
    
    # Normalize from [-1, 1] to [0, 1] range
    # The model outputs images in [-1, 1] (training range)
    # GIFs need [0, 1] or [0, 255] range
    gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2
    
    # Verify we have the right number of frames
    assert gen_samples.shape[0] == len(sampled_steps)
    
    # Convert to uint8 [0, 255] range for image format
    gen_samples = (gen_samples * 255).type(torch.uint8)
    
    # Reshape into grid format
    # Shape: (num_frames, grid_rows, grid_cols, height, width, channels)
    gen_samples = gen_samples.reshape(
        -1,                    # Number of frames
        gif_shape[0],          # Grid rows (3)
        gif_shape[1],          # Grid columns (3)
        train_dataset.size,    # Image height (32)
        train_dataset.size,    # Image width (32)
        train_dataset.depth,   # Channels (1 or 3)
    )

    # ============================================================
    # ADD TIMESTEP TEXT OVERLAY
    # ============================================================
    """
    Add text showing current timestep to the top-left image in each frame.
    
    Before:              After:
    ┌─────┬─────┬─────┐  ┌─────┬─────┬─────┐
    │     │     │     │  │ 999 │     │     │ ← timestep shown here
    ├─────┼─────┼─────┤  ├─────┼─────┼─────┤
    │     │     │     │  │     │     │     │
    ├─────┼─────┼─────┤  ├─────┼─────┼─────┤
    │     │     │     │  │     │     │     │
    └─────┴─────┴─────┘  └─────┴─────┴─────┘
    """
    
    def add_text_to_image(image, text):
        """
        Add white text on black background to an image.
        
        Args:
            image: Image tensor
            text: Text to display (the timestep)
        
        Returns:
            Image tensor with text overlay
        """
        # Create black background
        black_image = np.zeros_like(image.numpy())
        black_image = Image.fromarray(black_image, "RGB")
        
        # Draw white text
        draw = ImageDraw.Draw(black_image)
        font = ImageFont.load_default()
        draw.text((0, 0), text, (255, 255, 255), font=font)
        
        # Convert back to tensor
        black_image = torch.tensor(np.array(black_image))
        return black_image

    # Add timestep text to first image (top-left) in each frame
    for i in range(gen_samples.shape[0]):
        gen_samples[i, 0, 0] = add_text_to_image(
            gen_samples[i, 0, 0], 
            f"{sampled_steps[i]}"
        )

    # ============================================================
    # ARRANGE IMAGES INTO GRID
    # ============================================================
    """
    Convert from individual images to a single combined grid image.
    
    Input format:  (frames, rows, cols, H, W, C)
    Output format: (frames, grid_H, grid_W, C)
    
    Where grid_H = rows * H, grid_W = cols * W
    
    Example with 3x3 grid of 32x32 images:
    - Input: (1000, 3, 3, 32, 32, 3)
    - Output: (1000, 96, 96, 3)  [because 3*32=96]
    
    Visual transformation:
    ┌───┬───┬───┐
    │ 1 │ 2 │ 3 │      ┌─────────────┐
    ├───┼───┼───┤  →   │ 1  2  3     │
    │ 4 │ 5 │ 6 │      │ 4  5  6     │
    ├───┼───┼───┤      │ 7  8  9     │
    │ 7 │ 8 │ 9 │      └─────────────┘
    └───┴───┴───┘
    """
    
    def stack_samples(gen_samples, stack_dim):
        """
        Stack images along a dimension to create grid.
        
        Args:
            gen_samples: Tensor of images
            stack_dim: Dimension to stack along
        
        Returns:
            Stacked tensor
        """
        # Split into list of tensors
        gen_samples = list(torch.split(gen_samples, 1, dim=1))
        
        # Remove the split dimension
        for i in range(len(gen_samples)):
            gen_samples[i] = gen_samples[i].squeeze(1)
        
        # Concatenate along the specified dimension
        return torch.cat(gen_samples, dim=stack_dim)

    # Stack rows (dimension 2 - height)
    gen_samples = stack_samples(gen_samples, 2)
    
    # Stack columns (dimension 2 again - width, after height stacking)
    gen_samples = stack_samples(gen_samples, 2)

    # ============================================================
    # SAVE AS GIF
    # ============================================================
    # Create output directory if it doesn't exist
    output_file = f"{output_dir}/pred.gif"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Saving GIF to: {output_file}")
    
    # Save as animated GIF
    # duration=20 means 20ms per frame = 50 FPS
    imageio.mimsave(
        output_file, 
        list(gen_samples.squeeze(-1)), 
        format="GIF", 
        duration=20
    )
    
    print(f"GIF saved successfully! Total frames: {len(gen_samples)}")
    print(f"The GIF shows the denoising process from t={model.t_range-1} to t=1")


def train_model(config: dict) -> None:
    """
    Main training function for the diffusion model.
    
    This function:
    1. Sets up datasets and dataloaders
    2. Creates or loads the diffusion model
    3. Configures PyTorch Lightning trainer
    4. Trains the model
    5. Returns trained model and metadata
    
    Args:
        config: Dictionary containing training configuration:
            - diffusion_steps: Number of timesteps T (typically 1000)
            - dataset: Dataset name ("MNIST", "FashionMNIST", or "CIFAR10")
            - max_epoch: Number of training epochs
            - batch_size: Batch size for training
            - load_model: Whether to load a pretrained model
            - load_version_num: Version number to load (if load_model=True)
    
    Returns:
        tuple: (model, train_dataset, output_dir)
            - model: Trained DiffusionModel
            - train_dataset: Training dataset (for image dimensions)
            - output_dir: Directory where logs are saved
    """
    
    # ============================================================
    # OPTIONAL MODEL LOADING
    # ============================================================
    # Variables for loading a pretrained model checkpoint
    pass_version = None
    last_checkpoint = None

    if config['load_model']:
        # Set version to continue from
        pass_version = config["load_version_num"]
        
        # Find the latest checkpoint file for this version
        # Pattern: ./lightning_logs/{dataset}/version_{num}/checkpoints/*.ckpt
        checkpoint_pattern = (
            f"./lightning_logs/{config['dataset']}/"
            f"version_{config['load_version_num']}/checkpoints/*.ckpt"
        )
        last_checkpoint = glob.glob(checkpoint_pattern)[-1]
        
        print(f"Loading model from checkpoint: {last_checkpoint}")

    # ============================================================
    # CREATE DATASETS AND DATALOADERS
    # ============================================================
    """
    Dataset Loading:
    
    Train Dataset:
    ┌──────────────────────────────────────┐
    │  Load training images                │
    │  - MNIST: 60,000 images              │
    │  - FashionMNIST: 60,000 images       │
    │  - CIFAR10: 50,000 images            │
    │  Preprocessing:                      │
    │  - Resize to 32x32 (if needed)       │
    │  - Normalize to [-1, 1]              │
    └──────────────────────────────────────┘
    
    Validation Dataset:
    ┌──────────────────────────────────────┐
    │  Load test/validation images         │
    │  - MNIST: 10,000 images              │
    │  - FashionMNIST: 10,000 images       │
    │  - CIFAR10: 10,000 images            │
    │  Same preprocessing as training      │
    └──────────────────────────────────────┘
    """
    
    print(f"Loading dataset: {config['dataset']}")
    
    # Create training dataset
    train_dataset = DiffSet(True, config["dataset"])
    
    # Create validation dataset
    val_dataset = DiffSet(False, config["dataset"])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Image size: {train_dataset.size}x{train_dataset.size}")
    print(f"Image channels: {train_dataset.depth}")

    # Create data loaders
    # num_workers=4: Use 4 parallel processes for data loading
    # shuffle=True: Randomize training order each epoch
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        num_workers=4, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        num_workers=4, 
        shuffle=False
    )

    # ============================================================
    # CREATE OR LOAD MODEL
    # ============================================================
    """
    Model Creation:
    
    Parameters:
    - in_size: Total pixels (H × W × C)
    - t_range: Number of diffusion timesteps
    - img_depth: Number of channels (1 for grayscale, 3 for RGB)
    
    Example for CIFAR10:
    - in_size = 32 × 32 × 3 = 3,072
    - t_range = 1000
    - img_depth = 3
    """
    
    if config['load_model']:
        # Load from checkpoint
        print("Loading model from checkpoint...")
        model = DiffusionModel.load_from_checkpoint(
            last_checkpoint,
            in_size=train_dataset.size * train_dataset.size,
            t_range=config["diffusion_steps"],
            img_depth=train_dataset.depth,
        )
    else:
        # Create new model
        print("Creating new model...")
        model = DiffusionModel(
            train_dataset.size * train_dataset.size,
            config["diffusion_steps"],
            train_dataset.depth,
        )

    # ============================================================
    # SETUP TENSORBOARD LOGGER
    # ============================================================
    """
    TensorBoard Logger:
    
    Saves training metrics to:
    ./lightning_logs/{dataset}/version_{N}/
    
    You can view training progress with:
    $ tensorboard --logdir=lightning_logs
    
    Then open http://localhost:6006 in your browser
    """
    
    tb_logger = pl.loggers.TensorBoardLogger(
        "lightning_logs/",
        name=config["dataset"],
        version=pass_version,  # Continue from existing version or create new
    )
    
    print(f"Logging to: {tb_logger.log_dir}")

    # ============================================================
    # CREATE PYTORCH LIGHTNING TRAINER
    # ============================================================
    """
    Trainer Configuration:
    
    - max_epochs: How many times to iterate through full dataset
    - log_every_n_steps: Log metrics every N batches (10)
    - logger: TensorBoard logger for visualization
    
    The trainer handles:
    - Automatic GPU/CPU detection
    - Gradient computation and optimization
    - Training/validation loop
    - Checkpointing
    - Logging
    """
    
    trainer = pl.Trainer(
        max_epochs=config["max_epoch"], 
        log_every_n_steps=10, 
        logger=tb_logger
    )

    # ============================================================
    # TRAIN THE MODEL
    # ============================================================
    """
    Training Process:
    
    For each epoch:
      For each batch in training data:
        1. Sample random timesteps
        2. Add noise to images
        3. Predict noise with U-Net
        4. Compute MSE loss
        5. Backpropagate and update weights
      
      For each batch in validation data:
        1. Compute validation loss
        2. Log metrics
    
    Progress is automatically logged to TensorBoard.
    """
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    trainer.fit(model, train_loader, val_loader)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60 + "\n")

    return model, train_dataset, trainer.logger.log_dir


def get_config() -> dict:
    """
    Get training configuration.
    
    Returns:
        dict: Configuration dictionary with all training parameters
        
    Configuration Options:
    ┌──────────────────────────────────────────────────────────┐
    │ diffusion_steps: 1000                                    │
    │   Number of timesteps in diffusion process               │
    │   More steps = smoother denoising but slower sampling    │
    │                                                          │
    │ dataset: "CIFAR10"                                       │
    │   Options: "MNIST", "FashionMNIST", "CIFAR10"            │
    │                                                          │
    │ max_epoch: 10                                            │
    │   Number of training epochs                              │
    │   More epochs = better quality but longer training       │
    │                                                          │
    │ batch_size: 32                                           │
    │   Number of images per batch                             │
    │   Larger = faster but needs more GPU memory              │
    │                                                          │
    │ load_model: False                                        │
    │   Whether to load a pretrained checkpoint                │
    │                                                          │
    │ load_version_num: 1                                      │
    │   Which checkpoint version to load (if load_model=True)  │
    └──────────────────────────────────────────────────────────┘
    """
    return {
        "diffusion_steps": 1000,    # T in the DDPM paper
        "dataset": "CIFAR10",        # Which dataset to train on
        "max_epoch": 10,             # Training duration
        "batch_size": 32,            # Batch size for training
        "load_model": False,         # Start fresh or load checkpoint
        "load_version_num": 1,       # Checkpoint version to load
    }


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    """
    Main execution flow:
    
    1. Load configuration
    2. Train the model (or load pretrained)
    3. Generate sample images
    4. Create animated GIF showing denoising process
    
    The output GIF visually demonstrates how the diffusion
    model works by showing the gradual transformation from
    noise to coherent images.
    """
    
    print("="*60)
    print("DIFFUSION MODEL TRAINING AND SAMPLING")
    print("="*60 + "\n")
    
    # Get training configuration
    config = get_config()
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Train the model
    model, train_ds, output_dir = train_model(config)
    
    # Generate sample GIF
    print("\n" + "="*60)
    print("GENERATING SAMPLE GIF")
    print("="*60 + "\n")
    
    sample_gif(model, train_ds, output_dir)
    
    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)
    print(f"\nCheck {output_dir}/pred.gif to see the denoising process!")


# ============================================================
# USAGE NOTES
# ============================================================
"""
To run this script:
    $ python train.py

To view training progress in TensorBoard:
    $ tensorboard --logdir=lightning_logs
    Then open http://localhost:6006

To modify configuration:
    Edit the get_config() function above

To continue training from a checkpoint:
    Set load_model=True and load_version_num=<version> in config

Output files:
    - Model checkpoints: ./lightning_logs/{dataset}/version_{N}/checkpoints/
    - TensorBoard logs: ./lightning_logs/{dataset}/version_{N}/
    - Generated GIF: ./lightning_logs/{dataset}/version_{N}/pred.gif

The GIF shows:
    - A 3x3 grid of 9 generated samples
    - Each frame shows one denoising step
    - Timestep number displayed in top-left image
    - Final clean images held for 100 frames at the end
"""