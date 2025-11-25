"""
DENOISING DIFFUSION PROBABILISTIC MODEL                   

This implements the DDPM paper (Ho et al., 2020):                          
"Denoising Diffusion Probabilistic Models"                                 

Key Concepts:                                                               
┌──────────────────────────────────────────────────────────────┐           
│ FORWARD PROCESS (Adding Noise - Training Time)               │           
│                                                              │          
│  x₀ (clean) → x₁ → x₂ → ... → x_T (pure noise)               │           
│    ↓          ↓     ↓           ↓                            │           
│  + ε₁       + ε₂   + ε₃        + ε_T                         │           
│                                                              │           
│  At each step, add a small amount of Gaussian noise          │          
└──────────────────────────────────────────────────────────────┘          

┌──────────────────────────────────────────────────────────────┐          
│ REVERSE PROCESS (Removing Noise - Generation Time)           │           
│                                                              │          
│  x_T (noise) → x_T-1 → ... → x₁ → x₀ (clean image)           │           
│                                                              │           
│  Model learns to predict and remove the noise at each step   │           
└──────────────────────────────────────────────────────────────┘           
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
from modules import *


class DiffusionModel(pl.LightningModule):
    """
    Denoising Diffusion Probabilistic Model (DDPM).
    
    This model learns to gradually denoise images by predicting the noise
    added at each timestep of a diffusion process.
    
    Training: Given a clean image, add noise and predict it
    Sampling: Start from noise and iteratively denoise to generate images
    """
    
    def __init__(self, in_size, t_range, img_depth):
        """
        Args:
            in_size: Total number of pixels (height * width * channels)
            t_range: Number of diffusion timesteps (typically 1000)
            img_depth: Number of image channels (1 for grayscale, 3 for RGB)
        """
        super().__init__()
        
        # ============================================================
        # NOISE SCHEDULE PARAMETERS
        # ============================================================
        # These control how much noise is added at each timestep
        
        # β_small: Minimum noise level (at t=0, very little noise)
        self.beta_small = 1e-4  # 0.0001
        
        # β_large: Maximum noise level (at t=T, lots of noise)
        self.beta_large = 0.02  # 0.02
        
        # Total number of diffusion steps
        self.t_range = t_range
        
        # Total pixels in flattened image
        self.in_size = in_size

        # ============================================================
        # U-NET ARCHITECTURE
        # ============================================================
        # The neural network that predicts noise
        # - dim=64: Base channel dimension
        # - dim_mults=(1,2,4,8): Creates [64, 128, 256, 512] channels
        # - channels=img_depth: Input/output channels (1 or 3)
        self.unet = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=img_depth)

    def forward(self, x, t):
        """
        Forward pass: predict noise in image x at timestep t.
        
        Args:
            x: Noisy image (B, C, H, W)
            t: Timestep (B,)
        
        Returns:
            Predicted noise (B, C, H, W)
        """
        return self.unet(x, t)

    # ============================================================
    # NOISE SCHEDULE FUNCTIONS
    # ============================================================
    # These implement the variance schedule from the DDPM paper
    
    def beta(self, t):
        """
        β_t: Amount of noise added at timestep t (variance of noise).
        
        Linear interpolation between β_small and β_large:
        
        β_t = β_small + (t / T) * (β_large - β_small)
        
        Visual representation:
        
        Noise Level (β_t)
        ↑
        │                                    • β_large = 0.02
        │                                 •
        │                              •
        │                           •
        │                        •
        │                     •
        │                  •
        │               •
        │            •
        │         •
        │      •
        │   •
        │ • β_small = 0.0001
        └──────────────────────────────────────► Time (t)
        0                                      T=1000
        
        At t=0: Almost no noise (β ≈ 0.0001)
        At t=T: Lots of noise (β ≈ 0.02)
        
        Args:
            t: Timestep (integer from 0 to t_range)
        
        Returns:
            β_t: Noise level at timestep t
        """
        return self.beta_small + (t / self.t_range) * (self.beta_large - self.beta_small)

    def alpha(self, t):
        """
        α_t: Proportion of signal retained at timestep t.
        
        α_t = 1 - β_t
        
        Interpretation:
        - If β_t = 0.02 (2% noise added), then α_t = 0.98 (98% signal retained)
        - Higher t → higher β_t → lower α_t → less signal retained
        
        Signal Retention (α_t)
        ↑
        │ • ≈ 1.0 (almost all signal)
        │   •
        │     •
        │       •
        │         •
        │           •
        │             •
        │               •
        │                 •
        │                   •
        │                     • ≈ 0.98 (most signal)
        └──────────────────────────────────────► Time (t)
        0                                      T
        
        Args:
            t: Timestep
        
        Returns:
            α_t: Signal retention at timestep t
        """
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        """
        ᾱ_t (alpha-bar): Cumulative product of alphas from 0 to t.
        
        ᾱ_t = ∏(α_s) for s=0 to t = α_0 × α_1 × α_2 × ... × α_t
        
        This represents the TOTAL signal retained after t steps of noising.
        
        Example with simplified numbers:
        - α_0 = 0.99, α_1 = 0.98, α_2 = 0.97
        - ᾱ_0 = 0.99
        - ᾱ_1 = 0.99 × 0.98 = 0.9702
        - ᾱ_2 = 0.99 × 0.98 × 0.97 = 0.9411
        
        Cumulative Signal (ᾱ_t)
        ↑
        │ • 1.0 (all signal)
        │   •
        │      •
        │         •• (exponential decay)
        │             ••
        │                ••
        │                   •••
        │                       ••••
        │                            •••••
        │                                  ••••••••
        │                                           •••••••••••••
        └──────────────────────────────────────────────────────► Time (t)
        0                                                      T
        
        At t=0: ᾱ_0 ≈ 1.0 (clean image)
        At t=T: ᾱ_T ≈ 0.0 (pure noise)
        
        This allows us to jump directly to any timestep t without
        iterating through all previous timesteps!
        
        Args:
            t: Timestep
        
        Returns:
            ᾱ_t: Cumulative signal retention up to timestep t
        """
        return math.prod([self.alpha(j) for j in range(t)])

    # ============================================================
    # TRAINING (ALGORITHM 1 FROM DDPM PAPER)
    # ============================================================
    
    def get_loss(self, batch, batch_idx):
        """
        Training loss computation (Algorithm 1 from Ho et al., 2020).
        
        Training Process:
        ┌──────────────────────────────────────────────────────────┐
        │  1. Take a clean image x₀                                │
        │                                                          │
        │  2. Sample random timestep t ~ Uniform(0, T)             │
        │                                                          │
        │  3. Sample noise ε ~ N(0, I)                             │
        │                                                          │
        │  4. Create noisy image:                                  │
        │     x_t = √(ᾱ_t) * x₀ + √(1 - ᾱ_t) * ε                   │
        │                                                          │
        │  5. Predict the noise: ε̂ = model(x_t, t)                 │
        │                                                          │
        │  6. Compute loss: L = MSE(ε̂, ε)                          │
        │                                                          │
        │  The model learns to predict the noise we added!         │
        └──────────────────────────────────────────────────────────┘
        
        Visualization of the noising formula:
        
        x_t = √(ᾱ_t) * x₀ + √(1 - ᾱ_t) * ε
              ↑              ↑
              │              └─ Noise component (scaled)
              └─ Signal component (scaled)
        
        As t increases:
        - √(ᾱ_t) decreases (less signal)
        - √(1 - ᾱ_t) increases (more noise)
        
        At t=0:   √(ᾱ_0)=1,  √(1-ᾱ_0)≈0  → x_0 = x₀ (clean)
        At t=500: √(ᾱ_t)≈0.7, √(1-ᾱ_t)≈0.7 → 50% signal, 50% noise
        At t=T:   √(ᾱ_T)≈0,  √(1-ᾱ_T)≈1  → x_T ≈ ε (pure noise)
        
        Args:
            batch: Clean images (B, C, H, W)
        
        Returns:
            loss: MSE between predicted noise and actual noise
        """
        # ============================================================
        # STEP 1: Sample random timesteps for each image
        # ============================================================
        # Each image in the batch gets a different random timestep
        # Shape: (batch_size,)
        ts = torch.randint(0, self.t_range, [batch.shape[0]], device=self.device)
        
        # ============================================================
        # STEP 2: Sample random noise (same shape as images)
        # ============================================================
        # ε ~ N(0, I): Gaussian noise with mean=0, std=1
        # Shape: (batch_size, channels, height, width)
        epsilons = torch.randn(batch.shape, device=self.device)
        
        # ============================================================
        # STEP 3: Create noisy images using the forward diffusion formula
        # ============================================================
        noise_imgs = []
        for i in range(len(ts)):
            # Get cumulative signal retention for this timestep
            a_hat = self.alpha_bar(ts[i])
            
            # Apply forward diffusion formula:
            # x_t = √(ᾱ_t) * x₀ + √(1 - ᾱ_t) * ε
            noisy_image = (
                math.sqrt(a_hat) * batch[i] +           # Signal component
                math.sqrt(1 - a_hat) * epsilons[i]      # Noise component
            )
            noise_imgs.append(noisy_image)
        
        # Stack into batch: list of tensors → single tensor
        noise_imgs = torch.stack(noise_imgs, dim=0)
        
        # ============================================================
        # STEP 4: Predict the noise using the U-Net
        # ============================================================
        # Input: noisy images x_t and timesteps t
        # Output: predicted noise ε̂
        e_hat = self.forward(noise_imgs, ts)
        
        # ============================================================
        # STEP 5: Compute loss (MSE between predicted and actual noise)
        # ============================================================
        # Reshape to (batch_size, total_pixels) for loss computation
        # The model is learning: given x_t and t, predict ε
        loss = nn.functional.mse_loss(
            e_hat.reshape(-1, self.in_size),      # Predicted noise (flattened)
            epsilons.reshape(-1, self.in_size)    # Actual noise (flattened)
        )
        
        return loss

    # ============================================================
    # SAMPLING (ALGORITHM 2 FROM DDPM PAPER)
    # ============================================================
    
    def denoise_sample(self, x, t):
        """
        Single denoising step: remove noise to go from x_t to x_(t-1).
        
        This is the inner loop of Algorithm 2 from (Ho et al., 2020).
        
        Reverse Process (Sampling):
        ┌──────────────────────────────────────────────────────────┐
        │                                                          │
        │  Start: x_T ~ N(0, I)  (pure Gaussian noise)             │
        │           ↓                                              │
        │  Step T:  x_T → x_(T-1)  (denoise a little)              │
        │           ↓                                              │
        │  Step T-1: x_(T-1) → x_(T-2)  (denoise more)             │
        │           ↓                                              │
        │           ...                                            │
        │           ↓                                              │
        │  Step 1:  x_1 → x_0  (final denoising)                   │
        │           ↓                                              │
        │  Result: x_0 (clean generated image!)                    │
        │                                                          │
        └──────────────────────────────────────────────────────────┘
        
        Denoising Formula:
        
        x_(t-1) = (1/√α_t) * [x_t - ((1-α_t)/√(1-ᾱ_t)) * ε̂ ] + σ_t * z
                  ↑            ↑                              ↑
                  │            │                              └─ Random noise
                  │            └─ Subtract predicted noise        (for t>1)
                  └─ Rescale
        
        where:
        - ε̂ = model(x_t, t)  (predicted noise)
        - σ_t = √β_t  (noise to add for stochasticity)
        - z ~ N(0, I)  (random noise, only added if t > 1)
        
        Why add noise z?
        - Makes the reverse process stochastic (not deterministic)
        - Helps explore different possible images
        - Not needed at t=1 (final step should be deterministic)
        
        Args:
            x: Current noisy image x_t (B, C, H, W)
            t: Current timestep (scalar tensor)
        
        Returns:
            x: Denoised image x_(t-1) (B, C, H, W)
        """
        with torch.no_grad():  # No gradients needed during sampling
            # ============================================================
            # STEP 1: Sample random noise (if not the final step)
            # ============================================================
            if t > 1:
                # Add stochastic noise for all steps except the last
                z = torch.randn(x.shape)
            else:
                # Final step (t=1 → t=0) should be deterministic
                z = 0
            
            # ============================================================
            # STEP 2: Predict the noise in x_t
            # ============================================================
            # Repeat timestep t for each image in the batch
            # t.view(1).repeat(x.shape[0]) creates tensor [t, t, ..., t]
            e_hat = self.forward(x, t.view(1).repeat(x.shape[0]))
            
            # ============================================================
            # STEP 3: Compute denoising transformation
            # ============================================================
            # Formula: x_(t-1) = (1/√α_t) * [x_t - ((1-α_t)/√(1-ᾱ_t)) * ε̂] + √β_t * z
            
            # Pre-scale factor: 1/√α_t
            # This rescales because we're "undoing" the scaling from forward process
            pre_scale = 1 / math.sqrt(self.alpha(t))
            
            # Noise scale: (1 - α_t) / √(1 - ᾱ_t)
            # This determines how much of the predicted noise to subtract
            e_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
            
            # Posterior standard deviation: √β_t * z
            # Random noise to maintain stochasticity
            post_sigma = math.sqrt(self.beta(t)) * z
            
            # Apply the denoising formula
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            
            return x

    # ============================================================
    # PYTORCH LIGHTNING TRAINING HOOKS
    # ============================================================
    
    def training_step(self, batch, batch_idx):
        """
        Called for each training batch.
        
        Args:
            batch: Clean images from dataloader
            batch_idx: Index of current batch
        
        Returns:
            loss: Training loss (used for backpropagation)
        """
        loss = self.get_loss(batch, batch_idx)
        # Log to tensorboard/wandb
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Called for each validation batch.
        
        Args:
            batch: Clean images from validation dataloader
            batch_idx: Index of current batch
        """
        loss = self.get_loss(batch, batch_idx)
        # Log validation loss
        self.log("val/loss", loss)
        return

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        
        Returns:
            optimizer: Adam optimizer with learning rate 2e-4
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer


# ============================================================
# USAGE EXAMPLE
# ============================================================
"""
# ──────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────

from pytorch_lightning import Trainer

# Create model
model = DiffusionModel(
    in_size=32*32*3,  # For 32x32 RGB images
    t_range=1000,      # 1000 diffusion steps
    img_depth=3        # RGB channels
)

# Train
trainer = Trainer(max_epochs=100, accelerator='gpu')
trainer.fit(model, train_dataloader)

# ──────────────────────────────────────────────────────────
# SAMPLING (GENERATING NEW IMAGES)
# ──────────────────────────────────────────────────────────

model.eval()

# Start with pure noise
x = torch.randn(4, 3, 32, 32)  # 4 images, RGB, 32x32

# Iteratively denoise from t=1000 down to t=0
for t in reversed(range(1, model.t_range)):
    x = model.denoise_sample(x, torch.tensor(t))

# x now contains 4 generated images!

# ──────────────────────────────────────────────────────────
# WHY DOES THIS WORK?
# ──────────────────────────────────────────────────────────

Training teaches the model:
  "Given a noisy image at timestep t, predict the noise"

Sampling uses this knowledge in reverse:
  1. Start with pure noise (t=1000)
  2. Predict the noise at t=1000 and subtract it → get x_999
  3. Predict the noise at t=999 and subtract it → get x_998
  4. ... repeat 1000 times ...
  5. End with clean image (t=0)

Each step removes a little bit of noise, gradually
revealing a coherent image from random noise!
"""