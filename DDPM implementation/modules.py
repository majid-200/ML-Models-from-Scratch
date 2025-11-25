"""
DIFFUSION MODEL U-NET ARCHITECTURE                        

This file implements a U-Net architecture for diffusion models.           
Key components:                                                           
1. Time embedding (tells the model what noise level we're at)           
2. Encoder (downsampling path) - compresses image                       
3. Bottleneck (middle) - processes compressed representation            
4. Decoder (upsampling path) - reconstructs image                       
5. Skip connections - preserves fine details                            

The U-Net predicts noise to remove from a noisy image at time step t      
"""

import math
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

# ============================================================
# HELPER FUNCTIONS
# ============================================================
# Small utility functions used throughout the code

def exists(x):
    """Check if a value is not None."""
    return x is not None


def default(val, d):
    """
    Return val if it exists, otherwise return d (or d() if d is callable).
    
    Example:
        default(None, 10) -> 10
        default(5, 10) -> 5
        default(None, lambda: expensive_computation()) -> calls function only if needed
    """
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_tuple(t, length=1):
    """
    Ensure input is a tuple of specified length.
    
    Example:
        cast_tuple(3, length=4) -> (3, 3, 3, 3)
        cast_tuple((1, 2), length=2) -> (1, 2)
    """
    if isinstance(t, tuple):
        return t
    return ((t,) * length)


def identity(t, *args, **kwargs):
    """Identity function - returns input unchanged. Used as a no-op placeholder."""
    return t

# ============================================================
# UPSAMPLING AND DOWNSAMPLING MODULES
# ============================================================

def Upsample(dim, dim_out=None):
    """
    Upsampling module: doubles spatial dimensions (H, W).
    
    Visual representation:
    Input:  (B, dim, H, W)
            ┌─────────┐
            │  8x8    │
            │  image  │
            └─────────┘
                ↓ (nearest neighbor interpolation)
            ┌───────────────┐
            │    16x16      │  ← pixels duplicated
            │    image      │
            └───────────────┘
                ↓ (conv to refine)
    Output: (B, dim_out, H*2, W*2)
            ┌───────────────┐
            │    16x16      │
            │   refined     │
            └───────────────┘
    
    Steps:
    1. nn.Upsample: doubles size using nearest neighbor (each pixel → 2x2 block)
    2. Conv2d: refines the upsampled features with 3x3 convolution
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    """
    Downsampling module: halves spatial dimensions (H, W).
    
    Visual representation:
    Input:  (B, C, H, W)
            ┌───────────────┐
            │ A B │ C D │...│  16x16 image
            │ E F │ G H │...│  with C channels
            ├─────┼─────┤   │
            │ I J │ K L │...│
            │ M N │ O P │...│
            └───────────────┘
                ↓ (rearrange into 2x2 blocks)
            ┌───────────────┐
            │[ABEF][CDGH]...│  8x8 image
            │[IJMN][KLOP]...│  with C*4 channels
            └───────────────┘
                ↓ (1x1 conv to adjust channels)
    Output: (B, dim_out, H/2, W/2)
            ┌───────────────┐
            │   8x8 image   │
            │ dim_out chnls │
            └───────────────┘
    
    This is a "space-to-depth" operation:
    - Takes 2x2 pixel patches and stacks them as channels
    - Reduces spatial size by 2, increases channels by 4
    - Then uses 1x1 conv to map C*4 channels → dim_out channels
    """
    return nn.Sequential(
        # 'b c (h p1) (w p2) -> b (c p1 p2) h w' means:
        # Split height into patches of size p1=2, width into patches of size p2=2
        # Stack the 4 pixels in each 2x2 patch as additional channels
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )


# ============================================================
# NORMALIZATION
# ============================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (simpler than LayerNorm).
    
    Instead of normalizing to mean=0, std=1, this normalizes the vector
    to have RMS (root mean square) = 1, then scales by learnable parameter.
    
    Formula: RMSNorm(x) = (x / RMS(x)) * g * sqrt(dim)
    where RMS(x) = sqrt(mean(x²))
    
    Why use it?
    - Simpler than LayerNorm (no mean subtraction)
    - Works well in practice for transformers/diffusion models
    - Faster computation
    """
    def __init__(self, dim):
        super().__init__()
        # Learnable scale parameter (one per channel)
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        # F.normalize with dim=1 normalizes across the channel dimension
        # Multiplying by sqrt(num_channels) maintains scale
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


# ============================================================
# SINUSOIDAL POSITIONAL EMBEDDINGS (TIME ENCODING)
# ============================================================

class SinusoidalPosEmb(nn.Module):
    """
    Encodes the timestep 't' into a high-dimensional vector using sine/cosine.
    
    WHY? The diffusion model needs to know "how noisy is the current image?"
    - t=0: clean image (no noise)
    - t=999: pure noise
    
    We encode this scalar 't' into a rich vector representation using
    sinusoidal functions at different frequencies (like in Transformers).
    
    Mathematical intuition:
    ┌────────────────────────────────────────────────────────┐
    │  Time t=500 is encoded as:                             │
    │  [sin(500/10000^0), cos(500/10000^0),                  │
    │   sin(500/10000^(2/dim)), cos(500/10000^(2/dim)),      │
    │   sin(500/10000^(4/dim)), cos(500/10000^(4/dim)),      │
    │   ...]                                                 │
    │                                                        │
    │  Different frequencies capture different "scales"      │
    │  of time information (fine-grained and coarse-grained) │
    └────────────────────────────────────────────────────────┘
    
    Formula from "Attention is All You Need":
        PE(pos, 2i)   = sin(pos / 10000^(2i/dim))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B,) containing timestep values
        
        Returns:
            Tensor of shape (B, dim) with sinusoidal embeddings
        """
        device = x.device
        half_dim = self.dim // 2
        
        # Compute frequency scaling factors in log space for numerical stability
        # emb = [1/10000^(0/half_dim), 1/10000^(1/half_dim), ..., 1/10000^((half_dim-1)/half_dim)]
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # Multiply timestep by each frequency
        # x[:, None] shape: (B, 1)
        # emb[None, :] shape: (1, half_dim)
        # Result shape: (B, half_dim)
        emb = x[:, None] * emb[None, :]
        
        # Apply sin and cos, then concatenate
        # sin(emb) and cos(emb) each have shape (B, half_dim)
        # After concatenation: (B, dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        return emb
    
# ============================================================
# BUILDING BLOCKS
# ============================================================

class Block(nn.Module):
    """
    Basic convolutional block: Conv → GroupNorm → SiLU activation.
    
    Optionally modulates features using time embedding (scale and shift).
    
    Architecture:
    ┌─────────────────────────────────────────┐
    │  Input: (B, dim, H, W)                  │
    │    ↓                                    │
    │  Conv 3x3 (extracts features)           │
    │    ↓                                    │
    │  GroupNorm (normalizes)                 │
    │    ↓                                    │
    │  [Optional] Scale & Shift with time emb │
    │    ↓                                    │
    │  SiLU activation (smooth non-linearity) │
    │    ↓                                    │
    │  Output: (B, dim_out, H, W)             │
    └─────────────────────────────────────────┘
    
    The scale & shift operation (when time embedding is provided):
        output = normalized_features * (scale + 1) + shift
    This allows the time embedding to modulate the features.
    """
    def __init__(self, dim, dim_out, groups=8):
        """
        Args:
            dim: Input channels
            dim_out: Output channels
            groups: Number of groups for GroupNorm (typically 8)
        """
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()  # SiLU(x) = x * sigmoid(x), smooth activation

    def forward(self, x, scale_shift=None):
        """
        Args:
            x: Input features (B, dim, H, W)
            scale_shift: Optional tuple of (scale, shift) from time embedding
        """
        x = self.proj(x)
        x = self.norm(x)

        # Apply time-dependent modulation if provided
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """
    Residual block with time embedding conditioning.
    
    Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │  Input x ──────────────────────────────────┐             │
    │    │                                       │             │
    │    │                                       │             │
    │    ├─→ Block 1 (with time modulation) ─→   │             │
    │    │                                       │             │
    │    └─→ Block 2 ─────────────────────────→ ADD ─→ Output  │
    │                                           │              │
    │  Time embedding ─→ MLP ─→ scale & shift ──┘              │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
    
    The residual connection helps gradients flow during training and
    allows the network to learn incremental refinements.
    
    Time embedding processing:
    1. Time embedding passes through MLP (SiLU → Linear)
    2. Output is split into 'scale' and 'shift'
    3. These modulate the features in Block 1
    """
    # Everything after the * must be passed as a keyword argument (not positional).
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        """
        Args:
            dim: Input channels
            dim_out: Output channels
            time_emb_dim: Dimension of time embedding (if using time conditioning)
            groups: Number of groups for GroupNorm
        """
        super().__init__()
        
        # MLP to process time embedding into scale and shift parameters
        # Output size is dim_out * 2 because we split it into scale and shift
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        
        # If input and output dimensions differ, use 1x1 conv for residual
        # Otherwise, use identity (no transformation needed)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        """
        Args:
            x: Input features (B, dim, H, W)
            time_emb: Time embedding (B, time_emb_dim)
        """
        scale_shift = None
        
        if exists(self.mlp) and exists(time_emb):
            # Process time embedding
            time_emb = self.mlp(time_emb)  # (B, dim_out * 2)
            
            # Reshape for broadcasting with spatial dimensions
            # (B, dim_out * 2) → (B, dim_out * 2, 1, 1)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            
            # Split into scale and shift
            # Each will have shape (B, dim_out, 1, 1)
            scale_shift = time_emb.chunk(2, dim=1)

        # Apply blocks with time modulation
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        # Add residual connection
        return h + self.res_conv(x)


# ============================================================
# ATTENTION MECHANISM
# ============================================================

class Attend(nn.Module):
    """
    Computes scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T/√d)V
    
    Visual representation of attention:
    
    Query (what am I looking for?)
      ↓
    ┌─────────────────────────────────────────────┐
    │  Q · K^T = similarity scores                │  ← Key (what do I contain?)
    │                                             │
    │  [0.1  0.8  0.05  0.05]  ← attention to     │
    │                            different pixels │
    └─────────────────────────────────────────────┘
      ↓ softmax (normalize to sum to 1)
    ┌─────────────────────────────────────────────┐
    │  [0.05  0.85  0.05  0.05] · V = output      │  ← Value (what to output?)
    │                                             │
    │  Weighted sum of values based on attention  │
    └─────────────────────────────────────────────┘
    
    This allows each pixel to gather information from other pixels
    based on learned similarity patterns.
    """
    def __init__(self, dropout=0.):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        """
        Einstein notation guide:
            b - batch
            h - number of attention heads
            i, j - sequence positions (source and target)
            d - feature dimension per head
        
        Args:
            q: Query (B, heads, seq_len, dim_head)
            k: Key   (B, heads, seq_len, dim_head)
            v: Value (B, heads, seq_len, dim_head)
        
        Returns:
            Attended output (B, heads, seq_len, dim_head)
        """
        scale = q.shape[-1] ** -0.5  # 1/√d for scaled dot-product attention
        
        # Compute similarity: Q @ K^T
        # "b h i d, b h j d -> b h i j" means:
        # For each batch and head, compute dot product between
        # all pairs of positions (i, j)
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale
        
        # Convert similarities to attention weights via softmax
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Aggregate values using attention weights: Attention @ V
        # "b h i j, b h j d -> b h i d" means:
        # For each query position i, take weighted sum of all value positions j
        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)
        
        return out


class Attention(nn.Module):
    """
    Multi-head self-attention for images.
    
    Process:
    1. Normalize input features
    2. Project to Query, Key, Value
    3. Split into multiple attention heads
    4. Reshape spatial dimensions into sequence (flatten H×W)
    5. Compute attention for each head
    6. Concatenate heads and project to output
    
    Visual flow:
    Input: (B, C, H, W)
      ↓ normalize
      ↓ project to Q, K, V
    (B, C, H, W) → (B, heads, H×W, dim_head)
      ↓ attend (each pixel attends to all other pixels)
    (B, heads, H×W, dim_head)
      ↓ reshape back
    (B, C, H, W) ← output
    
    Why use attention in diffusion models?
    - Captures long-range dependencies (distant pixels can influence each other)
    - Complements convolutions (which are local)
    - Helps with coherent structure and fine details
    """
    def __init__(self, dim, heads=4, dim_head=32):
        """
        Args:
            dim: Input channel dimension
            heads: Number of attention heads
            dim_head: Dimension per head
        """
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend()

        # Single conv to project input → Q, K, V (concatenated)
        # Output has hidden_dim * 3 channels (will be split into Q, K, V)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        
        # Project concatenated head outputs back to original dimension
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        """
        Args:
            x: Input features (B, C, H, W)
        
        Returns:
            Attended features (B, C, H, W)
        """
        b, c, h, w = x.shape
        
        # Normalize
        x = self.norm(x)
        
        # Project to Q, K, V and split
        qkv = self.to_qkv(x).chunk(3, dim=1)
        
        # Reshape for multi-head attention
        # 'b (h c) x y -> b h (x y) c' means:
        # - Split channels into heads: (h * dim_head) → h separate heads
        # - Flatten spatial dims: (H, W) → single sequence of length H×W
        # Result: (B, heads, H×W, dim_head)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads), qkv)
        
        # Apply attention
        out = self.attend(q, k, v)
        
        # Reshape back to spatial format
        # 'b h (x y) d -> b (h d) x y' means:
        # - Unflatten sequence back to spatial: H×W → (H, W)
        # - Concatenate heads: h separate heads → (h * dim_head) channels
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        
        # Project back to original dimension
        return self.to_out(out)
    
# ============================================================
# U-NET ARCHITECTURE
# ============================================================

class Unet(nn.Module):
    """
    U-Net architecture for diffusion models.
    
    Overall Structure:
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Input Image (B, 3, 32, 32)                                 │
    │    ↓                                                        │
    │  ┌──────────┐  ──→ skip connection ──→  ┌──────────┐        │
    │  │ Encoder  │                           │ Decoder  │        │
    │  │ Block 1  │  ──→ skip connection ──→  │ Block 1  │        │
    │  └────↓─────┘                           └─────↑────┘        │
    │  ┌──────────┐  ──→ skip connection ──→  ┌──────────┐        │
    │  │ Encoder  │                           │ Decoder  │        │
    │  │ Block 2  │  ──→ skip connection ──→  │ Block 2  │        │
    │  └────↓─────┘                           └─────↑────┘        │
    │       ↓                                       ↑             │
    │  ┌──────────┐                          ┌──────────┐         │
    │  │Bottleneck│  ← processes compressed  │ Upsample │         │
    │  │  (Mid)   │     representation       │          │         │
    │  └──────────┘                          └──────────┘         │
    │                                                             │
    │  Time Embedding (t) → MLP → injected into all blocks        │
    │                                                             │
    │  Output: Predicted Noise (B, 3, 32, 32)                     │
    └─────────────────────────────────────────────────────────────┘
    
    Key Design Choices:
    1. Skip connections preserve fine details from encoder to decoder
    2. Time embedding tells each layer "how noisy is the image?"
    3. Attention layers capture long-range dependencies
    4. Gradually downsample then upsample (multi-scale processing)
    """
    
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8
    ):
        """
        Args:
            dim: Base channel dimension (all other dims are multiples of this)
            init_dim: Initial feature dimension after first conv (default: same as dim)
            out_dim: Output channels (default: same as input channels)
            dim_mults: Channel multipliers for each resolution level
                       e.g., (1, 2, 4, 8) with dim=64 gives [64, 128, 256, 512]
            channels: Number of input image channels (3 for RGB, 1 for grayscale)
            resnet_block_groups: Number of groups for GroupNorm in ResNet blocks
        """
        super().__init__()

        # ============================================================
        # DIMENSION SETUP
        # ============================================================
        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        
        # Initial convolution: converts input image to feature map
        # 7x7 conv with padding=3 maintains spatial dimensions
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        # Build channel dimensions for each level
        # If dim=64 and dim_mults=(1, 2, 4, 8):
        # dims = [64, 64, 128, 256, 512]
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        
        # Create (input, output) pairs for each level
        # in_out = [(64, 64), (64, 128), (128, 256), (256, 512)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Partial function for creating ResNet blocks with consistent group norm settings
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # ============================================================
        # TIME EMBEDDINGS
        # ============================================================
        # Time dimension is 4x the base dimension
        # This rich representation helps the model understand noise levels
        time_dim = dim * 4
        
        # Create sinusoidal positional encoding for timestep
        sinu_pos_emb = SinusoidalPosEmb(dim)
        
        # MLP to process time embedding:
        # dim → time_dim → time_dim
        # This transforms the sinusoidal encoding into a learned representation
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, time_dim),
            nn.GELU(),  # Smooth activation function
            nn.Linear(time_dim, time_dim)
        )

        # ============================================================
        # ENCODER (DOWNSAMPLING PATH)
        # ============================================================
        """
        Encoder structure at each level:
        ┌────────────────────────────────────────┐
        │  ResBlock 1 (with time embedding)      │
        │    ↓         (save to skip list)       │
        │  ResBlock 2 (with time embedding)      │
        │    ↓         (save to skip list)       │
        │  Attention                             │
        │    ↓                                   │
        │  Downsample (reduce spatial size)      │
        └────────────────────────────────────────┘
        
        Each level processes features, then reduces spatial dimensions by 2x
        while increasing channel dimensions.
        """
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out)) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Attention(dim_in),
                # Last layer doesn't downsample, just changes channels
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        # ============================================================
        # BOTTLENECK (MIDDLE)
        # ============================================================
        """
        Bottleneck processes the most compressed representation:
        ┌────────────────────────────────────────┐
        │  ResBlock (with time embedding)        │
        │    ↓                                   │
        │  Attention (capture global context)    │
        │    ↓                                   │
        │  ResBlock (with time embedding)        │
        └────────────────────────────────────────┘
        
        This is the "thinking" part at the smallest spatial resolution
        but highest channel dimension.
        """
        mid_dim = dims[-1]  # Highest channel dimension (e.g., 512)
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Attention(mid_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # ============================================================
        # DECODER (UPSAMPLING PATH)
        # ============================================================
        """
        Decoder structure at each level (reverse of encoder):
        ┌────────────────────────────────────────┐
        │  Concatenate with skip connection      │
        │    ↓                                   │
        │  ResBlock 1 (with time embedding)      │
        │    ↓                                   │
        │  Concatenate with skip connection      │
        │    ↓                                   │
        │  ResBlock 2 (with time embedding)      │
        │    ↓                                   │
        │  Attention                             │
        │    ↓                                   │
        │  Upsample (increase spatial size)      │
        └────────────────────────────────────────┘
        
        Skip connections bring back fine details from encoder.
        Input channels = dim_out + dim_in (current features + skip connection)
        """
        for ind, ((dim_in, dim_out)) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                # dim_out + dim_in because we concatenate skip connection
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Attention(dim_out),
                # Last layer doesn't upsample, just changes channels
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        # ============================================================
        # FINAL OUTPUT LAYERS
        # ============================================================
        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)

        # Final processing before output
        # Input has dim*2 channels because we concatenate with initial features
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        
        # 1x1 conv to map to output channels (typically same as input, e.g., 3 for RGB)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time):
        """
        Forward pass through the U-Net.
        
        Args:
            x: Noisy input image (B, channels, H, W)
            time: Timestep (B,) - indicates noise level
        
        Returns:
            Predicted noise (B, channels, H, W)
        
        Data flow visualization:
        
        Input (32x32)
            ↓
        [init conv] ─────────────────────────┐ (save for final skip)
            ↓                                │
        ┌──────────────────┐                 │
        │ Encoder Block 1  │ → skip 1, 2 ────┼─→ Decoder Block 1
        │   (32x32)        │                 │
        └────────┬─────────┘                 │
                 ↓                           │
        ┌──────────────────┐                 │
        │ Encoder Block 2  │ → skip 3, 4 ────┼─→ Decoder Block 2
        │   (16x16)        │                 │
        └────────┬─────────┘                 │
                 ↓                           │
        ┌──────────────────┐                 │
        │ Encoder Block 3  │ → skip 5, 6 ────┼─→ Decoder Block 3
        │   (8x8)          │                 │
        └────────┬─────────┘                 │
                 ↓                           │
        ┌──────────────────┐                 │
        │   Bottleneck     │                 │
        │   (4x4)          │                 │
        └────────┬─────────┘                 │
                 ↓                           │
        [Decoder path with upsampling]       │
                 ↓                           │
        [Final conv] ←───────────────────────┘
            ↓
        Output (32x32) - Predicted Noise
        
        The skip connections preserve information lost during downsampling.
        """
        
        # ============================================================
        # INITIAL PROCESSING
        # ============================================================
        # x shape: (BATCH, CHANNELS, HEIGHT, WIDTH)
        # time shape: (BATCH,)
        
        # Convert input image to feature representation
        x = self.init_conv(x)
        
        # Save initial features for final skip connection
        r = x.clone()
        
        # Process timestep into rich embedding
        # time: (B,) → (B, time_dim)
        t = self.time_mlp(time)

        # ============================================================
        # ENCODER PATH (DOWNSAMPLING)
        # ============================================================
        # List to store features for skip connections
        h = []

        for block1, block2, attn, downsample in self.downs:
            # First ResNet block with time conditioning
            x = block1(x, t)
            h.append(x)  # Save for skip connection
            
            # Second ResNet block with time conditioning
            x = block2(x, t)
            
            # Self-attention with residual connection
            # "+ x" is a residual connection around the attention
            x = attn(x) + x
            h.append(x)  # Save for skip connection
            
            # Reduce spatial dimensions
            x = downsample(x)

        # ============================================================
        # BOTTLENECK (MIDDLE)
        # ============================================================
        # Process at the most compressed spatial resolution
        x = self.mid_block1(x, t)
        
        # Attention with residual
        x = self.mid_attn(x) + x
        
        x = self.mid_block2(x, t)

        # ============================================================
        # DECODER PATH (UPSAMPLING)
        # ============================================================
        # Reverse the downsampling process, using skip connections
        for block1, block2, attn, upsample in self.ups:
            # Concatenate with skip connection from encoder
            # h.pop() retrieves the most recent saved feature
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            
            # Concatenate with another skip connection
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            
            # Attention with residual
            x = attn(x) + x
            
            # Increase spatial dimensions
            x = upsample(x)

        # ============================================================
        # FINAL OUTPUT
        # ============================================================
        # Concatenate with initial features (long skip connection)
        x = torch.cat((x, r), dim=1)
        
        # Final processing
        x = self.final_res_block(x, t)
        
        # Map to output channels (predict the noise)
        return self.final_conv(x)
    
# ============================================================
# USAGE EXAMPLE
# ============================================================
"""
# Create U-Net model
model = Unet(
    dim=64,              # Base channel dimension
    dim_mults=(1, 2, 4, 8),  # Channel multipliers: [64, 128, 256, 512]
    channels=3,          # RGB images
)

# Example inputs
batch_size = 4
noisy_images = torch.randn(batch_size, 3, 32, 32)  # Noisy images
timesteps = torch.randint(0, 1000, (batch_size,))   # Random timesteps

# Forward pass
predicted_noise = model(noisy_images, timesteps)
# Output shape: (4, 3, 32, 32) - same as input

# In diffusion training:
# 1. Take clean image
# 2. Add noise according to timestep t
# 3. Model predicts the noise
# 4. Loss = MSE(predicted_noise, actual_noise)
# 5. Model learns to denoise at all timesteps
"""