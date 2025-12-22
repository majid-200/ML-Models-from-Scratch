"""
LLAMA-STYLE TRANSFORMER IMPLEMENTATION
=======================================
This implementation follows the LLaMA architecture with:
- Rotary Position Embeddings (RoPE)
- RMS Normalization
- SwiGLU activation in feedforward
- Grouped Query Attention (GQA)
- KV Caching for efficient inference
"""

from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    """
    Configuration class for the Transformer model.
    
    Architecture Overview:
    ┌─────────────────────────────────────┐
    │         INPUT TOKENS                │
    │              ↓                      │
    │      Token Embeddings               │
    │              ↓                      │
    │    ┌──────────────────┐             │
    │    │  Encoder Block 1 │             │
    │    │  Encoder Block 2 │             │
    │    │       ...        │  (n_layers) │
    │    │  Encoder Block N │             │
    │    └──────────────────┘             │
    │              ↓                      │
    │       RMS Norm + Output             │
    │              ↓                      │
    │         LOGITS                      │
    └─────────────────────────────────────┘
    """
    
    # Model dimension (embedding size)
    # Each token is represented as a vector of size 'dim'
    dim: int = 4096
    
    # Number of transformer encoder blocks stacked on top of each other
    n_layers: int = 32
    
    # Number of attention heads for Queries
    # More heads = model can attend to different aspects simultaneously
    n_heads: int = 32
    
    # Number of attention heads for Keys and Values (for Grouped Query Attention)
    # If None, defaults to n_heads (standard multi-head attention)
    # If less than n_heads, multiple Q heads share the same K,V heads (more efficient)
    #
    # Example with n_heads=8, n_kv_heads=2:
    # Q heads: [Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8]
    # K,V heads: [K1, V1, K2, V2]
    # Grouping: Q1-Q4 use K1,V1 | Q5-Q8 use K2,V2
    n_kv_heads: Optional[int] = None
    
    # Size of vocabulary (number of unique tokens)
    # Set to -1 initially, will be updated based on tokenizer
    vocab_size: int = -1
    
    # Used to round hidden dimensions to multiples of this number
    # Helps with computational efficiency on GPUs
    multiple_of: int = 256
    
    # Optional multiplier for feedforward network hidden dimension
    ffn_dim_multiplier: Optional[float] = None
    
    # Small constant added for numerical stability in normalization
    norm_eps: float = 1e-5

    # === KV Cache Parameters (for efficient inference) ===
    # Maximum batch size for inference
    max_batch_size: int = 32
    
    # Maximum sequence length the model can handle
    # Longer sequences require more memory for KV cache
    max_seq_len: int = 2048

    # Device to run the model on ('cuda' or 'cpu')
    device: str = None


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    ====================================
    
    Simpler and faster alternative to LayerNorm used in LLaMA models.
    Instead of normalizing by mean and variance, only normalizes by RMS (root mean square).
    
    Mathematical Formula:
    ─────────────────────
    RMS(x) = sqrt(mean(x²) + ε)
    Output = (x / RMS(x)) * weight
    
    Visual Example (for one vector):
    ────────────────────────────────
    Input:  [1.0, 2.0, 3.0, 4.0]
                    ↓
    Square: [1.0, 4.0, 9.0, 16.0]
                    ↓
    Mean:   7.5
                    ↓
    RMS:    sqrt(7.5) = 2.74
                    ↓
    Norm:   [0.36, 0.73, 1.09, 1.46]  (input / RMS)
                    ↓
    Scale:  [0.36*w₁, 0.73*w₂, 1.09*w₃, 1.46*w₄]  (multiply by learned weights)
    
    Why RMS Norm?
    ─────────────
    • Faster: No mean subtraction needed
    • Simpler: Only one normalization statistic (RMS vs mean+variance)
    • Effective: Works as well as LayerNorm for transformers
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # Small epsilon for numerical stability (prevents division by zero)
        self.eps = eps
        
        # Learnable scale parameter (gamma)
        # Shape: (dim,) - one weight per dimension
        # Allows the model to learn the optimal scaling after normalization
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        """
        Normalize input by its Root Mean Square
        
        Input shape:  (Batch, Seq_Len, Dim)
        Output shape: (Batch, Seq_Len, Dim)
        
        Step-by-step:
        ────────────
        x.pow(2)              → Square each element
        .mean(-1, keepdim=True) → Mean across last dim: (B, Seq_Len, Dim) → (B, Seq_Len, 1)
        + self.eps            → Add small epsilon for stability
        torch.rsqrt(...)      → Reciprocal square root: 1/sqrt(x)
        x * ...               → Multiply input by reciprocal RMS
        """
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x) - more numerically stable than 1/sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        """
        Apply RMS normalization with learned scaling
        
        Flow:
        ────
        Input (B, Seq_Len, Dim) 
            → normalize (float precision for stability)
            → convert back to input dtype
            → scale by learned weights
        → Output (B, Seq_Len, Dim)
        """
        # Normalize in float32 for numerical stability, then convert back
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    """
    Precompute Rotary Position Embedding (RoPE) frequencies
    =======================================================
    
    RoPE encodes position information by rotating embeddings in 2D subspaces.
    Instead of adding position embeddings, we rotate the query and key vectors.
    
    Why RoPE?
    ─────────
    • Relative positions: Naturally encodes relative distances between tokens
    • Extrapolation: Can generalize to longer sequences than seen in training
    • No learned params: Purely based on sinusoidal frequencies
    
    The rotation angle increases with position, encoding position information.
    
    Mathematical Foundation:
    ───────────────────────
    For dimension pair (x, y), rotation matrix at position m:
    
    R(m, θ) = [cos(mθ)  -sin(mθ)]
              [sin(mθ)   cos(mθ)]
    
    Using complex numbers (more efficient):
    e^(i*m*θ) = cos(mθ) + i*sin(mθ)
    
    Frequency Formula (from RoPE paper):
    θᵢ = 10000^(-2i/d) where i = 0, 1, 2, ..., d/2-1
    
    Different dimensions rotate at different frequencies:
    • Low dimensions (i=0): Fast rotation, capture local patterns
    • High dimensions (i=d/2): Slow rotation, capture long-range patterns
    """
    
    # As written in the paragraph 3.2.2 of the paper
    # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    
    # Build the theta parameters (one for each dimension pair)
    # According to the formula: θᵢ = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    #
    # Example for head_dim=8:
    # theta_numerator = [0, 2, 4, 6]
    # theta = [10000^0, 10000^(-2/8), 10000^(-4/8), 10000^(-6/8)]
    #       = [1.0, 0.0464..., 0.00215..., 0.0001...]
    # Lower dimensions → higher frequencies → faster rotation
    #
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    
    # Calculate theta values for each dimension pair
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    
    # Construct position indices (m parameter in the rotation formula)
    # m = [0, 1, 2, 3, ..., seq_len-1]
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device=device)
    
    # Compute m * θ for all combinations of positions and frequencies
    # This is the rotation angle for each position and dimension pair
    #
    # Visual representation (seq_len=4, head_dim=4, so head_dim/2=2):
    #
    #           θ₀        θ₁
    #     ┌──────────┬──────────┐
    # m=0 │   0*θ₀   │   0*θ₁   │  Position 0 (no rotation)
    # m=1 │   1*θ₀   │   1*θ₁   │  Position 1
    # m=2 │   2*θ₀   │   2*θ₁   │  Position 2
    # m=3 │   3*θ₀   │   3*θ₁   │  Position 3
    #     └──────────┴──────────┘
    #
    # Multiply each position (m) by each frequency (theta) using outer product
    # Shape: (Seq_Len) outer_product (Head_Dim / 2) → (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    
    # Convert to complex numbers in polar form: R * e^(iθ)
    # For rotation, R=1 (magnitude), and angle=freqs
    #
    # Complex number representation:
    # e^(iθ) = cos(θ) + i*sin(θ)
    #
    # This gives us the rotation in the complex plane:
    # Shape: (Seq_Len, Head_Dim / 2)
    # Each complex number represents a 2D rotation for a dimension pair
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    """
    Apply Rotary Position Embeddings to input tensor
    ================================================
    
    Takes the query or key vectors and rotates them based on their position.
    This rotation encodes position information without adding parameters.
    
    Process Overview:
    ────────────────
    1. Convert real vectors to complex numbers (pair consecutive dimensions)
    2. Multiply by rotation complex numbers (this performs 2D rotations)
    3. Convert back to real numbers
    
    Detailed Example (head_dim=4, so 2 complex numbers per head):
    ─────────────────────────────────────────────────────────────
    
    Input vector x = [x₀, x₁, x₂, x₃]
    
    Step 1: Pair into complex numbers
    ──────────────────────────────────
    [x₀, x₁, x₂, x₃] → [x₀+ix₁,    x₂+ix₃]
                       ︸─────︸  ︸─────︸
                         complex₁  complex₂
    
    Step 2: Multiply by rotation (at position m)
    ─────────────────────────────────────────────
    Rotation factors: [e^(im*θ₀), e^(im*θ₁)]
    
    Result: [(x₀+ix₁)·e^(im*θ₀), (x₂+ix₃)·e^(im*θ₁)]
    
    This is equivalent to rotating (x₀,x₁) by angle m*θ₀
                       and rotating (x₂,x₃) by angle m*θ₁
    
    Visual (2D rotation of first dimension pair):
    ────────────────────────────────────────────
        y (x₁)
         ↑
         |     • (x₀', x₁')  ← After rotation
         |    /
         |   / m*θ₀
         |  /
         | /___________
         |/          • (x₀, x₁)  ← Before rotation
         └──────────────→ x (x₀)
    
    Step 3: Convert back to real
    ────────────────────────────
    [real₁+i·imag₁, real₂+i·imag₂] → [real₁, imag₁, real₂, imag₂]
    
    Shape Transformations:
    ─────────────────────
    (B, Seq_Len, H, Head_Dim) 
        → (B, Seq_Len, H, Head_Dim/2)      [view as complex]
        → (B, Seq_Len, H, Head_Dim/2)      [multiply by rotation]
        → (B, Seq_Len, H, Head_Dim/2, 2)   [view as real]
        → (B, Seq_Len, H, Head_Dim)        [flatten back]
    """
    
    # Convert real numbers to complex numbers by pairing consecutive dimensions
    # Treat each pair of real numbers as (real, imaginary) components
    #
    # Example: [a, b, c, d] → [a+ib, c+id]
    #
    # x.shape[:-1] preserves (B, Seq_Len, H)
    # -1, 2 means: group last dimension into pairs
    # 
    # Shape: (B, Seq_Len, H, Head_Dim) → (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # Reshape freqs_complex to broadcast correctly
    # Need to add batch dimension and head dimension for broadcasting
    #
    # Before: (Seq_Len, Head_Dim/2)
    # After:  (1, Seq_Len, 1, Head_Dim/2)
    #          ↑           ↑
    #      batch dim   head dim
    #
    # This allows broadcasting across all batches and all heads
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    
    # Perform the rotation via complex multiplication
    # 
    # Complex multiplication rotates the vector:
    # (a + ib) × (cos(θ) + i·sin(θ)) = rotation of (a,b) by angle θ
    #
    # Broadcasting:
    # (B, Seq_Len, H, Head_Dim/2) × (1, Seq_Len, 1, Head_Dim/2)
    #   ↓                              ↓
    # broadcasts across B and H    broadcasts across B and H
    #
    # Result: Each position gets rotated by its corresponding angle
    # Shape: (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    
    # Convert complex numbers back to real numbers
    # Separate real and imaginary parts back into consecutive dimensions
    #
    # Example: [a+ib, c+id] → [[a,b], [c,d]]
    #
    # Shape: (B, Seq_Len, H, Head_Dim/2) → (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    
    # Flatten the last two dimensions back to original shape
    # [[a,b], [c,d]] → [a, b, c, d]
    #
    # Shape: (B, Seq_Len, H, Head_Dim/2, 2) → (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    
    # Convert back to original dtype and device
    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat Key and Value tensors to match the number of Query heads
    ================================================================
    
    Used for Grouped Query Attention (GQA), where multiple Q heads share
    the same K,V heads. This function expands K,V to match Q heads.
    
    Why Grouped Query Attention?
    ────────────────────────────
    • Memory efficient: Fewer K,V heads = less memory for KV cache
    • Speed: Faster inference with smaller cache
    • Performance: Minimal accuracy loss compared to full Multi-Head Attention
    
    Example Scenario:
    ────────────────
    n_heads_q = 8 (Query heads)
    n_kv_heads = 2 (Key/Value heads)
    n_rep = 8 / 2 = 4 (repetitions needed)
    
    Visualization of repetition:
    ───────────────────────────
    
    Input K,V heads:     Output K,V heads (repeated):
    ┌────┬────┐          ┌────┬────┬────┬────┬────┬────┬────┬────┐
    │ K₁ │ K₂ │    →     │ K₁ │ K₁ │ K₁ │ K₁ │ K₂ │ K₂ │ K₂ │ K₂ │
    └────┴────┘          └────┴────┴────┴────┴────┴────┴────┴────┘
         2                              8
    
    Query heads:
    ┌────┬────┬────┬────┬────┬────┬────┬────┐
    │ Q₁ │ Q₂ │ Q₃ │ Q₄ │ Q₅ │ Q₆ │ Q₇ │ Q₈ │
    └────┴────┴────┴────┴────┴────┴────┴────┘
      ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓
    Use K₁ for Q₁-Q₄    │  Use K₂ for Q₅-Q₈
    
    Attention grouping:
    • Q₁, Q₂, Q₃, Q₄ all attend using K₁, V₁
    • Q₅, Q₆, Q₇, Q₈ all attend using K₂, V₂
    
    Shape Transformation Example:
    ────────────────────────────
    Input:  (B=2, Seq=10, KV_Heads=2, Dim=64)
    n_rep = 4
    
    Step 1: Add dimension
    (2, 10, 2, 64) → (2, 10, 2, 1, 64)
                              ↑
                         new axis
    
    Step 2: Expand
    (2, 10, 2, 1, 64) → (2, 10, 2, 4, 64)
                                 ↑
                            repeat 4x
    
    Step 3: Reshape
    (2, 10, 2, 4, 64) → (2, 10, 8, 64)
              ↑   ↑           ↑
            2 × 4 = 8 heads
    
    Output: (B=2, Seq=10, Q_Heads=8, Dim=64)
    """
    
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    
    # If no repetition needed (n_kv_heads == n_heads_q), return as-is
    # This happens in standard Multi-Head Attention
    if n_rep == 1:
        return x
    
    return (
        # Step 1: Add a new dimension for repetition
        # (B, Seq_Len, N_KV_Heads, Head_Dim)
        # → (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        #
        # x[:, :, :, None, :] is equivalent to x.unsqueeze(3)
        # The None (or unsqueeze) adds a dimension of size 1
        x[:, :, :, None, :]
        
        # Step 2: Expand along the new dimension
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        # → (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        #
        # expand() repeats the tensor n_rep times along dimension 3
        # This doesn't allocate new memory, just changes the view
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        
        # Step 3: Merge KV heads with repetitions
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        # → (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        #
        # Flattens dimensions 2 and 3 together
        # Result: Each KV head is now repeated n_rep times consecutively
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


class SelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with Grouped Query Attention (GQA) and KV Caching
    ===========================================================================
    
    Attention allows each token to look at all other tokens and decide which
    ones are most relevant. It's the core mechanism that makes transformers powerful.
    
    Attention Intuition:
    ───────────────────
    Imagine reading: "The cat sat on the mat because it was tired"
    When processing "it", attention helps the model look back and realize
    "it" refers to "cat" (high attention weight) not "mat" (low attention weight).
    
    High-Level Flow:
    ───────────────
    Input → [Q, K, V projections] → [Apply RoPE] → [Compute attention scores]
         → [Apply softmax] → [Weighted sum of values] → [Output projection]
    
    Mathematical Formula:
    ────────────────────
    Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V
    
    Where:
    • Q (Query): "What am I looking for?"
    • K (Key): "What do I contain?"
    • V (Value): "What information do I have?"
    • d: Head dimension (for scaling)
    
    Visualization (simplified with 3 tokens):
    ────────────────────────────────────────
    
    Token 1: "The"    Token 2: "cat"    Token 3: "sat"
       Q₁                Q₂                Q₃
       ↓                 ↓                 ↓
    Compare with:     Compare with:     Compare with:
    K₁  K₂  K₃        K₁  K₂  K₃        K₁  K₂  K₃
    ↓   ↓   ↓         ↓   ↓   ↓         ↓   ↓   ↓
    [Attention Scores] [Attention Scores] [Attention Scores]
    0.2 0.3 0.5       0.1 0.6 0.3       0.3 0.4 0.3
    ↓                 ↓                 ↓
    Weighted sum of:  Weighted sum of:  Weighted sum of:
    V₁  V₂  V₃        V₁  V₂  V₃        V₁  V₂  V₃
    
    Grouped Query Attention (GQA):
    ─────────────────────────────
    Instead of having separate K,V for each head, multiple Q heads share K,V heads:
    
    Standard MHA (32 heads):          GQA (32 Q heads, 8 KV heads):
    Q: [Q₁...Q₃₂]                     Q: [Q₁...Q₃₂]
    K: [K₁...K₃₂]  ← 32 KV heads      K: [K₁...K₈]  ← Only 8 KV heads!
    V: [V₁...V₃₂]                     V: [V₁...V₈]
    
    Memory: High                      Memory: 4x less KV cache
    Speed: Slower                     Speed: Faster inference
    
    KV Cache Concept:
    ────────────────
    During generation, we process one token at a time. Instead of recomputing
    K,V for all previous tokens, we cache them.
    
    Step 1: "The"                Step 2: "The cat"           Step 3: "The cat sat"
    Cache: [K₁, V₁]              Cache: [K₁, K₂, V₁, V₂]    Cache: [K₁, K₂, K₃, V₁, V₂, V₃]
           ↑ new                       ↑ new                        ↑ new
    
    Each step only computes new token's K,V, reuses cached ones for attention.
    """
    
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Number of Key/Value heads (for GQA)
        # If n_kv_heads is None, use same as n_heads (standard MHA)
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        
        # Number of Query heads
        self.n_heads_q = args.n_heads
        
        # How many times to repeat each KV head to match Q heads
        # Example: 32 Q heads / 8 KV heads = 4 repetitions
        self.n_rep = self.n_heads_q // self.n_kv_heads
        
        # Dimension of each attention head
        # Total model dim is split across all heads
        # Example: dim=4096, n_heads=32 → head_dim=128
        self.head_dim = args.dim // args.n_heads

        # Linear projections for Queries, Keys, Values, and Output
        # 
        # Weight matrix shapes:
        # wq: (dim, n_heads * head_dim) - Projects input to all Q heads
        # wk: (dim, n_kv_heads * head_dim) - Projects input to KV heads (fewer!)
        # wv: (dim, n_kv_heads * head_dim) - Projects input to KV heads
        # wo: (n_heads * head_dim, dim) - Projects concatenated heads back to dim
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # KV Cache: Store Keys and Values for all previous tokens
        # Preallocate for maximum batch size and sequence length
        # 
        # During inference:
        # • Start with empty cache
        # • Each new token: compute its K,V and store in cache
        # • Attention uses all cached K,V (growing sequence)
        #
        # Shape: (max_batch_size, max_seq_len, n_kv_heads, head_dim)
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_complex: torch.Tensor
    ):
        """
        Forward pass of self-attention with KV caching
        
        Args:
            x: Input tensor (B, Seq_Len, Dim) - usually Seq_Len=1 during inference
            start_pos: Position in sequence where this token starts (for caching)
            freqs_complex: Precomputed RoPE frequencies for position encoding
        
        Process Flow:
        ────────────
        1. Project input to Q, K, V
        2. Reshape for multi-head attention
        3. Apply rotary position embeddings (RoPE)
        4. Update KV cache with new K, V
        5. Retrieve all cached K, V (including new ones)
        6. Repeat KV heads to match Q heads (for GQA)
        7. Compute attention scores (Q·Kᵀ)
        8. Apply softmax to get attention weights
        9. Compute weighted sum of values
        10. Project output back to model dimension
        """
        
        batch_size, seq_len, _ = x.shape  # (B, 1, Dim) - typically seq_len=1 during inference

        # ════════════════════════════════════════════════════════
        # STEP 1-2: Project to Q, K, V and reshape for multi-head
        # ════════════════════════════════════════════════════════
        
        # Project input to Query space
        # (B, 1, Dim) → (B, 1, H_Q * Head_Dim)
        xq = self.wq(x)
        
        # Project input to Key space (fewer heads for GQA!)
        # (B, 1, Dim) → (B, 1, H_KV * Head_Dim)
        xk = self.wk(x)
        
        # Project input to Value space (fewer heads for GQA!)
        # (B, 1, Dim) → (B, 1, H_KV * Head_Dim)
        xv = self.wv(x)

        # Reshape to separate heads: (B, 1, H * Head_Dim) → (B, 1, H, Head_Dim)
        # This separates the projection into individual attention heads
        #
        # (B, 1, H_Q * Head_Dim) → (B, 1, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_Dim) → (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) → (B, 1, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # ════════════════════════════════════════════════════════
        # STEP 3: Apply Rotary Position Embeddings (RoPE)
        # ════════════════════════════════════════════════════════
        
        # Rotate queries by their position angle
        # This encodes "where" the token is in the sequence
        # (B, 1, H_Q, Head_Dim) → (B, 1, H_Q, Head_Dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        
        # Rotate keys by their position angle
        # This allows relative position computation via dot product
        # (B, 1, H_KV, Head_Dim) → (B, 1, H_KV, Head_Dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # ════════════════════════════════════════════════════════
        # STEP 4-5: Update and retrieve KV cache
        # ════════════════════════════════════════════════════════
        
        # Store the current token's K,V in the cache at position [start_pos]
        # 
        # Example: Generating "The cat sat"
        # Token 1 "The": start_pos=0, stores K,V at cache[0]
        # Token 2 "cat": start_pos=1, stores K,V at cache[1]
        # Token 3 "sat": start_pos=2, stores K,V at cache[2]
        #
        # Cache after 3 tokens: [K₁, K₂, K₃, ...]
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # Retrieve all K,V from start to current position (inclusive)
        # This includes all previous tokens plus the current one
        #
        # Example at token 3: retrieves cache[0:3] = [K₁, K₂, K₃]
        # 
        # (B, Seq_Len_KV, H_KV, Head_Dim) where Seq_Len_KV grows each step
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # ════════════════════════════════════════════════════════
        # STEP 6: Repeat KV heads for Grouped Query Attention
        # ════════════════════════════════════════════════════════
        
        # Expand KV heads to match the number of Q heads
        # Each KV head is shared by multiple Q heads
        #
        # (B, Seq_Len_KV, H_KV, Head_Dim) → (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # ════════════════════════════════════════════════════════
        # STEP 7-8: Compute attention scores and weights
        # ════════════════════════════════════════════════════════
        
        # Transpose to put heads first: (B, Seq, H, D) → (B, H, Seq, D)
        # This format is better for batch matrix multiplication
        #
        # (B, 1, H_Q, Head_Dim) → (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) → (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) → (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)

        # Compute attention scores: Q·Kᵀ / √d
        #
        # Intuition: How much does each query "match" each key?
        # Higher score = more relevant = pay more attention
        #
        # Matrix multiplication:
        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV)
        # → (B, H_Q, 1, Seq_Len_KV)
        #
        # Divide by √head_dim for scaled dot-product attention
        # (prevents extremely large values that would saturate softmax)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply softmax to convert scores to probabilities
        # Each row sums to 1.0 (distribution over all tokens)
        #
        # Example scores: [2.1, 0.5, -1.0]
        # After softmax: [0.7, 0.2, 0.1]  ← Attention weights
        #
        # (B, H_Q, 1, Seq_Len_KV) → (B, H_Q, 1, Seq_Len_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # ════════════════════════════════════════════════════════
        # STEP 9-10: Weighted sum and output projection
        # ════════════════════════════════════════════════════════
        
        # Compute weighted sum of values using attention weights
        #
        # Intuition: Mix together value vectors, weighted by attention
        # High attention weight → more of that token's value
        #
        # (B, H_Q, 1, Seq_Len_KV) @ (B, H_Q, Seq_Len_KV, Head_Dim)
        # → (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)
        
        # Reshape: Transpose and concatenate all heads
        # (B, H_Q, 1, Head_Dim) → (B, 1, H_Q, Head_Dim) → (B, 1, H_Q * Head_Dim)
        #
        # This merges all attention heads back into a single vector
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        
        # Final linear projection back to model dimension
        # (B, 1, H_Q * Head_Dim) → (B, 1, Dim)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with SwiGLU Activation
    =========================================================
    
    After attention mixes information between tokens, the feedforward network
    processes each token independently to transform the representations.
    
    Architecture: Two-layer MLP with SwiGLU activation
    
    Standard FFN:                    SwiGLU FFN (used here):
    ┌─────────┐                      ┌────────┐
    │ Input   │                      │ Input  │
    └────┬────┘                      └───┬────┘
         │                               │
         ↓                          ┌────┴────┐
    ┌─────────┐                  ┌──▼──┐   ┌──▼──┐
    │ Linear  │                  │ w1  │   │ w3  │
    │  + ReLU │                  └──┬──┘   └──┬──┘
    └────┬────┘                     │         │
         │                          ↓         ↓
         ↓                        [SiLU]   [Identity]
    ┌─────────┐                     │         │
    │ Linear  │                     └────┬────┘
    └────┬────┘                          ↓
         │                           [Multiply]  ← Gating!
         ↓                               │
    ┌─────────┐                          ↓
    │ Output  │                      ┌────────┐
    └─────────┘                      │   w2   │
                                     └────┬───┘
                                          ↓
                                     ┌────────┐
                                     │ Output │
                                     └────────┘
    
    SwiGLU (Swish-Gated Linear Unit):
    ─────────────────────────────────
    output = SiLU(W₁·x) ⊗ (W₃·x)
    
    Where:
    • SiLU(x) = x · sigmoid(x) [also called Swish]
    • ⊗ = element-wise multiplication (gating)
    • W₁, W₃ = different learned transformations
    
    Why SwiGLU?
    ───────────
    • Better performance: Outperforms ReLU and GELU in LLMs
    • Gating mechanism: W₃ acts as a learned gate for W₁
    • Smooth gradients: SiLU is smooth everywhere (helps training)
    
    Hidden Dimension Calculation:
    ────────────────────────────
    Standard transformer: hidden_dim = 4 * model_dim
    LLaMA adjustment: hidden_dim = (2/3) * 4 * model_dim
    
    Example: model_dim = 4096
    • Start: 4 * 4096 = 16384
    • Adjust: (2/3) * 16384 ≈ 10922
    • Round to multiple of 256: 10752
    
    This reduces parameters while maintaining performance.
    
    Example Forward Pass (simplified):
    ──────────────────────────────────
    Input: [1.0, 2.0, 3.0, 4.0]  (dim=4)
    
    After w1: [2.5, -1.0, 3.2, 0.5]  (hidden_dim)
    After SiLU: [2.3, -0.3, 3.1, 0.3]  (smooth activation)
    
    After w3: [1.0, 2.0, 0.5, 1.5]  (hidden_dim)
    
    Element-wise multiply (gating):
    [2.3, -0.3, 3.1, 0.3] ⊗ [1.0, 2.0, 0.5, 1.5]
    = [2.3, -0.6, 1.55, 0.45]
    
    After w2: [1.8, 2.1, 2.9, 3.5]  (back to dim=4)
    """
    
    def __init__(
        self,
        args: ModelArgs
    ):
        super().__init__()

        # Calculate hidden dimension (expansion factor)
        # Start with 4x the model dimension
        hidden_dim = 4 * args.dim
        
        # LLaMA uses 2/3 factor to reduce parameters
        # 4 * dim * 2/3 ≈ 2.67 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        
        # Apply optional multiplier (for experimentation)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        
        # Round to nearest multiple of multiple_of (typically 256)
        # This ensures efficient computation on modern hardware
        #
        # Formula: ceil(hidden_dim / multiple_of) * multiple_of
        # Example: hidden_dim=10922, multiple_of=256
        #   → (10922 + 255) // 256 = 43
        #   → 43 * 256 = 10752
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        # Three linear transformations (no bias for efficiency)
        #
        # w1: First transformation (input → hidden, then SiLU activation)
        # Shape: (dim → hidden_dim)
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        
        # w2: Output transformation (hidden → dim)
        # Shape: (hidden_dim → dim)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        
        # w3: Gate transformation (input → hidden, used for gating)
        # Shape: (dim → hidden_dim)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Forward pass with SwiGLU activation
        
        Process:
        ───────
        1. Apply w1 and SiLU activation
        2. Apply w3 (gate pathway)
        3. Element-wise multiply (gating)
        4. Apply w2 to return to model dimension
        
        Shape flow:
        ──────────
        (B, Seq_Len, Dim) 
            → [w1] → (B, Seq_Len, Hidden_Dim)
            → [SiLU] → (B, Seq_Len, Hidden_Dim)
            
        (B, Seq_Len, Dim)
            → [w3] → (B, Seq_Len, Hidden_Dim)
            
        [Multiply] → (B, Seq_Len, Hidden_Dim)
            → [w2] → (B, Seq_Len, Dim)
        """
        
        # First pathway: Linear transformation + SiLU activation
        # SiLU(x) = x * sigmoid(x) - smooth, non-monotonic activation
        #
        # (B, Seq_Len, Dim) → (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        
        # Second pathway: Linear transformation (acts as gate)
        # This determines how much of the swish activation to let through
        #
        # (B, Seq_Len, Dim) → (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        
        # Gating: Element-wise multiplication
        # The gate (x_V) modulates the activated values (swish)
        # This is the key innovation of GLU-style activations
        #
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) 
        # → (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        
        # Project back to model dimension
        # (B, Seq_Len, Hidden_Dim) → (B, Seq_Len, Dim)
        x = self.w2(x)
        
        return x


class EncoderBlock(nn.Module):
    """
    Single Transformer Encoder Block (Decoder-only architecture)
    ============================================================
    
    Despite being called "EncoderBlock", this is actually a decoder block
    used in decoder-only models like LLaMA and GPT. The naming is historical.
    
    Block Architecture (Pre-Norm variant):
    ─────────────────────────────────────
    
         Input (from previous block or embeddings)
           │
           │────────────────┐
           │                │  Residual Connection 1
           ↓                │
       RMSNorm              │
           │                │
           ↓                │
    Self-Attention          │
     (with RoPE)            │
           │                │
           └───────(+)──────┘
                    │
                    │────────────────┐
                    │                │  Residual Connection 2
                    ↓                │
                RMSNorm              │
                    │                │
                    ↓                │
              FeedForward            │
               (SwiGLU)              │
                    │                │
                    └────────(+)─────┘
                             │
                             ↓
                          Output (to next block)
    
    Key Components:
    ──────────────
    1. Pre-Layer Normalization (RMSNorm before each sub-layer)
    2. Self-Attention with RoPE (processes relationships between tokens)
    3. Feed-Forward Network (processes each token independently)
    4. Residual Connections (help gradients flow during training)
    
    Pre-Norm vs Post-Norm:
    ──────────────────────
    
    Post-Norm (original Transformer):    Pre-Norm (LLaMA, modern):
    ┌──────────┐                         ┌─────────┐
    │  Input   │                         │  Input  │
    └────┬─────┘                         └───┬─────┘
         │                                   │
         ↓                               ┌───┴───┐
    ┌─────────┐                          │       │
    │ SubLayer│                          ↓       │
    └────┬────┘                      ┌────────┐  │
         │                           │  Norm  │  │
         ↓                           └───┬────┘  │
    ┌────────┐                           ↓       │
    │  Norm  │                      ┌─────────┐  │
    └────┬───┘                      │SubLayer │  │
         │                          └────┬────┘  │
         ↓                               │       │
    ┌────────┐                           └──(+)──┘
    │ Output │                               │
    └────────┘                               ↓
                                         ┌────────┐
                                         │ Output │
                                         └────────┘
    
    Pre-Norm advantages:
    • More stable training (easier gradient flow)
    • Can train deeper models without special init
    • Better for very large models
    
    Residual Connections:
    ────────────────────
    output = input + transformation(input)
    
    Why residual connections?
    • Gradient flow: Gradients can flow directly back through addition
    • Identity mapping: Network can learn to skip layers if needed
    • Enables deep networks: Without them, deep networks are hard to train
    
    Example with numbers (simplified, dim=4):
    ────────────────────────────────────────
    
    Input: [1.0, 2.0, 3.0, 4.0]
    
    Step 1: Attention block
    ─────────────────────────
    Normalize: [0.5, 1.0, 1.5, 2.0]
    Attention: [0.2, 0.8, 1.2, 1.8]
    Add residual: [0.2+1.0, 0.8+2.0, 1.2+3.0, 1.8+4.0]
                = [1.2, 2.8, 4.2, 5.8]
    
    Step 2: Feedforward block
    ─────────────────────────
    Normalize: [0.6, 1.4, 2.1, 2.9]
    FFN: [0.3, 1.1, 1.9, 2.7]
    Add residual: [0.3+1.2, 1.1+2.8, 1.9+4.2, 2.7+5.8]
                = [1.5, 3.9, 6.1, 8.5]
    
    Output: [1.5, 3.9, 6.1, 8.5]
    
    Notice how the values grow through the network, while the residual
    connections preserve the original information flow.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()

        # Store configuration
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        # Initialize sub-layers
        # Self-attention mechanism (processes token relationships)
        self.attention = SelfAttention(args)
        
        # Feed-forward network (processes each token independently)
        self.feed_forward = FeedForward(args)

        # Normalization layers (RMSNorm)
        # Pre-normalization: normalize BEFORE the attention block
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # Pre-normalization: normalize BEFORE the feed-forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        """
        Forward pass through one transformer encoder block
        
        Args:
            x: Input tensor (B, Seq_Len, Dim)
            start_pos: Current position in sequence (for KV cache)
            freqs_complex: RoPE frequencies for position encoding
        
        Returns:
            Output tensor (B, Seq_Len, Dim)
        
        Flow Diagram:
        ────────────
        
        x (input)
        │
        ├──────────────────┐
        │                  │
        ↓                  │
        RMSNorm            │ (residual path)
        │                  │
        ↓                  │
        Self-Attention     │
        │                  │
        └───────(+)────────┘
                 │
        h (intermediate)
        │
        ├──────────────────┐
        │                  │
        ↓                  │
        RMSNorm            │ (residual path)
        │                  │
        ↓                  │
        FeedForward        │
        │                  │
        └────────(+)───────┘
                 │
                 ↓
        out (output to next block)
        """
        
        # ════════════════════════════════════════════════════════
        # BLOCK 1: Self-Attention with Residual Connection
        # ════════════════════════════════════════════════════════
        
        # Step 1a: Normalize input (pre-normalization)
        # (B, Seq_Len, Dim) → (B, Seq_Len, Dim)
        normalized = self.attention_norm(x)
        
        # Step 1b: Apply self-attention
        # Tokens look at each other and exchange information
        # (B, Seq_Len, Dim) → (B, Seq_Len, Dim)
        attn_output = self.attention.forward(normalized, start_pos, freqs_complex)
        
        # Step 1c: Add residual connection (x + attention(norm(x)))
        # This preserves the original input information
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) → (B, Seq_Len, Dim)
        h = x + attn_output
        
        # ════════════════════════════════════════════════════════
        # BLOCK 2: Feed-Forward with Residual Connection
        # ════════════════════════════════════════════════════════
        
        # Step 2a: Normalize intermediate output (pre-normalization)
        # (B, Seq_Len, Dim) → (B, Seq_Len, Dim)
        normalized = self.ffn_norm(h)
        
        # Step 2b: Apply feed-forward network
        # Process each token independently with non-linear transformation
        # (B, Seq_Len, Dim) → (B, Seq_Len, Dim)
        ffn_output = self.feed_forward.forward(normalized)
        
        # Step 2c: Add residual connection (h + ffn(norm(h)))
        # Again preserving information from previous steps
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) → (B, Seq_Len, Dim)
        out = h + ffn_output
        
        return out
    

class Transformer(nn.Module):
    """
    Complete LLaMA-style Transformer Model
    ======================================
    
    This is the full model that ties everything together. It's a decoder-only
    transformer designed for autoregressive language modeling (predicting next tokens).
    
    Complete Architecture Stack:
    ───────────────────────────
    
    Input: Token IDs [15, 432, 89, ...]
              │
              ↓
    ┌─────────────────────────┐
    │   Token Embeddings      │  Convert IDs to vectors
    │   (vocab_size → dim)    │
    └──────────┬──────────────┘
               │
               ↓
    ┌─────────────────────────┐
    │   Encoder Block 1       │
    │   • RMSNorm             │
    │   • Self-Attention      │
    │   • RMSNorm             │
    │   • FeedForward         │
    └──────────┬──────────────┘
               │
               ↓
    ┌─────────────────────────┐
    │   Encoder Block 2       │
    │   (same structure)      │
    └──────────┬──────────────┘
               │
               ⋮  (n_layers total)
               │
               ↓
    ┌─────────────────────────┐
    │   Encoder Block N       │
    └──────────┬──────────────┘
               │
               ↓
    ┌─────────────────────────┐
    │   Final RMSNorm         │  Stabilize outputs
    └──────────┬──────────────┘
               │
               ↓
    ┌─────────────────────────┐
    │   Output Projection     │  Project to vocabulary
    │   (dim → vocab_size)    │
    └──────────┬──────────────┘
               │
               ↓
    Output: Logits for each token in vocabulary
            [0.2, -1.5, 3.4, ..., 0.8]  (vocab_size floats)
               ↓
        softmax/sampling
               ↓
          Next token ID
    
    Autoregressive Generation:
    ─────────────────────────
    The model generates text one token at a time, using previous tokens
    as context (stored in KV cache for efficiency).
    
    Example generation of "The cat sat":
    
    Step 1: Input: [BOS]          → Output: "The"    (start_pos=0)
    Step 2: Input: [BOS, The]     → Output: "cat"    (start_pos=1)  
    Step 3: Input: [BOS, The, cat] → Output: "sat"   (start_pos=2)
    
    Note: During inference, we only process ONE new token at a time,
          but attention can see ALL previous tokens via KV cache.
    
    Key Design Choices:
    ──────────────────
    • Decoder-only: No encoder-decoder attention (unlike original Transformer)
    • Causal masking: Implicit through KV cache (can't see future tokens)
    • RoPE: Relative position encoding via rotation
    • GQA: Grouped query attention for efficiency
    • Pre-norm: RMSNorm before each sub-layer
    • SwiGLU: Gated activation in feedforward
    • No bias: All linear layers have bias=False (reduces parameters)
    
    Model Sizes (LLaMA examples):
    ─────────────────────────────
    LLaMA-7B:   dim=4096, n_layers=32, n_heads=32
    LLaMA-13B:  dim=5120, n_layers=40, n_heads=40
    LLaMA-65B:  dim=8192, n_layers=80, n_heads=64
    
    Parameters ≈ 12 × n_layers × dim² (rough estimate)
    """

    def __init__(self, args: ModelArgs):
        super().__init__()

        # Verify vocabulary size is set
        assert args.vocab_size != -1, "Vocab size must be set"

        # Store configuration
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        
        # ════════════════════════════════════════════════════════
        # Token Embeddings
        # ════════════════════════════════════════════════════════
        
        # Convert token IDs to dense vectors
        # Each of the vocab_size tokens gets a learnable dim-dimensional vector
        #
        # Example: vocab_size=32000, dim=4096
        # Token ID 15 → embedding vector of size 4096
        #
        # This is a lookup table: embeddings[token_id] = vector
        # Shape: (vocab_size, dim)
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        # ════════════════════════════════════════════════════════
        # Transformer Layers (Stack of Encoder Blocks)
        # ════════════════════════════════════════════════════════
        
        # Create a stack of n_layers encoder blocks
        # Each block has the same architecture but different learned parameters
        #
        # Information flows through blocks sequentially:
        # embeddings → block_1 → block_2 → ... → block_n → output
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        # ════════════════════════════════════════════════════════
        # Output Head
        # ════════════════════════════════════════════════════════
        
        # Final normalization before output projection
        # Stabilizes the representations before converting to logits
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # Project from model dimension to vocabulary size
        # Produces a score (logit) for each possible next token
        #
        # Shape: (dim, vocab_size)
        # Output: For each position, we get vocab_size scores
        # Higher score = model thinks that token is more likely
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        # ════════════════════════════════════════════════════════
        # Precompute RoPE Frequencies
        # ════════════════════════════════════════════════════════
        
        # Precompute all rotation frequencies for positions 0 to max_seq_len*2
        # *2 to allow for some extrapolation beyond training length
        #
        # These are used in every attention layer to encode position information
        # Precomputing saves time since they're the same for every forward pass
        #
        # Shape: (max_seq_len * 2, head_dim / 2)
        self.freqs_complex = precompute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads,  # head_dim
            self.args.max_seq_len * 2,           # sequence length
            device=self.args.device
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Forward pass through the entire transformer
        
        Args:
            tokens: Input token IDs (B, Seq_Len)
                   During training: Seq_Len can be long (e.g., 2048)
                   During inference: Seq_Len = 1 (process one token at a time)
            
            start_pos: Position in the sequence where these tokens start
                      Used for KV cache indexing
                      Example: When generating token 5, start_pos=4
        
        Returns:
            Logits: (B, Seq_Len, Vocab_Size)
                   Scores for each token in vocabulary at each position
                   Higher score = model predicts that token is more likely
        
        Complete Flow:
        ─────────────
        tokens (B, Seq_Len)
            ↓ [embedding lookup]
        embeddings (B, Seq_Len, Dim)
            ↓ [encoder block 1]
        hidden (B, Seq_Len, Dim)
            ↓ [encoder block 2]
        hidden (B, Seq_Len, Dim)
            ⋮
            ↓ [encoder block N]
        hidden (B, Seq_Len, Dim)
            ↓ [RMSNorm]
        normalized (B, Seq_Len, Dim)
            ↓ [output projection]
        logits (B, Seq_Len, Vocab_Size)
        
        Example Values:
        ──────────────
        Input: tokens = [15, 432]  (batch_size=1, seq_len=2)
        
        After embedding: [[vec_15], [vec_432]]  each vec is 4096-dim
        
        After all layers: [[hidden_1], [hidden_2]]  transformed representations
        
        After output: [[logits_1], [logits_2]]  each is 32000-dim (vocab_size)
        
        logits_1 = [0.2, -1.5, 3.4, ..., 0.8]  (scores for position 1)
        logits_2 = [1.1, 0.3, -0.5, ..., 2.1]  (scores for position 2)
        
        To get next token: apply softmax and sample from logits_2
        """
        
        # ════════════════════════════════════════════════════════
        # Input Validation and Embedding
        # ════════════════════════════════════════════════════════
        
        # Get batch size and sequence length
        # (B, Seq_Len)
        batch_size, seq_len = tokens.shape
        
        # During inference, we process one token at a time for efficiency
        # During training, seq_len can be longer
        assert seq_len == 1, "Only one token at a time can be processed"

        # Convert token IDs to embedding vectors
        # Lookup: For each token ID, get its corresponding embedding vector
        #
        # (B, Seq_Len) → (B, Seq_Len, Dim)
        # Example: token ID 15 → embedding_matrix[15] (4096-dim vector)
        h = self.tok_embeddings(tokens)

        # ════════════════════════════════════════════════════════
        # Get RoPE Frequencies for Current Positions
        # ════════════════════════════════════════════════════════
        
        # Extract the rotation frequencies for current positions
        # If start_pos=5 and seq_len=1, we get frequencies for position 5
        #
        # These frequencies encode position information that will be
        # applied in the attention mechanism of each layer
        #
        # Shape: (Seq_Len, Head_Dim/2)
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        
        # ════════════════════════════════════════════════════════
        # Process Through All Transformer Layers
        # ════════════════════════════════════════════════════════
        
        # Pass through each encoder block sequentially
        # Each layer:
        #   1. Applies self-attention (tokens interact)
        #   2. Applies feedforward (tokens processed independently)
        #   3. Uses residual connections and normalization
        #
        # The representation 'h' gets progressively refined through the stack
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        
        # ════════════════════════════════════════════════════════
        # Final Normalization and Output Projection
        # ════════════════════════════════════════════════════════
        
        # Apply final RMSNorm to stabilize the output
        # (B, Seq_Len, Dim) → (B, Seq_Len, Dim)
        h = self.norm(h)
        
        # Project to vocabulary size to get logits
        # Each position gets a score for every possible token
        #
        # (B, Seq_Len, Dim) @ (Dim, Vocab_Size) → (B, Seq_Len, Vocab_Size)
        #
        # Convert to float for numerical stability
        # (Mixed precision training might use float16 internally)
        output = self.output(h).float()
        
        return output