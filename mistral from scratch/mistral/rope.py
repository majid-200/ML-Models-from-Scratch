import torch
from typing import Tuple


"""
═══════════════════════════════════════════════════════════════════════════
                    ROTARY POSITION EMBEDDINGS (RoPE)
═══════════════════════════════════════════════════════════════════════════

RoPE is a way to encode position information into token embeddings using ROTATION.

WHY RoPE?
---------
Traditional positional embeddings:
- Add position vectors to token embeddings
- Position info can "fade" during attention
- Hard to extrapolate to longer sequences than seen in training

RoPE advantages:
- Encodes RELATIVE positions naturally
- Position info preserved through attention mechanism
- Better extrapolation to longer sequences
- Used in: LLaMA, Mistral, GPT-NeoX, PaLM, and many modern LLMs


THE CORE IDEA: ROTATION IN COMPLEX SPACE
-----------------------------------------

Think of each pair of dimensions as a 2D plane:

    Dimension space:     [d₀, d₁, d₂, d₃, d₄, d₅, ...]
                          └──┬──┘ └──┬──┘ └──┬──┘
    Group into pairs:     Pair 0   Pair 1   Pair 2
    
Each pair forms a 2D vector that we ROTATE by an angle θ.
The angle depends on the TOKEN'S POSITION in the sequence.

Visual of rotation in 2D:
    
    y │     • (x', y')  ← After rotation by angle θ
      │    ╱│
      │   ╱ │
      │  ╱  │
      │ ╱ θ │
      │╱────┼─── x
           (x, y) ← Original point
    
    Rotation formula:
    x' = x·cos(θ) - y·sin(θ)
    y' = x·sin(θ) + y·cos(θ)
    
    In complex form: (x + iy) · e^(iθ) = (x + iy) · (cos(θ) + i·sin(θ))


POSITION ENCODING:
------------------
Each position gets a DIFFERENT rotation angle for each dimension pair.

Position 0: [θ₀⁰, θ₁⁰, θ₂⁰, ...]  ← Small rotations
Position 1: [θ₀¹, θ₁¹, θ₂¹, ...]
Position 2: [θ₀², θ₁², θ₂², ...]  ← Larger rotations
Position t: [θ₀ᵗ, θ₁ᵗ, θ₂ᵗ, ...]

The frequencies decrease for higher dimensions, so:
- Low dimensions rotate fast (capture fine-grained position)
- High dimensions rotate slow (capture coarse-grained position)

═══════════════════════════════════════════════════════════════════════════
"""


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the rotation frequencies for RoPE.
    
    This creates complex numbers e^(i·θ) that represent rotations.
    Each position and each dimension pair gets its own rotation angle.
    
    Args:
        dim: Model dimension (e.g., 128 for head_dim)
        end: Maximum sequence length to precompute (e.g., 2048)
        theta: Base for frequency calculation (default 10000, from original paper)
    
    Returns:
        Tensor of shape [end, dim//2] containing complex numbers (complex64)
        Each complex number represents a rotation: e^(iθ) = cos(θ) + i·sin(θ)
    
    
    ═══════════════════════════════════════════════════════════════════
                        FREQUENCY COMPUTATION VISUAL
    ═══════════════════════════════════════════════════════════════════
    
    Step 1: Compute base frequencies for each dimension pair
    ────────────────────────────────────────────────────────────────
    
    For dimension pairs 0, 1, 2, ..., dim//2 - 1:
    
    freq_i = 1 / (theta^(2i/dim))
    
    Example with dim=8, theta=10000:
        Pair 0 (dims 0,1): freq = 1 / 10000^(0/8) = 1.0000
        Pair 1 (dims 2,3): freq = 1 / 10000^(2/8) = 0.1000  ← 10x slower
        Pair 2 (dims 4,5): freq = 1 / 10000^(4/8) = 0.0100  ← 100x slower
        Pair 3 (dims 6,7): freq = 1 / 10000^(6/8) = 0.0010  ← 1000x slower
    
    This creates a "spectrum" of frequencies:
    
        Fast ████████████░░░░░░░░░░░░░░░░░░░░ Slow
             Pair 0      Pair 1    Pair 2    Pair 3
             
    Fast frequencies (low dims) → encode fine position details
    Slow frequencies (high dims) → encode coarse position details
    
    
    Step 2: Multiply by position to get rotation angles
    ────────────────────────────────────────────────────────────────
    
    For each position t (0, 1, 2, ..., end-1):
        angle_t,i = t · freq_i
    
    This creates a matrix:
    
                     Dimension Pairs
                  0      1      2      3
              ┌─────────────────────────────
        Pos 0 │  0.0   0.0    0.0    0.0     ← No rotation at start
        Pos 1 │  1.0   0.1    0.01   0.001   ← Increasing angles
        Pos 2 │  2.0   0.2    0.02   0.002
        Pos 3 │  3.0   0.3    0.03   0.003
        Pos 4 │  4.0   0.4    0.04   0.004
         ...
    
    Notice: Later positions have larger angles (more rotation)
    
    
    Step 3: Convert to complex numbers (rotation in complex plane)
    ────────────────────────────────────────────────────────────────
    
    Each angle θ becomes: e^(iθ) = cos(θ) + i·sin(θ)
    
    This represents a rotation by angle θ in the complex plane.
    
    ═══════════════════════════════════════════════════════════════════
    """
    
    
    # STEP 1: Compute base frequencies for dimension pairs   
    # 
    # Create frequencies that decrease exponentially:
    # freq_i = 1.0 / (theta^(2i/dim))
    # 
    # torch.arange(0, dim, 2) gives [0, 2, 4, 6, ...] for dimension pairs
    # [: (dim // 2)] ensures we only take dim//2 elements
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # Shape: [dim // 2]
    # Example for dim=128: [1.0, 0.78, 0.61, 0.48, ..., 0.0001]
    #                       ↑                              ↑
    #                     fast                           slow
    
    
    # STEP 2: Create position indices (time steps)
    # 
    # t = [0, 1, 2, 3, ..., end-1]
    t = torch.arange(end, device=freqs.device)  # Shape: [end]
    
    
    # STEP 3: Compute all rotation angles
    # 
    # torch.outer(t, freqs) creates a matrix where:
    # result[i, j] = t[i] * freqs[j]
    # 
    # This gives us the rotation angle for position i, dimension pair j
    freqs = torch.outer(t, freqs).float()
    # Shape: [end, dim // 2]
    # 
    # Example visualization (simplified):
    #           dim_pair_0  dim_pair_1  dim_pair_2
    #   pos_0  [    0.0        0.0         0.0    ]
    #   pos_1  [    1.0        0.78        0.61   ]
    #   pos_2  [    2.0        1.56        1.22   ]
    #   pos_3  [    3.0        2.34        1.83   ]
    #    ...
    
    
    # STEP 4: Convert angles to complex rotation numbers     
    # 
    # torch.polar(magnitude, angle) creates complex numbers:
    # magnitude * e^(i * angle) = magnitude * (cos(angle) + i*sin(angle))
    # 
    # We use magnitude=1, so we get pure rotations: e^(i*angle)
    return torch.polar(torch.ones_like(freqs), freqs)
    # Shape: [end, dim // 2]
    # dtype: complex64
    # 
    # Each element is a complex number representing a rotation


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.
    
    This is where the actual position encoding happens
    We rotate each dimension pair by the precomputed angles.
    
    Args:
        xq: Query tensor [batch, seq_len, n_heads, head_dim]
        xk: Key tensor [batch, seq_len, n_heads, head_dim]
        freqs_cis: Precomputed rotation frequencies [seq_len, head_dim//2]
    
    Returns:
        Rotated query and key tensors (same shape as input)
    
    
    ═══════════════════════════════════════════════════════════════════
                        ROTATION APPLICATION VISUAL
    ═══════════════════════════════════════════════════════════════════
    
    Input tensor structure:
    ────────────────────────────────────────────────────────────────
    
    xq shape: [Seq, N_Heads, Head_Dim]
    
    Example: Seq=4, N_Heads=2, Head_Dim=8
    
        Token 0, Head 0: [x₀, x₁, x₂, x₃, x₄, x₅, x₆, x₇]
        Token 0, Head 1: [x₀, x₁, x₂, x₃, x₄, x₅, x₆, x₇]
        Token 1, Head 0: [x₀, x₁, x₂, x₃, x₄, x₅, x₆, x₇]
        Token 1, Head 1: [x₀, x₁, x₂, x₃, x₄, x₅, x₆, x₇]
        ...
    
    
    Reshape to pairs (view as complex):
    ────────────────────────────────────────────────────────────────
    
    Group consecutive dimensions into complex numbers (pairs):
    
        [x₀, x₁, x₂, x₃, x₄, x₅, x₆, x₇]
         └─┬─┘    └─┬─┘   └─┬─┘   └─┬─┘
          c₀       c₁      c₂      c₃      where c₀ = x₀ + i·x₁
    
    After reshape: [Seq, N_Heads, Head_Dim//2] of complex numbers
    
    
    Rotation in complex space:
    ────────────────────────────────────────────────────────────────
    
    For token at position t:
        c'ᵢ = cᵢ · e^(i·θₜ,ᵢ)
    
    Visual (2D projection of one dimension pair):
    
        Before rotation:          After rotation:
        
        y │                       y │     
          │   • c₀                  │       • c'₀
          │                         │      ╱
          │                         │     ╱ 
          │                         │    ╱θ
          └─────── x                └───────── x
          
    Each dimension pair rotates by a different angle!
    Different positions rotate by different amounts!
    
    
    Broadcasting for all heads:
    ────────────────────────────────────────────────────────────────
    
    xq_:      [Seq, N_Heads, Head_Dim//2]
    freqs_cis:[Seq,    1,    Head_Dim//2]  ← Added dimension
              └────────┬────────────────┘
                   broadcasts across heads
    
    Result:   [Seq, N_Heads, Head_Dim//2]
    
    Same rotation applied to all heads at each position!
    
    ═══════════════════════════════════════════════════════════════════
    """
    
    
    # STEP 1: Reshape to complex numbers (pair dimensions)   
    # 
    # Original: [Seq, N_Heads, Head_Dim] where Head_Dim is even
    # Goal: Group consecutive dims into complex numbers
    # 
    # Example: [a, b, c, d, e, f, g, h] → [(a,b), (c,d), (e,f), (g,h)]
    #          where (a,b) becomes complex number a + i·b
    
    # Reshape: [..., Head_Dim] → [..., Head_Dim//2, 2]
    # This groups dimensions into pairs: [[d0,d1], [d2,d3], [d4,d5], ...]
    # Then view_as_complex converts each pair [x, y] → x + i·y
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    # Shape: [Seq, N_Heads, Head_Dim//2] (complex64)
    
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # Shape: [Seq, N_Heads, Head_Dim//2] (complex64)
    
    
    # STEP 2: Prepare rotation frequencies for broadcasting  
    # 
    # freqs_cis original shape: [Seq, Head_Dim//2]
    # Add dimension for heads: [Seq, 1, Head_Dim//2]
    # This allows broadcasting across the N_Heads dimension
    freqs_cis = freqs_cis[:, None, :]
    # Shape: [Seq, 1, Head_Dim//2]
    # 
    # The '1' dimension will broadcast to match N_Heads
    

    # STEP 3: Apply rotation (complex multiplication)
    # 
    # Complex multiplication performs rotation:
    # (a + bi) · (cos(θ) + i·sin(θ)) = rotated vector
    # 
    # Broadcasting happens:
    # [Seq, N_Heads, Head_Dim//2] * [Seq, 1, Head_Dim//2]
    #                                      ↑
    #                                broadcasts to N_Heads
    xq_rotated = xq_ * freqs_cis
    # Shape: [Seq, N_Heads, Head_Dim//2] (complex64)
    
    
    # STEP 4: Convert back to real numbers                   
    # 
    # view_as_real converts: (a + bi) → [a, b]
    # This unpacks each complex number back into 2 real numbers
    # Shape becomes: [Seq, N_Heads, Head_Dim//2, 2]
    # 
    # flatten(2) merges last two dimensions: [..., Head_Dim//2, 2] → [..., Head_Dim]
    xq_out = torch.view_as_real(xq_rotated).flatten(2)
    # Shape: [Seq, N_Heads, Head_Dim]
    
    # Same process for keys
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    # Shape: [Seq, N_Heads, Head_Dim]
    
    # Convert back to original dtype (e.g., float16, bfloat16)
    return xq_out.type_as(xq), xk_out.type_as(xk)


"""
═══════════════════════════════════════════════════════════════════════════
                            KEY INSIGHTS & BENEFITS
═══════════════════════════════════════════════════════════════════════════

1. RELATIVE POSITION ENCODING
   ───────────────────────────
   When computing attention between tokens at positions i and j:
   
   Q_i · K_j^T = (RoPE(i) · q_i) · (RoPE(j) · k_j)^T
               = q_i · RoPE(j-i) · k_j^T
   
   The rotation naturally encodes RELATIVE distance (j-i)
   This is why RoPE works so well for understanding token relationships.


2. NO ADDITIONAL PARAMETERS
   ─────────────────────────
   - Traditional position embeddings: Need learned embedding matrix
   - RoPE: Just mathematical rotations, no parameters to learn
   - Reduces model size and training complexity


3. EXTRAPOLATION TO LONGER SEQUENCES
   ──────────────────────────────────
   - Trained on sequences of length N
   - Can generalize to length > N reasonably well
   - Rotation angles continue smoothly beyond training range
   
   Why it works:
   - Position 2048 is just "more rotation" than position 2047
   - Smooth, continuous function (unlike learned embeddings)


4. DIMENSION-SPECIFIC FREQUENCIES
   ────────────────────────────────
   Different dimension pairs rotate at different speeds:
   
   Low dims (fast):  Capture fine-grained position (nearby tokens)
   High dims (slow): Capture coarse-grained position (distant tokens)
   
   It's like having multiple "scales" of position information
   Similar to Fourier transforms or wavelets in signal processing.


5. EFFICIENT COMPUTATION
   ──────────────────────
   - Precompute freqs_cis once for max sequence length
   - Apply with simple complex multiplication
   - Very fast on GPU (native complex number support)


COMPARISON WITH OTHER POSITIONAL ENCODINGS:
───────────────────────────────────────────

Absolute (Learned):    [E₀, E₁, E₂, ..., Eₙ] added to tokens
   ✓ Simple
   ✗ Doesn't capture relative position well
   ✗ Extra parameters
   ✗ Poor extrapolation

Sinusoidal (Fixed):    sin/cos functions added to tokens
   ✓ No parameters
   ✓ Some relative position info
   ✗ Position info can fade through layers

RoPE (Rotary):         Rotation applied to Q/K
   ✓ No parameters
   ✓ Strong relative position encoding
   ✓ Preserved through attention
   ✓ Good extrapolation
   ✓ Used in most modern LLMs!

═══════════════════════════════════════════════════════════════════════════
"""