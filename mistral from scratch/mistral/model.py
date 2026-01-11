import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch import nn
from simple_parsing.helpers import Serializable

from mistral.rope import precompute_freqs_cis, apply_rotary_emb
from mistral.cache import CacheView, RotatingBufferCache
from mistral.moe import MoeArgs, MoeLayer

from xformers.ops.fmha import memory_efficient_attention


"""
═══════════════════════════════════════════════════════════════════════════
                    MISTRAL TRANSFORMER ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════

COMPLETE ARCHITECTURE OVERVIEW:
───────────────────────────────

                    Input Token IDs [1, 2, 3, 4, 5]
                             ↓
                    ┌────────────────────┐
                    │ Token Embeddings   │  vocab_size → dim
                    └────────────────────┘
                             ↓
            ╔════════════════════════════════════╗
            ║     Transformer Block 1            ║
            ║  ┌──────────────────────────────┐  ║
            ║  │ RMSNorm                      │  ║
            ║  │ Multi-Head Attention         │  ← RoPE + KV Cache
            ║  │  + Residual                  │  ║
            ║  └──────────────────────────────┘  ║
            ║  ┌──────────────────────────────┐  ║
            ║  │ RMSNorm                      │  ║
            ║  │ FeedForward / MoE            │  ← Dense or Sparse
            ║  │  + Residual                  │  ║
            ║  └──────────────────────────────┘  ║
            ╚════════════════════════════════════╝
                             ↓
            ╔════════════════════════════════════╗
            ║     Transformer Block 2            ║
            ║            ...                     ║
            ╚════════════════════════════════════╝
                             ↓
                           ...
                             ↓
            ╔════════════════════════════════════╗
            ║     Transformer Block N            ║
            ║            ...                     ║
            ╚════════════════════════════════════╝
                             ↓
                    ┌────────────────────┐
                    │ Final RMSNorm      │
                    └────────────────────┘
                             ↓
                    ┌────────────────────┐
                    │ Output Projection  │  dim → vocab_size
                    └────────────────────┘
                             ↓
                    Logits [vocab_size]
                             ↓
                    Softmax → Next Token


KEY COMPONENTS:
───────────────
1. Token Embeddings: Convert token IDs to vectors
2. Positional Encoding: RoPE (rotary position embeddings)
3. Attention: Multi-head with GQA (grouped query attention)
4. FeedForward/MoE: Dense or sparse expert layers
5. Normalization: RMSNorm (simpler than LayerNorm)
6. KV Cache: Efficient generation with sliding window

═══════════════════════════════════════════════════════════════════════════
"""


@dataclass
class ModelArgs(Serializable):
    """
    Configuration for the Mistral Transformer model.
    
    These hyperparameters define the model architecture.
    Example values for Mistral-7B:
        dim: 4096
        n_layers: 32
        n_heads: 32
        n_kv_heads: 8  (GQA with 4x fewer KV heads)
        sliding_window: 4096
    """
    dim: int                    # Model dimension (hidden size, e.g., 4096)
    n_layers: int              # Number of transformer blocks (e.g., 32)
    head_dim: int              # Dimension per attention head (e.g., 128)
    hidden_dim: int            # FFN intermediate dimension (e.g., 14336)
    n_heads: int               # Number of query heads (e.g., 32)
    n_kv_heads: int            # Number of key/value heads for GQA (e.g., 8)
    norm_eps: float            # Epsilon for RMSNorm (e.g., 1e-5)
    vocab_size: int            # Vocabulary size (e.g., 32000)

    max_batch_size: int = 0    # Maximum batch size for inference

    rope_theta: Optional[float] = None      # Base for RoPE frequencies
    sliding_window: Optional[int] = None    # Sliding window size (e.g., 4096)
    moe: Optional[MoeArgs] = None          # MoE configuration (if using experts)


@dataclass
class SimpleInputMetadata:
    """
    Minimal metadata when NOT using KV cache.
    
    Just contains token positions for RoPE - no cache management needed.
    """
    positions: torch.Tensor   

    @staticmethod
    def from_seqlens(seqlens: List[int], device: torch.device) -> "SimpleInputMetadata":
        """
        Generate positions for each token in the batch.
        
        Example:
            seqlens = [3, 2]  (2 sequences: lengths 3 and 2)
            positions = [0,1,2, 0,1]  (each sequence starts from 0)
        """
        return SimpleInputMetadata(
            positions=torch.cat([torch.arange(0, seqlen) for seqlen in seqlens]).to(
                device=device, dtype=torch.long
            )
        )


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int):
    """
    Repeat keys and values for Grouped Query Attention (GQA).
    
    GQA uses fewer KV heads than query heads to save memory.
    We repeat each KV head to match the number of query heads.
    
    Args:
        keys/values: [Seq, N_KV_Heads, Head_Dim]
        repeats: How many times to repeat (n_heads // n_kv_heads)
        dim: Dimension to repeat along (1 for heads)
    
    Returns:
        Repeated keys/values: [Seq, N_Heads, Head_Dim]
    
    
    ═══════════════════════════════════════════════════════════════════
                    GROUPED QUERY ATTENTION (GQA)
    ═══════════════════════════════════════════════════════════════════
    
    Standard Multi-Head Attention (MHA):
        Q heads: 32  [H₀, H₁, H₂, ..., H₃₁]
        K heads: 32  [H₀, H₁, H₂, ..., H₃₁]
        V heads: 32  [H₀, H₁, H₂, ..., H₃₁]
        
        Each query head has its own K/V head
        Memory: 32 + 32 = 64 heads for KV
    
    
    Grouped Query Attention (GQA) with 4 groups:
        Q heads: 32  [H₀, H₁, H₂, ..., H₃₁]
        K heads: 8   [H₀, H₁, H₂, H₃, H₄, H₅, H₆, H₇]
        V heads: 8   [H₀, H₁, H₂, H₃, H₄, H₅, H₆, H₇]
        
        Groups of 4 query heads share same K/V head:
            Q[0,1,2,3] → K[0], V[0]
            Q[4,5,6,7] → K[1], V[1]
            ...
        
        Memory: 8 + 8 = 16 heads for KV (4x reduction!)
    
    
    How repeat_kv works:
        Input K:  [H₀, H₁, H₂, H₃, H₄, H₅, H₆, H₇]
                   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
        repeat    ×4  ×4  ×4  ×4  ×4  ×4  ×4  ×4
                   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
        Output K: [H₀,H₀,H₀,H₀, H₁,H₁,H₁,H₁, ..., H₇,H₇,H₇,H₇]
                   └────┬────┘  └────┬────┘       └────┬────┘
                    Group 0      Group 1           Group 7
    
    Benefits:
    - 4x less KV cache memory
    - Minimal quality loss
    - Used by: Mistral, Llama 2, many modern LLMs
    
    ═══════════════════════════════════════════════════════════════════
    """
    # torch.repeat_interleave: [1,2,3] with repeats=2 → [1,1,2,2,3,3]
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


class Attention(nn.Module):
    """
    Multi-Head Attention with Grouped Query Attention (GQA).
    
    This implements the core attention mechanism:
    - Projects input to Q, K, V
    - Applies RoPE positional encoding
    - Computes scaled dot-product attention
    - Handles KV caching for efficient generation
    
    ═══════════════════════════════════════════════════════════════════
                        ATTENTION COMPUTATION
    ═══════════════════════════════════════════════════════════════════
    
    Step 1: Linear Projections
        Input: [Seq, Dim]
        Q: [Seq, N_Heads × Head_Dim]
        K: [Seq, N_KV_Heads × Head_Dim]  ← Fewer heads (GQA)
        V: [Seq, N_KV_Heads × Head_Dim]  ← Fewer heads (GQA)
    
    Step 2: Reshape to Multi-Head
        Q: [Seq, N_Heads, Head_Dim]
        K: [Seq, N_KV_Heads, Head_Dim]
        V: [Seq, N_KV_Heads, Head_Dim]
    
    Step 3: Apply RoPE (Positional Encoding)
        Q_rotated, K_rotated = RoPE(Q, K)
    
    Step 4: KV Caching (if generating)
        Cache K and V for reuse
        Retrieve past K, V from cache
    
    Step 5: Repeat KV for GQA
        K: [Seq, N_KV_Heads, Head_Dim] → [Seq, N_Heads, Head_Dim]
        V: [Seq, N_KV_Heads, Head_Dim] → [Seq, N_Heads, Head_Dim]
    
    Step 6: Scaled Dot-Product Attention
        Attention(Q,K,V) = softmax(Q·Kᵀ / √d) · V
        
        For each query:
            1. Compute similarity with all keys: Q·Kᵀ
            2. Scale by √head_dim
            3. Apply causal mask (can't see future)
            4. Softmax to get attention weights
            5. Weighted sum of values
    
    Step 7: Output Projection
        [Seq, N_Heads, Head_Dim] → [Seq, N_Heads × Head_Dim] → [Seq, Dim]
    
    ═══════════════════════════════════════════════════════════════════
    """
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads          # Query heads (e.g., 32)
        self.head_dim: int = args.head_dim        # Dimension per head (e.g., 128)
        self.n_kv_heads: int = args.n_kv_heads    # KV heads (e.g., 8 for GQA)

        # How many times to repeat KV heads to match query heads
        # Example: 32 query heads / 8 KV heads = 4 repeats
        self.repeats = self.n_heads // self.n_kv_heads

        # Scale factor for attention: 1/√d
        # Prevents dot products from getting too large (helps gradients)
        self.scale = self.args.head_dim**-0.5

        # ┌────────────────────────────────────────────────────────┐
        # │ Linear projections for Q, K, V                         │
        # └────────────────────────────────────────────────────────┘
        # Query projection: dim → n_heads × head_dim
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        
        # Key/Value projections: dim → n_kv_heads × head_dim (fewer heads!)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        
        # Output projection: n_heads × head_dim → dim
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[CacheView],
    ) -> torch.Tensor:
        """
        Forward pass through attention.
        
        Args:
            x: Input tensor [Seq, Dim]
            freqs_cis: RoPE frequencies [Seq, Head_Dim//2] (complex numbers)
            cache: Optional KV cache for efficient generation
        
        Returns:
            Output tensor [Seq, Dim]
        """
        seqlen_sum, _ = x.shape  # Total tokens in batch (concatenated sequences)

        # ┌────────────────────────────────────────────────────────┐
        # │ STEP 1: Project to Q, K, V                             │
        # └────────────────────────────────────────────────────────┘
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        # ┌────────────────────────────────────────────────────────┐
        # │ STEP 2: Reshape to multi-head format                   │
        # └────────────────────────────────────────────────────────┘
        # Queries: [Seq, Dim] → [Seq, N_Heads, Head_Dim]
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        
        # Keys/Values: [Seq, Dim] → [Seq, N_KV_Heads, Head_Dim]
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        
        # ┌────────────────────────────────────────────────────────┐
        # │ STEP 3: Apply RoPE positional encoding                 │
        # └────────────────────────────────────────────────────────┘
        # Rotates Q and K by position-dependent angles
        # This encodes position information directly into Q/K
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # ┌────────────────────────────────────────────────────────┐
        # │ STEP 4: Handle KV caching                              │
        # └────────────────────────────────────────────────────────┘
        if cache is None:
            # No caching - use current K, V directly
            # Used during training or first forward pass
            key, val = xk, xv
            
        elif cache.prefill:
            # Prefill mode: Processing prompt (multiple tokens at once)
            # Need both cached K/V AND new K/V for attention
            key, val = cache.interleave_kv(xk, xv)  # Combine: [cached..., new...]
            cache.update(xk, xv)  # Then update cache with new K/V
            
        else:
            # Generation mode: Generating one token at a time
            # First update cache with new K/V
            cache.update(xk, xv)
            
            # Then retrieve ALL cached K/V (including what we just added)
            key, val = cache.key, cache.value
            
            # Reshape for attention computation
            # [Batch, Window, Heads, Dim] → [Batch×Window, Heads, Dim]
            key = key.view(
                seqlen_sum * cache.sliding_window, self.n_kv_heads, self.head_dim
            )
            val = val.view(
                seqlen_sum * cache.sliding_window, self.n_kv_heads, self.head_dim
            )

        # ┌────────────────────────────────────────────────────────┐
        # │ STEP 5: Repeat K/V for Grouped Query Attention         │
        # └────────────────────────────────────────────────────────┘
        # Expand KV from n_kv_heads to n_heads by repeating
        # Example: 8 KV heads → 32 heads (repeat 4x)
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # ┌────────────────────────────────────────────────────────┐
        # │ STEP 6: Compute attention with xformers                │
        # └────────────────────────────────────────────────────────┘
        # xformers requires batch dimension, so add it: [Seq,...] → [1, Seq,...]
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        
        # memory_efficient_attention is optimized (Flash Attention style)
        # Computes: softmax(Q·Kᵀ / √d) · V with mask
        output = memory_efficient_attention(
            xq, key, val, None if cache is None else cache.mask
        )
        # Output: [1, Seq, N_Heads, Head_Dim]

        # ┌────────────────────────────────────────────────────────┐
        # │ STEP 7: Reshape and project output                     │
        # └────────────────────────────────────────────────────────┘
        # [1, Seq, N_Heads, Head_Dim] → [Seq, N_Heads * Head_Dim]
        output = output.view(seqlen_sum, self.n_heads * self.head_dim)
        
        # Final projection: [Seq, N_Heads * Head_Dim] → [Seq, Dim]
        return self.wo(output)


class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) with SwiGLU activation.
    
    This is applied after attention in each transformer block.
    
    ═══════════════════════════════════════════════════════════════════
                        SWIGLU FEEDFORWARD
    ═══════════════════════════════════════════════════════════════════
    
    Standard FFN:
        FFN(x) = W₂ · ReLU(W₁ · x)
        
        Problem: ReLU zeros out negative values → info loss
    
    
    SwiGLU (better):
        FFN(x) = W₂ · (SiLU(W₁ · x) ⊙ W₃ · x)
        
        Where:
        - SiLU(x) = x · sigmoid(x)  (smooth activation)
        - ⊙ = element-wise multiplication (gating)
        
        Benefits:
        - Smoother gradients than ReLU
        - Gating mechanism (W₃) controls information flow
        - Better performance empirically
        - Used by: LLaMA, Mistral, PaLM
    
    
    Architecture:
        Input: [Seq, Dim=4096]
               ↓
        ┌──────────────────────────┐
        │ W₁: [Dim, Hidden=14336]  │ ← Expand
        └──────────────────────────┘
               ↓
        ┌──────────────────────────┐
        │ SiLU activation          │
        └──────────────────────────┘
               ↓
        ┌──────────────────────────┐
        │ W₃: [Dim, Hidden=14336]  │ ← Gate
        └──────────────────────────┘
               ↓
        Element-wise multiply ⊙
               ↓
        ┌──────────────────────────┐
        │ W₂: [Hidden, Dim=4096]   │ ← Project back
        └──────────────────────────┘
               ↓
        Output: [Seq, Dim=4096]
    
    Note: Hidden dim is usually ~3.5x model dim
          (14336 / 4096 ≈ 3.5 for Mistral-7B)
    
    ═══════════════════════════════════════════════════════════════════
    """
    
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Three linear layers (no bias)
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)  # Expand
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)  # Project back
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)  # Gate

    def forward(self, x) -> torch.Tensor:
        """
        SwiGLU forward pass.
        
        Formula: W₂(SiLU(W₁(x)) ⊙ W₃(x))
        
        Args:
            x: [Seq, Dim]
        
        Returns:
            [Seq, Dim]
        """
        # SiLU(W₁(x)) ⊙ W₃(x)
        # silu is also called "swish": x * sigmoid(x)
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Simpler alternative to LayerNorm - only normalizes scale, not shift.
    
    ═══════════════════════════════════════════════════════════════════
                        RMSNORM vs LAYERNORM
    ═══════════════════════════════════════════════════════════════════
    
    LayerNorm (traditional):
        y = γ · (x - μ) / σ + β
        
        Where:
        - μ = mean(x)
        - σ = std(x)
        - γ, β = learned parameters
        
        Normalizes both mean and variance
    
    
    RMSNorm (simpler):
        y = γ · x / RMS(x)
        
        Where:
        - RMS(x) = √(mean(x²) + ε)
        - γ = learned parameter (no β!)
        
        Only normalizes scale (RMS), not mean
    
    
    Why RMSNorm?
    ────────────
    1. Simpler: No need to compute/subtract mean
    2. Faster: Fewer operations
    3. Fewer parameters: Only γ, no β
    4. Works just as well: Empirically similar performance
    5. Better gradients: More stable training
    
    Used by: T5, LLaMA, Mistral, many modern LLMs
    
    ═══════════════════════════════════════════════════════════════════
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: Model dimension
            eps: Small constant for numerical stability (avoid div by 0)
        """
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (one per dimension)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Compute RMS normalization: x / RMS(x)
        
        RMS(x) = √(mean(x²) + ε)
        
        Args:
            x: [Seq, Dim]
        
        Returns:
            Normalized x: [Seq, Dim]
        """
        # x.pow(2): Square each element
        # .mean(-1, keepdim=True): Mean over last dim (Dim), keep shape
        # + self.eps: Add epsilon for stability
        # torch.rsqrt: Reciprocal square root (1/√x) - faster than 1/sqrt(x)
        # Result: x / √(mean(x²) + ε)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Apply RMSNorm.
        
        Args:
            x: Input tensor [Seq, Dim]
        
        Returns:
            Normalized and scaled output [Seq, Dim]
        """
        # Normalize in float32 for stability, then convert back
        output = self._norm(x.float()).type_as(x)
        
        # Apply learned scale: γ · normalized(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    """
    Single Transformer Block.
    
    This is the core building block, repeated N times in the model.
    
    ═══════════════════════════════════════════════════════════════════
                    TRANSFORMER BLOCK ARCHITECTURE
    ═══════════════════════════════════════════════════════════════════
    
                        Input: x
                          ↓
                    ┌─────────────┐
                    │  RMSNorm    │
                    └─────────────┘
                          ↓
                    ┌─────────────┐
                    │  Attention  │  ← Multi-head + RoPE + KV cache
                    └─────────────┘
                          ↓
                        + ←─────── Residual connection
                          ↓
                        h = x + Attention(Norm(x))
                          ↓
                    ┌─────────────┐
                    │  RMSNorm    │
                    └─────────────┘
                          ↓
                    ┌─────────────┐
                    │  FFN / MoE  │  ← SwiGLU or Mixture of Experts
                    └─────────────┘
                          ↓
                        + ←─────── Residual connection
                          ↓
                        out = h + FFN(Norm(h))
                          ↓
                        Output
    
    
    KEY FEATURES:
    ─────────────
    1. Pre-Norm: Normalize BEFORE attention/FFN (more stable training)
    2. Residual connections: x + F(x) helps gradient flow
    3. Two sub-layers: Attention → FFN
    4. Optional MoE: Can use sparse experts instead of dense FFN
    
    ═══════════════════════════════════════════════════════════════════
    """
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        
        # Attention sub-layer
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # FFN sub-layer normalization
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        self.args = args

        # ┌────────────────────────────────────────────────────────┐
        # │ Feed-Forward: Dense or Mixture of Experts              │
        # └────────────────────────────────────────────────────────┘
        self.feed_forward: nn.Module
        if args.moe is not None:
            # Sparse MoE: Multiple expert networks with routing
            # Used in Mistral-8x7B (8 experts, activate 2 per token)
            self.feed_forward = MoeLayer(
                experts=[FeedForward(args=args) for _ in range(args.moe.num_experts)],
                gate=nn.Linear(args.dim, args.moe.num_experts, bias=False),
                moe_args=args.moe,
            )
        else:
            # Dense FFN: Single network for all tokens
            # Used in Mistral-7B
            self.feed_forward = FeedForward(args=args)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, cache: Optional[CacheView]
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input [Seq, Dim]
            freqs_cis: RoPE frequencies [Seq, Head_Dim//2]
            cache: Optional KV cache
        
        Returns:
            Output [Seq, Dim]
        """
        # ┌────────────────────────────────────────────────────────┐
        # │ Attention sub-layer with residual                      │
        # └────────────────────────────────────────────────────────┘
        # Normalize → Attention → Add residual
        r = self.attention.forward(self.attention_norm(x), freqs_cis, cache)
        h = x + r  # Residual connection
        
        # ┌────────────────────────────────────────────────────────┐
        # │ FFN sub-layer with residual                            │
        # └────────────────────────────────────────────────────────┘
        # Normalize → FFN/MoE → Add residual
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r  # Residual connection
        
        return out


class Transformer(nn.Module):
    """
    Complete Mistral Transformer model.
    
    This is the top-level class that combines all components:
    - Token embeddings
    - N transformer blocks  
    - Final normalization
    - Output projection to vocabulary
    
    Also supports pipeline parallelism for multi-GPU inference.
    
    ═══════════════════════════════════════════════════════════════════
                    PIPELINE PARALLELISM
    ═══════════════════════════════════════════════════════════════════
    
    For large models, layers can be split across multiple GPUs:
    
    Single GPU (no pipeline):
        GPU 0: [Embed → Block0...Block31 → Norm → Output]
    
    
    2-GPU Pipeline:
        GPU 0: [Embed → Block0...Block15] ──> Send to GPU 1
        GPU 1: Receive ← [Block16...Block31 → Norm → Output]
    
    
    4-GPU Pipeline:
        GPU 0: [Embed → Block0...Block7] ──>
        GPU 1: [Block8...Block15] ──>
        GPU 2: [Block16...Block23] ──>
        GPU 3: [Block24...Block31 → Norm → Output]
    
    
    Benefits:
    - Fit larger models than single GPU memory
    - Each GPU only needs ~1/N parameters
    - Layers computed sequentially (some idle time)
    
    Trade-offs:
    - Communication overhead between GPUs
    - GPU utilization not 100% (pipeline bubbles)
    - More complex than data parallelism
    
    ═══════════════════════════════════════════════════════════════════
    """
    
    def __init__(
        self,
        args: ModelArgs,
        pipeline_rank: int = 0,
        num_pipeline_ranks: int = 1,
    ):
        """
        Args:
            args: Model configuration
            pipeline_rank: Which GPU is this (0 to num_pipeline_ranks-1)
            num_pipeline_ranks: Total number of GPUs in pipeline
        """
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self._precomputed_freqs_cis: Optional[torch.Tensor] = None
        
        assert self.vocab_size > 0
        assert pipeline_rank < num_pipeline_ranks, (pipeline_rank, num_pipeline_ranks)
        
        self.pipeline_rank = pipeline_rank
        self.num_pipeline_ranks = num_pipeline_ranks
        
        # ┌────────────────────────────────────────────────────────┐
        # │ Rank-specific modules                                  │
        # └────────────────────────────────────────────────────────┘
        # Only some ranks need certain modules:
        
        self.tok_embeddings: Optional[nn.Embedding] = None
        self.norm: Optional[RMSNorm] = None
        self.output: Optional[nn.Linear] = None
        
        if pipeline_rank == 0:
            # First GPU: Has token embeddings
            self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
            
        if pipeline_rank == num_pipeline_ranks - 1:
            # Last GPU: Has final norm and output projection
            self.norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        # ┌────────────────────────────────────────────────────────┐
        # │ Distribute layers across GPUs                          │
        # └────────────────────────────────────────────────────────┘
        # Create all layers, but only keep those for this rank
        layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        
        # Divide layers evenly across ranks
        # Example: 32 layers, 4 GPUs → 8 layers per GPU
        num_layers_per_rank = math.ceil(self.n_layers / self.num_pipeline_ranks)
        offset = self.pipeline_rank * num_layers_per_rank
        end = min(self.n_layers, offset + num_layers_per_rank)
        
        # Keep only layers for this rank in a ModuleDict
        # Example for rank 1: layers 8-15 → {"8": Block8, "9": Block9, ..., "15": Block15}
        self.layers = nn.ModuleDict({str(i): layers[i] for i in range(offset, end)})
        self.n_local_layers = len(self.layers)

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype of model parameters (e.g., float16, bfloat16)."""
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        """Get device where model is located (e.g., cuda:0)."""
        return next(self.parameters()).device

    @property
    def freqs_cis(self) -> torch.Tensor:
        """
        Get precomputed RoPE frequencies.
        
        These are cached and reused across forward passes.
        Must be recomputed if device changes.
        
        Returns:
            Complex tensor [Max_Seq_Len, Head_Dim//2]
        """
        # ┌────────────────────────────────────────────────────────┐
        # │ Lazy initialization of RoPE frequencies                │
        # └────────────────────────────────────────────────────────┘
        if self._precomputed_freqs_cis is None:
            # Determine theta (base for frequency calculation)
            theta = self.args.rope_theta
            if theta is None:
                # Default theta depends on whether using sliding window
                # No sliding window: larger theta (1M) for longer sequences
                # With sliding window: smaller theta (10k) is sufficient
                theta = 1000000.0 if self.args.sliding_window is None else 10000.0
            
            # Precompute for up to 128k tokens (very long sequences)
            self._precomputed_freqs_cis = precompute_freqs_cis(
                self.args.head_dim, 128_000, theta
            )
        
        # ┌────────────────────────────────────────────────────────┐
        # │ Ensure frequencies are on correct device               │
        # └────────────────────────────────────────────────────────┘
        if self._precomputed_freqs_cis.device != self.device:
            self._precomputed_freqs_cis = self._precomputed_freqs_cis.to(
                device=self.device
            )
        
        return self._precomputed_freqs_cis

    def forward_partial(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        cache: Optional[RotatingBufferCache] = None,
    ) -> torch.Tensor:
        """
        Local forward pass for this pipeline rank.
        
        This handles:
        - Embeddings (rank 0 only)
        - Local transformer blocks
        - Communication between ranks
        - Final normalization (last rank only)
        
        Args:
            input_ids: Token IDs [Total_Tokens]
            seqlens: Length of each sequence in batch
            cache: Optional KV cache
        
        Returns:
            Activations for next rank OR normalized embeddings (last rank)
        
        
        ═══════════════════════════════════════════════════════════════
                    FORWARD PASS EXAMPLE (2 sequences)
        ═══════════════════════════════════════════════════════════════
        
        Input:
            input_ids = [1, 2, 3, 4, 5]  (concatenated tokens)
            seqlens = [3, 2]              (sequence lengths)
            
            Sequence 0: tokens [1, 2, 3]
            Sequence 1: tokens [4, 5]
        
        
        Rank 0 (first GPU):
        ───────────────────
        1. Token Embedding:
           [1,2,3,4,5] → [[e₁],[e₂],[e₃],[e₄],[e₅]]
           Shape: [5, 4096]
        
        2. Get RoPE frequencies for positions [0,1,2,0,1]
        
        3. Apply local layers (e.g., Block 0-15):
           [5, 4096] → [5, 4096]
        
        4. Send to rank 1:
           torch.distributed.send(activations, dst=1)
        
        
        Rank 1 (last GPU):
        ──────────────────
        1. Receive from rank 0:
           torch.distributed.recv(activations, src=0)
           Shape: [5, 4096]
        
        2. Apply local layers (e.g., Block 16-31):
           [5, 4096] → [5, 4096]
        
        3. Final RMSNorm:
           [5, 4096] → [5, 4096]
        
        4. Return normalized activations
        
        ═══════════════════════════════════════════════════════════════
        """
        # Validate batch size
        assert (
            len(seqlens) <= self.args.max_batch_size
        ), f"Max batch size is {self.args.max_batch_size}, got batch size of {len(seqlens)}"
        
        # Validate total tokens
        (num_toks,) = input_ids.shape
        assert sum(seqlens) == num_toks, (sum(seqlens), num_toks)
        
        # ┌────────────────────────────────────────────────────────┐
        # │ Get input metadata (positions, masks, cache info)      │
        # └────────────────────────────────────────────────────────┘
        if cache is not None:
            # Using cache: get full metadata (masks, cache positions, etc.)
            input_metadata = cache.get_input_metadata(seqlens)
        else:
            # No cache: just need positions for RoPE
            input_metadata = SimpleInputMetadata.from_seqlens(seqlens, self.device)

        # ┌────────────────────────────────────────────────────────┐
        # │ Token Embedding (rank 0 only)                          │
        # └────────────────────────────────────────────────────────┘
        if self.pipeline_rank == 0:
            # First rank: embed tokens
            assert self.tok_embeddings is not None
            h = self.tok_embeddings(input_ids)  # [num_toks] → [num_toks, dim]
        else:
            # Other ranks: receive embeddings from previous rank
            h = torch.empty(
                num_toks, self.args.dim, device=self.device, dtype=self.dtype
            )
            torch.distributed.recv(h, src=self.pipeline_rank - 1)

        # ┌────────────────────────────────────────────────────────┐
        # │ Get RoPE frequencies for current positions             │
        # └────────────────────────────────────────────────────────┘
        # Select frequencies for each token's position
        # Example: positions [0,1,2,0,1] → freqs_cis[[0,1,2,0,1]]
        freqs_cis = self.freqs_cis[input_metadata.positions]

        # ┌────────────────────────────────────────────────────────┐
        # │ Apply transformer blocks (local to this rank)          │
        # └────────────────────────────────────────────────────────┘
        for local_layer_id, layer in enumerate(self.layers.values()):
            if cache is not None:
                assert input_metadata is not None
                # Get cache view for this layer
                cache_view = cache.get_view(local_layer_id, input_metadata)
            else:
                cache_view = None
            
            # Apply transformer block
            h = layer(h, freqs_cis, cache_view)

        # ┌────────────────────────────────────────────────────────┐
        # │ Update cache sequence lengths                          │
        # └────────────────────────────────────────────────────────┘
        if cache is not None:
            # Increment total tokens seen by each sequence
            cache.update_seqlens(seqlens)
        
        # ┌────────────────────────────────────────────────────────┐
        # │ Communication and final processing                     │
        # └────────────────────────────────────────────────────────┘
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            # Not last rank: send to next rank
            torch.distributed.send(h, dst=self.pipeline_rank + 1)
            return h
        else:
            # Last rank: apply final normalization
            assert self.norm is not None
            return self.norm(h)

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        cache: Optional[RotatingBufferCache] = None,
    ) -> torch.Tensor:
        """
        Complete forward pass through the model.
        
        This adds the final output projection on top of forward_partial.
        
        Args:
            input_ids: Token IDs [Total_Tokens]
            seqlens: Length of each sequence in batch
            cache: Optional KV cache
        
        Returns:
            Logits over vocabulary [Total_Tokens, Vocab_Size]
        
        
        ═══════════════════════════════════════════════════════════════
                    COMPLETE FORWARD PASS
        ═══════════════════════════════════════════════════════════════
        
        Input: Token IDs [5] = [1, 2, 3, 4, 5]
               ↓
        ┌──────────────────────┐
        │ Token Embeddings     │  [5] → [5, 4096]
        └──────────────────────┘
               ↓
        ┌──────────────────────┐
        │ Transformer Blocks   │  [5, 4096] → [5, 4096]
        │ (32 layers)          │
        └──────────────────────┘
               ↓
        ┌──────────────────────┐
        │ Final RMSNorm        │  [5, 4096] → [5, 4096]
        └──────────────────────┘
               ↓
        ┌──────────────────────┐
        │ Output Projection    │  [5, 4096] → [5, 32000]
        └──────────────────────┘
               ↓
        Logits: [5, 32000]
               ↓
        For each token, get probability over all vocab:
            Token 1 logits: [0.1, 0.05, 0.8, ...]  (32k values)
            Token 2 logits: [0.2, 0.15, 0.6, ...]
            ...
        
        Highest logit → Most likely next token
        
        ═══════════════════════════════════════════════════════════════
        """
        # Get normalized embeddings from all transformer layers
        h = self.forward_partial(input_ids, seqlens, cache=cache)
        
        # ┌────────────────────────────────────────────────────────┐
        # │ Output projection to vocabulary                        │
        # └────────────────────────────────────────────────────────┘
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            # Not last rank: receive final logits from last rank
            outs = torch.empty(
                h.shape[0], self.vocab_size, device=h.device, dtype=h.dtype
            )
        else:
            # Last rank: project to vocabulary
            assert self.output is not None
            outs = self.output(h)  # [num_toks, dim] → [num_toks, vocab_size]
        
        # ┌────────────────────────────────────────────────────────┐
        # │ Broadcast final output to all ranks (if pipeline)      │
        # └────────────────────────────────────────────────────────┘
        if self.num_pipeline_ranks > 1:
            # All ranks need the logits (e.g., for sampling)
            torch.distributed.broadcast(outs, src=self.num_pipeline_ranks - 1)
        
        # Return in float32 for numerical stability
        return outs.float()

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Load model parameters, handling pipeline parallelism.
        
        Each rank only loads the parameters it needs:
        - Rank 0: embeddings + its layers
        - Middle ranks: only their layers
        - Last rank: its layers + norm + output
        
        Args:
            state_dict: Dictionary of parameter tensors
        """
        state_to_load = {}
        skipped = set([])
        
        for k, v in state_dict.items():
            # ┌────────────────────────────────────────────────────┐
            # │ Token embeddings (rank 0 only)                     │
            # └────────────────────────────────────────────────────┘
            if k.startswith("tok_embeddings"):
                if self.pipeline_rank == 0:
                    state_to_load[k] = v
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            
            # ┌────────────────────────────────────────────────────┐
            # │ Final norm and output (last rank only)             │
            # └────────────────────────────────────────────────────┘
            elif k.startswith("norm") or k.startswith("output"):
                if self.pipeline_rank == self.num_pipeline_ranks - 1:
                    state_to_load[k] = v
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            
            # ┌────────────────────────────────────────────────────┐
            # │ Transformer layers (load if in this rank)          │
            # └────────────────────────────────────────────────────┘
            elif k.startswith("layers"):
                # Extract layer ID from key (e.g., "layers.5.attention.wq.weight" → "5")
                layer_id = k.split(".")[1]
                if layer_id in self.layers:
                    state_to_load[k] = v
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            else:
                raise ValueError(f"Unexpected key {k}")
        
        # Verify we handled all parameters
        assert set(state_dict.keys()) == skipped.union(set(state_to_load.keys()))
        
        # Load the parameters for this rank
        super().load_state_dict(state_to_load, *args, **kwargs)

    @staticmethod
    def from_folder(
        folder: Path,
        max_batch_size: int = 1,
        num_pipeline_ranks: int = 1,
        device="cuda",
        dtype=torch.float16,
    ) -> "Transformer":
        """
        Load a trained model from a folder.
        
        Args:
            folder: Path to model directory (contains params.json and consolidated.00.pth)
            max_batch_size: Maximum batch size for inference
            num_pipeline_ranks: Number of GPUs for pipeline parallelism
            device: Device to load model on
            dtype: Data type for model parameters
        
        Returns:
            Loaded Transformer model
        
        
        Model folder structure:
        ───────────────────────
        model_folder/
        ├── params.json           ← Model configuration (dims, layers, etc.)
        └── consolidated.00.pth   ← Trained weights
        
        
        Loading process:
        ────────────────
        1. Read params.json → Create ModelArgs
        2. Create model with "meta" device (no memory allocated)
        3. Load weights with mmap (memory-mapped file, efficient)
        4. Move to target device and dtype
        """
        # ┌────────────────────────────────────────────────────────┐
        # │ Load model configuration                               │
        # └────────────────────────────────────────────────────────┘
        with open(folder / "params.json", "r") as f:
            model_args = ModelArgs.from_dict(json.load(f))
        model_args.max_batch_size = max_batch_size
        
        # ┌────────────────────────────────────────────────────────┐
        # │ Determine pipeline rank                                │
        # └────────────────────────────────────────────────────────┘
        if num_pipeline_ranks > 1:
            # Multi-GPU: get rank from distributed process group
            pipeline_rank = torch.distributed.get_rank()
        else:
            # Single GPU: always rank 0
            pipeline_rank = 0
        
        # ┌────────────────────────────────────────────────────────┐
        # │ Create model on meta device (no memory allocated)      │
        # └────────────────────────────────────────────────────────┘
        # Using "meta" device creates tensors without allocating memory
        # Useful for large models: allocate memory only after loading weights
        with torch.device("meta"):
            model = Transformer(
                model_args,
                pipeline_rank=pipeline_rank,
                num_pipeline_ranks=num_pipeline_ranks,
            )
        
        # ┌────────────────────────────────────────────────────────┐
        # │ Load weights from file                                 │
        # └────────────────────────────────────────────────────────┘
        # mmap=True: Memory-map the file (efficient for large files)
        # Don't load entire file into RAM at once
        loaded = torch.load(str(folder / "consolidated.00.pth"), mmap=True)
        
        # assign=True: Assign tensors directly without copying (efficient)
        model.load_state_dict(loaded, assign=True)
        
        # ┌────────────────────────────────────────────────────────┐
        # │ Move to target device and dtype                        │
        # └────────────────────────────────────────────────────────┘
        return model.to(device=device, dtype=dtype)


"""
═══════════════════════════════════════════════════════════════════════════
                        KEY ARCHITECTURE INSIGHTS
═══════════════════════════════════════════════════════════════════════════

1. GROUPED QUERY ATTENTION (GQA)
   ──────────────────────────────
   - Query heads: 32
   - KV heads: 8 (4x fewer!)
   - Memory: 4x less for KV cache
   - Quality: Minimal degradation
   
   This is THE key innovation that makes Mistral so efficient!


2. SLIDING WINDOW ATTENTION
   ─────────────────────────
   - Window size: 4096 tokens
   - Constant memory regardless of sequence length
   - Can generate infinitely long sequences
   - Local attention is sufficient for most tasks


3. SWIGLU ACTIVATION
   ──────────────────
   - FFN(x) = W₂(SiLU(W₁(x)) ⊙ W₃(x))
   - Better than ReLU/GELU
   - Gating mechanism
   - Used by most modern LLMs


4. RMSNORM
   ────────
   - Simpler than LayerNorm
   - Just normalizes scale, not mean
   - Faster and works just as well
   - Pre-norm architecture (normalize before sublayers)


5. MIXTURE OF EXPERTS (OPTIONAL)
   ──────────────────────────────
   - Mistral-7B: Dense (single FFN)
   - Mistral-8x7B: Sparse MoE (8 experts, activate 2)
   - 5x parameters, ~2x compute (sparse activation)
   - Each expert can specialize


6. PIPELINE PARALLELISM
   ────────────────────
   - Split layers across multiple GPUs
   - Each GPU handles subset of layers
   - Sequential processing (some idle time)
   - Enables models larger than single GPU memory


7. INFERENCE OPTIMIZATIONS
   ───────────────────────
   - KV cache: Reuse past computations (500x speedup!)
   - xformers: Flash attention (memory efficient)
   - bfloat16: Half precision (2x faster, same quality)
   - Batch processing: Process multiple sequences together


8. PARAMETER COUNT BREAKDOWN (Mistral-7B)
   ───────────────────────────────────────
   
   Token Embeddings:     32k × 4096 = 131M
   
   Per Layer (32 layers):
     Attention:
       wq: 4096 × 4096 = 16.8M
       wk: 4096 × 1024 = 4.2M   ← GQA savings!
       wv: 4096 × 1024 = 4.2M   ← GQA savings!
       wo: 4096 × 4096 = 16.8M
       Total: ~42M per layer
     
     FFN:
       w1: 4096 × 14336 = 58.7M
       w2: 14336 × 4096 = 58.7M
       w3: 4096 × 14336 = 58.7M
       Total: ~176M per layer
     
     Layer Total: ~218M × 32 = ~7B
   
   Output: 4096 × 32k = 131M
   
   Grand Total: ~7.24B parameters


9. MEMORY REQUIREMENTS (Mistral-7B, bfloat16)
   ───────────────────────────────────────────
   
   Model Weights:     7.24B × 2 bytes = ~14.5 GB
   KV Cache (4096):   2 × 32 × batch × 4096 × 8 × 128 × 2
                      ≈ 2 GB per sequence in batch
   Activations:       Depends on batch size and sequence length
   
   Total (batch=1):   ~17 GB (fits on 24GB GPU!)
   Total (batch=8):   ~30 GB (need 40GB+ GPU)


10. GENERATION SPEED ESTIMATES
    ──────────────────────────
    
    On A100 (80GB), bfloat16:
    - Prefill: ~1000 tokens/second
    - Generation: ~50-100 tokens/second (per sequence)
    
    Bottlenecks:
    - Prefill: Compute-bound (parallel processing)
    - Generation: Memory-bound (sequential, loading weights)
    
    This is why KV cache is crucial - avoids recomputation!

═══════════════════════════════════════════════════════════════════════════
"""