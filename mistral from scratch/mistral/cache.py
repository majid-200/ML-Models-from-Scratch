import torch
from typing import List, Tuple
from dataclasses import dataclass

from xformers.ops.fmha.attn_bias import (
    AttentionBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
)


"""
═══════════════════════════════════════════════════════════════════════════
                    KV CACHE WITH SLIDING WINDOW ATTENTION
═══════════════════════════════════════════════════════════════════════════

WHAT IS KV CACHE?
─────────────────
During text generation, we generate tokens ONE AT A TIME. Without caching,
we'd recompute attention for ALL previous tokens at EVERY step!

Without KV Cache (SLOW):
    Step 1: Generate token 1   → Compute attention for token 1
    Step 2: Generate token 2   → Compute attention for tokens 1,2  (redundant)
    Step 3: Generate token 3   → Compute attention for tokens 1,2,3  (very redundant)
    ...
    Time complexity: O(n²) where n is sequence length

With KV Cache (FAST):
    Step 1: Generate token 1   → Compute K₁, V₁, cache them
    Step 2: Generate token 2   → Reuse cached K₁, V₁, compute K₂, V₂
    Step 3: Generate token 3   → Reuse cached K₁, K₂, V₁, V₂, compute K₃, V₃
    ...
    Time complexity: O(n) - much faster!


WHAT IS SLIDING WINDOW ATTENTION?
──────────────────────────────────
Standard attention: Each token attends to ALL previous tokens
Memory: O(n²) grows quadratically!

Sliding Window: Each token only attends to last W tokens
Memory: O(n·W) - constant memory per token!

Visual (W=3):
    
    Token 0: [●] ─────────────────────────────  Attends to: self
    Token 1: [●][●] ──────────────────────────  Attends to: 0,1
    Token 2: [●][●][●] ───────────────────────  Attends to: 0,1,2
    Token 3:    [●][●][●] ────────────────────  Attends to: 1,2,3  (window slides!)
    Token 4:       [●][●][●] ─────────────────  Attends to: 2,3,4
    Token 5:          [●][●][●] ──────────────  Attends to: 3,4,5
                      └──┬──┘
                    Sliding Window (size 3)


ROTATING BUFFER CACHE
─────────────────────
Instead of growing the cache indefinitely, we use a FIXED-SIZE circular buffer.

Standard cache (grows unbounded):
    Step 1: [K₁, V₁]
    Step 2: [K₁, V₁, K₂, V₂]
    Step 3: [K₁, V₁, K₂, V₂, K₃, V₃]  ← Keeps growing!

Rotating cache (fixed size, window=3):
    Step 1: [K₁, V₁, __, __]
    Step 2: [K₁, V₁, K₂, V₂]
    Step 3: [K₃, V₃, K₂, V₂]  ← Overwrites oldest (K₁,V₁)!
            ↑
        Wrap around - writes to position 0

Like a circular buffer or ring buffer in OS!

Visual of rotating cache:
    
    Position:  0    1    2    ← Physical cache positions
             ┌────┬────┬────┐
    Step 3:  │ K₃ │ K₁ │ K₂ │  (after rotation)
             └────┴────┴────┘
                   ↑ 
              Next write position (oldest data)

To read in correct order, we "unrotate" the cache.


PREFILL vs GENERATION
─────────────────────
Prefill: Process entire prompt at once (parallel)
    Input: "Hello, how are you?"  (5 tokens)
    Process: All 5 tokens in ONE forward pass
    Output: K₁,K₂,K₃,K₄,K₅ and V₁,V₂,V₃,V₄,V₅

Generation: Generate one token at a time (sequential)
    Step 1: Use cached KV + new query → Generate token 6
    Step 2: Use cached KV + new query → Generate token 7
    Step 3: Use cached KV + new query → Generate token 8
    ...

Different attention masks needed for each phase!

═══════════════════════════════════════════════════════════════════════════
"""


@dataclass
class RotatingCacheInputMetadata:
    """
    Metadata for managing the rotating KV cache during attention computation.
    
    This contains all the bookkeeping information needed to:
    1. Track token positions for RoPE
    2. Determine what to cache
    3. Handle variable-length sequences in a batch
    4. Apply correct attention masks
    """
    
    
    # Position Information (for RoPE)
    # 
    positions: torch.Tensor  # [Total_Tokens] - Absolute position of each token in its sequence
    # Example: For 2 sequences of length 3 and 2 starting at positions 5 and 10:
    #          [5, 6, 7, 10, 11]
    
    
    # Cache Management                                       
    # 
    to_cache_mask: torch.Tensor  # [Total_Tokens] - Boolean mask: which tokens to cache
    # Example: For sliding_window=3 and sequences [5, 7, 2]:
    #          [0,0,1,1,1, 0,0,0,0,1,1,1, 1,1]
    #          Only last 3 tokens per sequence are cached (sliding window)
    
    cached_elements: torch.Tensor  # [Batch_Size] - How many tokens cached per sequence
    # Example: [3, 3, 2] (3 tokens cached from seq1, 3 from seq2, 2 from seq3)
    
    cache_positions: torch.Tensor  # [Num_Cached_Tokens] - Where in cache buffer to write
    # Example: [2,0,1, 5,3,4, 6,7]
    # These are indices into the flattened cache: batch_idx * window + (pos % window)
    
    
    # Attention Configuration 
    # 
    prefill: bool  # True if processing prompt, False if generating tokens
    
    mask: AttentionBias  # Attention mask (causal + sliding window)
    # Different mask types:
    # - Prefill (first chunk):  BlockDiagonalCausalMask
    # - Prefill (later chunks): BlockDiagonalMask  
    # - Generation:             BlockDiagonalCausalWithOffsetPaddedKeysMask
    
    seqlens: List[int]  # Length of each sequence in the batch
    # Example: [5, 7, 2] means 3 sequences with 5, 7, and 2 tokens

def interleave_list(l1: List[torch.Tensor], l2: List[torch.Tensor]):
    """
    Interleave two lists element by element.
    
    Purpose: Merge cached KV with new KV by alternating elements.
    
    Example:
        l1 = [cached_k1, cached_k2, cached_k3]  ← Old cached keys
        l2 = [new_k1, new_k2, new_k3]           ← New keys
        
        Result: [cached_k1, new_k1, cached_k2, new_k2, cached_k3, new_k3]
        
    This creates the full key/value sequence for attention:
        [past tokens..., new tokens]
    """
    assert len(l1) == len(l2)
    return [v for pair in zip(l1, l2) for v in pair]


def unrotate(cache: torch.Tensor, seqlen: int) -> torch.Tensor:
    """
    "Unrotate" a circular buffer cache to restore chronological order.
    
    The cache is a circular buffer that wraps around. This function
    rotates it back so tokens are in the correct temporal order.
    
    Args:
        cache: Cached tensor [Sliding_Window_Size, Num_Heads, Head_Dim]
        seqlen: Total number of tokens seen so far (including overwritten ones)
    
    Returns:
        Cache rotated to chronological order
    """
    assert cache.ndim == 3  # (Sliding_Window_Size, Num_Heads, Head_Dim)
    
    # Calculate where the next write will happen (this is right after the most recent write)
    position = seqlen % cache.shape[0]
    
    if seqlen < cache.shape[0]:
        # Cache isn't full yet - no rotation has happened
        # Just return the first `seqlen` elements (valid data)
        # Example: seqlen=2, cache_size=4 → return cache[0:2]
        return cache[:seqlen]
    elif position == 0:
        # Cache is exactly full or perfectly aligned
        # No rotation needed - data is already in order
        return cache
    else:
        # Cache has wrapped around - need to unrotate
        # Split at the next write position (which follows the most recent write)
        # Everything after position is the older data
        # Everything before position is newer data (including wrapped)
        # 
        # Example: seqlen=6, window=4, position=2
        #   Pos 0: token 4 (older)
        #   Pos 1: token 5 (newer) 
        #   Pos 2: next write ← split here
        #   Pos 3: token 3 (oldest)
        #
        # Current cache:
        #   Pos 0: token 4
        #   Pos 1: token 5 (most recent)
        #   Pos 2: token 2 (oldest still in cache)
        #   Pos 3: token 3
        #
        # Chronological order: [2,3,4,5]
        # position = 2 (next write)
        # cache[2:] = [token 2, token 3]  ← oldest data
        # cache[:2] = [token 4, token 5]  ← newest data
        # concat: [token 2, token 3, token 4, token 5] ✓
        
        return torch.cat([cache[position:], cache[:position]], dim=0)
    
class CacheView:
    """
    View into the KV cache for a specific layer.
    
    This provides methods to:
    - Update the cache with new key/value pairs
    - Retrieve cached keys/values interleaved with new ones
    - Access cache metadata
    """
    
    def __init__(self, cache_k: torch.Tensor, cache_v: torch.Tensor, 
                 metadata: RotatingCacheInputMetadata, kv_seqlens: torch.Tensor):
        """
        Args:
            cache_k: Key cache [Batch_Size, Sliding_Window, N_Heads_KV, Head_Dim]
            cache_v: Value cache [Batch_Size, Sliding_Window, N_Heads_KV, Head_Dim]
            metadata: Metadata about what/where to cache
            kv_seqlens: How many tokens cached per sequence [Batch_Size]
        """
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.kv_seqlens = kv_seqlens
        self.metadata = metadata

    def update(self, xk: torch.Tensor, xv: torch.Tensor):
        """
        Update the cache with new key/value pairs.
        
        Only updates positions indicated by metadata.to_cache_mask
        (the last `sliding_window` tokens of each sequence).
        
        Args:
            xk: New keys [Total_Tokens, N_Heads_KV, Head_Dim]
            xv: New values [Total_Tokens, N_Heads_KV, Head_Dim]
        """
        n_kv_heads, head_dim = self.cache_k.shape[-2:]
        
        
        # Flatten cache for easy indexing                        
        # 
        # Original: [Batch, Window, N_Heads, Head_Dim]
        # Flattened: [Batch * Window, N_Heads, Head_Dim]
        # This allows us to use simple 1D indices for cache_positions
        flat_cache_k = self.cache_k.view(-1, n_kv_heads, head_dim)
        flat_cache_v = self.cache_v.view(-1, n_kv_heads, head_dim)
        
        
        # Copy selected tokens to cache                          
        # 
        # index_copy_(dim, index, source):
        #   - Copies from source to self[index] along dim
        #   - Here: copies xk[to_cache_mask] to cache[cache_positions]
        # 
        # Only tokens where to_cache_mask=True are copied
        # They go to positions specified by cache_positions
        flat_cache_k.index_copy_(0, self.metadata.cache_positions, xk[self.metadata.to_cache_mask])
        flat_cache_v.index_copy_(0, self.metadata.cache_positions, xv[self.metadata.to_cache_mask])

    def interleave_kv(self, xk: torch.Tensor, xv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine cached KV with new KV.
        
        Returns: (all_keys, all_values)
        
        Args:
            xk: New keys [Total_Tokens, N_Heads_KV, Head_Dim]
            xv: New values [Total_Tokens, N_Heads_KV, Head_Dim]
        
        Returns:
            (keys, values) with cached data prepended
        
        
        ═══════════════════════════════════════════════════════════════
                        INTERLEAVE VISUALIZATION
        ═══════════════════════════════════════════════════════════════
        
        Purpose: Concatenate cached tokens with new tokens.
        
        Example: 2 sequences in batch
        
        Sequence 0:
            Cached: [K₀, K₁, K₂] (3 tokens from previous step)
            New:    [K₃, K₄]     (2 new tokens this step)
            Result: [K₀, K₁, K₂, K₃, K₄]  ← Full sequence 
        
        Sequence 1:
            Cached: [K₀, K₁]     (2 tokens from previous step)
            New:    [K₂]         (1 new token this step)
            Result: [K₀, K₁, K₂]  ← Full sequence 
        
        Interleaved output (concatenated batches):
            [K₀,K₁,K₂,K₃,K₄, K₀,K₁,K₂]
             └──Sequence 0─┘ └─Seq 1─┘
        
        ═══════════════════════════════════════════════════════════════
        """
        assert xk.ndim == xv.ndim == 3  # (Total_Tokens, N_Heads, Head_Dim)
        assert xk.shape == xv.shape

        if all([s == 0 for s in self.metadata.seqlens]):
            # No new tokens - shouldn't happen but handle gracefully
            return xk, xv

        
        # Split concatenated tensors back into per-sequence      
        # 
        # Input: All sequences concatenated [Seq1+Seq2+Seq3, H, D]
        # Output: List of individual sequences [(Seq1,H,D), (Seq2,H,D), ...]
        xk = torch.split(xk, self.metadata.seqlens)
        xv = torch.split(xv, self.metadata.seqlens)
        assert len(xk) == len(self.kv_seqlens), \
            f"Batch size is {len(self.kv_seqlens)}, got {len(xk)}"

        
        # Unrotate cached elements to chronological order        
        # 
        # For each sequence, unrotate its cache so tokens are in order
        # kv_seqlens[i] tells us how many tokens have been cached for sequence i
        cache_k = [unrotate(t, s) for t, s in zip(self.cache_k, self.kv_seqlens)]
        cache_v = [unrotate(t, s) for t, s in zip(self.cache_v, self.kv_seqlens)]

        
        # Interleave: [cached, new, cached, new, ...]
        # 
        # For each sequence: prepend its cached KV before new KV
        # Result: [cache_k[0], xk[0], cache_k[1], xk[1], ...]
        interleaved_k = interleave_list(cache_k, xk)
        interleaved_v = interleave_list(cache_v, xv)

        
        # Concatenate all sequences
        # 
        return torch.cat(interleaved_k, dim=0), torch.cat(interleaved_v, dim=0)

    @property
    def sliding_window(self):
        """Size of the sliding attention window."""
        return self.cache_k.shape[1]

    @property
    def key(self) -> torch.Tensor:
        """Get key cache for current batch (trim to actual batch size)."""
        return self.cache_k[:len(self.kv_seqlens)]

    @property
    def value(self) -> torch.Tensor:
        """Get value cache for current batch (trim to actual batch size)."""
        return self.cache_v[:len(self.kv_seqlens)]

    @property
    def prefill(self):
        """Whether this is prefill phase (True) or generation phase (False)."""
        return self.metadata.prefill

    @property
    def mask(self):
        """Attention mask for this step."""
        return self.metadata.mask
    
class RotatingBufferCache:
    """
    Rotating buffer cache for efficient KV caching with sliding window attention.
    
    This implements a fixed-size circular buffer that:
    - Stores Keys and Values for multiple layers
    - Handles variable-length sequences in batches
    - Automatically overwrites oldest entries when full
    - Supports sliding window attention (only recent tokens matter)
    
    ═══════════════════════════════════════════════════════════════════
                        CACHE STRUCTURE
    ═══════════════════════════════════════════════════════════════════
    
    Shape: [n_layers, max_batch_size, sliding_window, n_kv_heads, head_dim]
            └──┬───┘ └──────┬──────┘ └──────┬──────┘ └────┬────┘   └──┬─┘
               │            │               │             │           │
          Each layer    Max sequences   Window size    KV heads   Dimension
          has cache     in batch        (e.g., 4096)   per layer
    
    
    Memory comparison (for reference):
    
    Without cache (recompute everything):
        Memory: O(batch × seq_len × hidden_dim)
        Compute: O(seq_len²) per token  ← Very slow!
    
    With full cache (no sliding window):
        Memory: O(batch × max_seq_len × hidden_dim)  ← Grows unbounded!
        Compute: O(seq_len) per token  ← Fast
    
    With rotating cache (sliding window):
        Memory: O(batch × window × hidden_dim)  ← Fixed size!
        Compute: O(window) per token  ← Fast & efficient
    
    ═══════════════════════════════════════════════════════════════════
    """
    
    def __init__(self, n_layers: int, max_batch_size: int, sliding_window: int, 
                 n_kv_heads: int, head_dim: int):
        """
        Initialize the rotating buffer cache.
        
        Args:
            n_layers: Number of transformer layers (e.g., 32)
            max_batch_size: Maximum number of sequences in a batch (e.g., 8)
            sliding_window: Size of attention window (e.g., 4096 for Mistral)
            n_kv_heads: Number of key/value heads (e.g., 8 for GQA)
            head_dim: Dimension of each head (e.g., 128)
        """
        self.sliding_window = sliding_window
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        
        # Allocate cache tensors (uninitialized for speed)       
        # 
        # These will be filled as we process tokens
        # Shape: [Layers, Batch, Window, Heads, Dim]
        self.cache_k = torch.empty((
            n_layers,
            max_batch_size,
            sliding_window,
            n_kv_heads,
            head_dim
        ))
        self.cache_v = torch.empty((
            n_layers,
            max_batch_size,
            sliding_window,
            n_kv_heads,
            head_dim
        ))
        
        # Tracks how many tokens have been cached for each sequence
        # None initially, initialized when first used
        self.kv_seqlens = None

    def get_view(self, layer_id: int, metadata: RotatingCacheInputMetadata) -> CacheView:
        """
        Get a view into the cache for a specific layer.
        
        Args:
            layer_id: Which transformer layer (0 to n_layers-1)
            metadata: Metadata for this forward pass
        
        Returns:
            CacheView object for this layer
        
        This allows each layer to independently access its slice of the cache.
        """
        return CacheView(self.cache_k[layer_id], self.cache_v[layer_id], 
                        metadata, self.kv_seqlens)

    def reset(self):
        """
        Reset the cache (typically between different conversations).
        
        This doesn't deallocate memory, just marks all sequences as empty.
        """
        self.kv_seqlens = None

    def init_kvseqlens(self, batch_size: int):
        """
        Initialize sequence length tracker for a new batch.
        
        Args:
            batch_size: Number of sequences in the batch
        
        Creates a tensor of zeros: [0, 0, 0, ...] (one per sequence)
        This tracks total tokens seen (including overwritten ones).
        """
        self.kv_seqlens = torch.zeros((batch_size,), device=self.device, dtype=torch.long)

    @property
    def device(self):
        """Get the device where cache is stored (CPU/CUDA)."""
        return self.cache_k.device

    def to(self, device: torch.device, dtype: torch.dtype):
        """
        Move cache to a different device/dtype.
        
        Args:
            device: Target device (e.g., 'cuda:0')
            dtype: Target dtype (e.g., torch.bfloat16)
        
        Returns:
            self (for method chaining)
        """
        self.cache_k = self.cache_k.to(device=device, dtype=dtype)
        self.cache_v = self.cache_v.to(device=device, dtype=dtype)
        return self

    def update_seqlens(self, seqlens: List[int]):
        """
        Update the sequence length tracker after processing tokens.
        
        Args:
            seqlens: Number of tokens processed in this step for each sequence
        
        Example:
            Before: kv_seqlens = [10, 15, 5]  (tokens seen so far)
            seqlens = [2, 3, 1]                (new tokens this step)
            After:  kv_seqlens = [12, 18, 6]  (updated totals)
        """
        self.kv_seqlens += torch.tensor(seqlens, device=self.device, dtype=torch.long)

    def get_input_metadata(self, seqlens: List[int]) -> RotatingCacheInputMetadata:
        """
        Generate metadata for the current forward pass.
        
        It computes:
        1. Which tokens to cache (last sliding_window tokens)
        2. Where to put them in the circular buffer
        3. Absolute positions for RoPE
        4. Appropriate attention mask
        
        Args:
            seqlens: Length of each sequence in this batch [seq1_len, seq2_len, ...]
        
        Returns:
            RotatingCacheInputMetadata with all necessary information
        """
        
        
        # Initialize kv_seqlens if first use                     
        # 
        if self.kv_seqlens is None:
            self.init_kvseqlens(len(seqlens))
        
        assert len(seqlens) == len(self.kv_seqlens), \
            f"Batch size is {len(self.kv_seqlens)}, got {len(seqlens)}, " \
            f"did you forget to reset cache?"
        
        # Get current position for each sequence
        seqpos = self.kv_seqlens.tolist()
        
        assert len(seqlens) > 0, seqlens

        
        # STEP 1: Compute to_cache_mask                          
        # 
        # Create a boolean mask indicating which tokens to cache
        # True if token is in the last `sliding_window` positions of sequence
        # 
        # For each sequence, mark last min(seqlen, sliding_window) tokens as True
        masks = [
            [x >= seqlen - self.sliding_window for x in range(seqlen)]
            for seqlen in seqlens
        ]
        # Example: seqlen=5, window=3
        #   range(5) = [0,1,2,3,4]
        #   x >= 5-3 → [False, False, True, True, True]

        # Flatten all masks into a single tensor
        to_cache_mask = torch.tensor(sum(masks, []), device=self.device, dtype=torch.bool)

        
        # STEP 2: Count cached elements per sequence             
        # 
        cached_elements = torch.tensor([sum(mask) for mask in masks], 
                                      device=self.device, dtype=torch.long)

        
        # STEP 3: Compute absolute positions for RoPE            
        # 
        # Each token gets its absolute position in the sequence
        # Starting from seqpos (where this sequence left off)
        positions = torch.cat([
            torch.arange(pos, pos + seqlen) 
            for pos, seqlen in zip(seqpos, seqlens)
        ]).to(device=self.device, dtype=torch.long)

        
        # STEP 4: Compute batch indices for each token           
        # 
        # Each token needs to know which batch element it belongs to
        # Example: seqlens=[3,2,4] → batch_idx=[0,0,0,1,1,2,2,2,2]
        batch_idx = torch.tensor(
            sum([[i]*seqlen for i, seqlen in enumerate(seqlens)], []),
            device=self.device, 
            dtype=torch.long
        )

        
        # STEP 5: Compute cache positions (circular buffer)      
        # 
        # Formula: (position % window_size) + (batch_idx * window_size)
        # 
        # This maps each token to a unique location in the flattened cache:
        # - position % window_size: position within the circular buffer
        # - batch_idx * window_size: offset for this batch element
        cache_positions = positions % self.sliding_window + batch_idx * self.sliding_window

        
        # STEP 6: Determine prefill vs generation mode           
        # 
        # first_prefill: Very first forward pass (all seqpos are 0)
        first_prefill = seqpos[0] == 0
        
        # subsequent_prefill: Processing multiple tokens (continuing prompt)
        subsequent_prefill = any(seqlen > 1 for seqlen in seqlens)

        
        # STEP 7: Create appropriate attention mask              
        # 
        if first_prefill:
            # ═══════════════════════════════════════════════════════
            # FIRST PREFILL: Initial prompt processing
            # ═══════════════════════════════════════════════════════
            # 
            # Requirements:
            # - Causal mask (can't attend to future tokens)
            # - Block diagonal (sequences don't attend to each other)
            # - Local attention (sliding window)
            # 
            # Visual (window=3, seqlens=[4,3]):
            # 
            #     Seq0 tokens:   0  1  2  3     Seq1 tokens:  0  1  2
            #                 ┌──────────────┐              ┌────────┐
            #     Seq0:    0  │ ●  .  .  .   │              │        │
            #              1  │ ●  ●  .  .   │              │        │
            #              2  │ ●  ●  ●  .   │ ← window=3   │        │
            #              3  │ .  ●  ●  ●   │              │        │
            #                 └──────────────┘              │        │
            #     Seq1:    0                 │              │ ●  .  .│
            #              1                 │              │ ●  ●  .│
            #              2                 │              │ ●  ●  ●│
            #                                               └────────┘
            #              ● = can attend    . = cannot attend
            
            assert all([pos == 0 for pos in seqpos]), \
                f"first_prefill but seqpos={seqpos}"
            
            mask = BlockDiagonalCausalMask.from_seqlens(seqlens)\
                                          .make_local_attention(self.sliding_window)
            
        elif subsequent_prefill:
            # ═══════════════════════════════════════════════════════
            # SUBSEQUENT PREFILL: Continuing prompt in chunks
            # ═══════════════════════════════════════════════════════
            # 
            # Requirements:
            # - Attend to cached tokens + new tokens
            # - Block diagonal (sequences separate)
            # - Local attention from bottom-right
            # 
            # Example: window=3, cached=2 tokens, new=3 tokens
            # 
            #         Cached  |  New tokens
            #          0  1   |  2  3  4
            #       ┌──────────────────────┐
            #    2  │  ●  ●   │  ●  .  .   │  ← Can see cache + self
            #    3  │  .  ●   │  ●  ●  .   │  ← Window of 3
            #    4  │  .  .   │  ●  ●  ●   │  ← Window of 3
            #       └──────────────────────┘
            #       
            #  New queries attend to: (cached + new) within window
            
            mask = BlockDiagonalMask.from_seqlens(
                q_seqlen=seqlens,
                kv_seqlen=[
                    s + cached_s.clamp(max=self.sliding_window).item() 
                    for (s, cached_s) in zip(seqlens, self.kv_seqlens)
                ]
            ).make_local_attention_from_bottomright(self.sliding_window)
            
        else:
            # ═══════════════════════════════════════════════════════
            # GENERATION MODE: One token at a time
            # ═══════════════════════════════════════════════════════
            # 
            # Requirements:
            # - Single query token
            # - Attends to all cached keys (up to window size)
            # - Padded keys (not all sequences have same cache length)
            # 
            # Example: window=4, cached=[3,2,4] tokens for 3 sequences
            # 
            #     Seq0:  Q → [K K K _]  (3 cached, 1 empty)
            #     Seq1:  Q → [K K _ _]  (2 cached, 2 empty)
            #     Seq2:  Q → [K K K K]  (4 cached, 0 empty)
            #     
            #     Each Q attends to its own cached Ks (within window)
            
            mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=seqlens,
                kv_padding=self.sliding_window,
                kv_seqlen=(self.kv_seqlens + cached_elements)\
                          .clamp(max=self.sliding_window)\
                          .tolist()
            )

        
        # Package everything into metadata object                
        # 
        return RotatingCacheInputMetadata(
            positions=positions,
            to_cache_mask=to_cache_mask,
            cached_elements=cached_elements,
            cache_positions=cache_positions[to_cache_mask],  # Only cached positions
            prefill=first_prefill or subsequent_prefill,
            mask=mask,
            seqlens=seqlens,
        )


"""
═══════════════════════════════════════════════════════════════════════════
                        KEY CONCEPTS SUMMARY
═══════════════════════════════════════════════════════════════════════════

1. WHY KV CACHE?
   ─────────────
   Without cache: O(n²) computation per token (n = sequence length)
   With cache:    O(n) computation per token
   
   For a 1000-token generation:
   - Without cache: ~500,000 redundant computations
   - With cache: ~1,000 new computations
   
   Trade-off: Memory for speed (worth it)


2. WHY SLIDING WINDOW?
   ───────────────────
   Problem: Full cache grows unbounded O(n²) memory
   Solution: Only keep recent tokens O(n) memory
   
   Mistral uses window=4096:
   - Can generate sequences of ANY length
   - Constant memory per token
   - Still captures local context well
   
   Empirical finding: Most attention is local anyway
   Tokens rarely attend to very distant tokens.


3. WHY ROTATING BUFFER?
   ────────────────────
   Alternative: Shift entire cache left when full
   
   [K₀, K₁, K₂, K₃] → [K₁, K₂, K₃, K₄]  ← Expensive copy!
   
   Rotating buffer: Just overwrite oldest position
   
   [K₀, K₁, K₂, K₃] → [K₄, K₁, K₂, K₃]  ← O(1) operation!
   
   Trade-off: Need to "unrotate" when reading (still cheaper)


4. ATTENTION MASK TYPES
   ────────────────────
   
   BlockDiagonalCausalMask:
   - For first prompt chunk
   - Each sequence separate + causal
   
   BlockDiagonalMask:
   - For subsequent prompt chunks  
   - Attend to cache + new tokens
   
   BlockDiagonalCausalWithOffsetPaddedKeysMask:
   - For generation (1 token at a time)
   - Handles variable cache lengths
   
   These are from xformers library (optimized attention)

5. MEMORY LAYOUT TRICKS
   ────────────────────
   
   Why flatten the cache?
   Original: [Batch, Window, Heads, Dim]
   Flattened: [Batch*Window, Heads, Dim]
   
   Benefit: Can use simple 1D indices
   cache[batch_idx * window + pos] = new_value
   
   Makes circular buffer indexing trivial!


6. BATCH PROCESSING
   ────────────────
   
   Challenge: Variable-length sequences in batch
   
   Seq 0: "Hello world"       (2 tokens)
   Seq 1: "How are you doing" (4 tokens)  
   Seq 2: "Hi"                (1 token)
   
   Solution: Concatenate + track boundaries
   Tokens: [H,W, H,a,y,d, H]
   seqlens: [2, 4, 1]
   
   Attention masks ensure sequences don't cross-attend!


7. PREFILL CHUNKING
   ─────────────────
   
   Long prompts can be processed in chunks:
   
   Prompt: 10,000 tokens, window: 4,096
   
   Chunk 1: tokens 0-4095    (first_prefill)
   Chunk 2: tokens 4096-8191 (subsequent_prefill)
   Chunk 3: tokens 8192-9999 (subsequent_prefill)
   
   Then: Generate tokens one by one (generation mode)
   
   This allows processing arbitrarily long prompts!


8. CACHE LIFECYCLE
   ───────────────
   
   1. Initialization:
      cache = RotatingBufferCache(...)
      cache.to(device, dtype)
   
   2. Prefill (first chunk):
      metadata = cache.get_input_metadata(seqlens=[100])
      # Process tokens with BlockDiagonalCausalMask
      cache.update_seqlens([100])
   
   3. Generation loop:
      for _ in range(num_tokens):
          metadata = cache.get_input_metadata(seqlens=[1])
          # Generate next token
          cache.update_seqlens([1])
   
   4. Next conversation:
      cache.reset()  # Clear for new conversation


9. OPTIMIZATION OPPORTUNITIES
   ──────────────────────────
   
   Current implementation:
   ✓ Fixed memory budget
   ✓ Circular buffer (O(1) updates)
   ✗ Rectangular allocation (wasteful for short sequences)
   
   Better alternatives:
   - PagedAttention: Allocate cache in pages (like virtual memory)
   - Variable-length allocation: Only allocate what's needed
   - Flash Attention: Fused kernel (no separate cache storage)
   
   Mistral likely uses more optimized implementations in production!


10. COMMON PITFALLS
    ──────────────
    
    Pitfall 1: Forgetting to reset cache between conversations
    Solution: Always call cache.reset() for new prompts
    
    Pitfall 2: Mismatched batch sizes
    Solution: Check len(seqlens) == len(kv_seqlens)
    
    Pitfall 3: Cache too small for prompt
    Solution: Process prompt in chunks (subsequent_prefill)
    
    Pitfall 4: Wrong attention mask type
    Solution: Let get_input_metadata() handle it automatically
    
    Pitfall 5: Not updating kv_seqlens
    Solution: Always call update_seqlens() after processing

═══════════════════════════════════════════════════════════════════════════
"""