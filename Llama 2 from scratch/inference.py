"""
This module handles loading a trained LLaMA model and generating text.
It implements autoregressive text generation with various sampling strategies.

Key Components:
1. Model loading from checkpoints
2. Tokenization (text ↔ token IDs)
3. Text generation with temperature and top-p sampling
4. Batch processing of multiple prompts

Autoregressive Generation Overview:
───────────────────────────────────

The model generates text one token at a time, using previously generated
tokens as context:

Initial: "The cat"
Step 1: "The cat" → predict "sat"
Step 2: "The cat sat" → predict "on"
Step 3: "The cat sat on" → predict "the"
Step 4: "The cat sat on the" → predict "mat"

At each step, the model:
1. Takes all previous tokens as input
2. Produces probability distribution over vocabulary
3. Samples next token from distribution
4. Adds token to sequence and repeats
"""

from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer


class LLaMA:
    """
    LLaMA Inference Wrapper
    =======================
    
    Wraps the transformer model with tokenization and generation logic.
    Provides a high-level interface for text completion.
    
    Components:
    ──────────
    • model: The transformer neural network
    • tokenizer: Converts text ↔ token IDs
    • args: Model configuration (dimensions, max length, etc.)
    
    Example Usage:
    ─────────────
    llama = LLaMA.build(checkpoint_dir, tokenizer_path, ...)
    tokens, text = llama.text_completion(["Once upon a time"])
    print(text[0])  # Generated story continuation
    """

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        """
        Initialize LLaMA with model, tokenizer, and configuration
        
        Args:
            model: Trained transformer model
            tokenizer: SentencePiece tokenizer for text ↔ tokens
            model_args: Model configuration (dim, layers, etc.)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(
        checkpoints_dir: str, 
        tokenizer_path: str, 
        load_model: bool, 
        max_seq_len: int, 
        max_batch_size: int, 
        device: str
    ):
        """
        Build and load a LLaMA model from checkpoints
        
        This static method handles:
        1. Loading model weights from checkpoint files
        2. Loading tokenizer
        3. Setting up model configuration
        4. Moving model to appropriate device (CPU/GPU)
        
        Args:
            checkpoints_dir: Directory containing model checkpoints (.pth files)
            tokenizer_path: Path to SentencePiece tokenizer model
            load_model: Whether to load weights (False for structure only)
            max_seq_len: Maximum sequence length model can handle
            max_batch_size: Maximum batch size for generation
            device: 'cuda' or 'cpu'
        
        Returns:
            LLaMA instance ready for inference
        
        Directory Structure Expected:
        ────────────────────────────
        checkpoints_dir/
        ├── consolidated.00.pth  (model weights)
        └── params.json          (model config)
        
        tokenizer_path: tokenizer.model
        
        Loading Process Flow:
        ────────────────────
        1. Load checkpoint file (.pth) → model weights
        2. Load params.json → model architecture config
        3. Load tokenizer.model → text ↔ token conversion
        4. Create model structure with config
        5. Load weights into model structure
        6. Move to device and set precision
        """
        
        prev_time = time.time()
        
        # ════════════════════════════════════════════════════════
        # STEP 1: Load Model Checkpoint (Weights)
        # ════════════════════════════════════════════════════════
        
        if load_model:
            # Find all .pth checkpoint files in directory
            # sorted() ensures consistent ordering if multiple checkpoints exist
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
            
            # Use the first checkpoint (for multi-file checkpoints, would need to merge)
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            
            # Load checkpoint weights from disk
            # map_location="cpu" loads to CPU first (safer, works regardless of device)
            # We'll move to target device later
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()
        
        # ════════════════════════════════════════════════════════
        # STEP 2: Load Model Configuration
        # ════════════════════════════════════════════════════════
        
        # Load model architecture parameters from JSON
        # This tells us: dimension, number of layers, heads, etc.
        #
        # Example params.json:
        # {
        #   "dim": 4096,
        #   "n_layers": 32,
        #   "n_heads": 32,
        #   "n_kv_heads": 8,
        #   ...
        # }
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        # Create ModelArgs with loaded params + inference settings
        # **params unpacks the JSON dict into constructor arguments
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,        # Override with inference setting
            max_batch_size=max_batch_size,  # Override with inference setting
            device=device,                   # Set target device
            **params                         # Unpack architecture params from JSON
        )

        # ════════════════════════════════════════════════════════
        # STEP 3: Load Tokenizer
        # ════════════════════════════════════════════════════════
        
        # Initialize SentencePiece tokenizer
        # This handles converting text to token IDs and back
        #
        # Tokenizer functions:
        # • encode("Hello") → [123, 456]  (text to IDs)
        # • decode([123, 456]) → "Hello"       (IDs to text)
        # • vocab_size() → 32000               (number of tokens)
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        
        # Set vocab size in model args based on tokenizer
        # This determines output layer size (model predicts over vocab)
        model_args.vocab_size = tokenizer.vocab_size()
        
        # ════════════════════════════════════════════════════════
        # STEP 4: Set Default Tensor Type (Precision)
        # ════════════════════════════════════════════════════════
        
        # Set default dtype for all future tensor allocations
        # This affects memory usage and computation speed
        #
        # float16 (Half): 2 bytes per number, ~2x faster, slightly less precise
        # bfloat16: 2 bytes, better range than float16, good for mixed precision
        # float32 (Full): 4 bytes per number, most precise, slower
        
        if device == "cuda":
            # Use float16 on CUDA for speed (GPUs handle it well)
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            # Use bfloat16 on CPU (better numerical properties for CPU)
            torch.set_default_tensor_type(torch.BFloat16Tensor)
        
        # ════════════════════════════════════════════════════════
        # STEP 5: Create Model Structure
        # ════════════════════════════════════════════════════════
        
        # Instantiate the transformer with configuration
        # This creates all layers, attention heads, etc.
        # Initially with random weights (will be overwritten if loading checkpoint)
        model = Transformer(model_args).to(device)

        # ════════════════════════════════════════════════════════
        # STEP 6: Load Trained Weights into Model
        # ════════════════════════════════════════════════════════
        
        if load_model:
            # Remove 'rope.freqs' from checkpoint
            # This key exists in checkpoint but not in our model structure
            # (we compute RoPE frequencies on the fly in __init__)
            del checkpoint['rope.freqs']
            
            # Load the state dict (model weights) into the model
            # strict=True: All keys must match exactly (catch errors)
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")
        
        # Return wrapped model ready for inference
        return LLaMA(model, tokenizer, model_args)

    def text_completion(self, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        """
        Generate text completions for multiple prompts
        
        This is the main inference method that generates text autoregressively.
        It processes multiple prompts in parallel (batched generation).
        
        Args:
            prompts: List of input text prompts to complete
                    Example: ["Once upon a time", "The meaning of life is"]
            
            temperature: Controls randomness of predictions (0.0 to 1.0+)
                        • 0.0: Deterministic (always pick most likely token)
                        • 0.6: Balanced (default, good for most tasks)
                        • 1.0: Sample from true distribution
                        • >1.0: More random/creative
            
            top_p: Nucleus sampling threshold (0.0 to 1.0)
                  Only sample from tokens whose cumulative probability >= top_p
                  • 0.9: Sample from top 90% probability mass (default)
                  • 1.0: Consider all tokens
                  • 0.5: Only most likely tokens (more focused)
            
            max_gen_len: Maximum number of NEW tokens to generate
                        None = generate up to max_seq_len
        
        Returns:
            Tuple of (token_ids, decoded_text) for each prompt
            • token_ids: List of generated token ID lists
            • decoded_text: List of generated text strings
        
        Temperature Visualization:
        ─────────────────────────
        Original logits: [2.0, 1.0, 0.5, 0.2]
        
        Temperature = 0.1 (very focused):
        After softmax: [0.90, 0.08, 0.015, 0.005]
        → Almost always picks first token
        
        Temperature = 1.0 (balanced):
        After softmax: [0.50, 0.25, 0.15, 0.10]
        → Balanced sampling
        
        Temperature = 2.0 (very random):
        After softmax: [0.35, 0.25, 0.22, 0.18]
        → More uniform, creative but less coherent
        
        Top-P (Nucleus) Sampling Visualization:
        ──────────────────────────────────────
        Token probabilities (sorted): [0.50, 0.25, 0.15, 0.05, 0.03, 0.02]
        
        top_p = 0.9:
        Cumulative:                   [0.50, 0.75, 0.90, 0.95, 0.98, 1.00]
                                                     ↑ cutoff here (reaches 0.9)
        Sample from: [0.50, 0.25, 0.15] (renormalized)
        Ignore: [0.05, 0.03, 0.02] (tail of distribution)
        
        Generation Process Example:
        ──────────────────────────
        Prompt: "The cat"
        
        Step 0: tokens = [15, 432, <pad>, <pad>, <pad>]  (prompt + padding)
        Position: cur_pos = 2
        
        Step 1: Process token at position 1 (432)
        → Model outputs logits for position 2
        → Sample: token_id = 891 ("sat")
        → tokens = [15, 432, 891, <pad>, <pad>]
        
        Step 2: Process token at position 2 (891)
        → Model outputs logits for position 3
        → Sample: token_id = 67 ("on")
        → tokens = [15, 432, 891, 67, <pad>]
        
        Continue until EOS token or max length...
        """
        
        # ════════════════════════════════════════════════════════
        # SETUP: Initialize Generation Parameters
        # ════════════════════════════════════════════════════════
        
        # Set maximum generation length if not specified
        # -1 because we need space for at least one prompt token
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        
        # ════════════════════════════════════════════════════════
        # STEP 1: Tokenize All Prompts
        # ════════════════════════════════════════════════════════
        
        # Convert each text prompt to token IDs
        # add_bos=True: Add beginning-of-sequence token
        # add_eos=False: Don't add end token (we'll generate until we hit it)
        #
        # Example:
        # "Hello world" → [1, 15496, 3186]
        #                  ↑ BOS token
        prompt_tokens = [
            self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) 
            for prompt in prompts
        ]
        
        # ════════════════════════════════════════════════════════
        # STEP 2: Validate Batch Size and Sequence Lengths
        # ════════════════════════════════════════════════════════
        
        # Check batch size doesn't exceed model's capacity
        # (KV cache was allocated for max_batch_size)
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, \
            f"batch size must be less than or equal to {self.args.max_batch_size}"
        
        # Find longest prompt in batch
        # All prompts will be padded to this length for parallel processing
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        
        # Verify longest prompt fits in model's context window
        assert max_prompt_len <= self.args.max_seq_len, \
            f"prompt length must be less than or equal to {self.args.max_seq_len}"
        
        # Calculate total sequence length (prompt + generation)
        # Can't exceed model's maximum sequence length
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # ════════════════════════════════════════════════════════
        # STEP 3: Create Padded Token Tensor
        # ════════════════════════════════════════════════════════
        
        # Create tensor filled with padding tokens
        # Shape: (batch_size, total_len)
        #
        # Example with 2 prompts:
        # Initial: [[<pad>, <pad>, <pad>, <pad>, <pad>],
        #           [<pad>, <pad>, <pad>, <pad>, <pad>]]
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        
        # Populate initial tokens with prompt tokens
        # Shorter prompts remain padded on the right
        #
        # Example:
        # Prompt 1: [1, 15, 432]     → [1, 15, 432, <pad>, <pad>]
        # Prompt 2: [1, 98]          → [1, 98, <pad>, <pad>, <pad>]
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        
        # ════════════════════════════════════════════════════════
        # STEP 4: Set Up Generation Tracking
        # ════════════════════════════════════════════════════════
        
        # Track which sequences have reached EOS (end-of-sequence)
        # Once a sequence hits EOS, we stop generating for it
        # Shape: (batch_size,)
        eos_reached = torch.tensor([False] * batch_size, device=device)
        
        # Create mask: True for prompt tokens, False for padding
        # This prevents us from overwriting prompt tokens during generation
        #
        # Example:
        # tokens:     [1, 15, 432, <pad>, <pad>]
        # mask:       [T,  T,   T,   F,     F  ]
        # 
        # We only generate into positions where mask is False
        prompt_tokens_mask = tokens != pad_id
        
        # ════════════════════════════════════════════════════════
        # STEP 5: Autoregressive Generation Loop
        # ════════════════════════════════════════════════════════
        
        # Generate tokens one position at a time
        # Start from position 1 (position 0 is BOS token)
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
        
        for cur_pos in cur_iterator:
            """
            At each position, we:
            1. Feed the previous token to the model
            2. Get probability distribution for next token
            3. Sample a token from distribution
            4. Place token in current position
            5. Check if we've hit EOS
            
            Position Evolution Example:
            ──────────────────────────
            cur_pos=1: Process tokens[:,0:1], predict position 1
            cur_pos=2: Process tokens[:,1:2], predict position 2
            cur_pos=3: Process tokens[:,2:3], predict position 3
            ...
            
            Why tokens[:, cur_pos-1:cur_pos]?
            • cur_pos-1:cur_pos gives us a single token (the previous one)
            • During inference, we only process one NEW token at a time
            • Previous tokens are already in KV cache
            """
            
            with torch.no_grad():  # Don't track gradients during inference
                # Forward pass: Get logits for next token prediction
                # Input: (batch_size, 1) - single previous token
                # Output: (batch_size, 1, vocab_size) - scores for next token
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
            
            # ════════════════════════════════════════════════════════
            # STEP 6: Sample Next Token
            # ════════════════════════════════════════════════════════
            
            if temperature > 0:
                # ═══ Stochastic Sampling (with temperature and top-p) ═══
                
                # Apply temperature scaling to logits
                # Higher temperature → flatter distribution (more random)
                # Lower temperature → sharper distribution (more focused)
                #
                # logits[:, -1] gets the last position's logits
                # Shape: (batch_size, vocab_size)
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                
                # Apply top-p (nucleus) sampling
                # Only sample from most probable tokens that sum to top_p
                # This filters out unlikely tokens for better quality
                next_token = self._sample_top_p(probs, top_p)
                
            else:
                # ═══ Greedy Sampling (deterministic) ═══
                
                # Always pick the most likely token
                # temperature=0 means no randomness
                # Good for factual tasks, not creative writing
                next_token = torch.argmax(logits[:, -1], dim=-1)

            # Reshape to match tokens tensor shape
            # From (batch_size,) to (batch_size,)
            next_token = next_token.reshape(-1)
            
            # ════════════════════════════════════════════════════════
            # STEP 7: Update Token Sequence
            # ════════════════════════════════════════════════════════
            
            # Only replace padding tokens, preserve prompt tokens
            # where() acts as: if prompt_token then keep_original else use_generated
            #
            # Example:
            # Position 2 (prompt):     mask[2]=True  → keep tokens[2] (prompt)
            # Position 3 (generation): mask[3]=False → use next_token (generated)
            next_token = torch.where(
                prompt_tokens_mask[:, cur_pos],  # condition: is this a prompt token?
                tokens[:, cur_pos],               # if yes: keep existing token
                next_token                        # if no: use newly generated token
            )
            tokens[:, cur_pos] = next_token
            
            # ════════════════════════════════════════════════════════
            # STEP 8: Check for End-of-Sequence
            # ════════════════════════════════════════════════════════
            
            # Mark sequences that have generated an EOS token
            # Only count EOS in generated positions (not prompt)
            #
            # Logic:
            # • ~prompt_tokens_mask[:, cur_pos]: This is a generated position
            # • next_token == eos_id: We generated an EOS token
            # • Both conditions must be true to mark EOS reached
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & \
                          (next_token == self.tokenizer.eos_id())
            
            # If all sequences have reached EOS, stop generation early
            # No point continuing if everything is done
            if all(eos_reached):
                break

        # ════════════════════════════════════════════════════════
        # STEP 9: Post-process Generated Sequences
        # ════════════════════════════════════════════════════════
        
        out_tokens = []
        out_text = []
        
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut sequence at EOS token if present
            # Don't include EOS in output
            #
            # Example:
            # [1, 15, 432, 891, 2, <pad>] → [1, 15, 432, 891]
            #                   ↑ EOS (id=2)
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            
            # Store token IDs
            out_tokens.append(current_prompt_tokens)
            
            # Decode tokens back to text
            # [1, 15, 432, 891] → "The cat sat"
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        
        return (out_tokens, out_text)
    
    def _sample_top_p(self, probs, p):
        """
        Top-P (Nucleus) Sampling
        ========================
        
        Instead of sampling from the entire vocabulary, only sample from the
        smallest set of tokens whose cumulative probability exceeds p.
        
        This filters out the "long tail" of unlikely tokens, leading to:
        • More coherent text (avoids random unlikely words)
        • Controllable randomness (p=0.9 is a good default)
        • Better than top-k because it adapts to the distribution shape
        
        Algorithm Walkthrough:
        ─────────────────────
        
        Given: probs = [0.4, 0.3, 0.15, 0.1, 0.03, 0.02] (already softmaxed)
               p = 0.9 (we want top 90% of probability mass)
        
        Step 1: Sort probabilities (descending)
        ────────────────────────────────────────
        probs_sort = [0.4, 0.3, 0.15, 0.1, 0.03, 0.02]
        probs_idx =  [0,   1,   2,    3,   4,    5   ]  (original indices)
        
        Step 2: Compute cumulative sum
        ───────────────────────────────
        probs_sum = [0.4, 0.7, 0.85, 0.95, 0.98, 1.0]
                                      ↑ exceeds 0.9 here
        
        Step 3: Create mask for tokens to exclude
        ──────────────────────────────────────────
        We want to exclude tokens where cumsum (minus current) > p
        
        probs_sum - probs_sort = [0.0, 0.4, 0.7, 0.85, 0.95, 0.98]
        
        mask = (probs_sum - probs_sort) > 0.9
             = [F,   F,   F,   F,    T,    T  ]
                                     ↑     ↑
                               exclude these
        
        Why subtract probs_sort?
        • We want to keep a token if adding it FIRST TIME crosses threshold
        • Subtracting shifts cumsum so we check "cumsum before adding this token"
        • This ensures we include the token that crosses threshold
        
        Visual Example:
        ──────────────
        Token    Prob   Cumsum   Cumsum-Prob   > 0.9?   Keep?
        ─────────────────────────────────────────────────────────
        tok_0    0.4    0.4      0.0           No       ✓ Keep
        tok_1    0.3    0.7      0.4           No       ✓ Keep
        tok_2    0.15   0.85     0.7           No       ✓ Keep
        tok_3    0.1    0.95     0.85          No       ✓ Keep  (crosses here!)
        tok_4    0.03   0.98     0.95          Yes      ✗ Drop
        tok_5    0.02   1.0      0.98          Yes      ✗ Drop
        
        Step 4: Zero out excluded tokens
        ─────────────────────────────────
        probs_sort = [0.4, 0.3, 0.15, 0.1, 0.0, 0.0]
        
        Step 5: Renormalize (sum back to 1.0)
        ──────────────────────────────────────
        sum = 0.4 + 0.3 + 0.15 + 0.1 = 0.95
        probs_sort = [0.42, 0.32, 0.16, 0.10, 0.0, 0.0]
                      ↑ each divided by 0.95
        
        Step 6: Sample from filtered distribution
        ──────────────────────────────────────────
        Sample index from [0, 1, 2, 3] with probs [0.42, 0.32, 0.16, 0.10]
        Let's say we sample index 1
        
        Step 7: Map back to original vocabulary index
        ──────────────────────────────────────────────
        probs_idx[1] = 1  (original position in vocabulary)
        Return token_id = 1
        
        Args:
            probs: Probability distribution (B, vocab_size) - already softmaxed
            p: Cumulative probability threshold (typically 0.9)
        
        Returns:
            next_token: Sampled token indices (B, 1)
        """
        
        # ════════════════════════════════════════════════════════
        # STEP 1: Sort Probabilities (Descending)
        # ════════════════════════════════════════════════════════
        
        # Sort probabilities from highest to lowest
        # probs_sort: sorted probability values
        # probs_idx: original indices (for mapping back later)
        #
        # Shape: (B, vocab_size) → (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        
        # ════════════════════════════════════════════════════════
        # STEP 2: Compute Cumulative Sum
        # ════════════════════════════════════════════════════════
        
        # Calculate running sum of probabilities
        # At position i: sum of all probabilities from 0 to i (inclusive)
        #
        # Example: [0.4, 0.3, 0.2, 0.1]
        # Cumsum:  [0.4, 0.7, 0.9, 1.0]
        #
        # Shape: (B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        
        # ════════════════════════════════════════════════════════
        # STEP 3: Create Exclusion Mask
        # ════════════════════════════════════════════════════════
        
        # Determine which tokens to exclude from sampling
        # 
        # Key insight: We want to exclude tokens where the cumulative sum
        # (BEFORE including this token) already exceeds p
        #
        # Subtracting probs_sort shifts cumsum by one position:
        # • probs_sum[i] = sum from 0 to i (includes token i)
        # • probs_sum[i] - probs_sort[i] = sum from 0 to i-1 (excludes token i)
        #
        # mask[i] = True means "cumsum already exceeded p before this token"
        #          → exclude this token
        #
        # Shape: (B, vocab_size)
        mask = probs_sum - probs_sort > p
        
        # ════════════════════════════════════════════════════════
        # STEP 4: Zero Out Excluded Tokens
        # ════════════════════════════════════════════════════════
        
        # Set probability to 0.0 for all tokens we want to exclude
        # These tokens won't be sampled since they have zero probability
        #
        # Example:
        # Before: [0.4, 0.3, 0.15, 0.1, 0.03, 0.02]
        # Mask:   [F,   F,   F,    F,   T,    T  ]
        # After:  [0.4, 0.3, 0.15, 0.1, 0.0,  0.0 ]
        probs_sort[mask] = 0.0
        
        # ════════════════════════════════════════════════════════
        # STEP 5: Renormalize Probabilities
        # ════════════════════════════════════════════════════════
        
        # After zeroing out tokens, probabilities no longer sum to 1.0
        # Divide by sum to renormalize the distribution
        #
        # Example:
        # Before renorm: [0.4, 0.3, 0.15, 0.1, 0.0, 0.0]  (sum = 0.95)
        # After renorm:  [0.42, 0.32, 0.16, 0.11, 0.0, 0.0]  (sum = 1.0)
        #
        # div_() is in-place division
        # keepdim=True preserves dimensions for broadcasting
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        
        # ════════════════════════════════════════════════════════
        # STEP 6: Sample from Filtered Distribution
        # ════════════════════════════════════════════════════════
        
        # Randomly sample one token per batch element
        # multinomial() samples from categorical distribution
        # Higher probability → more likely to be sampled
        #
        # This samples an INDEX in the sorted array (0 to vocab_size-1)
        # Shape: (B, vocab_size) → (B, 1)
        next_token = torch.multinomial(probs_sort, num_samples=1)
        
        # ════════════════════════════════════════════════════════
        # STEP 7: Map Back to Original Vocabulary Indices
        # ════════════════════════════════════════════════════════
        
        # next_token is an index in the SORTED array
        # We need to map it back to the ORIGINAL vocabulary index
        #
        # gather() uses next_token as indices to select from probs_idx
        #
        # Example:
        # next_token = [2]  (position in sorted array)
        # probs_idx = [450, 123, 789, 56, ...]
        # gather(probs_idx, index=2) → 789  (original vocab position)
        #
        # Shape: (B, 1)
        next_token = torch.gather(probs_idx, -1, next_token)
        
        return next_token
    

if __name__ == '__main__':
    torch.manual_seed(0)

    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # Few shot promt
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as human:
        Name: majid
        Decision: 
        """
    ]

    model = LLaMA.build(
        checkpoints_dir='llama-2-7b/',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device
    )

    out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=64))
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)