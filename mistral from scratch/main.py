from mistral.cache import RotatingBufferCache
import logging
import torch
import fire
from typing import List
from pathlib import Path

from mistral.model import Transformer
from mistral.tokenizer import Tokenizer


"""
═══════════════════════════════════════════════════════════════════════════
                    TEXT GENERATION PIPELINE
═══════════════════════════════════════════════════════════════════════════

This module implements the complete text generation workflow:

1. SAMPLING: How to pick the next token from probability distribution
2. GENERATION: The main loop that generates tokens one by one
3. INTERACTIVE: Chat-like interface for experimenting
4. DEMO: Batch generation example

The pipeline combines all components:
    Tokenizer → Model → Sampling → Decoding

═══════════════════════════════════════════════════════════════════════════
"""


def sample_top_p(probs: torch.Tensor, p: float):
    """
    Nucleus (top-p) sampling: Sample from the smallest set of tokens 
    whose cumulative probability exceeds p.
    
    This provides a dynamic vocabulary size based on the probability distribution.
    
    Args:
        probs: Probability distribution [Vocab_Size]
        p: Cumulative probability threshold (e.g., 0.9)
    
    Returns:
        Sampled token ID
    
    
    ═══════════════════════════════════════════════════════════════════
                        TOP-P SAMPLING EXPLAINED
    ═══════════════════════════════════════════════════════════════════
    
    Problem: How to sample from 32,000 vocabulary tokens?
    
    Option 1: Greedy (always pick highest probability)
        ✗ Repetitive, boring outputs
        ✗ No diversity
        Example: "The cat sat on the mat. The cat sat on the mat. The cat..."
    
    Option 2: Sample from ALL tokens
        ✗ Includes very unlikely tokens
        ✗ Incoherent outputs
        Example: "The xylophone banana quantum refrigerator..."
    
    Option 3: Top-K (fixed K most likely tokens)
        ✓ Some diversity
        ✗ Fixed K doesn't adapt to distribution
        Example: K=10, but sometimes only 3 tokens are reasonable
    
    Option 4: Top-P (Nucleus Sampling) ← We use this!
        ✓ Dynamic vocabulary size
        ✓ Adapts to confidence of model
        ✓ Good balance of coherence and diversity
    
    
    HOW IT WORKS:
    ─────────────
    
    Example: p=0.9 (sample from top 90% of probability mass)
    
    Original probabilities (sorted):
        Token "the":    0.35  ──┐
        Token "a":      0.25  ──┤
        Token "this":   0.20  ──┤  90% cumsum
        Token "that":   0.10  ──┘  (0.35+0.25+0.20+0.10 = 0.90)
        Token "some":   0.05      ← Excluded (cumsum > 0.9)
        Token "those":  0.03      ← Excluded
        Token "xylo":   0.02      ← Excluded
        ... (rest)
    
    Process:
    1. Sort probabilities descending
    2. Compute cumulative sum
    3. Keep tokens until cumsum > p (0.9)
    4. Zero out other tokens
    5. Renormalize remaining probabilities
    6. Sample from this smaller distribution
    
    Result: Sample from {the, a, this, that} only
    
    
    ADAPTIVE BEHAVIOR:
    ──────────────────
    
    High confidence situation (one clear answer):
        "The capital of France is ___"
        Probs: Paris: 0.95, Lyon: 0.03, London: 0.02
        Top-p=0.9 → Only sample "Paris" (cumsum=0.95 > 0.9 after 1 token)
    
    Low confidence situation (creative writing):
        "The dragon ___"
        Probs: flew: 0.15, roared: 0.12, slept: 0.10, ...
        Top-p=0.9 → Sample from ~10 tokens (more diversity)
    
    ═══════════════════════════════════════════════════════════════════
    """
    assert 0 <= p <= 1, f"p must be in [0,1], got {p}"

    # ┌────────────────────────────────────────────────────────┐
    # │ STEP 1: Sort probabilities in descending order         │
    # └────────────────────────────────────────────────────────┘
    # probs_sort: sorted probabilities (highest first)
    # probs_idx: original indices of sorted probabilities
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    
    # ┌────────────────────────────────────────────────────────┐
    # │ STEP 2: Compute cumulative sum                         │
    # └────────────────────────────────────────────────────────┘
    # Example: [0.4, 0.3, 0.2, 0.1] → [0.4, 0.7, 0.9, 1.0]
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    
    # ┌────────────────────────────────────────────────────────┐
    # │ STEP 3: Create mask for tokens to exclude              │
    # └────────────────────────────────────────────────────────┘
    # Mask tokens where: cumsum - current_prob > p
    # This means: "cumsum BEFORE adding this token exceeded p"
    # 
    # Example with p=0.8:
    #   cumsum: [0.4, 0.7, 0.9, 1.0]
    #   probs:  [0.4, 0.3, 0.2, 0.1]
    #   cumsum - probs: [0.0, 0.4, 0.7, 0.9]
    #   mask (>0.8): [False, False, False, True]
    #   This excludes the last token (0.1 probability)
    mask = probs_sum - probs_sort > p
    
    # ┌────────────────────────────────────────────────────────┐
    # │ STEP 4: Zero out excluded tokens                       │
    # └────────────────────────────────────────────────────────┘
    probs_sort[mask] = 0.0
    
    # ┌────────────────────────────────────────────────────────┐
    # │ STEP 5: Renormalize (probabilities sum to 1)           │
    # └────────────────────────────────────────────────────────┘
    # After zeroing some tokens, remaining ones don't sum to 1
    # Example: [0.4, 0.3, 0.2, 0.0] → [0.44, 0.33, 0.22, 0.0]
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    
    # ┌────────────────────────────────────────────────────────┐
    # │ STEP 6: Sample from filtered distribution              │
    # └────────────────────────────────────────────────────────┘
    # multinomial samples indices based on probabilities
    # num_samples=1: sample one token
    next_token = torch.multinomial(probs_sort, num_samples=1)
    
    # ┌────────────────────────────────────────────────────────┐
    # │ STEP 7: Map back to original token IDs                 │
    # └────────────────────────────────────────────────────────┘
    # next_token is an index into probs_sort
    # Use gather to get the corresponding original token ID
    return torch.gather(probs_idx, -1, next_token)


def sample(logits: torch.Tensor, temperature: float, top_p: float):
    """
    Sample next token from logits using temperature and top-p sampling.
    
    Args:
        logits: Model output logits [Vocab_Size]
        temperature: Controls randomness (0 = greedy, >1 = more random)
        top_p: Nucleus sampling threshold (e.g., 0.9)
    
    Returns:
        Sampled token ID
    
    
    ═══════════════════════════════════════════════════════════════════
                        TEMPERATURE SCALING
    ═══════════════════════════════════════════════════════════════════
    
    Temperature controls the "sharpness" of probability distribution.
    
    Original logits: [2.0, 1.0, 0.5, 0.1]
    
    
    Temperature = 1.0 (default):
        Probs: softmax([2.0, 1.0, 0.5, 0.1])
             = [0.48, 0.18, 0.11, 0.07, ...]
        → Balanced distribution
    
    
    Temperature = 0.1 (low, more deterministic):
        Probs: softmax([2.0, 1.0, 0.5, 0.1] / 0.1)
             = softmax([20, 10, 5, 1])
             = [0.99, 0.01, 0.00, 0.00, ...]
        → Very peaked, almost always picks highest
        → More repetitive, focused outputs
        Use for: factual questions, code generation
    
    
    Temperature = 2.0 (high, more random):
        Probs: softmax([2.0, 1.0, 0.5, 0.1] / 2.0)
             = softmax([1.0, 0.5, 0.25, 0.05])
             = [0.32, 0.19, 0.15, 0.12, ...]
        → Flatter distribution
        → More diversity, creativity
        Use for: creative writing, brainstorming
    
    
    Temperature = 0 (greedy):
        → Always pick argmax (highest logit)
        → Completely deterministic
        → No randomness at all
        Use for: when you want consistent outputs
    
    ═══════════════════════════════════════════════════════════════════
    """
    if temperature > 0:
        # ┌────────────────────────────────────────────────────────┐
        # │ Stochastic sampling with temperature                   │
        # └────────────────────────────────────────────────────────┘
        # Scale logits by temperature, then convert to probabilities
        probs = torch.softmax(logits / temperature, dim=-1)
        
        # Sample using nucleus (top-p) sampling
        next_token = sample_top_p(probs, top_p)
    else:
        # ┌────────────────────────────────────────────────────────┐
        # │ Greedy sampling (temperature = 0)                      │
        # └────────────────────────────────────────────────────────┘
        # Always pick the token with highest logit (deterministic)
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

    return next_token.reshape(-1)


@torch.inference_mode()
def generate(
    prompts: List[str], 
    model: Transformer, 
    tokenizer: Tokenizer, 
    *, 
    max_tokens: int,  
    temperature: float, 
    chunk_size: int = None
):
    """
    Generate text completions for multiple prompts in parallel.
    
    This is the main generation function that orchestrates:
    1. Tokenization
    2. Prompt processing (prefill)
    3. Token-by-token generation
    4. Decoding back to text
    
    Args:
        prompts: List of text prompts to complete
        model: Transformer model
        tokenizer: Tokenizer for encoding/decoding
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0=greedy, >0=stochastic)
        chunk_size: Process prompt in chunks (for long prompts)
    
    Returns:
        generated_words: List of completed texts
        logprobs: Log probabilities for each token (for analysis)
    
    
    ═══════════════════════════════════════════════════════════════════
                    GENERATION PIPELINE OVERVIEW
    ═══════════════════════════════════════════════════════════════════
    
    Example: Generate completions for 2 prompts
    
    Input prompts:
        1. "The capital of France is"
        2. "Hello"
    
    
    PHASE 1: PREFILL (Process prompts in parallel)
    ───────────────────────────────────────────────────────────────────
    
    Tokenize:
        Prompt 1: [1, 450, 7483, 310, 3444, 338]  (6 tokens)
        Prompt 2: [1, 15043]                      (2 tokens)
    
    Process in chunks (if needed):
        Chunk 1: [1, 450, 7483, 310, 3444, 338, 1,     15043]
                  └───────   Prompt 1   ──────┘ └ Prompt 2 ┘
        
        Forward pass → Get logits for each position
        Cache K/V for all tokens
    
    Output: Logits for next token after each prompt
        Prompt 1 logits: [32000 values]
        Prompt 2 logits: [32000 values]
    
    
    PHASE 2: GENERATION (Generate tokens one at a time)
    ───────────────────────────────────────────────────────────────────
    
    Loop max_tokens times:
    
    Step 1: Sample next tokens
        Prompt 1: Sample from logits → token "Paris" (ID: 3681)
        Prompt 2: Sample from logits → token " world" (ID: 3186)
    
    Step 2: Forward pass with new tokens
        Input: [3681, 3186]
        seqlens: [1, 1]  (one new token per sequence)
        
        Use cached K/V from prompt + previous tokens
        Only compute attention for new tokens!
        
        Output: New logits for both sequences
    
    Step 3: Repeat until max_tokens reached
    
    
    Final outputs:
        Prompt 1: "The capital of France is Paris, a city..."
        Prompt 2: "Hello world! How are you today?..."
    
    ═══════════════════════════════════════════════════════════════════
    """
    # Set model to evaluation mode (disable dropout, etc.)
    model = model.eval()
    batch_size = len(prompts)
    vocabulary_size = model.args.vocab_size

    # ┌────────────────────────────────────────────────────────┐
    # │ STEP 1: Tokenize all prompts                           │
    # └────────────────────────────────────────────────────────┘
    # Convert text to token IDs, with BOS token prepended
    encoded_prompts = [tokenizer.encode(prompt, bos=True) for prompt in prompts]
    
    # Track length of each prompt (for batching)
    prompts_sequence_lengths = [len(x) for x in encoded_prompts]

    # ┌────────────────────────────────────────────────────────┐
    # │ STEP 2: Initialize KV cache                            │
    # └────────────────────────────────────────────────────────┘
    # Calculate required cache size:
    # Need to store: longest_prompt + all_generated_tokens
    cache_window = max(prompts_sequence_lengths) + max_tokens

    # Limit cache to sliding window size (if specified)
    if model.args.sliding_window is not None and cache_window > model.args.sliding_window:
        cache_window = model.args.sliding_window
    
    # Create rotating buffer cache
    # This will store K/V for all layers
    cache = RotatingBufferCache(
        model.n_local_layers,      # Number of layers in this rank
        model.args.max_batch_size,  # Max batch size
        cache_window,               # Cache size (tokens to remember)
        model.args.n_kv_heads,      # Number of KV heads (for GQA)
        model.args.head_dim,        # Dimension per head
    )

    # Move cache to same device/dtype as model
    cache.to(device=model.device, dtype=model.dtype)
    cache.reset()  # Clear any previous data
    
    # ┌────────────────────────────────────────────────────────┐
    # │ STEP 3: Initialize bookkeeping                         │
    # └────────────────────────────────────────────────────────┘
    # Track log probabilities for each generated token (for analysis)
    logprobs = [[] for _ in range(batch_size)]
    last_token_prelogits = None

    # ┌────────────────────────────────────────────────────────┐
    # │ STEP 4: Determine chunk size for prompt processing     │
    # └────────────────────────────────────────────────────────┘
    # For very long prompts, process in chunks to save memory
    max_prompt_len = max(prompts_sequence_lengths)
    if chunk_size is None:
        # Default: process entire prompt at once
        chunk_size = max_prompt_len

    # ┌────────────────────────────────────────────────────────┐
    # │ PHASE 1: PREFILL - Process prompts chunk by chunk      │
    # └────────────────────────────────────────────────────────┘
    """
    PREFILL PHASE EXPLANATION:
    ─────────────────────────
    
    Why chunk? Long prompts (1000+ tokens) can exhaust memory.
    Solution: Process in chunks, cache K/V incrementally.
    
    Example: Prompt with 10,000 tokens, chunk_size=4096
        Chunk 1: tokens 0-4095     → Process, cache K/V
        Chunk 2: tokens 4096-8191  → Process, cache K/V
        Chunk 3: tokens 8192-9999  → Process, cache K/V
    
    Each chunk attends to: previous cached tokens + current chunk
    (thanks to sliding window + cache interleaving)
    """
    for s in range(0, max_prompt_len, chunk_size):
        # Extract current chunk from each prompt
        # Example: s=0, chunk_size=5
        #   Prompt 1: [1,2,3,4,5,6,7,8] → [1,2,3,4,5]
        #   Prompt 2: [1,2,3]           → [1,2,3]
        prompt_chunks = [p[s:s+chunk_size] for p in encoded_prompts]
        
        # Ensure all chunks are non-empty
        assert all(len(p) > 0 for p in prompt_chunks)
        
        # ┌────────────────────────────────────────────────────┐
        # │ Forward pass through model                         │
        # └────────────────────────────────────────────────────┘
        # Concatenate all chunks into single tensor
        # Example: [[1,2,3], [4,5]] → [1,2,3,4,5]
        prelogits = model.forward(
            torch.tensor(sum(prompt_chunks, []), device=model.device, dtype=torch.long),
            seqlens=[len(p) for p in prompt_chunks],
            cache=cache
        )
        # prelogits shape: [Total_Tokens_In_Chunk, Vocab_Size]
        
        # Convert logits to log probabilities
        # This is numerically more stable than computing probabilities
        logits = torch.log_softmax(prelogits, dim=-1)

        # ┌────────────────────────────────────────────────────┐
        # │ Track log probabilities (for analysis/debugging)   │
        # └────────────────────────────────────────────────────┘
        if last_token_prelogits is not None:
            # Not first chunk: we can compute logprob for first token of this chunk
            # (it was predicted by last token of previous chunk)
            last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
            for i_seq in range(batch_size):
                # Get logprob of first token in current chunk
                # last_token_logits[i_seq, TOKEN_ID] = log P(TOKEN_ID | previous context)
                logprobs[i_seq].append(
                    last_token_logits[i_seq, prompt_chunks[i_seq][0]].item()
                )

        # For remaining tokens in chunk, compute their logprobs
        offset = 0
        for i_seq, sequence in enumerate(prompt_chunks):
            # For each token (except last), get logprob of next token
            # Example: sequence = [1,2,3,4]
            #   logits[offset+0] predicts token 2
            #   logits[offset+1] predicts token 3
            #   logits[offset+2] predicts token 4
            logprobs[i_seq].extend([
                logits[offset + i, sequence[i + 1]].item() 
                for i in range(len(sequence) - 1)
            ])
            offset += len(sequence)

        # ┌────────────────────────────────────────────────────┐
        # │ Save logits for last token of each sequence        │
        # └────────────────────────────────────────────────────┘
        # These will be used to predict first token of next chunk
        # OR to start generation phase
        # 
        # cumsum gives us end position of each sequence
        # Example: lens=[3,2] → cumsum=[3,5] → positions [2,4] (0-indexed: [2,4])
        last_token_positions = torch.tensor(
            [len(p) for p in prompt_chunks], 
            device=prelogits.device
        ).cumsum(dim=0) - 1
        
        last_token_prelogits = prelogits.index_select(0, last_token_positions)
        assert last_token_prelogits.shape == (batch_size, vocabulary_size)

    # ┌────────────────────────────────────────────────────────┐
    # │ PHASE 2: GENERATION - Generate tokens one by one       │
    # └────────────────────────────────────────────────────────┘
    """
    GENERATION PHASE EXPLANATION:
    ────────────────────────────
    
    Now we've processed all prompts and cached their K/V.
    Generate max_tokens new tokens, one at a time.
    
    Why one at a time? Can't know token N+2 until we generate N+1!
    This is autoregressive generation.
    
    Example loop (max_tokens=3):
        Step 1: last_token_prelogits → Sample → "Hello"
                Cache "Hello" K/V
                Forward → new logits
        
        Step 2: new logits → Sample → " world"
                Cache " world" K/V
                Forward → new logits
        
        Step 3: new logits → Sample → "!"
                Cache "!" K/V
                Forward → new logits
        
        Done! Generated: "Hello world!"
    """
    generated_tokens = []
    assert last_token_prelogits is not None
    
    for i_token in range(max_tokens):
        # ┌────────────────────────────────────────────────────┐
        # │ Sample next token for each sequence                │
        # └────────────────────────────────────────────────────┘
        # last_token_prelogits: [Batch_Size, Vocab_Size]
        # Returns: [Batch_Size] token IDs
        next_token = sample(last_token_prelogits, temperature=temperature, top_p=0.8)

        # ┌────────────────────────────────────────────────────┐
        # │ Record log probability of sampled tokens           │
        # └────────────────────────────────────────────────────┘
        last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
        for i in range(batch_size):
            # Get logprob of the token we actually sampled
            logprobs[i].append(last_token_logits[i, next_token[i]].item())

        # ┌────────────────────────────────────────────────────┐
        # │ Save generated token                               │
        # └────────────────────────────────────────────────────┘
        # Shape: [Batch_Size, 1] (add dimension for concatenation later)
        generated_tokens.append(next_token[:, None])
        
        # ┌────────────────────────────────────────────────────┐
        # │ Forward pass with new tokens                       │
        # └────────────────────────────────────────────────────┘
        # Process one token per sequence
        # Cache automatically handles:
        #   1. Storing new K/V
        #   2. Retrieving cached K/V
        #   3. Rotating buffer if window is full
        last_token_prelogits = model.forward(
            next_token, 
            seqlens=[1] * len(prompts),  # One token per sequence
            cache=cache
        )
        assert last_token_prelogits.shape == (batch_size, vocabulary_size)

    # ┌────────────────────────────────────────────────────────┐
    # │ STEP 5: Decode generated tokens back to text           │
    # └────────────────────────────────────────────────────────┘
    generated_words = []
    if generated_tokens:
        # Concatenate all generated tokens: [[t1], [t2], [t3]] → [t1, t2, t3]
        generated_tokens = torch.cat(generated_tokens, 1)  # [Batch, Max_Tokens]
        
        for i, x in enumerate(encoded_prompts):
            # Combine original prompt + generated tokens
            full_sequence = x + generated_tokens[i].tolist()
            
            # Decode back to text
            generated_words.append(tokenizer.decode(full_sequence))

    return generated_words, logprobs


def interactive(
    model_path: str, 
    max_tokens: int = 35, 
    temperature: float = 0.7, 
    instruct: bool = False
):
    """
    Interactive chat interface for experimenting with the model.
    
    This creates a REPL (Read-Eval-Print Loop) where you can:
    - Type prompts
    - Get model completions
    - Experiment with different inputs
    
    Args:
        model_path: Path to model directory
        max_tokens: Maximum tokens to generate per response
        temperature: Sampling temperature (0.7 = balanced)
        instruct: Whether to wrap prompts in instruction format
    
    
    ═══════════════════════════════════════════════════════════════════
                    INTERACTIVE MODE USAGE
    ═══════════════════════════════════════════════════════════════════
    
    Run with:
        python main.py interactive --model_path=/path/to/model
    
    Example session:
    
        Prompt: What is the capital of France?
        The capital of France is Paris, a city known for...
        =====================
        
        Prompt: Write a haiku about AI
        Silicon thinking
        Patterns emerge from data
        Intelligence grows
        =====================
        
        Prompt: ^C  (Ctrl+C to exit)
    
    
    Instruct mode (instruct=True):
        Wraps prompts in [INST]...[/INST] tags
        Used for instruction-tuned models
        
        Your input: "Translate to French: Hello"
        Sent to model: "[INST] Translate to French: Hello [/INST]"
    
    ═══════════════════════════════════════════════════════════════════
    """
    # ┌────────────────────────────────────────────────────────┐
    # │ Load model and tokenizer                               │
    # └────────────────────────────────────────────────────────┘
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(Path(model_path), max_batch_size=3)

    # ┌────────────────────────────────────────────────────────┐
    # │ Interactive loop                                       │
    # └────────────────────────────────────────────────────────┘
    while True:
        # Get user input
        prompt = input("Prompt: ")
        
        # Optionally wrap in instruction format
        if instruct:
            prompt = f"[INST] {prompt} [/INST]"
        
        # Generate completion
        res, _logprobs = generate(
            [prompt],
            transformer,
            tokenizer,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Print result
        print(res[0])
        print("=====================")


def demo(
    model_path: str, 
    max_tokens: int = 35, 
    temperature: float = 0, 
    num_pipeline_ranks: int = 1
):
    """
    Demo function showing batch generation with multiple prompts.
    
    This demonstrates:
    - Batched inference (multiple prompts at once)
    - Pipeline parallelism (optional multi-GPU)
    - Deterministic generation (temperature=0)
    
    Args:
        model_path: Path to model directory
        max_tokens: Tokens to generate per prompt
        temperature: 0 = greedy (deterministic), >0 = stochastic
        num_pipeline_ranks: Number of GPUs for pipeline parallelism
    
    
    ═══════════════════════════════════════════════════════════════════
                    DEMO USAGE
    ═══════════════════════════════════════════════════════════════════
    
    Single GPU:
        python main.py demo --model_path=/path/to/model
    
    Multi-GPU (4 GPUs):
        torchrun --nproc_per_node=4 main.py demo \
            --model_path=/path/to/model \
            --num_pipeline_ranks=4
    
    
    Batching benefits:
    ──────────────────
    Processing 3 prompts:
    
    Sequential (batch=1, 3 forward passes):
        Prompt 1: 100ms
        Prompt 2: 100ms  
        Prompt 3: 100ms
        Total: 300ms
    
    Batched (batch=3, 1 forward pass):
        All prompts: 120ms  ← 2.5x faster!
        Total: 120ms
    
    GPU utilization:
        Sequential: 33% (idle between prompts)
        Batched: 90% (fully utilized)
    
    ═══════════════════════════════════════════════════════════════════
    """
    # ┌────────────────────────────────────────────────────────┐
    # │ Setup distributed training (if multi-GPU)              │
    # └────────────────────────────────────────────────────────┘
    if num_pipeline_ranks > 1:
        # Initialize PyTorch distributed process group
        # Each GPU gets a unique rank (0, 1, 2, ...)
        torch.distributed.init_process_group()
        
        # Set current device to this process's rank
        # Rank 0 → GPU 0, Rank 1 → GPU 1, etc.
        torch.cuda.set_device(torch.distributed.get_rank())
        
        # Only rank 0 prints output (avoid duplicate prints)
        should_print = torch.distributed.get_rank() == 0
    else:
        # Single GPU: always print
        should_print = True
    
    # ┌────────────────────────────────────────────────────────┐
    # │ Load model and tokenizer                               │
    # └────────────────────────────────────────────────────────┘
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(
        Path(model_path), 
        max_batch_size=3,  # Support up to 3 sequences in batch
        num_pipeline_ranks=num_pipeline_ranks
    )

    # ┌────────────────────────────────────────────────────────┐
    # │ Generate completions for example prompts               │
    # └────────────────────────────────────────────────────────┘
    res, _logprobs = generate(
        [
            "This is a test made by me with the help of an AI assistant. I also like to play with videogames. Can you recommend me one game to play with?",
            "This is another great test",
            "This is a third test, mistral AI is very good at testing. ",
        ],
        transformer,
        tokenizer,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    # ┌────────────────────────────────────────────────────────┐
    # │ Print results (only on rank 0 for multi-GPU)           │
    # └────────────────────────────────────────────────────────┘
    if should_print:
        for x, l in zip(res, _logprobs):
            print(x)
            logging.debug('Logprobs: %s', l)
            print("=====================")


if __name__ == "__main__":
    """
    Entry point for command-line interface.
    
    Uses Python Fire to create CLI from functions.
    
    Available commands:
    ───────────────────
    
    1. interactive - Chat interface
       python main.py interactive \
           --model_path=/path/to/model \
           --max_tokens=50 \
           --temperature=0.7
    
    2. demo - Batch generation example
       python main.py demo \
           --model_path=/path/to/model \
           --max_tokens=35 \
           --temperature=0
    
    
    Fire automatically:
    ───────────────────
    - Parses command-line arguments
    - Maps to function parameters
    - Provides --help documentation
    - Handles type conversion
    
    Example:
        python main.py interactive --help
        → Shows all parameters for interactive()
    """
    logging.basicConfig(level=logging.INFO)
    fire.Fire({
        "interactive": interactive,
        "demo": demo,
    })


"""
═══════════════════════════════════════════════════════════════════════════
                    GENERATION STRATEGIES DEEP DIVE
═══════════════════════════════════════════════════════════════════════════

1. SAMPLING METHODS COMPARISON
   ────────────────────────────

   Greedy (temperature=0):
   ─────────────────────────
   Always pick highest probability token
   
   Pros:
   ✓ Deterministic (same input → same output)
   ✓ Often picks "safe" reasonable tokens
   ✓ Good for factual Q&A, translation
   
   Cons:
   ✗ Repetitive outputs
   ✗ No diversity
   ✗ Can get stuck in loops
   
   Example:
   "The cat sat on the mat. The cat sat on the mat. The cat..."


   Top-K Sampling:
   ───────────────
   Sample from K most likely tokens
   
   Pros:
   ✓ Some diversity
   ✓ Filters out very unlikely tokens
   
   Cons:
   ✗ Fixed K doesn't adapt to distribution
   ✗ Sometimes K too large (includes nonsense)
   ✗ Sometimes K too small (misses good options)
   
   Example (K=10):
   Always sample from top 10 tokens, even if:
   - Only 3 are reasonable (wastes probability on junk)
   - 20 are reasonable (misses good options)


   Top-P / Nucleus Sampling (what we use):
   ────────────────────────────────────────
   Sample from smallest set with cumulative prob > p
   
   Pros:
   ✓ Adaptive vocabulary size
   ✓ High confidence → few tokens (focused)
   ✓ Low confidence → many tokens (creative)
   ✓ Good balance of quality and diversity
   
   Cons:
   ✗ Slightly more complex to implement
   ✗ Can still occasionally pick weird tokens
   
   Example (p=0.9):
   High confidence: "The capital of France is ___"
   → Only sample "Paris" (90% prob on single token)
   
   Low confidence: "The dragon ___"
   → Sample from ~10 tokens (flew, roared, slept, ...)


   Temperature Scaling:
   ────────────────────
   Controls randomness of distribution
   
   T = 0.1 (low):
   - Very peaked distribution
   - Almost deterministic
   - Use for: code, math, factual answers
   
   T = 0.7 (medium):
   - Balanced distribution
   - Good default for most tasks
   - Use for: general chat, Q&A
   
   T = 1.0 (default):
   - Original distribution
   - Moderate randomness
   
   T = 1.5 (high):
   - Flatter distribution
   - More creative/unexpected
   - Use for: creative writing, brainstorming
   
   T = 2.0+ (very high):
   - Very flat distribution
   - Can produce nonsense
   - Use for: maximum creativity (at risk of coherence)


2. BATCHING STRATEGIES
   ────────────────────

   Static Batching:
   ────────────────
   - Fixed batch size
   - Pad sequences to same length
   - Simple but wasteful
   
   Example:
   Seq 1: [1,2,3,4,5] + [PAD, PAD, PAD]
   Seq 2: [1,2] + [PAD, PAD, PAD, PAD, PAD, PAD]
   → Lots of wasted computation on PAD tokens


   Dynamic Batching:
   ─────────────────
   - Group similar-length sequences
   - Minimal padding
   - More efficient
   
   Example:
   Batch 1: Lengths [10, 11, 12] → pad to 12
   Batch 2: Lengths [50, 48, 51] → pad to 51
   → Less waste than mixing [10, 50]


   Continuous Batching (advanced):
   ────────────────────────────────
   - Add/remove sequences dynamically
   - As one finishes (generates EOS), add new one
   - Maximizes GPU utilization
   - Used in production inference servers


3. MEMORY OPTIMIZATIONS
   ────────────────────

   KV Cache (what we use):
   ───────────────────────
   Memory: O(layers × batch × window × heads × dim)
   Speedup: ~100-500x vs recomputation
   Critical for real-time generation!


   Flash Attention:
   ────────────────
   - Fused attention kernel
   - Reduces memory by not materializing attention matrix
   - 2-4x faster than standard attention
   - Enabled via xformers.memory_efficient_attention


   Quantization:
   ─────────────
   - int8/int4 instead of float16
   - 2-4x less memory
   - Slight quality degradation
   - Enables larger models on smaller GPUs
   
   Example:
   Mistral-7B:
   - float16: ~14 GB
   - int8: ~7 GB
   - int4: ~3.5 GB


   Gradient Checkpointing (training only):
   ────────────────────────────────────────
   - Trade compute for memory
   - Recompute activations during backward pass
   - Allows training larger models
   - Not used in inference


4. GENERATION QUALITY TIPS
   ────────────────────────

   For Factual Q&A:
   ────────────────
   - temperature = 0 (greedy)
   - OR temperature = 0.1-0.3 (mostly deterministic)
   - Prioritize correctness over creativity


   For Creative Writing:
   ─────────────────────
   - temperature = 0.7-1.0
   - top_p = 0.9
   - Allow more randomness for interesting outputs


   For Code Generation:
   ─────────────────────
   - temperature = 0-0.2
   - top_p = 0.95
   - Need precision, but some creativity for variable names


   For Chat/Assistant:
   ───────────────────
   - temperature = 0.7
   - top_p = 0.9
   - Good balance of helpfulness and naturalness


   For Brainstorming:
   ──────────────────
   - temperature = 1.0-1.5
   - top_p = 0.95
   - Maximum diversity for idea generation


5. COMMON GENERATION ISSUES
   ─────────────────────────

   Repetition Loops:
   ─────────────────
   Problem: "The cat sat on the mat. The cat sat on the mat..."
   
   Causes:
   - Temperature too low (greedy)
   - Model gets stuck in local maximum
   
   Solutions:
   - Increase temperature (0.7-1.0)
   - Use repetition penalty (penalize recently used tokens)
   - Use top-p sampling (more diversity)


   Incoherent Output:
   ──────────────────
   Problem: "The xylophone banana quantum refrigerator..."
   
   Causes:
   - Temperature too high
   - Sampling from very low probability tokens
   
   Solutions:
   - Decrease temperature (0.5-0.7)
   - Lower top_p (0.8-0.9)
   - Use top-k to hard cutoff unlikely tokens


   Premature EOS:
   ──────────────
   Problem: Model generates end-of-sequence too early
   
   Causes:
   - Model thinks it's done
   - EOS token has high probability
   
   Solutions:
   - Suppress EOS token for first N tokens
   - Use min_length parameter
   - Adjust prompt to encourage continuation


   Off-Topic Generation:
   ─────────────────────
   Problem: Model ignores prompt, generates unrelated text
   
   Causes:
   - Weak prompt
   - Model not instruction-tuned
   
   Solutions:
   - Use clearer, more specific prompts
   - Use instruction format: [INST]...[/INST]
   - Use few-shot examples in prompt


6. PERFORMANCE BENCHMARKS
   ───────────────────────

   Typical speeds (Mistral-7B on A100, bfloat16):
   
   Prefill (prompt processing):
   - Batch=1: ~1000 tokens/sec
   - Batch=8: ~5000 tokens/sec
   - Compute-bound (can parallelize)
   
   Generation (autoregressive):
   - Batch=1: ~50-80 tokens/sec
   - Batch=8: ~200-300 tokens/sec
   - Memory-bound (limited by bandwidth)
   
   
   Bottleneck analysis:
   ────────────────────
   
   Prefill: Compute-bound
   - Lots of parallel matrix multiplications
   - GPU cores fully utilized
   - Can batch efficiently
   
   Generation: Memory-bound
   - Sequential (can't parallelize within sequence)
   - Spend time loading weights from memory
   - GPU cores underutilized
   - This is why KV cache is crucial!
   
   
   Rule of thumb:
   ──────────────
   Generation speed ≈ Model_Size_GB / Memory_Bandwidth_GB/s
   
   Example (A100 with 2TB/s bandwidth, Mistral-7B at 14GB):
   14 GB / 2000 GB/s ≈ 7ms per token ≈ 142 tokens/sec
   
   Actual: ~80 tokens/sec (overhead, batch effects, etc.)


7. DEBUGGING GENERATION
   ────────────────────

   Log Probabilities:
   ──────────────────
   Track logprobs to understand model confidence:
   
   High confidence (logprob ≈ 0):
   - Model is sure about this token
   - Example: "The capital of France is" → "Paris" (logprob: -0.05)
   
   Low confidence (logprob < -5):
   - Model uncertain, many options
   - Example: "The color could be" → "blue" (logprob: -3.2)
   
   Very low (logprob < -10):
   - Model surprised by this token
   - Might indicate bad generation or rare word


   Perplexity:
   ───────────
   Measure of how surprised model is by text:
   
   PPL = exp(-mean(logprobs))
   
   Low perplexity (< 10): Model confident, natural text
   High perplexity (> 100): Model confused, unnatural text


   Monitoring:
   ───────────
   During generation, watch for:
   - Sudden perplexity spikes (model getting confused)
   - Repetitive patterns (stuck in loop)
   - Decreasing logprobs (model getting uncertain)


8. PRODUCTION CONSIDERATIONS
   ──────────────────────────

   Latency vs Throughput:
   ──────────────────────
   Latency (time per request):
   - Use small batches (1-4)
   - Fast response for single user
   
   Throughput (requests per second):
   - Use large batches (16-32)
   - Maximize GPU utilization
   - Higher latency per request, but more total requests


   Serving Frameworks:
   ───────────────────
   - vLLM: High-throughput inference
   - TGI (Text Generation Inference): HuggingFace solution
   - TensorRT-LLM: NVIDIA optimized
   - Serve with FastAPI/gRPC for production


   Cost Optimization:
   ──────────────────
   - Use smaller models when possible (Mistral-7B vs GPT-4)
   - Batch requests aggressively
   - Cache common responses
   - Use int8 quantization (2x cheaper)
   - Spot instances for non-critical workloads


   Monitoring:
   ───────────
   Key metrics:
   - Tokens/second (throughput)
   - Latency p50, p95, p99
   - GPU utilization
   - Memory usage
   - Request queue depth

═══════════════════════════════════════════════════════════════════════════
"""