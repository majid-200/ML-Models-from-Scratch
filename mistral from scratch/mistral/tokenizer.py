from pathlib import Path
from sentencepiece import SentencePieceProcessor
from typing import List


"""
═══════════════════════════════════════════════════════════════════════════
                            TOKENIZATION OVERVIEW
═══════════════════════════════════════════════════════════════════════════

WHAT IS TOKENIZATION?
─────────────────────
Tokenization is the process of converting text into numbers that a model can process.

    "Hello world!" → [15043, 3186, 28808] → Model processes → [Output IDs] → "Hi there!"
     (human text)      (token IDs)          (neural network)    (token IDs)    (human text)


WHY NOT JUST USE CHARACTERS OR WORDS?
──────────────────────────────────────

Character-level (a, b, c, ...):
   ✓ Small vocabulary (~256 for ASCII)
   ✗ Very long sequences (inefficient)
   ✗ Model must learn to group characters into meaningful units
   
Word-level ("hello", "world", ...):
   ✗ HUGE vocabulary (millions of words)
   ✗ Can't handle unknown words
   ✗ No way to represent "uncommon" or misspelled words
   ✗ Wastes vocabulary on rare words

Subword tokenization (BPE, SentencePiece):
   ✓ Balanced vocabulary size (~32k-100k tokens)
   ✓ Can represent ANY text (even unknown words)
   ✓ Common words = single token, rare words = multiple tokens
   ✓ Efficient sequence lengths
   ✓ Used by ALL modern LLMs!


SENTENCEPIECE TOKENIZATION
───────────────────────────
Mistral uses SentencePiece with BPE (Byte-Pair Encoding) algorithm.

Example tokenization:
    "Hello world!" → ["▁Hello", "▁world", "!"]
                   → [15043, 3186, 28808]

Key features:
- Treats spaces as part of tokens (▁ represents space)
- Language-agnostic (works for any language)
- Reversible (decode gets back exact original text)
- Pre-trained vocabulary (learned from large corpus)


VOCABULARY STRUCTURE
────────────────────
Typical Mistral vocabulary (~32k tokens):

1. Special tokens (0-10):
   [PAD], [BOS], [EOS], [UNK], ...
   
2. Single bytes (11-266):
   Individual bytes for fallback (handles ANY UTF-8 text)
   
3. Common subwords (267-32000):
   "▁the", "▁and", "ing", "tion", "▁Hello", etc.


TOKEN IDS:
──────────
- BOS (Beginning Of Sequence): Marks start of text
- EOS (End Of Sequence): Marks end of text
- PAD (Padding): Used to make sequences same length in batches
- UNK (Unknown): Rare, since byte-level fallback exists

═══════════════════════════════════════════════════════════════════════════
"""


class Tokenizer:
    """
    Wrapper around SentencePiece tokenizer for Mistral models.
    
    This handles the conversion between human-readable text and token IDs
    that the neural network can process.
    
    ═══════════════════════════════════════════════════════════════════
                        TOKENIZATION PIPELINE
    ═══════════════════════════════════════════════════════════════════
    
    ENCODING (Text → Numbers):
    ──────────────────────────
    
    Input: "Hello world!"
       ↓
    [Normalization] (optional: lowercase, accent removal, etc.)
       ↓
    [Segmentation] "Hello world!" → ["▁Hello", "▁world", "!"]
       ↓                                   ↓
    [Vocabulary lookup]              Look up in vocab
       ↓                                   ↓
    Output: [1, 15043, 3186, 28808]  (with BOS token prepended)
            ↑
          BOS token (marks start)
    
    
    DECODING (Numbers → Text):
    ──────────────────────────
    
    Input: [1, 15043, 3186, 28808]
       ↓
    [Skip special tokens] → [15043, 3186, 28808]
       ↓
    [Vocabulary lookup] → ["▁Hello", "▁world", "!"]
       ↓
    [Concatenate] → "▁Hello▁world!"
       ↓
    [Replace ▁ with space] → " Hello world!"
       ↓
    Output: "Hello world!"
    
    ═══════════════════════════════════════════════════════════════════
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the tokenizer by loading a pre-trained SentencePiece model.
        
        Args:
            model_path: Path to the .model file (e.g., "tokenizer.model")
                       This file contains:
                       - The vocabulary (32k+ tokens)
                       - Merge rules (how to split text)
                       - Special token IDs
        
        The .model file is created during training on a large text corpus
        and captures the most common subword patterns in that corpus.
        """
        # Verify the model file exists before trying to load it
        assert Path(model_path).exists(), f"Tokenizer model not found: {model_path}"
        
        # Load the pre-trained SentencePiece model
        # This reads the vocabulary and all tokenization rules
        self._model = SentencePieceProcessor(model_file=model_path)
        
        # Sanity check: vocab_size should equal the number of pieces
        # vocab_size = number of unique token IDs
        # get_piece_size = number of token strings in vocabulary
        # These should always be equal in a valid model
        assert self._model.vocab_size() == self._model.get_piece_size(), \
            "Vocabulary size mismatch - model file may be corrupted"

    @property
    def n_words(self) -> int:
        """
        Get the vocabulary size (number of unique tokens).
        
        Returns:
            Number of tokens in vocabulary (typically ~32k for Mistral)
        
        Example:
            tokenizer.n_words → 32000
        
        This determines the size of the embedding layer in the model:
            Embedding(vocab_size, hidden_dim)
        """
        return self._model.vocab_size()

    @property
    def bos_id(self) -> int:
        """
        Get the Beginning-Of-Sequence token ID.
        
        BOS is prepended to the start of text to signal "this is the beginning".
        
        Returns:
            Token ID for BOS (typically 1)
        
        Example usage:
            Input text: "Hello world"
            Encoded:    [1, 15043, 3186]  ← BOS token (1) at start
                        ↑
                      BOS
        
        Why BOS matters:
        - Helps model distinguish between "start of text" and middle of text
        - Important for generation (model knows when it's generating first token)
        - Many models are trained to expect BOS at the start
        """
        return self._model.bos_id()

    @property
    def eos_id(self) -> int:
        """
        Get the End-Of-Sequence token ID.
        
        EOS signals "this is the end of the text".
        
        Returns:
            Token ID for EOS (typically 2)
        
        Example usage:
            Input text: "Hello world"
            Encoded:    [1, 15043, 3186, 2]  ← EOS token (2) at end
                                        ↑
                                       EOS
        
        Why EOS matters:
        - During generation, model outputs EOS to signal "I'm done"
        - Prevents infinite generation loops
        - Can indicate natural stopping points (end of sentence/document)
        
        Note: EOS is typically NOT added during encoding (done separately)
        but IS generated by the model during text generation.
        """
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        """
        Get the Padding token ID.
        
        PAD is used to make sequences the same length in a batch.
        
        Returns:
            Token ID for PAD (often -1 or a special value)
        
        Why padding is needed:
        ───────────────────────
        Neural networks process batches, but sentences have different lengths:
        
        Without padding (can't batch):
            Sequence 1: [1, 234, 567, 890]           (length 4)
            Sequence 2: [1, 111, 222]                (length 3)
            Sequence 3: [1, 999, 888, 777, 666, 555] (length 6)
        
        With padding (can batch):
            Sequence 1: [1, 234, 567, 890,   0,   0] (padded to 6)
            Sequence 2: [1, 111, 222,   0,   0,   0] (padded to 6)
            Sequence 3: [1, 999, 888, 777, 666, 555] (already 6)
                                        ↑
                                    PAD tokens
        
        The model learns to ignore PAD tokens (through attention masks).
        """
        return self._model.pad_id()

    def encode(self, s: str, bos: bool = True) -> List[int]:
        """
        Convert text string to list of token IDs.
        
        This is the primary interface for preparing text to feed into the model.
        
        Args:
            s: Input text string (any UTF-8 text)
            bos: Whether to prepend BOS token (default True)
                 - True: Good for standalone text/prompts
                 - False: Good when continuing existing sequence
        
        Returns:
            List of integer token IDs
        
        
        ═══════════════════════════════════════════════════════════════
                        ENCODING EXAMPLES
        ═══════════════════════════════════════════════════════════════
        
        Example 1: Simple sentence
        ──────────────────────────
        Input:  "Hello world"
        Output: [1, 15043, 3186]
                 ↑    ↑      ↑
               BOS "Hello" "▁world"
        
        
        Example 2: Rare/unknown word
        ─────────────────────────────
        Input:  "Supercalifragilisticexpialidocious"
        Output: [1, 5514, 9999, 1234, 5678, 9012, ...]
                 ↑  └────────────┬───────────────┘
               BOS    Broken into multiple subword tokens
        
        Common words = 1 token, Rare words = multiple tokens
        
        
        Example 3: Multiple languages
        ──────────────────────────────
        Input:  "Hello 世界"  (English + Chinese)
        Output: [1, 15043, 29871, 30640, 30계]
                 ↑    ↑       ↑      ↑     ↑
               BOS "Hello"  space  世    界
        
        SentencePiece handles ANY UTF-8 text (multilingual)!
        
        
        Example 4: Code
        ───────────────
        Input:  "def hello():"
        Output: [1, 1753, 22172, 5658]
                 ↑    ↑     ↑     ↑
               BOS  "def" "▁hello" "():"
        
        Code tokens are just like text tokens!
        
        ═══════════════════════════════════════════════════════════════
        """
        # Input validation: ensure we have a string
        assert isinstance(s, str), f"Input must be string, got {type(s)}"
        
        # ┌────────────────────────────────────────────────────┐
        # │ Core tokenization happens here                     │
        # └────────────────────────────────────────────────────┘
        # The SentencePiece model:
        # 1. Normalizes the text (if configured)
        # 2. Segments into subword tokens
        # 3. Looks up each token in vocabulary
        # 4. Returns list of integer IDs
        t = self._model.encode(s)
        
        # ┌────────────────────────────────────────────────────┐
        # │ Optionally prepend BOS (Beginning Of Sequence)     │
        # └────────────────────────────────────────────────────┘
        # Most models expect BOS at the start to signal "new text"
        # Use bos=False when continuing a sequence or when BOS was already added
        if bos:
            t = [self.bos_id, *t]  # Prepend BOS token ID
        
        return t
        # Result: [BOS_ID, token1_id, token2_id, token3_id, ...]

    def decode(self, t: List[int]) -> str:
        """
        Convert list of token IDs back to text string.
        
        This reverses the encoding process. Used for:
        - Reading model outputs (generation)
        - Debugging (see what tokens represent)
        - Displaying results to users
        
        Args:
            t: List of token IDs (integers)
        
        Returns:
            Human-readable text string
        
        
        ═══════════════════════════════════════════════════════════════
                        DECODING EXAMPLES
        ═══════════════════════════════════════════════════════════════
        
        Example 1: Standard decoding
        ─────────────────────────────
        Input:  [1, 15043, 3186]
        Step 1: Look up tokens → [<BOS>, "Hello", "▁world"]
        Step 2: Skip BOS (automatically)
        Step 3: Concatenate → "Hello▁world"
        Step 4: Replace ▁ → "Hello world"
        Output: "Hello world"
        
        
        Example 2: With punctuation
        ────────────────────────────
        Input:  [1, 15043, 3186, 28808]
        Tokens: [<BOS>, "Hello", "▁world", "!"]
        Output: "Hello world!"
        
        
        Example 3: Multiple sentences
        ──────────────────────────────
        Input:  [1, 15043, 29889, 1128, 526, 366, 29973]
        Tokens: [<BOS>, "Hello", ".", "▁How", "▁are", "▁you", "?"]
        Output: "Hello. How are you?"
        
        
        Example 4: Handling EOS
        ────────────────────────
        Input:  [1, 15043, 3186, 2]
        Tokens: [<BOS>, "Hello", "▁world", <EOS>]
        Output: "Hello world"  (EOS automatically stripped)
        
        Special tokens (BOS, EOS, PAD) are automatically handled!
        
        ═══════════════════════════════════════════════════════════════
        """
        # ┌────────────────────────────────────────────────────┐
        # │ Core decoding happens here                         │
        # └────────────────────────────────────────────────────┘
        # The SentencePiece model:
        # 1. Looks up each ID in vocabulary → subword strings
        # 2. Concatenates all subwords
        # 3. Replaces ▁ (space marker) with actual spaces
        # 4. Removes special tokens (BOS, EOS, PAD)
        # 5. Returns clean UTF-8 string
        return self._model.decode(t)


"""
═══════════════════════════════════════════════════════════════════════════
                        KEY TOKENIZATION CONCEPTS
═══════════════════════════════════════════════════════════════════════════

1. VOCABULARY SIZE TRADEOFFS
   ──────────────────────────
   
   Small vocab (8k tokens):
   ✓ Smaller embedding layer (fewer parameters)
   ✓ Less memory usage
   ✗ Longer sequences (inefficient)
   ✗ More tokens per word
   
   Large vocab (100k tokens):
   ✓ Shorter sequences (efficient)
   ✓ Fewer tokens per word
   ✗ Huge embedding layer (many parameters)
   ✗ More memory usage
   ✗ Rare tokens undertrained
   
   Sweet spot: 32k-64k tokens
   - Used by most modern LLMs (GPT-4, Llama, Mistral)
   - Good balance of efficiency and quality


2. BYTE-PAIR ENCODING (BPE) ALGORITHM
   ───────────────────────────────────
   
   Training process (simplified):
   
   Start with: Individual characters/bytes
               ['a', 'b', 'c', 'd', ..., 'z']
   
   Step 1: Find most common pair → "th"
           Add to vocabulary
   
   Step 2: Find next most common → "e "
           Add to vocabulary
   
   Step 3-32000: Repeat until target vocab size
   
   Result: Vocabulary of common subwords
           ["a", "b", ..., "the", "ing", "tion", "▁Hello", ...]
   
   Common patterns = single token
   Rare patterns = multiple tokens


3. TOKENIZATION ARTIFACTS
   ───────────────────────
   
   Tokenization affects model behavior!
   
   Example 1: Space sensitivity
   "hello" vs " hello" may tokenize differently:
   - "hello"  → [15043]
   - " hello" → [29871, 15043]  (space is separate token)
   
   Example 2: Case sensitivity
   "HELLO" vs "hello" are different tokens:
   - Model must learn both separately
   
   Example 3: Repetition issues
   "ha" (1 token) vs "haha" (2 tokens) vs "hahaha" (3 tokens)
   - Model may struggle with repeated patterns
   
   Example 4: Trailing spaces
   "test " vs "test" tokenize differently
   - Important for chat/instruct models


4. SPECIAL TOKEN USAGE IN PRACTICE
   ────────────────────────────────
   
   Typical sequence structure:
   
   [BOS] The quick brown fox [EOS]
     ↑                          ↑
   Start marker              End marker
   
   
   For chat models (instruction tuning):
   
   [BOS] [INST] Question here [/INST] Answer here [EOS]
          ↑                     ↑
     Instruction markers   Separates Q/A
   
   
   Batched sequences (training):
   
   Seq 1: [BOS] Hello world    [EOS] [PAD] [PAD]
   Seq 2: [BOS] Hi there       [EOS] [PAD] [PAD]
   Seq 3: [BOS] Good morning   [EOS] [PAD] [PAD]
                                ↑      ↑      ↑
                            Padding to same length


5. TOKENIZATION GOTCHAS
   ────────────────────
   
   Problem: Token boundaries don't align with words
   Example: "unhappiness"
            → ["un", "happiness"]  ✓ Good split
            → ["unh", "app", "iness"]  ✗ Bad split
   
   Problem: Numbers are inefficient
   Example: "3.14159265"
            → Might be 10+ tokens!
            → Consider special number tokenization
   
   Problem: Code indentation
   Example: Spaces in code may each be separate tokens
            → Inefficient for code models
            → Some models use special tab/indent tokens
   
   Problem: Multilingual imbalance
   Example: English "hello" = 1 token
            Chinese "你好" = 2+ tokens
            → Non-English text uses more tokens
            → Important for pricing/limits!


6. WHY SENTENCEPIECE?
   ──────────────────
   
   Alternatives: BERT's WordPiece, GPT-2's BPE, Tiktoken
   
   SentencePiece advantages:
   ✓ Language-agnostic (no need for pre-tokenization)
   ✓ Treats spaces as characters (reversible)
   ✓ Built-in vocabulary training
   ✓ Efficient C++ implementation
   ✓ Widely used (T5, LLaMA, Mistral, PaLM)
   
   Used by most modern open-source LLMs!

═══════════════════════════════════════════════════════════════════════════
"""