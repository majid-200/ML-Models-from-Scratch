import torch
import torch.nn as nn
from torch.utils.data import Dataset

"""
BILINGUAL DATASET FOR TRANSFORMER TRANSLATION
==============================================

Purpose: Prepares parallel text data for training a translation model
Example: English → French translation

Key Concepts:
1. Special Tokens: [SOS] (Start of Sequence), [EOS] (End of Sequence), [PAD] (Padding)
2. Encoder Input: Full source sentence with [SOS] and [EOS]
3. Decoder Input: Target sentence with [SOS] only (for teacher forcing)
4. Label: Target sentence with [EOS] only (what the model should predict)
5. Masks: Prevent attention to padding and future tokens
"""

class BilingualDataset(Dataset):
    """
    Dataset for sequence-to-sequence translation tasks
    
    EXAMPLE DATA FLOW:
    ------------------
    Source (English): "Hello world"
    Target (French):  "Bonjour monde"
    
    After tokenization and processing:
    
    encoder_input: [SOS] Hello world [EOS] [PAD] [PAD] ...
    decoder_input: [SOS] Bonjour monde [PAD] [PAD] [PAD] ...
    label:         Bonjour monde [EOS] [PAD] [PAD] [PAD] ...
    
    WHY DIFFERENT?
    - Encoder sees complete source with boundaries
    - Decoder input starts with [SOS] for first prediction
    - Label is shifted by 1 (what decoder should output next)
    """

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        """
        Args:
            ds: Dataset containing translation pairs
            tokenizer_src: Tokenizer for source language (e.g., English)
            tokenizer_tgt: Tokenizer for target language (e.g., French)
            src_lang: Source language code (e.g., "en")
            tgt_lang: Target language code (e.g., "fr")
            seq_len: Maximum sequence length (all sequences padded/truncated to this)
        """
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Special tokens as tensors (for efficient concatenation)
        # These mark sentence boundaries and padding
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        """Return number of translation pairs in dataset"""
        return len(self.ds)

    def __getitem__(self, idx):
        """
        Get a single training example with all necessary components
        
        Returns a dictionary containing:
        - encoder_input: Source sentence for encoder
        - decoder_input: Target sentence for decoder (teacher forcing)
        - encoder_mask: Mask for encoder (hide padding)
        - decoder_mask: Mask for decoder (hide padding + future tokens)
        - label: Ground truth for training (what decoder should predict)
        - src_text: Original source text (for debugging/logging)
        - tgt_text: Original target text (for debugging/logging)
        """
        # Get the translation pair from dataset
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # TOKENIZATION: Convert text to integer IDs
        # Example: "Hello world" → [5234, 8912]
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # CALCULATE PADDING
        # We need to fit: [SOS] + tokens + [EOS] + [PAD]... = seq_len
        
        # Encoder: needs both [SOS] and [EOS] (total: 2 special tokens)
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        
        # Decoder input: needs only [SOS] (total: 1 special token)
        # We don't add [EOS] here because that's what the model should predict!
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Safety check: if negative, the sentence is too long for seq_len
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            # Truncate to fit
            max_enc_len = self.seq_len - 2  # Reserve space for SOS and EOS
            max_dec_len = self.seq_len - 1  # Reserve space for SOS
            
            enc_input_tokens = enc_input_tokens[:max_enc_len]
            dec_input_tokens = dec_input_tokens[:max_dec_len]
            
            # Recalculate padding
            enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
            dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
            
            print(f"Truncated sentence {idx}")
            # raise ValueError("Sentence is too long")

        # ====================================================================
        # BUILD ENCODER INPUT
        # ====================================================================
        # Structure: [SOS] + tokens + [EOS] + [PAD] [PAD] [PAD] ...
        # 
        # Visual example (seq_len=10):
        # Original: "Hello world"
        # Tokens: [5234, 8912]
        # Result: [SOS] [5234] [8912] [EOS] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
        #          ^                    ^    ^---- padding to seq_len ----^
        #          |                    |
        #        Start                 End
        encoder_input = torch.cat(
            [
                self.sos_token,                                                    # [SOS]
                torch.tensor(enc_input_tokens, dtype=torch.int64),                 # Actual tokens
                self.eos_token,                                                    # [EOS]
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),  # Padding
            ],
            dim=0,
        )

        # ====================================================================
        # BUILD DECODER INPUT (for teacher forcing during training)
        # ====================================================================
        # Structure: [SOS] + tokens + [PAD] [PAD] [PAD] ...
        # Note: NO [EOS] - the model should learn to predict it!
        # 
        # Visual example (seq_len=10):
        # Original: "Bonjour monde"
        # Tokens: [3421, 7654]
        # Result: [SOS] [3421] [7654] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
        #          ^                  ^---- padding to seq_len ----^
        #          |
        #        Start (decoder begins here)
        decoder_input = torch.cat(
            [
                self.sos_token,                                                    # [SOS]
                torch.tensor(dec_input_tokens, dtype=torch.int64),                 # Actual tokens
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),  # Padding
            ],
            dim=0,
        )

        # ====================================================================
        # BUILD LABEL (ground truth for training)
        # ====================================================================
        # Structure: tokens + [EOS] + [PAD] [PAD] [PAD] ...
        # Note: NO [SOS] - this is what the decoder should OUTPUT
        # 
        # Visual example (seq_len=10):
        # Tokens: [3421, 7654]
        # Result: [3421] [7654] [EOS] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
        #         ^             ^      ^---- padding to seq_len ----^
        #         |             |
        #    First word     Last word + end marker
        #
        # TEACHER FORCING EXPLAINED:
        # At each step, decoder input and label are offset by 1:
        #   Decoder sees: [SOS]   [3421]  [7654]  [PAD] ...
        #   Should output: [3421] [7654]  [EOS]   [PAD] ...
        #                  ^       ^       ^
        #                  Step 1  Step 2  Step 3
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),                 # Actual tokens
                self.eos_token,                                                    # [EOS]
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),  # Padding
            ],
            dim=0,
        )

        # ====================================================================
        # SANITY CHECKS
        # ====================================================================
        # Ensure all sequences are exactly seq_len (for batching)
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # ====================================================================
        # CREATE MASKS
        # ====================================================================
        # Masks tell the attention mechanism which positions to ignore
        
        # ENCODER MASK: Hide padding tokens only
        # Shape: (1, 1, seq_len)
        # Visual: [1, 1, 1, 1, 1, 0, 0, 0, ...]  (1=attend, 0=ignore)
        #         ^-real tokens-^ ^-padding-^
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
        
        # DECODER MASK: Hide padding AND future tokens (causal mask)
        # Shape: (1, seq_len, seq_len)
        # This prevents the decoder from "cheating" by looking at future words
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0))
        
        return {
            "encoder_input": encoder_input,    # (seq_len)
            "decoder_input": decoder_input,    # (seq_len)
            "encoder_mask": encoder_mask,      # (1, 1, seq_len)
            "decoder_mask": decoder_mask,      # (1, seq_len, seq_len)
            "label": label,                    # (seq_len)
            "src_text": src_text,              # Original source text
            "tgt_text": tgt_text,              # Original target text
        }


def causal_mask(size):
    """
    Creates a causal (triangular) mask for the decoder
    
    Purpose: Prevent the decoder from looking at future tokens during training
    
    WHY NEEDED?
    During training, we give the decoder the full target sequence at once (teacher forcing).
    But we need to simulate autoregressive generation where each word can only depend
    on previous words, not future ones.
    
    VISUAL EXAMPLE (size=5):
    
    Attention matrix WITHOUT mask (bad - can see future):
        [can attend to all positions]
        Position:  0  1  2  3  4
               0: [1  1  1  1  1]  ← Position 0 can see positions 1,2,3,4 (WRONG!)
               1: [1  1  1  1  1]  ← Position 1 can see positions 2,3,4 (WRONG!)
               2: [1  1  1  1  1]
               3: [1  1  1  1  1]
               4: [1  1  1  1  1]
    
    Attention matrix WITH causal mask (good - can only see past):
        Position:  0  1  2  3  4
               0: [1  0  0  0  0]  ← Position 0 can only see itself
               1: [1  1  0  0  0]  ← Position 1 can see 0 and 1
               2: [1  1  1  0  0]  ← Position 2 can see 0, 1, and 2
               3: [1  1  1  1  0]  ← Position 3 can see 0, 1, 2, and 3
               4: [1  1  1  1  1]  ← Position 4 can see all previous
        
        1 = can attend (visible)
        0 = cannot attend (masked)
    
    This is also called:
    - Causal mask (because it enforces causal/temporal ordering)
    - Look-ahead mask (prevents looking ahead)
    - Triangular mask (shape is lower triangular)
    - Auto-regressive mask (enforces auto-regressive generation)
    
    Args:
        size: Sequence length
    
    Returns:
        Boolean mask of shape (1, size, size) where True = can attend
    
    IMPLEMENTATION DETAILS:
    ----------------------
    torch.triu() creates upper triangular matrix:
        torch.ones((5, 5)) = [[1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1]]
        
        torch.triu(..., diagonal=1) = [[0, 1, 1, 1, 1],  ← diagonal=1 means start from diagonal+1
                                       [0, 0, 1, 1, 1],
                                       [0, 0, 0, 1, 1],
                                       [0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0]]
        
        mask == 0 flips it to get:    [[1, 0, 0, 0, 0],  ← Now 1s are where we WANT to attend
                                       [1, 1, 0, 0, 0],
                                       [1, 1, 1, 0, 0],
                                       [1, 1, 1, 1, 0],
                                       [1, 1, 1, 1, 1]]
    
    REAL EXAMPLE WITH WORDS:
    ------------------------
    Sentence: [SOS] "I" "love" "coding" [EOS]
    
    When predicting "I":     Can see: [SOS]
    When predicting "love":  Can see: [SOS] "I"
    When predicting "coding": Can see: [SOS] "I" "love"
    When predicting [EOS]:   Can see: [SOS] "I" "love" "coding"
    
    This mimics how the model will work during inference (generation),
    where it truly can only see past tokens!
    """
    # Create upper triangular matrix (1s above diagonal)
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    
    # Flip it: 0s become 1s (can attend), 1s become 0s (cannot attend)
    return mask == 0  # Returns boolean tensor


"""
COMPLETE EXAMPLE WALKTHROUGH
=============================

Input pair:
    Source (en): "Hello world"
    Target (fr): "Bonjour monde"

After tokenization:
    src_tokens: [5234, 8912]
    tgt_tokens: [3421, 7654]

Assuming seq_len=10, the dataset returns:

encoder_input: [SOS, 5234, 8912, EOS, PAD, PAD, PAD, PAD, PAD, PAD]
               └───────────────────┘ └─────────────────────────────┘
                 Real sentence           Padding

decoder_input: [SOS, 3421, 7654, PAD, PAD, PAD, PAD, PAD, PAD, PAD]
               └──────────────┘  └────────────────────────────────┘
                Real sentence         Padding

label:         [3421, 7654, EOS, PAD, PAD, PAD, PAD, PAD, PAD, PAD]
               └──────────────┘  └────────────────────────────────┘
                What to predict      Ignore during loss

encoder_mask: [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]  (attend to real tokens only)

decoder_mask: [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],   (causal + padding mask)
               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
               [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],   ← Padding positions
               ...]
"""