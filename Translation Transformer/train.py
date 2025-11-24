from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

# import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

"""
TRANSFORMER TRAINING SCRIPT
===========================

This script handles:
1. Loading and tokenizing bilingual data
2. Building the transformer model
3. Training loop with teacher forcing
4. Validation with greedy decoding
5. Checkpoint saving and loading
6. Metrics tracking (loss, CER, WER, BLEU)
"""

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Greedy Decoding: Generate translation one token at a time
    
    INFERENCE PROCESS (how the model generates translations):
    ---------------------------------------------------------
    
    Unlike training (teacher forcing), during inference we don't have the target!
    We must generate word by word:
    
    Step 1: Start with [SOS]
        Input:  [SOS]
        Output: "Bonjour"
    
    Step 2: Add predicted word
        Input:  [SOS] Bonjour
        Output: "monde"
    
    Step 3: Continue...
        Input:  [SOS] Bonjour monde
        Output: [EOS]
    
    Step 4: Stop when we see [EOS] or reach max_len
    
    VISUAL PROCESS:
    ---------------
    Source: "Hello world" [already encoded by encoder]
    
    Decoder sees:  [SOS]               → Predicts: "Bonjour"
    Decoder sees:  [SOS] Bonjour       → Predicts: "monde"
    Decoder sees:  [SOS] Bonjour monde → Predicts: [EOS]
    
    Final output: [SOS] Bonjour monde [EOS]
    
    WHY "GREEDY"?
    At each step, we pick the word with highest probability (greedy choice).
    More advanced: Beam search (consider multiple possibilities)
    
    Args:
        model: Trained transformer
        source: Source sentence tokens (batch=1, seq_len)
        source_mask: Mask for source
        tokenizer_src: Source tokenizer
        tokenizer_tgt: Target tokenizer
        max_len: Maximum generation length (safety limit)
        device: cuda/cpu
    
    Returns:
        Generated token sequence
    """
    # Get special token IDs
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # OPTIMIZATION: Encode source once and reuse
    # The encoder output doesn't change during generation
    # Only the decoder input changes at each step
    encoder_output = model.encode(source, source_mask)
    
    # Initialize decoder input with just [SOS]
    # Shape: (1, 1) - batch_size=1, seq_len=1
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    # GENERATION LOOP
    while True:
        # Safety check: don't generate forever
        if decoder_input.size(1) == max_len:
            break

        # Build causal mask for current decoder input length
        # Mask grows as we generate more tokens
        # Step 1: (1, 1, 1)  - only [SOS]
        # Step 2: (1, 2, 2)  - [SOS] + first word
        # Step 3: (1, 3, 3)  - [SOS] + first + second word
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Run decoder with what we have so far
        # decoder_input grows each iteration: [SOS] → [SOS, w1] → [SOS, w1, w2] → ...
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get prediction for the NEXT token
        # out[:, -1] selects the last position (most recent prediction)
        # Shape: (1, vocab_size) - probability distribution over all words
        prob = model.project(out[:, -1])
        
        # Pick the word with highest probability (greedy choice)
        _, next_word = torch.max(prob, dim=1)
        
        # Append predicted word to decoder input
        # This becomes input for next iteration
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], 
            dim=1
        )

        # Stop if we predicted [EOS]
        if next_word == eos_idx:
            break

    # Return generated sequence (remove batch dimension)
    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, 
                   print_msg, global_step, writer, num_examples=2):
    """
    Run validation to check translation quality
    
    PURPOSE:
    --------
    - Test model on unseen data
    - Compute metrics: CER, WER, BLEU
    - Print example translations for inspection
    
    METRICS EXPLAINED:
    -----------------
    1. CER (Character Error Rate): % of characters wrong
       - Lower is better (0 = perfect)
       - Good for typos, spelling
    
    2. WER (Word Error Rate): % of words wrong
       - Lower is better (0 = perfect)
       - Good for overall word accuracy
    
    3. BLEU (Bilingual Evaluation Understudy): Translation quality score
       - Higher is better (1.0 = perfect, 0 = terrible)
       - Industry standard for machine translation
       - Considers n-gram overlap with reference
    
    EXAMPLE:
    --------
    Source:    "Hello world"
    Expected:  "Bonjour monde"
    Predicted: "Bonjour le monde"
    
    CER: 3 extra chars / 13 total ≈ 0.23
    WER: 1 wrong word / 2 total = 0.5
    BLEU: Partial match ≈ 0.6
    """
    model.eval()  # Set to evaluation mode (disables dropout, etc.)
    count = 0

    source_texts = []
    expected = []
    predicted = []

    # Try to get console width for pretty printing
    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80  # Default fallback

    # No gradient computation needed during validation (saves memory)
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (1, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)    # (1, 1, 1, seq_len)

            # Validation uses batch_size=1 for simplicity
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # Generate translation using greedy decoding
            model_out = greedy_decode(
                model, encoder_input, encoder_mask, 
                tokenizer_src, tokenizer_tgt, max_len, device
            )

            # Get text versions for display and metrics
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # Collect for metrics
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print comparison
            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            # Only validate on a few examples (for speed)
            if count == num_examples:
                print_msg('-' * console_width)
                break
    
    # Compute and log metrics
    if writer:
        # CER: Character-level accuracy
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # WER: Word-level accuracy
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # BLEU: Translation quality score
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


def get_all_sentences(ds, lang):
    """
    Generator that yields all sentences in a language
    Used for training tokenizer on the full dataset
    """
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    """
    Build or load a tokenizer for the specified language
    
    TOKENIZER PURPOSE:
    ------------------
    Converts text to integers that the model can process
    
    Example:
        Text: "Hello world"
        Tokens: [5234, 8912]
    
    WORDLEVEL TOKENIZER:
    --------------------
    - Simplest type: each word = one token
    - Vocabulary: all unique words in dataset
    - Unknown words → [UNK] token
    
    More advanced alternatives:
    - BPE (Byte Pair Encoding): subword tokens
    - WordPiece: used by BERT
    - SentencePiece: language-agnostic
    
    SPECIAL TOKENS:
    ---------------
    [UNK]: Unknown word (not in vocabulary)
    [PAD]: Padding (make all sequences same length)
    [SOS]: Start of sequence
    [EOS]: End of sequence
    
    Args:
        config: Configuration dict
        ds: Dataset to train tokenizer on
        lang: Language code (e.g., "en", "fr")
    
    Returns:
        Trained tokenizer
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
    if not Path.exists(tokenizer_path):
        # Build new tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()  # Split on whitespace
        
        # Train tokenizer with special tokens
        # min_frequency=2: ignore words that appear only once
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], 
            min_frequency=2
        )
        
        # Train on all sentences in the dataset
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        
        # Save for future use
        tokenizer.save(str(tokenizer_path))
    else:
        # Load existing tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer


def get_ds(config):
    """
    Load and prepare datasets
    
    PROCESS:
    --------
    1. Load raw bilingual dataset from HuggingFace
    2. Build/load tokenizers for both languages
    3. Split into train (90%) and validation (10%)
    4. Wrap in BilingualDataset (adds padding, masks, etc.)
    5. Create DataLoaders for batching
    
    Returns:
        train_dataloader: Batched training data
        val_dataloader: Validation data (batch_size=1)
        tokenizer_src: Source language tokenizer
        tokenizer_tgt: Target language tokenizer
    """
    # Load dataset (e.g., "opus_books" en-fr translations)
    ds_raw = load_dataset(
        f"{config['datasource']}", 
        f"{config['lang_src']}-{config['lang_tgt']}", 
        split='train'
    )

    # Build tokenizers (or load if already exist)
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Split dataset: 90% train, 10% validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Wrap in BilingualDataset (adds special tokens, padding, masks)
    train_ds = BilingualDataset(
        train_ds_raw, tokenizer_src, tokenizer_tgt, 
        config['lang_src'], config['lang_tgt'], config['seq_len']
    )
    val_ds = BilingualDataset(
        val_ds_raw, tokenizer_src, tokenizer_tgt, 
        config['lang_src'], config['lang_tgt'], config['seq_len']
    )

    # Find maximum sentence lengths (for debugging/info)
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    
    # Create DataLoaders
    # Training: batch multiple examples together for efficiency
    # Validation: batch_size=1 for simplicity during generation
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    """
    Build transformer model with specified configuration
    
    Args:
        config: Configuration dict with hyperparameters
        vocab_src_len: Size of source vocabulary
        vocab_tgt_len: Size of target vocabulary
    
    Returns:
        Initialized transformer model
    """
    model = build_transformer(
        vocab_src_len, 
        vocab_tgt_len, 
        config["seq_len"],      # Source max length
        config['seq_len'],      # Target max length
        d_model=config['d_model']  # Embedding dimension (e.g., 512)
    )
    return model


def train_model(config):
    """
    Main training loop
    
    TRAINING PROCESS OVERVIEW:
    --------------------------
    
    For each epoch:
        For each batch:
            1. Get source and target sentences
            2. Encode source with encoder
            3. Decode with decoder (using teacher forcing)
            4. Compare predictions with labels
            5. Compute loss
            6. Backpropagate
            7. Update weights
        
        After each epoch:
            - Run validation
            - Save checkpoint
    
    TEACHER FORCING EXPLAINED:
    --------------------------
    During training, we give the decoder the REAL target words,
    not its own predictions. This makes training more stable.
    
    Example:
        Target: "Bonjour monde"
        
        Training (teacher forcing):
            Decoder sees: [SOS]           → Should predict: Bonjour
            Decoder sees: [SOS] Bonjour   → Should predict: monde
            Decoder sees: [SOS] Bonjour monde → Should predict: [EOS]
        
        We use the REAL words, even if decoder predicted wrong!
    
    Inference (no teacher forcing):
        Decoder uses its own predictions at each step
    
    LOSS FUNCTION:
    --------------
    CrossEntropyLoss with:
    - ignore_index=[PAD]: Don't compute loss on padding tokens
    - label_smoothing=0.1: Prevents overconfidence (regularization)
    
    VISUAL TRAINING STEP:
    ---------------------
    
    Input:  "Hello world"  →  [ENCODER]  →  Context
                                              ↓
    Target: [SOS] Bonjour monde  →  [DECODER]  →  Predictions
                                                    ↓
    Label:  Bonjour monde [EOS]  ←  Compare  ←  [Loss]
                                                    ↓
                                              [Backprop]
                                                    ↓
                                           [Update Weights]
    """
    # =====================================================================
    # DEVICE SETUP
    # =====================================================================
    # Choose best available device: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
    device = "cuda" if torch.cuda.is_available() \
        else "mps" if torch.has_mps or torch.backends.mps.is_available() \
        else "cpu"
    
    print("Using device:", device)
    
    if device == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif device == 'mps':
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
    
    device = torch.device(device)

    # =====================================================================
    # SETUP: DATA, MODEL, OPTIMIZER
    # =====================================================================
    # Create directory for saving model weights
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # Load datasets and tokenizers
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    
    # Build model and move to device (GPU/CPU)
    model = get_model(
        config, 
        tokenizer_src.get_vocab_size(), 
        tokenizer_tgt.get_vocab_size()
    ).to(device)
    
    # TensorBoard for visualization (view with: tensorboard --logdir=runs)
    writer = SummaryWriter(config['experiment_name'])

    # Adam optimizer: adaptive learning rate for each parameter
    # eps=1e-9: numerical stability
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # =====================================================================
    # CHECKPOINT LOADING (if resuming training)
    # =====================================================================
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    
    # Determine which checkpoint to load (if any)
    model_filename = latest_weights_file_path(config) if preload == 'latest' \
        else get_weights_file_path(config, preload) if preload \
        else None
    
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    # =====================================================================
    # LOSS FUNCTION
    # =====================================================================
    # CrossEntropyLoss for classification (picking the right word)
    # ignore_index: Don't compute loss on [PAD] tokens
    # label_smoothing: Regularization to prevent overconfidence
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id('[PAD]'), 
        label_smoothing=0.1
    ).to(device)

    # =====================================================================
    # MAIN TRAINING LOOP
    # =====================================================================
    for epoch in range(initial_epoch, config['num_epochs']):
        # Free up GPU memory from previous epoch
        torch.cuda.empty_cache()
        
        # Set model to training mode (enables dropout, etc.)
        model.train()
        
        # Progress bar for this epoch
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        
        # ================================================================
        # ITERATE OVER BATCHES
        # ================================================================
        for batch in batch_iterator:
            # Move batch to device (GPU/CPU)
            encoder_input = batch['encoder_input'].to(device)  # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)    # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)    # (B, 1, seq_len, seq_len)

            # ============================================================
            # FORWARD PASS
            # ============================================================
            # Step 1: Encode source sentence
            encoder_output = model.encode(encoder_input, encoder_mask)
            # Shape: (B, seq_len, d_model)
            
            # Step 2: Decode with teacher forcing
            # Uses REAL target words, not predictions
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )
            # Shape: (B, seq_len, d_model)
            
            # Step 3: Project to vocabulary (get word probabilities)
            proj_output = model.project(decoder_output)
            # Shape: (B, seq_len, vocab_size)

            # Get ground truth labels
            label = batch['label'].to(device)  # (B, seq_len)

            # ============================================================
            # COMPUTE LOSS
            # ============================================================
            # Reshape to (B*seq_len, vocab_size) vs (B*seq_len)
            # This flattens the batch and sequence dimensions
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), 
                label.view(-1)
            )
            
            # Display loss in progress bar
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log loss to TensorBoard
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # ============================================================
            # BACKWARD PASS (Backpropagation)
            # ============================================================
            # Compute gradients
            loss.backward()

            # Update model weights using gradients
            optimizer.step()
            
            # Reset gradients for next iteration
            # set_to_none=True: more memory efficient than zero_grad()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # ================================================================
        # VALIDATION (after each epoch)
        # ================================================================
        run_validation(
            model, val_dataloader, tokenizer_src, tokenizer_tgt, 
            config['seq_len'], device, 
            lambda msg: batch_iterator.write(msg),  # Print function
            global_step, writer
        )

        # ================================================================
        # SAVE CHECKPOINT (after each epoch)
        # ================================================================
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),      # Model weights
            'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
            'global_step': global_step
        }, model_filename)


# =========================================================================
# ENTRY POINT
# =========================================================================
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)


"""
COMPLETE TRAINING EXAMPLE WALKTHROUGH
======================================

Let's trace through one training step:

1. BATCH DATA:
   encoder_input: [[SOS, 5234, 8912, EOS, PAD, PAD, ...],  # "Hello world"
                   [SOS, 1111, 2222, 3333, EOS, PAD, ...]]  # Another sentence
   
   decoder_input: [[SOS, 3421, 7654, PAD, PAD, PAD, ...],  # "Bonjour monde"
                   [SOS, 4444, 5555, 6666, PAD, PAD, ...]]  # Another translation
   
   label:         [[3421, 7654, EOS, PAD, PAD, PAD, ...],   # What to predict
                   [4444, 5555, 6666, EOS, PAD, PAD, ...]]

2. FORWARD PASS:
   encoder_input → [ENCODER] → encoder_output (context vectors)
   
   decoder_input + encoder_output → [DECODER] → decoder_output
   
   decoder_output → [PROJECTION] → logits (vocab_size predictions per position)

3. LOSS COMPUTATION:
   For each position in sequence:
       - Compare predicted word distribution with actual word
       - Ignore positions with [PAD]
   
   Example at position 1:
       Predicted distribution: [0.1, 0.05, 0.7, ...]  (vocab_size probabilities)
       Actual word: 3421
       Loss: -log(probability of word 3421)

4. BACKPROPAGATION:
   loss.backward() → Compute gradients for all parameters
   optimizer.step() → Update weights: weight = weight - learning_rate × gradient

5. REPEAT for all batches in epoch

6. VALIDATION:
   - Generate translations using greedy decode (no teacher forcing!)
   - Compare with ground truth
   - Compute metrics (CER, WER, BLEU)

7. SAVE CHECKPOINT:
   - Save model weights
   - Save optimizer state (for resuming training)
   - Save epoch number

TRAINING vs INFERENCE DIFFERENCE:
----------------------------------

TRAINING (Teacher Forcing):
    Input:  [SOS] Bonjour monde
    Output: Bonjour monde [EOS]
    Loss:   Compare with label at each position
    
    Even if model predicts wrong at position 1,
    we still give it the CORRECT word for position 2!

INFERENCE (Greedy Decode):
    Start:  [SOS]
    Step 1: Predict → Bonjour       (use this for next step)
    Step 2: Predict → monde          (use this for next step)
    Step 3: Predict → [EOS]          (stop)
    
    Model uses its OWN predictions at each step!

KEY HYPERPARAMETERS:
--------------------
- batch_size: Number of examples per batch (e.g., 8)
- learning_rate: How much to update weights (e.g., 0.0001)
- num_epochs: How many times to see full dataset (e.g., 20)
- d_model: Embedding dimension (e.g., 512)
- seq_len: Maximum sequence length (e.g., 350)
- label_smoothing: Regularization (e.g., 0.1)

TENSORBOARD USAGE:
------------------
To view training progress:
    tensorboard --logdir=runs

You'll see:
- Training loss curve (should decrease)
- Validation metrics (CER, WER, BLEU)
- Example translations
"""