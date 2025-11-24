from pathlib import Path

"""
CONFIGURATION FILE FOR TRANSFORMER TRAINING
============================================

This file centralizes all hyperparameters and settings for training.
Benefits:
- Easy to experiment with different settings
- No need to modify main training code
- Keep track of what settings produced which results
"""


def get_config():
    """
    Returns configuration dictionary with all hyperparameters
    
    HYPERPARAMETERS EXPLAINED:
    ==========================
    
    batch_size: How many translation pairs to process at once
    ┌─────────────────────────────────────────────────────── ─┐
    │ WHY IT MATTERS:                                         │
    │ - Larger batch: Faster training, more memory required   │
    │ - Smaller batch: Slower training, less memory           │
    │ - Typical values: 8, 16, 32, 64                         │
    │                                                         │
    │ Example with batch_size=8:                              │
    │   Process 8 sentences simultaneously                    │
    │   All padded to same length (seq_len)                   │
    │   GPU can parallelize computation                       │
    └─────────────────────────────────────────────────────────┘
    
    num_epochs: How many times to see the entire dataset
    ┌─────────────────────────────────────────────────────────┐
    │ ONE EPOCH = One pass through all training data          │
    │                                                         │
    │ Example: 10,000 sentences, batch_size=8                 │
    │   1 epoch = 10,000/8 = 1,250 batches                    │
    │   20 epochs = 20 × 1,250 = 25,000 training steps        │
    │                                                         │
    │ More epochs = More learning, but risk overfitting       │
    └─────────────────────────────────────────────────────────┘
    
    lr (learning_rate): How fast the model learns
    ┌─────────────────────────────────────────────────────────┐
    │ VISUALIZATION:                                          │
    │                                                         │
    │ Too large (e.g., 0.1):                                  │
    │   Loss ╱╲╱╲╱╲  (unstable, jumps around)                 │
    │                                                         │
    │ Good (e.g., 0.0001):                                    │
    │   Loss ╲_____  (smooth decrease)                        │
    │                                                         │
    │ Too small (e.g., 0.000001):                             │
    │   Loss ╲        (learns too slowly)                     │
    │                                                         │
    │ 10**-4 = 0.0001 is a good starting point                │
    └─────────────────────────────────────────────────────────┘
    
    seq_len: Maximum sequence length (in tokens)
    ┌─────────────────────────────────────────────────────────┐
    │ ALL sequences padded/truncated to this length           │
    │                                                         │
    │ Example with seq_len=350:                               │
    │   Short sentence (5 words):                             │
    │     [SOS] word1 word2 word3 word4 [EOS] [PAD]...[PAD]   │
    │     └───────────────────────────────────────────────┘   │
    │                    350 tokens total                     │
    │                                                         │
    │   Long sentence (400 words):                            │
    │     TRUNCATED to fit 350 tokens                         │
    │                                                         │
    │ Trade-off:                                              │
    │   - Longer: Can handle longer sentences, more memory    │
    │   - Shorter: Faster, less memory, may cut sentences     │
    │                                                         │
    │ Typical values: 128 (short), 512 (medium), 1024 (long)  │
    └─────────────────────────────────────────────────────────┘
    
    d_model: Embedding dimension / model width
    ┌─────────────────────────────────────────────────────────┐
    │ How many numbers represent each word                    │
    │                                                         │
    │ Visual representation:                                  │
    │   d_model=512:                                          │
    │   "Hello" → [0.2, -0.5, 0.8, ..., 0.1]  (512 numbers)   │
    │                                                         │
    │ Larger d_model:                                         │
    │     More expressive (can capture more meaning)          │
    │     More parameters (slower, more memory)               │
    │                                                         │
    │ Paper uses 512 (base model) or 1024 (big model)         │
    └─────────────────────────────────────────────────────────┘
    
    datasource: Which dataset to use
    ┌─────────────────────────────────────────────────────────┐
    │ Popular translation datasets:                           │
    │                                                         │
    │ - opus_books: Books translated to many languages        │
    │   Example: Harry Potter in English → French             │
    │   Size: ~millions of sentence pairs                     │
    │                                                         │
    │ - wmt14: Workshop on Machine Translation dataset        │
    │ - ted_talks: TED talk transcripts                       │
    │ - multi30k: Image captions in multiple languages        │
    │                                                         │
    │ These come from HuggingFace datasets library            │
    └─────────────────────────────────────────────────────────┘
    
    lang_src & lang_tgt: Source and target languages
    ┌─────────────────────────────────────────────────────────┐
    │ Language codes (ISO 639-1):                             │
    │   en = English                                          │
    │   it = Italian                                          │
    │   fr = French                                           │
    │   de = German                                           │
    │   es = Spanish                                          │
    │   etc.                                                  │
    │                                                         │
    │ Example: "en" → "fr" means English to French            │
    └─────────────────────────────────────────────────────────┘
    
    model_folder & model_basename: Where to save checkpoints
    ┌─────────────────────────────────────────────────────────┐
    │ During training, model is saved after each epoch        │
    │                                                         │
    │ Directory structure:                                    │
    │   opus_books_weights/                                   │
    │   ├── tmodel_00.pt  (after epoch 0)                     │
    │   ├── tmodel_01.pt  (after epoch 1)                     │
    │   ├── tmodel_02.pt  (after epoch 2)                     │
    │   └── ...                                               │
    │                                                         │
    │ Each .pt file contains:                                 │
    │   - Model weights                                       │
    │   - Optimizer state                                     │
    │   - Epoch number                                        │
    │   - Training step count                                 │
    └─────────────────────────────────────────────────────────┘
    
    preload: Which checkpoint to resume from
    ┌─────────────────────────────────────────────────────────┐
    │ Options:                                                │
    │   None or "": Start training from scratch               │
    │   "latest": Resume from most recent checkpoint          │
    │   "10": Resume from specific epoch (e.g., epoch 10)     │
    │                                                         │
    │ Use case:                                               │
    │   Training interrupted? Set preload="latest"            │
    │   to continue where you left off!                       │
    └─────────────────────────────────────────────────────────┘
    
    tokenizer_file: Where to save/load tokenizers
    ┌─────────────────────────────────────────────────────────┐
    │ Pattern: "tokenizer_{0}.json"                           │
    │ {0} gets replaced with language code                    │
    │                                                         │
    │ Results in:                                             │
    │   tokenizer_en.json  (English tokenizer)                │
    │   tokenizer_it.json  (Italian tokenizer)                │
    │                                                         │
    │ Tokenizer contains:                                     │
    │   - Vocabulary (all words → IDs)                        │
    │   - Special tokens ([SOS], [EOS], [PAD], [UNK])         │
    └─────────────────────────────────────────────────────────┘
    
    experiment_name: TensorBoard logging directory
    ┌─────────────────────────────────────────────────────────┐
    │ Logs training metrics for visualization                 │
    │                                                         │
    │ Directory: runs/tmodel/                                 │
    │   Contains event files for TensorBoard                  │
    │                                                         │
    │ To view:                                                │
    │   tensorboard --logdir=runs/tmodel                      │
    │                                                         │
    │ Shows:                                                  │
    │   - Training loss over time                             │
    │   - Validation metrics (CER, WER, BLEU)                 │
    │   - Example translations                                │
    └─────────────────────────────────────────────────────────┘
    """
    
    return {
        # ==================================================================
        # TRAINING HYPERPARAMETERS
        # ==================================================================
        "batch_size": 8,           # Number of sentence pairs per batch
                                   # Adjust based on GPU memory:
                                   # - 8-16 for 8GB GPU
                                   # - 32-64 for 16GB+ GPU
                                   # - 2-4 for CPU training
        
        "num_epochs": 20,          # How many times to see full dataset
                                   # More epochs = better learning (until overfitting)
                                   # 20 is reasonable for most datasets
        
        "lr": 10**-4,              # Learning rate = 0.0001
                                   # Adam optimizer default: 0.001
                                   # 10**-4 is more conservative and stable
        
        "seq_len": 350,            # Maximum sequence length (tokens)
                                   # Based on dataset statistics
                                   # Check training output for max lengths
                                   # Should be >= 95% of your sentences
        
        "d_model": 512,            # Model dimension (paper default)
                                   # All embeddings, attention outputs have this size
                                   # Larger = more capacity, more memory
                                   # 512 (base) or 1024 (big) from paper
        
        # ==================================================================
        # DATASET CONFIGURATION
        # ==================================================================
        "datasource": 'opus_books',  # HuggingFace dataset name
                                     # Popular options:
                                     # - 'opus_books': Book translations
                                     # - 'wmt14': News translations
                                     # - 'ted_talks': TED talk subtitles
        
        "lang_src": "en",          # Source language code (English)
        "lang_tgt": "fr",          # Target language code (French)
                                   # Change these for different language pairs
                                   # Must be supported by your dataset
        
        # ==================================================================
        # FILE PATHS AND PERSISTENCE
        # ==================================================================
        "model_folder": "weights",       # Folder name for model checkpoints
        "model_basename": "tmodel_",     # Prefix for checkpoint files
                                         # Results in: tmodel_00.pt, tmodel_01.pt, etc.
        
        "preload": "latest",       # Which checkpoint to load at start
                                   # Options:
                                   # - "latest": Most recent checkpoint
                                   # - "05": Specific epoch number
                                   # - None or "": Start from scratch
        
        "tokenizer_file": "tokenizer_{0}.json",  # Pattern for tokenizer files
                                                 # {0} replaced with lang code
        
        "experiment_name": "runs/tmodel"  # TensorBoard log directory
                                          # Each run creates timestamped subfolder
    }


def get_weights_file_path(config, epoch: str):
    """
    Generate full path to model checkpoint for a specific epoch
    
    PURPOSE:
    --------
    Construct consistent file paths for saving/loading model checkpoints
    
    EXAMPLE:
    --------
    config = get_config()
    path = get_weights_file_path(config, "05")
    
    Returns: "./opus_books_weights/tmodel_05.pt"
    
    PATH BREAKDOWN:
    ---------------
    .                     ← Current directory
    └── opus_books_weights/    ← {datasource}_{model_folder}
        └── tmodel_05.pt       ← {model_basename}{epoch}.pt
    
    USED WHEN:
    ----------
    - Saving checkpoint after each epoch
    - Loading specific checkpoint to resume training
    - Evaluating a specific trained model
    
    Args:
        config: Configuration dictionary
        epoch: Epoch number as string (e.g., "05", "10", "19")
    
    Returns:
        Full path string to checkpoint file
    """
    # Combine datasource and model_folder
    # Example: "opus_books" + "_" + "weights" = "opus_books_weights"
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    
    # Create filename with epoch
    # Example: "tmodel_" + "05" + ".pt" = "tmodel_05.pt"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    
    # Combine into full path
    # Path('.') = current directory
    # Example: "./opus_books_weights/tmodel_05.pt"
    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(config):
    """
    Find the most recent checkpoint file in the weights folder
    
    PURPOSE:
    --------
    Automatically resume training from where you left off
    No need to manually specify which epoch to load
    
    HOW IT WORKS:
    -------------
    1. Look in weights folder for all checkpoint files
    2. Sort them alphabetically (which sorts by epoch number)
    3. Return the last one (most recent)
    
    EXAMPLE:
    --------
    Weights folder contains:
        opus_books_weights/
        ├── tmodel_00.pt
        ├── tmodel_01.pt
        ├── tmodel_02.pt
        └── tmodel_03.pt
    
    This function returns: "./opus_books_weights/tmodel_03.pt"
    
    SORTING LOGIC:
    --------------
    Files are sorted alphabetically:
        tmodel_00.pt
        tmodel_01.pt
        tmodel_02.pt
        tmodel_03.pt  ← Latest (last in sorted order)
    
    This works because epoch numbers are zero-padded:
        "05" comes before "10" (alphabetically correct!)
        "5" would come AFTER "10" (alphabetically wrong!)
    
    USE CASES:
    ----------
    1. Training interrupted (power outage, error, etc.)
       Set preload="latest" to resume automatically
    
    2. Continue training for more epochs
       Just increase num_epochs and run again
    
    3. Don't remember which epoch you trained to
       "latest" finds it for you
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Full path to latest checkpoint, or None if no checkpoints exist
    """
    # Get weights folder path
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    
    # Create glob pattern to match all checkpoint files
    # Example: "tmodel_*" matches tmodel_00.pt, tmodel_01.pt, etc.
    # The * is a wildcard that matches anything
    model_filename = f"{config['model_basename']}*"
    
    # Find all files matching the pattern
    # glob returns an iterator of Path objects
    # Example results: [tmodel_00.pt, tmodel_01.pt, tmodel_02.pt]
    weights_files = list(Path(model_folder).glob(model_filename))
    
    # Check if any checkpoints exist
    if len(weights_files) == 0:
        return None  # No checkpoints found (first time training)
    
    # Sort files alphabetically
    # Because epochs are zero-padded (00, 01, 02...),
    # alphabetical sorting gives chronological order
    weights_files.sort()
    
    # Return the last file (most recent epoch)
    # weights_files[-1] gets last element
    return str(weights_files[-1])


"""
EXAMPLE USAGE SCENARIOS
========================

SCENARIO 1: Starting fresh training
------------------------------------
config = get_config()
# All defaults, preload=None → Trains from scratch
# Creates: opus_books_weights/tmodel_00.pt, tmodel_01.pt, ...

SCENARIO 2: Resuming interrupted training
------------------------------------------
config = get_config()
config['preload'] = 'latest'
# Finds most recent checkpoint
# Continues training from that epoch

SCENARIO 3: Training different language pair
---------------------------------------------
config = get_config()
config['lang_src'] = 'en'
config['lang_tgt'] = 'fr'  # Change to French
# Creates separate folder: opus_books_weights/
# Separate tokenizers: tokenizer_en.json, tokenizer_fr.json

SCENARIO 4: Experimenting with hyperparameters
-----------------------------------------------
config = get_config()
config['batch_size'] = 16      # Try larger batches
config['lr'] = 10**-5          # Try slower learning
config['experiment_name'] = 'runs/experiment2'  # New log folder
# Can compare results in TensorBoard!

SCENARIO 5: Loading specific checkpoint for inference
------------------------------------------------------
config = get_config()
checkpoint_path = get_weights_file_path(config, "15")
# Loads: opus_books_weights/tmodel_15.pt
# Use this model for translation

SCENARIO 6: GPU memory issues
------------------------------
config = get_config()
config['batch_size'] = 4       # Reduce batch size
config['seq_len'] = 256        # Reduce sequence length
config['d_model'] = 256        # Reduce model size
# All reduce memory usage

FILE ORGANIZATION
=================

After training, your directory looks like:

project/
├── config.py
├── model.py
├── dataset.py
├── train.py
├── opus_books_weights/           ← Model checkpoints
│   ├── tmodel_00.pt
│   ├── tmodel_01.pt
│   └── ...
├── tokenizer_en.json             ← English tokenizer
├── tokenizer_it.json             ← Italian tokenizer
└── runs/                         ← TensorBoard logs
    └── tmodel/
        └── events.out.tfevents...

TYPICAL HYPERPARAMETER VALUES
==============================

Small model (fast training, less accurate):
    batch_size: 16-32
    d_model: 256
    num_epochs: 10
    seq_len: 128

Medium model (balanced):
    batch_size: 8-16
    d_model: 512        ← Paper default
    num_epochs: 20
    seq_len: 350

Large model (best quality, slow training):
    batch_size: 4-8
    d_model: 1024
    num_epochs: 30+
    seq_len: 512

ADJUSTING FOR YOUR HARDWARE
============================

8GB GPU:
    batch_size: 4-8
    d_model: 512
    seq_len: 256-350

16GB+ GPU:
    batch_size: 16-32
    d_model: 512-1024
    seq_len: 512

CPU only (slow!):
    batch_size: 1-2
    d_model: 256-512
    seq_len: 128
    Consider using smaller dataset
"""