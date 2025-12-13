# Project: Transformer for Machine Translation

This project is a complete, from-scratch implementation of the original Transformer model, as introduced in the seminal paper **"Attention Is All You Need"** by Vaswani et al. The goal is to build a neural machine translation system that can translate text from a source language (e.g., English) to a target language (e.g., French).

The entire architecture is built using PyTorch and is heavily annotated to provide a clear, educational walkthrough of every component, from the input embeddings to the final prediction.

## Key Features

-   **Full Encoder-Decoder Architecture:** Implements the complete six-layer encoder and six-layer decoder stack as described in the paper.
-   **Multi-Head Self-Attention:** The core mechanism of the Transformer, allowing the model to weigh the importance of different words in a sequence.
-   **Positional Encoding:** Uses the original `sin`/`cos` positional encoding to give the model a sense of word order.
-   **Teacher Forcing:** Employs teacher forcing during training for faster convergence and stability.
-   **Greedy Decoding:** Includes a validation loop that uses greedy decoding to generate translations on unseen data, providing a clear view of the model's performance.
-   **Checkpointing:** Automatically saves and loads model checkpoints, allowing you to resume training from where you left off.
-   **TensorBoard Integration:** Logs training loss and validation metrics (CER, WER, BLEU) for easy visualization of the training process.
-   **Hugging Face Datasets:** Integrates with the `datasets` library to easily download and use standard translation datasets like `opus_books`.

---

## 🏛️ Architecture Deep Dive

This implementation faithfully recreates the components of the "Attention Is All You Need" architecture.

> **Overall Architecture:**
> `Input Sequence` → `[ENCODER]` → `Context Vector` → `[DECODER]` → `Output Sequence`

1.  **Input Embeddings & Positional Encoding:**
    -   Words are converted into dense vectors (`d_model=512`).
    -   Sinusoidal positional encodings are added to these vectors to inject information about the order of the words in the sequence.

2.  **The Encoder:**
    -   A stack of 6 identical layers. Each layer has two sub-layers:
        -   A **Multi-Head Self-Attention** mechanism that allows each word in the source sentence to attend to all other words in the source sentence.
        -   A simple, position-wise **Feed-Forward Network**.
    -   Residual connections and Layer Normalization are used around each sub-layer to stabilize training.

3.  **The Decoder:**
    -   Also a stack of 6 identical layers. Each layer has three sub-layers:
        -   A **Masked Multi-Head Self-Attention** mechanism. The mask ensures that when predicting a word, the model can only attend to previous words in the target sentence, preventing it from "cheating."
        -   A **Cross-Attention** mechanism where the queries come from the decoder, but the keys and values come from the output of the **Encoder**. This is the crucial step where the decoder looks at the source sentence to inform its translation.
        -   A position-wise **Feed-Forward Network**.
    -   Residual connections and Layer Normalization are also used here.

4.  **Final Projection Layer:**
    -   The output of the decoder (a vector of size `d_model`) is passed through a final linear layer and a softmax function to produce a probability distribution over the entire target vocabulary.

---

## 📂 File Breakdown

-   **`model.py`**: Contains the complete implementation of the Transformer architecture, including all sub-components like Multi-Head Attention, Positional Encoding, Encoder/Decoder blocks, and residual connections.
-   **`dataset.py`**: Defines the `BilingualDataset` class, which handles all data preprocessing. It tokenizes sentences, adds special tokens (`[SOS]`, `[EOS]`, `[PAD]`), creates the encoder/decoder inputs, and generates the necessary attention masks.
-   **`config.py`**: A centralized configuration file to manage all hyperparameters, file paths, and settings. This makes it easy to experiment without changing the main training code.
-   **`train.py`**: The main script that ties everything together. It handles data loading, model initialization, the complete training loop, validation, checkpointing, and logging to TensorBoard.

---

## ⚙️ How to Use

### 1. Installation
First, set up a virtual environment and install the required dependencies.

```bash
# Navigate to this project folder
cd Translation-Transformer

# Install dependencies
pip install torch datasets tokenizers torchmetrics tqdm tensorboard
```

### 2. Configuration
All training settings are controlled by `config.py`. Before training, you can modify this file to change:
-   **Dataset:** `datasource`, `lang_src`, `lang_tgt` (e.g., from `en-it` to `en-fr`).
-   **Hyperparameters:** `batch_size`, `num_epochs`, `lr`, `d_model`.
-   **Checkpointing:** `preload` to specify which model checkpoint to resume from (`'latest'` is the default).

### 3. Training the Model
Run the main training script. The script will automatically download the dataset, build the tokenizers (and save them for future use), and start the training process.

```bash
python train.py
```
The script will log progress to the console and save model checkpoints to the `weights/` directory after each epoch.

### 4. Monitoring with TensorBoard
You can monitor the training progress, including loss curves and validation metrics, using TensorBoard.

```bash
tensorboard --logdir=runs
```
Navigate to `http://localhost:6006/` in your browser to view the dashboard.

### 5. Inference and Translation
The `run_validation` function inside `train.py` provides a clear example of how to perform inference. It uses the `greedy_decode` function to translate sentences from the validation set and prints the source, target, and predicted translations to the console after each epoch.

---

## Example Hyperparameter Configuration

The `config.py` file allows for easy experimentation. Here are some key settings:

| Parameter         | Default Value       | Description                                                 |
| ----------------- | ------------------- | ----------------------------------------------------------- |
| `batch_size`      | `8`                 | Number of sentences per batch. Adjust based on GPU memory.  |
| `num_epochs`      | `20`                | Total number of training epochs.                            |
| `lr`              | `1e-4`              | The learning rate for the Adam optimizer.                   |
| `d_model`         | `512`               | The internal dimension of the model (as in the paper).      |
| `seq_len`         | `350`               | Maximum sequence length for padding/truncation.             |
| `datasource`      | `'opus_books'`      | The Hugging Face dataset to use.                            |
| `lang_src`/`lang_tgt` | `'en'` / `'it'` | The language pair for translation.                          |
| `preload`         | `'latest'`          | Resume training from the latest checkpoint. Set to `None` to start fresh. |
