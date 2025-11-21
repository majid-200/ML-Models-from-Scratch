# Project: Building a GPT-like Model

This project focuses on constructing a decoder-only transformer model from the ground up, similar in spirit to OpenAI's GPT-2. It is trained on the works of Shakespeare using simple **character-level tokenization** to demonstrate the core mechanics of the transformer architecture.

The project starts with a simple Bigram model as a baseline and progressively builds up to a full GPT implementation.

### Key Concepts Covered
-   Character-level Tokenization
-   Bigram Language Models
-   Transformer Architecture
-   Self-Attention Mechanism & Multi-Head Attention
-   Positional Embeddings
-   Layer Normalization and Residual Connections
-   Autoregressive Text Generation

---

## 📂 File Breakdown

#### 📜 `bigram.py`
A "Hello World" of language modeling. This script implements a simple Bigram model where the prediction for the next character depends only on the preceding character. It establishes the data loading and evaluation pipeline.

#### 🤖 `GPT.py` & `GPT-Scratch.ipynb`
The core of this project.
-   `GPT-Scratch.ipynb`: A narrative, cell-by-cell walkthrough explaining the "why" behind each component, from self-attention to the final model assembly.
-   `GPT.py`: A clean, runnable Python script of the final, complete model.

The model built here includes all the fundamental components of a transformer block:
-   **Multi-Head Self-Attention**: Allows tokens to communicate with each other and weigh the importance of different tokens in the context.
-   **Feed-Forward Network**: A simple MLP applied to each token position independently, adding computational depth.
-   **Residual Connections & Layer Normalization**: Critical for stabilizing training in deep networks by preventing vanishing/exploding gradients.

---

## ⚙️ Setup and Usage

1.  **Install dependencies:**
    ```bash
    pip install torch torchvision torchaudio notebook
    ```

2.  **Download the Dataset:** The `TinyShakespear.txt` dataset is required for training.
    ```bash
    # From the root directory of the repository:
    wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O gpt-from-scratch/TinyShakespear.txt
    ```

3.  **Explore the Notebook:** Launch Jupyter (`jupyter notebook`) and open `GPT-Scratch.ipynb` to follow the step-by-step guide.

4.  **Train the Model:** To run the full training process, execute the Python script:
    ```bash
    python gpt-from-scratch/GPT.py
    ```

---

## Acknowledgements
This project is heavily inspired by and based on the excellent tutorial by Andrej Karpathy:
- **[Let's build GPT: from scratch, in code, spelled out.](https://youtu.be/kCc8FmEb1nY?feature=shared)**
