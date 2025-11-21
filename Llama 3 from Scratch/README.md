# Project: Deconstructing Llama 3's Architecture

This project moves beyond the foundational GPT architecture to explore the specific, high-performance components used in Meta's Llama 3 model. The notebooks isolate and implement these key innovations from scratch.

### Key Concepts Covered
-   **Byte-Pair Encoding (BPE)**: A modern subword tokenization strategy.
-   **RMSNorm**: A simpler and more efficient alternative to LayerNorm.
-   **SwiGLU (Gated Linear Unit)**: An advanced feed-forward network.
-   **Rotary Positional Embeddings (RoPE)**: A sophisticated method for encoding positional information.
-   **Grouped-Query Attention (GQA)**: An optimized attention mechanism.
-   **QK Normalization**: An additional normalization step for training stability.

---

## 📂 File Breakdown

#### 🧠 `Llama_3_Attention.ipynb`
This notebook is a deep dive into the heart of the Llama 3 transformer block: its attention mechanism. It meticulously reconstructs the entire attention forward pass, explaining and implementing RoPE, QK Norm, and Grouped-Query Attention (GQA).

#### 🚀 `Llama_3_feed_forward.ipynb`
This notebook focuses on the other major component of the Llama 3 transformer block: the feed-forward network. It implements the **SwiGLU** variant and demonstrates the use of **RMSNorm** for pre-normalization, a key feature of the Llama architecture.

#### 🧩 `BPE_Scratch.ipynb` & `BPE_Scratch.py`
This notebook and script implement **Byte-Pair Encoding (BPE)** from scratch, a popular subword tokenization algorithm. It starts with a base vocabulary of individual characters and iteratively merges the most frequent adjacent pairs, creating a more efficient and semantically rich vocabulary. This is a crucial concept for understanding how LLMs handle vast and varied language data.

---

## ⚙️ Usage

1.  **Install dependencies:**
    ```bash
    pip install torch torchvision torchaudio notebook
    ```

2.  **Explore the Notebooks:** Launch Jupyter (`jupyter notebook`) and navigate into the `llama3-architecture-deep-dive` directory to explore the notebooks. Each notebook is self-contained and explains the concepts as they are implemented.

---

## Acknowledgements
This project's implementation of Llama 3 components is heavily inspired by the clear explanations from Vuk Rosić:
- **[Code Llama 3 From Scratch - Easy Math Explanations & Python Code](https://youtu.be/wcDV3l4CD14?feature=shared)**
