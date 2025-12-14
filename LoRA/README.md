# Project: LoRA (Low-Rank Adaptation) from Scratch

This project is a comprehensive, two-part guide to understanding and implementing Low-Rank Adaptation (LoRA), one of the most popular Parameter-Efficient Fine-Tuning (PEFT) techniques.

The project is structured into two Jupyter notebooks:
1.  **`SVD.ipynb`**: Explores the mathematical foundation of LoRA through Singular Value Decomposition (SVD) and low-rank matrix factorization.
2.  **`LoRA.ipynb`**: Provides a practical, hands-on implementation of LoRA to fine-tune a neural network on a specific task.

## Key Concepts Covered

-   **Singular Value Decomposition (SVD):** The core mathematical tool for decomposing matrices.
-   **Low-Rank Matrix Factorization:** How to approximate a large matrix with two smaller, low-rank matrices.
-   **Parameter-Efficient Fine-Tuning (PEFT):** The strategy of fine-tuning only a small subset of a model's parameters.
-   **LoRA Implementation:** How to inject trainable low-rank adapters (`A` and `B` matrices) into a frozen pre-trained model.
-   **Freezing Base Model Weights:** The critical step of keeping the original model parameters unchanged while only training the new adapters.
-   **PyTorch Parametrization:** Using `torch.nn.utils.parametrize` to elegantly and non-invasively add the LoRA computation to existing layers.

---

## 📂 Notebooks Overview

It is highly recommended to go through the notebooks in order to build a strong foundational understanding.

### 1. `SVD.ipynb` - The Mathematical Foundation

This notebook answers the question: **"Why does LoRA work?"** It walks through the concept of low-rank approximation, which is the cornerstone of LoRA's efficiency.

**You will learn:**
-   How to create a rank-deficient matrix, mimicking the "low-rank hypothesis" of model weight updates.
-   How to use SVD to decompose this matrix into its fundamental components (`U`, `Σ`, `V^T`).
-   How to reconstruct the original matrix perfectly using two smaller, low-rank matrices (`B` and `A`), just like in LoRA.
-   How this decomposition dramatically reduces the number of parameters required while achieving the same computational result.

### 2. `LoRA.ipynb` - The Practical Implementation

This notebook answers the question: **"How do you apply LoRA to a real neural network?"** It demonstrates the complete workflow of fine-tuning a pre-trained model using LoRA.

**The steps include:**
1.  **Pre-training a Base Model:** An intentionally oversized neural network is trained on the full MNIST dataset to simulate a large, pre-trained base model.
2.  **Injecting LoRA Adapters:** A `LoRAParametrization` class is defined and applied to the linear layers of the network, adding the trainable `A` and `B` matrices without modifying the original weights.
3.  **Freezing and Fine-tuning:** All original model weights are frozen, and only the newly added LoRA parameters (a tiny fraction of the total) are trained on a specific subset of the data (only images of the digit "9").
4.  **Evaluation and Verification:** The model's performance is tested with LoRA enabled and disabled to demonstrate that we successfully improved performance on the target task (recognizing "9") while preserving the model's general knowledge.

---

## ⚙️ How to Use

1.  **Installation:**
    First, set up a virtual environment and install the required dependencies from within this project folder.
    ```bash
    # Navigate to this project's folder
    cd LoRA

    # Install dependencies
    pip install torch torchvision numpy matplotlib tqdm notebook
    ```

2.  **Run the Notebooks:**
    Launch Jupyter and run the notebooks. It is recommended to start with `SVD.ipynb` to understand the theory before moving on to `LoRA.ipynb` for the implementation.

    ```bash
    jupyter notebook
    ```
    The notebooks are self-contained and explain each step in the markdown cells.

---

## Acknowledgements
This project is heavily inspired by and based on the excellent tutorial by Umar Jamil:
- **[LoRA: Low-Rank Adaptation of Large Language Models - Explained visually + PyTorch code from scratch](https://youtu.be/PXWYUTMt-AU?si=5HGoCa0C8c37d4Yb)**
