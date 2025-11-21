# Building ML Models from Scratch

This repository documents my journey into building foundational machine learning models from scratch. The goal is to demystify the core components of modern architectures by implementing them step-by-step, primarily using PyTorch.

Each project is self-contained in its own folder with a dedicated README explaining the concepts, architecture, and setup instructions.

---

## 🚀 Projects

| Project                                                   | Description                                                                                              | Key Architectures & Concepts        |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| **GPT-like Model**                 | Build a complete, character-level Generative Pre-trained Transformer from the ground up.                 | Transformer, Self-Attention, Bigram |
| **Llama 3 Deep Dive** | Explore and implement the key architectural innovations in Meta's Llama 3 model.                           | RoPE, SwiGLU, GQA, BPE Tokenizer    |
| *(More model implementations will be added here...)*      |                                                                                                          |                                     |

---

## 🔧 General Setup

1.  **Clone the Repository:**
   
    ```bash
    git clone https://github.com/your-username/LLM-From-Scratch.git
    cd LLM-From-Scratch
    ```

2.  **Create a Virtual Environment:**
   
    It's highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**

    Install dependencies based on the project you want to run. 

For project-specific setup, like downloading datasets or running scripts, please see the `README.md` inside each project folder.
