# Project: Implementing PaliGemma from Scratch

This project is a complete, from-scratch implementation of Google's PaliGemma, a powerful multimodal Vision-Language Model (VLM). The goal is to deconstruct and understand the core components that allow a model to process both images and text to generate relevant textual descriptions or answers.

The implementation is built in PyTorch and is heavily annotated to explain the flow of data and the purpose of each architectural choice.

## 🏛️ Core Architectural Components

PaliGemma's architecture is a fusion of two powerful models bridged by a simple projector:

1.  **SigLIP Vision Transformer:** A vision encoder that processes an image into a sequence of patch embeddings.
2.  **Gemma Language Model:** A decoder-only transformer that processes a sequence of tokens. It features optimizations like Grouped-Query Attention (GQA), Rotary Positional Embeddings (RoPE), and RMSNorm.
3.  **Multimodal Projector:** A simple linear layer that translates the visual features from the SigLIP's embedding space into the Gemma's language embedding space.
4.  **Embedding Merger:** The key logic that replaces special `<image>` tokens in the input prompt with the actual projected vision features, creating a unified sequence for the LLM.

---

## 📂 File Breakdown

This project is organized into several key files, each responsible for a specific part of the model's architecture or functionality.

#### 👁️ `modeling_siglip.py`
This file contains the implementation of the **SigLIP Vision Transformer**. It takes a raw image, divides it into patches, and uses a stack of transformer encoder layers to create a rich, contextualized representation of the image.

#### 🧠 `modeling_gemma.py`
This is the heart of the project. It contains the implementation of the **Gemma language model**, the **multimodal projector**, and the final `PaliGemmaForConditionalGeneration` class that orchestrates the entire process. It includes detailed implementations of GQA, RoPE, RMSNorm, and the crucial KV Cache for efficient text generation.

#### 🖼️ `processing_paligemma.py`
This file contains the **PaliGemmaProcessor**, which is responsible for all input preprocessing. It handles resizing and normalizing images, and it formats text prompts by prepending the special `<image>` tokens, ensuring the data is in the exact format the model expects.

#### 🛠️ `utils.py`
A utility script containing the `load_hf_model` function. This function is essential for loading pretrained model weights from the Hugging Face Hub (in `safetensors` format) into our custom model implementation, allowing us to run inference with a fully trained model.

#### 🚀 `inference.py`
An end-to-end inference script that demonstrates how to use the model. It loads a pretrained model, processes an image and a text prompt, and generates a textual response using an autoregressive loop with KV caching.

---

## ⚙️ How to Use

1.  **Install dependencies:**
    ```bash
    pip install torch transformers sefetensors Pillow numpy fire
    ```

3.  **Run Inference:** Use the `inference.py` script to generate text from an image. You will need to provide a path to a model from the Hugging Face Hub, a text prompt, and a path to an image file.

    **Example Usage:**
    ```bash
    python inference.py \
        --model_path="google/paligemma-3b-pt-224" \
        --prompt="describe this image" \
        --image_file_path="/path/to/your/image.jpg" \
        --do_sample=False
    ```
    *The script will automatically download the model from Hugging Face the first time you run it.*

---

## Acknowledgements
This project is heavily inspired by and based on the excellent tutorial by Umar Jamil:
- **[Coding a Multimodal (Vision) Language Model from scratch in PyTorch with full explanation](https://www.youtube.com/watch?v=vAmKB7iPkWw&t=8310s)**
