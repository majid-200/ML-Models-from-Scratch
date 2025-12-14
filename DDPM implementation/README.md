# Project: Denoising Diffusion Probabilistic Model (DDPM) from Scratch

This project is a from-scratch implementation of the Denoising Diffusion Probabilistic Model (DDPM), based on the seminal 2020 paper by Ho et al. The goal is to build and train a generative model that can create images by learning to reverse a gradual noising process.

The implementation is built using PyTorch and PyTorch Lightning, with a strong emphasis on clear, commented code to explain each component of the diffusion process and the underlying U-Net architecture.

## Key Features

-   **Full DDPM Implementation:** Implements both the forward (noising) and reverse (denoising) processes as described in the paper.
-   **U-Net with Attention:** A modern U-Net architecture serves as the noise predictor, complete with ResNet blocks, self-attention layers, and time embeddings.
-   **Sinusoidal Time Embeddings:** The model is conditioned on the noise level (timestep `t`) using sinusoidal embeddings, allowing it to learn how to denoise at any stage.
-   **PyTorch Lightning:** The training loop is managed by PyTorch Lightning for clean, boilerplate-free code, automatic checkpointing, and easy hardware acceleration.
-   **TensorBoard Logging:** All training and validation metrics are logged to TensorBoard for easy monitoring.
-   **Visual GIF Generation:** After training, the script automatically generates a GIF that visualizes the entire reverse diffusion process, showing how the model transforms pure noise into a coherent image step-by-step.

---

## 🏛️ How It Works: The Diffusion Process

The core idea behind DDPMs is to master a two-step process:

### 1. The Forward Process (Fixed)
We gradually add a small amount of Gaussian noise to an image over a large number of timesteps (`T=1000`). This process is a fixed Markov chain that eventually transforms any image into pure, unstructured noise.

> **`x₀ (clean image)` → `x₁` → `x₂` → ... → `x_T (pure noise)`**

### 2. The Reverse Process (Learned)
The model's task is to learn the reverse of this process. It is trained to predict the noise that was added at any given timestep `t`. By repeatedly predicting and subtracting this noise, the model can start from a random noise image (`x_T`) and gradually denoise it back into a clean, realistic image (`x₀`).

> **`x_T (pure noise)` → `x_(T-1)` → ... → `x₁` → `x₀ (generated image)`**

The neural network that performs this noise prediction is a **U-Net**, which is particularly well-suited for image-to-image tasks.

---

## 📂 File Breakdown

-   **`model.py`**: Implements the main `DiffusionModel` class. It contains the logic for the noise schedule (`beta`, `alpha`, `alpha_bar`), the training loss calculation (Algorithm 1), and the sampling/denoising step (Algorithm 2).
-   **`modules.py`**: Contains the complete implementation of the **U-Net architecture**. This includes all the building blocks like ResNet blocks, self-attention, time embeddings, and up/downsampling layers.
-   **`data.py`**: Defines the `DiffSet` class, which prepares standard vision datasets (MNIST, FashionMNIST, CIFAR10) for training. It handles resizing images to a consistent 32x32 and normalizing pixel values to the `[-1, 1]` range, which is crucial for diffusion models.
-   **`train.py`**: The main executable script. It ties everything together by handling data loading, model initialization, training with PyTorch Lightning, and finally, generating a sample GIF to visualize the denoising process.

---

## ⚙️ How to Use

### 1. Installation
Set up a virtual environment and install the required dependencies.

```bash
# Navigate to this project folder
cd DDPM implementation

# Install dependencies
pip install torch torchvision pytorch-lightning imageio tqdm Pillow numpy einops tensorboard
```

### 2. Configuration
The main settings are controlled by the `get_config()` function inside `train.py`. You can easily change:
-   `dataset`: `"MNIST"`, `"FashionMNIST"`, or `"CIFAR10"`.
-   `max_epoch`: The number of epochs to train for.
-   `batch_size`: Adjust based on your GPU memory.
-   `load_model`: Set to `True` to resume training from a checkpoint.

### 3. Training the Model
Run the main training script. It will automatically download the specified dataset, create the model, and start training.

```bash
python train.py
```
Training logs and model checkpoints will be saved to the `lightning_logs/` directory.

### 4. Monitoring with TensorBoard
Track the training loss and other metrics in real-time.

```bash
tensorboard --logdir=lightning_logs
```
Open `http://localhost:6006/` in your browser to view the dashboard.

### 5. The Output: Denoising GIF
After training is complete, the script will automatically use the trained model to generate a 3x3 grid of new images and save the entire step-by-step denoising process as a GIF named `pred.gif` inside the versioned log directory (e.g., `lightning_logs/CIFAR10/version_0/`).

---

## Example Output

The final output is an animated GIF showcasing the reverse diffusion process, from noise to image.

[Denoising Process GIF](lightning_logs/CIFAR10/version_1/pred.gif)

---

## Acknowledgements
