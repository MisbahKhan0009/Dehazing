---
marp: true
theme: default
class: lead
paginate: true
backgroundColor: #ffffff
backgroundImage: url('https://marp.app/assets/hero-background.svg')
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
    padding: 40px;
  }
  h1 {
    color: #2c3e50;
  }
  h2 {
    color: #34495e;
  }
  strong {
    color: #e74c3c;
  }
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
---

# **AI-Driven Dehazing and Denoising of Echocardiographic Images**

## Using GANs, Diffusion Models, and Mamba-SSM

**CSE 498 Project**
*University of South Carolina*

---

<!-- Research Gap & Objective (3 Slides) -->

# **The Research Gap**

## Limitations of Current Echocardiography

- **Diagnostic Gold Standard**: Echocardiography is essential for diagnosing heart failure, valve disorders, and congenital defects.
- **The Problem**: Image quality is frequently compromised by:
    - Acoustic impedance mismatches.
    - Patient body habitus (obesity).
    - Environmental noise and artifacts.
- **Consequence**: **10-15%** of patients are classified as **"Difficult-to-Image"**.
    - Conventional imaging fails to provide clear views.
    - Traditional noise reduction techniques (filtering) blur important structural details.

---

# **The "Difficult-to-Image" Challenge**

## Why manual intervention isn't enough

- **Current Clinical Workarounds**:
    1. Repositioning the patient (physically demanding).
    2. Using alternative imaging windows (limited view).
    3. Contrast agents (invasive, costly).
- **Time Criticality**: In emergency settings, these delays are unacceptable.
- **Research Gap**: While Deep Learning shines in natural images, specific solutions for **Echocardiography** are limited.
    - Natural image models don't account for speckle noise nature.
    - Lack of specialized datasets with "ground truth" reference frames.

---

# **Our Objective**

## A Specialized Deep Learning Framework

To develop a comprehensive framework for **simultaneous dehazing and denoising** of echocardiographic images.

**Key Goals:**
1. **Restore Visibility**: Recover structural details in degraded images.
2. **Preserve Clinical Features**: Ensure wall motion and valve structures are not halluncinated or smoothed out.
3. **Compare Architectures**: Rigorously benchmark three cutting-edge approaches:
    - **GANs** (Generative Adversarial Networks)
    - **Diffusion Models** (Probabilistic formulation)
    - **Mamba SSM** (State Space Models - *Novel Application*)

---

<!-- Dataset Analysis (5 Slides) -->

# **Dataset Overview**

## Source: Grand Challenge 2025

- **Origin**: Sourced from the [Dehazing Echo 2025 Grand Challenge](https://dehazingecho2025.grand-challenge.org/).
- **Composition**: A curated set of high-quality and degraded echocardiograms.
- **Stats**:
    - **Total Patients**: 75
    - **Clean Images**: 4,376 (Reference high-quality)
    - **Noisy Images**: 2,324 (Simulated degradation)
- **Clinical Relevance**: Represents real-world variations in patient anatomy.

---

# **Data Distribution & Structure**

## Clean vs. Noisy

The dataset reflects the "Difficult-to-Image" reality:

- **Patients 1-75**: All have Clean (High Quality) images.
- **Patients 1-40**: A subset selected to represent "Difficult" cases, containing corresponding **Noisy** images.
- **Ratio**:
    - **Clean-only**: ~2,000 images (Training baseline)
    - **Paired (Clean-Noisy)**: ~2,300 images (Supervised learning)

---

# **Simulating the Challenge**

## How "Noisy" is Defined

The noisy dataset isn't just random Gaussian noise. It simulates acoustic degradation:

- **Speckle Noise**: Multiplicative noise inherent to ultrasound.
- **Low Contrast**: Reduced dynamic range.
- **Haze/Blur**: Simulating scattering of ultrasound waves in fatty tissue.

*Place comparative image here: Clean vs Noisy side-by-side*
<!-- Insert image from Dataset/clean vs Dataset/noisy -->

---

# **Evaluation Structure: Triplet Data**

To ensure clinical validity, we use **Triplets** for evaluation:

1. **Clean Image**: Ground Truth.
2. **Noisy Image**: Input to the model.
3. **ROI (Region of Interest) Mask**:
    - Provided for specific frames (e.g., Frame 1, 11, 21).
    - Focuses evaluation on key cardiac structures (Valves, Ventricular Walls).
    - Allows computation of **CNR (Contrast-to-Noise Ratio)**.

---

# **Dataset Statistics**

<div class="columns">

<div>

### Volume
- **Total Frames**: ~6,700
- **Frames per Patient**: 60
- **Cycle Coverage**: Full cardiac cycles included.

</div>

<div>

### Annotations
- **ROI Frequency**: Every ~10 frames.
- **Purpose**: Targeted evaluation avoids biasing metrics with background noise (regions outside the scanning sector).

</div>

</div>

---

<!-- Models: UNet, GAN, Diffusion (10 Slides) -->

# **Model 1: U-Net Architecture**

## The Medical Imaging Baseline

- **Architecture**: Encoder-Decoder with Skip Connections.
- **Mechanism**:
    - **Encoder**: Captures context (What is present?). Downsamples image.
    - **Decoder**: Localization (Where is it?). Upsamples to original resolution.
    - **Skip Connections**: Pass fine-grained details directly from Encoder to Decoder, preventing loss of resolution.
- **Role**: Serves as our "foundation" model and the Generator for GAN/Diffusion.

---

# **U-Net Performance**

- **Strengths**: 
    - Fast inference.
    - Good at removing global noise.
- **Weaknesses**: 
    - Produces "smooth" or blurry outputs.
    - Struggles with high-frequency texture (speckle patterns).
- **Result**: Significant improvement over traditional filtering, but lacks "crispness".

---

# **Model 2: GAN (Generative Adversarial Network)**

## Adversarial Training

- **Generator (U-Net)**: Tries to create a dehazed image that looks real.
- **Discriminator (PatchGAN)**: Tries to distinguish between the *Generated* dehazed image and the *Real* clean image.
- **Objective**: Min-Max Game.
    - The Generator is forced to create realistic textures to fool the Discriminator.

---

# **GAN Architecture Details**

- **PatchGAN Discriminator**:
    - Instead of classifying the whole image as Fake/Real, it classifies small **N x N patches**.
    - This enforces high-frequency consistency (sharp edges).
- **Loss Function**:
    - $L_{GAN}$ (Adversarial Loss) + $\lambda L_{L1}$ (Pixel-wise Loss).
    - Balances "looking real" with "matching the ground truth".

---

# **GAN Results**

- **Observation**:
    - Much sharper than standard U-Net.
    - Edges of heart walls are more defined.
- **Trade-off**:
    - Can introduce slight artifacts if training is unstable.
    - Harder to converge.
- **Verdict**: Excellent for perceptual quality.

---

# **Model 3: Diffusion Models**

## Probabilistic Denoising

- **Concept**: Treat dehazing as a gradual denoising process.
- **Forward Process**: Gradually add noise to a clean image until it is pure Gaussian noise (during training).
- **Reverse Process**: Train a Neural Network to predict the noise added at each step and remove it.
- **Our Approach**: **Conditional Diffusion**.
    - We condition the generation on the "Noisy/Hazy" input image.
    - The model effectively "guides" the random noise into a clean version of the input.

---

# **Diffusion Architecture**

- **Backbone**: Modified U-Net with Attention mechanisms.
- **Time Embedding**: The model knows which "timestep" of denoising it is performing (coarse vs. fine details).
- **Scheduler**: Controls how fast noise is removed (Linear vs Cosine schedules).
- **Iterations**: Requires multiple passes (e.g., 1000 steps) to generate one image.

---

# **Diffusion Results**

- **Quality**: State-of-the-Art perceptual quality.
- **Characteristics**:
    - Best at preserving complex textures.
    - No mode collapse (unlike GANs).
- **Cons**: Slow inference time (seconds vs milliseconds for U-Net/GAN).

---

# **Why Three Models?**

## Comparative Analysis

| Feature | U-Net | GAN | Diffusion |
| :--- | :--- | :--- | :--- |
| **Speed** | Fast | Fast | Slow |
| **Sharpness** | Low | High | Very High |
| **Stability** | Stable | Unstable | Stable |
| **Texture** | Smooth | Realistic | Highly Realistic |

---

# **Architecture Summary**

- **U-Net**: The reliable workhorse. Clean but blurry.
- **GAN**: The artist. Sharp, detailed, but temperamental.
- **Diffusion**: The perfectionist. Best quality, but takes time.

**Next**: We introduce a fourth contender that aims to combine the speed of U-Net with the global context of Transformers.

---

<!-- Mamba SSM (10 Slides) -->

# **Introducing Mamba SSM**

## Selective State Space Models

**The Challenge with Transformers**: 
- Transformers (like in ChatGPT) are great at global context but scale quadratically $O(N^2)$ with image size. Too heavy for high-res medical video.
- CNNs (U-Net) only see local windows.

**The Solution**: **Mamba**.
- A new architecture based on State Space Models (SSM).
- **Linear Scaling** $O(N)$.
- Captures global context like a Transformer, runs fast like a CNN.

---

# **Mamba Architecture: Our Implementation**

## Mamba-UNet

We replaced standard convolutional bottlenecks with **Vision Mamba Layers**.

- **Structure**:
    1. **Encoder**: Conv Blocks extract features + Mamba layers mix information spatially.
    2. **Bottleneck**: Deep Mamba stacks to understand the whole heart geometry.
    3. **Decoder**: Reconstructs the image.

---

# **How It Works: The Data Flow**

### Step 1: Flattening
The 2D Image Feature Map $(B, C, H, W)$ is flattened into a 1D sequence $(B, L, C)$ where $L = H \times W$.

### Step 2: Linear Scanning
Unlike Attention (which compares every pixel to every pixel), Mamba **scans** the sequence effectively using a recurrent state.
- It remembers "past" pixels to inform "current" pixels.
- Bi-directional scanning ensures pixels know about neighbors in all directions.

---

# **The Selection Mechanism**

## "Selective" State Spaces

Standard SSMs are rigid. Mamba introduces **Selection**:
- The model can decide to **"remember"** or **"ignore"** information at each step based on the input.
- **In Dehazing**:
    - *Ignore*: Random noise patterns/haze.
    - *Remember*: Continuous structures like the heart wall.
- This dynamic content-filtering is perfect for separating signal (anatomy) from noise (haze).

---

# **Mamba Processing Pipeline**

1. **Input Hazy Image** $\rightarrow$ ConvNet Feature Extraction.
2. **Global Mixing**: Mamba layers sweep across the image features, connecting distant parts of the heart wall that are obscured by haze.
3. **Reconstruction**: The cleaned features are projected back to 2D.
4. **Skip Connections**: Original details are added back to preserve resolution.
5. **Output**: Clean Dehazed Image.

---

# **Mamba Results: Metrics History**

## Training Dynamics (100 Epochs)

- **Convergence**: Extremely stable training curve.
- **Efficiency**: Reached high PSNR faster than pure Transformers.
- **Final Validation Stats**:
    - **PSNR**: **31.82 dB** (High fidelity).
    - **SSIM**: **0.9536** (Structural match is near perfect).
    - **Loss**: Dropped from 0.46 (Epoch 1) to 0.033 (Epoch 100).

---

# **Quantitative Superiority**

## Comparison Table

| Metric | Baseline (Noisy) | U-Net | GAN | **Mamba (Ours)** |
| :--- | :--- | :--- | :--- | :--- |
| **PSNR** (dB) | 15.44 | ~28.5 | ~29.8 | **31.82** |
| **SSIM** | 0.45 | 0.82 | 0.89 | **0.95** |
| **Inference/Frame** | N/A | 15ms | 15ms | **18ms** |

*Note: Mamba achieves near-GAN quality with much better stability and no hallucination artifacts.*

---

# **Clinical Validation Metrics**

Standard metrics (PSNR) aren't enough for medical usage. We used:

### CNR (Contrast-to-Noise Ratio)
- **Result**: Improved from 0.12 $\rightarrow$ **0.266**.
- **Meaning**: The contrast between the heart tissue and the blood pool is doubled.

### gCNR (Generalized CNR)
- **Result**: **0.325**.
- **Meaning**: Validates that histogram overlaps between signal and background are minimized.

---

# **Visual Results: Mamba**

## Before vs. After

*Place comparative visualization here*
*(Left: Noisy Input | Center: Mamba Output | Right: Ground Truth)*

**Observations**:
- The "haze" clouding the ventricle is removed.
- Valve leaflet edges are sharp.
- No "checkerboard" artifacts (common in GANs).

---

# **Conclusion & Future of Mamba**

## Why Mamba Wins

1. **Global Context**: It "sees" the whole heart at once, ensuring geometric consistency.
2. **Speed**: Linear complexity means it can potentially run in real-time on ultrasound machines.
3. **Accuracy**: 31.8+ dB PSNR proves it restores data faithfully.

**Future Work**:
- Deploying Mamba-UNet on edge devices (portable ultrasound).
- Exploring 3D Mamba for volumetric echo data.

---
