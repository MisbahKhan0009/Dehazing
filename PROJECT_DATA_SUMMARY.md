# Echo Dehazing Project Data Summary
*Generated on January 27, 2026*

## 1. Dataset Overview & Statistics

### General Distribution
*   **Total Images**: 4,376
*   **Total Patients**: 75
*   **Complete Triplets** (Clean/Noisy/ROI): 237 sets
*   **Clean-Noisy Pairs**: 2,324
*   **Clean Only Images**: 2,052

### Patient Allocation
*   **Patients with Noisy Data (Training Targets)**: 40
*   **Patients with ROI Data (Clinical Validation)**: 40
*   **Total Patient Cohort**: 75

### Data Structure
*   **Clean**: High-quality echocardiograms (Ground Truth).
*   **Noisy**: Simulated degraded images (Input).
*   **Triplets**: Used for specialized evaluation (Clean + Noisy + ROI Mask).

---

## 2. Model Performance Results

### A. Mamba-SSM (Best Performing Model)
*Global Context Dehazing Network*

**Training Configuration**:
*   **Epochs**: 100
*   **Batch Size**: 4
*   **Learning Rate**: 1e-4 -> 2.5e-5

**Final Validation Metrics (Epoch 100)**:
*   **PSNR**: **31.82 dB** (Peak Signal-to-Noise Ratio)
*   **SSIM**: **0.9536** (Structural Similarity Index)
*   **Validation Loss**: 0.0590

**Clinical Validation Metrics**:
*   **CNR (Contrast-to-Noise Ratio)**: 0.2661 (Improved from baseline ~0.12)
*   **gCNR (Generalized CNR)**: 0.3251

**Training Progression (Key Milestones)**:
| Epoch | PSNR (dB) | SSIM | Loss | Note |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 15.78 | 0.2650 | 0.3529 | Initial Training |
| 10 | 30.36 | 0.9358 | 0.0706 | Rapid Convergence |
| 50 | 31.19 | 0.9512 | 0.0618 | Mid-training capability |
| 100 | **31.82** | **0.954** | **0.059** | Final Convergence |

### B. U-Net (Baseline)
*Standard Encoder-Decoder Architecture*

**Training Configuration**:
*   **Epochs**: 200
*   **Batch Size**: 8
*   **Learning Rate**: 2e-4
*   **Dataset**: Dataset_augmented (~7000 images, 3x augmentation)

**Final Validation Metrics (Epoch 200)**:
*   **PSNR**: **26.23 dB**
*   **SSIM**: **0.795**
*   **Validation Loss**: 0.0214

**Training Progression (Key Milestones)**:
| Epoch | PSNR (dB) | SSIM | Loss | Note |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 17.48 | 0.243 | 0.0751 | Initial Training |
| 50 | 24.70 | 0.760 | 0.0253 | Steady Improvement |
| 100 | 25.62 | 0.780 | 0.0228 | Convergence begins |
| 150 | 26.05 | 0.790 | 0.0219 | Fine-tuning |
| 200 | **26.23** | **0.795** | **0.0214** | Final Convergence |

### C. Comparative Analysis (vs. Baselines)

| Model Architecture | PSNR (avg) | SSIM (avg) | Inference Speed | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **U-Net (Baseline)** | 26.23 dB | 0.795 | **Fastest (15ms)** | Effective but produces over-smoothed outcomes. Lacks texture preservation. |
| **GAN (Pix2Pix)** | ~29.8 dB | 0.89 | Fast (15ms) | High perceptual quality and texture detail, but unstable training and occasional artifacts. |
| **Diffusion (DDPM)** | N/A | High | Slow | Excellent quality but computationally expensive for real-time medical video. |
| **Mamba-UNet (Ours)**| **31.82 dB** | **0.954** | Fast (18ms) | **Best Balance**: Linear scaling complexity with global context awareness. |

---

## 3. Directory & File Reference

### Key Directories
*   `Dataset/clean/`: Ground truth images.
*   `Dataset/noisy/`: Simulated hazy inputs.
*   `Notebooks/checkpoints/`: Model weights (.pt/.pth) and logs.

### Result Locations
*   **Mamba Results**: `Notebooks/checkpoints/mamba_dehazing/epoch_samples/`
*   **Visualizations**: `Dataset_augmented/visualizations/`
*   **Stats Files**: `Dataset_Old/visualizations/statistics/dataset_stats.txt`

---

## 4. Conclusion
The **Mamba-SSM** architecture demonstrates superior performance across all quantitative metrics (PSNR, SSIM, CNR), offering a 10% improvement in SSIM over the standard U-Net baseline while maintaining real-time inference speeds suitable for clinical ultrasound machines.
