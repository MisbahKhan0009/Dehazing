# Echocardiography Dehazing Using GANs

This project implements a deep learning approach for dehazing echocardiography images using Generative Adversarial Networks (GANs). The goal is to enhance the quality of ultrasound images captured from difficult-to-image subjects by removing noise and improving visibility.

## 🚀 Project Overview

Echocardiography is a crucial diagnostic tool in cardiology, but image quality can be significantly degraded by factors such as patient body habitus, acoustic shadowing, and poor acoustic windows. This project addresses these challenges by:

* **Training GANs** to predict clean images from noisy/hazy echocardiography data
* **Implementing advanced architectures** including U-Net generators and PatchGAN discriminators
* **Using medical-specific metrics** such as CNR (Contrast-to-Noise Ratio) and gCNR for evaluation
* **Providing comprehensive tools** for dataset analysis, training, and evaluation

## 📁 Project Structure

```
Dehazing/
├── README.md
├── .gitignore
├── .venv/
│
├── Dataset/
│   ├── clean/
│   ├── noisy/
│   ├── noisy_roi/
│   ├── dataset_mapping.csv
│   ├── Dataset_Analysis_Report.md
│   ├── FolderDescriptions.txt
│   ├── Scripts/
│   │   ├── analyze_dataset.py
│   │   ├── preview_visualizations.py
│   │   └── visualize_dataset.py
│   └── visualizations/
│
├── gan_training/
│   ├── config/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── scripts/
│   ├── checkpoints/
│   ├── logs/
│   ├── requirements.txt
│   ├── run_training.py
│   ├── test_setup.py
│   └── README.md
│
├── Notebooks/
│   └── Dataset_Explorer.ipynb
│
└── Papers/
    ├── 2401.00153v2.pdf
    └── Dehazing_Ultrasound_Using_Diffusion_Models.pdf
```

---

## 🛠️ Setup Instructions

### Prerequisites

* Python 3.8+
* CUDA-compatible GPU (recommended for training)
* 8GB+ RAM
* 5GB+ free disk space

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd Dehazing

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r gan_training/requirements.txt
```

---

### 2. Verify Dataset Structure

The dataset should be organized as follows in the `Dataset/` folder:

* `clean/`: Clean echocardiography images (reference/ground truth)
* `noisy/`: Noisy echocardiography images (input for training)
* `noisy_roi/`: Region of Interest masks for evaluation
* `dataset_mapping.csv`: Complete file mapping and metadata

---

### 🧠 Model Setup

This project requires a pre-trained model file (`unet_best.pth`) which is not included in the repository due to size limitations.

#### 🔽 Step 1: Download the Model

Download the **`unet_best.pth`** file from the [Releases page](https://github.com/MisbahKhan0009/Dehazing/releases/tag/Checkpoints).

#### 📁 Step 2: Place the File

After downloading, place the file in the following directory:

```
project_root/
└── checkpoints/
    └── unet_dehazing/
        └── unet_best.pth
```

> **Note:** If the `checkpoints/unet_dehazing` folders do not exist, create them manually.

#### ✅ Step 3: Verify

Once placed correctly, the project should automatically load the model when running the application.

---

### 3. Test Installation

```bash
# Navigate to training directory
cd gan_training

# Run setup verification
python test_setup.py
```

---

## 🚀 Quick Start

### Dataset Exploration

1. **Interactive Exploration**: Open `Notebooks/Dataset_Explorer.ipynb` in Jupyter
2. **Command-line Analysis**: Run scripts in `Dataset/Scripts/`

```bash
# Generate dataset visualizations
cd Dataset/Scripts
python visualize_dataset.py

# View analysis summary
python analysis_summary.py
```

### Training a Model

```bash
# Navigate to training directory
cd gan_training

# Start training with default settings
python run_training.py

# Or customize training
python run_training.py --epochs 200 --batch-size 8 --lr 0.0002
```

### Monitoring Training

```bash
# View training logs
tensorboard --logdir=logs

# Check training progress
tail -f logs/training.log
```

---

## 📊 Dataset Details

### Image Specifications

* **Format**: PNG grayscale images
* **Naming**: `patient-{id}-4C-frame-{number}.png`
* **Content**: 4-chamber echocardiography views
* **Resolution**: Variable (automatically resized during training)

### Data Categories

1. **Clean Images** (`Dataset/clean/`): High-quality images
2. **Noisy Images** (`Dataset/noisy/`): Low-quality challenging images
3. **ROI Masks** (`Dataset/noisy_roi/`): Region of Interest annotations

### Evaluation Metrics

* **PSNR** (Peak Signal-to-Noise Ratio)
* **SSIM** (Structural Similarity Index)
* **CNR** (Contrast-to-Noise Ratio)
* **gCNR** (Generalized CNR)
* **KS Test** (Kolmogorov-Smirnov statistical test)

---

## 🧠 Model Architecture

### Generator

* **Base**: U-Net architecture with skip connections
* **Input**: Grayscale noisy echocardiography image (256x256)
* **Output**: Enhanced image (256x256)
* **Features**: Attention mechanisms, spectral normalization

### Discriminator

* **Base**: PatchGAN
* **Variants**: Standard, conditional, multi-scale, attention-based
* **Purpose**: Distinguish between real and generated clean images

### Loss Functions

* **Adversarial Loss**
* **L1/L2 Loss**
* **Perceptual Loss (VGG)**
* **CNR Loss (medical-specific)**

---

## 📈 Results and Evaluation

Training results are automatically saved to:

* `gan_training/checkpoints/`: Model weights
* `gan_training/logs/`: Training metrics and TensorBoard logs
* `gan_training/results/`: Generated sample images

---

## 🔬 Research Context

This project is based on research in medical image dehazing and denoising:

* Addresses echocardiography image quality challenges
* Implements advanced GAN architectures
* Focuses on preserving diagnostic detail
* Uses domain-specific medical metrics

---

## 📚 References

See the `Papers/` directory for related research papers.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create a Pull Request

---

## 📄 License

[Specify your license here]

---

## 🆘 Support

For questions or issues:

1. Check documentation in each subdirectory
2. Review example notebooks
3. Run the setup test script
4. Open an issue with details

---

**Note:** This project is part of the **CSE498 coursework** and focuses on **medical image enhancement using deep learning**.

---

