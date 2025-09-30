# Echocardiography Dehazing Using GANs

This project implements a deep learning approach for dehazing echocardiography images using Generative Adversarial Networks (GANs). The goal is to enhance the quality of ultrasound images captured from difficult-to-image subjects by removing noise and improving visibility.

## 🚀 Project Overview

Echocardiography is a crucial diagnostic tool in cardiology, but image quality can be significantly degraded by factors such as patient body habitus, acoustic shadowing, and poor acoustic windows. This project addresses these challenges by:

- **Training GANs** to predict clean images from noisy/hazy echocardiography data
- **Implementing advanced architectures** including U-Net generators and PatchGAN discriminators  
- **Using medical-specific metrics** such as CNR (Contrast-to-Noise Ratio) and gCNR for evaluation
- **Providing comprehensive tools** for dataset analysis, training, and evaluation

## 📁 Project Structure

```
Dehazing/
├── README.md                   # This file - project overview and setup
├── .gitignore                  # Git ignore rules
├── .venv/                      # Python virtual environment (created locally)
│
├── Dataset/                    # 📊 Dataset and analysis tools
│   ├── clean/                  # Clean echocardiography images (easy-to-image subjects)
│   ├── noisy/                  # Noisy echocardiography images (difficult-to-image subjects)  
│   ├── noisy_roi/              # ROI annotations for evaluation metrics (CNR, gCNR, KS test)
│   ├── dataset_mapping.csv     # Complete mapping of all image files and metadata
│   ├── Dataset_Analysis_Report.md # Analysis results and statistics
│   ├── FolderDescriptions.txt  # Detailed folder descriptions
│   ├── Scripts/                # Analysis and visualization scripts
│   │   ├── analyze_dataset.py
│   │   ├── preview_visualizations.py
│   │   └── visualize_dataset.py
│   └── visualizations/         # Generated analysis visualizations
│
├── gan_training/               # 🤖 GAN Training Framework
│   ├── config/                 # Configuration management
│   │   ├── __init__.py
│   │   └── config.py           # Training hyperparameters and paths
│   ├── data/                   # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── dataset.py          # PyTorch dataset classes and data loaders
│   ├── models/                 # Neural network architectures
│   │   ├── __init__.py
│   │   ├── generator.py        # U-Net based generator models
│   │   ├── discriminator.py    # PatchGAN discriminator variants
│   │   └── losses.py           # Custom loss functions (CNR, perceptual, etc.)
│   ├── training/               # Training logic and utilities
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training orchestration
│   │   ├── train.py            # Main training script
│   │   └── utils.py            # Training utilities
│   ├── evaluation/             # Model evaluation and metrics
│   │   ├── __init__.py
│   │   └── metrics.py          # Medical and standard image quality metrics
│   ├── scripts/                # Training scripts and utilities
│   │   └── train.py            # Alternative training script
│   ├── checkpoints/            # Model checkpoints (created during training)
│   ├── logs/                   # Training logs and TensorBoard files
│   ├── data/                   # Training data cache (created during training)
│   ├── requirements.txt        # Python dependencies for training
│   ├── run_training.py         # Main training runner script
│   ├── test_setup.py           # Setup verification and testing
│   └── README.md               # Detailed training framework documentation
│
├── Notebooks/                  # 📓 Jupyter notebooks for exploration
│   └── Dataset_Explorer.ipynb  # Interactive dataset exploration and analysis
│
└── Papers/                     # 📚 Research papers and references
    ├── 2401.00153v2.pdf
    └── Dehazing_Ultrasound_Using_Diffusion_Models.pdf
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8+ 
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- 5GB+ free disk space

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

### 2. Verify Dataset Structure

The dataset should be organized as follows in the `Dataset/` folder:
- `clean/`: Clean echocardiography images (reference/ground truth)
- `noisy/`: Noisy echocardiography images (input for training)
- `noisy_roi/`: Region of Interest masks for evaluation
- `dataset_mapping.csv`: Complete file mapping and metadata

### 3. Test Installation

```bash
# Navigate to training directory
cd gan_training

# Run setup verification
python test_setup.py
```

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

## 📊 Dataset Details

### Image Specifications
- **Format**: PNG grayscale images
- **Naming**: `patient-{id}-4C-frame-{number}.png`
- **Content**: 4-chamber echocardiography views
- **Resolution**: Variable (automatically resized during training)

### Data Categories
1. **Clean Images** (`Dataset/clean/`): High-quality images from subjects with good acoustic windows
2. **Noisy Images** (`Dataset/noisy/`): Lower-quality images from subjects with challenging acoustic conditions
3. **ROI Masks** (`Dataset/noisy_roi/`): Region of Interest annotations for evaluation metrics

### Evaluation Metrics
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **CNR**: Contrast-to-Noise Ratio (medical-specific)
- **gCNR**: Generalized CNR
- **KS Test**: Kolmogorov-Smirnov statistical test

## 🧠 Model Architecture

### Generator
- **Base**: U-Net architecture with skip connections
- **Input**: Grayscale noisy echocardiography image (256x256)
- **Output**: Denoised/enhanced image (256x256)
- **Features**: Attention mechanisms, spectral normalization options

### Discriminator  
- **Base**: PatchGAN architecture
- **Variants**: Standard, conditional, multi-scale, spectral, attention-based
- **Purpose**: Distinguish between real clean images and generated outputs

### Loss Functions
- **Adversarial Loss**: Standard GAN objective
- **L1/L2 Loss**: Pixel-wise reconstruction
- **Perceptual Loss**: VGG-based feature matching
- **CNR Loss**: Custom medical imaging metric

## 📈 Results and Evaluation

Training results are automatically saved to:
- `gan_training/checkpoints/`: Model weights and optimizer states
- `gan_training/logs/`: Training metrics and TensorBoard logs  
- `gan_training/results/`: Generated sample images

## 🔬 Research Context

This project is based on research in medical image dehazing and denoising:
- Addresses challenges in echocardiography image quality
- Implements state-of-the-art GAN architectures for medical imaging
- Focuses on preserving clinical diagnostic information
- Uses domain-specific evaluation metrics

## 📚 References

See `Papers/` directory for relevant research papers and documentation.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

[Specify your license here]

## 🆘 Support

For questions or issues:
1. Check the documentation in each subdirectory
2. Review the example notebooks
3. Run the test setup script to verify installation
4. Open an issue with detailed error descriptions

---

**Note**: This project is part of CSE498 coursework and research into medical image enhancement using deep learning techniques.
