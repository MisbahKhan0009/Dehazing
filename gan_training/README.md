# Echocardiography Dehazing GAN Training Framework

This directory contains a comprehensive framework for training a GAN model to predict clean echocardiography images from noisy/hazy ones.

## Project Structure

```
gan_training/
├── config/                # Configuration management
│   ├── __init__.py
│   └── config.py          # Training configurations
├── data/                  # Data handling and preprocessing
│   ├── __init__.py
│   └── dataset.py         # Dataset classes and data loading
├── models/                # GAN model architectures
│   ├── __init__.py
│   ├── generator.py       # U-Net based generator
│   ├── discriminator.py   # PatchGAN discriminator variants
│   └── losses.py          # Custom loss functions (CNR, perceptual, etc.)
├── training/              # Training logic and utilities
│   ├── __init__.py
│   └── train.py           # Main training script
├── evaluation/            # Evaluation metrics and validation
│   ├── __init__.py
│   └── metrics.py         # Medical and standard image metrics
├── checkpoints/           # Model checkpoints (created during training)
├── logs/                  # Training logs and tensorboard files
├── results/               # Generated images and evaluation results
├── requirements.txt       # Python dependencies
├── run_training.py        # Simple training runner script
└── test_setup.py          # Setup verification script
```

## Quick Start

### 1. Test Setup

```bash
# Verify everything is working
python test_setup.py
```

### 2. Install Dependencies

```bash
# Install required packages
python run_training.py --install-deps
```

### 3. Start Training

```bash
# Start training with default configuration
python run_training.py

# Or use a specific configuration
python run_training.py --config high_quality
```

Available configurations:

- `default`: Standard training settings
- `quick_test`: Faster training with smaller batch size for testing
- `high_quality`: Higher resolution, more epochs, focused on image quality
- `medical_focused`: Optimized for medical image metrics like CNR

### 4. Monitor Training

```bash
# Launch tensorboard to monitor training
tensorboard --logdir experiments/
```

The training process will generate:

- Checkpoints in `experiments/[timestamp]/checkpoints/`
- Sample images in `experiments/[timestamp]/samples/`
- Logs in `experiments/[timestamp]/logs/`

### 5. Resume Training

To resume training from a checkpoint:

```bash
# Resume training from latest checkpoint
python run_training.py --resume experiments/latest/checkpoints/latest.pth
```

## Model Architecture

### Generator

The framework provides two generator architectures:

- **UNetGenerator**: Standard U-Net with skip connections
- **EnhancedUNetGenerator**: U-Net with attention and residual blocks

### Discriminator

Multiple discriminator architectures are available:

- **PatchGANDiscriminator**: Standard PatchGAN discriminator
- **ConditionalPatchGANDiscriminator**: Conditional discriminator that takes both input and target
- **MultiScaleDiscriminator**: Multi-scale discriminator for different patch sizes
- **AttentionDiscriminator**: Discriminator with self-attention mechanism

### Loss Functions

The training combines multiple loss functions:

- **Adversarial Loss**: Standard GAN loss (MSE or BCE)
- **Perceptual Loss**: Feature-based loss using pretrained VGG
- **SSIM Loss**: Structural similarity loss
- **CNR Loss**: Contrast-to-Noise Ratio loss for medical imaging
- **Edge Preservation Loss**: Preserves edge details

## Training Configuration

The `config/config.py` file contains configurable parameters for:

- Dataset paths and preprocessing
- Model architecture and hyperparameters
- Training settings (batch size, learning rates, etc.)
- Loss function weights
- Evaluation metrics

## Evaluation Metrics

The framework tracks multiple evaluation metrics:

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **CNR**: Contrast-to-Noise Ratio (medical-specific)
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **Edge Preservation**: Edge detail preservation measurement

## Custom Dataset

The dataset class expects the following structure:

- `clean/`: Clean echocardiography images
- `noisy/`: Noisy/hazy echocardiography images
- (Optional) `noisy_roi/`: ROI annotations for noisy images

## Advanced Usage

### Custom Configuration

Create a new configuration class in `config/config.py`:

```python
class MyCustomConfig(Config):
    def __init__(self):
        super().__init__()
        self.batch_size = 8
        self.g_lr = 1e-4
        self.d_lr = 4e-4
        # Add your custom configuration parameters
```

Then register it in the `get_config()` function.
