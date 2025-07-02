#!/usr/bin/env python3
"""
Configuration file for Echocardiography Dehazing GAN Training
Contains all hyperparameters and settings
"""

import os
from pathlib import Path
import torch
from torchvision.models import vgg19, VGG19_Weights


class Config:
    """Configuration class for GAN training"""

    def __init__(self):
        # Paths
        self.project_root = Path(__file__).parent.parent
        self.dataset_path = self.project_root.parent / "Dataset"
        self.csv_path = self.project_root.parent / "dataset_mapping.csv"
        self.checkpoint_dir = self.project_root / "checkpoints"
        self.log_dir = self.project_root / "logs"
        self.results_dir = self.project_root / "results"

        # Create directories if they don't exist
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        # Dataset parameters
        self.image_size = (256, 256)
        self.batch_size = 16
        self.num_workers = 4
        self.use_roi = True
        self.patient_split_ratio = (0.7, 0.2, 0.1)  # train, val, test
        self.patient_wise_split = True  # Whether to split dataset by patients

        # Model parameters
        self.generator_type = "unet"  # "unet" or "enhanced_unet"
        # "patch", "conditional_patch", "multiscale", "spectral", "attention"
        self.discriminator_type = "conditional_patch"
        self.use_attention = True
        self.generator_channels = 1
        self.discriminator_channels = 1
        self.ndf = 64  # Discriminator base feature maps
        self.discriminator_layers = 3
        self.use_spectral_norm = False

        # Training parameters
        self.num_epochs = 200
        self.learning_rate_g = 2e-4
        self.learning_rate_d = 2e-4
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.weight_decay = 1e-4
        self.resume_training = False  # Whether to resume training from a checkpoint
        self.resume_checkpoint = None  # Path to checkpoint file for resuming training

        # Loss function weights
        self.lambda_l1 = 100.0
        self.lambda_perceptual = 1.0
        self.lambda_ssim = 1.0
        self.lambda_cnr = 0.1
        self.lambda_edge = 10.0
        self.lambda_tv = 0.01
        self.lambda_gan = 1.0

        # GAN parameters
        self.gan_mode = 'lsgan'  # 'lsgan', 'vanilla', 'wgangp'
        self.discriminator_update_freq = 1
        self.generator_update_freq = 1
        self.gradient_penalty_lambda = 10.0

        # Progressive training
        self.use_progressive_training = False
        # Epochs to increase image resolution
        self.progressive_epochs = [50, 100, 150]
        self.progressive_sizes = [(128, 128), (192, 192), (256, 256)]

        # Curriculum learning
        self.use_curriculum_learning = False
        # Epochs to introduce harder samples
        self.curriculum_epochs = [30, 60, 90]
        self.curriculum_difficulty_levels = 3

        # Logging and checkpointing
        self.log_interval = 100  # Log every N iterations
        self.save_interval = 10  # Save checkpoint every N epochs
        self.eval_interval = 5   # Evaluate on validation set every N epochs
        self.generate_samples_interval = 10  # Generate sample images every N epochs
        self.num_sample_images = 8

        # Evaluation metrics
        self.eval_metrics = ['psnr', 'ssim', 'cnr', 'lpips']
        self.save_best_model = True
        self.best_metric = 'psnr'  # Metric to use for saving best model

        # Hardware
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mixed_precision = True  # Use automatic mixed precision
        self.dataloader_pin_memory = True

        # Reproducibility
        self.seed = 42
        self.deterministic = False  # Set to True for fully deterministic training

        # Advanced features
        self.use_ema = False  # Exponential Moving Average for generator
        self.ema_decay = 0.999
        self.use_self_supervised = False  # Use clean-only images for pre-training
        self.pretrain_epochs = 20

        # Data augmentation
        self.use_augmentation = True
        self.horizontal_flip_prob = 0.5
        self.rotation_degrees = 5
        self.brightness_jitter = 0.1
        self.contrast_jitter = 0.1

        # Early stopping
        self.use_early_stopping = True
        self.patience = 20
        self.min_delta = 1e-4

        # Learning rate scheduling
        self.use_lr_scheduler = True
        self.scheduler_type = 'cosine'  # 'cosine', 'step', 'plateau'
        self.lr_decay_epochs = [100, 150]
        self.lr_decay_gamma = 0.5
        self.cosine_eta_min = 1e-6

        # Validation
        self.validation_split = 'patient_wise'  # 'patient_wise' or 'random'
        self.cross_validation_folds = None  # Set to integer for k-fold CV

        # Model ensemble
        self.use_ensemble = False
        self.ensemble_size = 3

        # Experiment tracking
        self.use_tensorboard = True
        self.use_wandb = False
        self.wandb_project = "echocardiography-dehazing"
        self.experiment_name = "gan_baseline"

    def get_generator_config(self):
        """Get generator-specific configuration"""
        return {
            'n_channels': self.generator_channels,
            'n_classes': self.generator_channels,
            'use_attention': self.use_attention,
            'bilinear': True
        }

    def get_discriminator_config(self):
        """Get discriminator-specific configuration"""
        config = {
            'input_channels': self.discriminator_channels,
            'output_channels': self.discriminator_channels,
            'ndf': self.ndf,
            'n_layers': self.discriminator_layers
        }

        if self.discriminator_type == 'multiscale':
            config['num_scales'] = 3

        return config

    def get_loss_config(self):
        """Get loss function configuration"""
        return {
            'lambda_l1': self.lambda_l1,
            'lambda_perceptual': self.lambda_perceptual,
            'lambda_ssim': self.lambda_ssim,
            'lambda_cnr': self.lambda_cnr,
            'lambda_edge': self.lambda_edge,
            'lambda_tv': self.lambda_tv,
            'use_roi': self.use_roi
        }

    def get_optimizer_config(self):
        """Get optimizer configuration"""
        return {
            'generator': {
                'lr': self.learning_rate_g,
                'betas': (self.beta1, self.beta2),
                'weight_decay': self.weight_decay
            },
            'discriminator': {
                'lr': self.learning_rate_d,
                'betas': (self.beta1, self.beta2),
                'weight_decay': self.weight_decay
            }
        }

    def get_dataloader_config(self):
        """Get data loader configuration"""
        return {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'image_size': self.image_size,
            'use_roi': self.use_roi,
            'patient_split_ratio': self.patient_split_ratio
        }

    def update_from_dict(self, config_dict):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown configuration key '{key}'")

    def save_config(self, filepath):
        """Save configuration to file"""
        import json

        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            elif not key.startswith('_'):
                config_dict[key] = value

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def load_config(self, filepath):
        """Load configuration from file"""
        import json

        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        self.update_from_dict(config_dict)

    def print_config(self):
        """Print current configuration"""
        print("Configuration:")
        print("=" * 50)

        categories = {
            'Paths': ['dataset_path', 'csv_path', 'checkpoint_dir', 'log_dir', 'results_dir'],
            'Dataset': ['image_size', 'batch_size', 'num_workers', 'use_roi', 'patient_split_ratio'],
            'Model': ['generator_type', 'discriminator_type', 'use_attention', 'ndf'],
            'Training': ['num_epochs', 'learning_rate_g', 'learning_rate_d', 'beta1', 'beta2', 'resume_training', 'resume_checkpoint'],
            'Loss Weights': ['lambda_l1', 'lambda_perceptual', 'lambda_ssim', 'lambda_cnr', 'lambda_edge', 'lambda_tv'],
            'Hardware': ['device', 'mixed_precision', 'dataloader_pin_memory']
        }

        for category, keys in categories.items():
            print(f"\n{category}:")
            for key in keys:
                if hasattr(self, key):
                    value = getattr(self, key)
                    print(f"  {key}: {value}")


# Example configurations for different experiments
class QuickTestConfig(Config):
    """Quick test configuration for debugging"""

    def __init__(self):
        super().__init__()
        self.num_epochs = 5
        self.batch_size = 4
        self.num_workers = 0
        self.log_interval = 10
        self.save_interval = 2
        self.eval_interval = 2
        self.resume_training = False
        self.resume_checkpoint = None


class HighQualityConfig(Config):
    """High quality training configuration"""

    def __init__(self):
        super().__init__()
        self.num_epochs = 500
        self.batch_size = 8
        self.image_size = (512, 512)
        self.lambda_perceptual = 2.0
        self.lambda_ssim = 2.0
        self.use_progressive_training = True
        self.use_curriculum_learning = True
        self.resume_training = False
        self.resume_checkpoint = None


class MedicalFocusedConfig(Config):
    """Configuration focused on medical image quality"""

    def __init__(self):
        super().__init__()
        self.lambda_cnr = 1.0
        self.lambda_edge = 20.0
        self.lambda_ssim = 5.0
        self.discriminator_type = "attention"
        self.use_roi = True
        self.eval_metrics = ['psnr', 'ssim', 'cnr', 'gcnr', 'ks_test']
        self.resume_training = False
        self.resume_checkpoint = None


def get_config(config_name="default"):
    """Get configuration by name"""
    configs = {
        "default": Config,
        "quick_test": QuickTestConfig,
        "high_quality": HighQualityConfig,
        "medical_focused": MedicalFocusedConfig
    }

    if config_name not in configs:
        print(f"Unknown config name '{config_name}'. Using default.")
        config_name = "default"

    return configs[config_name]()


if __name__ == "__main__":
    # Test configuration
    config = Config()
    config.print_config()

    # Test saving and loading
    config.save_config("test_config.json")

    new_config = Config()
    new_config.load_config("test_config.json")

    print("\nLoaded configuration matches:",
          config.num_epochs == new_config.num_epochs)
