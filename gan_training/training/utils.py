"""
Training utilities for GAN model training.
Includes checkpoint management, seeding, and sample generation utilities.
"""

import os
import torch
import random
import numpy as np
from typing import Dict, Any, Optional, Tuple
import torchvision.utils as vutils
from PIL import Image
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False
) -> None:
    """
    Save model checkpoint.

    Args:
        state: Dictionary containing model state and metadata
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
        is_best: Whether this is the best model so far
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    torch.save(state, checkpoint_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)
        print(f"Best model saved to {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        generator: Generator model
        discriminator: Discriminator model
        optimizer_g: Generator optimizer
        optimizer_d: Discriminator optimizer
        device: Device to load models on

    Returns:
        Dictionary containing loaded metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(
        f"Best validation loss: {checkpoint.get('best_val_loss', 'Unknown')}")

    return checkpoint


def generate_sample_images(
    generator: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 4,
    save_path: Optional[str] = None,
    epoch: Optional[int] = None
) -> torch.Tensor:
    """
    Generate sample images for visualization during training.

    Args:
        generator: Generator model
        dataloader: Data loader for test/validation data
        device: Device for computation
        num_samples: Number of sample triplets to generate
        save_path: Path to save the visualization
        epoch: Current epoch number for filename

    Returns:
        Generated sample grid tensor
    """
    generator.eval()

    with torch.no_grad():
        # Get a batch of data
        data_iter = iter(dataloader)
        try:
            batch = next(data_iter)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                noisy_images = batch[0][:num_samples].to(device)
                clean_images = batch[1][:num_samples].to(device)
            else:
                raise ValueError(
                    "Batch should contain at least noisy and clean images")
        except (StopIteration, ValueError) as e:
            print(f"Error getting batch: {e}")
            return torch.empty(0)

        # Generate denoised images
        generated_images = generator(noisy_images)

        # Create comparison grid: [noisy, generated, clean] for each sample
        comparison_images = []
        for i in range(num_samples):
            comparison_images.extend([
                noisy_images[i],
                generated_images[i],
                clean_images[i]
            ])

        # Create grid
        grid = vutils.make_grid(
            comparison_images,
            nrow=3,  # 3 images per row (noisy, generated, clean)
            normalize=True,
            scale_each=True,
            padding=2,
            pad_value=1.0
        )

        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Create figure
            plt.figure(figsize=(15, 5 * num_samples))
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.axis('off')

            # Add labels
            for i in range(num_samples):
                y_pos = (i + 0.5) / num_samples
                plt.text(0.16, y_pos, 'Noisy', transform=plt.gca().transAxes,
                         ha='center', va='center', fontsize=12, color='white',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                plt.text(0.5, y_pos, 'Generated', transform=plt.gca().transAxes,
                         ha='center', va='center', fontsize=12, color='white',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                plt.text(0.84, y_pos, 'Clean', transform=plt.gca().transAxes,
                         ha='center', va='center', fontsize=12, color='white',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

            if epoch is not None:
                plt.title(
                    f'Sample Results - Epoch {epoch}', fontsize=16, pad=20)
            else:
                plt.title('Sample Results', fontsize=16, pad=20)

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Sample images saved to {save_path}")

    generator.train()
    return grid


def create_learning_curves_plot(
    train_losses: Dict[str, list],
    val_losses: Dict[str, list],
    save_path: str
) -> None:
    """
    Create and save learning curves plot.

    Args:
        train_losses: Dictionary of training losses over epochs
        val_losses: Dictionary of validation losses over epochs
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16)

    # Generator loss
    if 'generator' in train_losses and train_losses['generator']:
        axes[0, 0].plot(train_losses['generator'], label='Train', color='blue')
        if 'generator' in val_losses and val_losses['generator']:
            axes[0, 0].plot(val_losses['generator'],
                            label='Validation', color='red')
        axes[0, 0].set_title('Generator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # Discriminator loss
    if 'discriminator' in train_losses and train_losses['discriminator']:
        axes[0, 1].plot(train_losses['discriminator'],
                        label='Train', color='blue')
        if 'discriminator' in val_losses and val_losses['discriminator']:
            axes[0, 1].plot(val_losses['discriminator'],
                            label='Validation', color='red')
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # PSNR
    if 'psnr' in val_losses and val_losses['psnr']:
        axes[1, 0].plot(val_losses['psnr'],
                        label='Validation PSNR', color='green')
        axes[1, 0].set_title('PSNR')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('PSNR (dB)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # SSIM
    if 'ssim' in val_losses and val_losses['ssim']:
        axes[1, 1].plot(val_losses['ssim'],
                        label='Validation SSIM', color='purple')
        axes[1, 1].set_title('SSIM')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('SSIM')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Learning curves saved to {save_path}")


def log_metrics(
    metrics: Dict[str, float],
    epoch: int,
    phase: str = "train",
    log_file: Optional[str] = None
) -> None:
    """
    Log metrics to console and optionally to file.

    Args:
        metrics: Dictionary of metric values
        epoch: Current epoch
        phase: Phase (train/val/test)
        log_file: Optional log file path
    """
    log_str = f"Epoch {epoch} - {phase.capitalize()}:"
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            log_str += f" {metric_name}: {value:.4f}"
        else:
            log_str += f" {metric_name}: {value}"

    print(log_str)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(log_str + '\n')


def calculate_model_size(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Calculate model size in parameters and bytes.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_parameters, model_size_mb)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    # Estimate size in MB (assuming 32-bit floats)
    model_size_mb = total_params * 4 / (1024 * 1024)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB")

    return total_params, model_size_mb


def warmup_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    base_lr: float,
    current_epoch: int
) -> None:
    """
    Apply learning rate warmup.

    Args:
        optimizer: Optimizer to modify
        warmup_epochs: Number of warmup epochs
        base_lr: Base learning rate after warmup
        current_epoch: Current epoch (0-indexed)
    """
    if current_epoch < warmup_epochs:
        lr = base_lr * (current_epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
