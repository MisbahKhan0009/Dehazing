"""
Main training script for Echocardiography Dehazing GAN
"""

from training.trainer import GANTrainer
from evaluation.metrics import calculate_metrics
from models.losses import GANLoss, PerceptualLoss, SSIMLoss, CNRLoss
from models.discriminator import ConditionalPatchGANDiscriminator
from models.generator import UNetGenerator
from data.dataset import create_dataloaders, visualize_batch
from config.config import Config
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import time
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def setup_logging(config):
    """Setup logging configuration"""
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )


def setup_deterministic_training(config):
    """Setup deterministic training for reproducibility"""
    if config.deterministic:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.backends.cudnn.benchmark = True


def create_models(config):
    """Create generator and discriminator models"""
    # Generator
    if config.generator_type == "unet":
        generator = UNetGenerator(
            n_channels=config.generator_channels,
            n_classes=config.generator_channels,
            use_attention=config.use_attention,
            bilinear=True
        )
    else:
        raise ValueError(f"Unknown generator type: {config.generator_type}")

    # Discriminator
    if config.discriminator_type == "conditional_patch":
        discriminator = ConditionalPatchGANDiscriminator(
            input_channels=config.discriminator_channels,  # noisy image input channel
            output_channels=config.discriminator_channels,  # clean/generated output channel
            ndf=config.ndf,
            n_layers=config.discriminator_layers
        )
    else:
        raise ValueError(
            f"Unknown discriminator type: {config.discriminator_type}")

    return generator, discriminator


def create_optimizers(generator, discriminator, config):
    """Create optimizers for generator and discriminator"""
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=config.learning_rate_g,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )

    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=config.learning_rate_d,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )

    return optimizer_g, optimizer_d


def create_loss_functions(config):
    """Create loss functions"""
    losses = {}

    # GAN loss
    losses['gan'] = GANLoss(config.gan_mode).to(config.device)

    # Reconstruction losses
    losses['l1'] = nn.L1Loss()
    losses['mse'] = nn.MSELoss()

    # Perceptual loss
    if config.lambda_perceptual > 0:
        losses['perceptual'] = PerceptualLoss().to(config.device)

    # SSIM loss
    if config.lambda_ssim > 0:
        losses['ssim'] = SSIMLoss().to(config.device)

    # CNR loss (if using ROI)
    if config.lambda_cnr > 0 and config.use_roi:
        losses['cnr'] = CNRLoss().to(config.device)

    return losses


def create_schedulers(optimizer_g, optimizer_d, config):
    """Create learning rate schedulers"""
    schedulers = {}

    if config.use_lr_scheduler:
        if config.scheduler_type == 'cosine':
            schedulers['generator'] = optim.lr_scheduler.CosineAnnealingLR(
                optimizer_g,
                T_max=config.num_epochs,
                eta_min=config.cosine_eta_min
            )
            schedulers['discriminator'] = optim.lr_scheduler.CosineAnnealingLR(
                optimizer_d,
                T_max=config.num_epochs,
                eta_min=config.cosine_eta_min
            )
        elif config.scheduler_type == 'step':
            schedulers['generator'] = optim.lr_scheduler.MultiStepLR(
                optimizer_g,
                milestones=config.lr_decay_epochs,
                gamma=config.lr_decay_gamma
            )
            schedulers['discriminator'] = optim.lr_scheduler.MultiStepLR(
                optimizer_d,
                milestones=config.lr_decay_epochs,
                gamma=config.lr_decay_gamma
            )

    return schedulers


def save_checkpoint(generator, discriminator, optimizer_g, optimizer_d,
                    schedulers, epoch, losses, metrics, config, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'losses': losses,
        'metrics': metrics,
        'config': config.__dict__
    }

    # Add scheduler states if they exist
    if schedulers:
        checkpoint['scheduler_g_state_dict'] = schedulers.get(
            'generator', {}).state_dict() if schedulers.get('generator') else None
        checkpoint['scheduler_d_state_dict'] = schedulers.get(
            'discriminator', {}).state_dict() if schedulers.get('discriminator') else None

    # Save regular checkpoint
    checkpoint_path = Path(config.checkpoint_dir) / \
        f"checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = Path(config.checkpoint_dir) / "best_model.pth"
        torch.save(checkpoint, best_path)
        logging.info(f"New best model saved at epoch {epoch}")

    # Keep only last N checkpoints
    keep_checkpoints = 5
    checkpoints = sorted(
        Path(config.checkpoint_dir).glob("checkpoint_epoch_*.pth"))
    if len(checkpoints) > keep_checkpoints:
        for old_checkpoint in checkpoints[:-keep_checkpoints]:
            old_checkpoint.unlink()


def load_checkpoint(checkpoint_path, generator, discriminator, optimizer_g, optimizer_d, schedulers=None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

    if schedulers and 'scheduler_g_state_dict' in checkpoint:
        if schedulers.get('generator') and checkpoint['scheduler_g_state_dict']:
            schedulers['generator'].load_state_dict(
                checkpoint['scheduler_g_state_dict'])
        if schedulers.get('discriminator') and checkpoint['scheduler_d_state_dict']:
            schedulers['discriminator'].load_state_dict(
                checkpoint['scheduler_d_state_dict'])

    return checkpoint['epoch'], checkpoint.get('losses', {}), checkpoint.get('metrics', {})


def train_epoch(generator, discriminator, dataloader, optimizer_g, optimizer_d,
                losses, config, epoch, writer, global_step):
    """Train for one epoch"""
    generator.train()
    discriminator.train()

    epoch_losses = {
        'g_total': 0.0, 'g_gan': 0.0, 'g_l1': 0.0,
        'd_total': 0.0, 'd_real': 0.0, 'd_fake': 0.0
    }

    # Add other loss components
    if config.lambda_perceptual > 0:
        epoch_losses['g_perceptual'] = 0.0
    if config.lambda_ssim > 0:
        epoch_losses['g_ssim'] = 0.0
    if config.lambda_cnr > 0:
        epoch_losses['g_cnr'] = 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        noisy = batch['noisy'].to(config.device)
        clean = batch['clean'].to(config.device)
        roi = batch.get('roi', None)
        if roi is not None:
            roi = roi.to(config.device)

        batch_size = noisy.size(0)

        # Generate fake images
        fake_clean = generator(noisy)

        # =====================================
        # Train Discriminator
        # =====================================
        optimizer_d.zero_grad()

        # Real pairs (noisy, clean)
        real_pairs = torch.cat([noisy, clean], dim=1)
        pred_real = discriminator(real_pairs)
        loss_d_real = losses['gan'](pred_real, True)

        # Fake pairs (noisy, generated)
        fake_pairs = torch.cat([noisy, fake_clean.detach()], dim=1)
        pred_fake = discriminator(fake_pairs)
        loss_d_fake = losses['gan'](pred_fake, False)

        # Total discriminator loss
        loss_d_total = (loss_d_real + loss_d_fake) * 0.5
        loss_d_total.backward()
        optimizer_d.step()

        # =====================================
        # Train Generator
        # =====================================
        if (batch_idx + 1) % config.generator_update_freq == 0:
            optimizer_g.zero_grad()

            # Adversarial loss
            fake_pairs = torch.cat([noisy, fake_clean], dim=1)
            pred_fake = discriminator(fake_pairs)
            loss_g_gan = losses['gan'](
                pred_fake, True) * config.lambda_adversarial

            # L1 reconstruction loss
            loss_g_l1 = losses['l1'](fake_clean, clean) * config.lambda_l1

            # Total generator loss
            loss_g_total = loss_g_gan + loss_g_l1

            # Additional losses
            if config.lambda_perceptual > 0:
                loss_g_perceptual = losses['perceptual'](
                    fake_clean, clean) * config.lambda_perceptual
                loss_g_total += loss_g_perceptual
                epoch_losses['g_perceptual'] += loss_g_perceptual.item()

            if config.lambda_ssim > 0:
                loss_g_ssim = (
                    1 - losses['ssim'](fake_clean, clean)) * config.lambda_ssim
                loss_g_total += loss_g_ssim
                epoch_losses['g_ssim'] += loss_g_ssim.item()

            if config.lambda_cnr > 0 and roi is not None:
                loss_g_cnr = losses['cnr'](
                    fake_clean, clean, roi) * config.lambda_cnr
                loss_g_total += loss_g_cnr
                epoch_losses['g_cnr'] += loss_g_cnr.item()

            loss_g_total.backward()
            optimizer_g.step()

            # Update epoch losses
            epoch_losses['g_total'] += loss_g_total.item()
            epoch_losses['g_gan'] += loss_g_gan.item()
            epoch_losses['g_l1'] += loss_g_l1.item()

        # Update discriminator losses
        epoch_losses['d_total'] += loss_d_total.item()
        epoch_losses['d_real'] += loss_d_real.item()
        epoch_losses['d_fake'] += loss_d_fake.item()

        # Log to tensorboard
        if (batch_idx + 1) % config.log_interval == 0:
            step = global_step[0]
            writer.add_scalar('Train/G_Total', loss_g_total.item(), step)
            writer.add_scalar('Train/G_GAN', loss_g_gan.item(), step)
            writer.add_scalar('Train/G_L1', loss_g_l1.item(), step)
            writer.add_scalar('Train/D_Total', loss_d_total.item(), step)
            writer.add_scalar('Train/D_Real', loss_d_real.item(), step)
            writer.add_scalar('Train/D_Fake', loss_d_fake.item(), step)

            # Add learning rates
            writer.add_scalar('Train/LR_Generator',
                              optimizer_g.param_groups[0]['lr'], step)
            writer.add_scalar('Train/LR_Discriminator',
                              optimizer_d.param_groups[0]['lr'], step)

        global_step[0] += 1

        # Update progress bar
        progress_bar.set_postfix({
            'G_Loss': f"{loss_g_total.item():.4f}",
            'D_Loss': f"{loss_d_total.item():.4f}"
        })

    # Average losses over epoch
    num_batches = len(dataloader)
    for key in epoch_losses:
        epoch_losses[key] /= num_batches

    return epoch_losses


def validate_epoch(generator, discriminator, dataloader, losses, config, epoch, writer):
    """Validate for one epoch"""
    generator.eval()
    discriminator.eval()

    val_losses = {
        'g_total': 0.0, 'g_gan': 0.0, 'g_l1': 0.0,
        'd_total': 0.0, 'd_real': 0.0, 'd_fake': 0.0
    }

    all_metrics = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            # Move data to device
            noisy = batch['noisy'].to(config.device)
            clean = batch['clean'].to(config.device)
            roi = batch.get('roi', None)
            if roi is not None:
                roi = roi.to(config.device)

            # Generate fake images
            fake_clean = generator(noisy)

            # Calculate losses (same as training but no backward pass)
            # Real pairs
            real_pairs = torch.cat([noisy, clean], dim=1)
            pred_real = discriminator(real_pairs)
            loss_d_real = losses['gan'](pred_real, True)

            # Fake pairs
            fake_pairs = torch.cat([noisy, fake_clean], dim=1)
            pred_fake = discriminator(fake_pairs)
            loss_d_fake = losses['gan'](pred_fake, False)

            # Discriminator loss
            loss_d_total = (loss_d_real + loss_d_fake) * 0.5

            # Generator losses
            loss_g_gan = losses['gan'](
                pred_fake, True) * config.lambda_adversarial
            loss_g_l1 = losses['l1'](fake_clean, clean) * config.lambda_l1
            loss_g_total = loss_g_gan + loss_g_l1

            # Update validation losses
            val_losses['g_total'] += loss_g_total.item()
            val_losses['g_gan'] += loss_g_gan.item()
            val_losses['g_l1'] += loss_g_l1.item()
            val_losses['d_total'] += loss_d_total.item()
            val_losses['d_real'] += loss_d_real.item()
            val_losses['d_fake'] += loss_d_fake.item()

            # Calculate metrics for each image in batch
            for i in range(noisy.size(0)):
                metrics = calculate_metrics(
                    fake_clean[i:i+1],
                    clean[i:i+1],
                    roi[i:i+1] if roi is not None else None,
                    config.eval_metrics
                )
                all_metrics.append(metrics)

    # Average losses
    num_batches = len(dataloader)
    for key in val_losses:
        val_losses[key] /= num_batches

    # Average metrics
    avg_metrics = {}
    if all_metrics:
        for metric_name in all_metrics[0].keys():
            avg_metrics[metric_name] = np.mean(
                [m[metric_name] for m in all_metrics])

    # Log to tensorboard
    for key, value in val_losses.items():
        writer.add_scalar(f'Val/Loss_{key}', value, epoch)

    for key, value in avg_metrics.items():
        writer.add_scalar(f'Val/Metric_{key}', value, epoch)

    return val_losses, avg_metrics


def main():
    """Main training function"""
    # Load configuration
    config = Config()

    # Setup
    setup_logging(config)
    setup_deterministic_training(config)

    # Log configuration
    logging.info("Starting training with configuration:")
    config.print_config()

    # Save configuration
    config.save_config(Path(config.checkpoint_dir) / "config.json")

    # Create data loaders
    logging.info("Creating data loaders...")
    dataloaders, datasets = create_dataloaders(config)

    # Create models
    logging.info("Creating models...")
    generator, discriminator = create_models(config)
    generator.to(config.device)
    discriminator.to(config.device)

    # Print model info
    total_params_g = sum(p.numel() for p in generator.parameters())
    total_params_d = sum(p.numel() for p in discriminator.parameters())
    logging.info(f"Generator parameters: {total_params_g:,}")
    logging.info(f"Discriminator parameters: {total_params_d:,}")

    # Create optimizers
    optimizer_g, optimizer_d = create_optimizers(
        generator, discriminator, config)

    # Create schedulers
    schedulers = create_schedulers(optimizer_g, optimizer_d, config)

    # Create loss functions
    losses = create_loss_functions(config)

    # Setup tensorboard
    writer = SummaryWriter(config.log_dir)

    # Resume training if specified
    start_epoch = 0
    best_metric = 0.0
    global_step = [0]  # Use list for mutable reference

    if config.resume_training:
        checkpoint_path = Path(config.checkpoint_dir) / "latest_checkpoint.pth"
        if checkpoint_path.exists():
            logging.info(f"Resuming training from {checkpoint_path}")
            start_epoch, _, metrics = load_checkpoint(
                checkpoint_path, generator, discriminator,
                optimizer_g, optimizer_d, schedulers
            )
            best_metric = metrics.get(config.best_metric, 0.0)
            start_epoch += 1

    # Training loop
    logging.info("Starting training loop...")

    for epoch in range(start_epoch, config.num_epochs):
        epoch_start_time = time.time()

        # Train
        train_losses = train_epoch(
            generator, discriminator, dataloaders['train'],
            optimizer_g, optimizer_d, losses, config,
            epoch, writer, global_step
        )

        # Log training losses
        logging.info(f"Epoch {epoch} - Training losses:")
        for key, value in train_losses.items():
            logging.info(f"  {key}: {value:.6f}")

        # Validate
        if (epoch + 1) % config.eval_interval == 0:
            val_losses, val_metrics = validate_epoch(
                generator, discriminator, dataloaders['val'],
                losses, config, epoch, writer
            )

            # Log validation results
            logging.info(f"Epoch {epoch} - Validation losses:")
            for key, value in val_losses.items():
                logging.info(f"  {key}: {value:.6f}")

            logging.info(f"Epoch {epoch} - Validation metrics:")
            for key, value in val_metrics.items():
                logging.info(f"  {key}: {value:.6f}")

            # Check if this is the best model
            current_metric = val_metrics.get(config.best_metric, 0.0)
            is_best = current_metric > best_metric
            if is_best:
                best_metric = current_metric
        else:
            val_losses, val_metrics = {}, {}
            is_best = False

        # Update schedulers
        if schedulers:
            schedulers['generator'].step()
            schedulers['discriminator'].step()

        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(
                generator, discriminator, optimizer_g, optimizer_d,
                schedulers, epoch, train_losses, val_metrics, config, is_best
            )

        # Log timing
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")

        # Generate sample images
        if (epoch + 1) % config.generate_samples_interval == 0:
            generate_sample_images(
                generator, dataloaders['val'], config, epoch, writer)

    # Final save
    save_checkpoint(
        generator, discriminator, optimizer_g, optimizer_d,
        schedulers, config.num_epochs - 1, train_losses, val_metrics, config
    )

    logging.info("Training completed!")
    writer.close()


def generate_sample_images(generator, dataloader, config, epoch, writer):
    """Generate and save sample images"""
    generator.eval()

    with torch.no_grad():
        batch = next(iter(dataloader))
        noisy = batch['noisy'][:config.num_sample_images].to(config.device)
        clean = batch['clean'][:config.num_sample_images].to(config.device)

        fake_clean = generator(noisy)

        # Save images to tensorboard
        # Denormalize images for visualization
        noisy_vis = (noisy + 1) / 2  # [-1, 1] -> [0, 1]
        clean_vis = (clean + 1) / 2
        fake_vis = (fake_clean + 1) / 2

        writer.add_images('Samples/Noisy', noisy_vis, epoch)
        writer.add_images('Samples/Clean', clean_vis, epoch)
        writer.add_images('Samples/Generated', fake_vis, epoch)


if __name__ == "__main__":
    main()
