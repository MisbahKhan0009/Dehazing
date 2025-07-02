"""
Trainer class for GAN-based dehazing model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import numpy as np
from pathlib import Path
import logging
import os
from datetime import datetime
from tqdm import tqdm

from models.losses import GANLoss, PerceptualLoss, SSIMLoss
from evaluation.metrics import MetricsCalculator


class GANTrainer:
    """
    Trainer class handling the training loop and validation
    """

    def __init__(self,
                 generator,
                 discriminator,
                 config,
                 train_loader,
                 val_loader=None,
                 device='cuda',
                 experiment_name=None):
        """
        Args:
            generator: Generator model
            discriminator: Discriminator model
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            device: Device to train on (cuda/cpu)
            experiment_name: Name for the experiment (for logging)
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Create experiment name and directories
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name

        # Setup directories
        self.exp_dir = Path("experiments") / self.experiment_name
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.sample_dir = self.exp_dir / "samples"
        self.log_dir = self.exp_dir / "logs"

        for directory in [self.exp_dir, self.checkpoint_dir, self.sample_dir, self.log_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Setup optimizers
        self.g_optimizer = Adam(
            self.generator.parameters(),
            lr=config.g_lr,
            betas=(config.beta1, 0.999)
        )
        self.d_optimizer = Adam(
            self.discriminator.parameters(),
            lr=config.d_lr,
            betas=(config.beta1, 0.999)
        )

        # Setup schedulers
        self.g_scheduler = CosineAnnealingLR(
            self.g_optimizer,
            T_max=config.epochs,
            eta_min=config.g_lr * 0.1
        )
        self.d_scheduler = CosineAnnealingLR(
            self.d_optimizer,
            T_max=config.epochs,
            eta_min=config.d_lr * 0.1
        )

        # Setup loss functions
        self.gan_loss = GANLoss().to(device)
        self.perceptual_loss = PerceptualLoss().to(device)
        self.ssim_loss = SSIMLoss().to(device)

        # Setup metrics calculator
        self.metrics = MetricsCalculator(device=device)

        # Initialize tensorboard writer
        self.writer = SummaryWriter(str(self.log_dir))

        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Setup file handler for logging
        fh = logging.FileHandler(self.exp_dir / 'training.log')
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)

        # Log model architectures
        self.logger.info(f"Generator architecture:\n{self.generator}")
        self.logger.info(f"Discriminator architecture:\n{self.discriminator}")

    def train_step(self, batch):
        """
        Single training step
        Returns dictionary with losses
        """
        noisy = batch['noisy'].to(self.device)
        clean = batch['clean'].to(self.device)

        # Train discriminator
        self.d_optimizer.zero_grad()

        # Real loss
        real_cat = torch.cat([noisy, clean], dim=1)
        d_real = self.discriminator(real_cat)
        d_real_loss = self.gan_loss(d_real, True)

        # Fake loss
        fake = self.generator(noisy)
        fake_cat = torch.cat([noisy, fake.detach()], dim=1)
        d_fake = self.discriminator(fake_cat)
        d_fake_loss = self.gan_loss(d_fake, False)

        # Combined D loss
        d_loss = (d_real_loss + d_fake_loss) * 0.5
        d_loss.backward()
        self.d_optimizer.step()

        # Train generator
        self.g_optimizer.zero_grad()

        # GAN loss
        fake_cat = torch.cat([noisy, fake], dim=1)
        d_fake = self.discriminator(fake_cat)
        g_gan_loss = self.gan_loss(d_fake, True)

        # Content losses
        g_perceptual_loss = self.perceptual_loss(fake, clean)
        g_ssim_loss = self.ssim_loss(fake, clean)

        # Combined G loss
        g_loss = (g_gan_loss * self.config.lambda_gan +
                  g_perceptual_loss * self.config.lambda_perceptual +
                  g_ssim_loss * self.config.lambda_ssim)
        g_loss.backward()
        self.g_optimizer.step()

        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'g_gan_loss': g_gan_loss.item(),
            'g_perceptual_loss': g_perceptual_loss.item(),
            'g_ssim_loss': g_ssim_loss.item()
        }

    def validate(self):
        """
        Run validation and return metrics
        """
        if not self.val_loader:
            return {}

        self.generator.eval()
        self.discriminator.eval()

        val_metrics = []

        with torch.no_grad():
            for batch in self.val_loader:
                noisy = batch['noisy'].to(self.device)
                clean = batch['clean'].to(self.device)

                # Generate denoised image
                fake = self.generator(noisy)

                # Calculate metrics
                metrics_dict = self.metrics.calculate_metrics(fake, clean)
                val_metrics.append(metrics_dict)

        # Average metrics
        avg_metrics = {}
        for key in val_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in val_metrics])

        self.generator.train()
        self.discriminator.train()

        return avg_metrics

    def train(self, epochs, resume_path=None):
        """
        Full training loop
        Args:
            epochs: Number of epochs to train
            resume_path: Path to checkpoint to resume from (optional)
        """
        start_epoch = 0
        best_psnr = 0

        # Resume from checkpoint if provided
        if resume_path:
            start_epoch, best_psnr = self.load_checkpoint(resume_path)
            self.logger.info(
                f"Resuming from epoch {start_epoch} with best PSNR {best_psnr:.2f}")

        for epoch in range(start_epoch, epochs):
            epoch_losses = []

            # Training loop
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for i, batch in enumerate(pbar):
                losses = self.train_step(batch)
                epoch_losses.append(losses)

                # Update progress bar
                avg_losses = {k: np.mean([loss[k] for loss in epoch_losses])
                              for k in losses.keys()}
                pbar.set_postfix(avg_losses)

                # Log to tensorboard (every 100 iterations)
                if i % 100 == 0:
                    step = epoch * len(self.train_loader) + i
                    for k, v in losses.items():
                        self.writer.add_scalar(f'train/{k}', v, step)

                # Save samples (every 500 iterations)
                if i % 500 == 0:
                    self.save_samples(epoch, batch)

            # Update learning rates
            self.g_scheduler.step()
            self.d_scheduler.step()

            # Log learning rates
            self.writer.add_scalar('lr/generator',
                                   self.g_scheduler.get_last_lr()[0], epoch)
            self.writer.add_scalar('lr/discriminator',
                                   self.d_scheduler.get_last_lr()[0], epoch)

            # Validation
            val_metrics = self.validate()

            # Logging
            self.logger.info(f'Epoch {epoch+1}/{epochs}')
            self.logger.info(f'Train losses: {avg_losses}')
            if val_metrics:
                self.logger.info(f'Val metrics: {val_metrics}')
                # Log validation metrics to tensorboard
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f'val/{k}', v, epoch)

            # Save checkpoints
            self.save_checkpoint(
                'latest.pth', epoch=epoch, best_psnr=best_psnr)

            # Save best model
            if val_metrics.get('psnr', 0) > best_psnr:
                best_psnr = val_metrics['psnr']
                self.save_checkpoint(
                    'best_model.pth', epoch=epoch, best_psnr=best_psnr)

        self.writer.close()
        self.logger.info("Training completed!")

    def save_checkpoint(self, filename, epoch=None, best_psnr=None):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_scheduler_state_dict': self.g_scheduler.state_dict(),
            'd_scheduler_state_dict': self.d_scheduler.state_dict(),
            'best_psnr': best_psnr,
            'config': self.config
        }
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        """Load training checkpoint"""
        checkpoint = torch.load(path)

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(
            checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
        self.d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])

        epoch = checkpoint.get('epoch')
        best_psnr = checkpoint.get('best_psnr')

        self.logger.info(f"Loaded checkpoint from {path}")
        return epoch, best_psnr

    def save_samples(self, epoch, batch):
        """Save sample image outputs"""
        self.generator.eval()
        with torch.no_grad():
            noisy = batch['noisy'].to(self.device)
            clean = batch['clean'].to(self.device)
            fake = self.generator(noisy)

            # Create image grid
            img_grid = torch.cat([
                noisy[:8],
                fake[:8],
                clean[:8]
            ], dim=0)

            # Save grid
            grid = vutils.make_grid(img_grid, nrow=8, normalize=True)
            vutils.save_image(grid, self.sample_dir / f'epoch_{epoch:03d}.png')

            # Add to tensorboard
            self.writer.add_image('Samples/noisy_fake_clean', grid, epoch)

        self.generator.train()
