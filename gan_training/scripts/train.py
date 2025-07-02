#!/usr/bin/env python3
"""
Main training script for Echocardiography Dehazing GAN
"""

from models.discriminator import (ConditionalPatchGANDiscriminator,
                                  MultiScaleDiscriminator,
                                  SpectralNormDiscriminator,
                                  AttentionDiscriminator)
from training.utils import set_seed, save_checkpoint, load_checkpoint, generate_sample_images
from evaluation.metrics import MetricsCalculator, MetricsLogger
from data.dataset import EchocardiographyDataModule
from models.losses import GANLoss, CombinedLoss, GradientPenaltyLoss
from models.generator import UNetGenerator, EnhancedUNetGenerator
from config.config import get_config
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class EchoGANTrainer:
    """Main trainer class for Echocardiography GAN"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else 'cpu')

        # Set random seed for reproducibility
        set_seed(config.seed)

        # Initialize components
        self.setup_models()
        self.setup_optimizers()
        self.setup_losses()
        self.setup_data()
        self.setup_metrics()
        self.setup_logging()

        # Training state
        self.current_epoch = 0
        self.best_metric_value = -np.inf if config.best_metric != 'l1_error' else np.inf
        self.patience_counter = 0

        print(f"Trainer initialized on device: {self.device}")

    def setup_models(self):
        """Initialize generator and discriminator models"""
        print("Setting up models...")

        # Generator
        if self.config.generator_type == "unet":
            self.generator = UNetGenerator(
                **self.config.get_generator_config())
        elif self.config.generator_type == "enhanced_unet":
            self.generator = EnhancedUNetGenerator(
                **self.config.get_generator_config())
        else:
            raise ValueError(
                f"Unknown generator type: {self.config.generator_type}")

        # Discriminator
        disc_config = self.config.get_discriminator_config()
        if self.config.discriminator_type == "conditional_patch":
            self.discriminator = ConditionalPatchGANDiscriminator(
                **disc_config)
        elif self.config.discriminator_type == "multiscale":
            self.discriminator = MultiScaleDiscriminator(**disc_config)
        elif self.config.discriminator_type == "spectral":
            self.discriminator = SpectralNormDiscriminator(**disc_config)
        elif self.config.discriminator_type == "attention":
            self.discriminator = AttentionDiscriminator(**disc_config)
        else:
            raise ValueError(
                f"Unknown discriminator type: {self.config.discriminator_type}")

        # Move to device
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        # Print model info
        gen_params = sum(p.numel() for p in self.generator.parameters())
        disc_params = sum(p.numel() for p in self.discriminator.parameters())
        print(f"Generator parameters: {gen_params:,}")
        print(f"Discriminator parameters: {disc_params:,}")

    def setup_optimizers(self):
        """Initialize optimizers and schedulers"""
        opt_config = self.config.get_optimizer_config()

        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            **opt_config['generator']
        )

        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            **opt_config['discriminator']
        )

        # Learning rate schedulers
        if self.config.use_lr_scheduler:
            if self.config.scheduler_type == 'cosine':
                self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer_g,
                    T_max=self.config.num_epochs,
                    eta_min=self.config.cosine_eta_min
                )
                self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer_d,
                    T_max=self.config.num_epochs,
                    eta_min=self.config.cosine_eta_min
                )
            elif self.config.scheduler_type == 'step':
                self.scheduler_g = optim.lr_scheduler.MultiStepLR(
                    self.optimizer_g,
                    milestones=self.config.lr_decay_epochs,
                    gamma=self.config.lr_decay_gamma
                )
                self.scheduler_d = optim.lr_scheduler.MultiStepLR(
                    self.optimizer_d,
                    milestones=self.config.lr_decay_epochs,
                    gamma=self.config.lr_decay_gamma
                )
        else:
            self.scheduler_g = None
            self.scheduler_d = None

    def setup_losses(self):
        """Initialize loss functions"""
        self.gan_loss = GANLoss(self.config.gan_mode).to(self.device)
        self.combined_loss = CombinedLoss(
            **self.config.get_loss_config()).to(self.device)

        if self.config.gan_mode == 'wgangp':
            self.gradient_penalty = GradientPenaltyLoss(
                self.config.gradient_penalty_lambda).to(self.device)
        else:
            self.gradient_penalty = None

    def setup_data(self):
        """Initialize data loaders"""
        print("Setting up data loaders...")

        self.data_module = EchocardiographyDataModule(
            dataset_path=self.config.dataset_path,
            csv_path=self.config.csv_path,
            **self.config.get_dataloader_config()
        )

        self.dataloaders = self.data_module.get_all_dataloaders()

        print(f"Train batches: {len(self.dataloaders['train'])}")
        print(f"Val batches: {len(self.dataloaders['val'])}")
        print(f"Test batches: {len(self.dataloaders['test'])}")

    def setup_metrics(self):
        """Initialize metrics calculator and logger"""
        self.metrics_calculator = MetricsCalculator(self.device)
        self.metrics_logger = MetricsLogger()

    def setup_logging(self):
        """Initialize logging"""
        if self.config.use_tensorboard:
            self.writer = SummaryWriter(self.config.log_dir)
        else:
            self.writer = None

    def train_discriminator(self, batch):
        """Train discriminator for one step"""
        self.optimizer_d.zero_grad()

        noisy = batch['noisy'].to(self.device)
        clean = batch['clean'].to(self.device)

        # Generate fake images
        with torch.no_grad():
            fake = self.generator(noisy)

        # Real loss
        if self.config.discriminator_type == "multiscale":
            real_pred = self.discriminator(noisy, clean)
            fake_pred = self.discriminator(noisy, fake)

            # Multi-scale loss
            loss_d_real = 0
            loss_d_fake = 0
            for real_p, fake_p in zip(real_pred, fake_pred):
                loss_d_real += self.gan_loss(real_p, True)
                loss_d_fake += self.gan_loss(fake_p, False)

            loss_d_real /= len(real_pred)
            loss_d_fake /= len(fake_pred)
        else:
            real_pred = self.discriminator(noisy, clean)
            fake_pred = self.discriminator(noisy, fake)

            loss_d_real = self.gan_loss(real_pred, True)
            loss_d_fake = self.gan_loss(fake_pred, False)

        # Total discriminator loss
        loss_d = (loss_d_real + loss_d_fake) * 0.5

        # Gradient penalty for WGAN-GP
        if self.gradient_penalty is not None:
            gp_loss = self.gradient_penalty(
                self.discriminator, clean, fake, noisy)
            loss_d += gp_loss

        loss_d.backward()
        self.optimizer_d.step()

        return {
            'loss_d': loss_d.item(),
            'loss_d_real': loss_d_real.item(),
            'loss_d_fake': loss_d_fake.item()
        }

    def train_generator(self, batch):
        """Train generator for one step"""
        self.optimizer_g.zero_grad()

        noisy = batch['noisy'].to(self.device)
        clean = batch['clean'].to(self.device)
        roi = batch['roi'].to(self.device) if batch['has_roi'][0] else None

        # Generate fake images
        fake = self.generator(noisy)

        # Adversarial loss
        if self.config.discriminator_type == "multiscale":
            fake_pred = self.discriminator(noisy, fake)
            loss_g_gan = 0
            for pred in fake_pred:
                loss_g_gan += self.gan_loss(pred, True)
            loss_g_gan /= len(fake_pred)
        else:
            fake_pred = self.discriminator(noisy, fake)
            loss_g_gan = self.gan_loss(fake_pred, True)

        # Combined reconstruction loss
        loss_g_combined, loss_dict = self.combined_loss(fake, clean, roi)

        # Total generator loss
        loss_g = self.config.lambda_gan * loss_g_gan + loss_g_combined

        loss_g.backward()
        self.optimizer_g.step()

        # Combine loss dictionaries
        g_losses = {
            'loss_g': loss_g.item(),
            'loss_g_gan': loss_g_gan.item(),
            'loss_g_combined': loss_g_combined.item()
        }
        g_losses.update({f'loss_g_{k}': v for k, v in loss_dict.items()})

        return g_losses, fake

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()

        epoch_losses = {}

        pbar = tqdm(self.dataloaders['train'], desc=f'Epoch {epoch}')
        for i, batch in enumerate(pbar):
            # Train discriminator
            if i % self.config.discriminator_update_freq == 0:
                d_losses = self.train_discriminator(batch)

                # Update epoch losses
                for key, value in d_losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = []
                    epoch_losses[key].append(value)

            # Train generator
            if i % self.config.generator_update_freq == 0:
                g_losses, fake_images = self.train_generator(batch)

                # Update epoch losses
                for key, value in g_losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = []
                    epoch_losses[key].append(value)

            # Update progress bar
            if i % self.config.log_interval == 0:
                avg_losses = {k: np.mean(v[-10:])
                              for k, v in epoch_losses.items() if v}
                pbar.set_postfix(avg_losses)

                # Log to tensorboard
                if self.writer:
                    step = epoch * len(self.dataloaders['train']) + i
                    for key, value in avg_losses.items():
                        self.writer.add_scalar(f'train/{key}', value, step)

        # Calculate epoch averages
        epoch_avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}

        return epoch_avg_losses

    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.generator.eval()
        self.discriminator.eval()

        all_metrics = []

        with torch.no_grad():
            for batch in tqdm(self.dataloaders['val'], desc='Validation'):
                noisy = batch['noisy'].to(self.device)
                clean = batch['clean'].to(self.device)
                roi = batch['roi'].to(
                    self.device) if batch['has_roi'][0] else None

                # Generate images
                fake = self.generator(noisy)

                # Calculate metrics
                metrics = self.metrics_calculator.calculate_all_metrics(
                    fake, clean, roi, denormalize=True
                )
                all_metrics.append(metrics)

        # Average metrics across batches
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if not np.isnan(m[key])]
            avg_metrics[key] = np.mean(values) if values else 0.0

        # Log metrics
        self.metrics_logger.log_metrics(avg_metrics, epoch, 'val')

        if self.writer:
            for key, value in avg_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)

        return avg_metrics

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'config': self.config.__dict__,
            'best_metric_value': self.best_metric_value
        }

        if self.scheduler_g:
            checkpoint['scheduler_g_state_dict'] = self.scheduler_g.state_dict()
            checkpoint['scheduler_d_state_dict'] = self.scheduler_d.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.config.checkpoint_dir / \
            f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.config.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model saved at epoch {epoch}")

        # Save latest checkpoint
        latest_path = self.config.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(
            checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

        if self.scheduler_g and 'scheduler_g_state_dict' in checkpoint:
            self.scheduler_g.load_state_dict(
                checkpoint['scheduler_g_state_dict'])
            self.scheduler_d.load_state_dict(
                checkpoint['scheduler_d_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_metric_value = checkpoint.get('best_metric_value', -np.inf)

        print(f"Checkpoint loaded from epoch {self.current_epoch}")

    def train(self):
        """Main training loop"""
        print("Starting training...")

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_losses = self.train_epoch(epoch)

            # Update learning rate
            if self.scheduler_g:
                self.scheduler_g.step()
                self.scheduler_d.step()

            # Validate
            if epoch % self.config.eval_interval == 0:
                val_metrics = self.validate_epoch(epoch)

                # Print metrics
                print(f"\nEpoch {epoch} Results:")
                self.metrics_calculator.print_metrics(
                    train_losses, "Training Losses")
                self.metrics_calculator.print_metrics(
                    val_metrics, "Validation Metrics")

                # Check for best model
                current_metric = val_metrics.get(self.config.best_metric, 0)
                is_best = False

                if self.config.best_metric in ['l1_error', 'l2_error', 'ks_statistic', 'lpips']:
                    # Lower is better
                    if current_metric < self.best_metric_value:
                        self.best_metric_value = current_metric
                        is_best = True
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                else:
                    # Higher is better
                    if current_metric > self.best_metric_value:
                        self.best_metric_value = current_metric
                        is_best = True
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1

                # Early stopping
                if self.config.use_early_stopping and self.patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

                # Save checkpoint
                if epoch % self.config.save_interval == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)

            # Generate sample images
            if epoch % self.config.generate_samples_interval == 0:
                self.generate_sample_images(epoch)

        print("Training completed!")

        # Save final metrics
        metrics_path = self.config.log_dir / 'metrics_history.json'
        self.metrics_logger.save_metrics(metrics_path)

        if self.writer:
            self.writer.close()

    def generate_sample_images(self, epoch):
        """Generate and save sample images"""
        self.generator.eval()

        with torch.no_grad():
            # Get a batch from validation set
            batch = next(iter(self.dataloaders['val']))
            noisy = batch['noisy'][:self.config.num_sample_images].to(
                self.device)
            clean = batch['clean'][:self.config.num_sample_images].to(
                self.device)

            # Generate images
            fake = self.generator(noisy)

            # Save images
            output_dir = self.config.results_dir / f'epoch_{epoch}'
            output_dir.mkdir(exist_ok=True)

            generate_sample_images(
                noisy.cpu(), clean.cpu(), fake.cpu(),
                output_dir, epoch, self.config.num_sample_images
            )

        self.generator.train()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Train Echocardiography Dehazing GAN')
    parser.add_argument('--config', type=str, default='default',
                        help='Configuration name (default, quick_test, high_quality, medical_focused)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device to use')

    args = parser.parse_args()

    # Get configuration
    config = get_config(args.config)

    # Set GPU device
    if torch.cuda.is_available() and args.gpu >= 0:
        config.device = f'cuda:{args.gpu}'
    else:
        config.device = 'cpu'

    # Print configuration
    config.print_config()

    # Save configuration
    config_path = config.log_dir / 'config.json'
    config.save_config(config_path)

    # Initialize trainer
    trainer = EchoGANTrainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
