"""
Dataset class for Echocardiography Dehazing
Handles loading and preprocessing of clean, noisy, and ROI images
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import matplotlib.pyplot as plt


def create_dataloaders(config):
    """
    Create train, validation, and test dataloaders from config

    Args:
        config: Configuration object with dataset settings

    Returns:
        Dictionary of dataloaders and datasets
    """
    # Create dataset with transforms
    data_module = EchocardiographyDataModule(
        dataset_path=config.dataset_path,
        csv_path=config.csv_path,
        batch_size=config.batch_size,
        image_size=config.image_size,
        use_roi=config.use_roi,
        num_workers=config.num_workers,
        patient_split_ratio=config.patient_split_ratio
    )

    # Get dataloaders
    dataloaders = data_module.get_all_dataloaders()

    # Create datasets for each split
    datasets = {
        'train': EchocardiographyDataset(
            dataset_path=config.dataset_path,
            csv_path=config.csv_path,
            mode='train',
            transform=data_module.train_transform,
            use_roi=config.use_roi,
            patient_split=data_module.patient_split,
            image_size=config.image_size
        ),
        'val': EchocardiographyDataset(
            dataset_path=config.dataset_path,
            csv_path=config.csv_path,
            mode='val',
            transform=data_module.val_transform,
            use_roi=config.use_roi,
            patient_split=data_module.patient_split,
            image_size=config.image_size
        ),
        'test': EchocardiographyDataset(
            dataset_path=config.dataset_path,
            csv_path=config.csv_path,
            mode='test',
            transform=data_module.val_transform,
            use_roi=config.use_roi,
            patient_split=data_module.patient_split,
            image_size=config.image_size
        )
    }

    # Return dataloaders and datasets
    return dataloaders, datasets


def visualize_batch(batch, num_samples=4, save_path=None):
    """
    Visualize a batch of data with clean, noisy, and ROI images

    Args:
        batch: Dictionary with clean, noisy, and roi_mask tensors
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization (optional)
    """
    # Get data
    clean = batch['clean'].detach().cpu()
    noisy = batch['noisy'].detach().cpu()

    # Denormalize if needed (assuming [-1, 1] normalization)
    clean = clean * 0.5 + 0.5
    noisy = noisy * 0.5 + 0.5

    # Create figure
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 2 * num_samples))

    for i in range(min(num_samples, len(clean))):
        # Clean image
        if num_samples > 1:
            axes[i, 0].imshow(clean[i].squeeze(), cmap='gray')
            axes[i, 0].set_title('Clean')
            axes[i, 0].axis('off')

            # Noisy image
            axes[i, 1].imshow(noisy[i].squeeze(), cmap='gray')
            axes[i, 1].set_title('Noisy')
            axes[i, 1].axis('off')
        else:
            axes[0].imshow(clean[i].squeeze(), cmap='gray')
            axes[0].set_title('Clean')
            axes[0].axis('off')

            # Noisy image
            axes[1].imshow(noisy[i].squeeze(), cmap='gray')
            axes[1].set_title('Noisy')
            axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


class EchocardiographyDataset(Dataset):
    """
    Dataset class for echocardiography dehazing
    Loads clean, noisy, and ROI images based on the dataset mapping CSV
    """

    def __init__(self,
                 dataset_path,
                 csv_path,
                 mode='train',
                 transform=None,
                 use_roi=True,
                 patient_split=None,
                 image_size=(256, 256)):
        """
        Args:
            dataset_path: Path to the dataset folder containing clean, noisy, noisy_roi
            csv_path: Path to the dataset mapping CSV file
            mode: 'train', 'val', or 'test'
            transform: Optional transform to be applied
            use_roi: Whether to load ROI annotations
            patient_split: Dictionary with train/val/test patient lists
            image_size: Target image size (H, W)
        """
        self.dataset_path = Path(dataset_path)
        self.mode = mode
        self.transform = transform
        self.use_roi = use_roi
        self.image_size = image_size

        # Load dataset mapping
        self.df = pd.read_csv(csv_path)

        # Filter data based on mode and patient split
        if patient_split is not None:
            if mode in patient_split:
                patient_ids = patient_split[mode]
                self.df = self.df[self.df['patient_id'].isin(patient_ids)]

        # Filter based on data availability
        if mode == 'train':
            # Use clean-noisy pairs for training
            self.df = self.df[self.df['clean_noisy_pair'] == True]
        elif mode == 'val':
            # Use triplets for validation if available
            if use_roi:
                self.df = self.df[self.df['complete_triplet'] == True]
            else:
                self.df = self.df[self.df['clean_noisy_pair'] == True]
        elif mode == 'test':
            # Use all available pairs for testing
            self.df = self.df[self.df['clean_noisy_pair'] == True]

        # Reset index
        self.df = self.df.reset_index(drop=True)

        print(f"{mode.upper()} dataset: {len(self.df)} samples")

        # Default transforms if none provided
        if self.transform is None:
            self.transform = self.get_default_transform()

    def get_default_transform(self):
        """Get default image transforms"""
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ]
        return transforms.Compose(transform_list)

    def load_image(self, image_path):
        """Load and preprocess image"""
        if not os.path.exists(image_path):
            # Return zero image if file doesn't exist
            return np.zeros(self.image_size, dtype=np.uint8)

        # Load image as grayscale
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return np.zeros(self.image_size, dtype=np.uint8)

        # Resize image
        image = cv2.resize(image, self.image_size)

        # Normalize to 0-255 range
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        return image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        row = self.df.iloc[idx]

        # Construct image paths
        clean_path = self.dataset_path / "clean" / row['clean_image']
        noisy_path = self.dataset_path / "noisy" / row['noisy_image']

        # Load clean and noisy images
        clean_img = self.load_image(clean_path)
        noisy_img = self.load_image(noisy_path)

        # Convert to PIL Images for transforms
        clean_img = Image.fromarray(clean_img)
        noisy_img = Image.fromarray(noisy_img)

        # Apply transforms
        if self.transform:
            clean_img = self.transform(clean_img)
            noisy_img = self.transform(noisy_img)

        sample = {
            'clean': clean_img,
            'noisy': noisy_img,
            'patient_id': row['patient_id'],
            'frame_number': row['frame_number'],
            'clean_filename': row['clean_image'],
            'noisy_filename': row['noisy_image']
        }

        # Load ROI if available and requested
        if self.use_roi and row['has_roi']:
            roi_path = self.dataset_path / "noisy_roi" / row['noisy_roi_image']
            roi_img = self.load_image(roi_path)
            roi_img = Image.fromarray(roi_img)

            # ROI transform (no normalization, just resize and to tensor)
            roi_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()
            ])
            roi_img = roi_transform(roi_img)

            # Binarize ROI (threshold at 0.5)
            roi_img = (roi_img > 0.5).float()

            sample['roi'] = roi_img
            sample['roi_filename'] = row['noisy_roi_image']
            sample['has_roi'] = True
        else:
            sample['roi'] = torch.zeros(1, *self.image_size)
            sample['roi_filename'] = ''
            sample['has_roi'] = False

        return sample


class EchocardiographyDataModule:
    """
    Data module for handling dataset creation and loading
    """

    def __init__(self,
                 dataset_path,
                 csv_path,
                 batch_size=16,
                 num_workers=4,
                 image_size=(256, 256),
                 use_roi=True,
                 patient_split_ratio=(0.7, 0.2, 0.1)):
        """
        Args:
            dataset_path: Path to dataset folder
            csv_path: Path to dataset mapping CSV
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            image_size: Target image size
            use_roi: Whether to use ROI annotations
            patient_split_ratio: (train, val, test) split ratios
        """
        self.dataset_path = dataset_path
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.use_roi = use_roi

        # Load dataset mapping and create patient split
        self.df = pd.read_csv(csv_path)
        self.patient_split = self.create_patient_split(patient_split_ratio)

        # Create transforms
        self.train_transform = self.get_train_transform()
        self.val_transform = self.get_val_transform()

    def create_patient_split(self, split_ratio):
        """Create patient-wise train/val/test split"""
        # Get unique patients with noisy data
        patients_with_noisy = self.df[self.df['has_noisy']
                                      == True]['patient_id'].unique()
        patients_with_noisy = sorted(patients_with_noisy)

        n_patients = len(patients_with_noisy)
        n_train = int(n_patients * split_ratio[0])
        n_val = int(n_patients * split_ratio[1])

        # Random split (you can make this reproducible with np.random.seed)
        np.random.seed(42)  # For reproducibility
        shuffled_patients = np.random.permutation(patients_with_noisy)

        train_patients = shuffled_patients[:n_train].tolist()
        val_patients = shuffled_patients[n_train:n_train+n_val].tolist()
        test_patients = shuffled_patients[n_train+n_val:].tolist()

        patient_split = {
            'train': train_patients,
            'val': val_patients,
            'test': test_patients
        }

        print(f"Patient split:")
        print(f"  Train: {len(train_patients)} patients - {train_patients}")
        print(f"  Val: {len(val_patients)} patients - {val_patients}")
        print(f"  Test: {len(test_patients)} patients - {test_patients}")

        return patient_split

    def get_train_transform(self):
        """Get training transforms with augmentation"""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def get_val_transform(self):
        """Get validation transforms (no augmentation)"""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def get_dataloader(self, mode='train', shuffle=None):
        """Get data loader for specified mode"""
        if shuffle is None:
            shuffle = (mode == 'train')

        transform = self.train_transform if mode == 'train' else self.val_transform

        dataset = EchocardiographyDataset(
            dataset_path=self.dataset_path,
            csv_path=self.csv_path,
            mode=mode,
            transform=transform,
            use_roi=self.use_roi,
            patient_split=self.patient_split,
            image_size=self.image_size
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=(mode == 'train')
        )

        return dataloader

    def get_all_dataloaders(self):
        """Get all data loaders"""
        return {
            'train': self.get_dataloader('train'),
            'val': self.get_dataloader('val'),
            'test': self.get_dataloader('test')
        }


def visualize_batch(batch, num_samples=4):
    """Visualize a batch of samples"""
    fig, axes = plt.subplots(3, num_samples, figsize=(16, 12))

    for i in range(min(num_samples, batch['clean'].size(0))):
        # Denormalize images
        clean_img = batch['clean'][i].squeeze().cpu().numpy()
        noisy_img = batch['noisy'][i].squeeze().cpu().numpy()

        # Convert from [-1, 1] to [0, 1]
        clean_img = (clean_img + 1) / 2
        noisy_img = (noisy_img + 1) / 2

        # Plot clean image
        axes[0, i].imshow(clean_img, cmap='gray')
        axes[0, i].set_title(f'Clean - Patient {batch["patient_id"][i]}')
        axes[0, i].axis('off')

        # Plot noisy image
        axes[1, i].imshow(noisy_img, cmap='gray')
        axes[1, i].set_title(f'Noisy - Frame {batch["frame_number"][i]}')
        axes[1, i].axis('off')

        # Plot ROI if available
        if batch['has_roi'][i]:
            roi_img = batch['roi'][i].squeeze().cpu().numpy()
            axes[2, i].imshow(roi_img, cmap='gray')
            axes[2, i].set_title('ROI')
        else:
            axes[2, i].text(0.5, 0.5, 'No ROI', ha='center',
                            va='center', transform=axes[2, i].transAxes)
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()


def test_dataset():
    """Test dataset functionality"""
    # Paths (adjust as needed)
    dataset_path = "../Dataset"
    csv_path = "../dataset_mapping.csv"

    # Create data module
    data_module = EchocardiographyDataModule(
        dataset_path=dataset_path,
        csv_path=csv_path,
        batch_size=4,
        num_workers=0,  # Set to 0 for testing
        image_size=(256, 256),
        use_roi=True
    )

    # Get data loaders
    dataloaders = data_module.get_all_dataloaders()

    # Test train dataloader
    train_loader = dataloaders['train']
    print(f"Train loader: {len(train_loader)} batches")

    # Get a batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Clean shape: {batch['clean'].shape}")
    print(f"Noisy shape: {batch['noisy'].shape}")
    print(f"ROI shape: {batch['roi'].shape}")
    print(f"Has ROI: {batch['has_roi']}")

    # Visualize batch
    visualize_batch(batch)


if __name__ == "__main__":
    test_dataset()
