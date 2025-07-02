#!/usr/bin/env python3
"""
Test script to verify the GAN training setup
Tests dataset loading, model creation, and basic functionality
"""

import sys
import os
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def test_config():
    """Test configuration loading"""
    print("Testing configuration...")
    try:
        from config.config import Config, get_config

        config = Config()
        print(f"✓ Default config loaded")
        print(f"  - Dataset path: {config.dataset_path}")
        print(f"  - Batch size: {config.batch_size}")
        print(f"  - Image size: {config.image_size}")

        # Test different configs
        quick_config = get_config("quick_test")
        print(
            f"✓ Quick test config loaded (epochs: {quick_config.num_epochs})")

        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def test_dataset():
    """Test dataset loading"""
    print("\nTesting dataset loading...")
    try:
        from config.config import Config
        from data.dataset import EchocardiographyDataset
        import pandas as pd

        config = Config()

        # Check if CSV exists
        if not config.csv_path.exists():
            print(f"✗ Dataset CSV not found: {config.csv_path}")
            return False

        # Load CSV
        df = pd.read_csv(config.csv_path)
        print(f"✓ Dataset CSV loaded ({len(df)} samples)")

        # Test dataset creation (without actual data loading)
        try:
            dataset = EchocardiographyDataset(
                dataset_path=config.dataset_path,
                csv_path=config.csv_path,
                mode='train'
            )
            print(f"✓ Train dataset created ({len(dataset)} samples)")
        except Exception as e:
            print(f"✗ Dataset creation failed: {e}")
            return False

        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        traceback.print_exc()
        return False


def test_models():
    """Test model creation"""
    print("\nTesting model creation...")
    try:
        from config.config import Config

        config = Config()

        # Test imports
        from models.generator import UNetGenerator
        from models.discriminator import ConditionalPatchGANDiscriminator
        print("✓ Model imports successful")

        # Create models (without loading to GPU)
        generator = UNetGenerator(
            n_channels=1,
            n_classes=1,
            use_attention=config.use_attention,
            bilinear=True
        )

        discriminator = ConditionalPatchGANDiscriminator(
            input_channels=1,  # noisy input
            output_channels=1,  # clean/generated
            ndf=config.ndf,
            n_layers=config.discriminator_layers
        )

        print("✓ Models created successfully")
        print(
            f"  - Generator params: {sum(p.numel() for p in generator.parameters()):,}")
        print(
            f"  - Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        traceback.print_exc()
        return False


def test_losses():
    """Test loss functions"""
    print("\nTesting loss functions...")
    try:
        from models.losses import GANLoss, PerceptualLoss, SSIMLoss
        print("✓ Loss function imports successful")

        # Test loss creation
        gan_loss = GANLoss('lsgan')
        print("✓ GAN loss created")

        # Test other losses (might fail if dependencies not installed)
        try:
            perceptual_loss = PerceptualLoss()
            print("✓ Perceptual loss created")
        except:
            print("⚠ Perceptual loss creation failed (may need torchvision)")

        try:
            ssim_loss = SSIMLoss()
            print("✓ SSIM loss created")
        except:
            print("⚠ SSIM loss creation failed (may need additional dependencies)")

        return True
    except Exception as e:
        print(f"✗ Loss test failed: {e}")
        traceback.print_exc()
        return False


def test_metrics():
    """Test evaluation metrics"""
    print("\nTesting evaluation metrics...")
    try:
        from evaluation.metrics import MetricsCalculator
        print("✓ Metrics imports successful")

        # Create calculator
        calculator = MetricsCalculator(device='cpu')
        print("✓ Metrics calculator created")

        # Test with dummy data
        import numpy as np
        img1 = np.random.rand(1, 1, 64, 64).astype(np.float32)
        img2 = np.random.rand(1, 1, 64, 64).astype(np.float32)

        # Test basic metrics
        psnr_val = calculator.calculate_psnr(img1, img2)
        ssim_val = calculator.calculate_ssim(img1, img2)
        print(
            f"✓ Basic metrics calculated (PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.3f})")

        return True
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        traceback.print_exc()
        return False


def test_file_structure():
    """Test file structure"""
    print("\nTesting file structure...")

    required_files = [
        "config/config.py",
        "data/dataset.py",
        "models/generator.py",
        "models/discriminator.py",
        "models/losses.py",
        "evaluation/metrics.py",
        "training/train.py",
        "requirements.txt"
    ]

    required_dirs = [
        "checkpoints",
        "logs",
        "results"
    ]

    all_good = True

    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            all_good = False

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (missing)")
            all_good = False

    return all_good


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("GAN TRAINING SETUP VERIFICATION")
    print("=" * 60)

    tests = [
        ("File Structure", test_file_structure),
        ("Configuration", test_config),
        ("Dataset", test_dataset),
        ("Models", test_models),
        ("Loss Functions", test_losses),
        ("Metrics", test_metrics)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{len(results)} tests")

    if passed == len(results):
        print("\n🎉 All tests passed! You're ready to start training.")
    else:
        print(
            f"\n⚠️  {len(results) - passed} test(s) failed. Please check the issues above.")
        print("You may need to install dependencies or check file paths.")


if __name__ == "__main__":
    run_all_tests()
