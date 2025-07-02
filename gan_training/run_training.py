#!/usr/bin/env python3
"""
Simple runner script for GAN training
Run this to start training with default or custom configuration
"""

import sys
import os
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def install_requirements():
    """Install required packages"""
    import subprocess

    print("Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    return True


def check_dataset():
    """Check if dataset is available"""
    dataset_path = project_root.parent / "Dataset"
    csv_path = project_root.parent / "dataset_mapping.csv"

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return False

    if not csv_path.exists():
        print(f"Error: Dataset mapping CSV not found at {csv_path}")
        return False

    # Check folders
    clean_path = dataset_path / "clean"
    noisy_path = dataset_path / "noisy"

    if not clean_path.exists():
        print(f"Error: Clean images folder not found at {clean_path}")
        return False

    if not noisy_path.exists():
        print(f"Error: Noisy images folder not found at {noisy_path}")
        return False

    print("Dataset check passed!")
    return True


def run_training(config_name="default", install_deps=False):
    """Run training with specified configuration"""

    if install_deps:
        if not install_requirements():
            return

    if not check_dataset():
        return

    # Import training script
    try:
        from training.train import main
        from config.config import get_config

        # Set configuration
        os.environ['TRAINING_CONFIG'] = config_name

        print(f"Starting training with {config_name} configuration...")
        main()

    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all dependencies are installed.")
        print("Run with --install-deps flag to install requirements.")
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Run GAN training for echocardiography dehazing")

    parser.add_argument(
        '--config',
        type=str,
        default='default',
        choices=['default', 'quick_test', 'high_quality', 'medical_focused'],
        help='Configuration to use for training'
    )

    parser.add_argument(
        '--install-deps',
        action='store_true',
        help='Install required dependencies before training'
    )

    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check dataset and dependencies, do not start training'
    )

    args = parser.parse_args()

    if args.check_only:
        print("Checking dataset and dependencies...")
        check_dataset()
        return

    print("=" * 60)
    print("ECHOCARDIOGRAPHY DEHAZING GAN TRAINING")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Install dependencies: {args.install_deps}")
    print("=" * 60)

    run_training(args.config, args.install_deps)


if __name__ == "__main__":
    main()
