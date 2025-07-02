#!/usr/bin/env python3
"""
Quick preview script to display generated visualizations
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import os


def show_sample_visualizations():
    """Display sample visualizations in a grid"""
    viz_path = Path("visualizations")

    if not viz_path.exists():
        print("Visualizations folder not found. Please run visualize_dataset.py first.")
        return

    # Get sample files
    triplet_files = list((viz_path / "triplets").glob("*.png"))
    pair_files = list((viz_path / "pairs").glob("*.png"))
    overlay_comp_files = list(
        (viz_path / "roi_overlays" / "comparison").glob("*.png"))

    if not (triplet_files and pair_files and overlay_comp_files):
        print("Sample files not found. Please ensure visualizations are generated.")
        return

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(20, 15))
    fig.suptitle('Echocardiography Dataset Visualizations',
                 fontsize=16, fontweight='bold')

    # Show triplet comparison
    triplet_img = mpimg.imread(str(triplet_files[0]))
    axes[0].imshow(triplet_img)
    axes[0].set_title(
        f"Triplet Comparison: Clean, Noisy, and ROI", fontsize=14)
    axes[0].axis('off')

    # Show pair comparison
    pair_img = mpimg.imread(str(pair_files[0]))
    axes[1].imshow(pair_img)
    axes[1].set_title(f"Pair Comparison: Clean and Noisy", fontsize=14)
    axes[1].axis('off')

    # Show ROI overlay comparison
    overlay_img = mpimg.imread(str(overlay_comp_files[0]))
    axes[2].imshow(overlay_img)
    axes[2].set_title(
        f"ROI Overlay: Noisy, ROI, and 50% Opacity Overlay", fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("sample_visualizations_preview.png",
                dpi=300, bbox_inches='tight')
    plt.show()

    print("\nGenerated visualizations saved in 'visualizations/' folder:")
    print(f"- {len(triplet_files)} triplet comparison images")
    print(f"- {len(pair_files)} pair comparison images")
    print(f"- {len(overlay_comp_files)} ROI overlay comparison images")
    print(f"- Individual overlay images in roi_overlays/individual/")
    print(f"- Dataset statistics in statistics/")


if __name__ == "__main__":
    show_sample_visualizations()
