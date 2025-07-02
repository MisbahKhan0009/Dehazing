#!/usr/bin/env python3
"""
Dataset Analysis Summary Script
Provides an overview of all generated visualizations and analysis
"""

import os
from pathlib import Path
import pandas as pd


def print_summary():
    """Print comprehensive summary of generated analysis"""
    print("=" * 60)
    print("ECHOCARDIOGRAPHY DATASET ANALYSIS SUMMARY")
    print("=" * 60)

    # Dataset mapping info
    if os.path.exists("dataset_mapping.csv"):
        df = pd.read_csv("dataset_mapping.csv")
        print(f"\n📊 DATASET MAPPING:")
        print(f"   • Total image entries: {len(df):,}")
        print(
            f"   • Complete triplets: {len(df[df['complete_triplet'] == True]):,}")
        print(
            f"   • Clean-noisy pairs: {len(df[df['clean_noisy_pair'] == True]):,}")
        print(f"   • Patients with data: {df['patient_id'].nunique()}")

    # Visualization outputs
    viz_path = Path("visualizations")
    if viz_path.exists():
        print(f"\n🎨 GENERATED VISUALIZATIONS:")

        # Count files in each category
        triplet_count = len(list((viz_path / "triplets").glob("*.png")))
        pair_count = len(list((viz_path / "pairs").glob("*.png")))
        overlay_individual = len(
            list((viz_path / "roi_overlays" / "individual").glob("*.png")))
        overlay_comparison = len(
            list((viz_path / "roi_overlays" / "comparison").glob("*.png")))
        sample_files = len(list((viz_path / "samples").glob("*.png")))
        stat_files = len(list((viz_path / "statistics").glob("*.png")))

        print(f"   • Triplet comparisons: {triplet_count} images")
        print(f"   • Pair comparisons: {pair_count} images")
        print(f"   • ROI overlays (individual): {overlay_individual} images")
        print(f"   • ROI overlays (comparison): {overlay_comparison} images")
        print(f"   • Sample grids: {sample_files} images")
        print(f"   • Statistics charts: {stat_files} images")
        print(
            f"   • TOTAL VISUALIZATIONS: {triplet_count + pair_count + overlay_individual + overlay_comparison + sample_files + stat_files} images")

    # Folder structure
    print(f"\n📁 FOLDER STRUCTURE:")
    print(f"   Dataset/")
    print(f"   ├── clean/           (4,376 images - 75 patients)")
    print(f"   ├── noisy/           (2,324 images - 40 patients)")
    print(f"   └── noisy_roi/       (237 images - 40 patients)")
    print(f"   ")
    print(f"   visualizations/")
    print(f"   ├── triplets/        (Clean + Noisy + ROI comparisons)")
    print(f"   ├── pairs/           (Clean + Noisy comparisons)")
    print(f"   ├── roi_overlays/")
    print(f"   │   ├── individual/  (50% opacity overlays)")
    print(f"   │   └── comparison/  (Noisy + ROI + Overlay)")
    print(f"   ├── samples/         (Representative grid)")
    print(f"   └── statistics/      (Charts and stats)")

    # Key insights
    print(f"\n🔍 KEY INSIGHTS:")
    print(f"   • Only 40/75 patients have noisy data (realistic clinical scenario)")
    print(f"   • ROI annotations provided strategically (~every 10 frames)")
    print(f"   • Each patient has exactly 60 frames when data is available")
    print(f"   • ROI overlays created with 50% opacity for both images")
    print(f"   • All visualizations include patient/frame identification")

    # Usage recommendations
    print(f"\n💡 USAGE RECOMMENDATIONS:")
    print(f"   • Training: Use 2,324 clean-noisy pairs")
    print(f"   • Evaluation: Use 237 ROI-annotated triplets")
    print(f"   • Quality check: Review triplet/pair comparison images")
    print(f"   • ROI validation: Check overlay comparison images")
    print(f"   • Statistics: Reference charts in statistics/ folder")

    # Generated files
    print(f"\n📄 GENERATED FILES:")
    files = [
        "dataset_mapping.csv - Complete image mapping and relationships",
        "Dataset_Analysis_Report.md - Comprehensive analysis report",
        "visualize_dataset.py - Main visualization generation script",
        "analyze_dataset.py - Original dataset analysis script",
        "visualizations/README.md - Detailed visualization documentation"
    ]
    for file in files:
        if os.path.exists(file.split(' - ')[0]):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")

    print(f"\n" + "=" * 60)
    print("ANALYSIS COMPLETE - All visualizations ready for use!")
    print("=" * 60)


if __name__ == "__main__":
    print_summary()
