#!/usr/bin/env python3
"""
Dataset Visualization Script
Creates pair, triplet, and overlay visualizations for the echocardiography dataset
"""

import os
import glob
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def load_and_normalize_image(image_path):
    """Load and normalize image to 0-255 range"""
    if not os.path.exists(image_path):
        return None
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Normalize to 0-255 range
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img


def create_comparison_image(images, labels, title, patient_id, frame_num):
    """Create a comparison image with the given images and labels"""
    if not images or not all(img is not None for img in images):
        return None

    # Get image dimensions from first image
    if len(images[0].shape) == 3:  # RGB image
        height, width, _ = images[0].shape
    else:  # Grayscale image
        height, width = images[0].shape

    n_images = len(images)
    spacing = 20

    # Create canvas for the combined image
    combined = np.zeros(
        (height, width * n_images + spacing * (n_images-1), 3), dtype=np.uint8)

    # Place each image with spacing
    for i, img in enumerate(images):
        # Convert to RGB if needed
        if len(img.shape) == 2:  # Grayscale
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:  # Already RGB
            img_rgb = img

        x_offset = i * (width + spacing)
        combined[:, x_offset:x_offset+width] = img_rgb
        if i < n_images - 1:  # Add separator except after last image
            combined[:, x_offset+width:x_offset +
                     width+spacing] = [100, 100, 100]

    # Convert to PIL for adding text
    pil_img = Image.fromarray(combined)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    # Add labels and title
    for i, label in enumerate(labels):
        x_offset = i * (width + spacing) + 10
        draw.text((x_offset, 10), label, fill=(255, 255, 255), font=font)

    # Add patient/frame info
    draw.text((10, height - 30), f"{title} - Patient {patient_id} - Frame {frame_num}",
              fill=(255, 255, 255), font=font)

    return np.array(pil_img)


def create_overlay_image(noisy_img, roi_img, alpha=0.5):
    """Create an overlay of ROI on noisy image with specified opacity"""
    if noisy_img is None or roi_img is None:
        return None

    # Convert to RGB for visualization
    noisy_rgb = cv2.cvtColor(noisy_img, cv2.COLOR_GRAY2RGB)
    roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2RGB)

    # Create overlay with alpha blending
    overlay = cv2.addWeighted(noisy_rgb, alpha, roi_rgb, alpha, 0)
    return overlay


def process_dataset(dataset_path, output_path):
    """Process the dataset and create visualizations"""
    # Setup paths
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    clean_path = dataset_path / "clean"
    noisy_path = dataset_path / "noisy"
    roi_path = dataset_path / "noisy_roi"

    # Create output directories
    for dir_path in [
        output_path,
        output_path / "triplets",
        output_path / "pairs",
        output_path / "roi_overlays",
        output_path / "roi_overlays" / "individual",
        output_path / "roi_overlays" / "comparison"
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Get list of clean images
    clean_images = sorted(glob.glob(str(clean_path / "*.png")))

    # Process each clean image
    processed_pairs = 0
    processed_triplets = 0
    processed_overlays = 0

    print("\nProcessing dataset...")

    for clean_file in clean_images[:100]:  # Limit to first 100 for testing
        # Extract patient and frame info
        filename = os.path.basename(clean_file)
        parts = filename.replace('.png', '').split('-')
        if len(parts) != 5:
            continue

        patient_id = parts[1]
        frame_num = parts[4]

        # Construct corresponding noisy and ROI paths
        noisy_file = str(noisy_path / filename)
        roi_file = str(roi_path / filename)

        # Load images
        clean_img = load_and_normalize_image(clean_file)
        noisy_img = load_and_normalize_image(noisy_file)
        roi_img = load_and_normalize_image(roi_file)

        if clean_img is not None and noisy_img is not None:
            # Create and save pair comparison
            pair_img = create_comparison_image(
                [clean_img, noisy_img],
                ["CLEAN", "NOISY"],
                "Pair Comparison",
                patient_id,
                frame_num
            )
            if pair_img is not None:
                output_file = output_path / "pairs" / \
                    f"pair_patient_{patient_id}_frame_{frame_num}.png"
                cv2.imwrite(str(output_file), cv2.cvtColor(
                    pair_img, cv2.COLOR_RGB2BGR))
                processed_pairs += 1

        if all(img is not None for img in [clean_img, noisy_img, roi_img]):
            # Create and save triplet comparison
            triplet_img = create_comparison_image(
                [clean_img, noisy_img, roi_img],
                ["CLEAN", "NOISY", "ROI"],
                "Triplet Comparison",
                patient_id,
                frame_num
            )
            if triplet_img is not None:
                output_file = output_path / "triplets" / \
                    f"triplet_patient_{patient_id}_frame_{frame_num}.png"
                cv2.imwrite(str(output_file), cv2.cvtColor(
                    triplet_img, cv2.COLOR_RGB2BGR))
                processed_triplets += 1

            # Create and save ROI overlay
            if noisy_img is not None and roi_img is not None:
                # Individual overlay
                overlay_img = create_overlay_image(noisy_img, roi_img)
                if overlay_img is not None:
                    output_file = output_path / "roi_overlays" / "individual" / \
                        f"overlay_patient_{patient_id}_frame_{frame_num}.png"
                    cv2.imwrite(str(output_file), cv2.cvtColor(
                        overlay_img, cv2.COLOR_RGB2BGR))

                    # Comparison (noisy, ROI, overlay)
                    comparison_img = create_comparison_image(
                        [noisy_img, roi_img, overlay_img],
                        ["NOISY", "ROI", "OVERLAY (50%)"],
                        "ROI Overlay Comparison",
                        patient_id,
                        frame_num
                    )
                    if comparison_img is not None:
                        output_file = output_path / "roi_overlays" / "comparison" / \
                            f"comparison_patient_{patient_id}_frame_{frame_num}.png"
                        cv2.imwrite(str(output_file), cv2.cvtColor(
                            comparison_img, cv2.COLOR_RGB2BGR))
                        processed_overlays += 1

        # Print progress
        if (processed_pairs + processed_triplets) % 10 == 0:
            print(
                f"Processed: {processed_pairs} pairs, {processed_triplets} triplets, {processed_overlays} overlays")

    print("\nVisualization generation complete!")
    print(f"Generated:")
    print(f"- {processed_pairs} pair comparisons")
    print(f"- {processed_triplets} triplet comparisons")
    print(f"- {processed_overlays} ROI overlay comparisons")
    print(f"\nResults saved in: {output_path}")

    def setup_output_directories(self):
        """Create organized folder structure for outputs"""
        directories = [
            self.output_path,
            self.output_path / "triplets",
            self.output_path / "pairs",
            self.output_path / "roi_overlays",
            self.output_path / "roi_overlays" / "individual",
            self.output_path / "roi_overlays" / "comparison",
            self.output_path / "samples",
            self.output_path / "statistics"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def load_image(self, image_path):
        """Load and return image as numpy array"""
        if not image_path.exists():
            return None
        return cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    def create_triplet_comparison(self, patient_id, frame_number, clean_img, noisy_img, roi_img):
        """Create comparison image showing clean, noisy, and ROI side by side"""
        height, width = clean_img.shape

        # Create a combined image
        combined = np.zeros((height, width * 3 + 40, 3), dtype=np.uint8)

        # Convert grayscale to RGB for better visualization
        clean_rgb = cv2.cvtColor(clean_img, cv2.COLOR_GRAY2RGB)
        noisy_rgb = cv2.cvtColor(noisy_img, cv2.COLOR_GRAY2RGB)
        roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2RGB)

        # Place images side by side with spacing
        combined[:, :width] = clean_rgb
        combined[:, width+20:width*2+20] = noisy_rgb
        combined[:, width*2+40:width*3+40] = roi_rgb

        # Add separating lines
        combined[:, width:width+20] = [100, 100, 100]  # Gray separator
        combined[:, width*2+20:width*2+40] = [100, 100, 100]  # Gray separator

        # Convert to PIL for text addition
        pil_img = Image.fromarray(combined)
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()

        # Add labels
        draw.text((10, 10), "CLEAN", fill=(255, 255, 255), font=font)
        draw.text((width + 30, 10), "NOISY", fill=(255, 255, 255), font=font)
        draw.text((width*2 + 50, 10), "ROI", fill=(255, 255, 255), font=font)
        draw.text((10, height - 30),
                  f"Patient {patient_id} - Frame {frame_number}", fill=(255, 255, 255), font=font)

        return np.array(pil_img)

    def create_pair_comparison(self, patient_id, frame_number, clean_img, noisy_img):
        """Create comparison image showing clean and noisy side by side"""
        height, width = clean_img.shape

        # Create a combined image
        combined = np.zeros((height, width * 2 + 20, 3), dtype=np.uint8)

        # Convert grayscale to RGB
        clean_rgb = cv2.cvtColor(clean_img, cv2.COLOR_GRAY2RGB)
        noisy_rgb = cv2.cvtColor(noisy_img, cv2.COLOR_GRAY2RGB)

        # Place images side by side with spacing
        combined[:, :width] = clean_rgb
        combined[:, width+20:width*2+20] = noisy_rgb

        # Add separating line
        combined[:, width:width+20] = [100, 100, 100]  # Gray separator

        # Convert to PIL for text addition
        pil_img = Image.fromarray(combined)
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()

        # Add labels
        draw.text((10, 10), "CLEAN", fill=(255, 255, 255), font=font)
        draw.text((width + 30, 10), "NOISY", fill=(255, 255, 255), font=font)
        draw.text((10, height - 30),
                  f"Patient {patient_id} - Frame {frame_number}", fill=(255, 255, 255), font=font)

        return np.array(pil_img)

    def create_roi_overlay(self, noisy_img, roi_img, alpha=0.5):
        """Create overlay of ROI on noisy image with specified opacity"""
        # Convert to RGB
        noisy_rgb = cv2.cvtColor(noisy_img, cv2.COLOR_GRAY2RGB)
        roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2RGB)

        # Apply alpha blending
        overlay = cv2.addWeighted(noisy_rgb, alpha, roi_rgb, alpha, 0)

        return overlay

    def create_roi_overlay_comparison(self, patient_id, frame_number, noisy_img, roi_img, overlay_img):
        """Create comparison showing noisy, ROI, and overlay"""
        height, width = noisy_img.shape

        # Create a combined image
        combined = np.zeros((height, width * 3 + 40, 3), dtype=np.uint8)

        # Convert grayscale to RGB
        noisy_rgb = cv2.cvtColor(noisy_img, cv2.COLOR_GRAY2RGB)
        roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2RGB)

        # Place images side by side with spacing
        combined[:, :width] = noisy_rgb
        combined[:, width+20:width*2+20] = roi_rgb
        combined[:, width*2+40:width*3+40] = overlay_img

        # Add separating lines
        combined[:, width:width+20] = [100, 100, 100]
        combined[:, width*2+20:width*2+40] = [100, 100, 100]

        # Convert to PIL for text addition
        pil_img = Image.fromarray(combined)
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()

        # Add labels
        draw.text((10, 10), "NOISY", fill=(255, 255, 255), font=font)
        draw.text((width + 30, 10), "ROI", fill=(255, 255, 255), font=font)
        draw.text((width*2 + 50, 10), "OVERLAY (50%)",
                  fill=(255, 255, 255), font=font)
        draw.text((10, height - 30),
                  f"Patient {patient_id} - Frame {frame_number}", fill=(255, 255, 255), font=font)

        return np.array(pil_img)

    def process_triplets(self, max_samples=20):
        """Process and create triplet comparison images"""
        print("Creating triplet comparison images...")

        triplets = self.df[self.df['complete_triplet']
                           == True].head(max_samples)

        for idx, row in triplets.iterrows():
            patient_id = row['patient_id']
            frame_number = row['frame_number']

            # Load images
            clean_img = self.load_image(self.clean_path / row['clean_image'])
            noisy_img = self.load_image(self.noisy_path / row['noisy_image'])
            roi_img = self.load_image(self.roi_path / row['noisy_roi_image'])

            if clean_img is None or noisy_img is None or roi_img is None:
                continue

            # Create triplet comparison
            triplet_comparison = self.create_triplet_comparison(
                patient_id, frame_number, clean_img, noisy_img, roi_img
            )

            # Save triplet comparison
            output_filename = f"triplet_patient_{patient_id}_frame_{frame_number}.png"
            cv2.imwrite(str(self.output_path / "triplets" / output_filename),
                        cv2.cvtColor(triplet_comparison, cv2.COLOR_RGB2BGR))

        print(f"Created {len(triplets)} triplet comparison images")

    def process_pairs(self, max_samples=30):
        """Process and create pair comparison images"""
        print("Creating pair comparison images...")

        # Get pairs that don't have ROI (clean-noisy only)
        pairs = self.df[(self.df['clean_noisy_pair'] == True) &
                        (self.df['has_roi'] == False)].head(max_samples)

        for idx, row in pairs.iterrows():
            patient_id = row['patient_id']
            frame_number = row['frame_number']

            # Load images
            clean_img = self.load_image(self.clean_path / row['clean_image'])
            noisy_img = self.load_image(self.noisy_path / row['noisy_image'])

            if clean_img is None or noisy_img is None:
                continue

            # Create pair comparison
            pair_comparison = self.create_pair_comparison(
                patient_id, frame_number, clean_img, noisy_img
            )

            # Save pair comparison
            output_filename = f"pair_patient_{patient_id}_frame_{frame_number}.png"
            cv2.imwrite(str(self.output_path / "pairs" / output_filename),
                        cv2.cvtColor(pair_comparison, cv2.COLOR_RGB2BGR))

        print(f"Created {len(pairs)} pair comparison images")

    def process_roi_overlays(self, max_samples=20):
        """Process and create ROI overlay images"""
        print("Creating ROI overlay images...")

        roi_samples = self.df[self.df['has_roi'] == True].head(max_samples)

        for idx, row in roi_samples.iterrows():
            patient_id = row['patient_id']
            frame_number = row['frame_number']

            # Load images
            noisy_img = self.load_image(self.noisy_path / row['noisy_image'])
            roi_img = self.load_image(self.roi_path / row['noisy_roi_image'])

            if noisy_img is None or roi_img is None:
                continue

            # Create ROI overlay with 50% opacity
            overlay_img = self.create_roi_overlay(
                noisy_img, roi_img, alpha=0.5)

            # Save individual overlay
            overlay_filename = f"overlay_patient_{patient_id}_frame_{frame_number}.png"
            cv2.imwrite(str(self.output_path / "roi_overlays" / "individual" / overlay_filename),
                        cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

            # Create comparison image (noisy, ROI, overlay)
            comparison_img = self.create_roi_overlay_comparison(
                patient_id, frame_number, noisy_img, roi_img, overlay_img
            )

            # Save comparison
            comparison_filename = f"comparison_patient_{patient_id}_frame_{frame_number}.png"
            cv2.imwrite(str(self.output_path / "roi_overlays" / "comparison" / comparison_filename),
                        cv2.cvtColor(comparison_img, cv2.COLOR_RGB2BGR))

        print(f"Created {len(roi_samples)} ROI overlay images")


if __name__ == "__main__":
    # Process the dataset
    process_dataset("Dataset", "visualizations")
