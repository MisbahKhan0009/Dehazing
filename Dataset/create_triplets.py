import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Define folder paths
noisy_folder = Path(r"c:\Users\mkhan\Documents\Projects\CSE498\Dehazing\Dehazing\Dataset\noisy")
output_folder = Path(r"c:\Users\mkhan\Documents\Projects\CSE498\Dehazing\Dehazing\Dataset\Output")
clean_folder = Path(r"c:\Users\mkhan\Documents\Projects\CSE498\Dehazing\Dehazing\Dataset\clean")
triplet_output_folder = Path(r"c:\Users\mkhan\Documents\Projects\CSE498\Dehazing\Dehazing\Dataset\triplets")

# Create output folder if it doesn't exist
triplet_output_folder.mkdir(exist_ok=True)

# Get all PNG files from each folder
noisy_files = {f.name: f for f in noisy_folder.glob("*.png")}
output_files = {f.name: f for f in output_folder.glob("*.png")}
clean_files = {f.name: f for f in clean_folder.glob("*.png")}

# Find common files that exist in all three folders
common_files = set(noisy_files.keys()) & set(output_files.keys()) & set(clean_files.keys())

print(f"Found {len(common_files)} matching files across all three folders")

# Process each matching file
for idx, filename in enumerate(sorted(common_files), 1):
    print(f"Processing {idx}/{len(common_files)}: {filename}")
    
    try:
        # Load images
        noisy_img = Image.open(noisy_files[filename])
        output_img = Image.open(output_files[filename])
        clean_img = Image.open(clean_files[filename])
        
        # Get dimensions
        width, height = noisy_img.size
        
        # Define label height and padding
        label_height = 40
        padding = 10
        
        # Create new image for triplet (3 images side by side with labels)
        triplet_width = width * 3 + padding * 2
        triplet_height = height + label_height
        triplet_img = Image.new('RGB', (triplet_width, triplet_height), 'white')
        
        # Create drawing context for labels
        draw = ImageDraw.Draw(triplet_img)
        
        # Try to use a decent font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Add labels
        labels = ["Noisy", "Output", "Ground Truth"]
        positions = [0, width + padding, (width + padding) * 2]
        
        for label, x_pos in zip(labels, positions):
            # Calculate text position to center it
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = x_pos + (width - text_width) // 2
            text_y = 10
            
            # Draw text
            draw.text((text_x, text_y), label, fill='black', font=font)
        
        # Paste images below labels
        triplet_img.paste(noisy_img, (0, label_height))
        triplet_img.paste(output_img, (width + padding, label_height))
        triplet_img.paste(clean_img, ((width + padding) * 2, label_height))
        
        # Save triplet image
        output_path = triplet_output_folder / f"triplet_{filename}"
        triplet_img.save(output_path)
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        continue

print(f"\nCompleted! Saved {len(common_files)} triplet images to {triplet_output_folder}")
