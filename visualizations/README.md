# Dataset Visualization Outputs

This directory contains comprehensive visualizations and analysis of the echocardiography dehazing dataset.

## Generated Folder Structure

```
visualizations/
├── triplets/                  # Triplet comparison images (Clean + Noisy + ROI)
├── pairs/                     # Pair comparison images (Clean + Noisy)
├── roi_overlays/              # ROI overlay visualizations
│   ├── individual/            # Individual overlay images (50% opacity)
│   └── comparison/            # Side-by-side comparison (Noisy, ROI, Overlay)
├── samples/                   # Sample grid visualization
└── statistics/                # Dataset statistics and charts
```

## Image Types Generated

### 1. Triplet Comparisons (`triplets/`)

- **Purpose**: Shows Clean, Noisy, and ROI images side by side
- **Format**: Three images horizontally aligned with labels
- **Naming**: `triplet_patient_{ID}_frame_{FRAME}.png`
- **Count**: 20 samples from patients with complete triplet data

### 2. Pair Comparisons (`pairs/`)

- **Purpose**: Shows Clean and Noisy images side by side
- **Format**: Two images horizontally aligned with labels
- **Naming**: `pair_patient_{ID}_frame_{FRAME}.png`
- **Count**: 30 samples from clean-noisy pairs without ROI

### 3. ROI Overlays (`roi_overlays/`)

#### Individual Overlays (`individual/`)

- **Purpose**: ROI mask overlaid on noisy image with 50% opacity
- **Format**: Single blended image
- **Naming**: `overlay_patient_{ID}_frame_{FRAME}.png`
- **Opacity**: Both noisy and ROI images at 50% opacity each

#### Overlay Comparisons (`comparison/`)

- **Purpose**: Shows original noisy, ROI mask, and overlay side by side
- **Format**: Three images horizontally aligned
- **Naming**: `comparison_patient_{ID}_frame_{FRAME}.png`
- **Count**: 20 samples from patients with ROI data

### 4. Sample Grid (`samples/`)

- **Purpose**: Overview grid showing representative samples
- **Content**:
  - Top row: Clean, Noisy, ROI from a triplet
  - Bottom row: Clean, Noisy from a pair + ROI overlay example
- **Filename**: `dataset_samples_grid.png`

### 5. Statistics (`statistics/`)

- **Dataset Stats**: Text file with comprehensive statistics
- **Visualization**: Charts showing data distribution and patient availability
- **Files**:
  - `dataset_stats.txt`: Numerical statistics
  - `dataset_statistics.png`: Visual charts

## Key Features

### Image Processing

- **Grayscale to RGB conversion** for better visualization
- **Text labels** on each image section
- **Gray separators** between image sections
- **Patient and frame information** displayed on each image

### ROI Overlay Technique

- **50% opacity blending** using OpenCV's `addWeighted` function
- **Alpha blending formula**: `result = 0.5 * noisy + 0.5 * roi`
- **Preserves both image details** while showing ROI boundaries

### Folder Organization

- **Logical separation** by visualization type
- **Consistent naming** conventions
- **Scalable structure** for additional analysis

## Usage Examples

### For Training Data Selection

```python
# Load CSV mapping to identify triplets
df = pd.read_csv('dataset_mapping.csv')
triplets = df[df['complete_triplet'] == True]

# Use triplet comparison images to visually verify data quality
# Files in visualizations/triplets/ correspond to triplet rows
```

### For ROI Analysis

```python
# Individual overlays show ROI boundaries clearly
# Comparison images help validate ROI accuracy
# Files in roi_overlays/ folders provide visual verification
```

### For Dataset Understanding

```python
# Statistics provide quantitative overview
# Sample grid provides qualitative overview
# Pair/triplet comparisons show data relationships
```

## Technical Details

### Image Dimensions

- **Original**: Preserved from source images
- **Triplets**: Width × 3 + 40 pixels (for separators)
- **Pairs**: Width × 2 + 20 pixels (for separator)
- **Overlays**: Original dimensions maintained

### Color Coding

- **Clean images**: Natural grayscale
- **Noisy images**: Natural grayscale
- **ROI masks**: Natural grayscale
- **Overlays**: RGB blended at 50% opacity each
- **Separators**: Gray (RGB: 100, 100, 100)

### Text Annotations

- **Font**: DejaVu Sans Bold (fallback to default)
- **Size**: 16pt for labels, 16pt for patient/frame info
- **Color**: White for visibility on dark backgrounds
- **Position**: Top-left for labels, bottom-left for info

## Statistics Summary

Based on the analysis:

- **Total Images**: 4,376
- **Complete Triplets**: 237
- **Clean-Noisy Pairs**: 2,324
- **Clean Only**: 2,052
- **Patients with Noisy Data**: 40
- **Patients with ROI Data**: 40
- **Total Patients**: 75

## Recommendations

1. **Use triplet images** for visual quality assessment
2. **Use pair images** for training data validation
3. **Use overlay images** for ROI boundary verification
4. **Reference statistics** for dataset planning and splits
5. **Check sample grid** for overall dataset understanding

## Generated Scripts

- `visualize_dataset.py`: Main visualization generation script
- `preview_visualizations.py`: Quick preview script for sample outputs
- `analyze_dataset.py`: Original dataset analysis script
- `dataset_mapping.csv`: Complete image mapping file
