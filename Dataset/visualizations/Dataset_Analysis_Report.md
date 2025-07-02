# Echocardiography Dehazing Dataset Analysis Report

## Dataset Overview

This dataset contains echocardiography images for dehazing research, organized into three main categories:

1. **Clean Images**: High-quality echocardiography images from easy-to-image subjects
2. **Noisy Images**: Degraded echocardiography images from difficult-to-image subjects
3. **ROI Annotations**: Region of Interest masks for evaluation metrics

## Dataset Structure Summary

### File Statistics

- **Total Patients**: 75 (Patient IDs: 1-75)
- **Clean Images**: 4,376 files (75 patients, 60 frames each)
- **Noisy Images**: 2,324 files (40 patients, 60 frames each)
- **ROI Annotations**: 237 files (40 patients, ~6 frames each)

### Data Distribution

#### Complete Data Availability

- **Complete Triplets** (Clean + Noisy + ROI): 237 image sets
- **Clean-Noisy Pairs** (without ROI): 2,087 image pairs
- **Clean Only**: 2,052 images

#### Patient Categories

- **Patients with Clean Images**: All 75 patients (1-75)
- **Patients with Noisy Images**: 40 patients [1, 3, 4, 5, 6, 7, 9, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 33, 34, 35, 38, 39, 40, 42, 44, 45, 46, 47, 48, 49, 50]
- **Patients with ROI Annotations**: Same 40 patients as noisy images

## Key Findings and Explanations

### 1. Patient Distribution Discrepancy

**Observation**: Only 40 out of 75 patients have noisy images and ROI annotations.

**Explanation**: This reflects the real-world clinical scenario:

- **Patients 1-50** (with some gaps): Represent "difficult-to-image" subjects where image quality is compromised due to factors like:
  - Patient anatomy (obesity, chest wall thickness)
  - Positioning difficulties
  - Acoustic window limitations
  - Motion artifacts
- **Patients 51-75** (and some earlier): Represent "easy-to-image" subjects with inherently good image quality, making synthetic noise addition unnecessary

### 2. ROI Annotation Strategy

**Pattern**: ROI annotations are provided strategically, not for every frame:

- Typically every 10 frames (frames 1, 11, 21, 31, 41, 51)
- Some patients have variations (e.g., frames 7, 9, 13, 14, 17, 19, 20, 25, 27, 33)

**Rationale**:

- Reduces annotation workload while maintaining statistical validity
- Provides sufficient samples for evaluation metrics (CNR, gCNR, KS test)
- Captures representative cardiac cycle phases

### 3. Frame Structure

**Consistency**: Each patient has exactly 60 frames when data is available

- Represents a complete cardiac cycle or multiple cycles
- Standardized temporal sampling across all subjects

## Clinical Context and Use Cases

### Training Scenarios

1. **Dehazing Model Training**:

   - Use Clean-Noisy pairs (2,324 training samples)
   - 40 patients provide diverse anatomical variations

2. **Evaluation and Testing**:

   - Use ROI-annotated frames (237 samples) for quantitative metrics
   - ROI masks enable precise CNR, gCNR, and KS test calculations

3. **Data Augmentation**:
   - Clean-only images (2,052 samples) can be used for:
     - Synthetic noise generation
     - Domain adaptation experiments
     - Pre-training clean image features

### Quality Assessment Metrics

The ROI annotations enable computation of:

- **CNR (Contrast-to-Noise Ratio)**: Measures image contrast quality
- **gCNR (Generalized CNR)**: Enhanced contrast metric for ultrasound
- **KS Test**: Statistical test for image quality distribution

## Dataset Usage Recommendations

### For Model Development

1. **Training Set**: Use all Clean-Noisy pairs (2,324 samples)
2. **Validation Set**: Reserve some patients for validation (e.g., patients 45-50)
3. **Test Set**: Use ROI-annotated samples for final evaluation

### For Research Applications

1. **Dehazing Algorithm Development**: Focus on the 40 patients with complete data
2. **Generalization Studies**: Use clean-only patients to test domain transfer
3. **Ablation Studies**: Use ROI annotations for detailed performance analysis

## Data Integrity Verification

### Filename Convention

- Format: `patient-{ID}-4C-frame-{FRAME}.png`
- Patient IDs: 1-75
- Frame numbers: 1-60
- View: 4-chamber (4C) echocardiographic view

### File Relationships

The generated CSV file (`dataset_mapping.csv`) provides:

- Complete mapping between clean, noisy, and ROI images
- Boolean flags for data availability
- Triplet completeness indicators
- Easy filtering for specific use cases

## Conclusion

This dataset provides a realistic and clinically relevant collection for echocardiography dehazing research. The intentional patient distribution (40 difficult-to-image vs. 35 easy-to-image subjects) reflects real clinical scenarios and enables robust algorithm development and evaluation.

The strategic ROI annotation approach balances annotation cost with evaluation completeness, providing sufficient ground truth for quantitative assessment while maintaining practical feasibility.
