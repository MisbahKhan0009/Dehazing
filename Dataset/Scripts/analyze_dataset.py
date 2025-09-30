#!/usr/bin/env python3
"""
Dataset Analysis Script for Echocardiography Dehazing Dataset
"""

import os
import pandas as pd
from collections import defaultdict
import re


def analyze_dataset():
    """Analyze the dataset structure and create comprehensive mapping."""

    dataset_path = "/home/mejba/Documents/Dehazing/Dataset"

    # Read folder descriptions
    with open(os.path.join(dataset_path, "FolderDescriptions.txt"), 'r') as f:
        descriptions = f.read()

    print("Dataset Analysis: Echocardiography Dehazing Dataset")
    print("=" * 60)
    print("Folder Descriptions:")
    print(descriptions)
    print("=" * 60)

    # Get file lists for each folder
    folders = ['clean', 'noisy', 'noisy_roi']
    file_data = {}

    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
            file_data[folder] = sorted(files)
            print(f"{folder.upper()} folder: {len(files)} files")
        else:
            file_data[folder] = []
            print(f"{folder.upper()} folder: Not found")

    print("=" * 60)

    # Parse filenames to extract patient and frame information
    def parse_filename(filename):
        """Extract patient number and frame number from filename."""
        match = re.match(r'patient-(\d+)-4C-frame-(\d+)\.png', filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None

    # Organize data by patient and frame
    organized_data = defaultdict(lambda: defaultdict(dict))

    for folder in folders:
        for filename in file_data[folder]:
            patient, frame = parse_filename(filename)
            if patient is not None and frame is not None:
                organized_data[patient][frame][folder] = filename

    # Create comprehensive dataset mapping
    dataset_records = []

    # Get all unique patients and frames
    all_patients = sorted(organized_data.keys())

    print(f"Analysis Results:")
    print(f"Total patients found: {len(all_patients)}")
    print(f"Patient range: {min(all_patients)} to {max(all_patients)}")

    # Analyze patient distribution across folders
    patients_per_folder = {folder: set() for folder in folders}

    for patient in all_patients:
        for frame in organized_data[patient]:
            for folder in folders:
                if folder in organized_data[patient][frame]:
                    patients_per_folder[folder].add(patient)

    print("\nPatient distribution across folders:")
    for folder in folders:
        patients = sorted(patients_per_folder[folder])
        print(f"{folder}: {len(patients)} patients - {patients}")

    # Find missing patients in each folder
    all_patients_set = set(all_patients)
    print("\nMissing patients per folder:")
    for folder in folders:
        missing = sorted(all_patients_set - patients_per_folder[folder])
        if missing:
            print(f"{folder}: Missing patients {missing}")
        else:
            print(f"{folder}: No missing patients")

    # Create detailed records for CSV
    for patient in all_patients:
        frames = sorted(organized_data[patient].keys())

        for frame in frames:
            record = {
                'patient_id': patient,
                'frame_number': frame,
                'clean_image': organized_data[patient][frame].get('clean', ''),
                'noisy_image': organized_data[patient][frame].get('noisy', ''),
                'noisy_roi_image': organized_data[patient][frame].get('noisy_roi', ''),
                'has_clean': 'clean' in organized_data[patient][frame],
                'has_noisy': 'noisy' in organized_data[patient][frame],
                'has_roi': 'noisy_roi' in organized_data[patient][frame],
                'complete_triplet': all(folder in organized_data[patient][frame] for folder in folders),
                'clean_noisy_pair': all(folder in organized_data[patient][frame] for folder in ['clean', 'noisy']),
            }
            dataset_records.append(record)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(dataset_records)

    # Add derived columns for analysis
    df['total_data_types'] = df[['has_clean',
                                 'has_noisy', 'has_roi']].sum(axis=1)

    # Save to CSV
    csv_path = os.path.join(
        "/home/mejba/Documents/Dehazing", "dataset_mapping.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nDataset mapping saved to: {csv_path}")

    # Generate summary statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY STATISTICS")
    print("=" * 60)

    total_records = len(df)
    print(f"Total image records: {total_records}")

    # Statistics by data availability
    complete_triplets = len(df[df['complete_triplet']])
    clean_noisy_pairs = len(df[df['clean_noisy_pair']])
    only_clean = len(df[(df['has_clean']) & (
        ~df['has_noisy']) & (~df['has_roi'])])
    only_noisy = len(df[(~df['has_clean']) & (
        df['has_noisy']) & (~df['has_roi'])])
    only_roi = len(df[(~df['has_clean']) & (
        ~df['has_noisy']) & (df['has_roi'])])

    print(f"\nData Availability:")
    print(f"Complete triplets (clean + noisy + ROI): {complete_triplets}")
    print(
        f"Clean-Noisy pairs (without ROI): {clean_noisy_pairs - complete_triplets}")
    print(f"Only clean images: {only_clean}")
    print(f"Only noisy images: {only_noisy}")
    print(f"Only ROI images: {only_roi}")

    # Frame distribution analysis
    print(f"\nFrame Analysis:")
    frame_counts = df['frame_number'].value_counts().sort_index()
    print(
        f"Frame range: {frame_counts.index.min()} to {frame_counts.index.max()}")
    print(f"Most common frames: {frame_counts.head().to_dict()}")

    # ROI frame pattern analysis
    roi_frames = df[df['has_roi']]['frame_number'].unique()
    roi_frames_sorted = sorted(roi_frames)
    print(f"\nROI Annotation Pattern:")
    print(f"ROI annotations available for frames: {roi_frames_sorted}")
    print(f"ROI frame pattern appears to be every 10 frames starting from 1 (1, 11, 21, 31, 41, 51)")

    # Patient-wise analysis
    patient_summary = df.groupby('patient_id').agg({
        'frame_number': ['count', 'min', 'max'],
        'has_clean': 'sum',
        'has_noisy': 'sum',
        'has_roi': 'sum',
        'complete_triplet': 'sum'
    }).round(2)

    patient_summary.columns = ['total_frames', 'min_frame', 'max_frame',
                               'clean_count', 'noisy_count', 'roi_count', 'triplet_count']

    print(f"\nPatient Summary (first 10 patients):")
    print(patient_summary.head(10))

    # Identify data inconsistencies and gaps
    print(f"\n" + "=" * 60)
    print("DATA INCONSISTENCIES AND EXPLANATIONS")
    print("=" * 60)

    # Missing patients analysis
    clean_patients = set(df[df['has_clean']]['patient_id'])
    noisy_patients = set(df[df['has_noisy']]['patient_id'])
    roi_patients = set(df[df['has_roi']]['patient_id'])

    print(f"\nPatient Coverage:")
    print(
        f"Patients with clean images: {len(clean_patients)} - {sorted(clean_patients)}")
    print(
        f"Patients with noisy images: {len(noisy_patients)} - {sorted(noisy_patients)}")
    print(
        f"Patients with ROI annotations: {len(roi_patients)} - {sorted(roi_patients)}")

    missing_in_noisy = clean_patients - noisy_patients
    missing_in_clean = noisy_patients - clean_patients

    if missing_in_noisy:
        print(
            f"\nPatients missing in noisy folder: {sorted(missing_in_noisy)}")
        print("Explanation: These patients may represent 'easy-to-image' subjects that don't have corresponding noisy versions.")

    if missing_in_clean:
        print(
            f"\nPatients missing in clean folder: {sorted(missing_in_clean)}")
        print("Explanation: These patients may represent 'difficult-to-image' subjects that don't have clean reference versions.")

    # ROI sampling explanation
    print(f"\nROI Annotation Strategy:")
    print("ROI annotations are provided for a subset of frames (typically every 10 frames).")
    print("This is a common practice in medical imaging to reduce annotation workload while")
    print("maintaining representative samples for evaluation metrics (CNR, gCNR, KS test).")

    return df


if __name__ == "__main__":
    df = analyze_dataset()
    print(f"\nAnalysis complete! Check the generated CSV file for detailed mappings.")
