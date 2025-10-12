# Augmentation script usage

This folder contains `augment_dataset.py`, which expands the dataset by copying originals and generating paired augmentations that keep clean/noisy/ROI images aligned per frame.

Quick start (Windows PowerShell):

- Run 2 augmentations per sample into a new folder next to `Dataset`:

  - python c:\Users\mkhan\Documents\Projects\CSE498\Dehazing\Dataset\Scripts\augment_dataset.py --input-root c:\Users\mkhan\Documents\Projects\CSE498\Dehazing\Dataset --output-root c:\Users\mkhan\Documents\Projects\CSE498\Dehazing\Dataset_augmented --n-augs 2 --seed 1337

Key flags:

- --input-root: Path to the source dataset containing clean/, noisy/, and noisy_roi/ plus dataset_mapping.csv
- --output-root: Destination folder to write originals + augmented data (subfolders clean/, noisy/, noisy_roi/ will be created)
- --n-augs: Number of augmented variants per (patient, frame)
- --limit: Optional cap for a quick dry run
- --seed: Reproducible random seed
- --skip-copy-originals: If set, only augmented images are written; originals are not copied

Outputs:

- Originals are copied to the output folder
- Augmented images are created with suffixes like `_aug01`, `_aug02` etc.
- A CSV mapping `dataset_mapping_augmented.csv` is written at the output root with both original and augmented rows, preserving pairs/triplets alignment

Transforms applied (paired):

- Horizontal flip (50%), rare vertical flip (10%)
- Small rotation (~±10°), mild brightness/contrast changes (±10%)
- Optional Gaussian blur and light Gaussian noise

Note: To disable vertical flips entirely, set its probability to 0.0 inside `make_transform_plan`.
