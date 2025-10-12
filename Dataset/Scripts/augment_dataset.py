#!/usr/bin/env python3
"""
Augment Echocardiography Dehazing Dataset (paired clean/noisy with optional ROI).

- Reads the existing mapping CSV (Dataset/dataset_mapping.csv) or scans folders.
- Copies originals to a new output dataset folder.
- Creates N augmented variants per frame using safe, label-preserving transforms.
  The same transform plan is applied to all available modalities (clean, noisy, noisy_roi)
  for a given (patient, frame) so pairs/triplets remain aligned.
 - Writes an updated mapping CSV including augmented rows.

Default output folder: <repo>/Dataset_augmented

Usage examples:
  python Dataset/Scripts/augment_dataset.py \
    --input-root Dataset \
    --output-root Dataset_augmented \
    --n-augs 2 \
    --limit 100

Notes:
- This script only uses the Python standard library + Pillow and numpy to avoid extra deps.
- If albumentations is available, you can extend it easily; current ops are PIL-based.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError as e:
    raise SystemExit(
        "Pillow (PIL) is required. Please install it with: pip install Pillow"
    ) from e


# -----------------------------
# Data structures
# -----------------------------


@dataclass
class Sample:
    patient_id: int
    frame_number: int
    clean: Optional[str]
    noisy: Optional[str]
    noisy_roi: Optional[str]

    @property
    def has_clean(self) -> bool:
        return bool(self.clean)

    @property
    def has_noisy(self) -> bool:
        return bool(self.noisy)

    @property
    def has_roi(self) -> bool:
        return bool(self.noisy_roi)

    @property
    def is_pair(self) -> bool:
        return self.has_clean and self.has_noisy

    @property
    def is_triplet(self) -> bool:
        return self.has_clean and self.has_noisy and self.has_roi


# -----------------------------
# Image IO helpers
# -----------------------------


def load_image(path: Path) -> Image.Image:
    img = Image.open(path)
    return img


def save_image(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve PNG; default params are fine
    img.save(path)


# -----------------------------
# Transform plan (deterministic per-sample)
# -----------------------------


@dataclass
class TransformPlan:
    do_hflip: bool
    do_vflip: bool
    rotate_deg: float  # e.g., between -10 and 10
    brightness: float  # multiplier around 1.0
    contrast: float  # multiplier around 1.0
    do_blur: bool
    blur_radius: float
    # Optional subtle gaussian noise (applied to all equally)
    do_noise: bool
    noise_std: float


def make_transform_plan(rng: random.Random) -> TransformPlan:
    # Safe-ish defaults for ultrasound: avoid extreme rotations or flips by default.
    # Horizontal flip sometimes okay; vertical flip off by default (rare in ultrasound).
    do_hflip = rng.random() < 0.5
    do_vflip = rng.random() < 0.1  # keep vertical flips rare; set to 0.0 to disable
    rotate_deg = rng.uniform(-10, 10)
    brightness = rng.uniform(0.9, 1.1)
    contrast = rng.uniform(0.9, 1.1)
    do_blur = rng.random() < 0.2
    blur_radius = rng.uniform(0.5, 1.2)
    do_noise = rng.random() < 0.2
    noise_std = rng.uniform(2.0, 6.0)
    return TransformPlan(
        do_hflip=do_hflip,
        do_vflip=do_vflip,
        rotate_deg=rotate_deg,
        brightness=brightness,
        contrast=contrast,
        do_blur=do_blur,
        blur_radius=blur_radius,
        do_noise=do_noise,
        noise_std=noise_std,
    )


def apply_plan(img: Image.Image, plan: TransformPlan, rng: random.Random) -> Image.Image:
    # Work on a copy
    out = img.copy()

    # 1) Flips
    if plan.do_hflip:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if plan.do_vflip:
        out = out.transpose(Image.FLIP_TOP_BOTTOM)

    # 2) Small rotation around center. Keep size, fill with black.
    if abs(plan.rotate_deg) > 0.1:
        out = out.rotate(plan.rotate_deg, resample=Image.BICUBIC, expand=False, fillcolor=0)

    # 3) Brightness / contrast (safe, applied to all aligned modalities)
    if abs(plan.brightness - 1.0) > 1e-3:
        out = ImageEnhance.Brightness(out).enhance(plan.brightness)
    if abs(plan.contrast - 1.0) > 1e-3:
        out = ImageEnhance.Contrast(out).enhance(plan.contrast)

    # 4) Optional mild blur
    if plan.do_blur:
        out = out.filter(ImageFilter.GaussianBlur(radius=plan.blur_radius))

    # 5) Optional mild gaussian noise added to all available images equally
    if plan.do_noise:
        # Convert to numpy, add noise, clip, and back to PIL preserving mode
        arr = np.array(out).astype(np.float32)
        noise = rng.normalvariate(0.0, plan.noise_std)
        # For multi-channel, broadcast noise per pixel
        noise_arr = np.random.default_rng().normal(0.0, plan.noise_std, size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise_arr, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)

    return out


# -----------------------------
# Mapping & workflow
# -----------------------------


def read_mapping(mapping_csv: Path) -> List[Sample]:
    samples: List[Sample] = []
    with mapping_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        # Expect columns from existing mapping
        for row in reader:
            try:
                pid = int(row.get("patient_id", "0") or 0)
                frame = int(row.get("frame_number", "0") or 0)
            except ValueError:
                # Skip header-like or malformed rows
                continue

            samples.append(
                Sample(
                    patient_id=pid,
                    frame_number=frame,
                    clean=(row.get("clean_image") or ""),
                    noisy=(row.get("noisy_image") or ""),
                    noisy_roi=(row.get("noisy_roi_image") or ""),
                )
            )
    return samples


def scan_folders(input_root: Path) -> List[Sample]:
    """Fallback if mapping CSV is missing: infer from filenames present."""
    def parse_name(name: str) -> Optional[Tuple[int, int]]:
        # pattern: patient-<id>-4C-frame-<frame>.png
        try:
            base = Path(name).stem
            # Split and pick ints
            # patient-<id>-4C-frame-<frame>
            parts = base.split("-")
            pid = int(parts[1])
            frame = int(parts[-1].split("frame")[-1]) if "frame" in parts[-1] else int(parts[-1])
            return pid, frame
        except Exception:
            return None

    clean_dir = input_root / "clean"
    noisy_dir = input_root / "noisy"
    roi_dir = input_root / "noisy_roi"

    index: Dict[Tuple[int, int], Dict[str, str]] = {}

    for folder, key in [(clean_dir, "clean"), (noisy_dir, "noisy"), (roi_dir, "noisy_roi")]:
        if not folder.exists():
            continue
        for fn in folder.glob("*.png"):
            parsed = parse_name(fn.name)
            if not parsed:
                continue
            pid, frame = parsed
            index.setdefault((pid, frame), {})[key] = fn.name

    out: List[Sample] = []
    for (pid, frame), d in sorted(index.items()):
        out.append(
            Sample(
                patient_id=pid,
                frame_number=frame,
                clean=d.get("clean", ""),
                noisy=d.get("noisy", ""),
                noisy_roi=d.get("noisy_roi", ""),
            )
        )
    return out


def copy_originals(input_root: Path, output_root: Path) -> None:
    for sub in ["clean", "noisy", "noisy_roi"]:
        src = input_root / sub
        dst = output_root / sub
        if src.exists():
            dst.mkdir(parents=True, exist_ok=True)
            for fn in src.glob("*.png"):
                out_path = dst / fn.name
                if not out_path.exists():
                    shutil.copy2(fn, out_path)


def augment_dataset(
    samples: List[Sample],
    input_root: Path,
    output_root: Path,
    n_augs: int,
    limit: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []

    # Prepare quick path accessors
    def src_path(kind: str, name: str) -> Path:
        return input_root / kind / name

    def dst_name(base_name: str, aug_idx: int) -> str:
        stem = Path(base_name).stem
        suffix = Path(base_name).suffix
        return f"{stem}_aug{aug_idx:02d}{suffix}"

    processed = 0
    for s in samples:
        if limit is not None and processed >= limit:
            break

        # Add original row to mapping output (reference to copied originals)
        rows.append(
            {
                "patient_id": s.patient_id,
                "frame_number": s.frame_number,
                "clean_image": s.clean or "",
                "noisy_image": s.noisy or "",
                "noisy_roi_image": s.noisy_roi or "",
                "has_clean": bool(s.clean),
                "has_noisy": bool(s.noisy),
                "has_roi": bool(s.has_roi),
                "complete_triplet": s.is_triplet,
                "clean_noisy_pair": s.is_pair,
                "augmentation_id": "orig",
                "is_augmented": False,
            }
        )

        # Generate N augmentations, apply the same plan to all present images for this sample
        for k in range(1, n_augs + 1):
            plan = make_transform_plan(rng)

            clean_name_aug: Optional[str] = None
            noisy_name_aug: Optional[str] = None
            roi_name_aug: Optional[str] = None

            if s.clean:
                in_path = src_path("clean", s.clean)
                img = load_image(in_path)
                img_aug = apply_plan(img, plan, rng)
                clean_name_aug = dst_name(s.clean, k)
                save_image(img_aug, output_root / "clean" / clean_name_aug)

            if s.noisy:
                in_path = src_path("noisy", s.noisy)
                img = load_image(in_path)
                img_aug = apply_plan(img, plan, rng)
                noisy_name_aug = dst_name(s.noisy, k)
                save_image(img_aug, output_root / "noisy" / noisy_name_aug)

            if s.noisy_roi:
                in_path = src_path("noisy_roi", s.noisy_roi)
                img = load_image(in_path)
                img_aug = apply_plan(img, plan, rng)
                roi_name_aug = dst_name(s.noisy_roi, k)
                save_image(img_aug, output_root / "noisy_roi" / roi_name_aug)

            # Record augmented mapping row
            rows.append(
                {
                    "patient_id": s.patient_id,
                    "frame_number": s.frame_number,
                    "clean_image": clean_name_aug or "",
                    "noisy_image": noisy_name_aug or "",
                    "noisy_roi_image": roi_name_aug or "",
                    "has_clean": bool(clean_name_aug),
                    "has_noisy": bool(noisy_name_aug),
                    "has_roi": bool(roi_name_aug),
                    "complete_triplet": bool(clean_name_aug and noisy_name_aug and roi_name_aug),
                    "clean_noisy_pair": bool(clean_name_aug and noisy_name_aug),
                    "augmentation_id": f"aug{k:02d}",
                    "is_augmented": True,
                }
            )

        processed += 1

    return rows


def write_mapping(rows: List[Dict[str, object]], out_csv: Path) -> None:
    # Derive total_data_types for parity with original report
    for r in rows:
        r["total_data_types"] = int(bool(r["has_clean"])) + int(bool(r["has_noisy"])) + int(bool(r["has_roi"]))

    fieldnames = [
        "patient_id",
        "frame_number",
        "clean_image",
        "noisy_image",
        "noisy_roi_image",
        "has_clean",
        "has_noisy",
        "has_roi",
        "complete_triplet",
        "clean_noisy_pair",
        "augmentation_id",
        "is_augmented",
        "total_data_types",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Augment Dehazing dataset (paired clean/noisy/roi)")
    parser.add_argument("--input-root", type=str, default="Dataset", help="Path to input dataset root containing clean/noisy/noisy_roi")
    parser.add_argument("--output-root", type=str, default="Dataset_augmented", help="Output dataset root for originals + augmented")
    parser.add_argument("--mapping-csv", type=str, default=None, help="Path to existing mapping CSV; defaults to <input-root>/dataset_mapping.csv")
    parser.add_argument("--n-augs", type=int, default=1, help="Number of augmented variants per sample")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of (patient,frame) samples for quick run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--skip-copy-originals", action="store_true", help="Do not copy original images into output folder")

    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    mapping_csv = Path(args.mapping_csv).resolve() if args.mapping_csv else (input_root / "dataset_mapping.csv")

    if mapping_csv.exists():
        samples = read_mapping(mapping_csv)
    else:
        print(f"Mapping CSV not found at {mapping_csv}. Scanning folders instead…")
        samples = scan_folders(input_root)

    if not args.skip_copy_originals:
        print("Copying original images to output folder…")
        copy_originals(input_root, output_root)

    print(f"Preparing to augment {len(samples)} samples (limit={args.limit}, n_augs={args.n_augs})…")
    rows = augment_dataset(
        samples=samples,
        input_root=input_root,
        output_root=output_root,
        n_augs=max(0, args.n_augs),
        limit=args.limit,
        seed=args.seed,
    )

    out_csv = output_root / "dataset_mapping_augmented.csv"
    write_mapping(rows, out_csv)
    print(f"Augmented mapping written to: {out_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
