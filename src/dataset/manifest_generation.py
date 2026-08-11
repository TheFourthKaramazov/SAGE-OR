#!/usr/bin/env python3
"""
Generate Manifest for 4D-OR Dataset

Creates a unified manifest CSV containing metadata for all frames across all takes.
Also saves resized/processed RGB images for faster training.

Usage:
    python -m src.dataset.manifest_generation --data-dir data/4D-OR

Output structure:
    processed_data/
    ├── manifest.csv                          # Master index
    └── processed_images/
        ├── take01/
        │   ├── frame_0000_cam1.jpg           # Resized RGB (518x392)
        │   ├── frame_0000_cam2.jpg
        │   └── ...
        └── take02/
            └── ...

The manifest includes:
    - take_id: Take number (1-10)
    - scan_idx: Logical frame index (0, 1, 2, ...)
    - file_frame: Actual file frame number (000001, 000031, 000061, ...)
    - split: train/val/test assignment
    - cam{1-6}_rgb: Paths to ORIGINAL RGB images
    - cam{1-6}_processed: Paths to PROCESSED/resized RGB images
    - cache_path: Path to pi3 cache file

Frame Mapping:
    file_frame = scan_idx * 30 + 1  (formatted as 6 digits)
    Example: scan_idx=2 -> file_frame=000061
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def scan_idx_to_file_frame(scan_idx: int) -> str:
    """Convert logical scan index to file frame string."""
    return f"{scan_idx * 30 + 1:06d}"


def get_take_dirs(data_dir: Path) -> list[Path]:
    """Find all take directories in the 4D-OR dataset."""
    take_dirs = []
    for i in range(1, 11):  # Takes 1-10
        take_dir = data_dir / f"export_holistic_take{i}_processed"
        if take_dir.exists():
            take_dirs.append(take_dir)
    return sorted(take_dirs)


def parse_take_id(take_dir: Path) -> int:
    """Extract take ID from directory name."""
    # e.g., "export_holistic_take1_processed" -> 1
    name = take_dir.name
    return int(name.split("take")[1].split("_")[0])


def get_frame_count(take_dir: Path) -> int:
    """Count unique scan indices across all cameras."""
    frames_by_scan = get_available_frames(take_dir)
    return len(frames_by_scan)


def get_available_frames(take_dir: Path) -> dict[int, dict[int, str]]:
    """
    Get available frames for this take, grouped by scan_idx.

    Returns:
        Dict mapping scan_idx -> {cam_id: file_frame}
        Each scan_idx has at most one file_frame per camera.
    """
    colorimage_dir = take_dir / "colorimage"

    # Collect file_frames per camera, grouped by scan_idx
    # scan_idx -> {cam_id: file_frame}
    frames_by_scan: dict[int, dict[int, str]] = {}

    for cam_id in range(1, 7):
        for img_path in colorimage_dir.glob(f"camera{cam_id:02d}_colorimage-*.jpg"):
            file_frame = img_path.stem.split("-")[1]
            scan_idx = file_frame_to_scan_idx(file_frame)

            if scan_idx not in frames_by_scan:
                frames_by_scan[scan_idx] = {}

            # Store this camera's file_frame for this scan_idx
            frames_by_scan[scan_idx][cam_id] = file_frame

    return frames_by_scan


def file_frame_to_scan_idx(file_frame: str) -> int:
    """Convert file frame string to logical scan index."""
    # file_frame = scan_idx * 30 + 1
    # scan_idx = (file_frame - 1) / 30
    frame_num = int(file_frame)
    return (frame_num - 1) // 30


def load_split_assignments(data_dir: Path) -> dict[tuple[int, int], str]:
    """
    Load train/val/test split assignments from OR_4D_data.

    Returns:
        Dict mapping (take_id, scan_idx) -> split
    """
    splits = {}
    or_data_dir = data_dir / "OR_4D_data"

    for split_name, filename in [
        ("train", "relationships_train.json"),
        ("val", "relationships_validation.json"),
        ("test", "relationships_test.json"),
    ]:
        filepath = or_data_dir / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            continue

        with open(filepath) as f:
            data = json.load(f)

        for scan in data.get("scans", []):
            take_id = scan["take_idx"]
            scan_idx = int(scan["scan"])
            splits[(take_id, scan_idx)] = split_name

    return splits


def process_and_save_image(
    src_path: Path,
    dst_path: Path,
    target_size: tuple[int, int] = (518, 392),
) -> bool:
    """Resize and save an image."""
    try:
        img = Image.open(src_path).convert("RGB")
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img_resized.save(dst_path, quality=95)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False


def generate_manifest(
    data_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    target_size: tuple[int, int] = (518, 392),
    process_images: bool = True,
) -> pd.DataFrame:
    """
    Generate manifest CSV for all frames in 4D-OR dataset.
    Optionally processes and saves resized RGB images.

    Args:
        data_dir: Path to 4D-OR dataset root
        output_dir: Path to output directory (for manifest.csv and processed_images/)
        cache_dir: Path to cache directory (for pi3 cache paths in manifest)
        target_size: (width, height) for resized images
        process_images: If True, resize and save images

    Returns:
        DataFrame with manifest data
    """
    take_dirs = get_take_dirs(data_dir)
    if not take_dirs:
        raise ValueError(f"No take directories found in {data_dir}")

    print(f"Found {len(take_dirs)} takes")

    # Load split assignments
    splits = load_split_assignments(data_dir)
    print(f"Loaded {len(splits)} split assignments")

    processed_images_dir = output_dir / "processed_images"
    manifest_path = output_dir / "manifest.csv"

    rows = []
    total_images = 0
    processed_count = 0

    for take_dir in take_dirs:
        take_id = parse_take_id(take_dir)
        frames_by_scan = get_available_frames(take_dir)
        print(f"\n  Take {take_id}: {len(frames_by_scan)} unique scan indices")

        colorimage_dir = take_dir / "colorimage"

        take_processed_dir = processed_images_dir / f"take{take_id:02d}"
        take_cache_dir = cache_dir / f"take{take_id:02d}"

        scan_indices = sorted(frames_by_scan.keys())
        scan_iter = tqdm(scan_indices, desc=f"Take {take_id}") if process_images else scan_indices

        for scan_idx in scan_iter:
            cam_frames = frames_by_scan[scan_idx]  # {cam_id: file_frame}

            # Get split assignment
            split = splits.get((take_id, scan_idx), "unknown")

            row = {
                "take_id": take_id,
                "scan_idx": scan_idx,
                "split": split,
                "cache_path": str(take_cache_dir / f"frame_{scan_idx:04d}.npz"),
            }

            # Add paths for each camera
            for cam_id in range(1, 7):
                file_frame = cam_frames.get(cam_id)  # May be None if camera missing

                if file_frame:
                    # Original RGB path
                    rgb_filename = f"camera{cam_id:02d}_colorimage-{file_frame}.jpg"
                    rgb_path = colorimage_dir / rgb_filename
                    row[f"cam{cam_id}_rgb"] = str(rgb_path) if rgb_path.exists() else ""

                    # Processed RGB path
                    processed_filename = f"frame_{scan_idx:04d}_cam{cam_id}.jpg"
                    processed_path = take_processed_dir / processed_filename
                    row[f"cam{cam_id}_processed"] = str(processed_path) if rgb_path.exists() else ""

                    # Process and save image if requested (skip if already exists)
                    if process_images and rgb_path.exists():
                        total_images += 1
                        if processed_path.exists():
                            processed_count += 1  # Already done
                        elif process_and_save_image(rgb_path, processed_path, target_size):
                            processed_count += 1
                else:
                    # Camera not available for this scan_idx
                    row[f"cam{cam_id}_rgb"] = ""
                    row[f"cam{cam_id}_processed"] = ""

            rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by take_id, then scan_idx
    df = df.sort_values(["take_id", "scan_idx"]).reset_index(drop=True)

    # Save manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    print(f"\nManifest saved to: {manifest_path}")

    # Print summary
    print(f"\nSummary:")
    print(f"  Total frames: {len(df)}")
    print(f"  Split distribution:")
    for split, count in df["split"].value_counts().items():
        print(f"    {split}: {count}")

    if process_images:
        print(f"\nImage processing:")
        print(f"  Processed: {processed_count}/{total_images}")
        print(f"  Target size: {target_size[0]}x{target_size[1]}")
        print(f"  Output: {processed_images_dir}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate manifest for 4D-OR dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/4D-OR",
        help="Path to 4D-OR dataset root (default: data/4D-OR)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="processed_data",
        help="Output directory for manifest.csv and processed_images/ (default: processed_data)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="caches/4d_or_cache",
        help="Cache directory path (for manifest references) (default: caches/4d_or_cache)"
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=518,
        help="Target image width (default: 518)"
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=392,
        help="Target image height (default: 392)"
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image processing (only generate manifest)"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    target_size = (args.target_width, args.target_height)

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1

    print("=" * 60)
    print("4D-OR Manifest Generation")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")
    print(f"Process images: {not args.skip_images}")
    print()

    try:
        df = generate_manifest(
            data_dir=data_dir,
            output_dir=output_dir,
            cache_dir=cache_dir,
            target_size=target_size,
            process_images=not args.skip_images,
        )
        print("\nManifest generation complete!")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
