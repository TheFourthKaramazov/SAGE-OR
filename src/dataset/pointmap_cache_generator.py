#!/usr/bin/env python3
"""
Generate pi3 Cache for 4D-OR Dataset

Generates geometry supervision signals from pi3 (Pi-Cubed) for all frames in 4D-OR.
Each frame consists of 6 synchronized camera images.

Default resolution: 518x392 (preserves 4:3 aspect ratio of original 2048x1536 images)

Outputs per frame (saved as .npz):
    - xyz_local: [N, H, W, 3] float16 - Local pointmaps (N = 1-6 cameras)
    - conf: [N, H, W, 1] float16 - Confidence scores
    - camera_poses: [N, 4, 4] float32 - Estimated camera poses
    - camera_ids: [N] int8 - Which cameras are present (1-6)

Usage:
    python -m src.dataset.pointmap_cache_generator --manifest processed_data/manifest.csv

Options:
    --debug         Save PLY visualizations for spot-checking
    --resume        Skip already processed frames
    --takes 1 2 3   Process only specific takes
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "external" / "pi3"))

# Lazy imports (only when needed)
Pi3 = None


def lazy_import_pi3():
    """Lazily import pi3 to allow running without it installed."""
    global Pi3
    if Pi3 is None:
        from pi3.models.pi3 import Pi3 as _Pi3
        Pi3 = _Pi3




def load_images_for_pi3(
    image_paths: list[Path],
    target_size: tuple[int, int] | None = None,
    pixel_limit: int = 255000
) -> tuple[torch.Tensor, list[tuple[int, int]]] | tuple[None, None]:
    """
    Load images for pi3 inference.

    pi3 expects images as [N, 3, H, W] tensors in [0, 1] range,
    with H and W divisible by 14 (patch size).

    Args:
        image_paths: List of image file paths (6 cameras)
        target_size: Optional (H, W) tuple to resize to
        pixel_limit: Maximum pixels per image for auto-sizing

    Returns:
        imgs: [N, 3, H, W] tensor in [0, 1] range
        original_sizes: List of (W, H) tuples
    """
    to_tensor = transforms.ToTensor()

    sources = []
    original_sizes = []

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            original_sizes.append(img.size)  # (W, H)
            sources.append(img)
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")
            return None, None

    if not sources:
        print(f"No images loaded")
        return None, None

    # Determine target size
    if target_size is not None:
        target_h, target_w = target_size
    else:
        # Auto-compute based on first image and pixel limit
        first_img = sources[0]
        W_orig, H_orig = first_img.size

        import math
        scale = math.sqrt(pixel_limit / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
        W_target, H_target = W_orig * scale, H_orig * scale

        # Make divisible by 14 (pi3 patch size)
        k, m = round(W_target / 14), round(H_target / 14)
        while (k * 14) * (m * 14) > pixel_limit:
            if k / m > W_target / H_target:
                k -= 1
            else:
                m -= 1
        target_w, target_h = max(1, k) * 14, max(1, m) * 14

    # Resize and convert to tensors
    tensor_list = []
    for img_pil in sources:
        resized_img = img_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        img_tensor = to_tensor(resized_img)  # [3, H, W] in [0, 1]
        tensor_list.append(img_tensor)

    imgs = torch.stack(tensor_list, dim=0)  # [6, 3, H, W]
    return imgs, original_sizes


def save_ply(xyz: np.ndarray, rgb: np.ndarray, path: Path):
    """Save colored point cloud to PLY file."""
    try:
        import open3d as o3d
    except ImportError:
        print(f"Skipping PLY save (open3d not installed): {path}")
        return

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(str(path), pc, compressed=True)


def process_frame(
    model: torch.nn.Module,
    image_paths: list[Path],
    camera_ids: list[int],
    output_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    target_size: tuple[int, int] | None = None,
    debug: bool = False,
) -> bool:
    """
    Process a single frame (1-6 cameras) through pi3 and save cache.

    Args:
        model: Loaded pi3 model
        image_paths: List of camera image paths (1-6 cameras)
        camera_ids: List of camera IDs (1-6) corresponding to image_paths
        output_path: Path to save .npz file
        device: Torch device
        dtype: Torch dtype for inference
        target_size: Optional (H, W) for resizing
        debug: If True, save PLY visualization

    Returns:
        True if successful, False otherwise
    """
    # Load images
    imgs, original_sizes = load_images_for_pi3(image_paths, target_size)
    if imgs is None:
        return False

    imgs = imgs.to(device)

    # Clear GPU cache before inference
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Run pi3 inference
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype, enabled=(device.type == "cuda")):
            res = model(imgs[None])  # Add batch dimension: [1, 6, 3, H, W]

    # Extract outputs (remove batch dimension)
    # res['local_points']: [1, N, H, W, 3] -> [N, H, W, 3]
    # res['conf']: [1, N, H, W, 1] -> [N, H, W, 1]
    # res['camera_poses']: [1, N, 4, 4] -> [N, 4, 4]
    # res['points']: [1, N, H, W, 3] -> [N, H, W, 3] (global, for debug)
    # where N = number of available cameras (1-6)

    num_cams = len(camera_ids)
    xyz_local = res["local_points"][0].cpu()  # [N, H, W, 3]
    conf_raw = res["conf"][0].cpu()  # [N, H, W, 1]
    camera_poses = res["camera_poses"][0].cpu()  # [N, 4, 4]

    # Apply sigmoid to confidence (pi3 outputs raw logits)
    conf = torch.sigmoid(conf_raw)  # [N, H, W, 1]

    # Convert to float16 for storage efficiency
    xyz_local_f16 = xyz_local.numpy().astype(np.float16)  # [N, H, W, 3]
    conf_f16 = conf.numpy().astype(np.float16)  # [N, H, W, 1]
    camera_poses_f32 = camera_poses.numpy().astype(np.float32)  # [N, 4, 4]

    # Save to NPZ
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        xyz_local=xyz_local_f16,
        conf=conf_f16,
        camera_poses=camera_poses_f32,
        camera_ids=np.array(camera_ids, dtype=np.int8),  # Which cameras (1-6)
        num_cameras=num_cams,
        original_sizes=np.array(original_sizes),
    )

    # Debug: Save PLY visualization
    if debug:
        xyz_global = res["points"][0].cpu().numpy()  # [N, H, W, 3]

        xyz_list, rgb_list = [], []
        for i, img_path in enumerate(image_paths):
            try:
                orig = Image.open(img_path).convert("RGB")
                xyz = xyz_global[i]  # [H, W, 3]
                H_viz, W_viz = xyz.shape[:2]

                # Resize image to match point cloud resolution
                rgb = np.array(orig.resize((W_viz, H_viz), Image.LANCZOS))

                # Flatten and add all points
                xyz_list.append(xyz.reshape(-1, 3))
                rgb_list.append(rgb.reshape(-1, 3))
            except Exception as e:
                print(f"Warning: Failed to process image {img_path}: {e}")
                continue

        if xyz_list:
            xyz_all = np.concatenate(xyz_list, 0)
            rgb_all = np.concatenate(rgb_list, 0)

            # Filter out NaN/Inf points
            valid_pts = np.isfinite(xyz_all).all(axis=1)
            if valid_pts.any():
                ply_path = output_path.with_suffix(".ply")
                save_ply(xyz_all[valid_pts], rgb_all[valid_pts], ply_path)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate pi3 cache for 4D-OR dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="processed_data/manifest.csv",
        help="Path to manifest CSV (default: processed_data/manifest.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="caches/4d_or_cache",
        help="Output cache directory (default: caches/4d_or_cache)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save PLY visualizations for debugging"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already processed frames"
    )
    parser.add_argument(
        "--takes",
        type=int,
        nargs="+",
        default=None,
        help="Process only specific takes (e.g., --takes 1 2 3)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of frames to process (for debugging)"
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=392,
        help="Target image height (default: 392, divisible by 14, preserves 4:3 aspect ratio)"
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=518,
        help="Target image width (default: 518, divisible by 14, preserves 4:3 aspect ratio)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pi3 checkpoint (default: HuggingFace)"
    )

    args = parser.parse_args()

    # Validate target size is divisible by 14
    if args.target_height % 14 != 0 or args.target_width % 14 != 0:
        print("Error: target-height and target-width must be divisible by 14")
        return 1

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    target_size = (args.target_height, args.target_width)

    # Check manifest exists
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        print("Run 'python -m src.dataset.manifest_generation' first")
        return 1

    # Load manifest
    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest with {len(df)} frames")

    # Filter by takes if specified
    if args.takes:
        df = df[df["take_id"].isin(args.takes)]
        print(f"Filtered to {len(df)} frames for takes {args.takes}")

    # Limit number of frames if specified (for debugging)
    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {len(df)} frames (--limit {args.limit})")

    if len(df) == 0:
        print("No frames to process")
        return 0

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    print(f"\n{'=' * 60}")
    print("pi3 Cache Generation for 4D-OR")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")
    print(f"Debug mode: {args.debug}")
    print(f"Resume mode: {args.resume}")
    print(f"Output directory: {output_dir}")
    print()

    # Import and load pi3 model
    lazy_import_pi3()

    print("Loading pi3 model...")
    if args.checkpoint:
        print(f"Loading from checkpoint: {args.checkpoint}")
        model = Pi3().to(device).eval()
        if args.checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file
            weight = load_file(args.checkpoint)
        else:
            weight = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(weight)
    else:
        print("Loading from HuggingFace: yyfz233/Pi3")
        model = Pi3.from_pretrained("yyfz233/Pi3")
        model = model.to(device).eval()

    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Process frames
    processed = 0
    skipped = 0
    failed = 0

    for take_id in sorted(df["take_id"].unique()):
        take_df = df[df["take_id"] == take_id]

        print(f"\nProcessing Take {take_id} ({len(take_df)} frames)...")

        for _, row in tqdm(take_df.iterrows(), total=len(take_df), desc=f"Take {take_id}"):
            scan_idx = row["scan_idx"]

            # Use cache_path from manifest if available, otherwise construct it
            if "cache_path" in row and pd.notna(row["cache_path"]):
                output_path = Path(row["cache_path"])
            else:
                output_path = output_dir / f"take{take_id:02d}" / f"frame_{scan_idx:04d}.npz"

            # Skip if already exists (resume mode)
            if args.resume and output_path.exists():
                skipped += 1
                continue

            # Gather image paths for available cameras (1-6)
            image_paths = []
            camera_ids = []
            for cam_id in range(1, 7):
                # Prefer processed images (already resized)
                processed_col = f"cam{cam_id}_processed"
                rgb_col = f"cam{cam_id}_rgb"

                processed_path = row[processed_col] if processed_col in row else ""
                if processed_path and pd.notna(processed_path) and Path(processed_path).exists():
                    image_paths.append(Path(processed_path))
                    camera_ids.append(cam_id)
                else:
                    # Fallback to original
                    rgb_path = row[rgb_col] if rgb_col in row else ""
                    if rgb_path and pd.notna(rgb_path) and Path(rgb_path).exists():
                        image_paths.append(Path(rgb_path))
                        camera_ids.append(cam_id)
                    # Skip missing cameras silently

            if len(image_paths) == 0:
                print(f"Warning: No cameras available for take {take_id}, scan {scan_idx}")
                failed += 1
                continue

            # Process frame
            success = process_frame(
                model=model,
                image_paths=image_paths,
                camera_ids=camera_ids,
                output_path=output_path,
                device=device,
                dtype=dtype,
                target_size=target_size,
                debug=args.debug,
            )

            if success:
                processed += 1
            else:
                failed += 1

            # Clear GPU cache periodically
            if processed % 100 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()

    # Save metadata
    metadata = {
        "teacher_model": "pi3",
        "model_source": args.checkpoint or "yyfz233/Pi3",
        "target_size": list(target_size),
        "max_cameras": 6,
        "total_frames": processed,
        "note": "Each frame may have 1-6 cameras; check camera_ids in NPZ",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print("CACHE GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Processed: {processed}")
    print(f"Skipped (resume): {skipped}")
    print(f"Failed: {failed}")
    print(f"Cache saved to: {output_dir}")
    print(f"\nPer-frame contents:")
    print(f"  xyz_local:     [N, {target_size[0]}, {target_size[1]}, 3] float16")
    print(f"  conf:          [N, {target_size[0]}, {target_size[1]}, 1] float16 (sigmoid applied)")
    print(f"  camera_poses:  [N, 4, 4] float32")
    print(f"  camera_ids:    [N] int8 (which cameras 1-6 are present)")
    print(f"  num_cameras:   N (1-6)")
    print(f"\nNote: N = number of available cameras per frame (varies 1-6)")

    return 0


if __name__ == "__main__":
    exit(main())
