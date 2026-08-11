#!/usr/bin/env python3
"""
Config-driven SAM3 Mask Generator

Features:
- Per-prompt resolution selection (high-res for small objects, low-res for large)
- Configurable confidence thresholds per prompt
- All masks resized to common output resolution
- Modular design using YAML config

Usage:
    python -m src.dataset.mask_generator --config configs/sam3_prompts.yaml
    python -m src.dataset.mask_generator --config configs/sam3_prompts.yaml --takes 1 3 5
    python -m src.dataset.mask_generator --config configs/sam3_prompts.yaml --num-frames 20
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def lazy_load_sam3():
    """Lazily import SAM3."""
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    return build_sam3_image_model, Sam3Processor


def resize_mask(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize mask to target resolution using nearest neighbor."""
    if mask.shape[0] == target_hw[0] and mask.shape[1] == target_hw[1]:
        return mask
    return cv2.resize(
        mask.astype(np.uint8),
        (target_hw[1], target_hw[0]),  # cv2 uses (W, H)
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)


def get_masks_numpy(masks, scores, boxes, threshold: float):
    """Convert SAM3 outputs to numpy arrays, filtered by confidence."""
    if masks is None or (torch.is_tensor(masks) and masks.numel() == 0):
        return np.array([], dtype=bool).reshape(0, 0, 0), np.array([], dtype=np.float32), np.array([], dtype=np.float32).reshape(0, 4)

    if torch.is_tensor(masks):
        masks_np = masks.cpu().numpy()
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0, :, :]
    else:
        masks_np = np.array(masks)

    if torch.is_tensor(scores):
        scores_np = scores.cpu().numpy()
    else:
        scores_np = np.array(scores) if scores is not None else np.array([])

    if torch.is_tensor(boxes):
        boxes_np = boxes.cpu().numpy()
    else:
        boxes_np = np.array(boxes) if boxes is not None else np.array([])

    if len(scores_np) == 0:
        h, w = masks_np.shape[-2:] if masks_np.ndim >= 2 else (0, 0)
        return np.array([], dtype=bool).reshape(0, h, w), np.array([], dtype=np.float32), np.array([], dtype=np.float32).reshape(0, 4)

    valid = scores_np > threshold
    if not valid.any():
        h, w = masks_np.shape[-2:] if masks_np.ndim >= 2 else (0, 0)
        return np.array([], dtype=bool).reshape(0, h, w), np.array([], dtype=np.float32), np.array([], dtype=np.float32).reshape(0, 4)

    return (
        masks_np[valid].astype(bool),
        scores_np[valid].astype(np.float32),
        boxes_np[valid].astype(np.float32) if len(boxes_np) > 0 and boxes_np.ndim == 2 else np.array([], dtype=np.float32).reshape(0, 4)
    )


class MaskGenerator:
    """Config-driven SAM3 mask generator with dual-resolution support."""

    def __init__(self, config: dict):
        self.config = config
        self.output_hw = tuple(config['output_resolution'])
        self.prompts = config['prompts']
        self.resolutions = config['resolutions']

        self.low_res_prompts = {k: v for k, v in self.prompts.items() if v['resolution'] == 'low'}
        self.high_res_prompts = {k: v for k, v in self.prompts.items() if v['resolution'] == 'high'}

        self.processor = None

    def load_model(self):
        """Load SAM3 model."""
        if self.processor is None:
            print("Loading SAM3 model...")
            build_sam3_image_model, Sam3Processor = lazy_load_sam3()
            model = build_sam3_image_model()
            self.processor = Sam3Processor(model)
            print("SAM3 loaded!")

    def _run_prompts(self, image: Image.Image, prompts: dict) -> dict:
        """Run a set of prompts on an image."""
        state = self.processor.set_image(image)
        results = {}

        for name, prompt_config in prompts.items():
            try:
                output = self.processor.set_text_prompt(state=state, prompt=prompt_config['text'])
                masks, scores, boxes = get_masks_numpy(
                    output.get("masks"),
                    output.get("scores"),
                    output.get("boxes"),
                    threshold=prompt_config['confidence'],
                )
                results[name] = {"masks": masks, "scores": scores, "boxes": boxes}
            except Exception as e:
                print(f"Error with prompt '{prompt_config['text']}': {e}")
                results[name] = {
                    "masks": np.array([], dtype=bool).reshape(0, 0, 0),
                    "scores": np.array([], dtype=np.float32),
                    "boxes": np.array([], dtype=np.float32).reshape(0, 4),
                }

        return results

    def process_image(
        self,
        low_res_image: Optional[Image.Image],
        high_res_image: Optional[Image.Image],
    ) -> dict:
        """Process an image with dual-resolution prompts."""
        results = {}

        if low_res_image and self.low_res_prompts:
            low_results = self._run_prompts(low_res_image, self.low_res_prompts)
            results.update(low_results)

        if high_res_image and self.high_res_prompts:
            high_results = self._run_prompts(high_res_image, self.high_res_prompts)
            results.update(high_results)

        # Resize all masks to output resolution
        for name, data in results.items():
            if data['masks'].size > 0 and data['masks'].shape[1:] != self.output_hw:
                resized_masks = np.stack([
                    resize_mask(m, self.output_hw) for m in data['masks']
                ])
                data['masks'] = resized_masks

        return results

    def save_results(self, results: dict, output_path: Path):
        """Save detection results to compressed npz."""
        save_dict = {}
        for name, data in results.items():
            save_dict[f"{name}_masks"] = data["masks"]
            save_dict[f"{name}_scores"] = data["scores"]
            save_dict[f"{name}_boxes"] = data["boxes"]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **save_dict)


def get_images_from_manifest(
    manifest_path: Path,
    takes: Optional[List[int]] = None,
    num_frames: Optional[int] = None,
) -> List[dict]:
    """Get list of images to process from manifest.

    Args:
        manifest_path: Path to manifest CSV
        takes: Optional list of take IDs to filter
        num_frames: Number of FRAMES to sample (each frame has 6 cameras)
    """
    df = pd.read_csv(manifest_path)

    if takes:
        df = df[df["take_id"].isin(takes)]

    # Sample frames first, then expand to all cameras
    if num_frames and num_frames < len(df):
        indices = np.linspace(0, len(df) - 1, num_frames, dtype=int)
        df = df.iloc[indices]

    images = []
    for _, row in df.iterrows():
        take_id = row["take_id"]
        scan_idx = row["scan_idx"]

        for cam_id in range(1, 7):
            processed_col = f"cam{cam_id}_processed"
            original_col = f"cam{cam_id}_rgb"

            processed_path = row.get(processed_col, "")
            original_path = row.get(original_col, "")

            if not processed_path or pd.isna(processed_path):
                continue

            images.append({
                "processed_path": Path(processed_path) if processed_path and not pd.isna(processed_path) else None,
                "original_path": Path(original_path) if original_path and not pd.isna(original_path) else None,
                "take_id": take_id,
                "scan_idx": scan_idx,
                "cam_id": cam_id,
                "output_name": f"frame_{scan_idx:04d}_cam{cam_id}.npz",
            })

    return images


def main():
    parser = argparse.ArgumentParser(description="Config-driven SAM3 mask generator")
    parser.add_argument("--config", type=str, default="configs/sam3_prompts.yaml")
    parser.add_argument("--manifest", type=str, default="processed_data/manifest.csv")
    parser.add_argument("--output-dir", type=str, default="caches/sam3_masks")
    parser.add_argument("--takes", type=int, nargs="+", default=None)
    parser.add_argument("--num-frames", type=int, default=None, help="Number of frames to process (6 images per frame)")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    resume = not args.no_resume

    config = load_config(config_path)

    print("\n" + "=" * 60)
    print("SAM3 MASK GENERATOR")
    print("=" * 60)
    print(f"Output resolution: {config['output_resolution']}")
    print(f"\nPrompts:")
    for name, cfg in config['prompts'].items():
        print(f"  {name}: '{cfg['text']}' @ {cfg['resolution']} res, conf>{cfg['confidence']}")

    images = get_images_from_manifest(manifest_path, args.takes, args.num_frames)
    print(f"\nFound {len(images)} images")

    to_process = []
    for img in images:
        output_path = output_dir / f"take{img['take_id']:02d}" / img["output_name"]
        if resume and output_path.exists():
            continue
        to_process.append(img)

    print(f"To process: {len(to_process)}")

    if len(to_process) == 0:
        print("Nothing to process!")
        return 0

    generator = MaskGenerator(config)
    generator.load_model()

    print(f"\nProcessing {len(to_process)} images...")
    start_time = time.time()
    processed_count = 0
    error_count = 0

    for img in tqdm(to_process, desc="Generating masks"):
        try:
            low_res_image = None
            high_res_image = None

            if img["processed_path"] and img["processed_path"].exists():
                low_res_image = Image.open(img["processed_path"]).convert("RGB")

            if img["original_path"] and img["original_path"].exists():
                high_res_image = Image.open(img["original_path"]).convert("RGB")

            if low_res_image is None and high_res_image is None:
                continue

            results = generator.process_image(low_res_image, high_res_image)

            output_path = output_dir / f"take{img['take_id']:02d}" / img["output_name"]
            generator.save_results(results, output_path)
            processed_count += 1

        except Exception as e:
            print(f"\nError: {e}")
            error_count += 1

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Time: {total_time/60:.1f} minutes")

    metadata = {
        "created": datetime.now().isoformat(),
        "config": config,
        "processed": processed_count,
        "errors": error_count,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return 0


if __name__ == "__main__":
    exit(main())
