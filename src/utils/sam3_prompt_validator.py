"""
Validate SAM3 prompt configurations before full cache generation.

Processes a sample of frames and generates:
- Visualizations with detected masks overlaid on images
- Detection statistics (instance counts, coverage rates)

This enables rapid validation of new prompts for OR context adaptation.
"""

import sys
from pathlib import Path
import argparse
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

from src.dataset.mask_generator import MaskGenerator


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_images_from_manifest(manifest_path, take_id):
    """Load image paths for a specific take from manifest."""
    manifest = pd.read_csv(manifest_path)
    take_frames = manifest[manifest['take_id'] == take_id]

    images = []
    for _, row in take_frames.iterrows():
        for cam_id in range(1, 7):
            processed_path = row.get(f'cam{cam_id}_processed')
            original_path = row.get(f'cam{cam_id}_rgb')

            if pd.notna(processed_path):
                images.append({
                    'take_id': take_id,
                    'scan_idx': row['scan_idx'],
                    'cam_id': cam_id,
                    'processed_path': Path(processed_path) if pd.notna(processed_path) else None,
                    'original_path': Path(original_path) if pd.notna(original_path) else None,
                })

    return images


def visualize_masks(image, results, prompt_names, output_path):
    """Create visualization with detected masks overlaid."""
    image_np = np.array(image)

    # Create subplot grid
    n_prompts = len(prompt_names)
    n_cols = min(3, n_prompts)
    n_rows = (n_prompts + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    if n_prompts == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes

    for idx, prompt_name in enumerate(prompt_names):
        ax = axes[idx]
        ax.imshow(image_np)

        n_instances = 0
        if prompt_name in results and 'masks' in results[prompt_name]:
            masks = results[prompt_name]['masks']
            scores = results[prompt_name]['scores']
            n_instances = len(masks)

            # Overlay masks with distinct colors
            colors = plt.cm.rainbow(np.linspace(0, 1, max(1, len(masks))))

            for mask, score, color in zip(masks, scores, colors):
                # Create colored overlay
                overlay = image_np.copy().astype(np.float32)
                colored_mask = np.zeros_like(image_np, dtype=np.float32)
                colored_mask[mask > 0] = np.array(color[:3]) * 255
                overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)

                ax.imshow(overlay.astype(np.uint8))

                # Add confidence score at centroid
                y_coords, x_coords = np.where(mask > 0)
                if len(y_coords) > 0:
                    cy, cx = int(y_coords.mean()), int(x_coords.mean())
                    ax.text(cx, cy, f'{score:.2f}', color='white', fontsize=8,
                           weight='bold', bbox=dict(boxstyle='round',
                           facecolor=color, alpha=0.5))

        ax.set_title(f"{prompt_name}: {n_instances} instances", fontsize=10)
        ax.axis('off')

    # Hide unused subplots
    for idx in range(n_prompts, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def print_statistics(detection_counts, prompt_names):
    """Print detection statistics."""
    print("\n" + "="*60)
    print("DETECTION STATISTICS")
    print("="*60)

    for prompt_name in prompt_names:
        counts = detection_counts.get(prompt_name, [])
        if not counts or sum(counts) == 0:
            print(f"\n{prompt_name}: NO DETECTIONS")
            continue

        total = sum(counts)
        n_frames = len(counts)
        n_detected = sum(c > 0 for c in counts)

        print(f"\n{prompt_name}:")
        print(f"  Frames with detections: {n_detected}/{n_frames} ({100*n_detected/n_frames:.1f}%)")
        print(f"  Avg instances/frame: {np.mean(counts):.2f}")
        print(f"  Max instances: {max(counts)}")
        print(f"  Total instances: {total}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate SAM3 prompts before full cache generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m src.utils.sam3_prompt_validator --config configs/sam3_prompts.yaml --take 1

This will process all frames from Take 1, sample every 5th frame for visualization,
and output statistics + visualizations to outputs/sam3_validation/.
        """
    )
    parser.add_argument('--config', required=True,
                       help='SAM3 config file with prompts to validate')
    parser.add_argument('--manifest', default='processed_data/manifest.csv',
                       help='Path to manifest CSV')
    parser.add_argument('--take', type=int, default=1,
                       help='Take ID to process')
    parser.add_argument('--sample-every', type=int, default=5,
                       help='Visualize every Nth frame (default: 5)')
    parser.add_argument('--output-dir', default='outputs/sam3_validation',
                       help='Output directory for visualizations')
    parser.add_argument('--prompts', nargs='*', default=None,
                       help='Specific prompts to validate (default: all prompts in config)')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which prompts to validate
    if args.prompts:
        prompts_to_validate = args.prompts
    else:
        # Validate all prompts in config
        prompts_to_validate = list(config['prompts'].keys())

    print(f"\nValidating SAM3 prompts on Take {args.take}")
    print(f"Config: {args.config}")
    print(f"Prompts to validate: {prompts_to_validate}")
    print(f"Output: {output_dir}")

    # Load images
    images = get_images_from_manifest(args.manifest, args.take)
    sampled_images = images[::args.sample_every]
    print(f"\nFound {len(images)} images")
    print(f"Sampling every {args.sample_every}th: {len(sampled_images)} images")

    # Initialize SAM3
    generator = MaskGenerator(config)
    generator.load_model()

    # Process images
    detection_counts = {name: [] for name in prompts_to_validate}

    for img_info in tqdm(sampled_images, desc="Processing"):
        try:
            # Load images
            low_res = None
            high_res = None

            if img_info['processed_path'] and img_info['processed_path'].exists():
                low_res = Image.open(img_info['processed_path']).convert('RGB')

            if img_info['original_path'] and img_info['original_path'].exists():
                high_res = Image.open(img_info['original_path']).convert('RGB')

            if low_res is None:
                continue

            # Run SAM3
            results = generator.process_image(low_res, high_res)

            # Count detections
            for prompt_name in prompts_to_validate:
                n = len(results.get(prompt_name, {}).get('masks', []))
                detection_counts[prompt_name].append(n)

            # Save visualization
            output_path = output_dir / f"take{args.take:02d}_scan{img_info['scan_idx']:04d}_cam{img_info['cam_id']}.png"
            results_subset = {k: v for k, v in results.items() if k in prompts_to_validate}
            visualize_masks(low_res, results_subset, prompts_to_validate, output_path)

        except Exception as e:
            print(f"\nError processing {img_info}: {e}")
            continue

    # Print statistics
    print_statistics(detection_counts, prompts_to_validate)
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
