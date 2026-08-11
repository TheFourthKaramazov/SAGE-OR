"""
DINOv3 feature extractor for dense patch features.

Usage:
    extractor = DINOv3Extractor(device='cuda')

    # Get dense patch features
    features = extractor.extract(image)  # [1, H_patches, W_patches, 1024]

    # Get mask-pooled features
    features = extractor.extract_masked(image, masks)  # [N_masks, 1024]
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Add dinov3 to path
DINOV3_PATH = Path(__file__).parent.parent.parent / 'external' / 'dinov3'
sys.path.insert(0, str(DINOV3_PATH))

from dinov3.hub.backbones import dinov3_vits16, dinov3_vitb16, dinov3_vitl16, dinov3_vit7b16


# Model configurations
MODEL_CONFIGS = {
    'dinov3_vits16': {
        'loader': dinov3_vits16,
        'feat_dim': 384,
        'weights_file': 'dinov3_vits16_pretrain.pth',
        'description': 'ViT-S/16 (22M params, 384-dim)'
    },
    'dinov3_vitb16': {
        'loader': dinov3_vitb16,
        'feat_dim': 768,
        'weights_file': 'dinov3_vitb16_pretrain.pth',
        'description': 'ViT-B/16 (86M params, 768-dim)'
    },
    'dinov3_vitl16': {
        'loader': dinov3_vitl16,
        'feat_dim': 1024,
        'weights_file': 'dinov3_vitl16_pretrain.pth',
        'description': 'ViT-L/16 (300M params, 1024-dim)'
    },
    'dinov3_vit7b16': {
        'loader': dinov3_vit7b16,
        'feat_dim': 4096,
        'weights_file': 'dinov3_vit7b16_pretrain.pth',
        'description': 'ViT-7B/16 (6.7B params, 4096-dim)'
    }
}


class DINOv3Extractor(nn.Module):
    """
    DINOv3 dense feature extractor.

    Supports ViT-L/16 (1024-dim) and ViT-7B/16 (4096-dim).
    Images 518x392 -> 32x24 patches (16x16 patch size).
    """

    def __init__(
        self,
        model_name: str = 'dinov3_vitl16',
        weights_path: str = None,
        device: str = 'cuda',
    ):
        """
        Args:
            model_name: 'dinov3_vitl16' or 'dinov3_vit7b16'
            weights_path: Path to weights file (optional, will use default)
            device: 'cuda' or 'cpu'
        """
        super().__init__()

        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")

        self.model_name = model_name
        self.device = device
        self.patch_size = 16

        config = MODEL_CONFIGS[model_name]

        # Default weights path
        if weights_path is None:
            weights_path = str(
                Path(__file__).parent.parent.parent / 'weights' / 'dinov3' / config['weights_file']
            )

        # Load model architecture without pretrained weights
        print(f"Loading DINOv3 {config['description']} architecture...")
        self.model = config['loader'](pretrained=False)

        # Load weights manually
        print(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=True)

        self.model = self.model.to(device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Feature dimension from config
        self.feat_dim = config['feat_dim']
        print(f"Loaded {model_name} with {self.feat_dim}-dim features")

        # ImageNet normalization
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image for DINOv3.

        Args:
            image: [B, 3, H, W] RGB image in [0, 1] or [0, 255]

        Returns:
            Normalized image ready for DINOv3
        """
        # Convert to float if needed
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        elif image.max() > 1.0:
            image = image / 255.0

        # Get mean/std on same device as image
        mean = self.mean.to(image.device)
        std = self.std.to(image.device)

        # Normalize with ImageNet stats
        image = (image - mean) / std

        return image

    def _get_patch_grid_size(self, H: int, W: int) -> Tuple[int, int]:
        """Get output patch grid size for given input size."""
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        return h_patches, w_patches

    @torch.no_grad()
    def extract(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract dense patch features.

        Args:
            image: [B, 3, H, W] or [3, H, W] RGB image

        Returns:
            features: [B, H_patches, W_patches, feat_dim] dense patch features
        """
        # Handle single image
        if image.dim() == 3:
            image = image.unsqueeze(0)

        B, C, H, W = image.shape

        # Preprocess first (works on any device)
        image = self._preprocess(image)

        # Move to model device
        image = image.to(self.device)

        # Get patch grid size
        h_patches, w_patches = self._get_patch_grid_size(H, W)

        # Extract features - DINOv3 returns dict with patch tokens
        output = self.model.forward_features(image)

        # Get patch tokens (excluding CLS and storage tokens)
        if 'x_norm_patchtokens' in output:
            patch_tokens = output['x_norm_patchtokens']  # [B, N_patches, feat_dim]
        else:
            # Fallback: get all tokens and exclude special tokens
            x = output['x_norm']
            n_special = 1 + self.model.n_storage_tokens  # CLS + storage tokens
            patch_tokens = x[:, n_special:, :]

        # Reshape to spatial grid
        features = patch_tokens.reshape(B, h_patches, w_patches, self.feat_dim)

        return features

    @torch.no_grad()
    def extract_masked(
        self,
        image: torch.Tensor,
        masks: torch.Tensor,
        pool_type: str = 'mean'
    ) -> torch.Tensor:
        """
        Extract mask-pooled features.

        Args:
            image: [B, 3, H, W] or [3, H, W] RGB image
            masks: [N, H, W] boolean masks at image resolution
            pool_type: 'mean' or 'max'

        Returns:
            features: [N, feat_dim] pooled features per mask
        """
        # Handle single image
        if image.dim() == 3:
            image = image.unsqueeze(0)

        assert image.shape[0] == 1, "Batch size must be 1 for mask pooling"

        B, C, H, W = image.shape
        N = masks.shape[0]

        # Extract dense features
        features = self.extract(image)  # [1, h_patches, w_patches, feat_dim]
        features = features.squeeze(0)   # [h_patches, w_patches, feat_dim]

        h_patches, w_patches = features.shape[:2]

        # Resize masks to patch resolution
        masks_tensor = masks.float().to(self.device)  # [N, H, W]
        masks_resized = F.interpolate(
            masks_tensor.unsqueeze(1),  # [N, 1, H, W]
            size=(h_patches, w_patches),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [N, h_patches, w_patches]

        # Threshold resized masks
        masks_resized = masks_resized > 0.5

        # Pool features for each mask
        pooled = []
        for i in range(N):
            mask = masks_resized[i]  # [h_patches, w_patches]

            if mask.sum() == 0:
                # Empty mask - return zeros
                pooled.append(torch.zeros(self.feat_dim, device=self.device))
            else:
                masked_features = features[mask]  # [num_patches, feat_dim]

                if pool_type == 'mean':
                    pooled.append(masked_features.mean(dim=0))
                elif pool_type == 'max':
                    pooled.append(masked_features.max(dim=0)[0])
                else:
                    raise ValueError(f"Unknown pool_type: {pool_type}")

        return torch.stack(pooled, dim=0)  # [N, feat_dim]

    def get_output_size(self, H: int, W: int) -> Tuple[int, int]:
        """Get output feature map size for given input size."""
        return self._get_patch_grid_size(H, W)
