"""
Unified Visual Encoder Interface

Provides a common interface for all visual feature extractors:
- DINOv3 (ViT-S/B/L/7B)
- ResNet-50
- EfficientNet-B5/B5-NS
- CLIP ViT-L/14

Usage:
    encoder = create_encoder('dinov3_vitl16', device='cuda')
    encoder = create_encoder('resnet50', device='cuda')
    encoder = create_encoder('clip_vitl14', device='cuda')

    features = encoder.extract_masked(image, masks)  # [N, feat_dim]
"""

import torch
import torch.nn as nn
from typing import Optional


class VisualEncoder(nn.Module):
    """Base interface for all visual encoders."""

    @property
    def feat_dim(self) -> int:
        """Output feature dimension."""
        raise NotImplementedError

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
        raise NotImplementedError


def create_encoder(encoder_name: str, device: str = 'cuda') -> VisualEncoder:
    """
    Factory to create any visual encoder.

    Args:
        encoder_name: Model identifier
            - DINOv3: 'dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16', 'dinov3_vit7b16'
            - ResNet: 'resnet50'
            - EfficientNet: 'efficientnet_b5', 'efficientnet_b5_ns'
            - CLIP: 'clip_vitl14', 'clip_vitl14_336'
        device: 'cuda' or 'cpu'

    Returns:
        Encoder instance conforming to VisualEncoder interface
    """
    # DINOv3 models
    if encoder_name.startswith('dinov3'):
        from .dinov3 import DINOv3Extractor
        return DINOv3Extractor(model_name=encoder_name, device=device)

    # ResNet models
    elif encoder_name.startswith('resnet'):
        from .resnet import ResNetExtractor
        variant = encoder_name  # 'resnet50', 'resnet101', etc.
        return ResNetExtractor(variant=variant, device=device)

    # EfficientNet models
    elif encoder_name.startswith('efficientnet'):
        from .efficientnet import EfficientNetExtractor
        if '_ns' in encoder_name:
            # NoisyStudent variant via timm
            # encoder_name = 'efficientnet_b5_ns' -> 'tf_efficientnet_b5_ns'
            variant = encoder_name.replace('efficientnet_', 'tf_efficientnet_')
            use_timm = True
        else:
            # Standard ImageNet via torchvision
            variant = encoder_name
            use_timm = False
        return EfficientNetExtractor(variant=variant, use_timm=use_timm, device=device)

    # CLIP models
    elif encoder_name.startswith('clip'):
        from .clip_encoder import CLIPExtractor
        if '336' in encoder_name:
            model_name = 'openai/clip-vit-large-patch14-336'
        else:
            model_name = 'openai/clip-vit-large-patch14'
        return CLIPExtractor(model_name=model_name, device=device)

    else:
        raise ValueError(
            f"Unknown encoder: {encoder_name}\n"
            f"Supported: dinov3_*, resnet*, efficientnet_*, clip_*"
        )


def get_encoder_info(encoder_name: str) -> dict:
    """Get encoder specifications without loading the model."""

    specs = {
        # DINOv3
        'dinov3_vits16': {'feat_dim': 384, 'description': 'DINOv3 ViT-S/16'},
        'dinov3_vitb16': {'feat_dim': 768, 'description': 'DINOv3 ViT-B/16'},
        'dinov3_vitl16': {'feat_dim': 1024, 'description': 'DINOv3 ViT-L/16'},
        'dinov3_vit7b16': {'feat_dim': 4096, 'description': 'DINOv3 ViT-7B/16'},

        # ResNet
        'resnet50': {'feat_dim': 2048, 'description': 'ResNet-50 (ImageNet)'},

        # EfficientNet
        'efficientnet_b5': {'feat_dim': 2048, 'description': 'EfficientNet-B5 (ImageNet)'},
        'efficientnet_b5_ns': {'feat_dim': 512, 'description': 'EfficientNet-B5-NS (NoisyStudent)'},

        # CLIP
        'clip_vitl14': {'feat_dim': 1024, 'description': 'CLIP ViT-L/14 (224px)'},
        'clip_vitl14_336': {'feat_dim': 1024, 'description': 'CLIP ViT-L/14 (336px)'},
    }

    if encoder_name in specs:
        return specs[encoder_name]
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")
