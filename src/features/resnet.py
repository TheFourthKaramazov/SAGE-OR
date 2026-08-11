"""
ResNet Feature Extractor for Mask-Pooled Features

Uses torchvision ResNet models with ImageNet pretraining.
Extracts features from final conv layer (layer4) before global pooling.

Usage:
    extractor = ResNetExtractor(variant='resnet50', device='cuda')
    features = extractor.extract_masked(image, masks)  # [N_masks, 2048]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple


class ResNetExtractor(nn.Module):
    """
    ResNet feature extractor with mask pooling.

    Supports ResNet-50, ResNet-101, ResNet-152 from torchvision.
    """

    def __init__(
        self,
        variant: str = 'resnet50',
        device: str = 'cuda',
    ):
        """
        Args:
            variant: 'resnet50', 'resnet101', or 'resnet152'
            device: 'cuda' or 'cpu'
        """
        super().__init__()

        self.variant = variant
        self.device = device

        # Model configurations
        model_loaders = {
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152,
        }

        if variant not in model_loaders:
            raise ValueError(f"Unknown ResNet variant: {variant}. Choose from {list(model_loaders.keys())}")

        print(f"Loading {variant} with ImageNet pretrained weights...")

        # Load pretrained model
        model = model_loaders[variant](weights='IMAGENET1K_V2')

        # Extract feature extractor (remove final avgpool and fc)
        self.feature_extractor = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,  # Output: [B, 2048, H/32, W/32] for ResNet-50/101/152
        )

        self.feature_extractor = self.feature_extractor.to(device)
        self.feature_extractor.eval()

        # Freeze all parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Feature dimension (ResNet layer4 output channels)
        self._feat_dim = 2048

        print(f"Loaded {variant} with {self._feat_dim}-dim features")

        # ImageNet normalization
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    @property
    def feat_dim(self) -> int:
        """Output feature dimension."""
        return self._feat_dim

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image for ResNet.

        Args:
            image: [B, 3, H, W] RGB image in [0, 1] or [0, 255]

        Returns:
            Normalized image ready for ResNet
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

    @torch.no_grad()
    def extract(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract dense feature map.

        Args:
            image: [B, 3, H, W] or [3, H, W] RGB image

        Returns:
            features: [B, feat_dim, H_feat, W_feat] dense feature map
        """
        # Handle single image
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Preprocess
        image = self._preprocess(image)

        # Move to model device
        image = image.to(self.device)

        # Extract features
        features = self.feature_extractor(image)

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
        features = self.extract(image)  # [1, feat_dim, H_feat, W_feat]
        features = features.squeeze(0)   # [feat_dim, H_feat, W_feat]

        feat_dim, H_feat, W_feat = features.shape

        # Resize masks to feature map resolution
        masks_tensor = masks.float().to(self.device)  # [N, H, W]
        masks_resized = F.interpolate(
            masks_tensor.unsqueeze(1),  # [N, 1, H, W]
            size=(H_feat, W_feat),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [N, H_feat, W_feat]

        # Threshold resized masks
        masks_resized = masks_resized > 0.5

        # Pool features for each mask
        pooled = []
        for i in range(N):
            mask = masks_resized[i]  # [H_feat, W_feat]

            if mask.sum() == 0:
                # Empty mask - return zeros
                pooled.append(torch.zeros(self.feat_dim, device=self.device))
            else:
                # Features is [feat_dim, H_feat, W_feat]
                # Mask is [H_feat, W_feat]
                # Select features where mask is True
                masked_features = features[:, mask]  # [feat_dim, num_pixels]

                if pool_type == 'mean':
                    pooled.append(masked_features.mean(dim=1))
                elif pool_type == 'max':
                    pooled.append(masked_features.max(dim=1)[0])
                else:
                    raise ValueError(f"Unknown pool_type: {pool_type}")

        return torch.stack(pooled, dim=0)  # [N, feat_dim]
