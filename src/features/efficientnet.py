"""
EfficientNet Feature Extractor for Mask-Pooled Features

Supports both torchvision (ImageNet) and timm (NoisyStudent) variants.

Usage:
    # Standard ImageNet pretrained
    extractor = EfficientNetExtractor(variant='efficientnet_b5', device='cuda')

    # NoisyStudent variant (LABRAD-OR)
    extractor = EfficientNetExtractor(variant='tf_efficientnet_b5_ns', use_timm=True, device='cuda')

    features = extractor.extract_masked(image, masks)  # [N_masks, 2048]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class EfficientNetExtractor(nn.Module):
    """
    EfficientNet feature extractor with mask pooling.

    Supports torchvision (ImageNet) and timm (NoisyStudent) variants.
    """

    def __init__(
        self,
        variant: str = 'efficientnet_b5',
        use_timm: bool = False,
        device: str = 'cuda',
    ):
        """
        Args:
            variant: Model variant
                - torchvision: 'efficientnet_b5', 'efficientnet_b7', etc.
                - timm: 'tf_efficientnet_b5_ns', 'tf_efficientnet_b7_ns', etc.
            use_timm: If True, use timm library; else use torchvision
            device: 'cuda' or 'cpu'
        """
        super().__init__()

        self.variant = variant
        self.use_timm = use_timm
        self.device = device

        if use_timm:
            # Use timm for NoisyStudent variants
            import timm
            print(f"Loading {variant} from timm with NoisyStudent pretrained weights...")
            model = timm.create_model(variant, pretrained=True, features_only=True)
            # Get feature dimension from last feature layer
            self._feat_dim = model.feature_info.channels()[-1]
        else:
            # Use torchvision for standard ImageNet variants
            import torchvision.models as models

            model_loaders = {
                'efficientnet_b0': models.efficientnet_b0,
                'efficientnet_b1': models.efficientnet_b1,
                'efficientnet_b2': models.efficientnet_b2,
                'efficientnet_b3': models.efficientnet_b3,
                'efficientnet_b4': models.efficientnet_b4,
                'efficientnet_b5': models.efficientnet_b5,
                'efficientnet_b6': models.efficientnet_b6,
                'efficientnet_b7': models.efficientnet_b7,
            }

            if variant not in model_loaders:
                raise ValueError(
                    f"Unknown torchvision EfficientNet variant: {variant}\n"
                    f"Supported: {list(model_loaders.keys())}"
                )

            print(f"Loading {variant} from torchvision with ImageNet pretrained weights...")
            model = model_loaders[variant](weights='IMAGENET1K_V1')

            # Extract feature extractor (features module, before avgpool and classifier)
            self.feature_extractor = model.features

            # Feature dimension (output channels of last conv layer)
            # EfficientNet-B5: 2048 channels
            self._feat_dim = self.feature_extractor[-1][0].out_channels

        if use_timm:
            self.model = model
        else:
            self.feature_extractor = self.feature_extractor.to(device)
            self.feature_extractor.eval()

        # Move to device and freeze
        if use_timm:
            self.model = self.model.to(device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

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
        Preprocess image for EfficientNet.

        Args:
            image: [B, 3, H, W] RGB image in [0, 1] or [0, 255]

        Returns:
            Normalized image
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
        if self.use_timm:
            # timm features_only=True returns list of feature maps
            feature_maps = self.model(image)
            features = feature_maps[-1]  # Get final feature map
        else:
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
                masked_features = features[:, mask]  # [feat_dim, num_pixels]

                if pool_type == 'mean':
                    pooled.append(masked_features.mean(dim=1))
                elif pool_type == 'max':
                    pooled.append(masked_features.max(dim=1)[0])
                else:
                    raise ValueError(f"Unknown pool_type: {pool_type}")

        return torch.stack(pooled, dim=0)  # [N, feat_dim]
