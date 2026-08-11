"""
CLIP Vision Encoder for Mask-Pooled Features

Uses HuggingFace transformers CLIP vision encoder.
Supports ViT-L/14 at 224px and 336px resolutions.

Usage:
    # Standard 224px
    extractor = CLIPExtractor(model_name='openai/clip-vit-large-patch14', device='cuda')

    # High-res 336px (MM-OR)
    extractor = CLIPExtractor(model_name='openai/clip-vit-large-patch14-336', device='cuda')

    features = extractor.extract_masked(image, masks)  # [N_masks, 1024]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor
from typing import Tuple


class CLIPExtractor(nn.Module):
    """
    CLIP Vision Transformer feature extractor with mask pooling.

    Uses patch tokens before final pooling for dense features.
    """

    def __init__(
        self,
        model_name: str = 'openai/clip-vit-large-patch14-336',
        device: str = 'cuda',
    ):
        """
        Args:
            model_name: HuggingFace model identifier
                - 'openai/clip-vit-large-patch14' (224px)
                - 'openai/clip-vit-large-patch14-336' (336px)
            device: 'cuda' or 'cpu'
        """
        super().__init__()

        self.model_name = model_name
        self.device = device

        print(f"Loading CLIP vision encoder: {model_name}...")

        # Load CLIP vision model
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Load image processor for preprocessing
        self.processor = CLIPImageProcessor.from_pretrained(model_name)

        # Feature dimension (CLIP-L is 1024-dim)
        self._feat_dim = self.model.config.hidden_size

        # Patch size
        self.patch_size = self.model.config.patch_size

        # Image size
        self.image_size = self.model.config.image_size

        print(f"Loaded CLIP with {self._feat_dim}-dim features")
        print(f"  Patch size: {self.patch_size}, Image size: {self.image_size}")

    @property
    def feat_dim(self) -> int:
        """Output feature dimension."""
        return self._feat_dim

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image for CLIP.

        Args:
            image: [B, 3, H, W] RGB image in [0, 1] or [0, 255]

        Returns:
            Preprocessed image ready for CLIP
        """
        # Convert to float if needed
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        elif image.max() > 1.0:
            image = image / 255.0

        # CLIP uses its own preprocessing (resize + normalize)
        # For simplicity, we'll just normalize with CLIP stats here
        # Note: CLIP expects images resized to model.config.image_size

        # Resize to CLIP's expected input size
        if image.shape[2] != self.image_size or image.shape[3] != self.image_size:
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )

        # CLIP normalization (from CLIPImageProcessor)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image.device).view(1, 3, 1, 1)

        image = (image - mean) / std

        return image

    def _get_patch_grid_size(self, H: int, W: int) -> Tuple[int, int]:
        """Get patch grid size for given input size."""
        # After preprocessing, images are resized to self.image_size
        h_patches = self.image_size // self.patch_size
        w_patches = self.image_size // self.patch_size
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

        # Preprocess
        image = self._preprocess(image)

        # Move to model device
        image = image.to(self.device)

        # Get patch grid size
        h_patches, w_patches = self._get_patch_grid_size(H, W)

        # Extract features
        outputs = self.model(pixel_values=image, output_hidden_states=True)

        # Get patch tokens (last hidden state, excluding CLS token)
        # Shape: [B, num_patches + 1, hidden_size]
        last_hidden_state = outputs.last_hidden_state

        # Remove CLS token (first token)
        patch_tokens = last_hidden_state[:, 1:, :]  # [B, num_patches, feat_dim]

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
        # Original masks are at original image resolution, but CLIP resizes to self.image_size
        # So we need to resize masks to patch grid of the resized image
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
