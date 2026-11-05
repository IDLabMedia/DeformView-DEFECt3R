"""
FeatUp upsampler (JBU) used to lift low-resolution DINO features to the input
resolution.  Adapted from MEt3R (`MEt3R._interpolate`).

FeatUp ships the backbone and its upsampler as one torch.hub object; it is
loaded once and shared with `dino.py`.
"""

from functools import lru_cache

import torch
from torch import Tensor
from torch.nn import Module, functional as F

from .utils import freeze_model, convert_to_buffer


FEATUP_HUB = "mhamilton723/FeatUp"


@lru_cache(maxsize=None)
def load_featup_hub(backbone: str = "dino16", use_norm: bool = True):
    """Load (once) the FeatUp hub model: `.model` = backbone, `.upsampler` = JBU."""
    return torch.hub.load(FEATUP_HUB, backbone, use_norm=use_norm)


class FeatUp(Module):

    def __init__(self, backbone: str = "dino16", use_norm: bool = True, freeze: bool = True) -> None:
        super().__init__()
        self.upsampler_model = load_featup_hub(backbone, use_norm).upsampler
        if freeze:
            freeze_model(self.upsampler_model)
            convert_to_buffer(self.upsampler_model, persistent=False)

    def forward(self, lr_feat: Tensor, images: Tensor) -> Tensor:
        """
        Args:
            lr_feat: (B, C, h, w) low-resolution backbone features
            images:  (B, 3, H, W) guidance images in [0, 1]
        Returns:
            (B, C, H, W) upsampled features
        """
        feat = self.upsampler_model(lr_feat, images)
        # Important for specific backbone which may not return with correct dimensions
        feat = F.interpolate(feat, (images.shape[-2:]), mode="bilinear")
        return feat
