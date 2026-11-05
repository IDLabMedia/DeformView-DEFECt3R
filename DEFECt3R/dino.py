"""
DINO ViT-S/16 feature extractor (low resolution, 384-d).

Uses the DINO backbone shipped with FeatUp ("dino16"), with FeatUp's ImageNet
normalisation, exactly as in MEt3R (`MEt3R._get_features`).
"""

from torch import Tensor
from torch.nn import Module

from .featup import load_featup_hub
from .utils import freeze_model, convert_to_buffer


class DINO(Module):

    def __init__(self, backbone: str = "dino16", use_norm: bool = True, freeze: bool = True) -> None:
        super().__init__()
        from featup.util import norm
        self.norm = norm
        self.feature_model = load_featup_hub(backbone, use_norm).model
        if freeze:
            freeze_model(self.feature_model)
            convert_to_buffer(self.feature_model, persistent=False)

    def forward(self, images: Tensor) -> Tensor:
        """
        Args:
            images: (B, 3, H, W) in [0, 1]
        Returns:
            (B, 384, H/16, W/16) patch features
        """
        return self.feature_model(self.norm(images))
