"""
MEt3R baseline (Asim et al., CVPR 2025) as evaluated in the paper.

Uses the same DINO + FeatUp features and correspondences as DEFECt3R (imported
from the DEFECt3R package) and scores every overlap pixel with the cosine
distance of the two views' features:

    score = 1 - cos(f1, f2) = ||f1/|f1| - f2/|f2| ||^2 / 2      in [0, 2]

Pair-level MEt3R scores are aggregated from this map (mean = original MEt3R).

Usage:
    model = MEt3R(correspondence="mast3r").to("cuda")
    out = model(img1, img2)       # images in [-1, 1], (3, H, W) or (B, 3, H, W)
    out["heatmap"]                # cosine distance, NaN outside overlap
"""

from typing import Optional

from torch import Tensor

from DEFECt3R.aligned_features import AlignedFeatureModel


# Pixel-level decision thresholds, selected on the validation split by maximising F1.
THRESHOLDS = {
    "mast3r": 0.3343, # based on DeformView validation split
    "gt":     0.3538, # based on DeformView validation split
}


class MEt3R(AlignedFeatureModel):

    def __init__(self, correspondence: str = "mast3r", img_size: int = 512,
                 threshold: Optional[float] = None) -> None:
        """
        Args:
            correspondence: "mast3r" (estimated) or "gt" (GT poses + depth, DeformView only)
            img_size:       input resolution
            threshold:      override the pixel-level decision threshold
        """
        super().__init__(
            correspondence=correspondence,
            use_mast3r_features=False,
            img_size=img_size,
            threshold=THRESHOLDS.get(correspondence) if threshold is None else threshold,
        )
        self.eval()

    def score(self, dino_diff: Tensor, mast3r_diff: Optional[Tensor]) -> Tensor:
        # dino_diff is the difference of L2-normalised features: ||diff||^2 / 2 = 1 - cos
        return 0.5 * (dino_diff ** 2).sum(dim=-1)
