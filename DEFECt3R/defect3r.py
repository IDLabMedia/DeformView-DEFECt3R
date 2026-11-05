"""
DEFECt3R: pixel-level geometric inconsistency localization for image pairs.

Pipeline (see the paper, Section 4):
    1. DINO ViT-S/16 features of both views, upsampled with FeatUp     (dino.py, featup.py)
    2. pixel correspondences from MASt3R or from GT poses + depth       (mast3r.py, gt_correspondence.py)
    3. per overlap pixel: L2-normalised feature differences of the two
       views (DINO, optionally + MASt3R descriptors)                   (aligned_features.py)
    4. lightweight MLP → inconsistency probability per pixel            (classifier.py)

Three trained variants exist (features × correspondence):
    dino+mast3r / mast3r,   dino / mast3r,   dino / gt

Usage:
    model = DEFECt3R(features="dino+mast3r", correspondence="mast3r").to("cuda")
    out = model(img1, img2)                  # images in [-1, 1], (3, H, W) or (B, 3, H, W)
    out["heatmap"]                           # inconsistency probability, NaN outside overlap
    out["prediction"]                        # heatmap > model.threshold
"""

from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor

from .aligned_features import AlignedFeatureModel, CORRESPONDENCES
from .classifier import DinoMASt3RMLP, DinoMLP


WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

# Trained variants: classifier, weights and the pixel-level decision threshold
# (selected on the validation split by maximising F1).
# Original training runs:
#   dino_mast3r__mast3r.pt  checkpoints_full_pairs_s3_sampled_bg_inbalanced_20260709_093944/best_model.pt
#   dino__mast3r.pt         checkpoints_full_pairs_s3_sampled_bg_inbalanced_dino_only_20260714_044942/best_model.pt
#   dino__gt.pt             checkpoints_gt_poses_sampled_bg_inbalanced_20260712_192222/best_model_E26.pt
VARIANTS = {
    ("dino+mast3r", "mast3r"): dict(
        classifier=DinoMASt3RMLP,
        weights="dino_mast3r__mast3r.pt",   # trained on DeformView train split
        threshold=0.3965,                   # based on DeformView validation split
    ),
    ("dino", "mast3r"): dict(
        classifier=DinoMLP,
        weights="dino__mast3r.pt",  # trained on DeformView train split
        threshold=0.3665,           # based on DeformView validation split
    ),
    ("dino", "gt"): dict(
        classifier=DinoMLP,
        weights="dino__gt.pt",  # trained on DeformView train split
        threshold=0.4011,       # based on DeformView validation split
    ),
}

FEATURES = ("dino", "dino+mast3r")

INFERENCE_CHUNK = 65536   # pixels per MLP forward pass


def load_image(path, img_size: int = 512) -> Tensor:
    """Load an image as a (3, img_size, img_size) tensor in [-1, 1]."""
    pre = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Lambda(lambda x: x * 2 - 1),
    ])
    return pre(Image.open(path).convert("RGB"))


class DEFECt3R(AlignedFeatureModel):

    def __init__(
        self,
        features: str = "dino+mast3r",
        correspondence: str = "mast3r",
        img_size: int = 512,
        weights: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> None:
        """
        Args:
            features:       "dino" or "dino+mast3r" (the latter needs MASt3R correspondences)
            correspondence: "mast3r" (estimated) or "gt" (GT poses + depth, DeformView only)
            img_size:       input resolution (the models were trained at 512)
            weights:        override the checkpoint of the selected variant
            threshold:      override the pixel-level decision threshold
        """
        if features not in FEATURES:
            raise ValueError(f"features must be one of {FEATURES}, got {features!r}")
        if correspondence not in CORRESPONDENCES:
            raise ValueError(f"correspondence must be one of {CORRESPONDENCES}, got {correspondence!r}")
        if (features, correspondence) not in VARIANTS:
            raise ValueError(f"No trained DEFECt3R for features={features!r} with "
                             f"correspondence={correspondence!r} (MASt3R features require "
                             f"MASt3R correspondences). Available: {list(VARIANTS)}")
        cfg = VARIANTS[(features, correspondence)]

        super().__init__(
            correspondence=correspondence,
            use_mast3r_features=(features == "dino+mast3r"),
            img_size=img_size,
            threshold=cfg["threshold"] if threshold is None else threshold,
        )
        self.features = features

        self.classifier = cfg["classifier"]()
        weights_path = Path(weights) if weights is not None else WEIGHTS_DIR / cfg["weights"]
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.classifier.load_state_dict(ckpt["model_state"])

        self.eval()

    def score(self, dino_diff: Tensor, mast3r_diff: Optional[Tensor]) -> Tensor:
        """Inconsistency probability per pixel (sigmoid of the MLP logit)."""
        probs = []
        for start in range(0, len(dino_diff), INFERENCE_CHUNK):
            d = dino_diff[start:start + INFERENCE_CHUNK]
            if mast3r_diff is None:
                logits = self.classifier(d)
            else:
                logits = self.classifier(d, mast3r_diff[start:start + INFERENCE_CHUNK])
            probs.append(torch.sigmoid(logits))
        return torch.cat(probs)
