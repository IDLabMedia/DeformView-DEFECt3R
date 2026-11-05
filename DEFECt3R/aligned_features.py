"""
Shared pipeline of DEFECt3R and MEt3R: DINO + FeatUp features of both views,
pixel correspondences (MASt3R or GT), and the L2-normalised feature
differences of every overlap pixel in the frame of view 1.

Subclasses implement `score(dino_diff, mast3r_diff)`, which maps the feature
differences of the overlap pixels to an inconsistency score per pixel.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn import Module

from .dino import DINO
from .featup import FeatUp

CORRESPONDENCES = ("mast3r", "gt")


class AlignedFeatureModel(Module):

    def __init__(self, correspondence: str = "mast3r", use_mast3r_features: bool = False,
                 img_size: int = 512, threshold: Optional[float] = None) -> None:
        super().__init__()
        if correspondence not in CORRESPONDENCES:
            raise ValueError(f"correspondence must be one of {CORRESPONDENCES}, got {correspondence!r}")
        if use_mast3r_features and correspondence != "mast3r":
            raise ValueError("MASt3R features require MASt3R correspondences")

        self.correspondence_type = correspondence
        self.use_mast3r_features = use_mast3r_features
        self.img_size = img_size
        self.threshold = threshold

        self.dino = DINO()
        self.featup = FeatUp()

        if correspondence == "mast3r":
            from .mast3r import MASt3R
            self.correspondence = MASt3R(img_size=img_size)
        else:
            from .gt_correspondence import GTCorrespondence
            self.correspondence = GTCorrespondence(img_size=img_size)

    def score(self, dino_diff: Tensor, mast3r_diff: Optional[Tensor]) -> Tensor:
        """(N, 384) DINO diffs [+ (N, 24) MASt3R diffs] → (N,) inconsistency scores."""
        raise NotImplementedError

    @torch.inference_mode()
    def forward(
        self,
        img1: Tensor,
        img2: Tensor,
        depth1: Optional[Tensor] = None,
        depth2: Optional[Tensor] = None,
        pose1: Optional[Tensor] = None,
        pose2: Optional[Tensor] = None,
    ) -> dict:
        """
        Args:
            img1, img2:     (3, H, W) or (B, 3, H, W) images in [-1, 1]
            depth1, depth2: (H, W) or (B, H, W) depth maps   — only for correspondence="gt"
            pose1, pose2:   (4, 4) or (B, 4, 4) camera-to-world — only for correspondence="gt"

        Returns dict (in the frame of view 1, without batch dim for unbatched input):
            heatmap:      (B, H, W) float  inconsistency score, NaN where no score
            prediction:   (B, H, W) bool   heatmap > threshold (only if a threshold is set)
            multiplicity: (B, H, W) int    number of overlap pixels that produced the
                                           score at each location
        """
        unbatched = img1.dim() == 3
        if unbatched:
            img1, img2 = img1[None], img2[None]
            if depth1 is not None:
                depth1, depth2, pose1, pose2 = depth1[None], depth2[None], pose1[None], pose2[None]

        images = torch.stack([img1, img2], dim=1)   # (B, 2, 3, H, W)
        b, _, _, h, w = images.shape
        device = images.device

        # ── DINO + FeatUp ─────────────────────────────────────────────────────
        images01 = rearrange((images + 1) / 2, "b k c h w -> (b k) c h w")
        hr_feat = self.featup(self.dino(images01), images01)
        hr_feat = rearrange(hr_feat, "(b k) c h w -> b k c h w", k=2)

        heatmap = torch.full((b, h, w), float("nan"), device=device)
        multiplicity = torch.zeros((b, h, w), dtype=torch.int32, device=device)

        # ── Correspondences + per-pixel feature differences ──────────────────
        if self.correspondence_type == "mast3r":
            corr = self.correspondence(images, hr_feat)
            for i in range(b):
                overlap_mask = corr["overlap_mask"][i].bool()
                vr = corr["row_coords"][i, 0][overlap_mask]
                vc = corr["col_coords"][i, 0][overlap_mask]

                dino_f1 = hr_feat[i, 0].float()   # (C, H_r, W_r)
                dino_f2 = hr_feat[i, 1].float()
                H_r, W_r = dino_f1.shape[-2], dino_f1.shape[-1]
                dino_f1 = dino_f1.permute(1, 2, 0)
                dino_f2 = dino_f2.permute(1, 2, 0)
                dino_diff = F.normalize(dino_f1[vr, vc], dim=-1) - F.normalize(dino_f2[vr, vc], dim=-1)

                mast3r_diff = None
                if self.use_mast3r_features:
                    desc = corr["descriptors"][i].float()   # (2, 24, H_m, W_m)
                    m1 = F.interpolate(desc[0].unsqueeze(0), size=(H_r, W_r), mode="bilinear",
                                       align_corners=False).squeeze(0).permute(1, 2, 0)
                    m2 = F.interpolate(desc[1].unsqueeze(0), size=(H_r, W_r), mode="bilinear",
                                       align_corners=False).squeeze(0).permute(1, 2, 0)
                    mast3r_diff = F.normalize(m1[vr, vc], dim=-1) - F.normalize(m2[vr, vc], dim=-1)

                heatmap[i, vr, vc] = self.score(dino_diff, mast3r_diff)
                multiplicity[i].index_put_((vr, vc), torch.ones_like(vr, dtype=torch.int32),
                                           accumulate=True)
        else:
            if depth1 is None or depth2 is None or pose1 is None or pose2 is None:
                raise ValueError("correspondence='gt' requires depth1, depth2, pose1 and pose2")
            poses = torch.stack([pose1, pose2], dim=1).to(device=device, dtype=torch.float32)
            depths = torch.stack([depth1, depth2], dim=1).to(device=device, dtype=torch.float32)
            corr = self.correspondence(hr_feat, poses, depths)
            for i in range(b):
                vr, vc = torch.where(corr["overlap_mask"][i])
                feat1 = corr["rendering"][i, 0].float().permute(1, 2, 0)   # (H, W, C)
                feat2 = corr["rendering"][i, 1].float().permute(1, 2, 0)   # view 2 warped to view 1
                dino_diff = F.normalize(feat1[vr, vc], dim=-1) - F.normalize(feat2[vr, vc], dim=-1)
                heatmap[i, vr, vc] = self.score(dino_diff, None)
                multiplicity[i, vr, vc] = 1

        out = {"heatmap": heatmap, "multiplicity": multiplicity}
        if self.threshold is not None:
            out["prediction"] = heatmap > self.threshold
        if unbatched:
            out = {k: v[0] for k, v in out.items()}
        return out
