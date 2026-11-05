"""
MASt3R correspondences, adapted from MEt3R (`MEt3R.forward`, backbone="mast3r").

MASt3R predicts a point map for both views in the frame of view 1.  A focal
length is estimated from the confidence-weighted canonical point map, and both
point clouds are rendered into view 1 with PyTorch3D.  This gives the overlap
mask, the per-pixel fragment coordinates and the MASt3R descriptors (24-d) that
DEFECt3R uses.

Requires the MASt3R repository as a git submodule in third_party/mast3r
(git submodule update --init --recursive).
"""

import sys
import os.path as path
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module
from jaxtyping import Float
from einops import rearrange, repeat

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
)

from .utils import freeze_model, convert_to_buffer

HERE_PATH = path.normpath(path.dirname(__file__))
MASt3R_REPO_PATH = path.normpath(path.join(HERE_PATH, 'third_party/mast3r'))
DUSt3R_REPO_PATH = path.normpath(path.join(HERE_PATH, 'third_party/mast3r/dust3r'))
MASt3R_LIB_PATH = path.join(MASt3R_REPO_PATH, 'mast3r')
DUSt3R_LIB_PATH = path.join(DUSt3R_REPO_PATH, 'dust3r')
# check the presence of models directory in repo to be sure its cloned
if path.isdir(MASt3R_LIB_PATH) and path.isdir(DUSt3R_LIB_PATH):
    # workaround for sibling import
    sys.path.insert(0, MASt3R_REPO_PATH)
    sys.path.insert(0, DUSt3R_REPO_PATH)
else:
    raise ImportError(f"mast3r and dust3r is not initialized, could not find: {MASt3R_LIB_PATH}.\n "
                      "Did you forget to run 'git submodule update --init --recursive' ?")
from dust3r.utils.geometry import xy_grid

MASt3R_WEIGHTS = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"


class MASt3R(Module):

    def __init__(self, img_size: int = 512, weights: str = MASt3R_WEIGHTS, freeze: bool = True) -> None:
        super().__init__()
        from mast3r.model import AsymmetricMASt3R
        self.backbone_model = AsymmetricMASt3R.from_pretrained(weights)
        if freeze:
            freeze_model(self.backbone_model)
            convert_to_buffer(self.backbone_model, persistent=False)

        self.set_rasterizer(image_size=img_size, points_per_pixel=10, bin_size=0)
        self.compositor = AlphaCompositor()

    def set_rasterizer(self, image_size, points_per_pixel=10, bin_size=0, **kwargs) -> None:
        raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            points_per_pixel=points_per_pixel,
            bin_size=bin_size,
            **kwargs
        )
        self.rasterizer = PointsRasterizer(cameras=None, raster_settings=raster_settings)

    def render(
        self,
        point_clouds: Pointclouds,
        **kwargs
    ) -> Tuple[
            Float[Tensor, "b h w c"],
            Float[Tensor, "b 2 h w n"]
    ]:
        """Adoped from Pytorch3D https://pytorch3d.readthedocs.io/en/latest/modules/renderer/points/renderer.html

        Args:
            point_clouds (pytorch3d.structures.PointCloud): Point cloud object to render

        Returns:
            images (Float[Tensor, "b h w c"]): Rendered images
            zbuf (Float[Tensor, "b k h w n"]): Z-buffers for points per pixel
        """
        with torch.autocast("cuda", enabled=False):
            fragments = self.rasterizer(point_clouds, **kwargs)

        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images, fragments.zbuf, fragments.idx[..., 0]

    def forward(self, images: Tensor, features: Tensor) -> dict:
        """
        Args:
            images:   (B, 2, 3, H, W) image pairs in [-1, 1]
            features: (B, 2, C, H, W) per-pixel features that are unprojected on the
                      point clouds and rendered into view 1

        Returns dict:
            overlap_mask: (B, H, W)        1 where both renderings are valid
            row_coords:   (B, 2, H, W)     source row of the point rendered at each pixel
            col_coords:   (B, 2, H, W)     source column of the point rendered at each pixel
            descriptors:  (B, 2, 24, H, W) MASt3R descriptors (in their own view)
            confidence:   (B, 2, H, W)     MASt3R confidences
            rendering:    (B, 2, C, H, W)  `features` of both views rendered into view 1
        """
        b, k, _, h, w = images.shape

        view1 = {"img": images[:, 0, ...], "instance": [""]}
        view2 = {"img": images[:, 1, ...], "instance": [""]}
        pred1, pred2 = self.backbone_model(view1, view2)

        mast3r_features = torch.stack([
            rearrange(pred1["desc"], "b h w c -> b c h w"),
            rearrange(pred2["desc"], "b h w c -> b c h w"),
        ], dim=1)
        mast3r_confidence = torch.stack([pred1["conf"], pred2["conf"]], dim=1).detach()

        ptmps = torch.stack([pred1["pts3d"], pred2["pts3d_in_other_view"]], dim=1).detach()
        conf = torch.stack([pred1["conf"], pred2["conf"]], dim=1).detach()

        # NOTE: Get canonical point map using the confidences
        confs11 = conf.unsqueeze(-1) - 0.999
        canon = (confs11 * ptmps).sum(1) / confs11.sum(1)

        # Define principal point
        pp = torch.tensor([w / 2, h / 2], device=canon.device)

        # NOTE: Estimating fx and fy for a given canonical point map
        B, H, W, THREE = canon.shape
        assert THREE == 3

        # centered pixel grid
        pixels = xy_grid(W, H, device=canon.device).view(1, -1, 2) - pp.view(-1, 1, 2)  # B,HW,2
        canon = canon.flatten(1, 2)  # (B, HW, 3)

        # direct estimation of focal
        u, v = pixels.unbind(dim=-1)
        x, y, z = canon.unbind(dim=-1)
        fx_votes = (u * z) / x
        fy_votes = (v * z) / y

        # assume square pixels, hence same focal for X and Y
        f_votes = torch.stack((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
        focal = torch.nanmedian(f_votes, dim=-2)[0]

        # Normalized focal length
        focal[..., 0] = 1 + focal[..., 0] / w
        focal[..., 1] = 1 + focal[..., 1] / h
        focal = repeat(focal, "b c -> (b k) c", k=2)
        # NOTE: Unproject feature on the point cloud
        ptmps = rearrange(ptmps, "b k h w c -> (b k) (h w) c", b=b, k=2)
        features = rearrange(features, "b k c h w -> (b k) (h w) c", k=2)
        point_cloud = Pointclouds(points=ptmps, features=features)

        # NOTE: Project and Render
        R = torch.eye(3)
        R[0, 0] *= -1
        R[1, 1] *= -1
        R = repeat(R, "... -> (b k) ...", b=b, k=2)
        T = torch.zeros((3, ))
        T = repeat(T, "... -> (b k) ...", b=b, k=2)

        # Define Pytorch3D camera for projection
        cameras = PerspectiveCameras(device=ptmps.device, R=R, T=T, focal_length=focal)
        # Render via point rasterizer to get projected features
        with torch.autocast("cuda", enabled=False):
            rendering, zbuf, frag_idx = self.render(point_cloud, cameras=cameras, background_color=[-10000] * features.shape[-1])
            frag_idx = rearrange(frag_idx, "(b k) h w -> b k h w", b=b, k=2)
            row_coords = torch.div(frag_idx, w, rounding_mode="floor").clamp(min=0)
            col_coords = (frag_idx % w).clamp(min=0)
        rendering = rearrange(rendering, "(b k) h w c -> b k c h w", b=b, k=2)

        # Compute overlapping mask
        non_overlap_mask = (rendering == -10000)
        overlap_mask = (1 - non_overlap_mask.float()).prod(2).prod(1)

        # Zero out regions which do not overlap
        rendering[non_overlap_mask] = 0.0

        return {
            "overlap_mask": overlap_mask,
            "row_coords": row_coords,
            "col_coords": col_coords,
            "descriptors": mast3r_features,
            "confidence": mast3r_confidence,
            "rendering": rendering,
        }
