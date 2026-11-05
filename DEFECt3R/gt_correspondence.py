"""
Ground-truth correspondences from GT camera poses + depth maps, adapted from
MEt3RGT (met3r_gt.py).

For every pixel in view 1, unproject with depth_1 / pose_1 and reproject into
view 2 (and vice versa).  A pixel is in the overlap when both directions are
valid and unoccluded.  View-2 features are then sampled at the projected
locations, which aligns them with view 1.

Only applicable when GT poses and depth maps are available (DeformView).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


# ─────────────────────────────────────────────────────────────────────────────
# Intrinsics helper
# ─────────────────────────────────────────────────────────────────────────────

def build_intrinsics(H: int, W: int, yfov_deg: float = 45.0,
                     device=None) -> Tensor:
    """
    Build (3, 3) intrinsics matrix K from image size and vertical FOV.
    Assumes square pixels.  Matches render_views_room.py (yfov=45°).
    """
    fy = (H / 2.0) / math.tan(math.radians(yfov_deg / 2.0))
    # pyrender scales fx by aspect ratio: fx = fy * (W/H) for standard cameras
    fx = fy * (W / H)
    cx, cy = W / 2.0, H / 2.0
    K = torch.zeros(3, 3, dtype=torch.float32, device=device)
    K[0, 0] = fx;  K[0, 2] = cx
    K[1, 1] = fy;  K[1, 2] = cy
    K[2, 2] = 1.0
    return K


# ─────────────────────────────────────────────────────────────────────────────
# GT-based warping
# ─────────────────────────────────────────────────────────────────────────────

def compute_warp_grid(
    depth_src: Tensor,
    K: Tensor,
    pose_src: Tensor,
    pose_dst: Tensor,
    depth_dst_map: Optional[Tensor] = None,
    occlusion_tol: float = 0.05,
    occlusion_abs_tol: float = 0.02,
) -> Tuple[Tensor, Tensor]:
    """
    For every pixel in the SOURCE view, unproject to world using depth_src and
    pose_src, then project into the DESTINATION view using pose_dst.

    Returns a sampling grid (for use with F.grid_sample) and a validity mask.

    Args:
        depth_src     : (B, H, W)  depth from pyrender — camera-axis distance,
                                   positive, 0 = background
        K             : (3, 3)     shared intrinsics
        pose_src      : (B, 4, 4)  camera-to-world for source view
        pose_dst      : (B, 4, 4)  camera-to-world for destination view
        depth_dst_map : (B, H, W)  actual depth map of the destination view.
                                   When provided, an occlusion check is run:
                                   pixels where the destination view sees a
                                   closer surface (i.e. the source point is
                                   occluded) are masked out.  This produces
                                   real holes in the overlap mask wherever
                                   the two views see different geometry.
        occlusion_tol : relative tolerance for the depth consistency check
                        (default 5%).  Increase to be more permissive.

    Returns:
        grid  (B, H, W, 2)  sampling coords in [-1, 1] for F.grid_sample
        mask  (B, H, W)     True = valid, unoccluded correspondence
    """
    B, H, W = depth_src.shape
    device   = depth_src.device
    dtype    = depth_src.dtype

    K = K.to(device=device, dtype=dtype)

    fx = K[0, 0]; fy = K[1, 1]
    cx = K[0, 2]; cy = K[1, 2]

    # Pixel grid
    v_grid, u_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )  # both (H, W)

    d = depth_src  # (B, H, W)

    # ── Unproject depth_src pixels to camera-space (source) ─────────────────
    # pyrender: camera looks along –Z, Y-up in camera space (OpenGL convention).
    # The depth array returned by pyrender is already vertically flipped to
    # standard image convention (row 0 = top), so:
    #   • u increases right  →  X_c = (u - cx) / fx * depth  (no sign change)
    #   • v increases DOWN   →  a point at positive Y_c (UP) has SMALL v
    #                            Y_c = -(v - cy) / fy * depth
    X_c = ( u_grid[None] - cx) / fx * d          # (B, H, W)
    Y_c = -(v_grid[None] - cy) / fy * d          # (B, H, W)  ← Y flipped
    Z_c = -d                                      # (B, H, W), negative = in front

    # Homogeneous: (B, 4, H*W)
    ones = torch.ones_like(d)
    pts_src = torch.stack([X_c, Y_c, Z_c, ones], dim=1).reshape(B, 4, H * W)

    # ── World coords via pose_src (camera-to-world) ──────────────────────────
    pts_w = torch.bmm(pose_src, pts_src)  # (B, 4, HW)

    # ── Camera coords in destination via pose_dst^{-1} ───────────────────────
    pose_dst_inv = torch.linalg.inv(pose_dst)
    pts_dst      = torch.bmm(pose_dst_inv, pts_w)            # (B, 4, HW)
    X2, Y2, Z2   = pts_dst[:, 0], pts_dst[:, 1], pts_dst[:, 2]

    # Depth in destination view: depth_dst = -Z2  (camera looks along –Z)
    depth_dst = -Z2  # (B, HW)

    # ── Project into destination image ───────────────────────────────────────
    safe_d = depth_dst.clamp(min=1e-6)
    u2 =  fx * X2 / safe_d + cx   # (B, HW)
    v2 = -fy * Y2 / safe_d + cy   # (B, HW)  ← Y flipped (Y-up cam, Y-down image)

    # Normalise to [-1, 1] for F.grid_sample (grid_sample expects x=col, y=row)
    u2_n = u2 / (W - 1) * 2 - 1   # (B, HW)
    v2_n = v2 / (H - 1) * 2 - 1   # (B, HW)

    grid = torch.stack([u2_n, v2_n], dim=-1).reshape(B, H, W, 2)

    # ── Base validity mask ────────────────────────────────────────────────────
    in_image = (
        (u2_n >= -1) & (u2_n <= 1) &
        (v2_n >= -1) & (v2_n <= 1)
    )
    valid_depth_src = (d > 0).reshape(B, H * W)
    valid_depth_dst = (depth_dst > 0)
    mask = (valid_depth_src & valid_depth_dst & in_image).reshape(B, H, W)

    # ── Occlusion check ───────────────────────────────────────────────────────
    # Compare reprojected depth with the actual depth of the destination view.
    # If the destination sees a closer surface, the source point is occluded.
    if depth_dst_map is not None:
        # Sample the destination depth map at the projected pixel locations.
        # Reshape projected coords to (B, 1, HW, 2) for grid_sample.
        sample_grid = torch.stack([u2_n, v2_n], dim=-1).reshape(B, 1, H * W, 2)
        depth_actual = F.grid_sample(
            depth_dst_map.to(dtype=dtype).unsqueeze(1),   # (B, 1, H, W)
            sample_grid,
            mode="nearest",        # nearest-neighbour keeps depth crisp
            padding_mode="zeros",  # out-of-bounds → 0 (already caught by in_image)
            align_corners=True,
        ).reshape(B, H * W)   # (B, HW)

        # A point is NOT occluded when the actual destination depth is CLOSE to
        # the expected reprojected depth — in BOTH directions:
        #   • depth_actual << depth_dst → something closer blocks our point (classic occlusion)
        #   • depth_actual >> depth_dst → destination sees different geometry entirely
        #     (e.g. the object back-face projects to a wall pixel in the other view)
        # Both cases mean there is no valid correspondence.
        not_occluded = (
            (depth_actual > 0) &
            (depth_actual >= depth_dst * (1.0 - occlusion_tol) - occlusion_abs_tol) &
            (depth_actual <= depth_dst * (1.0 + occlusion_tol) + occlusion_abs_tol)
        )
        mask = mask & not_occluded.reshape(B, H, W)

    return grid, mask


# ─────────────────────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────────────────────

class GTCorrespondence(Module):
    """
    Aligns view-2 features to view 1 using GT poses and depth maps
    (the warping part of MEt3RGT.forward).

    Args:
        img_size          : image size; intrinsics are precomputed for it.
        yfov_deg          : vertical FOV of the renderer (DeformView: 45°).
        occlusion_tol     : relative depth tolerance of the occlusion check.
        occlusion_abs_tol : absolute depth tolerance of the occlusion check.
    """

    def __init__(
        self,
        img_size: int = 512,
        yfov_deg: float = 45.0,
        occlusion_tol: float = 0.05,
        occlusion_abs_tol: float = 0.02,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.yfov_deg = yfov_deg
        self.occlusion_tol = occlusion_tol
        self.occlusion_abs_tol = occlusion_abs_tol
        self.register_buffer("_K", build_intrinsics(img_size, img_size, yfov_deg), persistent=False)

    def forward(self, features: Tensor, poses: Tensor, depths: Tensor) -> dict:
        """
        Args:
            features : (B, 2, C, H, W) per-pixel features of both views
            poses    : (B, 2, 4, 4)    camera-to-world matrices
            depths   : (B, 2, H, W)    depth maps (0 = background)

        Returns dict:
            overlap_mask : (B, H, W)        bilateral, unoccluded overlap (bool)
            rendering    : (B, 2, C, H, W)  view-1 features and view-2 features
                                            warped into view 1 (0 outside overlap)
        """
        b, k, C, h, w = features.shape
        device = features.device

        # Build intrinsics for this resolution
        K = (
            self._K.to(device=device, dtype=features.dtype)
            if (h == w == self.img_size)
            else build_intrinsics(h, w, self.yfov_deg, device=device).to(dtype=features.dtype)
        )

        feat1 = features[:, 0]   # (B, C_feat, H, W)
        feat2 = features[:, 1]   # (B, C_feat, H, W)

        # ── GT-pose warping: project view-2 features into view-1 frame ────────
        depth1 = depths[:, 0]   # (B, H, W)
        depth2 = depths[:, 1]   # (B, H, W)
        pose1  = poses[:, 0]    # (B, 4, 4)
        pose2  = poses[:, 1]    # (B, 4, 4)

        # Warp feat2 → view1 frame  (occlusion check: view2's depth map)
        grid_2to1, mask_2to1 = compute_warp_grid(depth1, K, pose1, pose2,
                                                  depth_dst_map=depth2,
                                                  occlusion_tol=self.occlusion_tol,
                                                  occlusion_abs_tol=self.occlusion_abs_tol)
        feat2_in_view1 = F.grid_sample(
            feat2, grid_2to1,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )  # (B, C_feat, H, W); zeros where invalid

        # Validity of the reverse direction (view2 → view1)
        _, mask_1to2 = compute_warp_grid(depth2, K, pose2, pose1,
                                         depth_dst_map=depth1,
                                         occlusion_tol=self.occlusion_tol,
                                         occlusion_abs_tol=self.occlusion_abs_tol)

        # ── Overlap mask: pixels visible in both views ────────────────────────
        # mask_2to1: pixels in view1 whose view2 correspondence is valid
        # Warp mask_1to2 back into view1 frame to find bilateral overlap
        mask_1to2_f = mask_1to2.float().unsqueeze(1)   # (B, 1, H, W)
        mask_1to2_in_view1 = F.grid_sample(
            mask_1to2_f, grid_2to1,
            mode="nearest", padding_mode="zeros", align_corners=True,
        ).squeeze(1).bool()  # (B, H, W)

        overlap_mask = mask_2to1 & mask_1to2_in_view1  # (B, H, W)

        # ── Aligned feature pair (both in view-1 frame) ───────────────────────
        rendering = torch.stack([feat1, feat2_in_view1], dim=1)  # (B, 2, C, H, W)

        # Zero out non-overlapping regions (matches MEt3R convention)
        invalid = ~overlap_mask.unsqueeze(1)  # (B, 1, H, W) → broadcasts over C
        rendering[:, 0].masked_fill_(invalid, 0.0)
        rendering[:, 1].masked_fill_(invalid, 0.0)

        return {"overlap_mask": overlap_mask, "rendering": rendering}
