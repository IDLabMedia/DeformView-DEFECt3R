"""
dataset.py
──────────
Loading of DeformView pairs: images, deformation masks, foreground masks, and
(for GT correspondences) depth maps + camera poses.

Pair files (splits/*.csv) store paths relative to the data root:

    <data_root>/DeformView_renders/<object>/
        camera_params.npz                       camera ring (shared by both meshes)
        original/  {angle}.png                  render of the original mesh
                   {angle}_fg.png               foreground (object) mask
                   {angle}_depth.npy            depth map
        deformed/  {angle}.png                  render of the deformed mesh
                   {angle}_mask.png             deformation mask (ground truth)
                   {angle}_depth.npy            depth map
"""

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from .camera_pose import camera_pose_from_file


RENDERS_DIR = "DeformView_renders"

IMG_SIZE = 512
SPLITS_DIR = Path(__file__).resolve().parent / "splits"        # pair list of the MASt3R runs
SPLITS_GT_DIR = Path(__file__).resolve().parent / "split_GT"   # pair list of the GT runs


def read_pairs(pair_file) -> list[dict]:
    """Read a pair file → list of dicts (pair_idx, img1, img2, mask, pair_type)."""
    with open(pair_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["pair_idx"] = int(r["pair_idx"])
        r["mask"] = r["mask"] or None
    return rows


class DeformView:

    def __init__(self, root, img_size: int = IMG_SIZE):
        self.root = Path(root)
        self.img_size = img_size
        self._preprocess = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2 - 1),
        ])

    # ── Images / masks ────────────────────────────────────────────────────────

    def load_image(self, rel_path: str) -> torch.Tensor:
        """(3, H, W) tensor in [-1, 1]."""
        return self._preprocess(Image.open(self.root / rel_path).convert("RGB"))

    def load_deform_mask(self, rel_path) -> np.ndarray:
        """Red deformation mask → bool (H, W). Zeros if there is no mask."""
        size = self.img_size
        if rel_path is None or not (self.root / rel_path).exists():
            return np.zeros((size, size), dtype=bool)
        arr = np.array(Image.open(self.root / rel_path).convert("RGB").resize((size, size), Image.NEAREST))
        return (arr[:, :, 0] > 128) & (arr[:, :, 1] < 64) & (arr[:, :, 2] < 64)

    def load_fg_mask(self, img_rel_path: str) -> np.ndarray:
        """Foreground mask of the view of an image → bool (H, W). Zeros if missing.
        Always the mask of the original render at that angle (also for deformed images)."""
        size = self.img_size
        p = self.root / img_rel_path
        fp = p.parent.parent / "original" / f"{p.stem}_fg.png"
        if not fp.exists():
            return np.zeros((size, size), dtype=bool)
        arr = np.array(Image.open(fp).convert("L").resize((size, size), Image.NEAREST))
        return arr > 128

    # ── GT geometry ───────────────────────────────────────────────────────────

    def load_depth(self, img_rel_path: str):
        """Depth map (H, W) float32 of the render of an image, or None if not found."""
        p = self.root / img_rel_path
        fp = p.parent / f"{p.stem}_depth.npy"
        if not fp.exists():
            return None
        d = np.load(str(fp)).astype(np.float32)
        if d.shape != (self.img_size, self.img_size):
            d = F.interpolate(
                torch.from_numpy(d)[None, None],
                size=(self.img_size, self.img_size), mode="nearest",
            )[0, 0].numpy()
        return d

    def load_pose(self, img_rel_path: str, dtype=np.float32):
        """4×4 camera-to-world pose (OpenGL convention) of an image, or None if not found.
        The deformed renders use the cameras of the original mesh."""
        p = self.root / img_rel_path
        fp = p.parent.parent / "camera_params.npz"
        if not fp.exists():
            return None
        return camera_pose_from_file(p.name, str(fp))["pose_matrix"].astype(dtype)
