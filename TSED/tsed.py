"""
TSED: Thresholded Symmetric Epipolar Distance (Yu et al., ICCV 2023), pair-level only.

For an image pair:
  1. Detect SIFT keypoints and match them (Lowe's ratio test).
  2. Build the fundamental matrix F from the camera poses: GT poses (DeformView)
     or poses estimated by MASt3R (DUSt3R PairViewer).
  3. Compute the Symmetric Epipolar Distance (SED) per match.
  4. Pair score = median SED (inf when fewer than MIN_MATCHES matches).
     A pair is inconsistent when score >= T_error.

Usage:
    model = TSED(correspondence="mast3r", device="cuda")
    out = model("view1.png", "view2.png")                      # MASt3R poses
    out = TSED(correspondence="gt")(p1, p2, pose1=P1, pose2=P2)  # GT poses (OpenGL cam-to-world)
    out["score"], out["n_matches"]

Requires OpenCV (opencv-python-headless<5); MASt3R poses use the MASt3R submodule of DEFECt3R.
"""

import contextlib
import io
import math
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError("OpenCV could not be imported. Install with: "
                      "pip install \"opencv-python-headless<5\"") from e

CORRESPONDENCES = ("mast3r", "gt")

MIN_MATCHES = 10      # minimum SIFT matches; fewer → score = inf
SIFT_RATIO  = 0.75    # Lowe's ratio test threshold

# T_error selected on the validation split (F1 of the inconsistent class).
THRESHOLDS = {
    "mast3r": 0.104, # based on DeformView validation split
    "gt":     0.112, # based on DeformView validation split
}

MASt3R_WEIGHTS = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"


# ─────────────────────────────────────────────────────────────────────────────
# Intrinsics & fundamental matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_intrinsics(yfov_deg, img_size):
    """Standard pinhole K from vertical FOV and square image size."""
    yfov = np.deg2rad(yfov_deg)
    f    = img_size / (2.0 * np.tan(yfov / 2.0))
    c    = img_size / 2.0
    return np.array([[f, 0, c], [0, f, c], [0, 0, 1]], dtype=np.float64)


def build_intrinsics_from_focal(focal_px, img_size):
    """Standard pinhole K (principal point at image centre)."""
    c = img_size / 2.0
    return np.array([
        [focal_px, 0,        c],
        [0,        focal_px, c],
        [0,        0,        1],
    ], dtype=np.float64)


def skew(t):
    """Skew-symmetric matrix:  skew(t) @ v  ==  t × v"""
    return np.array([
        [ 0,    -t[2],  t[1]],
        [ t[2],  0,    -t[0]],
        [-t[1],  t[0],  0   ],
    ], dtype=np.float64)


_D = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL → OpenCV axis flip (Y and Z)


def build_fundamental_matrix(cam_pose1, cam_pose2, K, opengl: bool):
    """
    Fundamental matrix F (pixel coords) from two cam-to-world poses.

    GT poses are in OpenGL convention (camera looks along −Z, Y-up) and are
    converted to OpenCV convention:  pose_cv = pose_gl @ diag(1, −1, −1, 1).
    MASt3R / DUSt3R poses are already in OpenCV convention (+Z, Y-down).

        T_rel = inv(pose2) @ pose1
        E     = skew(t) @ R
        F     = K^{−T} @ E @ K^{−1}

    Epipolar constraint:  p2^T @ F @ p1 ≈ 0  for pixel homogeneous coords.
    """
    if opengl:
        cam_pose1 = cam_pose1 @ _D
        cam_pose2 = cam_pose2 @ _D
    T_rel  = np.linalg.inv(cam_pose2) @ cam_pose1
    R      = T_rel[:3, :3]
    t      = T_rel[:3, 3]
    E      = skew(t) @ R
    Ki     = np.linalg.inv(K)
    F      = Ki.T @ E @ Ki
    return F


def symmetric_epipolar_distance(p1_px, p2_px, F):
    """
    Symmetric Epipolar Distance (pixels) for one matched pair.

    SED = 0.5 · (d(p2, F @ p1) + d(p1, F^T @ p2))
    where d(p, l) = |p^T l| / ‖l[:2]‖  is point-to-line distance.
    """
    p1 = np.array([p1_px[0], p1_px[1], 1.0])
    p2 = np.array([p2_px[0], p2_px[1], 1.0])

    l2 = F @ p1
    d2 = abs(p2 @ l2) / (np.linalg.norm(l2[:2]) + 1e-10)

    l1 = F.T @ p2
    d1 = abs(p1 @ l1) / (np.linalg.norm(l1[:2]) + 1e-10)

    return 0.5 * (d1 + d2)


# ─────────────────────────────────────────────────────────────────────────────
# SIFT matching
# ─────────────────────────────────────────────────────────────────────────────

def get_sift_matches(img1_path, img2_path):
    img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)

    if img1 is None:
        raise FileNotFoundError(f"Cannot read: {img1_path}")
    if img2 is None:
        raise FileNotFoundError(f"Cannot read: {img2_path}")

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return [], kp1 or [], kp2 or []

    bf   = cv2.BFMatcher()
    raw  = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < SIFT_RATIO * n.distance]
    return good, kp1, kp2


# ─────────────────────────────────────────────────────────────────────────────
# MASt3R pose estimation (DUSt3R PairViewer)
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _quiet():
    """Suppress Python-level stdout/stderr (e.g. DUSt3R confidence prints)."""
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        yield


class MASt3RPoses:
    """Relative camera poses + focal lengths of an image pair, estimated by MASt3R."""

    def __init__(self, img_size: int = 512, device: str = "cuda"):
        import DEFECt3R.mast3r  # noqa: F401  (puts the MASt3R submodule on sys.path)
        from mast3r.model import AsymmetricMASt3R
        self.img_size = img_size
        self.device = device
        self.model = AsymmetricMASt3R.from_pretrained(MASt3R_WEIGHTS).to(device).eval()

    def __call__(self, img1_path, img2_path):
        """
        Returns:
            cam_pose1, cam_pose2 : (4, 4) camera-to-world (anchored to image 1)
            focal1, focal2       : focal lengths in pixels at img_size
        """
        import torch
        from dust3r.utils.image import load_images
        from dust3r.inference import inference
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

        with torch.no_grad():
            images = load_images([str(img1_path), str(img2_path)], size=self.img_size, verbose=False)
            pairs = [(images[0], images[1]), (images[1], images[0])]
            output = inference(pairs, self.model, self.device, batch_size=1, verbose=False)

            # PairViewer: closed-form pose recovery for a single image pair
            with _quiet():
                scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PairViewer)

            poses  = scene.get_im_poses().cpu().numpy()   # (2, 4, 4)
            focals = scene.get_focals().cpu().numpy()     # (2,)

        return poses[0].astype(np.float64), poses[1].astype(np.float64), \
               float(focals[0]), float(focals[1])


# ─────────────────────────────────────────────────────────────────────────────
# TSED
# ─────────────────────────────────────────────────────────────────────────────

class TSED:

    def __init__(self, correspondence: str = "mast3r", img_size: int = 512, yfov_deg: float = 45.0,
                 threshold: Optional[float] = None, device: str = "cuda"):
        """
        Args:
            correspondence: "mast3r" (poses estimated by MASt3R) or "gt" (GT poses, DeformView only)
            img_size:       image size (pixels)
            yfov_deg:       vertical FOV for the GT intrinsics (DeformView: 45°)
            threshold:      override T_error
            device:         device for MASt3R
        """
        if correspondence not in CORRESPONDENCES:
            raise ValueError(f"correspondence must be one of {CORRESPONDENCES}, got {correspondence!r}")
        self.correspondence_type = correspondence
        self.img_size = img_size
        self.threshold = THRESHOLDS[correspondence] if threshold is None else threshold
        if correspondence == "mast3r":
            self.pose_estimator = MASt3RPoses(img_size=img_size, device=device)
        else:
            self.K = build_intrinsics(yfov_deg, img_size)

    def __call__(self, img1_path, img2_path, pose1: Optional[np.ndarray] = None,
                 pose2: Optional[np.ndarray] = None) -> dict:
        """
        Args:
            img1_path, img2_path: image files
            pose1, pose2:         (4, 4) camera-to-world, OpenGL — only for correspondence="gt"

        Returns dict:
            score       median SED (pixels); inf if fewer than MIN_MATCHES matches
            median_sed  median SED over all matches (inf without matches)
            n_matches   number of SIFT matches
            note        "" or "missing_pose" (pose estimation failed; pair not scoreable)
            prediction  score >= threshold (inconsistent), None if not scoreable
        """
        if self.correspondence_type == "gt":
            if pose1 is None or pose2 is None:
                raise ValueError("correspondence='gt' requires pose1 and pose2")
            K, opengl = self.K, True
        else:
            try:
                pose1, pose2, focal1, focal2 = self.pose_estimator(img1_path, img2_path)
            except Exception:
                return dict(score=math.inf, median_sed=math.inf, n_matches=0,
                            note="missing_pose", prediction=None)
            K, opengl = build_intrinsics_from_focal(0.5 * (focal1 + focal2), self.img_size), False

        matches, kp1, kp2 = get_sift_matches(img1_path, img2_path)
        if len(matches) == 0:
            median_sed = math.inf
        else:
            F = build_fundamental_matrix(pose1, pose2, K, opengl=opengl)
            seds = [
                symmetric_epipolar_distance(kp1[m.queryIdx].pt, kp2[m.trainIdx].pt, F)
                for m in matches
            ]
            median_sed = float(np.median(seds))

        n_matches = len(matches)
        score = median_sed if n_matches >= MIN_MATCHES else math.inf
        return dict(score=score, median_sed=median_sed, n_matches=n_matches, note="",
                    prediction=score >= self.threshold)
