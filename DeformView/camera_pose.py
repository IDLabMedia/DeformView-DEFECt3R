#!/usr/bin/env python3
"""
camera_pose.py
──────────────
Reconstruct the exact camera pose for any rendered view from the DeformView
pipeline, and visualise the full camera ring.

Usage
─────
# Print pose for a single file
python camera_pose.py --params /path/to/object/camera_params.npz --file 045.png

# Just generate the diagram (works without --file)
python camera_pose.py --params /path/to/object/camera_params.npz --diagram diagram.png

# Both at once
python camera_pose.py --params /path/to/object/camera_params.npz --file 045.png --diagram diagram.png
"""

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch


# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match render_views_room.py exactly
# ─────────────────────────────────────────────────────────────────────────────

DISTANCE_SCALE  = 0.8    # radius = max(DISTANCE_SCALE * obj_radius, 0.8)
ELEVATION_DEG   = 20.0   # fixed elevation above horizon
Z_ADJUST_FRAC   = 0.2    # eye_z -= Z_ADJUST_FRAC * obj_radius
STEP_DEG        = 15.0   # views at 0, 15, 30, … 345
FOV_DEG         = 45.0   # camera yfov (perspective)


# ─────────────────────────────────────────────────────────────────────────────
# Core maths
# ─────────────────────────────────────────────────────────────────────────────

def _look_at(eye: np.ndarray, target: np.ndarray,
             up: np.ndarray = np.array([0., 0., 1.])) -> np.ndarray:
    """4×4 camera-to-world pose matrix (same convention as render_views_room)."""
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-9
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-9
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up) + 1e-9

    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = true_up
    T[:3, 2] = -forward
    T[:3, 3] = eye
    return T


def camera_pose_from_degree(deg: int, target: np.ndarray,
                             obj_radius: float) -> dict:
    """
    Reconstruct the exact camera pose used when rendering '{deg:03d}.png'.

    Parameters
    ----------
    deg        : azimuth angle in degrees (= int(filename stem))
    target     : 3-D look-at point (from camera_params.npz)
    obj_radius : object radius (from camera_params.npz)

    Returns
    -------
    dict with keys:
        pose_matrix   (4,4)  camera-to-world transform (OpenGL convention)
        position      (3,)   camera centre in world coordinates
        forward       (3,)   unit vector camera is looking toward
        right         (3,)   camera right axis
        up            (3,)   camera up axis
        target        (3,)   look-at point
        azimuth_deg   float
        elevation_deg float
        orbit_radius  float  horizontal orbital radius
        fov_deg       float
    """
    elev   = math.radians(ELEVATION_DEG)
    az     = math.radians(deg)
    radius = max(DISTANCE_SCALE * obj_radius, 0.8)

    # Eye position before Z adjustment (used for look_at direction)
    eye_pre = np.array([
        target[0] + radius * math.cos(elev) * math.cos(az),
        target[1] + radius * math.cos(elev) * math.sin(az),
        target[2] + radius * math.sin(elev),
    ])

    # Build look_at matrix from eye_pre (this is what render_object does)
    pose = _look_at(eye_pre, target)

    # Apply the Z tweak: lowers the stored translation, NOT the look direction
    pose[2, 3] -= Z_ADJUST_FRAC * obj_radius

    position = pose[:3, 3]          # actual world-space camera centre
    forward  = -pose[:3, 2]         # camera looks along -Z in its own frame
    right    =  pose[:3, 0]
    up       =  pose[:3, 1]

    return {
        "pose_matrix":   pose,
        "position":      position,
        "forward":       forward,
        "right":         right,
        "up":            up,
        "target":        target.copy(),
        "azimuth_deg":   float(deg),
        "elevation_deg": ELEVATION_DEG,
        "orbit_radius":  float(radius),
        "fov_deg":       FOV_DEG,
    }


def camera_pose_from_file(filename: str, camera_params_path: str) -> dict:
    """
    Convenience wrapper: load camera_params.npz and compute the pose for
    a filename like '045.png' or '045' or just the integer 45.
    """
    params     = np.load(camera_params_path)
    target     = params["target"].astype(np.float64)
    obj_radius = float(params["radius"])

    stem = Path(filename).stem          # '045.png' → '045'
    deg  = int(stem)

    return camera_pose_from_degree(deg, target, obj_radius)


def all_camera_poses(camera_params_path: str) -> list[dict]:
    """Return poses for all 24 views (0°, 15°, …, 345°)."""
    params     = np.load(camera_params_path)
    target     = params["target"].astype(np.float64)
    obj_radius = float(params["radius"])
    n          = int(round(360.0 / STEP_DEG))
    return [
        camera_pose_from_degree(int(i * STEP_DEG), target, obj_radius)
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise_camera_ring(camera_params_path: str, output_path: str,
                          highlight_deg: int | None = None) -> None:
    """
    Save a two-panel diagram showing all camera positions:
      Left  — top-down view (XY plane)
      Right — side view (XZ plane, looking along Y)
    """
    poses  = all_camera_poses(camera_params_path)
    target = poses[0]["target"]

    positions = np.array([p["position"] for p in poses])   # (24, 3)
    azimuths  = [p["azimuth_deg"] for p in poses]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("DeformView Camera Ring", fontsize=14, fontweight="bold")

    colours = plt.cm.hsv(np.linspace(0, 1, len(poses) + 1)[:-1])

    # ── Left: top-down (XY) ──────────────────────────────────────────────────
    ax = axes[0]
    ax.set_aspect("equal")
    ax.set_title("Top-down view  (XY plane)", fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(alpha=0.25)

    # Orbit circle (ideal horizontal)
    r  = poses[0]["orbit_radius"]
    th = np.linspace(0, 2 * math.pi, 300)
    ax.plot(target[0] + r * np.cos(th),
            target[1] + r * np.sin(th),
            color="lightgray", lw=1, zorder=0)

    # Target marker
    ax.scatter(target[0], target[1], s=80, color="black",
               zorder=5, marker="+", linewidths=2)
    ax.text(target[0] + 0.02 * r, target[1] + 0.02 * r,
            "target", fontsize=7, color="black")

    for i, (p, col) in enumerate(zip(poses, colours)):
        x, y = p["position"][0], p["position"][1]
        # Arrow: from camera toward target (projected onto XY)
        dx = target[0] - x
        dy = target[1] - y
        norm = math.hypot(dx, dy) + 1e-9
        alen = 0.18 * r
        ax.annotate("", xy=(x + alen * dx / norm, y + alen * dy / norm),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color=col,
                                    lw=1.5, mutation_scale=12))
        # Camera dot
        is_hi = (highlight_deg is not None and
                 int(p["azimuth_deg"]) == highlight_deg)
        ax.scatter(x, y, s=60 if not is_hi else 180,
                   color=col, zorder=4,
                   edgecolors="black" if is_hi else "none",
                   linewidths=1.5)
        ax.text(x + 0.06 * r, y + 0.06 * r,
                f"{int(p['azimuth_deg']):03d}°", fontsize=6.5, color=col)

    # ── Right: side view (XZ, Y integrated) ─────────────────────────────────
    ax = axes[1]
    ax.set_aspect("equal")
    ax.set_title("Side view  (XZ plane)", fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Z (height)")
    ax.grid(alpha=0.25)

    # Orbit ellipse in XZ (radius_x = orbit_radius * cos(elev), height = sin(elev)*r)
    elev_rad = math.radians(ELEVATION_DEG)
    rx = r * math.cos(elev_rad)
    rz = r * math.sin(elev_rad)
    ax.plot(target[0] + rx * np.cos(th),
            target[2] + rz * np.ones_like(th),
            color="lightgray", lw=1, ls="--", zorder=0,
            label="horizontal projection")
    ax.scatter(target[0], target[2], s=80, color="black",
               zorder=5, marker="+", linewidths=2)
    ax.text(target[0] + 0.02 * rx, target[2] + 0.015 * rz,
            "target", fontsize=7, color="black")

    for p, col in zip(poses, colours):
        x, z = p["position"][0], p["position"][2]
        dx = target[0] - x
        dz = target[2] - z
        norm = math.hypot(dx, dz) + 1e-9
        alen = 0.18 * r
        is_hi = (highlight_deg is not None and
                 int(p["azimuth_deg"]) == highlight_deg)
        ax.annotate("", xy=(x + alen * dx / norm, z + alen * dz / norm),
                    xytext=(x, z),
                    arrowprops=dict(arrowstyle="->", color=col,
                                    lw=1.5, mutation_scale=12))
        ax.scatter(x, z, s=60 if not is_hi else 180,
                   color=col, zorder=4,
                   edgecolors="black" if is_hi else "none",
                   linewidths=1.5)

    # Elevation angle annotation on one camera
    p0 = poses[0]
    cx, cz = p0["position"][0], p0["position"][2]
    ax.annotate(f"elev {ELEVATION_DEG:.0f}°",
                xy=(cx, cz), xytext=(cx + 0.3 * rx, cz + 0.2 * rz),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1),
                fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Diagram saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helper
# ─────────────────────────────────────────────────────────────────────────────

def print_pose(p: dict) -> None:
    print("\n" + "═" * 56)
    print(f"  Camera pose for azimuth {int(p['azimuth_deg']):03d}°")
    print("═" * 56)
    print(f"  Position   : {p['position']}")
    print(f"  Forward    : {p['forward']}")
    print(f"  Right      : {p['right']}")
    print(f"  Up         : {p['up']}")
    print(f"  Target     : {p['target']}")
    print(f"  Orbit r    : {p['orbit_radius']:.4f}")
    print(f"  Elevation  : {p['elevation_deg']:.1f}°")
    print(f"  FOV (yfov) : {p['fov_deg']:.1f}°")
    print("\n  Pose matrix (camera-to-world, OpenGL convention):")
    for row in p["pose_matrix"]:
        print("    " + "  ".join(f"{v:+.6f}" for v in row))
    print("═" * 56)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", required=True,
                    help="Path to camera_params.npz for the object")
    ap.add_argument("--file", default=None,
                    help="Render filename, e.g. '045.png' or just '045'")
    ap.add_argument("--diagram", default=None,
                    help="Output path for the camera-ring diagram PNG")
    args = ap.parse_args()

    highlight = None
    if args.file:
        pose = camera_pose_from_file(args.file, args.params)
        print_pose(pose)
        highlight = int(pose["azimuth_deg"])

    if args.diagram:
        visualise_camera_ring(args.params, args.diagram, highlight_deg=highlight)
    elif args.file is None:
        ap.error("Provide --file and/or --diagram.")


if __name__ == "__main__":
    main()
