"""
Shared logic of the pixel- and pair-level evaluation.

Every model × features × correspondence × pair-file combination is run once:
the per-pair results are written to

    results/<pair list>/<split>/<run>/per_pair.csv          one row per pair (pair-level scores,
                                                            pixel counts)
    results/<pair list>/<split>/<run>/pixels/pair_XXXXX.npz per-pixel scores + GT of the overlap
                                                            pixels (heatmap models only)

Both evaluation scripts call `run_inference`, which only computes the pairs that
are not yet in per_pair.csv (so an interrupted run resumes, and pixel- and
pair-level evaluation share the same model outputs).
"""

import argparse
import csv
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from DeformView.dataset import DeformView, read_pairs, SPLITS_DIR, SPLITS_GT_DIR  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODELS = ("DEFECt3R", "MEt3R", "TSED")
FEATURES = ("dino", "dino+mast3r")
CORRESPONDENCES = ("mast3r", "gt")

# Pair-level aggregation of the pixel scores (over all overlap pixels).
AGGREGATIONS = {
    "mean": lambda s: float(np.mean(s)),
    "p99":  lambda s: float(np.percentile(s, 99)),
}

# Pair-level decision thresholds, selected on the validation split (paper, Table 2).
# (model, features, correspondence) → {aggregation: threshold}
PAIR_THRESHOLDS = {
    ("DEFECt3R", "dino+mast3r", "mast3r"): {"p99": 0.3497},
    ("DEFECt3R", "dino",        "mast3r"): {"p99": 0.3339},
    ("DEFECt3R", "dino",        "gt"):     {"p99": 0.3622},
    ("MEt3R",    "dino",        "mast3r"): {"p99": 0.3232, "mean": 0.0336},
    ("MEt3R",    "dino",        "gt"):     {"p99": 0.2639, "mean": 0.0850},
}

# TSED produces one score per pair (median symmetric epipolar distance).
TSED_SCORE = "median_sed"

# Pairs skipped by the original pipelines (kept identical to reproduce the paper).
MIN_OVERLAP_PIXELS_MASt3R_PAIR = 1000    # MASt3R correspondences, pair-level
MIN_OVERLAP_PIXELS_GT          = 10000   # GT correspondences
MIN_DEFORM_VISIBLE_GT          = 225     # GT: deformed region visible in the overlap

PAIR_FIELDS = ["pair_idx", "img1", "img2", "mask", "pair_type", "status"]
HEATMAP_FIELDS = ["n_overlap", "n_fg", "n_pos", "n_deform"] + [f"agg_{a}" for a in AGGREGATIONS]
TSED_FIELDS = ["n_matches", "median_sed", "score", "note"]


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def build_parser(description: str, level: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--features", default="dino", choices=FEATURES,
                        help="dino+mast3r is only available for DEFECt3R (ignored for TSED)")
    parser.add_argument("--corr", required=True, choices=CORRESPONDENCES,
                        help="correspondences: estimated by MASt3R, or GT poses + depth (DeformView only)")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"),
                        help="DeformView split. The pair lists of the paper are used: "
                             "DeformView/split_GT/ for MEt3R and DEFECt3R with GT correspondences, "
                             "DeformView/splits/ otherwise (incl. TSED with GT poses)")
    parser.add_argument("--pairs", type=Path, default=None,
                        help="custom pair file (overrides --split)")
    parser.add_argument("--data_root", type=Path, required=True,
                        help="DeformView data root")
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true",
                        help="run the model under fp16 autocast (stored as a separate run)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the decision threshold")
    if level == "pair":
        parser.add_argument("--agg", nargs="+", default=["p99"], choices=list(AGGREGATIONS),
                            help="aggregation(s) of the pixel scores into a pair score "
                                 "(ignored for TSED)")
    return parser


def validate_args(args, level: str) -> None:
    if args.model == "TSED":
        if level == "pixel":
            raise SystemExit("TSED only produces pair-level scores; use eval_pair_level.py.")
        args.features = None
        args.amp = False
        if level == "pair":
            args.agg = [TSED_SCORE]
    if args.features == "dino+mast3r" and (args.model != "DEFECt3R" or args.corr != "mast3r"):
        raise SystemExit("--features dino+mast3r is only available for DEFECt3R with --corr mast3r.")
    if level == "pair" and args.threshold is not None and len(args.agg) > 1:
        raise SystemExit("--threshold can only be combined with a single --agg.")


def pair_file(args) -> Path:
    if args.pairs is not None:
        return args.pairs
    # TSED (also with GT poses) was evaluated on the MASt3R pair list
    splits_dir = SPLITS_GT_DIR if args.corr == "gt" and args.model != "TSED" else SPLITS_DIR
    return splits_dir / f"{args.split}.csv"


def is_prefiltered(args) -> bool:
    """split_GT/ is the GT pipeline's filtered pair list: its skip rules are already applied."""
    return pair_file(args).resolve().parent == SPLITS_GT_DIR.resolve()


def run_dir(args) -> Path:
    name = args.model
    if args.features is not None:
        name += "_" + args.features.replace("+", "-")
    name += "_" + args.corr
    if args.amp:
        name += "_amp"
    pairs = pair_file(args)
    return args.results_dir / pairs.parent.name / pairs.stem / name


def pixel_threshold(args):
    if args.threshold is not None:
        return args.threshold
    if args.model == "DEFECt3R":
        from DEFECt3R import VARIANTS
        return VARIANTS[(args.features, args.corr)]["threshold"]
    if args.model == "MEt3R":
        from MEt3R import THRESHOLDS
        return THRESHOLDS[args.corr]
    return None


def pair_threshold(args, agg: str):
    if args.threshold is not None:
        return args.threshold
    if args.model == "TSED":
        from TSED import THRESHOLDS
        return THRESHOLDS[args.corr]
    return PAIR_THRESHOLDS.get((args.model, args.features, args.corr), {}).get(agg)


def csv_fields(args) -> list:
    return PAIR_FIELDS + (TSED_FIELDS if args.model == "TSED" else HEATMAP_FIELDS)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(args):
    if args.model == "DEFECt3R":
        from DEFECt3R import DEFECt3R
        return DEFECt3R(features=args.features, correspondence=args.corr).to(args.device)
    if args.model == "MEt3R":
        from MEt3R import MEt3R
        return MEt3R(correspondence=args.corr).to(args.device)
    if args.model == "TSED":
        from TSED import TSED
        return TSED(correspondence=args.corr, device=args.device)
    raise NotImplementedError(args.model)


# ─────────────────────────────────────────────────────────────────────────────
# Inference (computed once per run, cached in per_pair.csv + pixels/)
# ─────────────────────────────────────────────────────────────────────────────

def _load_gt_geometry(data: DeformView, pair: dict) -> dict:
    geometry = {}
    for k, key in (("1", "img1"), ("2", "img2")):
        depth = data.load_depth(pair[key])
        pose = data.load_pose(pair[key])
        if depth is None or pose is None:
            raise FileNotFoundError(f"Missing depth map or camera_params.npz for {pair[key]}")
        geometry[f"depth{k}"] = torch.from_numpy(depth)
        geometry[f"pose{k}"] = torch.from_numpy(pose)
    return geometry


def process_tsed_pair(model, data: DeformView, pair: dict, row: dict) -> dict:
    poses = {}
    if model.correspondence_type == "gt":
        for k, key in (("1", "img1"), ("2", "img2")):
            pose = data.load_pose(pair[key], dtype=np.float64)
            if pose is None:
                raise FileNotFoundError(f"Missing camera_params.npz for {pair[key]}")
            poses[f"pose{k}"] = pose
    out = model(data.root / pair["img1"], data.root / pair["img2"], **poses)
    return {**row, "status": "ok", **{k: out[k] for k in TSED_FIELDS}}


def process_pair(model, data: DeformView, pair: dict, args, pixels_dir: Path) -> dict:
    row = {k: pair[k] for k in ("pair_idx", "img1", "img2", "mask", "pair_type")}
    row["mask"] = row["mask"] or ""
    if args.model == "TSED":
        return process_tsed_pair(model, data, pair, row)

    device = args.device
    img1 = data.load_image(pair["img1"]).to(device)
    img2 = data.load_image(pair["img2"]).to(device)
    geometry = _load_gt_geometry(data, pair) if args.corr == "gt" else {}
    geometry = {k: v.to(device) for k, v in geometry.items()}

    deform_mask = data.load_deform_mask(pair["mask"])
    fg_mask = data.load_fg_mask(pair["img1"])
    pair_type = pair["pair_type"]
    # GT, in the frame of view 1:
    #   label  = inconsistent pixel (deformed region of a real_def pair)
    #   deform = deformed region visible in view 1 (real_def and def_def)
    label_map = deform_mask if pair_type == "real_def" else np.zeros_like(deform_mask)
    deform_map = np.zeros_like(deform_mask) if pair_type == "real_real" else deform_mask

    amp = torch.autocast("cuda", dtype=torch.float16) if args.amp and "cuda" in device else nullcontext()
    try:
        with amp:
            out = model(img1, img2, **geometry)
    except Exception as e:   # the original pipelines skipped pairs that failed
        tqdm.write(f"  [error pair {pair['pair_idx']}] {e}")
        return {**row, "status": f"error: {e}", "n_overlap": 0, "n_fg": 0, "n_pos": 0, "n_deform": 0,
                **{f"agg_{a}": float("nan") for a in AGGREGATIONS}}

    heatmap = out["heatmap"].float().cpu().numpy()
    multiplicity = out["multiplicity"].cpu().numpy()
    rows, cols = np.nonzero(multiplicity)
    scores = heatmap[rows, cols].astype(np.float32)
    count = multiplicity[rows, cols].astype(np.uint16)
    label = label_map[rows, cols]
    fg = fg_mask[rows, cols]
    deform = deform_map[rows, cols]

    all_scores = np.repeat(scores, count)
    row.update(
        status="ok",
        n_overlap=int(count.sum()),
        n_fg=int(count[fg].sum()),
        n_pos=int(count[label].sum()),
        n_deform=int(count[deform].sum()),
        **{f"agg_{a}": (f(all_scores) if len(all_scores) else float("nan"))
           for a, f in AGGREGATIONS.items()},
    )
    np.savez_compressed(
        pixels_dir / f"pair_{pair['pair_idx']:05d}.npz",
        rows=rows.astype(np.int16), cols=cols.astype(np.int16),
        scores=scores, count=count, label=label, fg=fg, deform=deform,
    )
    return row


def _parse_row(r: dict) -> dict:
    r["pair_idx"] = int(r["pair_idx"])
    for k in ("n_overlap", "n_fg", "n_pos", "n_deform", "n_matches"):
        if k in r:
            r[k] = int(r[k])
    for k in [f"agg_{a}" for a in AGGREGATIONS] + ["median_sed", "score"]:
        if k in r:
            r[k] = float(r[k])   # also parses "inf"
    return r


def run_inference(args) -> tuple[list[dict], Path]:
    """Compute (or load) the per-pair results of a run. Returns (rows, run_dir)."""
    pairs = read_pairs(pair_file(args))
    out_dir = run_dir(args)
    pixels_dir = out_dir / "pixels"
    (out_dir if args.model == "TSED" else pixels_dir).mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_pair.csv"

    done = {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done[int(r["pair_idx"])] = _parse_row(r)

    todo = [p for p in pairs if p["pair_idx"] not in done]
    print(f"Run: {out_dir}")
    print(f"Pairs: {len(pairs)}  (cached: {len(pairs) - len(todo)}, to compute: {len(todo)})")

    if todo:
        model = build_model(args)
        data = DeformView(args.data_root)
        new_file = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields(args))
            if new_file:
                writer.writeheader()
            for pair in tqdm(todo, desc="Inference", dynamic_ncols=True):
                r = process_pair(model, data, pair, args, pixels_dir)
                if r["status"] == "ok":   # failed pairs are skipped now and retried next run
                    writer.writerow(r)
                    f.flush()
                done[r["pair_idx"]] = r
        del model
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    return [done[p["pair_idx"]] for p in pairs], out_dir


def load_pixels(out_dir: Path, pair_idx: int) -> dict:
    with np.load(out_dir / "pixels" / f"pair_{pair_idx:05d}.npz") as d:
        return {k: d[k] for k in d.files}


# ─────────────────────────────────────────────────────────────────────────────
# Pair selection (identical to the original pipelines)
# ─────────────────────────────────────────────────────────────────────────────

def keep_pair(row: dict, level: str, corr: str, prefiltered: bool = False) -> bool:
    if row["status"] != "ok":
        return False
    if "note" in row:   # TSED: only pairs without a pose are excluded
        return row["note"] != "missing_pose"
    deformed = row["pair_type"] in ("real_def", "def_def")
    if corr == "gt":
        # skip rules of the GT precompute; already applied to the split_GT pair lists
        if not prefiltered and (row["n_overlap"] < MIN_OVERLAP_PIXELS_GT
                                or (deformed and row["n_deform"] < MIN_DEFORM_VISIBLE_GT)):
            return False
    elif level == "pixel":
        if row["n_overlap"] == 0 or row["n_fg"] == 0:
            return False
    elif row["n_overlap"] < MIN_OVERLAP_PIXELS_MASt3R_PAIR:
        return False
    if level == "pair" and row["pair_type"] == "real_def" and row["n_pos"] == 0:
        return False
    return True


def save_metrics_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {path}")
