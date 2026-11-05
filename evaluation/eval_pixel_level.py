#!/usr/bin/env python3
"""
Pixel-level geometric inconsistency localization on DeformView (paper, Table 1).

Each pixel is classified and compared with the GT mask. Only foreground pixels
of view 1 inside the overlap are evaluated. Metrics are pooled over all pixels
of the split, overall and per pair type:
    entire set / real_def : AUC, AP, F1, precision, recall
    def_def (deformed / non-deformed region), real_real : FPR

Examples:
    python eval_pixel_level.py --model DEFECt3R --features dino+mast3r --corr mast3r --data_root /data/DeformView
    python eval_pixel_level.py --model DEFECt3R --features dino --corr gt --split val --data_root /data/DeformView
    python eval_pixel_level.py --model MEt3R --corr mast3r --data_root /data/DeformView
"""

import numpy as np
from tqdm import tqdm

from common import (build_parser, validate_args, run_inference, keep_pair, is_prefiltered,
                    load_pixels, pixel_threshold, save_metrics_csv)
from metrics import binary_metrics

PAIR_TYPES = {"real_real": 0, "real_def": 1, "def_def": 2}


def collect_pixels(rows, out_dir, corr, prefiltered):
    """Concatenate the foreground overlap pixels of all kept pairs."""
    scores, labels, deform, types = [], [], [], []
    n_pairs = 0
    for r in tqdm([r for r in rows if keep_pair(r, "pixel", corr, prefiltered)], desc="Loading pixels",
                  dynamic_ncols=True):
        px = load_pixels(out_dir, r["pair_idx"])
        fg = px["fg"]
        if not fg.any():
            continue
        # every overlap pixel counts once, also when several share a location
        count = px["count"][fg]
        scores.append(np.repeat(px["scores"][fg], count))
        labels.append(np.repeat(px["label"][fg], count))
        deform.append(np.repeat(px["deform"][fg], count))
        types.append(np.full(int(count.sum()), PAIR_TYPES[r["pair_type"]], dtype=np.int8))
        n_pairs += 1
    return (np.concatenate(scores), np.concatenate(labels).astype(bool),
            np.concatenate(deform).astype(bool), np.concatenate(types), n_pairs)


def main():
    parser = build_parser(__doc__, level="pixel")
    args = parser.parse_args()
    validate_args(args, level="pixel")

    rows, out_dir = run_inference(args)
    threshold = pixel_threshold(args)
    scores, labels, deform, types, n_pairs = collect_pixels(rows, out_dir, args.corr,
                                                            is_prefiltered(args))

    print(f"\nPixel-level evaluation  ({n_pairs} pairs, {len(scores):,} foreground pixels, "
          f"threshold={threshold})")

    subsets = [
        ("Entire set",                   np.ones_like(labels),               ["auc", "f1", "precision", "recall", "ap"]),
        ("Original-Deformed",            types == PAIR_TYPES["real_def"],             ["auc", "f1", "precision", "recall", "ap"]),
        ("Deformed-Deformed (deformed)", (types == PAIR_TYPES["def_def"]) & deform,   ["fpr"]),
        ("Deformed-Deformed (non-def.)", (types == PAIR_TYPES["def_def"]) & ~deform,  ["fpr"]),
        ("Original-Original",            types == PAIR_TYPES["real_real"],            ["fpr"]),
    ]
    results = []
    for name, sel, keys in subsets:
        if not sel.any():
            continue
        m = binary_metrics(scores[sel], labels[sel], threshold)
        print(f"\n  {name}  (n={m['n']:,}  pos={m['n_pos']:,}  neg={m['n_neg']:,})")
        for k in keys:
            print(f"    {k:<10} {m[k]:.3f}")
        results.append({"subset": name, **m})

    save_metrics_csv(out_dir / "pixel_metrics.csv", results)


if __name__ == "__main__":
    main()
