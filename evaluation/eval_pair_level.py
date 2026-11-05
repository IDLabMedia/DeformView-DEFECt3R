#!/usr/bin/env python3
"""
Pair-level geometric inconsistency classification on DeformView (paper, Table 2).

A pair is inconsistent when it is an original-deformed (real_def) pair. Heatmap
models (DEFECt3R, MEt3R) are aggregated into one score per pair over all overlap
pixels (--agg, several at once possible); TSED directly scores a pair by its
median symmetric epipolar distance. Reports AUC, AP and precision / recall / F1
of the inconsistent class at the validation-selected threshold.

Examples:
    python eval_pair_level.py --model DEFECt3R --features dino+mast3r --corr mast3r --data_root /data/DeformView
    python eval_pair_level.py --model MEt3R --corr gt --agg p99 mean --data_root /data/DeformView
    python eval_pair_level.py --model TSED --corr gt --data_root /data/DeformView
"""

import numpy as np

from common import (build_parser, validate_args, run_inference, keep_pair, is_prefiltered,
                    pair_threshold, save_metrics_csv)
from metrics import binary_metrics


def main():
    parser = build_parser(__doc__, level="pair")
    args = parser.parse_args()
    validate_args(args, level="pair")

    rows, out_dir = run_inference(args)
    kept = [r for r in rows if keep_pair(r, "pair", args.corr, is_prefiltered(args))]
    labels = np.array([r["pair_type"] == "real_def" for r in kept])

    print(f"\nPair-level evaluation  ({len(kept)} of {len(rows)} pairs: "
          f"{int(labels.sum())} inconsistent, {int((~labels).sum())} consistent)")

    tsed = args.model == "TSED"
    results = []
    for agg in args.agg:
        scores = np.array([r["score" if tsed else f"agg_{agg}"] for r in kept])
        threshold = pair_threshold(args, agg)
        # TSED predicts inconsistent when median SED >= T_error
        m = binary_metrics(scores, labels, threshold, inclusive=tsed)
        print(f"\n  Aggregation: {agg}   threshold: "
              f"{'n/a (no validated threshold; pass --threshold)' if threshold is None else threshold}")
        for k in ("auc", "f1", "precision", "recall", "ap"):
            print(f"    {k:<10} {m[k]:.3f}")
        results.append({"aggregation": agg, **m})

    save_metrics_csv(out_dir / "pair_metrics.csv", results)


if __name__ == "__main__":
    main()
