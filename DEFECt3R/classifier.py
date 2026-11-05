"""
Per-pixel inconsistency classifiers (copied from the training scripts so the
released state dicts load as-is).
"""

import torch
import torch.nn as nn


class DinoMASt3RMLP(nn.Module):
    """
    DINO + MASt3R classifier (~63k parameters).
    Input:  dino_diff (N, 384), mast3r_diff (N, 24) — L2-normalised feature diffs
    Output: logit     (N,)
    """
    def __init__(self):
        super().__init__()
        self.dino_branch = nn.Sequential(
            nn.Linear(384, 128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 64),  nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(88, 64), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, dino_diff, mast3r_diff):
        d = self.dino_branch(dino_diff)
        return self.head(torch.cat([d, mast3r_diff], dim=1)).squeeze(-1)


class DinoMLP(nn.Module):
    """
    DINO-only classifier.
    Input:  dino_diff (N, 384) — L2-normalised DINO feature diff
    Output: logit     (N,)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(384, 128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 64),  nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, dino_diff):
        return self.net(dino_diff).squeeze(-1)
