#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""English version of the summary bar chart: how much dynamin information
clathrin carries after length adjustment. Numbers from verified runs.

    python3 scripts/fig_summary_en.py [--dpi N]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BARS = [
    ("best single number\n(brightness)", 0.543, "0.45"),
    ("trained model,\nbox mean", 0.595, "#1b9e77"),
    ("trained model,\nfit amplitude", 0.657, "#d95f02"),
]
LEN_WB = 0.596


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "Clathrin Analysis", "report", "summary_auc_EN.png"))
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(BARS))
    ax.bar(x, [v for _, v, _ in BARS], 0.55, color=[c for _, _, c in BARS])
    for xi, (_, v, _) in zip(x, BARS):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    ax.axhline(0.5, color="0.6", lw=1, ls=":")
    ax.axhline(LEN_WB, color="#7570b3", lw=1.3, ls="--")
    ax.text(2.42, LEN_WB + 0.005, "lifetime alone (0.596)", fontsize=9,
            color="#7570b3", ha="right",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
    ax.set_xticks(x, [n for n, _, _ in BARS], fontsize=9.5)
    ax.set_ylim(0.45, 0.70)
    ax.set_ylabel("AUC within lifetime groups", fontsize=10)
    ax.set_title("Dynamin information carried by clathrin (length-adjusted)",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"obrazek -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
