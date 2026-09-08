#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Obrazky experimentu dynamin -> klatrin (anglicky, pro kolegy):
1) mozaika confusion matic logisticke regrese (pooled + 4 lifetime groups):
   radky = dynamin->clathrin (prevalence-matched prah), dynamin->clathrin
   (GMM prah) a dynamin->SI (puvodni beh) pro primou srovnatelnost;
2) sweep prahu klatrinoveho stitku (none_endobs).

Cisla z Clathrin Analysis/report/dyn_to_clc_results.json a z kolegova
sweep JSONu; nic se nepocita znovu.

    python3 scripts/fig_dyn_to_clc.py [--dpi N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colormaps  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "Clathrin Analysis", "report", "dyn_to_clc_results.json")
HIS = os.path.join(ROOT, "external", "Shape2Fate_Fake2Emulate",
                   "dynamin_v2_confusion_sweep.json")
OUTDIR = os.path.join(ROOT, "Clathrin Analysis", "report")
CFG = "none_endobs"


def draw_panel(ax, m, neg, pos):
    tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]
    grid = [[tn, fp], [fn, tp]]
    fr = [[tn / max(tn + fp, 1), fp / max(tn + fp, 1)],
          [fn / max(fn + tp, 1), tp / max(fn + tp, 1)]]
    cmap = colormaps["Blues"]
    for r in range(2):
        for c in range(2):
            f = fr[r][c]
            ax.add_patch(plt.Rectangle((c, 1 - r), 1, 1, color=cmap(0.15 + 0.7 * f)))
            ax.text(c + 0.5, 1.5 - r, f"{grid[r][c]:,}\n{f * 100:.0f} %",
                    ha="center", va="center", fontsize=9,
                    color="white" if f > 0.55 else "black")
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5], ["pred −", "pred +"], fontsize=8)
    ax.set_yticks([1.5, 0.5], [neg, pos], fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def blocks_ours(rec):
    p = rec["pooled"]
    out = [("pooled", p, p["n"], p["prevalence"])]
    for b in rec["bands"]:
        out.append((f"{b['band']} frames", b, b["n"], b["prevalence"]))
    return out


def blocks_his():
    cfg = json.load(open(HIS))["configs"][CFG]
    L = "logistic regression"
    out = [("pooled", cfg["pooled"]["models"][L], cfg["n_tracks"], cfg["prevalence"])]
    for b in cfg["bands"]:
        out.append((f"{b['band']} frames", b["models"][L], b["n"],
                    b["n_productive"] / b["n"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()
    R = json.load(open(RES))

    rows = [
        (f"dynamin → clathrin\n(prevalence-matched thr,\n"
         f"κ vs SI = {R['prev']['configs'][CFG]['kappa_vs_SI']:.2f})",
         "Clc−", "Clc+", blocks_ours(R["prev"]["configs"][CFG])),
        (f"dynamin → clathrin\n(GMM thr,\n"
         f"κ vs SI = {R['gmm']['configs'][CFG]['kappa_vs_SI']:.2f})",
         "Clc−", "Clc+", blocks_ours(R["gmm"]["configs"][CFG])),
        ("dynamin → SI\n(original run)", "abortive", "productive", blocks_his()),
    ]
    fig, axes = plt.subplots(3, 5, figsize=(12.6, 8.6))
    for ri, (title, neg, pos, blocks) in enumerate(rows):
        for ci, (col, m, n, prev) in enumerate(blocks):
            ax = axes[ri][ci]
            draw_panel(ax, m, neg, pos)
            ax.set_title(f"{col}\nn = {n:,} · {prev * 100:.0f}% pos.", fontsize=8.5)
            ax.text(1, -0.42,
                    f"AUC {m['auc']:.3f} · thr {m['threshold']:.3f}\n"
                    f"sens {m['sensitivity']:.2f} · spec {m['specificity']:.2f}",
                    ha="center", va="top", fontsize=8)
        axes[ri][0].text(-0.55, 1.0, title, transform=axes[ri][0].transAxes,
                         ha="right", va="center", fontsize=9.5, fontweight="bold")
    for y in (0.652, 0.345):
        fig.add_artist(plt.Line2D([0.03, 0.985], [y, y], color="0.85", lw=0.8,
                                  transform=fig.transFigure))
    fig.suptitle("Logistic regression on dynamin features: label replaced by "
                 "clathrin-derived productivity (no filter, end-observed)",
                 fontsize=11.5, y=0.995)
    fig.text(0.5, 0.005,
             "Same dynamin features, filters, film-grouped folds and model as the "
             "original run; only the label changes. Each panel uses its own "
             "in-sample Youden threshold.",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0.06, 0.02, 1.0, 0.97))
    fig.subplots_adjust(hspace=0.95, wspace=0.45)
    out1 = os.path.join(OUTDIR, "dyn_to_clc_confusion_EN.png")
    fig.savefig(out1, dpi=args.dpi)
    plt.close(fig)
    print(f"obrazek -> {out1}")

    # ---- sweep prahu ----
    sw = R["sweep_none_endobs"]
    prev = [s["prevalence"] for s in sw]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.plot(prev, [s["auc"] for s in sw], "o-", color="#1b9e77", label="pooled AUC")
    ax.plot(prev, [s["wb_auc"] for s in sw], "s-", color="#d95f02",
            label="AUC within lifetime groups")
    m = R["_meta"]
    prev_pt = R["prev"]["configs"][CFG]["pooled"]["prevalence"]
    ax.axvline(prev_pt, color="#7570b3", ls="--", lw=1.2)
    ax.text(prev_pt + 0.006, 0.53,
            f"SI-matched prevalence ({prev_pt:.2f},\nthr = {m['thr_prev']:.0f} AU)",
            fontsize=8.5, color="#7570b3")
    ax.axhline(0.5, color="0.6", lw=1, ls=":")
    ax.set_xlabel("fraction of tracks labelled productive (label threshold sweep)")
    ax.set_ylabel("AUC")
    ax.set_title("Dynamin → clathrin: performance vs. clathrin label threshold",
                 fontsize=10.5)
    ax.invert_xaxis()
    ax.legend(fontsize=9)
    fig.tight_layout()
    out2 = os.path.join(OUTDIR, "dyn_to_clc_sweep_EN.png")
    fig.savefig(out2, dpi=args.dpi)
    print(f"obrazek -> {out2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
