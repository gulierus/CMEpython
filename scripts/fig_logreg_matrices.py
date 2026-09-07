#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Confusion matice LOGISTICKE REGRESE ve stylu kolegovych mozaik:
sloupce = pooled + 4 delkove skupiny, radky = tri experimenty
(klatrin->dynamin amplituda, klatrin->dynamin box-mean,
dynamin->SI referencni beh kolegu). Pod kazdym panelem AUC, prah,
sens a spec. Cisla se ctou primo ze sweep JSONu, nic se nepocita znovu.

    python3 scripts/fig_logreg_matrices.py [--print]
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
LOG = "logistic regression"
CFG = "none_endobs"

ROWS = [
    ("klatrin → dynamin\n(amplituda)", "dynamin−", "dynamin+",
     os.path.join(ROOT, "CME_for_Helios", "runs", "clavg_amp_238740",
                  "dynamin_v2_confusion_sweep.json")),
    ("klatrin → dynamin\n(box-mean)", "dynamin−", "dynamin+",
     os.path.join(ROOT, "CME_for_Helios", "runs", "clavg_box_238741",
                  "dynamin_v2_confusion_sweep.json")),
    ("dynamin → SI\n(referenční běh kolegů)", "abortivní", "produktivní",
     os.path.join(ROOT, "external", "Shape2Fate_Fake2Emulate",
                  "dynamin_v2_confusion_sweep.json")),
]


def blocks(path):
    cfg = json.load(open(path, encoding="utf-8"))["configs"][CFG]
    out = [("smíchaně (pooled)", cfg["pooled"]["models"][LOG],
            cfg["n_tracks"], cfg["prevalence"])]
    for b in cfg["bands"]:
        out.append((f"{b['band']} snímků", b["models"][LOG],
                    b["n"], b["n_productive"] / b["n"]))
    return out


def cz(v, nd=3):
    return f"{v:.{nd}f}".replace(".", ",")


def draw_panel(ax, m, neg, pos):
    tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]
    grid = [[tn, fp], [fn, tp]]           # radky = skutecnost (neg, pos)
    fracs = [[tn / max(tn + fp, 1), fp / max(tn + fp, 1)],
             [fn / max(fn + tp, 1), tp / max(fn + tp, 1)]]
    cmap = colormaps["Blues"]
    for r in range(2):
        for c in range(2):
            f = fracs[r][c]
            ax.add_patch(plt.Rectangle((c, 1 - r), 1, 1, color=cmap(0.15 + 0.7 * f)))
            ax.text(c + 0.5, 1.5 - r, f"{grid[r][c]:,}".replace(",", " ")
                    + f"\n{f * 100:.0f} %",
                    ha="center", va="center", fontsize=9,
                    color="white" if f > 0.55 else "black")
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5], ["− verdikt", "+ verdikt"], fontsize=8)
    ax.set_yticks([1.5, 0.5], [neg, pos], fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--print", action="store_true", dest="do_print")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--out", default=os.path.join(ROOT, "Clathrin Analysis",
                                                  "report", "fig_matice_logreg.png"))
    args = ap.parse_args()

    fig, axes = plt.subplots(len(ROWS), 5, figsize=(12.6, 8.4))
    for ri, (title, neg, pos, path) in enumerate(ROWS):
        for ci, (col, m, n, prev) in enumerate(blocks(path)):
            ax = axes[ri][ci]
            draw_panel(ax, m, neg, pos)
            if ri == 0:
                pass
            ax.set_title(f"{col}\nn = {n:,}".replace(",", " ")
                         + f" · podíl + {prev * 100:.0f} %", fontsize=8.5)
            ax.text(1, -0.42,
                    f"AUC {cz(m['auc'])} · práh {cz(m['threshold'])}\n"
                    f"sens {cz(m['sensitivity'], 2)} · spec {cz(m['specificity'], 2)}",
                    ha="center", va="top", fontsize=8)
            if args.do_print:
                print(f"{title.splitlines()[0]:>28} | {col:>18} | n {n:>6,} "
                      f"prev {prev:.3f} | tn {m['tn']} fp {m['fp']} fn {m['fn']} "
                      f"tp {m['tp']} | AUC {m['auc']:.3f} thr {m['threshold']:.3f} "
                      f"sens {m['sensitivity']:.3f} spec {m['specificity']:.3f}")
        axes[ri][0].text(-0.55, 1.0, title, transform=axes[ri][0].transAxes,
                         ha="right", va="center", fontsize=10.5, fontweight="bold")
    fig.suptitle("Logistická regrese: confusion matice smíchaně a po délkových "
                 "skupinách (konfigurace bez filtru, end-observed)",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0.06, 0.0, 1.0, 0.97))
    fig.subplots_adjust(hspace=0.95, wspace=0.45)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"obrazek -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
