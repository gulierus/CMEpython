#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Confusion matice LOGISTICKE REGRESE ve stylu kolegovych mozaik:
sloupce = pooled + 4 delkove skupiny, radky = tri experimenty
(klatrin->dynamin amplituda, klatrin->dynamin box-mean,
dynamin->SI referencni beh kolegu). Pod kazdym panelem AUC, prah,
sens a spec. Cisla se ctou primo ze sweep JSONu, nic se nepocita znovu.

    python3 scripts/fig_logreg_matrices.py [--lang cz|en] [--dpi N] [--print]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colormaps  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = "logistic regression"
CFG = "none_endobs"

PATHS = [
    os.path.join(ROOT, "CME_for_Helios", "runs", "clavg_amp_238740",
                 "dynamin_v2_confusion_sweep.json"),
    os.path.join(ROOT, "CME_for_Helios", "runs", "clavg_box_238741",
                 "dynamin_v2_confusion_sweep.json"),
    os.path.join(ROOT, "external", "Shape2Fate_Fake2Emulate",
                 "dynamin_v2_confusion_sweep.json"),
]

TXT = {
    "cz": dict(
        rows=[("klatrin → dynamin\n(amplituda)", "dynamin−", "dynamin+"),
              ("klatrin → dynamin\n(box-mean)", "dynamin−", "dynamin+"),
              ("dynamin → SI\n(referenční běh kolegů)", "abortivní", "produktivní")],
        pooled="smíchaně (pooled)", band="{} snímků",
        head="n = {n} · podíl + {p:.0f} %",
        pred=["− verdikt", "+ verdikt"],
        stats="AUC {auc} · práh {thr}\nsens {sens} · spec {spec}",
        title="Logistická regrese: confusion matice smíchaně a po délkových "
              "skupinách (konfigurace bez filtru, end-observed)",
        foot="Každý panel má vlastní práh (Youdenovo J na daném výběru, in-sample). "
             "Smíchané panely nafukuje délka; srovnává se po skupinách.",
        dec=",", thous=" "),
    "en": dict(
        rows=[("clathrin → dynamin\n(fit amplitude)", "Dyn−", "Dyn+"),
              ("clathrin → dynamin\n(box mean)", "Dyn−", "Dyn+"),
              ("dynamin → SI\n(colleagues' reference run)", "abortive", "productive")],
        pooled="pooled", band="{} frames",
        head="n = {n} · {p:.0f}% pos.",
        pred=["pred −", "pred +"],
        stats="AUC {auc} · thr {thr}\nsens {sens} · spec {spec}",
        title="Logistic regression: confusion matrices, pooled and per lifetime "
              "group (no filter, end-observed)",
        foot="Each panel uses its own Youden threshold (chosen in-sample). "
             "Pooled panels are inflated by lifetime; compare within groups.",
        dec=".", thous=","),
}


def blocks(path, T):
    cfg = json.load(open(path, encoding="utf-8"))["configs"][CFG]
    out = [(T["pooled"], cfg["pooled"]["models"][LOG],
            cfg["n_tracks"], cfg["prevalence"])]
    for b in cfg["bands"]:
        out.append((T["band"].format(b["band"]), b["models"][LOG],
                    b["n"], b["n_productive"] / b["n"]))
    return out


def draw_panel(ax, m, neg, pos, T):
    tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]
    grid = [[tn, fp], [fn, tp]]           # radky = skutecnost (neg, pos)
    fracs = [[tn / max(tn + fp, 1), fp / max(tn + fp, 1)],
             [fn / max(fn + tp, 1), tp / max(fn + tp, 1)]]
    cmap = colormaps["Blues"]
    for r in range(2):
        for c in range(2):
            f = fracs[r][c]
            ax.add_patch(plt.Rectangle((c, 1 - r), 1, 1,
                                       color=cmap(0.15 + 0.7 * f)))
            ax.text(c + 0.5, 1.5 - r,
                    f"{grid[r][c]:,}".replace(",", T["thous"])
                    + f"\n{f * 100:.0f} %",
                    ha="center", va="center", fontsize=9,
                    color="white" if f > 0.55 else "black")
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5], T["pred"], fontsize=8)
    ax.set_yticks([1.5, 0.5], [neg, pos], fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", choices=["cz", "en"], default="cz")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--print", action="store_true", dest="do_print")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    T = TXT[args.lang]
    out = args.out or os.path.join(
        ROOT, "Clathrin Analysis", "report",
        "fig_matice_logreg.png" if args.lang == "cz"
        else "confusion_matrices_logreg_EN.png")

    def num(v, nd=3):
        return f"{v:.{nd}f}".replace(".", T["dec"])

    fig, axes = plt.subplots(3, 5, figsize=(12.6, 8.6))
    for ri, (title, neg, pos) in enumerate(T["rows"]):
        for ci, (col, m, n, prev) in enumerate(blocks(PATHS[ri], T)):
            ax = axes[ri][ci]
            draw_panel(ax, m, neg, pos, T)
            head = T["head"].format(n=f"{n:,}".replace(",", T["thous"]), p=prev * 100)
            ax.set_title(f"{col}\n{head}", fontsize=8.5)
            ax.text(1, -0.42, T["stats"].format(
                auc=num(m["auc"]), thr=num(m["threshold"]),
                sens=num(m["sensitivity"], 2), spec=num(m["specificity"], 2)),
                ha="center", va="top", fontsize=8)
            if args.do_print:
                print(f"{title.splitlines()[0]:>28} | {col:>18} | n {n:>6,} "
                      f"prev {prev:.3f} | tn {m['tn']} fp {m['fp']} fn {m['fn']} "
                      f"tp {m['tp']} | AUC {m['auc']:.3f} thr {m['threshold']:.3f} "
                      f"sens {m['sensitivity']:.3f} spec {m['specificity']:.3f}")
        axes[ri][0].text(-0.55, 1.0, title, transform=axes[ri][0].transAxes,
                         ha="right", va="center", fontsize=10.5, fontweight="bold")
    for y in (0.652, 0.345):   # jemne oddelovaci linky mezi radky experimentu
        fig.add_artist(plt.Line2D([0.03, 0.985], [y, y], color="0.85",
                                  lw=0.8, transform=fig.transFigure))
    fig.suptitle(T["title"], fontsize=12, y=0.995)
    fig.text(0.5, 0.005, T["foot"], ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0.06, 0.02, 1.0, 0.97))
    fig.subplots_adjust(hspace=0.95, wspace=0.45)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f"obrazek -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
