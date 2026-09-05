#!/usr/bin/env python3
"""Prahova analyza samotne klatrinove intenzity proti dynaminovemu stitku.

Pro kazdy readout (clc_A = PSF amplituda, box5_norm = box-mean/biexp)
a statistiku dahy (max a prumer pres zivot) spocita:
  - pooled AUC vs dynaminova klasifikace (significantSlave, fixed masks),
  - within-band AUC (delkova pasma 6-9/10-19/20-39/40+ jako ve sweepu,
    vazeni poctem dvojic) -- delkove ocisteny signal,
  - Youdenuv prah, sens/spec a confusion matici u nej,
  - baseline "jen delka" (track_len jako skore).

AUC pres Mann-Whitney poradi (bez sklearn).  Vystup: tabulka na stdout
+ JSON vedle samplovanych dat.

    python3 scripts/clavg_threshold_analysis.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANDS = [(6, 9), (10, 19), (20, 39), (40, 10 ** 9)]


def auc_rank(score: np.ndarray, y: np.ndarray) -> float:
    """AUC = Mann-Whitney U / (n1*n0), s prumernym poradim pro shody."""
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    order = pd.Series(score).rank(method="average").to_numpy()
    return float((order[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wb_auc(score, y, tlen):
    num = den = 0.0
    out = []
    for lo, hi in BANDS:
        m = (tlen >= lo) & (tlen <= hi)
        a = auc_rank(score[m], y[m])
        n1 = int(y[m].sum()); n0 = int(m.sum()) - n1
        w = n1 * n0
        out.append({"band": f"{lo}-{hi if hi < 10**8 else '+'}", "auc": a,
                    "n": int(m.sum()), "prev": n1 / max(m.sum(), 1)})
        if np.isfinite(a):
            num += a * w; den += w
    return num / den if den else np.nan, out


def youden(score, y):
    order = np.argsort(-score)
    s, yy = score[order], y[order]
    n1 = yy.sum(); n0 = len(yy) - n1
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    sens = tp / n1; spec = 1 - fp / n0
    j = sens + spec - 1
    # prah mezi po sobe jdoucimi skore; vezmi bod s max J
    k = int(np.argmax(j))
    thr = s[k] if k == len(s) - 1 else (s[k] + s[k + 1]) / 2
    tp_, fp_ = int(tp[k]), int(fp[k])
    fn_, tn_ = int(n1 - tp_), int(n0 - fp_)
    return dict(threshold=float(thr), sens=float(sens[k]), spec=float(spec[k]),
                J=float(j[k]), tp=tp_, fp=fp_, fn=fn_, tn=tn_)


def main() -> int:
    files = sorted(glob.glob(os.path.join(ROOT, "Clathrin Analysis", "sampled",
                                          "*-clathrin-avg.csv")))
    lab = pd.read_csv(os.path.join(ROOT, "lr registered",
                                   "classification_comparison_by_track.csv"))
    rows = []
    for p in files:
        movie = int(os.path.basename(p).split("_DNMsiRNA_")[1].split("-")[0])
        d = pd.read_csv(p, usecols=["particle", "track_len", "clc_A", "clc_valid",
                                    "box5_norm"])
        g = d.groupby("particle")
        t = pd.DataFrame({
            "movie": movie,
            "track_len": g.track_len.first(),
            "amp_max": g.clc_A.max(), "amp_mean": g.clc_A.mean(),
            "box_max": g.box5_norm.max(), "box_mean": g.box5_norm.mean(),
        }).reset_index()
        rows.append(t)
    tr = pd.concat(rows, ignore_index=True).merge(
        lab[["movie", "particle", "cme_significant_slave"]],
        on=["movie", "particle"], validate="one_to_one")
    y = tr.cme_significant_slave.to_numpy(int)
    tlen = tr.track_len.to_numpy(int)
    keep = tlen >= 6
    tr, y, tlen = tr[keep], y[keep], tlen[keep]
    print(f"drah {len(tr):,} (track_len >= 6), dynamin+ {y.mean()*100:.1f} %\n")

    res = {}
    stats = [("box_max", "box-mean, max pres zivot"), ("box_mean", "box-mean, prumer"),
             ("amp_max", "PSF amplituda, max pres zivot"), ("amp_mean", "PSF amplituda, prumer"),
             ("track_len", "JEN DELKA (baseline)")]
    print(f"{'skore':>30} {'pooled':>7} {'wb':>7} {'prah':>9} {'sens':>6} {'spec':>6}  confusion (tp/fp/fn/tn)")
    for col, name in stats:
        s = tr[col].to_numpy(float)
        pooled = auc_rank(s, y)
        wb, bands = wb_auc(s, y, tlen)
        yd = youden(s, y)
        res[col] = {"name": name, "pooled_auc": pooled, "wb_auc": wb,
                    "bands": bands, "youden": yd}
        print(f"{name:>30} {pooled:7.3f} {wb:7.3f} {yd['threshold']:9.3g} "
              f"{yd['sens']:6.2f} {yd['spec']:6.2f}  "
              f"{yd['tp']:,}/{yd['fp']:,}/{yd['fn']:,}/{yd['tn']:,}")
    print("\npo pasmech (box_max):")
    for b in res["box_max"]["bands"]:
        print(f"  {b['band']:>7}: AUC {b['auc']:.3f}  n {b['n']:>6,}  dynamin+ {b['prev']*100:5.1f} %")

    out = os.path.join(ROOT, "Clathrin Analysis", "sampled", "threshold_analysis.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nulozeno: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
