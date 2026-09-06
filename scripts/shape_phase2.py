#!/usr/bin/env python3
"""Patro 2: model osudu jamky z MNOZSTVI a VYZNAMNOSTI dynaminu.

Patro 1 ukazalo, ze informace neni ve tvaru krivky, ale v tom, kolik
dynaminu prislo a jestli byl vyznamny (lokalni per-frame test cmeAnalysis).
Tady se z toho stavi model:

  blok D "mnozstvi+vyznamnost" (vse z amplitudy proti lokalnimu okoli):
    n_sig, frac_sig          pocet/podil vyznamnych snimku (lokalni test)
    n_sig_last10             vyznamne snimky v poslednich 10
    longest_sig_run          nejdelsi souvisly beh vyznamnych snimku
    lvl_mean, lvl_max        prumer a maximum amplitudy
    lvl_burst                max - median (vyska burstu nad vlastni hladinou)
    term_delta               prumer poslednich 10 minus prumer pred nimi
    peak_over_sigma          max A/A_pstd (sila nejsilnejsiho snimku)

Stitky: bezne SI pravidlo (max SI > 0.7) kvuli srovnatelnosti s referenci
(box-mean logreg within-band 0.585); zpevnene stitky jako druhy sloupec.

Vyhodnoceni: OOF po filmech, AUC pooled + within-band; logistika i
nelinearni kontrola (HistGradientBoosting).

    python3 scripts/shape_phase2.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANDS = [(6, 9), (10, 19), (20, 39), (40, 10 ** 9)]
D_COLS = ["n_sig", "frac_sig", "n_sig_last10", "longest_sig_run",
          "lvl_mean", "lvl_max", "lvl_burst", "term_delta", "peak_over_sigma"]


def auc_rank(score, y):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(score).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wb_auc(score, y, tlen):
    num = den = 0.0; per = []
    for lo, hi in BANDS:
        m = (tlen >= lo) & (tlen <= hi)
        a = auc_rank(score[m], y[m])
        n1 = int(y[m].sum()); w = n1 * (int(m.sum()) - n1)
        per.append(a)
        if np.isfinite(a):
            num += a * w; den += w
    return (num / den if den else np.nan), per


def longest_run(b):
    best = cur = 0
    for v in b:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def load_tracks():
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "lr registered", "measured",
                                           "*-lr-trajectories-dynamin.csv"))):
        movie = int(os.path.basename(p).split("_DNMsiRNA_")[1].split("-")[0])
        d = pd.read_csv(p, usecols=["particle", "frame", "cls", "dnm_A",
                                    "dnm_A_pstd", "dnm_signif"])
        for pid, t in d.groupby("particle"):
            t = t.sort_values("frame")
            A = t.dnm_A.to_numpy(float)
            L = len(A)
            if L < 6:
                continue
            sig = t.dnm_signif.to_numpy(int)
            k = min(10, L)
            pstd = t.dnm_A_pstd.to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                snr = np.where(pstd > 0, A / pstd, 0.0)
            cls = t.cls.to_numpy(float)
            rows.append(dict(
                movie=movie, particle=int(pid), track_len=L,
                si_plain=int(np.nanmax(cls) > 0.7),
                si_hard=1 if (cls > 0.7).sum() >= 3
                        else (0 if np.nanmax(cls) < 0.65 else -1),
                n_sig=int(sig.sum()), frac_sig=float(sig.mean()),
                n_sig_last10=int(sig[-k:].sum()),
                longest_sig_run=longest_run(sig),
                lvl_mean=float(A.mean()), lvl_max=float(A.max()),
                lvl_burst=float(A.max() - np.median(A)),
                term_delta=float(A[-k:].mean() - (A[:-k].mean() if L > k else A.mean())),
                peak_over_sigma=float(np.nanmax(snr))))
    return pd.DataFrame(rows)


def oof(model_maker, X, y, groups):
    out = np.full(len(y), np.nan)
    for trn, tst in GroupKFold(n_splits=5).split(X, y, groups):
        m = model_maker()
        m.fit(X[trn], y[trn])
        out[tst] = m.predict_proba(X[tst])[:, 1]
    return out


def main() -> int:
    tr = load_tracks()
    logit = lambda: make_pipeline(StandardScaler(),
                                  LogisticRegression(C=1.0, max_iter=2000,
                                                     random_state=0))
    gbm = lambda: HistGradientBoostingClassifier(random_state=0)
    out = {}
    for label_name, ycol in (("bezne stitky (max SI > 0.7)", "si_plain"),
                             ("zpevnene stitky", "si_hard")):
        d = tr[tr[ycol] >= 0].reset_index(drop=True)
        y = d[ycol].to_numpy(int)
        tlen = d.track_len.to_numpy(int)
        g = d.movie.to_numpy(int)
        print(f"\n=== {label_name}: n={len(d):,}, produktivnich {y.mean()*100:.1f} % ===")
        print(f"{'model':>26} | {'pooled':>6} {'wb':>6} | pasma 6-9/10-19/20-39/40+")
        runs = [
            ("A jen delka", logit, ["track_len"]),
            ("D mnozstvi+vyznamnost", logit, D_COLS),
            ("D+A", logit, D_COLS + ["track_len"]),
            ("D+A nelinearni (GBM)", gbm, D_COLS + ["track_len"]),
        ]
        for name, mk, cols in runs:
            s = oof(mk, d[cols].to_numpy(float), y, g)
            pooled = auc_rank(s, y)
            wb, per = wb_auc(s, y, tlen)
            out[f"{ycol}:{name}"] = dict(pooled=pooled, wb=wb, bands=per)
            print(f"{name:>26} | {pooled:6.3f} {wb:6.3f} | "
                  + "  ".join(f"{a:.3f}" for a in per))
    print("\nreference: kolegova box-mean logreg wb 0.585; nase amplituda "
          "(27 featur, jeho pipeline) wb 0.542-0.560; jen delka drive 0.62")
    od = os.path.join(ROOT, "out", "shape_phase2")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"JSON -> {od}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
