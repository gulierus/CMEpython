#!/usr/bin/env python3
"""Patro 1 testu hypotezy "produktivni a abortivni jamky maji zvlastni prubeh
dynaminu": popisne krivky + souboj tri modelu.

Stitky (zpevnene, nezavisle na dynaminu):
  produktivni_hard = SI > 0.7 v >= 3 snimcich
  abortivni_hard   = max SI < 0.65
  seda zona (mezi) se z treninku i hodnoceni vynechava.

Vstupy per draha z cmeAnalysis amplitudy (dnm_A, tecka proti vlastnimu okoli):
  A (delka):   [track_len]
  B (uroven):  [prumer A, max A, max A - median A]
  C (tvar):    krivka z-normalizovana na vlastni drahu ((A-mean)/std) a
               prevzorkovana na 20 bodu zlomku zivota 0..1 -> nese JEN tvar,
               ne uroven ani delku.

Vyhodnoceni: 5-fold OOF s foldy po filmech; AUC pooled + within-band
(4 delkova pasma, vazeni poctem dvojic). Hypoteza je podprena, kdyz C
porazi A i B UVNITR pasem.

    python3 scripts/shape_phase1.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANDS = [(6, 9), (10, 19), (20, 39), (40, 10 ** 9)]
GRID = np.linspace(0.0, 1.0, 20)
ORANGE, TEAL, GREY = "#d95f02", "#1b9e77", "0.45"


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
        per.append((f"{lo}-{'+' if hi > 10**8 else hi}", a, int(m.sum())))
        if np.isfinite(a):
            num += a * w; den += w
    return (num / den if den else np.nan), per


def load_tracks():
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "lr registered", "measured",
                                           "*-lr-trajectories-dynamin.csv"))):
        movie = int(os.path.basename(p).split("_DNMsiRNA_")[1].split("-")[0])
        d = pd.read_csv(p, usecols=["particle", "frame", "cls", "dnm_A",
                                    "dnm_signif", "track_len"])
        for pid, t in d.groupby("particle"):
            t = t.sort_values("frame")
            A = t.dnm_A.to_numpy(float)
            L = len(A)
            if L < 6:
                continue
            cls = t.cls.to_numpy(float)
            n_over = int((cls > 0.7).sum())
            max_si = float(np.nanmax(cls))
            mu, sd = float(A.mean()), float(A.std())
            z = (A - mu) / (sd + 1e-9)
            frac = np.arange(L) / (L - 1)
            shape = np.interp(GRID, frac, z)
            med = float(np.median(A))
            sig_last10 = int(t.dnm_signif.to_numpy()[-min(10, L):].sum())
            rows.append(dict(
                movie=movie, particle=int(pid), track_len=L,
                n_over=n_over, max_si=max_si,
                lvl_mean=mu, lvl_max=float(A.max()), lvl_burst=float(A.max() - med),
                phi_peak=float(np.argmax(A)) / (L - 1),
                sig_last10=sig_last10,
                **{f"s{i}": shape[i] for i in range(len(GRID))}))
    return pd.DataFrame(rows)


def oof_scores(X, y, groups, seed=0):
    out = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=2000,
                                                random_state=seed))
        pipe.fit(X[tr], y[tr])
        out[te] = pipe.predict_proba(X[te])[:, 1]
    return out


def main() -> int:
    tr = load_tracks()
    prod = tr.n_over >= 3
    abort = tr.max_si < 0.65
    tr["label"] = np.where(prod, 1, np.where(abort, 0, -1))
    gray = int((tr.label == -1).sum())
    d = tr[tr.label >= 0].reset_index(drop=True)
    y = d.label.to_numpy(int)
    tlen = d.track_len.to_numpy(int)
    g = d.movie.to_numpy(int)
    print(f"drah >=6 snimku: {len(tr):,} | jiste produktivni {int(prod.sum()):,} "
          f"| jiste abortivni {int(abort.sum()):,} | seda zona vynechana {gray:,}")
    print(f"hodnocena mnozina: {len(d):,} drah, produktivnich {y.mean()*100:.1f} %\n")

    scols = [f"s{i}" for i in range(len(GRID))]
    MODELS = [
        ("A jen delka", ["track_len"]),
        ("B jen uroven", ["lvl_mean", "lvl_max", "lvl_burst"]),
        ("C jen tvar", scols),
        ("C+A tvar+delka", scols + ["track_len"]),
        ("plny B+C+A", scols + ["track_len", "lvl_mean", "lvl_max", "lvl_burst"]),
    ]
    res = {}
    print(f"{'model':>18} | {'pooled':>6} {'wb':>6} | pasma 6-9 / 10-19 / 20-39 / 40+")
    for name, cols in MODELS:
        s = oof_scores(d[cols].to_numpy(float), y, g)
        pooled = auc_rank(s, y)
        wb, per = wb_auc(s, y, tlen)
        res[name] = dict(pooled=pooled, wb=wb,
                         bands={b: a for b, a, _ in per})
        print(f"{name:>18} | {pooled:6.3f} {wb:6.3f} | "
              + "  ".join(f"{a:.3f}" for _, a, _ in per))

    # popisna cast: median tvarove krivky + IQR, phi_peak, terminalni vyznamnost
    S = d[scols].to_numpy(float)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax = axes[0]
    for lab, col, nm in ((1, ORANGE, "produktivni"), (0, TEAL, "abortivni")):
        m = y == lab
        med = np.median(S[m], axis=0)
        lo, hi = np.percentile(S[m], [25, 75], axis=0)
        ax.fill_between(GRID, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(GRID, med, color=col, lw=2, label=f"{nm} (n={m.sum():,})")
    ax.set_xlabel("zlomek zivota drahy"); ax.set_ylabel("dynamin, z-normalizovany na drahu")
    ax.set_title("Median tvaru krivky (pas = IQR)", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    for lab, col, nm in ((1, ORANGE, "produktivni"), (0, TEAL, "abortivni")):
        m = y == lab
        ax.hist(d.phi_peak[m], bins=20, density=True, histtype="step",
                color=col, lw=2, label=nm)
    ax.set_xlabel("poloha vrcholu dynaminu v zivote (0=vznik, 1=zanik)")
    ax.set_ylabel("hustota"); ax.set_title("Kdy dynamin vrcholi", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[2]
    rates = []
    for lab in (1, 0):
        m = y == lab
        rates.append([(d.sig_last10[m] >= k).mean() for k in (1, 2, 3, 5)])
    x = np.arange(4)
    ax.bar(x - 0.17, rates[0], 0.34, color=ORANGE, label="produktivni")
    ax.bar(x + 0.17, rates[1], 0.34, color=TEAL, label="abortivni")
    ax.set_xticks(x, [">=1", ">=2", ">=3", ">=5"])
    ax.set_xlabel("vyznamnych dynaminovych snimku v poslednich 10")
    ax.set_ylabel("podil drah"); ax.set_title("Terminalni vyznamnost (lokalni test)", fontsize=10)
    ax.legend(fontsize=8)
    out_dir = os.path.join(ROOT, "out", "shape_phase1")
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "shape_phase1.png"), dpi=160)
    plt.close(fig)

    res["_meta"] = dict(n=len(d), n_prod=int(y.sum()), gray=gray,
                        labels="prod: >=3 snimku SI>0.7; abort: max SI<0.65",
                        grid_pts=len(GRID))
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nulozen obrazek + JSON -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
