# -*- coding: utf-8 -*-
# CMEpython -- Python port mereni intenzity dynaminu z cmeAnalysis.
# Copyright (C) 2026 Ruslan Guliev
#
# Odvozeno z cmeAnalysis (DanuserLab, UT Southwestern), GPL-3.0.
# Tento program je svobodny software: muzete jej sirit a upravovat podle
# podminek GNU General Public License verze 3 nebo pozdejsi. Sirenu je
# BEZ JAKEKOLI ZARUKY. Viz soubor LICENSE.
"""Reprodukce kvantitativnich tvrzeni o datasetu z README.

Tri analyzy, ktere README cituje a ktere by jinak byly neoveritelne
konstanty (presne tohle vytkl adversarialni audit -- cisla bez receptu):

  1. kalibrace testu vyznamnosti: falesne pozitivni na cistem pozadi
     a vliv korekce na korelovany sum (tau)
  2. photobleaching: pokles amplitudy po kontrole na delku drahy
     a OBOUSTRANNOU cenzuru, pokles pozadi per film
  3. rozptyl mezi filmy: dynamin vs pozadi tehoz kanalu vs clathrin

Pevne seedy; behove poradi analyz lze vybrat argumentem
(fp | bleach | crossfilm | all). Vysledky se tisknou, nic se nezapisuje.

Spusteni:  python3 scripts/dataset_findings.py [all]
"""

from __future__ import annotations

import csv
import glob
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cmepython.slave_intensity import fit_gaussian_2d_point   # noqa: E402
from cmepython.psf_calibration import detect_candidates        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MOVIE_DIR = ROOT / "reconstructed registered"
MEASURED_DIR = ROOT / "measured"
MOVIE16 = MOVIE_DIR / "U2OS_DYNAMIN_MSTAYGOLD_GREEN_SNAP_CLC_RED_DNMsiRNA_16_RR.tif"

SIGMA_SLAVE = 2.6424      # psf_calibration.json, cely dataset
SIGMA_MASTER = 1.4187
NCH = 3
CH_CLC, CH_DNM = 0, 2


def _page(tf, t, c):
    return tf.pages[t * NCH + c].asarray()


# ---------------------------------------------------------------- 1) FP + tau

def fp_and_tau(frame=80, n_null=1500, n_sig=600, seed=1):
    """Kalibrace testu na prazdnych pozicich + vliv tau korekce.

    KLICOVY DETAIL (nalezeny auditem): nulova sada se musi cistit od
    slabych zdroju POD detekcnim prahem. Pozice >4*sigma od detekci
    pri k=3.0 jeste obsahuji realne slabe spoty; teprve vylouceni
    detekci pri k=1.5 dava ciste pozadi. S kontaminovanou nulou vyjde
    FP ~1 %, s cistou ~0.3 %.
    """
    import tifffile
    from scipy.spatial import cKDTree
    from scipy.stats import t as st, norm
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    kL = norm.ppf(1 - 0.05 / 2)
    with tifffile.TiffFile(MOVIE16) as tf:
        T = tf.series[0].shape[0]
        dnm = _page(tf, frame, CH_DNM).astype(float)
        clc = _page(tf, frame, CH_CLC).astype(float)
    Y, X = dnm.shape
    ir = (float(dnm.min()), float(dnm.max()))

    # nulove pozice: daleko od detekci pri VOLNEM prahu k=1.5
    yd, xd = detect_candidates(dnm, SIGMA_SLAVE, k=1.5, max_spots=100000)
    tree = cKDTree(np.c_[yd, xd])
    w = int(np.ceil(4 * SIGMA_SLAVE)) + 3
    cy = rng.integers(w, Y - w, 60000)
    cx = rng.integers(w, X - w, 60000)
    d, _ = tree.query(np.c_[cy, cx], k=1)
    keep = d > 4 * SIGMA_SLAVE
    cy, cx = cy[keep][:n_null], cx[keep][:n_null]

    # signalni pozice: detekce na clathrinu
    sy, sx = detect_candidates(clc, SIGMA_MASTER, k=5.0, max_spots=n_sig)

    # tau z autokorelace pozadi (1D integral; 2D integral je ~20 --
    # ktera hodnota je "ta prava" je modelova volba, viz README)
    hp = dnm - gaussian_filter(dnm, 3)
    m = np.abs(hp) < np.percentile(np.abs(hp), 90)
    n = hp.copy(); n[~m] = np.nan

    def corr(lag):
        A, B = n[:, :-lag].ravel(), n[:, lag:].ravel()
        k = np.isfinite(A) & np.isfinite(B)
        return float(np.corrcoef(A[k], B[k])[0, 1])

    tau = 1 + 2 * sum(max(corr(l), 0) for l in (1, 2, 3))

    def run(ys, xs, use_tau):
        hits = tot = 0
        for y, x in zip(ys, xs):
            r = fit_gaussian_2d_point(dnm, float(x), float(y), SIGMA_SLAVE,
                                      mode="Ac", i_range=ir)
            if not r["valid"]:
                continue
            npx, sA, sr = r["npx"], r["A_pstd"], r["sigma_r"]
            if use_tau:
                npx = npx / tau
                sA = sA * np.sqrt(tau)
            SE = sr / np.sqrt(2 * (npx - 1)) * kL
            df = (npx - 1) * (sA**2 + SE**2)**2 / (sA**4 + SE**4)
            sc = np.sqrt((sA**2 + SE**2) / npx)
            pv = float(st.cdf(-((r["A"] - sr * kL) / sc), df))
            hits += pv < 0.05
            tot += 1
        return hits, tot

    print(f"[FP+tau] frame={frame}, seed={seed}, tau(1D)={tau:.2f}")
    print(f"{'populace':>22}{'bez tau':>12}{'s tau':>12}")
    for lbl, ys, xs in (("cista nula", cy, cx), ("signal (CCP)", sy, sx)):
        h0, t0 = run(ys, xs, False)
        h1, t1 = run(ys, xs, True)
        print(f"{lbl:>22}{100*h0/t0:>10.2f} %{100*h1/t1:>10.2f} %"
              f"   ({t0} pozic)")
    print()


# ---------------------------------------------------------------- 2) bleaching

def bleaching():
    """Pokles amplitudy po kontrole delky a OBOUSTRANNE cenzury.

    Levou cenzuru (t0 == 0) je nutne vyloucit: drahy zacinajici v prvnim
    snimku jsou fragmenty uz existujicich jasnych struktur a nafukuji
    zdanlivy pokles (puvodni odhady -44 az -62 %).
    """
    tracks = []
    for p in sorted(MEASURED_DIR.glob("*.csv")):
        per = {}
        with open(p) as f:
            for r in csv.DictReader(f):
                a = float(r["dnm_A"])
                if not np.isfinite(a):
                    continue
                k = int(float(r["particle"]))
                t = int(float(r["frame"]))
                d = per.setdefault(k, [10**9, -1, -np.inf])
                d[0] = min(d[0], t); d[1] = max(d[1], t); d[2] = max(d[2], a)
        tracks += [(v[0], v[1] - v[0] + 1, v[2]) for v in per.values()]
    tr = np.array(tracks)
    t0, L, A = tr[:, 0], tr[:, 1], tr[:, 2]
    clean = (t0 > 0) & (t0 + L < 141)          # oboustranne necenzurovane

    print(f"[bleaching] {len(tr):,} drah, {clean.sum():,} oboustranne "
          "necenzurovanych")
    starts = [(1, 30), (30, 60), (60, 90), (90, 130)]
    print(f"{'delka':>8}" + "".join(f"{f'{a}-{b}':>11}" for a, b in starts)
          + f"{'zmena':>9}")
    for lo, hi in ((10, 15), (15, 25), (25, 40), (40, 70)):
        vals = []
        for a, b in starts:
            m = clean & (L >= lo) & (L < hi) & (t0 >= a) & (t0 < b)
            vals.append(np.median(A[m]) if m.sum() >= 40 else np.nan)
        row = f"{f'{lo}-{hi}':>8}" + "".join(
            f"{v:>11.0f}" if np.isfinite(v) else f"{'-':>11}" for v in vals)
        if np.isfinite(vals[0]) and np.isfinite(vals[-1]):
            row += f"{100*(vals[-1]/vals[0]-1):>8.0f} %"
        print(row)

    print("\npokles pozadi (median dnm_c, prvnich vs poslednich 15 snimku):")
    for p in sorted(MEASURED_DIR.glob("*.csv")):
        early, late = [], []
        with open(p) as f:
            for r in csv.DictReader(f):
                c = float(r["dnm_c"]); t = int(float(r["frame"]))
                if not np.isfinite(c):
                    continue
                (early if t < 15 else late if t >= 136 else []).append(c)
        if len(early) > 50 and len(late) > 50:
            k = re.search(r"_(\d+)-traj", p.name).group(1)
            print(f"  film {k:>3}: {100*(np.median(late)/np.median(early)-1):+6.1f} %")
    print()


# ---------------------------------------------------------------- 3) crossfilm

def crossfilm(frame=50):
    """Rozptyl mezi filmy: dynamin vs pozadi tehoz kanalu vs clathrin.

    Nejsilnejsi dukaz biologickeho puvodu je same-channel disociace:
    amplitudy ch2 kolisaji ~17x, pozadi TEHOZ kanalu jen ~2.3x. Clathrin
    je podpurny (a mirne zavisly na zvolenem snimku).
    """
    import tifffile
    tn = {re.search(r"_(\d+)-traj", p.name).group(1): p
          for p in sorted(MEASURED_DIR.glob("*.csv"))}
    mn = {re.search(r"_(\d+)_RR", p.name).group(1): p
          for p in sorted(MOVIE_DIR.glob("*.tif"))}
    keys = sorted(set(tn) & set(mn), key=int)

    D, BG, CL = [], [], []
    for k in keys:
        per = {}
        with open(tn[k]) as f:
            for r in csv.DictReader(f):
                a = float(r["dnm_A"])
                if np.isfinite(a):
                    i = int(float(r["particle"]))
                    per[i] = max(per.get(i, -np.inf), a)
        D.append(np.median(list(per.values())))
        with tifffile.TiffFile(mn[k]) as tf:
            dnm = _page(tf, frame, CH_DNM)
            clc = _page(tf, frame, CH_CLC)
        BG.append(float(np.median(dnm)))
        ys, xs = detect_candidates(clc, SIGMA_MASTER, k=5.0, max_spots=400)
        ir = (float(clc.min()), float(clc.max()))
        ca = [fit_gaussian_2d_point(clc, float(x), float(y), SIGMA_MASTER,
                                    mode="Ac", i_range=ir)["A"]
              for y, x in zip(ys, xs)]
        CL.append(float(np.nanmedian(ca)))

    D, BG, CL = map(np.array, (D, BG, CL))

    def cv(v):
        return 100 * v.std() / v.mean()

    print(f"[crossfilm] {len(keys)} filmu, clathrin na framu {frame}")
    print(f"{'velicina':>22}{'max/min':>10}{'CV':>9}")
    print(f"{'dynamin A (max/track)':>22}{D.max()/D.min():>10.2f}{cv(D):>8.1f}%")
    print(f"{'pozadi ch2':>22}{BG.max()/BG.min():>10.2f}{cv(BG):>8.1f}%")
    print(f"{'clathrin A':>22}{CL.max()/CL.min():>10.2f}{cv(CL):>8.1f}%")
    print(f"korelace dynamin~clathrin: r = {np.corrcoef(D, CL)[0,1]:+.3f} "
          f"(n = {len(keys)} -- nerozlisitelne od nuly)")
    print()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("fp", "all"):
        fp_and_tau()
    if which in ("bleach", "all"):
        bleaching()
    if which in ("crossfilm", "all"):
        crossfilm()
