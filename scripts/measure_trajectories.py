# -*- coding: utf-8 -*-
# CMEpython -- Python port mereni intenzity dynaminu z cmeAnalysis.
# Copyright (C) 2026 Ruslan Guliev
#
# Odvozeno z cmeAnalysis (DanuserLab, UT Southwestern), GPL-3.0.
# Tento program je svobodny software: muzete jej sirit a upravovat podle
# podminek GNU General Public License verze 3 nebo pozdejsi. Sirenu je
# BEZ JAKEKOLI ZARUKY. Viz soubor LICENSE.
"""Zmeri intenzitu dynaminu na pozicich z trajectories/*.csv.

Ke kazdemu radku vstupniho CSV pripoji hodnoty z portu cmeAnalysis
(runDetection.m:180-190). Puvodni sloupce zustavaji nedotcene.

Pridane sloupce:
    dnm_A         amplituda gaussovky nad lokalnim pozadim  <- hledana intenzita
    dnm_c         fitovane lokalni pozadi
    dnm_A_pstd    smerodatna odchylka amplitudy
    dnm_sigma_r   sd rezidui
    dnm_pval      p-hodnota testu proti sumu
    dnm_signif    prosel test na hladine alpha
    dnm_x, dnm_y  pozice po pripadne lokalizaci ('xyAc')
    dnm_valid     fit probehl (False = okraj obrazu -> NaN)
    track_len     delka trajektorie ve snimcich (pro pozdejsi filtrovani)

Spusteni:  python3 measure_trajectories.py
"""

from __future__ import annotations

import csv
import glob
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cmepython import measure_movie  # noqa: E402

TRAJ_DIR = "trajectories"
MOVIE_DIR = "reconstructed registered"
OUT_DIR = "measured"

# psf_calibration.json, odhad pres vsech 23 filmu podminky
SIGMA_SLAVE = 2.6424        # ch2 -- zprumerovany raw dynamin, 2x zvetseny
SIGMA_MASTER = 1.4187       # ch0 -- rekonstruovany clathrin
SLAVE_CHANNEL = 2

NEW_COLS = ["dnm_A", "dnm_c", "dnm_A_pstd", "dnm_sigma_r", "dnm_pval",
            "dnm_signif", "dnm_x", "dnm_y", "dnm_valid", "track_len"]


def pair_files():
    tn = {re.search(r"_(\d+)-traj", p).group(1): p
          for p in sorted(glob.glob(f"{TRAJ_DIR}/*.csv"))}
    mn = {re.search(r"_(\d+)_RR", p).group(1): p
          for p in sorted(glob.glob(f"{MOVIE_DIR}/*.tif"))}
    return [(k, tn[k], mn[k]) for k in sorted(set(tn) & set(mn), key=int)]


def read_csv(path):
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        return rd.fieldnames, list(rd)


def main():
    Path(OUT_DIR).mkdir(exist_ok=True)
    pairs = pair_files()
    print(f"{len(pairs)} filmu | sigma_slave={SIGMA_SLAVE}  sigma_master={SIGMA_MASTER}  "
          f"kanal={SLAVE_CHANNEL}\n", flush=True)

    t_all = time.time()
    for k, tp, mp in pairs:
        t0 = time.time()
        fields, rows = read_csv(tp)
        coords = np.array([[float(r["frame"]), float(r["y"]), float(r["x"])]
                           for r in rows], dtype=float)

        # delka trajektorie -- pro pozdejsi filtrovani (cmeAnalysis pouziva
        # Cutoff_f = 5 snimku, runSlaveChannelClassification.m)
        pid = np.array([int(float(r["particle"])) for r in rows])
        counts = np.bincount(pid)
        track_len = counts[pid]

        res = measure_movie(mp, coords, SIGMA_SLAVE, SIGMA_MASTER,
                            slave_channel=SLAVE_CHANNEL, workers=0)

        out = Path(OUT_DIR) / f"{Path(tp).stem}-dynamin.csv"
        with open(out, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(fields + NEW_COLS)
            for i, r in enumerate(rows):
                wr.writerow([r[c] for c in fields] + [
                    f"{res['A'][i]:.4f}", f"{res['c'][i]:.4f}",
                    f"{res['A_pstd'][i]:.4f}", f"{res['sigma_r'][i]:.4f}",
                    f"{res['pval_Ar'][i]:.6g}", int(res["hval_Ar"][i]),
                    f"{res['x'][i]:.4f}", f"{res['y'][i]:.4f}",
                    int(res["valid"][i]), int(track_len[i]),
                ])

        A = res["A"]
        ok = np.isfinite(A)
        dt = time.time() - t0
        print(f"  film {k:>3}: {len(rows):>7,} bodu  "
              f"A_med={np.nanmedian(A):>8.1f}  "
              f"signif={100*res['hval_Ar'].mean():>5.1f}%  "
              f"NaN={100*(~ok).mean():>4.1f}%  "
              f"{dt:>5.1f}s", flush=True)

    print(f"\nhotovo za {(time.time()-t_all)/60:.1f} min -> {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
