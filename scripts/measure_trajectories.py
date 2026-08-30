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

import argparse
import csv
import glob
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cmepython import measure_movie  # noqa: E402

# Vychozi hodnoty = prvni (rekonstruovany) dataset; vse jde prepsat z CLI.
TRAJ_DIR = "trajectories"
MOVIE_DIR = "reconstructed registered"
OUT_DIR = "measured"
SIGMA_SLAVE = 2.6424        # psf_calibration.json
SIGMA_MASTER = 1.4187
SLAVE_CHANNEL = 2
MASTER_CHANNEL = 0

NEW_COLS = ["dnm_A", "dnm_c", "dnm_A_pstd", "dnm_sigma_r", "dnm_pval",
            "dnm_signif", "dnm_x", "dnm_y", "dnm_valid", "track_len"]
MASTER_COLS = ["clc_A", "clc_c", "clc_A_pstd", "clc_pval", "clc_signif"]


def movie_number(name):
    """Cislo filmu z nazvu: prvni cislo za podtrzitkem (…_16-lr-trajectories,
    …_16_RR.tif, …_16_LRR.tif)."""
    m = re.search(r"_(\d+)(?:[-_]|$)", Path(name).stem)
    return m.group(1) if m else None


def pair_files(traj_dir, movie_dir):
    tn = {movie_number(p): p for p in sorted(glob.glob(f"{traj_dir}/*.csv"))}
    mn = {}
    for p in sorted(glob.glob(f"{movie_dir}/*.tif")):
        k = movie_number(p)
        if k in mn:
            print(f"VAROVANI: vic filmu s cislem {k}: {mn[k]}, {p}", file=sys.stderr)
        mn[k] = p
    return [(k, tn[k], mn[k]) for k in sorted(set(tn) & set(mn), key=int)]


def read_csv(path):
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        return rd.fieldnames, list(rd)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", default=TRAJ_DIR)
    ap.add_argument("--movie-dir", default=MOVIE_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--sigma-slave", type=float, default=SIGMA_SLAVE)
    ap.add_argument("--sigma-master", type=float, default=SIGMA_MASTER)
    ap.add_argument("--slave-channel", type=int, default=SLAVE_CHANNEL)
    ap.add_argument("--master-channel", type=int, default=MASTER_CHANNEL)
    ap.add_argument("--also-master", action="store_true",
                    help="zmerit i amplitudu master (clathrin) kanalu na tychz pozicich")
    ap.add_argument("--workers", type=int, default=0, help="0 = vsechna jadra, 1 = serialne")
    a = ap.parse_args(argv)

    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    pairs = pair_files(a.traj_dir, a.movie_dir)
    if not pairs:
        print("zadne pary trajektorie<->film", file=sys.stderr)
        return 1
    print(f"{len(pairs)} filmu | sigma_slave={a.sigma_slave}  sigma_master={a.sigma_master}  "
          f"slave ch={a.slave_channel}" + (f"  + master ch={a.master_channel}" if a.also_master else "")
          + "\n", flush=True)

    t_all = time.time()
    for k, tp, mp in pairs:
        t0 = time.time()
        fields, rows = read_csv(tp)
        coords = np.array([[float(r["frame"]), float(r["y"]), float(r["x"])]
                           for r in rows], dtype=float)
        pid = np.array([int(float(r["particle"])) for r in rows])
        counts = np.bincount(pid)
        track_len = counts[pid]

        res = measure_movie(mp, coords, a.sigma_slave, a.sigma_master,
                            slave_channel=a.slave_channel,
                            master_channel=a.master_channel,
                            workers=(None if a.workers == 1 else a.workers))
        resm = None
        if a.also_master:
            # master kanal: mereni "sam na sebe" -- sigma_slave := sigma_master
            resm = measure_movie(mp, coords, a.sigma_master, a.sigma_master,
                                 slave_channel=a.master_channel,
                                 master_channel=a.master_channel,
                                 validate=False,
                                 workers=(None if a.workers == 1 else a.workers))

        out = Path(a.out_dir) / f"{Path(tp).stem}-dynamin.csv"
        cols = fields + NEW_COLS + (MASTER_COLS if resm else [])
        with open(out, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(cols)
            for i, r in enumerate(rows):
                row = [r[c] for c in fields] + [
                    f"{res['A'][i]:.4f}", f"{res['c'][i]:.4f}",
                    f"{res['A_pstd'][i]:.4f}", f"{res['sigma_r'][i]:.4f}",
                    f"{res['pval_Ar'][i]:.6g}", int(res["hval_Ar"][i]),
                    f"{res['x'][i]:.4f}", f"{res['y'][i]:.4f}",
                    int(res["valid"][i]), int(track_len[i]),
                ]
                if resm:
                    row += [f"{resm['A'][i]:.4f}", f"{resm['c'][i]:.4f}",
                            f"{resm['A_pstd'][i]:.4f}", f"{resm['pval_Ar'][i]:.6g}",
                            int(resm["hval_Ar"][i])]
                wr.writerow(row)

        A = res["A"]
        ok = np.isfinite(A)
        print(f"  film {k:>3}: {len(rows):>7,} bodu  "
              f"A_med={np.nanmedian(A):>8.1f}  "
              f"signif={100*res['hval_Ar'].mean():>5.1f}%  "
              f"NaN={100*(~ok).mean():>4.1f}%  "
              + (f"clc_med={np.nanmedian(resm['A']):>8.1f}  " if resm else "")
              + f"{time.time()-t0:>5.1f}s", flush=True)

    print(f"\nhotovo za {(time.time()-t_all)/60:.1f} min -> {a.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
