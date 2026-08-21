# -*- coding: utf-8 -*-
# CMEpython -- Python port mereni intenzity dynaminu z cmeAnalysis.
# Copyright (C) 2026 Ruslan Guliev
#
# Odvozeno z cmeAnalysis (DanuserLab, UT Southwestern), GPL-3.0.
# Tento program je svobodny software: muzete jej sirit a upravovat podle
# podminek GNU General Public License verze 3 nebo pozdejsi. Sirenu je
# BEZ JAKEKOLI ZARUKY. Viz soubor LICENSE.
"""Klasifikace dynamin-pozitivnich drah pres cely dataset.

Pro kazdy film: maska bunky (max-projekce master kanalu) -> rozdeleni
pozadi bg95 a mira falesnych detekci p_detection (16 snimku) -> oba
Aguetovy testy na kazde draze. Vstup jsou vystupy measure_trajectories.py
(measured/*.csv), vystup measured/*-classified.csv, jeden radek na drahu.

Spusteni:  python3 scripts/classify_trajectories.py
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
from cmepython.classification import background_stats, classify_tracks  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MEASURED_DIR = ROOT / "measured"
MOVIE_DIR = ROOT / "reconstructed registered"

SIGMA_SLAVE = 2.6424     # psf_calibration.json, cely dataset
SLAVE_CHANNEL = 2
MASTER_CHANNEL = 0

# Pocet pixelu fitovaciho okna -- ODVOZENY ze sigmy, ne magicka konstanta.
# Plati pro detekcni cestu bez maskovani sousedu (mereni v measured/ presne
# tak vzniklo); pri zmene sigmy se prepocita sam.
NPX = (2 * int(np.ceil(4 * SIGMA_SLAVE)) + 1) ** 2

OUT_COLS = ["particle", "track_len", "n_detected", "thr_master",
            "significant_master", "n_above_bg", "thr_slave",
            "significant_slave", "max_A"]


def load_measured(path):
    """Nacte measured CSV do {particle: vektory} + {frame: (ys, xs)}."""
    per, byframe = {}, {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            pid = int(float(r["particle"]))
            t = int(float(r["frame"]))
            d = per.setdefault(pid, {k: [] for k in
                                     ("A", "A_pstd", "sigma_r", "SE_sigma_r",
                                      "hval_Ar")})
            d["A"].append(float(r["dnm_A"]))
            d["A_pstd"].append(float(r["dnm_A_pstd"]))
            d["sigma_r"].append(float(r["dnm_sigma_r"]))
            # SE_sigma_r v CSV neni -- dopocita se stejne jako v mereni:
            # SE = sigma_r / sqrt(2*(NPX-1)), kde NPX je odvozene z
            # SIGMA_SLAVE (viz vyse). Klasifikace pak npx z pomeru zpetne
            # rekonstruuje presne jako MATLAB.
            d["SE_sigma_r"].append(float(r["dnm_sigma_r"])
                                   / np.sqrt(2 * (NPX - 1)))
            d["hval_Ar"].append(float(r["dnm_signif"]))
            byframe.setdefault(t, [[], []])
            # parsovat a testovat konecnost -- ne porovnavat retezec "nan",
            # ktery by selhal na CSV z jineho zapisovace ("NaN", prazdno)
            try:
                dy = float(r["dnm_y"]); dx = float(r["dnm_x"])
            except ValueError:
                dy = dx = float("nan")
            byframe[t][0].append(dy if np.isfinite(dy) else float(r["y"]))
            byframe[t][1].append(dx if np.isfinite(dx) else float(r["x"]))
    for d in per.values():
        for k in d:
            d[k] = np.asarray(d[k], dtype=float)
    byframe = {t: (np.asarray(v[0]), np.asarray(v[1]))
               for t, v in byframe.items()}
    return per, byframe


def main():
    pairs = []
    for p in sorted(MEASURED_DIR.glob("*-dynamin.csv")):
        k = re.search(r"_(\d+)-traj", p.name).group(1)
        mv = sorted(MOVIE_DIR.glob(f"*_{k}_RR.tif"))
        if len(mv) == 1:
            pairs.append((k, p, mv[0]))
        else:
            print(f"VAROVANI: {p.name} ma {len(mv)} odpovidajicich filmu"
                  " -- preskakuji", file=sys.stderr)
    print(f"{len(pairs)} filmu | sigma_slave={SIGMA_SLAVE}\n", flush=True)

    summary = []
    for k, mp, tif in pairs:
        t0 = time.time()
        per, byframe = load_measured(mp)
        stats = background_stats(tif, byframe, SIGMA_SLAVE,
                                 slave_channel=SLAVE_CHANNEL,
                                 master_channel=MASTER_CHANNEL)
        res = classify_tracks(per, stats["bg95"], stats["p_detection"])

        out = mp.with_name(mp.name.replace("-dynamin.csv", "-classified.csv"))
        with open(out, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(OUT_COLS)
            for pid in sorted(res):
                r = res[pid]
                wr.writerow([pid, r["track_len"], r["n_detected"],
                             f"{r['thr_master']:.0f}",
                             int(r["significant_master"]),
                             r["n_above_bg"], f"{r['thr_slave']:.0f}",
                             int(r["significant_slave"]),
                             f"{r['max_A']:.2f}"])

        lens = np.array([r["track_len"] for r in res.values()])
        sm = np.array([r["significant_master"] for r in res.values()])
        ss = np.array([r["significant_slave"] for r in res.values()])
        long = lens >= 5                       # konvence Cutoff_f=5
        summary.append((k, len(res), stats["bg95"], stats["p_detection"],
                        100 * sm[long].mean(), 100 * ss[long].mean()))
        print(f"  film {k:>3}: {len(res):>6,} drah  bg95={stats['bg95']:>8.1f}"
              f"  pDet={stats['p_detection']:.4f}"
              f"  master+={100*sm[long].mean():>5.1f}%"
              f"  slave+={100*ss[long].mean():>5.1f}%  (drahy >=5 sn.)"
              f"  {time.time()-t0:>5.1f}s", flush=True)

    print("\nhotovo -> measured/*-classified.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
