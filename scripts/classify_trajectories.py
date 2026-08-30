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

import argparse
import csv
import glob
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cmepython.classification import background_stats, classify_track  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Vychozi = prvni dataset; vse prepsatelne z CLI.
MEASURED_DIR = ROOT / "measured"
MOVIE_DIR = ROOT / "reconstructed registered"
SIGMA_SLAVE = 2.6424
SLAVE_CHANNEL = 2
MASTER_CHANNEL = 0

OUT_COLS = ["particle", "track_len", "n_detected", "thr_master",
            "significant_master", "n_above_bg", "thr_slave",
            "significant_slave", "max_A", "max_master_A", "max_si"]


def movie_number(name):
    m = re.search(r"_(\d+)(?:[-_]|$)", Path(name).stem)
    return m.group(1) if m else None


def load_measured(path, npx, si_col=None):
    """Nacte measured CSV do {particle: vektory} + {frame: (ys, xs)}.

    npx : pocet pixelu fitovaciho okna (odvozeny ze sigmy), pro
          rekonstrukci SE_sigma_r stejne jako v mereni.
    Pokud CSV obsahuje clc_A (master amplituda), ulozi se i max_master_A
    pro podminku amplitudoveho pomeru (:151-153).
    """
    per, byframe = {}, {}
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        has_master = "clc_A" in rd.fieldnames
        si_col = si_col if (si_col and si_col in rd.fieldnames) else \
            next((c for c in ("SI", "si", "shape_index", "cls") if c in rd.fieldnames), None)
        for r in rd:
            pid = int(float(r["particle"]))
            t = int(float(r["frame"]))
            d = per.setdefault(pid, {k: [] for k in
                                     ("A", "A_pstd", "sigma_r", "SE_sigma_r",
                                      "hval_Ar", "master_A", "si")})
            d["A"].append(float(r["dnm_A"]))
            d["A_pstd"].append(float(r["dnm_A_pstd"]))
            d["sigma_r"].append(float(r["dnm_sigma_r"]))
            d["SE_sigma_r"].append(float(r["dnm_sigma_r"]) / np.sqrt(2 * (npx - 1)))
            d["hval_Ar"].append(float(r["dnm_signif"]))
            d["master_A"].append(float(r["clc_A"]) if has_master else np.nan)
            try:
                d["si"].append(float(r[si_col]) if si_col else np.nan)
            except ValueError:
                d["si"].append(np.nan)
            byframe.setdefault(t, [[], []])
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measured", default=str(MEASURED_DIR))
    ap.add_argument("--movies", default=str(MOVIE_DIR))
    ap.add_argument("--masks", default=None,
                    help="slozka s maskami bunek (*_MASK.tif); bez ni se maska "
                         "spocita z max-projekce master kanalu (getCellMask)")
    ap.add_argument("--sigma-slave", type=float, default=SIGMA_SLAVE)
    ap.add_argument("--slave-channel", type=int, default=SLAVE_CHANNEL)
    ap.add_argument("--master-channel", type=int, default=MASTER_CHANNEL)
    ap.add_argument("--si-col", default=None)
    ap.add_argument("--out-suffix", default="-classified.csv")
    a = ap.parse_args(argv)

    npx = (2 * int(np.ceil(4 * a.sigma_slave)) + 1) ** 2
    mdir, vdir = Path(a.measured), Path(a.movies)
    masks = {}
    if a.masks:
        for p in Path(a.masks).glob("*.tif"):
            masks[movie_number(p)] = p

    pairs = []
    for p in sorted(mdir.glob("*-dynamin.csv")):
        k = movie_number(p)
        mv = sorted(x for x in vdir.glob("*.tif") if movie_number(x) == k)
        if len(mv) == 1:
            pairs.append((k, p, mv[0]))
        else:
            print(f"VAROVANI: {p.name} ma {len(mv)} odpovidajicich filmu -- preskakuji",
                  file=sys.stderr)
    print(f"{len(pairs)} filmu | sigma_slave={a.sigma_slave} npx={npx} | masky: "
          f"{'dodane (' + str(len(masks)) + ')' if masks else 'z max-projekce'}\n", flush=True)

    for k, mp, tif in pairs:
        t0 = time.time()
        per, byframe = load_measured(mp, npx, a.si_col)
        cellmask = None
        if k in masks:
            import tifffile
            cellmask = tifffile.imread(masks[k]) > 0
        stats = background_stats(tif, byframe, a.sigma_slave,
                                 slave_channel=a.slave_channel,
                                 master_channel=a.master_channel,
                                 cellmask=cellmask)
        res = {}
        for pid, d in per.items():
            mm = np.nanmax(d["master_A"]) if np.isfinite(d["master_A"]).any() else None
            r = classify_track(d["A"], d["A_pstd"], d["sigma_r"], d["SE_sigma_r"],
                               d["hval_Ar"], stats["bg95"], stats["p_detection"],
                               master_max_A=mm)
            r.pop("significant_vs_background")
            r["max_master_A"] = mm if mm is not None else np.nan
            r["max_si"] = float(np.nanmax(d["si"])) if np.isfinite(d["si"]).any() else np.nan
            res[pid] = r

        out = mp.with_name(mp.name.replace("-dynamin.csv", a.out_suffix))
        with open(out, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(OUT_COLS)
            for pid in sorted(res):
                r = res[pid]
                wr.writerow([pid, r["track_len"], r["n_detected"],
                             f"{r['thr_master']:.0f}", int(r["significant_master"]),
                             r["n_above_bg"], f"{r['thr_slave']:.0f}",
                             int(r["significant_slave"]), f"{r['max_A']:.2f}",
                             f"{r['max_master_A']:.2f}", f"{r['max_si']:.4f}"])

        sm = np.array([r["significant_master"] for r in res.values()])
        ss = np.array([r["significant_slave"] for r in res.values()])
        print(f"  film {k:>3}: {len(res):>6,} drah  bg95={stats['bg95']:>8.1f}"
              f"  pDet={stats['p_detection']:.4f}"
              f"  master+={100*sm.mean():>5.1f}%  slave+={100*ss.mean():>5.1f}%"
              f"  {time.time()-t0:>5.1f}s", flush=True)

    print(f"\nhotovo -> {mdir}/*{a.out_suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
