# -*- coding: utf-8 -*-
# CMEpython -- Python port mereni intenzity dynaminu z cmeAnalysis.
# Copyright (C) 2026 Ruslan Guliev
#
# Odvozeno z cmeAnalysis (DanuserLab, UT Southwestern), GPL-3.0.
# Tento program je svobodny software: muzete jej sirit a upravovat podle
# podminek GNU General Public License verze 3 nebo pozdejsi. Sirenu je
# BEZ JAKEKOLI ZARUKY. Viz soubor LICENSE.
"""Porovna Python port proti skutecne MATLAB MEX binarce fitGaussian2D.

Dvoukrokovy postup:

  1) MATLAB vyexportuje okna z realnych dat plus vysledky MEX fitu:
         matlab -batch export_fits          (viz matlab/export_fits.m)
     Vznikne matlab/mex_reference.mat.

  2) Tento skript pusti Python port na TATAZ okna se STEJNOU inicializaci
     a porovna kazdou vracenou velicinu.

Zamerne obchazi fitGaussians2D.m a vola MEX primo -- izoluje tim jedinou
komponentu, kterou nelze precist ze zdrojaku, misto aby ji hledal skrz
nekolik vrstev obalu.

Spusteni:  python3 scripts/validate_against_matlab.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cmepython.slave_intensity import fit_gaussian_2d_point  # noqa: E402

REF = Path(__file__).resolve().parent.parent / "matlab" / "mex_reference.mat"

# Prah shody. Vse krome dx/dy sedi na ~1e-8; dx/dy jsou citlivejsi, protoze
# poloha je v nelinearnim fitu nejhure podmineny parametr.
TOL = {"dx": 1e-5, "dy": 1e-5}
TOL_DEFAULT = 1e-6


def main():
    if not REF.exists():
        print(f"chybi {REF}\nspust nejdriv: matlab -batch export_fits")
        return 1

    m = loadmat(REF)
    wins = m["wins"]
    init = m["init"]
    w4 = int(m["w4"][0, 0])
    sigma = float(m["SIGMA"][0, 0])
    i_range = tuple(float(v) for v in m["iRange"].ravel())
    n = wins.shape[2]
    ctr = float(w4)

    print(f"{n} oken z realnych dat | w4={w4} | sigma={sigma:.4f} | npx={(2*w4+1)**2}\n")

    py = {k: [] for k in ("A", "c", "A_pstd", "RSS", "sigma_r",
                          "A_xy", "dx", "dy", "A_pstd_xy")}
    for k in range(n):
        win = wins[:, :, k]
        r = fit_gaussian_2d_point(win, ctr, ctr, sigma, mode="Ac",
                                  A_init=float(init[k, 2]), c_init=float(init[k, 4]),
                                  i_range=i_range)
        py["A"].append(r["A"]); py["c"].append(r["c"])
        py["A_pstd"].append(r["A_pstd"]); py["RSS"].append(r["RSS"])
        py["sigma_r"].append(r["sigma_r"])

        r2 = fit_gaussian_2d_point(win, ctr, ctr, sigma, mode="xyAc",
                                   A_init=float(r["A"]), c_init=float(r["c"]),
                                   i_range=i_range)
        py["A_xy"].append(r2["A"]); py["A_pstd_xy"].append(r2["A_pstd"])
        py["dx"].append(r2["x"] - ctr); py["dy"].append(r2["y"] - ctr)

    pairs = [
        ("A ('Ac')",       m["res_Ac"][:, 2],      py["A"]),
        ("c ('Ac')",       m["res_Ac"][:, 4],      py["c"]),
        ("A_pstd ('Ac')",  m["res_Ac_std"][:, 2],  py["A_pstd"]),
        ("RSS",            m["rss_Ac"].ravel(),    py["RSS"]),
        ("sigma_r",        m["std_Ac"].ravel(),    py["sigma_r"]),
        ("A ('xyAc')",     m["res_xy"][:, 2],      py["A_xy"]),
        ("A_pstd ('xyAc')", m["res_xy_std"][:, 2], py["A_pstd_xy"]),
        ("dx",             m["res_xy"][:, 0],      py["dx"]),
        ("dy",             m["res_xy"][:, 1],      py["dy"]),
    ]

    print(f"{'velicina':>17}{'n':>7}{'median rel':>14}{'max rel':>13}{'korelace':>11}  verdikt")
    print("-" * 76)
    failed = []
    for name, ml, p in pairs:
        ml = np.asarray(ml, dtype=float).ravel()
        p = np.asarray(p, dtype=float).ravel()
        ok = np.isfinite(ml) & np.isfinite(p)
        if ok.sum() < 10:
            print(f"{name:>17}{ok.sum():>7}   -- prilis malo platnych hodnot --")
            continue
        rel = np.abs(ml[ok] - p[ok]) / np.maximum(np.abs(ml[ok]), 1e-12)
        med, mx = float(np.median(rel)), float(np.max(rel))
        corr = float(np.corrcoef(ml[ok], p[ok])[0, 1])
        tol = TOL.get(name, TOL_DEFAULT)
        good = med < tol
        if not good:
            failed.append(name)
        print(f"{name:>17}{ok.sum():>7}{med:>14.2e}{mx:>13.2e}{corr:>11.6f}"
              f"  {'OK' if good else 'NESEDI'}")

    if failed:
        print(f"\nNESHODA: {', '.join(failed)}")
        return 1
    print("\nVse sedi na urovni strojove presnosti.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
