# -*- coding: utf-8 -*-
# CMEpython -- Python port mereni intenzity dynaminu z cmeAnalysis.
# Copyright (C) 2026 Ruslan Guliev
#
# Odvozeno z cmeAnalysis (DanuserLab, UT Southwestern), GPL-3.0.
# Tento program je svobodny software: muzete jej sirit a upravovat podle
# podminek GNU General Public License verze 3 nebo pozdejsi. Sirenu je
# BEZ JAKEKOLI ZARUKY. Viz soubor LICENSE.
"""Kalibrace sigma PSF pres CELY dataset -- zrcadli runDetection.m:66-92.

cmeAnalysis vzorkuje ~40 snimku rozprostrenych pres VSECHNY filmy podminky
(`nf = round(40/nd)` na snimek na film), slouci je a odhadne jednu sigmu
na kanal pro cely dataset.

Spousteni:  python3 calibrate_dataset.py "reconstructed registered"
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cmepython.psf_calibration import estimate_psf_sigma, apply_sigma_clamp  # noqa: E402
from cmepython.measure import movie_layout  # noqa: E402

TOTAL_FRAMES = 40          # runDetection.m:66 -- celkovy rozpocet snimku
CHANNELS = (0, 1, 2)


def frame_indices(n_movies, movie_length):
    """runDetection.m:66,72 -- nf snimku na film, rovnomerne rozprostrenych."""
    nf = max(1, round(TOTAL_FRAMES / n_movies))
    return np.unique(np.round(np.linspace(0, movie_length - 1, nf)).astype(int))


def load_frames(args):
    path, channel, n_movies = args
    with tifffile.TiffFile(path) as tf:
        T, C, Y, X, page = movie_layout(tf)
        idx = frame_indices(n_movies, T)
        return [tf.pages[page(t, channel)].asarray().astype(np.float64) for t in idx]


def main(folder):
    paths = sorted(Path(folder).glob("*.tif"))
    if not paths:
        print(f"zadne .tif v {folder}"); return 1
    nd = len(paths)
    nf = max(1, round(TOTAL_FRAMES / nd))
    print(f"dataset: {nd} filmu, {nf} snimek/film -> ~{nd * nf} snimku na kanal\n")

    results = {}
    for ch in CHANNELS:
        print(f"--- kanal {ch} ---", flush=True)
        with ProcessPoolExecutor() as ex:
            batches = list(ex.map(load_frames, [(str(p), ch, nd) for p in paths]))
        frames = [f for b in batches for f in b]
        print(f"  nacteno {len(frames)} snimku, meri se...", flush=True)

        sigma, detail = estimate_psf_sigma(frames, return_detail=True)
        clamped, fired = apply_sigma_clamp(sigma)
        detail["sigma_clamped"] = clamped
        detail["clamp_fired"] = fired
        results[ch] = detail

        print(f"  spotu:        {detail['n_spots']}")
        print(f"  median sigma: {detail['median']:.3f} px  "
              f"(IQR {detail['iqr'][0]:.2f}-{detail['iqr'][1]:.2f})")
        print(f"  GMM:          {detail['n_components']} komponent (BIC)")
        for c in detail["components"]:
            mark = "  <== vybrana" if c["selected"] else ""
            print(f"     mu={c['mu']:.3f}  sd={c['sigma']:.3f}  "
                  f"w={c['weight']:.3f}  vrchol={c['peak']:.2f}{mark}")
        print(f"  SIGMA:        {sigma:.4f} px"
              f"{'  -> CLAMP na 1.1' if fired else '  (clamp neaktivni)'}\n", flush=True)

    out = Path("psf_calibration.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"ulozeno: {out}")

    print("\n=== SHRNUTI ===")
    print(f"{'kanal':>7}{'sigma':>10}{'po clampu':>12}{'spotu':>9}{'komponent':>11}")
    for ch, d in results.items():
        print(f"{ch:>7}{d['sigma']:>10.4f}{d['sigma_clamped']:>12.4f}"
              f"{d['n_spots']:>9}{d['n_components']:>11}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "reconstructed registered"))
