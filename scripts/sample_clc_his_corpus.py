#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nasampluje klatrinovou intenzitu z noveho AVG datasetu na pozicich
KOLEGOVA korpusu (processed_v2, 47 923 drah vcetne tech, ktere nase
drivejsi trajektorie neobsahuji). Ucel: odvodit z klatrinu binarni
stitek produktivity pro experiment dynamin -> klatrin.

Vystup:
  Clathrin Analysis/his_corpus_clc/<film>-clc.csv.gz   per-frame hodnoty
  Clathrin Analysis/his_corpus_clc/per_track.csv       per-track souhrny

    python3 scripts/sample_clc_his_corpus.py
"""
from __future__ import annotations

import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_cme_corpus import fit_frame_means  # noqa: E402
from compute_boxmean import box_means  # noqa: E402
from cmepython import measure_movie  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "external", "Shape2Fate_Fake2Emulate",
                      "data", "dynamin_u2os", "processed_v2")
MOVIES = os.path.join(ROOT, "Clathrin Analysis")
OUT = os.path.join(ROOT, "Clathrin Analysis", "his_corpus_clc")
SIGMA = 1.6423   # kalibrace na AVG datasetu


def film_no(p):
    return int(re.search(r"_(\d+)-processed", os.path.basename(p)).group(1))


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    movies = {int(re.search(r"_(\d+)_AVG", os.path.basename(p)).group(1)): p
              for p in glob.glob(os.path.join(MOVIES, "*_AVG.tif"))}
    per_track = []
    for path in sorted(glob.glob(os.path.join(CORPUS, "*processed-trajectories.csv.gz"))):
        film = film_no(path)
        t0 = time.time()
        d = pd.read_csv(path, usecols=["particle", "x", "y", "frame"])
        d = d.sort_values(["particle", "frame"]).reset_index(drop=True)
        stack = tifffile.imread(movies[film])
        assert d.frame.max() < len(stack), (film, d.frame.max(), len(stack))
        means = stack.reshape(len(stack), -1).mean(axis=1, dtype=np.float64)
        fitted, info = fit_frame_means(means)
        frames = d.frame.to_numpy(int)
        box_raw = box_means(stack, d.y.to_numpy(float), d.x.to_numpy(float), frames)
        box_norm = box_raw / fitted[frames]

        coords = d[["frame", "y", "x"]].to_numpy(float)
        res = measure_movie(movies[film], coords, SIGMA, SIGMA,
                            slave_channel=0, master_channel=0,
                            validate=False, workers=0)
        out = pd.DataFrame({
            "particle": d.particle.astype(int), "frame": frames,
            "clc_A": res["A"], "clc_valid": res["valid"].astype(int),
            "box5_norm": box_norm,
        })
        out.to_csv(os.path.join(OUT, f"{film}-clc.csv.gz"), index=False,
                   float_format="%.6g", compression="gzip")

        A = np.where(res["valid"].astype(bool), res["A"], np.nan)
        g = out.assign(A=A).groupby("particle")
        pt = pd.DataFrame({
            "film": film,
            "n_frames": g.frame.size(),
            "amp_max": g.A.max(), "amp_mean": g.A.mean(),
            "box_max": g.box5_norm.max(), "box_mean": g.box5_norm.mean(),
        }).reset_index()
        per_track.append(pt)
        print(f"film {film:>3}: {len(d):>7,} radku  {pt.shape[0]:>6,} drah  "
              f"bleach {info['method']:<18} {time.time()-t0:.0f}s", flush=True)

    allt = pd.concat(per_track, ignore_index=True)
    allt.to_csv(os.path.join(OUT, "per_track.csv"), index=False, float_format="%.6g")
    print(f"\ncelkem {len(allt):,} drah  ->  {OUT}/per_track.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
