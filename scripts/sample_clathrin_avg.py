#!/usr/bin/env python3
"""Krok 1 klatrinoveho experimentu: nasamplovani klatrinove intenzity
z NOVEHO jednokanaloveho datasetu (*_AVG.tif, prumer 9 raw SIM snimku,
neprekalovane intenzity) na pozicich STEJNYCH trajektorii jako dosud.

Dve metody na kazdy radek trajektorie:
  A) cmeAnalysis: PSF fit 2D gaussianu -> amplituda clc_A nad lokalnim
     pozadim clc_c (jednotky kamery), per-frame test vyznamnosti.
  B) box-mean (metoda Shape2Fate): prumer 5x5 px okenka na pozici jamky
     (box5_raw), vydeleny biexponencialnim fitem prumeru celych snimku
     (box5_norm; kompenzace bleachingu, 1 = prumer bunky).

Kontrola zarovnani: stary dvoukanalovy dataset je per-film linearni
skalou tychz dat -- pro kazdy film se fituje old_ch0 = a*new + b na
vzorku pozic a reportuje se a, b, R^2 (R^2 ~ 1 potvrzuje shodu souradnic).

Vystup: <out-dir>/<traj-stem>-clathrin-avg.csv per film + manifest.json
+ README.md (samostatny sdilitelny balicek pro kolegy).

    python3 scripts/sample_clathrin_avg.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_cme_corpus import fit_frame_means, film_number, git_hash  # noqa: E402
from compute_boxmean import box_means  # noqa: E402
from cmepython import measure_movie  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MEAS_COLS = ["clc_A", "clc_c", "clc_A_pstd", "clc_sigma_r", "clc_pval",
             "clc_signif", "clc_x", "clc_y", "clc_valid"]


def load_sigma(path: str) -> float:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    ch = sorted(d)[0]
    return float(d[ch].get("sigma_clamped", d[ch]["sigma"]))


def affine_check(old_movie: str, stack_new: np.ndarray, df: pd.DataFrame,
                 rng: np.random.Generator, n: int = 2000) -> dict:
    """old_ch0 = a*new + b na vzorku pozic (box 5x5). R^2 ~ 1 = souradnice sedi."""
    sub = df.sample(n=min(n, len(df)), random_state=int(rng.integers(1 << 31)))
    img = tifffile.imread(old_movie)
    old = img[:, 0] if img.ndim == 4 else img
    a_new = box_means(stack_new, sub.y.to_numpy(float), sub.x.to_numpy(float),
                      sub.frame.to_numpy(int))
    a_old = box_means(old.astype(np.float32), sub.y.to_numpy(float),
                      sub.x.to_numpy(float), sub.frame.to_numpy(int))
    m = np.isfinite(a_new) & np.isfinite(a_old)
    a, b = np.polyfit(a_new[m], a_old[m], 1)
    r2 = float(np.corrcoef(a_new[m], a_old[m])[0, 1] ** 2)
    return {"a": float(a), "b": float(b), "r2": r2, "n": int(m.sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", default=os.path.join(ROOT, "lr registered", "trajectories"))
    ap.add_argument("--movies", default=os.path.join(ROOT, "Clathrin Analysis"))
    ap.add_argument("--old-movies", default=os.path.join(ROOT, "lr registered"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "Clathrin Analysis", "sampled"))
    ap.add_argument("--calibration", default=os.path.join(ROOT, "Clathrin Analysis", "psf_calibration_avg.json"))
    ap.add_argument("--sigma", type=float, default=None, help="prepsat sigma z kalibrace")
    ap.add_argument("--workers", type=int, default=0, help="procesy uvnitr measure_movie (0 = vsechna jadra)")
    args = ap.parse_args()

    sigma = args.sigma if args.sigma is not None else load_sigma(args.calibration)
    trajs = {film_number(p): p for p in glob.glob(os.path.join(args.traj_dir, "*.csv"))}
    movies = {film_number(p): p for p in glob.glob(os.path.join(args.movies, "*.tif"))}
    old_movies = {film_number(p): p for p in glob.glob(os.path.join(args.old_movies, "*.tif"))}
    films = sorted(set(trajs) & set(movies))
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{len(films)} filmu | sigma = {sigma:.4f} px | out = {args.out_dir}\n", flush=True)

    rng = np.random.default_rng(0)
    manifest = {"built": _dt.datetime.now().isoformat(timespec="seconds"),
                "cmepython_git": git_hash(), "sigma_px": sigma,
                "traj_dir": args.traj_dir, "movie_dir": args.movies, "films": {}}
    for f in films:
        t0 = time.time()
        df = pd.read_csv(trajs[f])
        coords = df[["frame", "y", "x"]].to_numpy(float)
        counts = df.groupby("particle")["frame"].transform("size")

        res = measure_movie(movies[f], coords, sigma, sigma,
                            slave_channel=0, master_channel=0, validate=False,
                            workers=(None if args.workers == 1 else args.workers))

        stack = tifffile.imread(movies[f])          # (T, Y, X) float32
        means = stack.reshape(len(stack), -1).mean(axis=1, dtype=np.float64)
        fitted, info = fit_frame_means(means)
        frames = df.frame.to_numpy(int)
        box_raw = box_means(stack, df.y.to_numpy(float), df.x.to_numpy(float), frames)
        box_norm = box_raw / fitted[frames]

        aff = affine_check(old_movies[f], stack, df, rng) if f in old_movies else None

        out = df.copy()
        out["track_len"] = counts.to_numpy(int)
        out["clc_A"] = res["A"]; out["clc_c"] = res["c"]
        out["clc_A_pstd"] = res["A_pstd"]; out["clc_sigma_r"] = res["sigma_r"]
        out["clc_pval"] = res["pval_Ar"]; out["clc_signif"] = res["hval_Ar"].astype(int)
        out["clc_x"] = res["x"]; out["clc_y"] = res["y"]
        out["clc_valid"] = res["valid"].astype(int)
        out["box5_raw"] = box_raw
        out["frame_mean_fit"] = fitted[frames]
        out["box5_norm"] = box_norm

        stem = os.path.splitext(os.path.basename(trajs[f]))[0]
        out_path = os.path.join(args.out_dir, f"{stem}-clathrin-avg.csv")
        out.to_csv(out_path, index=False, float_format="%.6g")

        v = res["valid"].astype(bool)
        manifest["films"][str(f)] = {
            "n_rows": int(len(out)), "n_tracks": int(out.particle.nunique()),
            "n_invalid": int((~v).sum()), "bleach_fit": info["method"],
            "clc_A_median": float(np.nanmedian(res["A"][v])),
            "box5_norm_median": float(np.nanmedian(box_norm[np.isfinite(box_norm)])),
            "affine_vs_old": aff}
        a_txt = (f"  old=a*new+b: a={aff['a']:.2f} b={aff['b']:.0f} R2={aff['r2']:.5f}"
                 if aff else "")
        print(f"film {f:>3}: {len(out):>7,} radku  {out.particle.nunique():>6,} drah  "
              f"neplatnych {(~v).sum():>3}  bleach {info['method']:<18}"
              f"{a_txt}  {time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(readme(manifest))
    print(f"\nhotovo -> {args.out_dir}")
    return 0


def readme(m: dict) -> str:
    films = m["films"]
    return f"""# Clathrin intensity sampled from the AVG dataset

Built {m['built']} by `scripts/sample_clathrin_avg.py` (CMEpython {m['cmepython_git']}).
Source movies: single-channel clathrin, one frame = average of 9 raw SIM frames,
intensities NOT rescaled (raw camera units). Positions: the same CCP trajectories
as in all previous experiments (`{os.path.basename(m['traj_dir'])}`), coordinates
transfer directly (per-film affine check against the old registered dataset gives
R^2 ~ 1, see `manifest.json`).

One CSV per film: original trajectory columns + the sampled intensities.

## Method A -- cmeAnalysis (PSF fit)

2D Gaussian fit at each track position (sigma = {m['sigma_px']:.4f} px, calibrated
on this dataset): amplitude above local background. Two-step fit as in
cmeAnalysis runDetection: fixed position first, then free position accepted if
it moves < 3 sigma and improves the amplitude.

| column | meaning |
|---|---|
| clc_A | amplitude above local background (camera units) |
| clc_c | fitted local background (camera units) |
| clc_A_pstd | amplitude uncertainty |
| clc_sigma_r | residual std of the fit |
| clc_pval, clc_signif | per-frame test A > k*sigma_r (alpha 0.05) |
| clc_x, clc_y | fitted position |
| clc_valid | 0 = too close to the image border (values NaN) |

## Method B -- box mean (the Shape2Fate readout)

Mean of the 5x5 px box centred at the rounded track position, and the same value
divided by a biexponential fit of the whole-frame means of the film
(a*exp(-t/tau1)+b*exp(-t/tau2)+c) -- the bleaching compensation used by the
original dynamin readout. After division 1 = the cell average at that time.

| column | meaning |
|---|---|
| box5_raw | 5x5 box mean, camera units |
| frame_mean_fit | fitted whole-frame mean used as divisor |
| box5_norm | box5_raw / frame_mean_fit (dimensionless) |

box5_raw includes the local background (camera offset ~96 counts) and any diffuse
or neighbouring signal in the box; nothing is subtracted. NaN outside the image.

## Films

{len(films)} films, {sum(f['n_rows'] for f in films.values()):,} rows,
{sum(f['n_tracks'] for f in films.values()):,} tracks. Per-film details
(bleach-fit method, affine scale of the old dataset, medians): `manifest.json`.

Internal delivery for the collaboration; ask before sharing outside the project.
"""


if __name__ == "__main__":
    sys.exit(main())
