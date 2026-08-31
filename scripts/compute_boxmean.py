#!/usr/bin/env python3
"""Box-mean 5x5 readout dynaminu na pozicich dodanych drah (replika Matyasova
mereni na nasich datech): prumer 5x5 pixelu dynaminoveho kanalu, vydeleny
biexponencialnim fitem prumeru snimku (per-cell normalizace, ~1 = prumer bunky).

Vystup: <measured>/<film>-boxmean.csv  (particle, frame, box5_norm)
Slouzi protokolu ukolu B k porovnani readoutu cmeAnalysis vs. box-mean
na tychz drahach a stejnem operacnim bode.

    python3 scripts/compute_boxmean.py            # vychozi lr cesty
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from build_cme_corpus import dynamin_frame_means, fit_frame_means, film_number  # noqa: E402


def box_means(stack: np.ndarray, ys: np.ndarray, xs: np.ndarray, frames: np.ndarray,
              half: int = 2) -> np.ndarray:
    """Prumer (2*half+1)^2 okenka kolem zaokrouhlene pozice; NaN mimo obraz."""
    T, H, W = stack.shape
    out = np.full(len(ys), np.nan)
    yi = np.rint(ys).astype(int)
    xi = np.rint(xs).astype(int)
    ok = (frames >= 0) & (frames < T) & (yi >= half) & (yi < H - half) & \
         (xi >= half) & (xi < W - half)
    for t in np.unique(frames[ok]):
        sel = ok & (frames == t)
        img = stack[t]
        # kumulativni soucty pro rychle okenko
        cs = np.cumsum(np.cumsum(img.astype(np.float64), axis=0), axis=1)
        cs = np.pad(cs, ((1, 0), (1, 0)))
        y0, y1 = yi[sel] - half, yi[sel] + half + 1
        x0, x1 = xi[sel] - half, xi[sel] + half + 1
        s = cs[y1, x1] - cs[y0, x1] - cs[y1, x0] + cs[y0, x0]
        out[sel] = s / (2 * half + 1) ** 2
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measured", default=os.path.join(ROOT, "lr registered", "measured"))
    ap.add_argument("--movies", default=os.path.join(ROOT, "lr registered"))
    ap.add_argument("--slave-channel", type=int, default=1)
    ap.add_argument("--half", type=int, default=2, help="polovina okna (2 -> 5x5)")
    args = ap.parse_args()

    movies = {film_number(p): p for p in glob.glob(os.path.join(args.movies, "*.tif"))}
    files = sorted(glob.glob(os.path.join(args.measured, "*-lr-trajectories-dynamin.csv")))
    if not files:
        sys.exit(f"zadne *-lr-trajectories-dynamin.csv v {args.measured}")

    import tifffile
    for path in files:
        t0 = time.time()
        film = film_number(os.path.basename(path).split("-lr-trajectories")[0])
        df = pd.read_csv(path, usecols=["particle", "x", "y", "frame"])
        img = tifffile.imread(movies[film])
        stack = img[:, args.slave_channel] if img.ndim == 4 else img
        means = stack.reshape(len(stack), -1).mean(axis=1, dtype=np.float64)
        fitted, info = fit_frame_means(means)
        raw = box_means(stack, df["y"].to_numpy(float), df["x"].to_numpy(float),
                        df["frame"].to_numpy(int), half=args.half)
        norm = raw / fitted[df["frame"].to_numpy(int)]
        out = pd.DataFrame({"particle": df["particle"].astype(int),
                            "frame": df["frame"].astype(int),
                            "box5_norm": norm})
        dst = path.replace("-lr-trajectories-dynamin.csv", "-boxmean.csv")
        out.to_csv(dst, index=False)
        print(f"film {film:>3}: {len(out):>7,} radku  fit {info['method']:<14} "
              f"NaN {np.isnan(norm).sum():>3}  {time.time()-t0:5.1f}s", flush=True)
    print("\nhotovo -> *-boxmean.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
