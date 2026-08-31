#!/usr/bin/env python3
"""Sestavi korpus pro Matyasuv trenovaci kod (Shape2Fate, vetev
release/dynamin-confusion-v1) z nasich cmeAnalysis mereni.

Vstup:  lr registered/measured/*-lr-trajectories-dynamin.csv  (vystup
        scripts/measure_trajectories.py: puvodni sloupce Shape2Fate + dnm_* / clc_*)
Vystup: <out-root>/cme_raw/         amplituda cmeAnalysis v surovych jednotkach kamery
        <out-root>/cme_normalized/  amplituda / biexponencialni fit prumeru snimku
                                     (stejna normalizace, jakou pouziva Matyasuv readout)

Kazda varianta je adresar ve formatu, ktery cte jeho `load_v2_frame()`:
  *_<film>-processed-trajectories.csv se sloupci
  particle,x,y,cls,frame,cluster,closest_ccp_dist,second_closest_ccp_dist,
  dynamin_intensity_5x5,dynamin_intensity_3x3,dynamin_intensity_1x1  (+ nase dnm_*/clc_*)
  thresholds_q90.json  (90. percentil per-frame excesu pres cely korpus, per readout)

Konvence jeho kodu: excess = dynamin_intensity - 1.  Proto zapisujeme
dynamin_intensity_* = 1 + amplituda, aby excess == amplituda cmeAnalysis.  Vsechny
tri "readouty" (5x5/3x3/1x1) nesou stejnou hodnotu -- cmeAnalysis nema velikost
boxu, amplituda je jedna.  Sloupce dnm_* zustavaji v puvodnich jednotkach.

Vzdalenosti k nejblizsi / druhe nejblizsi CCP se pocitaji z dodanych trajektorii
(v kazdem snimku pres vsechny dodane drahy), tj. z podmnoziny, ne z plne detekce.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NUM_RE = re.compile(r"_(\d+)(?:[-_]|$)")
READOUT_COLS = ("dynamin_intensity_5x5", "dynamin_intensity_3x3", "dynamin_intensity_1x1")
BASELINE = 1.0
FAR = 9999.0  # vzdalenost, kdyz ve snimku neni dost jinych drah


def film_number(name: str) -> int | None:
    m = _NUM_RE.search(os.path.basename(name))
    return int(m.group(1)) if m else None


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------- vzdalenosti
def neighbour_distances(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pro kazdy radek vzdalenost k nejblizsi a druhe nejblizsi jine draze
    ve stejnem snimku (px)."""
    closest = np.full(len(df), FAR)
    second = np.full(len(df), FAR)
    xy = df[["x", "y"]].to_numpy(dtype=float)
    for _, idx in df.groupby("frame").indices.items():
        pts = xy[idx]
        n = len(pts)
        if n < 2:
            continue
        k = min(3, n)
        d, _ = cKDTree(pts).query(pts, k=k)
        closest[idx] = d[:, 1]
        if k >= 3:
            second[idx] = d[:, 2]
    return closest, second


# ---------------------------------------------------------------- normalizace
def biexp(x, a, tau1, b, tau2, c):
    return a * np.exp(-x / tau1) + b * np.exp(-x / tau2) + c


def fit_frame_means(means: np.ndarray) -> tuple[np.ndarray, dict]:
    """Stejny fit jako dynamin/estimator.py:normalize_biexponential (Shape2Fate):
    a*exp(-t/tau1) + b*exp(-t/tau2) + c na prumer snimku, t = index snimku."""
    t = np.arange(len(means), dtype=float)
    m0 = float(means[0])
    span = max(float(len(means)), 2.0)
    bounds = ([0, 1e-3, 0, 1e-3, 0], [np.inf, 1e4, np.inf, 1e4, np.inf])
    # 1. Matyasuv start; 2.-3. jine starty (pomala + rychla slozka); pokud biexponenciala
    # nezkonverguje, 4. jednoducha exponenciala a*exp(-t/tau)+c (b = 0), a az nakonec konstanta.
    starts = [
        [0.2 * m0, span / 4, 0.2 * m0, span * 2, 0.6 * m0],
        [0.1 * m0, span / 10, 0.1 * m0, span * 10, 0.8 * m0],
        [0.05 * m0, span / 2, 0.05 * m0, span * 5, 0.9 * m0],
    ]
    errors = []
    for k, p0 in enumerate(starts):
        try:
            popt, _ = curve_fit(biexp, t, means, p0=p0, bounds=bounds, maxfev=20000)
            fitted = biexp(t, *popt)
            info = {"method": "biexponential", "start": k,
                    "params": dict(zip(["a", "tau1", "b", "tau2", "c"], popt.tolist()))}
            break
        except Exception as e:  # noqa: BLE001
            errors.append(str(e))
    else:
        try:
            single = lambda x, a, tau, c: a * np.exp(-x / tau) + c  # noqa: E731
            popt, _ = curve_fit(single, t, means, p0=[0.1 * m0, span, 0.9 * m0],
                                bounds=([0, 1e-3, 0], [np.inf, 1e4, np.inf]), maxfev=20000)
            fitted = single(t, *popt)
            info = {"method": "single_exponential", "params": dict(zip(["a", "tau", "c"], popt.tolist())),
                    "biexp_errors": errors}
        except Exception as e:  # noqa: BLE001
            fitted = np.full(len(means), float(np.mean(means)))
            info = {"method": "constant_fallback", "biexp_errors": errors, "error": str(e)}
    fitted = np.where(np.isfinite(fitted) & (fitted > 0), fitted, np.mean(means))
    return fitted, info


def dynamin_frame_means(movie_path: str, channel: int) -> np.ndarray:
    import tifffile
    img = tifffile.imread(movie_path)
    if img.ndim == 4:        # T, C, Y, X
        stack = img[:, channel]
    elif img.ndim == 3:      # T, Y, X (jednokanalove)
        if channel != 0:
            raise ValueError(f"{movie_path}: jednokanalovy film, kanal {channel} neexistuje")
        stack = img
    else:
        raise ValueError(f"{movie_path}: necekany tvar {img.shape}")
    return stack.reshape(len(stack), -1).mean(axis=1, dtype=np.float64)


# ---------------------------------------------------------------- hlavni
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measured", default=os.path.join(ROOT, "lr registered", "measured"))
    ap.add_argument("--movies", default=os.path.join(ROOT, "lr registered"))
    ap.add_argument("--out-root", default=os.path.join(ROOT, "CME_for_Helios", "data"))
    ap.add_argument("--variant", choices=["raw", "normalized", "both"], default="both")
    ap.add_argument("--slave-channel", type=int, default=1, help="kanal dynaminu ve filmu (lr: 1)")
    ap.add_argument("--calibration", default=os.path.join(ROOT, "lr registered", "psf_calibration_lr.json"))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.measured, "*-lr-trajectories-dynamin.csv")))
    if not files:
        print(f"zadne *-lr-trajectories-dynamin.csv v {args.measured}", file=sys.stderr)
        return 1
    movies = {film_number(p): p for p in glob.glob(os.path.join(args.movies, "*.tif"))}
    variants = ["raw", "normalized"] if args.variant == "both" else [args.variant]
    calib = {}
    if os.path.exists(args.calibration):
        with open(args.calibration, encoding="utf-8") as fh:
            calib = json.load(fh)

    out_dirs = {v: os.path.join(args.out_root, f"cme_{v}") for v in variants}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    manifest = {v: {"films": {}, "n_rows": 0, "n_tracks": 0, "n_invalid_rows": 0} for v in variants}
    excess_all = {v: [] for v in variants}
    norm_rows, norm_params = [], {}

    for path in files:
        film = film_number(os.path.basename(path).split("-lr-trajectories")[0])
        base = os.path.basename(path).split("-lr-trajectories")[0]
        df = pd.read_csv(path).sort_values(["particle", "frame"]).reset_index(drop=True)
        closest, second = neighbour_distances(df)
        valid = df["dnm_valid"].astype(bool) & np.isfinite(df["dnm_A"])
        A = np.where(valid, df["dnm_A"].to_numpy(dtype=float), 0.0)
        n_invalid = int((~valid).sum())

        factor = None
        if "normalized" in variants:
            if film not in movies:
                print(f"film {film}: chybi film pro normalizaci v {args.movies}", file=sys.stderr)
                return 1
            means = dynamin_frame_means(movies[film], args.slave_channel)
            fitted, info = fit_frame_means(means)
            norm_params[film] = info
            frames = df["frame"].to_numpy(dtype=int)
            if frames.max() >= len(fitted):
                print(f"film {film}: snimek {frames.max()} mimo film ({len(fitted)} snimku)", file=sys.stderr)
                return 1
            factor = fitted[frames]
            norm_rows.extend({"film": film, "frame": i, "frame_mean": float(means[i]), "fit": float(fitted[i])}
                             for i in range(len(means)))

        for v in variants:
            amp = A if v == "raw" else A / factor
            out = pd.DataFrame({
                "particle": df["particle"].astype(int),
                "x": df["x"], "y": df["y"], "cls": df["cls"],
                "frame": df["frame"].astype(int), "cluster": df["cluster"].astype(int),
                "closest_ccp_dist": closest, "second_closest_ccp_dist": second,
            })
            for c in READOUT_COLS:
                out[c] = BASELINE + amp
            keep = [c for c in df.columns if c.startswith("dnm_") or c.startswith("clc_") or c == "track_len"]
            for c in keep:
                out[c] = df[c].to_numpy()
            if v == "normalized":
                out["dnm_norm_factor"] = factor
                out["dnm_A_norm"] = amp
            out.to_csv(os.path.join(out_dirs[v], f"{base}-processed-trajectories.csv"), index=False)
            excess_all[v].append(amp)
            m = manifest[v]
            m["films"][str(film)] = {"n_rows": int(len(out)), "n_tracks": int(out.particle.nunique()),
                                     "n_invalid_rows": n_invalid,
                                     "frames": [int(out.frame.min()), int(out.frame.max())]}
            m["n_rows"] += int(len(out)); m["n_tracks"] += int(out.particle.nunique())
            m["n_invalid_rows"] += n_invalid
        print(f"film {film:>3}: {len(df):>7,} radku  {df.particle.nunique():>6,} drah  "
              f"neplatnych {n_invalid}" + (f"  norm fit {norm_params[film]['method']}" if factor is not None else ""),
              flush=True)

    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    for v in variants:
        exc = np.concatenate(excess_all[v])
        q90 = float(np.quantile(exc, 0.90))
        thr = {"5x5": q90, "3x3": q90, "1x1": q90}
        with open(os.path.join(out_dirs[v], "thresholds_q90.json"), "w", encoding="utf-8") as fh:
            json.dump(thr, fh)
        manifest[v].update({"variant": v, "built": stamp, "cmepython_git": git_hash(),
                            "sigma": calib, "thresholds_q90": thr,
                            "excess_quantiles": {q: float(np.quantile(exc, q / 100)) for q in (50, 90, 95, 99)}})
        if v == "normalized":
            manifest[v]["normalization"] = {"method": "biexponential fit of whole-frame means of the dynamin channel, "
                                                      "a*exp(-t/tau1)+b*exp(-t/tau2)+c (as Shape2Fate dynamin/estimator.py)",
                                            "per_film": {str(k): d for k, d in norm_params.items()}}
            pd.DataFrame(norm_rows).to_csv(os.path.join(out_dirs[v], "normalization_frame_means.csv"), index=False)
        with open(os.path.join(out_dirs[v], "corpus_manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest[v], fh, indent=1)
        with open(os.path.join(out_dirs[v], "README.md"), "w", encoding="utf-8") as fh:
            fh.write(readme(v, manifest[v]))
        print(f"\n[{v}] {manifest[v]['n_tracks']:,} drah, {manifest[v]['n_rows']:,} radku, "
              f"q90 excess = {q90:.4g}  ->  {out_dirs[v]}")
    return 0


def readme(v: str, m: dict) -> str:
    what = ("amplituda A z cmeAnalysis fitu (2D gaussian nad lokalnim pozadim) v surovych jednotkach kamery"
            if v == "raw" else
            "amplituda A z cmeAnalysis fitu vydelena biexponencialnim fitem prumeru snimku dynaminoveho kanalu "
            "(stejna normalizace jako v Shape2Fate readoutu; bezrozmerne, v jednotkach prumeru bunky)")
    return f"""# cmeAnalysis corpus, variant `{v}`

Built {m['built']} by `scripts/build_cme_corpus.py` (CMEpython {m['cmepython_git']}).
{m['n_tracks']:,} tracks, {m['n_rows']:,} rows, {len(m['films'])} films (U2OS, lr registered, 2 s/frame, 79.1 nm/px).
PSF sigma used for the fits: {json.dumps(m['sigma'])}.

Format is the one read by `load_v2_frame()` of Shape2Fate (branch release/dynamin-confusion-v1):
`*_<film>-processed-trajectories.csv` + `thresholds_q90.json`.

## What `dynamin_intensity_*` means here

`dynamin_intensity_5x5 == dynamin_intensity_3x3 == dynamin_intensity_1x1 == 1 + amplitude`, where amplitude is
{what}.
Matyas's code subtracts the baseline 1, so its `excess` equals the cmeAnalysis amplitude directly.
Invalid fits (image border) are written as amplitude 0 (count: {m['n_invalid_rows']} rows).

`thresholds_q90.json`: 90th percentile of the per-frame amplitude over the whole corpus = {m['thresholds_q90']['5x5']:.6g}
(the same rule Matyas's `dynamin_v2_pertrack.py` applies to the box-mean readout).

## Columns

| column | meaning |
|---|---|
| particle, x, y, cls, frame, cluster | as delivered by Shape2Fate (cls = per-frame shape index) |
| closest_ccp_dist, second_closest_ccp_dist | px to the nearest / second-nearest other delivered track in the same frame (9999 = fewer than 2/3 tracks in the frame) |
| dynamin_intensity_5x5/3x3/1x1 | 1 + amplitude (see above) |
| dnm_A, dnm_c | cmeAnalysis amplitude and local background, camera units |
| dnm_A_pstd, dnm_sigma_r | amplitude uncertainty, residual std of the fit |
| dnm_pval, dnm_signif | per-frame test of A > 0 (cmeAnalysis, alpha 0.05) |
| dnm_x, dnm_y | fitted position (free 'xyAc' fit accepted if shift < 3 sigma_master) |
| dnm_valid | fit inside the image |
| track_len | frames in the track |
| clc_* | the same fit on the clathrin channel (bonus) |
{'| dnm_norm_factor, dnm_A_norm | fitted frame mean used as divisor, and A / factor |' if v == 'normalized' else ''}

Neighbour distances are computed over the delivered (pre-filtered) trajectories, not over the full detection set.
"""


if __name__ == "__main__":
    sys.exit(main())
