#!/usr/bin/env python3
"""Sestavi KLATRINOVY korpus (pozitivni kontrola) pro Matyasuv trenovaci kod,
paralelne pres filmy.

Stejny format a stejna logika jako scripts/build_cme_corpus.py, ale misto
dynaminove amplitudy (dnm_A) se pouzije amplituda cmeAnalysis fitu na
KLATRINOVEM kanalu (clc_A) v pozicich tychz drah.  Ucel: pozitivni kontrola --
umi pipeline najit signal, kdyz v datech je?  Klatrin neni nezavisla validace
SI (SI se pocita z tvaru teze struktury), vysledek se cte jen jako kontrola
metody, ne jako biologie.

Vstup:  lr registered/measured/*-lr-trajectories-dynamin.csv  (sloupce clc_*)
Vystup: <out-root>/clathrin_raw/         clc_A v surovych jednotkach kamery
        <out-root>/clathrin_normalized/  clc_A / biexp fit prumeru snimku
                                          KLATRINOVEHO kanalu (lr: kanal 0)

POZOR na nazvoslovi: sloupce vystupu se musi jmenovat dynamin_intensity_5x5/
3x3/1x1 -- to je formatovy kontrakt `load_v2_frame()` v Shape2Fate (vetev
release/dynamin-confusion-v1), zadna volba.  Obsahem je zde 1 + clc_A.
Konvence jeho kodu: excess = dynamin_intensity - 1, takze excess == clc_A.

Paralelizace: jeden proces na film (--workers, default = pocet jader).

    python3 scripts/build_clathrin_corpus.py --workers 8
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_cme_corpus import (  # noqa: E402
    BASELINE, FAR, READOUT_COLS, dynamin_frame_means, film_number,
    fit_frame_means, git_hash, neighbour_distances,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def process_film(path: str, movie: str | None, variants: list[str],
                 out_dirs: dict[str, str], master_channel: int) -> dict:
    """Zpracuje jeden film: sousedske vzdalenosti, klatrinova amplituda,
    normalizace, zapis obou variant.  Bezi v samostatnem procesu."""
    base = os.path.basename(path).split("-lr-trajectories")[0]
    film = film_number(base)
    df = pd.read_csv(path).sort_values(["particle", "frame"]).reset_index(drop=True)
    closest, second = neighbour_distances(df)
    valid = df["dnm_valid"].astype(bool) & np.isfinite(df["clc_A"])
    A = np.where(valid, df["clc_A"].to_numpy(dtype=float), 0.0)
    res = {"film": film, "base": base, "n_rows": len(df),
           "n_tracks": int(df.particle.nunique()), "n_invalid": int((~valid).sum()),
           "frames": [int(df.frame.min()), int(df.frame.max())], "amp": {}}

    factor = None
    if "normalized" in variants:
        if movie is None:
            raise RuntimeError(f"film {film}: chybi .tif pro normalizaci")
        means = dynamin_frame_means(movie, master_channel)  # zde: klatrinovy kanal
        fitted, info = fit_frame_means(means)
        frames = df["frame"].to_numpy(dtype=int)
        if frames.max() >= len(fitted):
            raise RuntimeError(f"film {film}: snimek {frames.max()} mimo film ({len(fitted)})")
        factor = fitted[frames]
        res["norm_info"] = info
        res["norm_rows"] = [{"film": film, "frame": i, "frame_mean": float(means[i]),
                             "fit": float(fitted[i])} for i in range(len(means))]

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
            out["clc_norm_factor"] = factor
            out["clc_A_norm"] = amp
        out.to_csv(os.path.join(out_dirs[v], f"{base}-processed-trajectories.csv"), index=False)
        res["amp"][v] = amp
    return res


def readme(v: str, m: dict) -> str:
    what = ("clathrin-channel amplitude A from the cmeAnalysis fit (2D gaussian above local "
            "background) in raw camera units" if v == "raw" else
            "clathrin-channel amplitude A from the cmeAnalysis fit divided by a biexponential "
            "fit of the whole-frame means of the CLATHRIN channel (dimensionless)")
    return f"""# Clathrin corpus (positive control), variant `{v}`

Built {m['built']} by `scripts/build_clathrin_corpus.py` (CMEpython {m['cmepython_git']}).
{m['n_tracks']:,} tracks, {m['n_rows']:,} rows, {len(m['films'])} films (U2OS, lr registered, 2 s/frame, 79.1 nm/px).

Same format and the same `load_v2_frame()` contract as the `cme_raw`/`cme_normalized` corpora.
**Naming caveat:** the columns are called `dynamin_intensity_5x5/3x3/1x1` because that is the
format contract of the training code, but here they carry `1 + clathrin amplitude`, so the
code's `excess` equals the clathrin amplitude ({what}).

**Purpose:** positive control of the pipeline. The SI label is computed from the shape of the
same clathrin structure, so this corpus is NOT an independent validation of SI -- read the
result only as "can the pipeline find signal when it exists", next to the dynamin corpora.
Invalid fits are written as amplitude 0 ({m['n_invalid_rows']} rows).
`thresholds_q90.json`: 90th percentile of the per-frame amplitude over the whole corpus
= {m['thresholds_q90']['5x5']:.6g}.

Internal delivery for the collaboration; ask before sharing outside the project.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measured", default=os.path.join(ROOT, "lr registered", "measured"))
    ap.add_argument("--movies", default=os.path.join(ROOT, "lr registered"))
    ap.add_argument("--out-root", default=os.path.join(ROOT, "CME_for_Helios", "data"))
    ap.add_argument("--variant", choices=["raw", "normalized", "both"], default="both")
    ap.add_argument("--master-channel", type=int, default=0,
                    help="kanal klatrinu ve filmu (lr: 0); pouziva se pro normalizaci")
    ap.add_argument("--calibration", default=os.path.join(ROOT, "lr registered", "psf_calibration_lr.json"))
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="pocet paralelnich procesu (default: pocet jader)")
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

    out_dirs = {v: os.path.join(args.out_root, f"clathrin_{v}") for v in variants}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(process_film, p, movies.get(film_number(
                    os.path.basename(p).split("-lr-trajectories")[0])),
                    variants, out_dirs, args.master_channel): p for p in files}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            extra = f"  norm fit {r['norm_info']['method']}" if "norm_info" in r else ""
            print(f"film {r['film']:>3}: {r['n_rows']:>7,} radku  {r['n_tracks']:>6,} drah  "
                  f"neplatnych {r['n_invalid']}{extra}", flush=True)
    results.sort(key=lambda r: r["film"])

    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    for v in variants:
        exc = np.concatenate([r["amp"][v] for r in results])
        q90 = float(np.quantile(exc, 0.90))
        thr = {"5x5": q90, "3x3": q90, "1x1": q90}
        with open(os.path.join(out_dirs[v], "thresholds_q90.json"), "w", encoding="utf-8") as fh:
            json.dump(thr, fh)
        man = {"variant": f"clathrin_{v}", "built": stamp, "cmepython_git": git_hash(),
               "channel": "clathrin (master)", "sigma": calib, "thresholds_q90": thr,
               "n_rows": int(sum(r["n_rows"] for r in results)),
               "n_tracks": int(sum(r["n_tracks"] for r in results)),
               "n_invalid_rows": int(sum(r["n_invalid"] for r in results)),
               "films": {str(r["film"]): {"n_rows": r["n_rows"], "n_tracks": r["n_tracks"],
                                          "n_invalid_rows": r["n_invalid"], "frames": r["frames"]}
                         for r in results},
               "excess_quantiles": {q: float(np.quantile(exc, q / 100)) for q in (50, 90, 95, 99)}}
        if v == "normalized":
            man["normalization"] = {
                "method": "biexponential fit of whole-frame means of the CLATHRIN channel, "
                          "a*exp(-t/tau1)+b*exp(-t/tau2)+c (as Shape2Fate dynamin/estimator.py)",
                "per_film": {str(r["film"]): r["norm_info"] for r in results}}
            pd.DataFrame([row for r in results for row in r["norm_rows"]]).to_csv(
                os.path.join(out_dirs[v], "normalization_frame_means.csv"), index=False)
        with open(os.path.join(out_dirs[v], "corpus_manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=1)
        with open(os.path.join(out_dirs[v], "README.md"), "w", encoding="utf-8") as fh:
            fh.write(readme(v, man))
        print(f"\n[clathrin_{v}] {man['n_tracks']:,} drah, {man['n_rows']:,} radku, "
              f"q90 excess = {q90:.4g}  ->  {out_dirs[v]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
