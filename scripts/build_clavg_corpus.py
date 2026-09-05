#!/usr/bin/env python3
"""Krok 2 klatrinoveho experimentu: korpusy pro Matyasuv sweep, ve kterych
je VSTUPEM klatrinova intenzita z noveho AVG datasetu a STITKEM dynaminova
klasifikace produktivity (cmeAnalysis default significantSlave, fixed masks).

Dve varianty:
  clavg_amp   dynamin_intensity_* = 1 + clc_A   (PSF amplituda, jednotky
              kamery; excess = clc_A, stejna konvence jako korpusy cme_*)
  clavg_box   dynamin_intensity_* = box5_norm   (box-mean/biexp fit; jeho kod
              odecita 1, takze excess = box5_norm - 1 == puvodni readout)

STITEK -- formatovy trik: jeho pipeline pocita label jako max(cls) > 0.7
(Shape Index). Sloupec `cls` zde NAHRAZUJEME hodnotou 1.0 (dynamin+) /
0.0 (dynamin-), takze stejne pravidlo vyrobi presne dynaminovy stitek
a trenovaci kod bezi beze zmeny radku. Puvodni per-frame Shape Index
zustava ve sloupci `si_cls` (jen pro provenienci, kod ho necte).

Paralelne pres filmy.  python3 scripts/build_clavg_corpus.py --workers 8
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
    READOUT_COLS, film_number, git_hash, neighbour_distances,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANTS = ("clavg_amp", "clavg_box")


def process_film(path: str, labels: pd.DataFrame, out_dirs: dict) -> dict:
    base = os.path.basename(path).split("-lr-trajectories")[0]
    film = film_number(base)
    df = pd.read_csv(path).sort_values(["particle", "frame"]).reset_index(drop=True)
    lab = labels[labels.movie == int(film)][["particle", "cme_significant_slave"]]
    df = df.merge(lab, on="particle", how="left", validate="many_to_one")
    if df.cme_significant_slave.isna().any():
        raise RuntimeError(f"film {film}: {int(df.cme_significant_slave.isna().sum())} "
                           "radku bez dynaminoveho stitku")
    closest, second = neighbour_distances(df)

    res = {"film": film, "base": base, "n_rows": len(df),
           "n_tracks": int(df.particle.nunique()),
           "prev_dnm": float(df.groupby("particle").cme_significant_slave.first().mean()),
           "amp": {}}
    for v in out_dirs:
        if v == "clavg_amp":
            valid = df.clc_valid.astype(bool) & np.isfinite(df.clc_A)
            intensity = 1.0 + np.where(valid, df.clc_A.to_numpy(float), 0.0)
        else:
            b = df.box5_norm.to_numpy(float)
            valid = np.isfinite(b)
            intensity = np.where(valid, b, 1.0)   # NaN -> excess 0
        res.setdefault("n_invalid", {})[v] = int((~valid).sum())
        out = pd.DataFrame({
            "particle": df["particle"].astype(int),
            "x": df["x"], "y": df["y"],
            "cls": df["cme_significant_slave"].astype(float),   # STITEK, viz docstring
            "frame": df["frame"].astype(int), "cluster": df["cluster"].astype(int),
            "closest_ccp_dist": closest, "second_closest_ccp_dist": second,
        })
        for c in READOUT_COLS:
            out[c] = intensity
        out["si_cls"] = df["cls"]                               # puvodni Shape Index
        for c in ("track_len", "clc_A", "clc_c", "clc_signif", "clc_valid",
                  "box5_raw", "box5_norm", "frame_mean_fit"):
            out[c] = df[c].to_numpy()
        out.to_csv(os.path.join(out_dirs[v], f"{base}-processed-trajectories.csv"), index=False)
        res["amp"][v] = intensity - 1.0
    return res


def readme(v: str, m: dict) -> str:
    what = ("`1 + clc_A`: cmeAnalysis PSF-fit amplitude above local background, raw camera "
            "units of the AVG dataset (excess = clc_A)" if v == "clavg_amp" else
            "`box5_norm`: 5x5 box mean divided by the biexponential fit of whole-frame "
            "means (the original Shape2Fate readout; the code's excess = box5_norm - 1)")
    return f"""# Clathrin->dynamin corpus, variant `{v}`

Built {m['built']} by `scripts/build_clavg_corpus.py` (CMEpython {m['cmepython_git']}).
{m['n_tracks']:,} tracks, {m['n_rows']:,} rows, {len(m['films'])} films. Input intensities
sampled from the NEW single-channel AVG dataset (unrescaled camera units) at the same CCP
trajectories as all previous experiments.

**LABEL CAVEAT (read this):** the training code derives the label as `max(cls) > 0.7`.
Here the `cls` column is REPLACED by 1.0/0.0 = the cmeAnalysis default dynamin
classification (significantSlave, corrected masks), so the very same rule yields the
dynamin-productivity label and the code runs unchanged. The original per-frame Shape Index
is preserved in the extra column `si_cls` (not read by the code). Dynamin+ prevalence:
{m['prev_dnm']*100:.1f} % of tracks.

`dynamin_intensity_5x5/3x3/1x1` all carry the same value ({what}).
`thresholds_q90.json` = 90th percentile of the per-frame excess over the whole corpus
= {m['thresholds_q90']['5x5']:.6g}.

Expect a very large length confound: both the dynamin label and clathrin intensity grow
with track lifetime (dynamin+ rises from ~24 % to ~95 % across length bands). Pooled
numbers are therefore inflated by design; within-band results are the primary reading.

Internal delivery for the collaboration; ask before sharing outside the project.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sampled", default=os.path.join(ROOT, "Clathrin Analysis", "sampled"))
    ap.add_argument("--labels", default=os.path.join(ROOT, "lr registered",
                                                     "classification_comparison_by_track.csv"))
    ap.add_argument("--out-root", default=os.path.join(ROOT, "CME_for_Helios", "data"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.sampled, "*-clathrin-avg.csv")))
    if not files:
        print(f"zadne *-clathrin-avg.csv v {args.sampled}", file=sys.stderr)
        return 1
    labels = pd.read_csv(args.labels)
    out_dirs = {v: os.path.join(args.out_root, v) for v in VARIANTS}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(process_film, p, labels, out_dirs): p for p in files}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(f"film {r['film']:>3}: {r['n_rows']:>7,} radku  {r['n_tracks']:>6,} drah  "
                  f"dynamin+ {r['prev_dnm']*100:5.1f} %  neplatnych amp/box "
                  f"{r['n_invalid']['clavg_amp']}/{r['n_invalid']['clavg_box']}", flush=True)
    results.sort(key=lambda r: r["film"])

    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    n_tracks = int(sum(r["n_tracks"] for r in results))
    prev = float(np.average([r["prev_dnm"] for r in results],
                            weights=[r["n_tracks"] for r in results]))
    for v in VARIANTS:
        exc = np.concatenate([r["amp"][v] for r in results])
        q90 = float(np.quantile(exc, 0.90))
        with open(os.path.join(out_dirs[v], "thresholds_q90.json"), "w", encoding="utf-8") as fh:
            json.dump({"5x5": q90, "3x3": q90, "1x1": q90}, fh)
        man = {"variant": v, "built": stamp, "cmepython_git": git_hash(),
               "label": "cmeAnalysis significantSlave (default, fixed masks) via cls:=1/0",
               "prev_dnm": prev, "n_rows": int(sum(r["n_rows"] for r in results)),
               "n_tracks": n_tracks, "thresholds_q90": {"5x5": q90, "3x3": q90, "1x1": q90},
               "films": {str(r["film"]): {"n_rows": r["n_rows"], "n_tracks": r["n_tracks"],
                                          "prev_dnm": r["prev_dnm"],
                                          "n_invalid": r["n_invalid"][v]} for r in results},
               "excess_quantiles": {q: float(np.quantile(exc, q / 100)) for q in (50, 90, 95, 99)}}
        with open(os.path.join(out_dirs[v], "corpus_manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=1)
        with open(os.path.join(out_dirs[v], "README.md"), "w", encoding="utf-8") as fh:
            fh.write(readme(v, man))
        print(f"\n[{v}] {n_tracks:,} drah, dynamin+ {prev*100:.1f} %, "
              f"q90 excess = {q90:.4g}  ->  {out_dirs[v]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
