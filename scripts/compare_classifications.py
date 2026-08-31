#!/usr/bin/env python3
"""Ukol B: porovnani dvou binarnich klasifikaci na tychz trajektoriich.

  - SI klasifikace:       max SI pres zivot drahy > 0.7
  - cmeAnalysis (vychozi): significantSlave, tj. t-test amplitud proti bg95 filmu
                           s binomickym prahem na pocet vyznamnych snimku
                           (vychozi rezim 's' dle ccpSorter.m:46; varianty 'm'
                           a 'm&s' reportovany jako citlivostni)

Vstup:  lr registered/measured/*-classified.csv  (vystup scripts/classify_trajectories.py)
Vystup: confusion matrix + pocty skupin (stdout), JSON se vsemi cisly.
Zadny delkovy filtr se neaplikuje (dodane drahy jsou > 5 snimku).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANDS = [(6, 9), (10, 19), (20, 39), (40, 10 ** 9)]


def load(measured_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(measured_dir, "*-classified.csv")))
    if not files:
        sys.exit(f"zadne *-classified.csv v {measured_dir}")
    parts = []
    for f in files:
        m = re.search(r"_(\d+)-", os.path.basename(f))
        parts.append(pd.read_csv(f).assign(film=int(m.group(1)) if m else -1))
    return pd.concat(parts, ignore_index=True)


def confusion(si: np.ndarray, cme: np.ndarray) -> dict:
    a = int((si & cme).sum())        # SI+ & cme+
    b = int((si & ~cme).sum())       # SI+ & cme-
    c = int((~si & cme).sum())       # SI- & cme+
    d = int((~si & ~cme).sum())      # SI- & cme-
    n = a + b + c + d
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / n ** 2
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    orr = (a * d) / (b * c) if b * c else float("inf")
    return {
        "n": n,
        "SI+cme+": a, "SI+cme-": b, "SI-cme+": c, "SI-cme-": d,
        "P(cme+|SI+)": a / (a + b) if a + b else float("nan"),
        "P(cme+|SI-)": c / (c + d) if c + d else float("nan"),
        "P(SI+|cme+)": a / (a + c) if a + c else float("nan"),
        "P(SI+|cme-)": b / (b + d) if b + d else float("nan"),
        "agreement": po, "kappa": kappa, "odds_ratio": orr,
        # SI bran jako reference (jen konvence zobrazeni, ne tvrzeni o pravde):
        "sensitivity_cme_vs_SI": a / (a + b) if a + b else float("nan"),
        "specificity_cme_vs_SI": d / (c + d) if c + d else float("nan"),
    }


def mantel_haenszel(df: pd.DataFrame, si_col: str, cme_col: str, n_strata: int = 12) -> float:
    q = pd.qcut(df["track_len"], n_strata, duplicates="drop")
    num = den = 0.0
    for _, g in df.groupby(q, observed=True):
        si = g[si_col].to_numpy(bool)
        cme = g[cme_col].to_numpy(bool)
        a = float((si & cme).sum()); b = float((si & ~cme).sum())
        c = float((~si & cme).sum()); d = float((~si & ~cme).sum())
        n = len(g)
        num += a * d / n
        den += b * c / n
    return num / den if den else float("nan")


def print_matrix(tag: str, r: dict) -> None:
    w = 9
    print(f"\n--- {tag} (n = {r['n']:,}) ---")
    print(f"{'':>14}{'cme+':>{w}}{'cme-':>{w}}{'celkem':>{w}}")
    print(f"{'SI+ (prod.)':>14}{r['SI+cme+']:>{w},}{r['SI+cme-']:>{w},}{r['SI+cme+'] + r['SI+cme-']:>{w},}")
    print(f"{'SI- (abort.)':>14}{r['SI-cme+']:>{w},}{r['SI-cme-']:>{w},}{r['SI-cme+'] + r['SI-cme-']:>{w},}")
    print(f"{'celkem':>14}{r['SI+cme+'] + r['SI-cme+']:>{w},}{r['SI+cme-'] + r['SI-cme-']:>{w},}{r['n']:>{w},}")
    print(f"shoda {r['agreement']:.3f} | kappa {r['kappa']:.3f} | OR {r['odds_ratio']:.2f} | "
          f"P(cme+|SI+) {r['P(cme+|SI+)']:.3f} | P(cme+|SI-) {r['P(cme+|SI-)']:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measured", default=os.path.join(ROOT, "lr registered", "measured"))
    ap.add_argument("--si-threshold", type=float, default=0.7)
    ap.add_argument("--out", default=os.path.join(ROOT, "lr registered", "classification_comparison.json"))
    args = ap.parse_args()

    df = load(args.measured)
    df["si_pos"] = df["max_si"] > args.si_threshold
    defs = {
        "s (significantSlave, vychozi)": df["significant_slave"].astype(bool),
        "m (significantMaster)": df["significant_master"].astype(bool),
        "m&s (obe podminky)": df["significant_slave"].astype(bool) & df["significant_master"].astype(bool),
    }
    print(f"{df.film.nunique()} filmu, {len(df):,} drah | SI prah {args.si_threshold} "
          f"| SI+ celkem {int(df.si_pos.sum()):,} ({df.si_pos.mean():.1%}) | delkovy filtr: zadny")

    out = {"si_threshold": args.si_threshold, "n_films": int(df.film.nunique()),
           "n_tracks": int(len(df)), "definitions": {}}

    for name, cme in defs.items():
        df["_cme"] = cme
        r = confusion(df.si_pos.to_numpy(bool), cme.to_numpy(bool))
        r["OR_length_adjusted_MH12"] = mantel_haenszel(df, "si_pos", "_cme")
        print_matrix(name, r)
        # dlouhodoba pasma
        bands = {}
        for lo, hi in BANDS:
            g = df[(df.track_len >= lo) & (df.track_len <= hi)]
            bands[f"{lo}-{hi if hi < 10**9 else 'inf'}"] = confusion(
                g.si_pos.to_numpy(bool), g["_cme"].to_numpy(bool))
        r["by_length_band"] = bands
        # po filmech
        r["by_film"] = {int(f): confusion(g.si_pos.to_numpy(bool), g["_cme"].to_numpy(bool))
                        for f, g in df.groupby("film")}
        out["definitions"][name] = r

    prim = out["definitions"]["s (significantSlave, vychozi)"]
    print(f"\nOR (surove) {prim['odds_ratio']:.2f} -> OR ocistene o delku (Mantel-Haenszel, 12 strat) "
          f"{prim['OR_length_adjusted_MH12']:.2f}")
    print("pasma (vychozi definice):")
    for band, r in prim["by_length_band"].items():
        print(f"  {band:>7}: n={r['n']:>6,}  cme+ {100 * (r['SI+cme+'] + r['SI-cme+']) / r['n']:5.1f} %  "
              f"SI+ {100 * (r['SI+cme+'] + r['SI+cme-']) / r['n']:5.1f} %  OR {r['odds_ratio']:5.2f}")

    df.drop(columns="_cme", inplace=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nulozeno: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
