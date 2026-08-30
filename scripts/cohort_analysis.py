# -*- coding: utf-8 -*-
# CMEpython -- Python port mereni intenzity dynaminu z cmeAnalysis.
# Copyright (C) 2026 Ruslan Guliev
#
# Odvozeno z cmeAnalysis (DanuserLab, UT Southwestern), GPL-3.0.
# Tento program je svobodny software: muzete jej sirit a upravovat podle
# podminek GNU General Public License verze 3 nebo pozdejsi. Sirenu je
# BEZ JAKEKOLI ZARUKY. Viz soubor LICENSE.
"""Prubehy dynaminove intenzity podle lifetime kohort, rozdelene podle
maximalniho shape indexu (SI) drahy.

Vstup: measured/*-dynamin.csv (vystup measure_trajectories.py) + filmy
pro zmereni bufferu (5 snimku pred vznikem a po zaniku drahy, gap cesta
portu = interpTrack). Vystup: krivky (CSV) a grafy (PNG).

Spusteni:
  python3 scripts/cohort_analysis.py --measured measured \\
      --movies "reconstructed registered" --framerate 2 \\
      --sigma-slave 2.6424 --sigma-master 1.4187 \\
      --slave-channel 2 --master-channel 0 --si-col cls --out out/cohorts
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cmepython.cohorts import DEFAULT_BOUNDS_S, cohort_curves   # noqa: E402
from cmepython.slave_intensity import dynamin_intensity_gap    # noqa: E402
from cmepython.measure import movie_layout                     # noqa: E402

SI_CANDIDATES = ("SI", "si", "shape_index", "shapeindex", "max_SI", "cls")
GROUPS = ("productive", "abortive", "all")


# ---------------------------------------------------------------- nacteni

def detect_si_column(fieldnames, requested=None):
    if requested:
        if requested not in fieldnames:
            raise SystemExit(f"sloupec SI '{requested}' v CSV neni; k dispozici: {fieldnames}")
        return requested
    for c in SI_CANDIDATES:
        if c in fieldnames:
            return c
    raise SystemExit(f"nenasel jsem sloupec se shape indexem; zadej --si-col. Sloupce: {fieldnames}")


def load_tracks(path, si_col, framerate):
    """Drahy z jednoho measured CSV: {particle: dict}."""
    per = {}
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        si_col = detect_si_column(rd.fieldnames, si_col)
        for r in rd:
            pid = int(float(r["particle"]))
            d = per.setdefault(pid, dict(frames=[], A=[], c=[], y=[], x=[], si=[]))
            d["frames"].append(int(float(r["frame"])))
            d["A"].append(float(r["dnm_A"]))
            d["c"].append(float(r["dnm_c"]))
            d["y"].append(float(r["y"]))
            d["x"].append(float(r["x"]))
            try:
                d["si"].append(float(r[si_col]))
            except ValueError:
                d["si"].append(np.nan)
    tracks = {}
    for pid, d in per.items():
        order = np.argsort(d["frames"])
        fr = np.asarray(d["frames"])[order]
        tracks[pid] = dict(
            frames=fr,
            A=np.asarray(d["A"], float)[order],
            c=np.asarray(d["c"], float)[order],
            y=np.asarray(d["y"], float)[order],
            x=np.asarray(d["x"], float)[order],
            si=np.asarray(d["si"], float)[order],
            lifetime_s=float(fr.size) * framerate,
            max_si=float(np.nanmax(d["si"])) if np.isfinite(d["si"]).any() else np.nan,
        )
    return tracks, si_col


# ---------------------------------------------------------------- buffery

def measure_buffers(movie_path, tracks, b, sigma_slave, sigma_master, slave_channel):
    """Buffer snimky pred/po draze -- runTrackProcessing.m:520-545 pres
    interpTrack: pozice = prvni/posledni bod drahy, A/c init = prvni/
    posledni namerene hodnoty. Snimky mimo film -> NaN (draha se pak v
    kohortach vyradi, pokud jsou buffery vyzadovany)."""
    import tifffile
    with tifffile.TiffFile(str(movie_path)) as tf:
        T, C, Y, X, idx = movie_layout(tf)
        # pozadavky seskupene po snimcich, kazdy snimek se cte jednou
        req = {}
        for pid, t in tracks.items():
            t["sbA"] = np.full(b, np.nan)
            t["ebA"] = np.full(b, np.nan)
            f0, f1 = int(t["frames"][0]), int(t["frames"][-1])
            for k in range(b):
                fs, fe = f0 - b + k, f1 + 1 + k
                if 0 <= fs < T:
                    req.setdefault(fs, []).append((pid, "sbA", k, t["y"][0], t["x"][0], t["A"][0], t["c"][0]))
                if 0 <= fe < T:
                    req.setdefault(fe, []).append((pid, "ebA", k, t["y"][-1], t["x"][-1], t["A"][-1], t["c"][-1]))
        for fr in sorted(req):
            img = tf.pages[idx(fr, slave_channel)].asarray().astype(np.float64)[None, ...]
            for pid, slot, k, y, x, A0, c0 in req[fr]:
                if not (np.isfinite(A0) and np.isfinite(c0)):
                    continue
                r = dynamin_intensity_gap(img, 0, y, x, sigma_slave=sigma_slave,
                                          sigma_master=sigma_master,
                                          A_init=A0, c_init=c0, labels=None)
                tracks[pid][slot][k] = r["A"]
    return tracks


# ---------------------------------------------------------------- vystupy

def write_curves(res_by_group, out_dir):
    p = out_dir / "cohort_curves.csv"
    with open(p, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["group", "cohort", "bounds_s", "t_s", "mean", "sem",
                     "median", "q25", "q75", "n_tracks", "n_movies"])
        for g, res in res_by_group.items():
            for r in res:
                for i, t in enumerate(r["t"]):
                    wr.writerow([g, r["cohort"], f"{r['bounds'][0]:.0f}-{r['bounds'][1]:.0f}",
                                 f"{t:.1f}", f"{r['mean'][i]:.3f}", f"{r['sem'][i]:.3f}",
                                 f"{r['median'][i]:.3f}", f"{r['q25'][i]:.3f}",
                                 f"{r['q75'][i]:.3f}", r["n_tracks"], r["n_movies"]])
    return p


def plot_all(res_by_group, out_dir, framerate, title_note=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nc = len(next(iter(res_by_group.values())))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(nc - 1, 1)) for i in range(nc)]

    # obr. 1: jeden panel na skupinu, kohorty barevne (styl cmeAnalysis)
    fig, axes = plt.subplots(1, len(res_by_group), figsize=(5.2 * len(res_by_group), 4.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (g, res) in zip(axes, res_by_group.items()):
        for r, col in zip(res, colors):
            if r["n_tracks"] == 0:
                continue
            ax.fill_between(r["t"], r["mean"] - r["sem"], r["mean"] + r["sem"],
                            color=col, alpha=0.18, linewidth=0)
            ax.plot(r["t"], r["mean"], color=col, lw=1.8,
                    label=f"{r['bounds'][0]:.0f}–{r['bounds'][1]:.0f} s  (n={r['n_tracks']})")
        ax.axvline(0, color="0.5", lw=0.8, ls="--")
        ax.set_title(f"{g}{title_note}")
        ax.set_xlabel("čas od vzniku dráhy [s]")
        ax.legend(fontsize=7.5, frameon=False)
    axes[0].set_ylabel("intenzita dynaminu (A, kamerové jednotky)")
    fig.suptitle("Průběh dynaminu podle lifetime kohort — průměr ± SEM přes filmy", y=1.02)
    fig.tight_layout()
    p1 = out_dir / "cohorts_by_group.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # obr. 2: jeden panel na kohortu, skupiny prekryte
    fig, axes = plt.subplots(2, int(np.ceil(nc / 2)), figsize=(4.6 * np.ceil(nc / 2), 7.6))
    axes = axes.ravel()
    gcol = {"productive": "#B4610E", "abortive": "#2A6F77", "all": "0.35"}
    for c in range(nc):
        ax = axes[c]
        any_data = False
        for g, res in res_by_group.items():
            r = res[c]
            if r["n_tracks"] == 0:
                continue
            any_data = True
            ax.fill_between(r["t"], r["mean"] - r["sem"], r["mean"] + r["sem"],
                            color=gcol.get(g, "k"), alpha=0.15, linewidth=0)
            ax.plot(r["t"], r["mean"], color=gcol.get(g, "k"), lw=1.8,
                    label=f"{g} (n={r['n_tracks']})")
        r0 = next(iter(res_by_group.values()))[c]
        ax.set_title(f"kohorta {r0['bounds'][0]:.0f}–{r0['bounds'][1]:.0f} s")
        ax.axvline(0, color="0.5", lw=0.8, ls="--")
        if any_data:
            ax.legend(fontsize=7.5, frameon=False)
        ax.set_xlabel("čas od vzniku [s]")
    for ax in axes[nc:]:
        ax.axis("off")
    fig.suptitle("SI-produktivní vs SI-abortivní dráhy — průměr ± SEM přes filmy", y=1.0)
    fig.tight_layout()
    p2 = out_dir / "cohorts_productive_vs_abortive.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p1, p2


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measured", default="measured")
    ap.add_argument("--movies", default="reconstructed registered")
    ap.add_argument("--framerate", type=float, default=2.0, help="s/snimek")
    ap.add_argument("--sigma-slave", type=float, required=True)
    ap.add_argument("--sigma-master", type=float, required=True)
    ap.add_argument("--slave-channel", type=int, default=2)
    ap.add_argument("--master-channel", type=int, default=0)
    ap.add_argument("--buffer-frames", type=int, default=5)
    ap.add_argument("--bounds", type=float, nargs="+", default=list(DEFAULT_BOUNDS_S))
    ap.add_argument("--si-col", default=None)
    ap.add_argument("--si-threshold", type=float, default=0.7)
    ap.add_argument("--require-inside", action="store_true",
                    help="vyradit drahy dotykajici se prvniho/posledniho snimku "
                         "(pro nepredfiltrovana data)")
    ap.add_argument("--max-movies", type=int, default=None, help="jen prvnich N filmu (zkouska)")
    ap.add_argument("--out", default="out/cohorts")
    a = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    mdir, vdir, out = root / a.measured, root / a.movies, root / a.out
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(mdir.glob("*-dynamin.csv"))
    if a.max_movies:
        files = files[:a.max_movies]
    all_tracks, si_col_used = [], None
    t_all = time.time()
    for p in files:
        m = re.search(r"_(\d+)(?:[-_]|$)", p.stem)      # prvni cislo za podtrzitkem
        if not m:
            print(f"VAROVANI: z nazvu {p.name} nejde precist cislo filmu -- preskakuji",
                  file=sys.stderr)
            continue
        k = m.group(1)
        mv = sorted(vdir.glob(f"*_{k}_*.tif"))
        if len(mv) != 1:
            print(f"VAROVANI: {p.name}: {len(mv)} filmu -- preskakuji", file=sys.stderr)
            continue
        t0 = time.time()
        tracks, si_col_used = load_tracks(p, a.si_col, a.framerate)
        import tifffile
        with tifffile.TiffFile(str(mv[0])) as tf:
            T = movie_layout(tf)[0]
        if a.require_inside:
            tracks = {pid: t for pid, t in tracks.items()
                      if t["frames"][0] > 0 and t["frames"][-1] < T - 1}
        # jen drahy, ktere vubec mohou spadnout do kohorty -- setri buffery
        lo = min(a.bounds)
        tracks = {pid: t for pid, t in tracks.items() if t["lifetime_s"] >= lo}
        if a.buffer_frames:
            measure_buffers(mv[0], tracks, a.buffer_frames, a.sigma_slave,
                            a.sigma_master, a.slave_channel)
        for pid, t in tracks.items():
            all_tracks.append(dict(movie=k, particle=pid, A=t["A"],
                                   sbA=t.get("sbA"), ebA=t.get("ebA"),
                                   lifetime_s=t["lifetime_s"], max_si=t["max_si"]))
        print(f"  film {k:>3}: {len(tracks):>6,} drah v rozsahu kohort  {time.time()-t0:5.1f}s",
              flush=True)

    def group_of(t):
        if not np.isfinite(t["max_si"]):
            return None
        return "productive" if t["max_si"] > a.si_threshold else "abortive"

    res_by_group = {}
    for g in GROUPS:
        sel = [t for t in all_tracks if g == "all" or group_of(t) == g]
        res_by_group[g] = cohort_curves(sel, tuple(a.bounds), a.framerate, a.buffer_frames)

    pc = write_curves(res_by_group, out)
    p1, p2 = plot_all(res_by_group, out, a.framerate)

    print(f"\nSI sloupec: '{si_col_used}', prah max SI > {a.si_threshold}; "
          f"buffer {a.buffer_frames} snimku; {a.framerate} s/snimek\n")
    print(f"{'kohorta [s]':>12}{'produktivni':>13}{'abortivni':>11}{'vsechny':>9}"
          f"{'vrchol prod.':>14}{'vrchol abort.':>15}")
    print("-" * 74)
    for c in range(len(a.bounds) - 1):
        rp, ra, rl = (res_by_group[g][c] for g in GROUPS)
        pk = lambda r: f"{np.nanmax(r['mean']):.0f}" if r["n_tracks"] else "-"
        print(f"{f'{a.bounds[c]:.0f}-{a.bounds[c+1]:.0f}':>12}{rp['n_tracks']:>13,}"
              f"{ra['n_tracks']:>11,}{rl['n_tracks']:>9,}{pk(rp):>14}{pk(ra):>15}")
    print(f"\ncelkem {time.time()-t_all:.0f} s -> {pc.name}, {p1.name}, {p2.name} v {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
