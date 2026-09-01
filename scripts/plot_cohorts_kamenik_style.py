# -*- coding: utf-8 -*-
"""Kohortove prubehy ve stylu plotIntensityCohorts (Kamenikuv obrazek):
dva panely (dynamin-pozitivni / dynamin-negativni podle significantSlave),
v kazdem oba kanaly (master klatrin cerveno-oranzove, slave dynamin zelene),
kohorty odsazene vedle sebe, prumer +- SEM pres filmy, cerne pozadi,
pocty drah nad kohortami a podil pozitivnich v titulku.

  python3 scripts/plot_cohorts_kamenik_style.py \
      --measured "lr registered/measured" --movies "lr registered" \
      --out "lr registered/cohorts"
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
from cmepython.cohorts import DEFAULT_BOUNDS_S, cohort_curves      # noqa: E402
from cmepython.slave_intensity import dynamin_intensity_gap        # noqa: E402
from cmepython.measure import movie_layout                         # noqa: E402


def load_tracks(path, framerate):
    per = {}
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            pid = int(float(r["particle"]))
            d = per.setdefault(pid, dict(frames=[], A=[], c=[], Am=[], cm=[], y=[], x=[]))
            d["frames"].append(int(float(r["frame"])))
            d["A"].append(float(r["dnm_A"]));  d["c"].append(float(r["dnm_c"]))
            d["Am"].append(float(r["clc_A"])); d["cm"].append(float(r["clc_c"]))
            d["y"].append(float(r["y"]));      d["x"].append(float(r["x"]))
    tracks = {}
    for pid, d in per.items():
        o = np.argsort(d["frames"])
        fr = np.asarray(d["frames"])[o]
        tracks[pid] = dict(frames=fr,
                           A=np.asarray(d["A"], float)[o],  c=np.asarray(d["c"], float)[o],
                           Am=np.asarray(d["Am"], float)[o], cm=np.asarray(d["cm"], float)[o],
                           y=np.asarray(d["y"], float)[o],  x=np.asarray(d["x"], float)[o],
                           lifetime_s=float(fr.size) * framerate)
    return tracks


def measure_buffers(movie_path, tracks, b, sigma, channel, a_key, c_key, out_prefix):
    """Buffery pred vznikem/po zaniku pro dany kanal (gap cesta portu)."""
    import tifffile
    with tifffile.TiffFile(str(movie_path)) as tf:
        T, C, Y, X, idx = movie_layout(tf)
        req = {}
        for pid, t in tracks.items():
            t[out_prefix + "s"] = np.full(b, np.nan)
            t[out_prefix + "e"] = np.full(b, np.nan)
            f0, f1 = int(t["frames"][0]), int(t["frames"][-1])
            for k in range(b):
                fs, fe = f0 - b + k, f1 + 1 + k
                if 0 <= fs < T:
                    req.setdefault(fs, []).append((pid, out_prefix + "s", k,
                                                   t["y"][0], t["x"][0], t[a_key][0], t[c_key][0]))
                if 0 <= fe < T:
                    req.setdefault(fe, []).append((pid, out_prefix + "e", k,
                                                   t["y"][-1], t["x"][-1], t[a_key][-1], t[c_key][-1]))
        for fr in sorted(req):
            img = tf.pages[idx(fr, channel)].asarray().astype(np.float64)[None, ...]
            for pid, slot, k, y, x, A0, c0 in req[fr]:
                if not (np.isfinite(A0) and np.isfinite(c0)):
                    continue
                r = dynamin_intensity_gap(img, 0, y, x, sigma_slave=sigma, sigma_master=sigma,
                                          A_init=A0, c_init=c0, labels=None)
                tracks[pid][slot][k] = r["A"]


def load_slave_flags(measured_dir, film):
    p = sorted(Path(measured_dir).glob(f"*_{film}-lr-trajectories-classified.csv"))
    flags = {}
    if not p:
        return flags
    with open(p[0], newline="") as f:
        for r in csv.DictReader(f):
            flags[int(float(r["particle"]))] = bool(int(r["significant_slave"]))
    return flags


def kamenik_panel(ax, res_m, res_s, framerate, gap_s=6.0):
    import matplotlib.pyplot as plt
    nc = len(res_m)
    cm_master = plt.get_cmap("autumn")      # cervena -> oranzova
    cm_slave = plt.get_cmap("summer")       # tmave zelena -> svetla
    x0, ticks = 0.0, []
    for c in range(nc):
        rm, rs = res_m[c], res_s[c]
        if rm["n_tracks"] == 0:
            continue
        t = rm["t"] - rm["t"][0]
        colM = cm_master(0.15 + 0.7 * c / max(nc - 1, 1))
        colS = cm_slave(0.10 + 0.6 * c / max(nc - 1, 1))
        for r, col in ((rm, colM), (rs, colS)):
            ax.fill_between(x0 + t, r["mean"] - r["sem"], r["mean"] + r["sem"],
                            color=col, alpha=0.45, linewidth=0)
            ax.plot(x0 + t, r["mean"], color=col, lw=1.4)
        ax.text(x0 + t[-1] / 2, ax.get_ylim()[1] * 0.0 + 1.0, "", fontsize=8)
        ticks.append((x0, f"{rm['bounds'][0]:.0f}-{rm['bounds'][1]:.0f} s", rm["n_tracks"], t[-1]))
        x0 += t[-1] + gap_s
    return ticks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measured", default="lr registered/measured")
    ap.add_argument("--movies", default="lr registered")
    ap.add_argument("--framerate", type=float, default=2.0)
    ap.add_argument("--sigma-dnm", type=float, default=1.4356)
    ap.add_argument("--sigma-clc", type=float, default=1.6424)
    ap.add_argument("--dnm-channel", type=int, default=1)
    ap.add_argument("--clc-channel", type=int, default=0)
    ap.add_argument("--buffer-frames", type=int, default=5)
    ap.add_argument("--bounds", type=float, nargs="+", default=list(DEFAULT_BOUNDS_S))
    ap.add_argument("--max-movies", type=int, default=None)
    ap.add_argument("--out", default="lr registered/cohorts")
    a = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    mdir, vdir, out = root / a.measured, root / a.movies, root / a.out
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(mdir.glob("*-lr-trajectories-dynamin.csv"))
    if a.max_movies:
        files = files[:a.max_movies]
    all_tracks, pos_frac = [], []
    for p in files:
        film = re.search(r"_(\d+)-lr", p.stem).group(1)
        mv = sorted(vdir.glob(f"*_{film}_*.tif"))
        if len(mv) != 1:
            print(f"VAROVANI: film {film}: {len(mv)} tif -- preskakuji", file=sys.stderr)
            continue
        t0 = time.time()
        tracks = load_tracks(p, a.framerate)
        flags = load_slave_flags(mdir, film)
        lo = min(a.bounds)
        tracks = {pid: t for pid, t in tracks.items() if t["lifetime_s"] >= lo and pid in flags}
        measure_buffers(mv[0], tracks, a.buffer_frames, a.sigma_dnm, a.dnm_channel, "A", "c", "bD")
        measure_buffers(mv[0], tracks, a.buffer_frames, a.sigma_clc, a.clc_channel, "Am", "cm", "bM")
        n_pos = sum(flags[pid] for pid in tracks)
        pos_frac.append(n_pos / max(len(tracks), 1))
        for pid, t in tracks.items():
            all_tracks.append(dict(movie=film, particle=pid, pos=flags[pid],
                                   A=t["A"], sbA=t["bDs"], ebA=t["bDe"],
                                   Am=t["Am"], sbM=t["bMs"], ebM=t["bMe"],
                                   lifetime_s=t["lifetime_s"]))
        print(f"  film {film:>3}: {len(tracks):>6,} drah (pozitivnich {n_pos/max(len(tracks),1):.0%})"
              f"  {time.time()-t0:5.1f}s", flush=True)

    res = {}
    for name, sel in (("pos", [t for t in all_tracks if t["pos"]]),
                      ("neg", [t for t in all_tracks if not t["pos"]])):
        res[name + "_dnm"] = cohort_curves(sel, tuple(a.bounds), a.framerate, a.buffer_frames)
        sel_m = [dict(t, A=t["Am"], sbA=t["sbM"], ebA=t["ebM"]) for t in sel]
        res[name + "_clc"] = cohort_curves(sel_m, tuple(a.bounds), a.framerate, a.buffer_frames)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    pf = 100 * np.asarray(pos_frac)
    titles = [f"dynamin-pozitivní (significantSlave)   +: {pf.mean():.1f}±{pf.std(ddof=1):.1f} %",
              f"dynamin-negativní   −: {100-pf.mean():.1f}±{pf.std(ddof=1):.1f} %"]
    for ax, grp, title in zip(axes, ("pos", "neg"), titles):
        ax.set_facecolor("black")
        ticks = kamenik_panel(ax, res[grp + "_clc"], res[grp + "_dnm"], a.framerate)
        ymax = ax.get_ylim()[1]
        for x0, lbl, n, width in ticks:
            ax.text(x0 + width / 2, ymax * 0.97, f"{n:,}".replace(",", " "),
                    color="white", ha="center", va="top", fontsize=9)
        ax.set_xticks([x0 for x0, *_ in ticks])
        ax.set_xticklabels([lbl for _, lbl, *_ in ticks], rotation=45, ha="right")
        ax.set_xlabel("lifetime kohorta")
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="x", color="0.35", lw=0.5)
    axes[0].set_ylabel("intenzita (A, kamerové jednotky)")
    fig.suptitle("Kohortové průběhy: klatrin (červené) a dynamin (zelené), průměr ± SEM přes filmy",
                 fontsize=11)
    fig.tight_layout()
    p = out / "cohorts_kamenik_style.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"\npodil pozitivnich pres filmy: {pf.mean():.1f} ± {pf.std(ddof=1):.1f} %")
    print(f"hotovo -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
