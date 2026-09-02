#!/usr/bin/env python3
"""Protokol k ukolu B (v4): porovnani klasifikaci SI vs. cmeAnalysis.

Text protokolu se ridi pravidly psani z VU_RG (PRAVIDLA_PSANI.md): autorsky
plural, kratke vety, cisla v tabulkach a popiscich (ne v proze), bloky
"Diskuze:", reader-guide pred slozitymi obrazky, neutralni titulky panelu,
zadne pomlcky v beznem textu, desetinne carky.

Vstupy: lr registered/measured/*-classified*.csv (+ *-boxmean.csv z
scripts/compute_boxmean.py; bez nich se casti o box-mean vynechaji).
Vystup: fig1..fig8, protokol.md, protokol.tex, protokol.pdf (xelatex).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm as _norm, rankdata, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANDS = [(6, 9), (10, 19), (20, 39), (40, 10 ** 9)]
BAND_LABELS = ["6–9", "10–19", "20–39", "40+"]
N_STRATA = 12
BLUE, ORANGE, GREEN, PURPLE = "#2b7bba", "#e07a29", "#27ae60", "#8e44ad"


def fmt_n(v):
    return f"{int(v):,}".replace(",", " ")


def cz(v, nd=2):
    """Ceske desetinne cislo (carka)."""
    return f"{v:.{nd}f}".replace(".", ",")


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def sci_md(p):
    mant, e = f"{p:.0e}".split("e")
    return f"{mant}·10{str(int(e)).translate(_SUP)}"


def sci_tex(p):
    mant, e = f"{p:.0e}".split("e")
    return f"${mant}\\cdot10^{{{int(e)}}}$"


# =============================================================== zakladni statistiky
def counts(si, cme):
    a = int((si & cme).sum()); b = int((si & ~cme).sum())
    c = int((~si & cme).sum()); d = int((~si & ~cme).sum())
    return a, b, c, d


def rank_auc(score, label):
    label = np.asarray(label, bool)
    ok = np.isfinite(score)
    score, label = np.asarray(score, float)[ok], label[ok]
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def within_band_auc(df, col, si_col="si_pos", n_strata=N_STRATA):
    q = pd.qcut(df["track_len"], n_strata, duplicates="drop")
    num = den = 0.0
    for _, g in df.groupby(q, observed=True):
        si = g[si_col].to_numpy(bool)
        if si.all() or not si.any():
            continue
        w = si.sum() * (~si).sum()
        a = rank_auc(g[col].to_numpy(float), si)
        if np.isfinite(a):
            num += a * w
            den += w
    return num / den if den else np.nan


def mh_or(df, si_col="si_pos", cme_col="cme_pos", n_strata=N_STRATA):
    q = pd.qcut(df["track_len"], n_strata, duplicates="drop")
    num = den = 0.0
    for _, g in df.groupby(q, observed=True):
        a, b, c, d = counts(g[si_col].to_numpy(bool), g[cme_col].to_numpy(bool))
        n = len(g)
        num += a * d / n
        den += b * c / n
    return num / den if den else np.nan


def base_metrics(df):
    si = df["si_pos"].to_numpy(bool)
    cme = df["cme_pos"].to_numpy(bool)
    a, b, c, d = counts(si, cme)
    n = a + b + c + d
    sens = a / (a + b) if a + b else np.nan
    spec = d / (c + d) if c + d else np.nan
    ppv = a / (a + c) if a + c else np.nan
    npvv = d / (b + d) if b + d else np.nan
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / n ** 2
    mcc_den = np.sqrt(float(a + c) * (a + b) * (d + b) * (d + c))
    return {
        "n": n, "si_rate": (a + b) / n, "cme_rate": (a + c) / n,
        "sens": sens, "spec": spec, "fpr": 1 - spec, "fnr": 1 - sens,
        "ppv": ppv, "npv": npvv,
        "f1": 2 * ppv * sens / (ppv + sens) if (ppv + sens) else np.nan,
        "bacc": (sens + spec) / 2,
        "mcc": (a * d - b * c) / mcc_den if mcc_den else np.nan,
        "auc": (sens + spec) / 2,
        "agreement": po,
        "kappa": (po - pe) / (1 - pe) if pe < 1 else np.nan,
        "or_crude": (a * d) / (b * c) if b * c else np.inf,
        "or_mh": mh_or(df),
        "wb_auc": within_band_auc(df, "cme_pos"),
        "auc_len_si": rank_auc(df["track_len"].to_numpy(float), si),
        "auc_len_cme": rank_auc(df["track_len"].to_numpy(float), cme),
    }


def matched_metrics(df, col, prefix):
    g = df[np.isfinite(df[col])].copy()
    thr = float(np.quantile(g[col], 0.90))
    g["_call"] = g[col] > thr
    si = g["si_pos"].to_numpy(bool)
    call = g["_call"].to_numpy(bool)
    a, b, c, d = counts(si, call)
    po = (a + d) / len(g)
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / len(g) ** 2
    return {
        f"{prefix}_thr": thr,
        f"{prefix}_rate": call.mean(),
        f"{prefix}_sens": a / (a + b) if a + b else np.nan,
        f"{prefix}_fpr": c / (c + d) if c + d else np.nan,
        f"{prefix}_kappa": (po - pe) / (1 - pe) if pe < 1 else np.nan,
        f"{prefix}_or": (a * d) / (b * c) if b * c else np.inf,
        f"{prefix}_or_mh": mh_or(g, cme_col="_call"),
        f"{prefix}_wb_call": within_band_auc(g, "_call"),
        f"{prefix}_auc": rank_auc(g[col].to_numpy(float), si),
        f"{prefix}_wb": within_band_auc(g, col),
    }


def draw_metrics(df, have_box):
    m = base_metrics(df)
    m.update(matched_metrics(df, "max_A", "cq"))
    if have_box:
        m.update(matched_metrics(df, "box_peak", "bq"))
    return m


def film_bootstrap_multi(df, have_box, n_boot=400, seed=0):
    films = df["film"].unique()
    by_film = {f: g for f, g in df.groupby("film")}
    rng = np.random.default_rng(seed)
    draws: dict[str, list] = {}
    for i in range(n_boot):
        pick = rng.choice(films, size=len(films), replace=True)
        m = draw_metrics(pd.concat([by_film[f] for f in pick], ignore_index=True), have_box)
        for k, v in m.items():
            draws.setdefault(k, []).append(v)
        if (i + 1) % 100 == 0:
            print(f"  bootstrap {i + 1}/{n_boot}", flush=True)
    ci = {}
    for k, vals in draws.items():
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        ci[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)
    return ci


# =============================================================== HL + van Elteren
def hl_shift(x_pos, x_neg, rng, n_pairs=1_000_000):
    if len(x_pos) == 0 or len(x_neg) == 0:
        return np.nan
    i = rng.integers(0, len(x_pos), n_pairs)
    j = rng.integers(0, len(x_neg), n_pairs)
    return float(np.median(x_pos[i] - x_neg[j]))


def hl_by_band(df, col="max_A", n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    films = df["film"].unique()
    by_film = {f: g for f, g in df.groupby("film")}
    rows = []
    for lbl, (lo, hi) in zip(BAND_LABELS, BANDS):
        g = df[(df.track_len >= lo) & (df.track_len <= hi)]
        xp = g[g.si_pos][col].dropna().to_numpy()
        xn = g[~g.si_pos][col].dropna().to_numpy()
        est = hl_shift(xp, xn, rng)
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(films, size=len(films), replace=True)
            s = pd.concat([by_film[f] for f in pick], ignore_index=True)
            s = s[(s.track_len >= lo) & (s.track_len <= hi)]
            boots.append(hl_shift(s[s.si_pos][col].dropna().to_numpy(),
                                  s[~s.si_pos][col].dropna().to_numpy(), rng, n_pairs=200_000))
        boots = np.asarray(boots, float)
        boots = boots[np.isfinite(boots)]
        lo_ci, hi_ci = (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if len(boots) else (np.nan, np.nan)
        rows.append({"band": lbl, "n": len(g),
                     "med_pos": float(np.median(xp)), "med_neg": float(np.median(xn)),
                     "hl": est, "lo": float(lo_ci), "hi": float(hi_ci)})
        print(f"  HL {lbl:>6}: {est:+.0f} ADU [{lo_ci:+.0f}; {hi_ci:+.0f}]", flush=True)
    return rows


def van_elteren(df, col="max_A", n_strata=N_STRATA):
    q = pd.qcut(df["track_len"], n_strata, duplicates="drop")
    T = V = 0.0
    for _, g in df.groupby(q, observed=True):
        x = g[col].to_numpy(float)
        si = g["si_pos"].to_numpy(bool)
        ok = np.isfinite(x)
        x, si = x[ok], si[ok]
        n1, n0 = int(si.sum()), int((~si).sum())
        n = n1 + n0
        if n1 == 0 or n0 == 0:
            continue
        r = rankdata(x)
        W = r[si].sum()
        E = n1 * (n + 1) / 2
        _, tcounts = np.unique(x, return_counts=True)
        tie = ((tcounts ** 3 - tcounts).sum()) / (n * (n - 1)) if n > 1 else 0.0
        Var = n1 * n0 / 12 * (n + 1 - tie)
        w = 1.0 / (n + 1)
        T += w * (W - E)
        V += w * w * Var
    z = T / np.sqrt(V) if V > 0 else np.nan
    return z, float(2 * _norm.sf(abs(z)))


# =============================================================== nacitani
def load_classified(measured_dir, suffix="-classified.csv", si_threshold=0.7):
    files = sorted(glob.glob(os.path.join(measured_dir, f"*{suffix}")))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f).assign(
        film=int(re.search(r"_(\d+)-", os.path.basename(f)).group(1))) for f in files],
        ignore_index=True)
    df["si_pos"] = df["max_si"] > si_threshold
    df["cme_pos"] = df["significant_slave"].astype(bool)
    return df


def load_boxpeaks(measured_dir):
    files = sorted(glob.glob(os.path.join(measured_dir, "*-boxmean.csv")))
    if not files:
        return None, None
    peaks, frames = [], []
    for f in files:
        film = int(re.search(r"_(\d+)-", os.path.basename(f)).group(1))
        d = pd.read_csv(f)
        g = d.groupby("particle")["box5_norm"].max().rename("box_norm_max").reset_index()
        g["film"] = film
        peaks.append(g)
        d["film"] = film
        frames.append(d)
    peaks = pd.concat(peaks, ignore_index=True)
    peaks["box_peak"] = peaks["box_norm_max"] - 1.0
    return peaks, pd.concat(frames, ignore_index=True)


def coverage_rows(measured_dir):
    rows, tot = [], {"rows": 0, "invalid": 0, "tracks": 0, "no_si": 0}
    for f in sorted(glob.glob(os.path.join(measured_dir, "*-lr-trajectories-dynamin.csv"))):
        film = int(re.search(r"_(\d+)-", os.path.basename(f)).group(1))
        d = pd.read_csv(f, usecols=["particle", "dnm_valid", "dnm_A", "cls"])
        inv = int((~(d["dnm_valid"].astype(bool) & np.isfinite(d["dnm_A"]))).sum())
        nosi = int(d.groupby("particle")["cls"].apply(lambda s: not np.isfinite(s).any()).sum())
        nt = d["particle"].nunique()
        rows.append((film, nt, len(d), inv, nosi))
        tot["rows"] += len(d); tot["invalid"] += inv; tot["tracks"] += nt; tot["no_si"] += nosi
    return rows, tot


# =============================================================== obrazky
def _matrix_panel(ax, g, title, cell_fs=10):
    si = g["si_pos"].to_numpy(bool); cme = g["cme_pos"].to_numpy(bool)
    a, b, c, d = counts(si, cme)
    m = np.array([[c, a], [d, b]], dtype=float)
    colpct = m / np.maximum(m.sum(axis=0, keepdims=True), 1) * 100
    ax.imshow(colpct, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{fmt_n(m[i, j])}\n{colpct[i, j]:.0f} %",
                    ha="center", va="center", fontsize=cell_fs,
                    color="white" if colpct[i, j] > 55 else "black")
    ax.set_xticks([0, 1], ["abortivní\n(SI−)", "produktivní\n(SI+)"], fontsize=8.5)
    ax.set_yticks([0, 1], ["dynamin+", "dynamin−"], fontsize=8.5)
    orr = (a * d) / (b * c) if b * c else np.nan
    ax.set_title(f"{title}\nn = {fmt_n(len(g))} · OR {cz(orr)}", fontsize=9.5)


def fig_mosaic(df, out):
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.9))
    _matrix_panel(axes[0], df, "všechny dráhy", cell_fs=11)
    for ax, lbl, (lo, hi) in zip(axes[1:], BAND_LABELS, BANDS):
        _matrix_panel(ax, df[(df.track_len >= lo) & (df.track_len <= hi)], f"{lbl} snímků")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)


def fig_length(df, M, out):
    rates_cme, rates_si, ors, ns = [], [], [], []
    for lo, hi in BANDS:
        g = df[(df.track_len >= lo) & (df.track_len <= hi)]
        a, b, c, d = counts(g.si_pos.to_numpy(bool), g.cme_pos.to_numpy(bool))
        rates_cme.append((a + c) / len(g) * 100); rates_si.append((a + b) / len(g) * 100)
        ors.append((a * d) / (b * c) if b * c else np.nan); ns.append(len(g))
    x = np.arange(len(BANDS))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.bar(x - 0.2, rates_cme, 0.38, label="dynamin+ (cmeAnalysis)", color=BLUE)
    ax1.bar(x + 0.2, rates_si, 0.38, label="SI+ (max SI > 0,7)", color=ORANGE)
    for xi, (rc, rs, n) in enumerate(zip(rates_cme, rates_si, ns)):
        ax1.text(xi - 0.2, rc + 1.5, f"{rc:.0f}", ha="center", fontsize=9)
        ax1.text(xi + 0.2, rs + 1.5, f"{rs:.0f}", ha="center", fontsize=9)
        ax1.text(xi, -10, f"n={fmt_n(n)}", ha="center", fontsize=8, color="0.4")
    ax1.set_xticks(x, BAND_LABELS); ax1.set_ylim(0, 104)
    ax1.set_xlabel("délka dráhy (snímky)"); ax1.set_ylabel("% pozitivních")
    ax1.set_title("Podíl pozitivních drah po pásmech", fontsize=10.5)
    ax1.legend(fontsize=9)
    ax2.plot(x, ors, "o-", color=BLUE, label="OR v pásmu")
    ax2.axhline(1, color="0.6", lw=1, ls=":")
    ax2.axhline(M["or_crude"], color="#c0392b", lw=1.2, ls="--",
                label=f"OR poolované {cz(M['or_crude'])}")
    ax2.axhline(M["or_mh"], color=GREEN, lw=1.2, ls="--",
                label=f"OR očištěné o délku {cz(M['or_mh'])}")
    for xi, o in zip(x, ors):
        ax2.text(xi, o + 0.07, cz(o), ha="center", fontsize=9)
    ax2.set_xticks(x, BAND_LABELS); ax2.set_xlabel("délka dráhy (snímky)")
    ax2.set_ylabel("odds ratio"); ax2.set_ylim(0.8, max(M["or_crude"], max(ors)) + 0.5)
    ax2.set_title("Odds ratio po pásmech", fontsize=10.5)
    ax2.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_films(df, out):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for f, g in df.groupby("film"):
        a, b, c, d = counts(g.si_pos.to_numpy(bool), g.cme_pos.to_numpy(bool))
        ax.scatter((a + b) / len(g) * 100, (a + c) / len(g) * 100,
                   s=np.sqrt(len(g)) * 3, color=BLUE, alpha=0.75)
        ax.annotate(str(f), ((a + b) / len(g) * 100, (a + c) / len(g) * 100),
                    textcoords="offset points", xytext=(5, 3), fontsize=8, color="0.3")
    ax.plot([0, 100], [0, 100], ls=":", color="0.6", lw=1, label="shodné podíly")
    ax.set_xlim(0, 60); ax.set_ylim(0, 100)
    ax.set_xlabel("podíl SI+ ve filmu (%)"); ax.set_ylabel("podíl dynamin+ ve filmu (%)")
    ax.set_title("Podíl pozitivních drah po filmech", fontsize=10.5)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_amplitude(df, hl_rows, out):
    si, no = df[df.si_pos]["max_A"].dropna(), df[~df.si_pos]["max_A"].dropna()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    bins = np.geomspace(max(1.0, min(si.min(), no.min())), max(si.max(), no.max()), 60)
    ax1.hist(no, bins=bins, density=True, alpha=0.6, color=ORANGE, label="SI− (abortivní)")
    ax1.hist(si, bins=bins, density=True, alpha=0.6, color=BLUE, label="SI+ (produktivní)")
    ax1.set_xscale("log"); ax1.set_xlabel("max. amplituda dynaminu na dráze [ADU]")
    ax1.set_ylabel("hustota"); ax1.legend(fontsize=9)
    ax1.set_title("Rozdělení maximální amplitudy podle SI třídy", fontsize=10.5)
    y = np.arange(len(hl_rows))
    ax2.barh(y, [r["hl"] for r in hl_rows], xerr=np.array(
        [[r["hl"] - r["lo"] for r in hl_rows], [r["hi"] - r["hl"] for r in hl_rows]]),
        color=BLUE, alpha=0.8, capsize=3)
    ax2.axvline(0, color="0.4", lw=1)
    ax2.set_yticks(y, [r["band"] for r in hl_rows])
    ax2.set_xlabel("Hodgesův–Lehmannův posun SI+ − SI− [ADU]")
    ax2.set_ylabel("délka dráhy (snímky)")
    ax2.set_title("Posun amplitud po pásmech", fontsize=10.5)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_alpha(sweep_rows, out):
    al = [r["alpha"] for r in sweep_rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.plot(al, [100 * r["cme_rate"] for r in sweep_rows], "o-", color=BLUE,
             label="dynamin+ celkem")
    ax1.plot(al, [100 * r["sens"] for r in sweep_rows], "o-", color=GREEN,
             label="P(dyn+ | SI+)")
    ax1.plot(al, [100 * r["fpr"] for r in sweep_rows], "o-", color=ORANGE,
             label="P(dyn+ | SI−)")
    ax1.set_xscale("log"); ax1.set_xlabel("α (citlivost testu cmeAnalysis)")
    ax1.set_ylabel("% drah"); ax1.set_ylim(0, 100)
    ax1.set_title("Podíl pozitivních drah podle α", fontsize=10.5)
    ax1.legend(fontsize=9)
    ax2.plot(al, [r["kappa"] for r in sweep_rows], "o-", color=BLUE, label="Cohenovo κ")
    ax2.plot(al, [r["or_mh"] for r in sweep_rows], "o-", color=GREEN,
             label="OR očištěné o délku")
    ax2.plot(al, [r["wb_auc"] for r in sweep_rows], "o-", color=PURPLE,
             label="within-band AUC")
    ax2.axhline(1, color="0.6", lw=1, ls=":"); ax2.axhline(0.5, color="0.8", lw=1, ls=":")
    ax2.set_xscale("log"); ax2.set_xlabel("α (citlivost testu cmeAnalysis)")
    ax2.set_title("Míry shody podle α", fontsize=10.5)
    ax2.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_matched(M, CI, have_box, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = ["cmeAnalysis\namplituda"] + (["box-mean 5×5\n(stejné dráhy)"] if have_box else [])
    keys = ["cq"] + (["bq"] if have_box else [])
    x = np.arange(len(keys))
    for shift, mk, color, lab in [(-0.17, "auc", BLUE, "pooled AUC"),
                                  (0.17, "wb", PURPLE, "within-band AUC")]:
        vals = [M[f"{k}_{mk}"] for k in keys]
        err = np.array([[M[f"{k}_{mk}"] - CI[f"{k}_{mk}"][0] for k in keys],
                        [CI[f"{k}_{mk}"][1] - M[f"{k}_{mk}"] for k in keys]])
        ax1.bar(x + shift, vals, 0.3, yerr=err, capsize=3, color=color, label=lab)
        for xi, v in zip(x + shift, vals):
            ax1.text(xi, v + 0.012, cz(v, 3), ha="center", fontsize=9)
    ax1.axhline(0.5, color="0.6", lw=1, ls=":")
    ax1.set_xticks(x, labels); ax1.set_ylim(0.45, 0.85)
    ax1.set_ylabel("AUC (spojitá max. amplituda → SI+)")
    ax1.set_title("AUC spojité amplitudy", fontsize=10.5)
    ax1.legend(fontsize=9)
    rows = [("výchozí call", M["or_mh"], CI["or_mh"]),
            ("q90 call, cmeAnalysis", M["cq_or_mh"], CI["cq_or_mh"])]
    if have_box:
        rows.append(("q90 call, box-mean", M["bq_or_mh"], CI["bq_or_mh"]))
    y = np.arange(len(rows))
    ax2.barh(y, [r[1] for r in rows], xerr=np.array(
        [[r[1] - r[2][0] for r in rows], [r[2][1] - r[1] for r in rows]]),
        color=[BLUE, GREEN, ORANGE][:len(rows)], alpha=0.85, capsize=3)
    ax2.axvline(1, color="0.4", lw=1)
    ax2.set_yticks(y, [r[0] for r in rows]); ax2.invert_yaxis()
    ax2.set_xlabel("odds ratio očištěné o délku (MH, 12 strat)")
    ax2.set_title("OR po očištění o délku", fontsize=10.5)
    for yi, r in zip(y, rows):
        ax2.text(r[1] + 0.03, yi, cz(r[1]), va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_agreement(df, frames_cme, frames_box, out):
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    g = df[np.isfinite(df["max_A"]) & np.isfinite(df["box_norm_max"])]
    hb = ax.hexbin(np.log10(np.clip(g["max_A"], 1, None)),
                   np.log10(np.clip(g["box_norm_max"], 1e-2, None)),
                   gridsize=45, cmap="Blues", bins="log")
    rho_t = spearmanr(g["max_A"], g["box_norm_max"]).correlation
    rho_f = np.nan
    if frames_cme is not None:
        rho_f = spearmanr(frames_cme, frames_box).correlation
    ax.set_xlabel("log10 max. amplitudy cmeAnalysis [ADU]")
    ax.set_ylabel("log10 max. box-mean 5×5 (normováno na buňku)")
    ax.set_title("Maximální amplituda obou readoutů na stejné dráze", fontsize=10.5)
    fig.colorbar(hb, ax=ax, label="počet drah (log)")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return rho_t, rho_f


def fig_perframe(df, out):
    df = df.copy()
    df["frac_sig"] = df["n_above_bg"] / df["track_len"].clip(lower=1)
    edges = np.unique(np.geomspace(6, df.track_len.max(), 12).astype(int))
    df["lbin"] = pd.cut(df.track_len, edges, include_lowest=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for si_val, color, lab in [(True, BLUE, "SI+ (produktivní)"), (False, ORANGE, "SI− (abortivní)")]:
        g = df[df.si_pos == si_val].groupby("lbin", observed=True)["frac_sig"]
        mid = [iv.mid for iv in g.mean().index]
        ax.errorbar(mid, g.mean(), yerr=g.sem(), fmt="o-", color=color, label=lab, capsize=2)
    ax.set_xscale("log")
    ax.set_xlabel("délka dráhy (snímky)")
    ax.set_ylabel("podíl snímků významně nad pozadím")
    ax.set_title("Podíl významných snímků podle délky dráhy", fontsize=10.5)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# =============================================================== tabulky
METRIC_ROWS = [
    ("n drah", "n", "d"),
    ("podíl SI+ (produktivních)", "si_rate", "%"),
    ("podíl dynamin+ (cmeAnalysis)", "cme_rate", "%"),
    ("sensitivity = P(dyn+ | SI+)", "sens", "3"),
    ("specificity = P(dyn− | SI−)", "spec", "3"),
    ("false-positive rate = P(dyn+ | SI−)", "fpr", "3"),
    ("false-negative rate = P(dyn− | SI+)", "fnr", "3"),
    ("precision (PPV) = P(SI+ | dyn+)", "ppv", "3"),
    ("NPV = P(SI− | dyn−)", "npv", "3"),
    ("F1", "f1", "3"),
    ("balanced accuracy", "bacc", "3"),
    ("Matthewsův korelační koeficient (MCC)", "mcc", "3"),
    ("ROC AUC (u binárního callu rovna balanced accuracy)", "auc", "3"),
    ("shoda (accuracy)", "agreement", "3"),
    ("Cohenovo κ", "kappa", "3"),
    ("odds ratio, poolované", "or_crude", "2"),
    ("odds ratio, očištěné o délku (Mantel–Haenszel, 12 strat)", "or_mh", "2"),
    ("AUC jen mezi drahami podobné délky (within-band, 12 strat)", "wb_auc", "3"),
    ("AUC samotné délky dráhy → SI+", "auc_len_si", "3"),
    ("AUC samotné délky dráhy → dynamin+", "auc_len_cme", "3"),
]


def _fmt(v, kind):
    if kind == "d":
        return fmt_n(v)
    if kind == "%":
        return cz(100 * v, 1) + " %"
    return cz(v, 2) if kind == "2" else cz(v, 3)


def metric_table_rows(M, CI):
    rows = []
    for label, key, kind in METRIC_ROWS:
        lo, hi = CI.get(key, (np.nan, np.nan))
        ci = "—" if key == "n" else f"[{_fmt(lo, kind)}; {_fmt(hi, kind)}]"
        rows.append((label, _fmt(M[key], kind), ci))
    return rows


def matched_table_rows(M, CI, have_box):
    def row(label, key, kind="3"):
        lo, hi = CI.get(key, (np.nan, np.nan))
        return [label, _fmt(M[key], kind), f"[{_fmt(lo, kind)}; {_fmt(hi, kind)}]"]
    rows = [
        row("pooled AUC, cmeAnalysis amplituda", "cq_auc"),
        row("within-band AUC, cmeAnalysis amplituda", "cq_wb"),
        row("OR očištěné o délku, q90 call cmeAnalysis", "cq_or_mh", "2"),
        row("Cohenovo κ, q90 call cmeAnalysis", "cq_kappa"),
    ]
    if have_box:
        rows += [
            row("pooled AUC, box-mean 5×5", "bq_auc"),
            row("within-band AUC, box-mean 5×5", "bq_wb"),
            row("OR očištěné o délku, q90 call box-mean", "bq_or_mh", "2"),
            row("Cohenovo κ, q90 call box-mean", "bq_kappa"),
        ]
    return rows


def band_table_rows(df):
    rows = []
    for lbl, (lo, hi) in zip(BAND_LABELS, BANDS):
        g = df[(df.track_len >= lo) & (df.track_len <= hi)]
        a, b, c, d = counts(g.si_pos.to_numpy(bool), g.cme_pos.to_numpy(bool))
        orr = (a * d) / (b * c) if b * c else np.nan
        rows.append((lbl, fmt_n(len(g)), cz(100 * (a + c) / len(g), 1) + " %",
                     cz(100 * (a + b) / len(g), 1) + " %", cz(orr)))
    return rows


def hl_table_rows(hl_rows):
    return [(r["band"], fmt_n(r["n"]), fmt_n(r["med_pos"]), fmt_n(r["med_neg"]),
             f"{r['hl']:+,.0f}".replace(",", " "),
             f"[{r['lo']:+,.0f}; {r['hi']:+,.0f}]".replace(",", " ")) for r in hl_rows]


def sweep_table_rows(sweep_rows):
    return [(cz(r["alpha"], 3).rstrip("0").rstrip(","), cz(100 * r["cme_rate"], 1) + " %",
             cz(r["sens"], 3), cz(r["fpr"], 3), cz(r["kappa"], 3), cz(r["or_mh"]),
             cz(r["wb_auc"], 3)) for r in sweep_rows]


def coverage_table_rows(cov_rows, cov_tot):
    rows = [(str(f), fmt_n(nt), fmt_n(nr), fmt_n(inv), fmt_n(nosi))
            for f, nt, nr, inv, nosi in cov_rows]
    rows.append(("celkem", fmt_n(cov_tot["tracks"]), fmt_n(cov_tot["rows"]),
                 fmt_n(cov_tot["invalid"]), fmt_n(cov_tot["no_si"])))
    return rows


# =============================================================== markdown
def md_table(rows, header):
    head = "| " + " | ".join(header) + " |"
    sep = "|" + "|".join("---" for _ in header) + "|"
    body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}"


def build_markdown(M, CI, df, sweep_rows, hl_rows, ve, cov, args, meta):
    have_box = meta["have_box"]
    t_cov = md_table(coverage_table_rows(*cov), ["film", "drah", "snímků", "selhaných fitů", "drah bez SI"])
    t_met = md_table(metric_table_rows(M, CI), ["metrika", "hodnota", "95% CI"])
    t_band = md_table(band_table_rows(df), ["délka (snímky)", "n", "dynamin+", "SI+", "OR v pásmu"])
    t_amp = md_table(matched_table_rows(M, CI, have_box), ["metrika", "hodnota", "95% CI"])
    t_hl = md_table(hl_table_rows(hl_rows), ["délka", "n", "med. SI+", "med. SI−", "HL posun [ADU]", "95% CI"])
    t_sw = md_table(sweep_table_rows(sweep_rows),
                    ["α", "dynamin+", "P(dyn+\\|SI+)", "P(dyn+\\|SI−)", "κ", "OR očišt.", "wb AUC"]) \
        if sweep_rows else "(sweep nebyl spočítán)"

    box_diskuze = ""
    if have_box:
        box_diskuze = f"""**Diskuze:**

Z tabulky 4 a obrázku 5 plyne, že oba *readouty* dávají v mezích intervalů spolehlivosti stejný
výsledek. Platí to pro spojité hodnocení i pro stejný operační bod. Hodnota OR pro *box-mean*
na našich drahách současně odpovídá hodnotě publikované dříve na plném korpusu (přibližně 1,5),
což naši repliku nezávisle potvrzuje. Usuzujeme tedy, že rozdíl mezi částí 3 a předchozí
analýzou nebyl vlastností měření, ale operačního bodu výchozího pravidla.
"""
    agree_part = ""
    if have_box:
        agree_part = f"""## 7. Shoda readoutů

Zbývá ověřit, že oba *readouty* měří stejnou veličinu.

![shoda readoutů](fig7_agreement.png)

*Obrázek 6: Maximální amplituda cmeAnalysis (vodorovná osa, logaritmicky) proti maximálnímu
box-mean na stejné dráze (svislá osa, logaritmicky). Barva udává počet drah. Spearmanova
korelace je {cz(meta['rho_track'])} po drahách a {cz(meta['rho_frame'])} po snímcích.*

**Diskuze:**

Korelace je po drahách vysoká a po snímcích střední, viz popisek obrázku 6. Dá se říct,
že jde o dvě měření stejné veličiny s jinou citlivostí na pozadí, ne o dvě různé veličiny.
To je v souladu s tím, že v části 6 dávají oba *readouty* stejné výsledky.
"""

    return f"""# Porovnání binárních klasifikací: dynamin (cmeAnalysis) proti Shape Indexu

Datum {meta['date']} · CMEpython {meta['git']} · vygenerováno `scripts/report_classification_comparison.py`

## 1. Co se porovnává a proč

Každé dráze klatrinové jamky přiřadíme dva nezávislé štítky a změříme, jak moc se shodují.

Nejprve štítek tvarový. Dráhu nazveme *produktivní*, pokud její Shape Index někdy za život
překročí {cz(args.si_threshold, 1)}. Dále štítek dynaminový podle pravidla cmeAnalysis
s výchozími parametry. V každém snímku nafitujeme na pozici jamky skvrnku tvaru PSF
({meta['sigma_note']}) a získáme amplitudu nad lokálním pozadím. Dráhu označíme za
*dynamin-pozitivní*, když počet snímků s amplitudou významně nad 95. percentilem pozadí filmu
překročí binomický práh odpovídající délce dráhy. Jde o test „s", tzv. *significantSlave*,
což je výchozí režim cmeAnalysis.

Pracujeme s {meta['n_films']} filmy „lr registered" (U2OS, 2 s/snímek, 79,1 nm/px)
a {fmt_n(M['n'])} dodanými drahami. Dráhy jsou předfiltrované již od dodavatele, a to na délku
přes 5 snímků, polohu uvnitř masky a úplnost v záznamu. Žádný další délkový filtr
neaplikujeme. Pozadí filmů pro klasifikaci počítáme s opravenými buněčnými maskami
(„masks fixed"; původní masky byly poškozené interpolací při uložení, překryv
s opravenými je podle filmu 0,89 až 0,97). Poznamenejme, že ani jeden štítek není
*ground truth* (skutečný stav).
Měříme tedy shodu dvou nedokonalých měření, ne správnost jednoho z nich.

### Pokrytí

{t_cov}

*Tabulka 1: Pokrytí dat po filmech.*

**Diskuze:**

Z tabulky 1 plyne, že fit neselhal na žádném snímku a oba štítky má každá dodaná dráha.
Fit totiž selhává jen na okraji obrazu a dodané dráhy jsou od okraje odfiltrované.
Nevzniká tedy žádný dodatečný výběr, který by mohl korelovat s délkou dráhy.

## 2. Výchozí binární call

Obrázek 1 čteme takto. Sloupce dělí dráhy podle SI na abortivní a produktivní. Řádky je dělí
podle dynaminového štítku. Procenta udávají podíl ve sloupci, tedy jakou část abortivních,
resp. produktivních drah pravidlo označilo za pozitivní. První panel ukazuje všechny dráhy
dohromady. Další čtyři panely ukazují totéž po délkových pásmech.

![mozaika confusion matic](fig1_mosaic.png)

*Obrázek 1: Confusion matice výchozího pravidla, poolovaná a po délkových pásmech. Poolované
odds ratio {cz(M['or_crude'])} se po očištění o délku (Mantel–Haenszel, 12 strat) snižuje na
{cz(M['or_mh'])} [{cz(CI['or_mh'][0])}; {cz(CI['or_mh'][1])}].*

**Diskuze:**

Z obrázku 1 plyne dvojí. Za prvé, dynamin-pozitivní je většina produktivních drah, avšak
současně i polovina abortivních. Za druhé, poolovaná matice naznačuje silnější asociaci, než
jaká platí uvnitř pásem. Oba štítky totiž vznikají operátorem typu „stalo se to někdy za
život", a proto rostou s délkou dráhy samy od sebe. Smícháním krátkých a dlouhých drah pak
vzniká zdánlivá asociace, tzv. Simpsonův jev. Po stratifikaci délkou zbývá odds ratio, jehož
interval spolehlivosti zahrnuje jedničku, viz popisek obrázku 1.

Poznamenejme ještě jednu věc. Výchozí pravidlo označí za pozitivní přes polovinu drah
(tabulka 2). *Box-mean* call z předchozí analýzy má z konstrukce pozitivních jen desetinu.
Přímé srovnání κ mezi oběma pravidly by proto nebylo korektní. Srovnáváme je až na stejném
operačním bodě v části 6.

## 3. Metriková tabulka výchozího callu

Tabulka 2 uvádí všechny běžně reportované metriky. Dynaminový štítek v ní hodnotíme jako
„prediktor" SI štítku; jde o konvenci zobrazení. Intervaly spolehlivosti počítáme
bootstrapem ({args.n_boot} tahů), ve kterém resamplujeme celé filmy. Dráhy stejného filmu
totiž nejsou nezávislé.

{t_met}

*Tabulka 2: Metriky výchozího callu s 95% intervaly spolehlivosti (bootstrap po filmech).*

**Diskuze:**

Z tabulky 2 plyne trojí. Za prvé, shoda nad rámec náhody je nízká (Cohenovo κ). Za druhé,
*within-band* AUC, tedy AUC počítaná jen mezi drahami podobné délky, klesá téměř k 0,5.
Rozdíl mezi poolovanou a *within-band* hodnotou říká, kolik zdánlivé shody dodávala délka.
Za třetí, samotná délka dráhy predikuje dynaminový štítek lépe než štítek SI. Dynaminové
pravidlo je tedy délkou tažené více než SI, které má ověřovat.

## 4. Závislost na délce dráhy

{t_band}

*Tabulka 3: Podíly pozitivních drah a odds ratio po délkových pásmech.*

![délková závislost](fig2_length.png)

*Obrázek 2: Vlevo podíly pozitivních drah po pásmech, vpravo odds ratio v pásmech proti
poolované a očištěné hodnotě.*

Obrázek 3 ukazuje podíl snímků významně nad pozadím v závislosti na délce dráhy, zvlášť pro
obě SI třídy. Svislé úsečky jsou střední chyby průměru.

![per-frame pozitivita](fig8_perframe.png)

*Obrázek 3: Podíl významných snímků roste s délkou dráhy v obou SI třídách; produktivní
dráhy leží o několik procentních bodů výš.*

**Diskuze:**

Pravidlo cmeAnalysis délku explicitně zohledňuje, binomický práh totiž s délkou roste.
Podíl pozitivních drah přesto roste napříč pásmy (tabulka 3). Z obrázku 3 usuzujeme proč.
S délkou roste již pozitivita jednotlivých snímků a binomická korekce pak nemá co
korigovat. Produktivní dráhy leží o několik procentních bodů výš, což odpovídá slabému
amplitudovému signálu z části 6. Dominantní je ovšem společný růst obou tříd. Dlouhé dráhy
tedy leží v místech s více dynaminem. Může jít o vlastnost membránové oblasti. Současně může
jít o selekci trackování, dráhy v jasných místech se totiž snáze trackují dlouho. Rozhodnout
mezi těmito vysvětleními z těchto dat neumíme.

## 5. Rozdíly mezi filmy

![po filmech](fig3_films.png)

*Obrázek 4: Podíl pozitivních drah po filmech; každý bod je jeden film, velikost bodu
odpovídá počtu drah. Všechny body leží vysoko nad diagonálou shodných podílů.*

**Diskuze:**

Ve všech filmech označí pravidlo cmeAnalysis výrazně více drah než SI. Rozptyl mezi filmy je
současně značný. Právě proto počítáme všechny intervaly spolehlivosti bootstrapem po filmech,
ne po drahách.

## 6. Amplitudová větev na stejném operačním bodě

Binární call cmeAnalysis nemá použitelný rozsah operačních bodů, jak ukážeme v části 8.
Korektní srovnání s *box-mean* callem proto postavíme jinak. Nejprve vezmeme maximální
amplitudu na dráhu a práh položíme na její 90. percentil. Tím dostaneme stejný podíl
pozitivních, jaký má *box-mean* call, a to desetinu drah. Dále hodnotíme amplitudu i spojitě,
tedy bez prahu, pomocí AUC. Aby srovnání nezáviselo na korpusu, spočítali jsme *box-mean* 5×5
vlastní implementací na stejných drahách a filmech, a to jako průměr 5×5 px dynaminového
kanálu dělený biexponenciálním fitem průměrů snímků buňky. Oběma způsobům odečtu říkáme
dále *readouty*; readout je způsob, jakým se z obrazu získá číslo „kolik dynaminu je na
daném místě v daném snímku" (průměr okénka u box-mean, amplituda PSF fitu u cmeAnalysis).

{t_amp}

*Tabulka 4: Amplitudová větev; oba readouty na stejných drahách a stejném operačním bodě.*

![amplitudová větev](fig6_matched.png)

*Obrázek 5: Vlevo AUC spojité amplitudy (pooled a within-band), vpravo odds ratio po
očištění o délku pro výchozí call a oba q90 cally.*

{box_diskuze}
{agree_part}

## 8. Amplituda dynaminu podle SI třídy

![amplitudy](fig4_amplitude.png)

*Obrázek 7: Vlevo rozdělení maximální amplitudy podle SI třídy (mediány SI+
{fmt_n(meta['med_si'])} ADU a SI− {fmt_n(meta['med_no'])} ADU; Mann–Whitney
{meta['mw_p_md']}). Vpravo Hodgesovy–Lehmannovy posuny po pásmech s 95% intervaly po filmech.*

{t_hl}

*Tabulka 5: Posun amplitud SI+ proti SI− po délkových pásmech. Souhrnný stratifikovaný test
(van Elteren): z = {cz(ve[0])}, p ≈ {sci_md(ve[1])}; p-hodnota nezohledňuje clustering po
filmech, intervaly posunů ano.*

**Diskuze:**

Posun amplitud mezi třídami je prokazatelný, avšak malý, viz tabulka 5. Poolovaný rozdíl
mediánů je z velké části opět efekt délky. Delší dráhy mají totiž vyšší maximum i vyšší podíl
SI+. Uvnitř pásem je posun řádově menší, ale ve všech pásmech kladný. To je stejný slabý
signál, jaký ukazuje *within-band* AUC v části 6.

## 9. Citlivost na α

Pravidlo cmeAnalysis nevrací spojité skóre, ale přímo binární rozhodnutí. Jeho jediným
parametrem citlivosti je hladina α statistického testu, výchozí hodnota je 0,05. Sweep přes α
je tedy obdobou *threshold-sensitivity* analýzy.

{t_sw}

*Tabulka 6: Vliv hladiny α na podíl pozitivních a míry shody.*

![alpha sweep](fig5_alpha.png)

*Obrázek 8: Vlevo podíly pozitivních drah podle α, vpravo míry shody podle α.*

**Diskuze:**

Křivky jsou téměř ploché. Amplitudy jsou totiž proti nejistotě fitu obrovské a p-hodnoty
jednotlivých snímků leží prakticky jen u nuly, nebo u jedničky. Posun α přes dva řády
přeznačí jen zlomek snímků. Slabá shoda se SI tedy není důsledkem nevhodně zvolené
citlivosti. Současně z toho plyne, že binární call nemá použitelný rozsah operačních bodů.

## 10. Shrnutí

1. Výchozí binární call cmeAnalysis se pro rozlišení produktivních drah nehodí. Označí přes
   polovinu drah, jeho operační bod nelze hladinou α posunout a délkou dráhy je tažen více
   než SI (tabulka 2, část 9).
2. Amplituda použitelná je, avšak signál v ní je slabý. Na spojité škále i na stejném
   operačním bodě dává v mezích intervalů spolehlivosti stejný výsledek jako *box-mean*
   (tabulka 4). Oba readouty přitom měří stejnou veličinu (část 7).
3. Posun amplitud mezi SI třídami je prokazatelný, ale malý (tabulka 5).
4. Délková závislost vzniká již na úrovni jednotlivých snímků (obrázek 3). Usuzujeme, že jde
   o vlastnost prostředí drah, ne o selhání binomické korekce.
5. Ani jedna štítek není *ground truth*. Podle modelu skupiny může přibližně pětina
   abortivních drah legitimně nést dynamin, a ani dokonalá měření by proto nedala shodu 100 %.

"""


# =============================================================== TeX
TEX_HEAD = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.2cm]{geometry}
\usepackage{fontspec}
\usepackage[czech]{babel}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{float}
\usepackage[hidelinks]{hyperref}
\raggedbottom
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\begin{document}
"""


def tex_escape(s):
    return (s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
             .replace("#", r"\#").replace("κ", r"$\kappa$").replace("α", r"$\alpha$")
             .replace("ρ", r"$\rho$").replace("σ", r"$\sigma$").replace("×", r"$\times$")
             .replace("→", r"$\rightarrow$").replace("−", "--").replace("|", r"$\mid$"))


def tex_table(rows, colspec, header, caption, label):
    body = "\n".join(" & ".join(tex_escape(str(v)) for v in r) + r" \\" for r in rows)
    return (f"\\begin{{table}}[H]\\centering\\small\n"
            f"\\begin{{tabular}}{{{colspec}}}\n\\toprule {header} \\\\ \\midrule\n"
            f"{body}\n\\bottomrule\\end{{tabular}}\n"
            f"\\caption{{{tex_escape(caption)}}}\\label{{{label}}}\\end{{table}}")


def tex_fig(fname, caption, label):
    return (f"\\begin{{figure}}[H]\\centering"
            f"\\includegraphics[width=\\linewidth]{{{fname}}}"
            f"\\caption{{{tex_escape(caption)}}}\\label{{{label}}}\\end{{figure}}")


def build_tex(M, CI, df, sweep_rows, hl_rows, ve, cov, args, meta):
    e = tex_escape
    have_box = meta["have_box"]
    t_cov = tex_table(coverage_table_rows(*cov), "lrrrr",
                      "film & drah & snímků & selhaných fitů & drah bez SI",
                      "Pokrytí dat po filmech.", "tab:cov")
    t_met = tex_table(metric_table_rows(M, CI), "p{8.6cm}rl", "metrika & hodnota & 95\\,\\% CI",
                      f"Metriky výchozího callu s 95% intervaly spolehlivosti "
                      f"(bootstrap {args.n_boot} tahů po filmech).", "tab:met")
    t_band = tex_table(band_table_rows(df), "lrrrr",
                       "délka (snímky) & n & dynamin+ & SI+ & OR v pásmu",
                       "Podíly pozitivních drah a odds ratio po délkových pásmech.", "tab:band")
    t_amp = tex_table(matched_table_rows(M, CI, have_box), "p{8.6cm}rl",
                      "metrika & hodnota & 95\\,\\% CI",
                      "Amplitudová větev; oba readouty na stejných drahách a stejném "
                      "operačním bodě.", "tab:amp")
    t_hl = tex_table(hl_table_rows(hl_rows), "lrrrrl",
                     "délka & n & med. SI+ & med. SI-- & HL posun [ADU] & 95\\,\\% CI",
                     f"Posun amplitud SI+ proti SI-- po pásmech. Souhrnný stratifikovaný test "
                     f"(van Elteren): z = {cz(ve[0])}, p $\\approx$ {sci_tex(ve[1])}; p-hodnota nezohledňuje "
                     f"clustering po filmech, intervaly posunů ano.", "tab:hl")
    t_sw = tex_table(sweep_table_rows(sweep_rows), "lrrrrrr",
                     "$\\alpha$ & dyn+ & P(dyn+$\\mid$SI+) & P(dyn+$\\mid$SI--) & $\\kappa$ & OR očišt. & wb AUC",
                     "Vliv hladiny $\\alpha$ na podíl pozitivních a míry shody.", "tab:sw") \
        if sweep_rows else ""

    f_mosaic = tex_fig("fig1_mosaic.png",
                       f"Confusion matice výchozího pravidla, poolovaná a po délkových pásmech. "
                       f"Poolované odds ratio {cz(M['or_crude'])} se po očištění o délku "
                       f"(Mantel--Haenszel, 12 strat) snižuje na {cz(M['or_mh'])} "
                       f"[{cz(CI['or_mh'][0])}; {cz(CI['or_mh'][1])}].", "fig:mosaic")
    f_len = tex_fig("fig2_length.png", "Vlevo podíly pozitivních drah po pásmech, vpravo odds "
                    "ratio v pásmech proti poolované a očištěné hodnotě.", "fig:len")
    f_pf = tex_fig("fig8_perframe.png", "Podíl významných snímků roste s délkou dráhy v obou "
                   "SI třídách; produktivní dráhy leží o několik procentních bodů výš.", "fig:pf")
    f_films = tex_fig("fig3_films.png", "Podíl pozitivních drah po filmech; každý bod je jeden "
                      "film, velikost bodu odpovídá počtu drah. Všechny body leží vysoko nad "
                      "diagonálou shodných podílů.", "fig:films")
    f_amp = tex_fig("fig6_matched.png", "Vlevo AUC spojité amplitudy (pooled a within-band), "
                    "vpravo odds ratio po očištění o délku pro výchozí call a oba q90 cally.",
                    "fig:amp")
    f_agree = tex_fig("fig7_agreement.png",
                      f"Maximální amplituda cmeAnalysis proti maximálnímu box-mean na stejné "
                      f"dráze (obě osy logaritmicky, barva udává počet drah). Spearmanova "
                      f"korelace je {cz(meta['rho_track'])} po drahách a {cz(meta['rho_frame'])} "
                      f"po snímcích.", "fig:agree") if have_box else ""
    f_hist = tex_fig("fig4_amplitude.png",
                     f"Vlevo rozdělení maximální amplitudy podle SI třídy (mediány SI+ "
                     f"{fmt_n(meta['med_si'])}\\,ADU a SI-- {fmt_n(meta['med_no'])}\\,ADU; "
                     f"Mann--Whitney {meta['mw_p_tex']}). Vpravo Hodgesovy--Lehmannovy posuny "
                     f"po pásmech s 95% intervaly po filmech.", "fig:hist")
    f_al = tex_fig("fig5_alpha.png", "Vlevo podíly pozitivních drah podle $\\alpha$, vpravo "
                   "míry shody podle $\\alpha$.", "fig:al")

    box_diskuze = ""
    if have_box:
        box_diskuze = (
            "\\textbf{Diskuze:}\n\n"
            "Z tabulky~\\ref{tab:amp} a obrázku~\\ref{fig:amp} plyne, že oba \\emph{readouty} "
            "dávají v~mezích intervalů spolehlivosti stejný výsledek. Platí to pro spojité "
            "hodnocení i pro stejný operační bod. Hodnota OR pro \\emph{box-mean} na našich "
            "drahách současně odpovídá hodnotě publikované dříve na plném korpusu (přibližně "
            "1{,}5), což naši repliku nezávisle potvrzuje. Usuzujeme tedy, že rozdíl mezi "
            "částí~3 a předchozí analýzou nebyl vlastností měření, ale operačního bodu "
            "výchozího pravidla.\n")
    agree_part = ""
    if have_box:
        agree_part = (
            "\\section*{7\\; Shoda readoutů}\n"
            "Zbývá ověřit, že oba \\emph{readouty} měří stejnou veličinu.\n"
            + f_agree + "\n"
            "\\textbf{Diskuze:}\n\n"
            "Korelace je po drahách vysoká a po snímcích střední, viz popisek "
            "obrázku~\\ref{fig:agree}. Dá se říct, že jde o dvě měření stejné veličiny s~jinou "
            "citlivostí na pozadí, ne o dvě různé veličiny. To je v~souladu s~tím, že "
            "v~části~6 dávají oba \\emph{readouty} stejné výsledky.\n")

    return TEX_HEAD + f"""
{{\\LARGE\\bfseries Porovnání binárních klasifikací:\\\\dynamin (cmeAnalysis) proti Shape Indexu}}\\\\[4pt]
{{\\small Datum {meta['date']} \\;·\\; CMEpython {meta['git']} \\;·\\;
\\texttt{{scripts/report\\_classification\\_comparison.py}}}}

\\section*{{1\\; Co se porovnává a proč}}
Každé dráze klatrinové jamky přiřadíme dva nezávislé štítky a změříme, jak moc se shodují.
\\par
Nejprve štítek tvarový. Dráhu nazveme \\emph{{produktivní}}, pokud její Shape Index někdy za život
překročí {cz(args.si_threshold, 1)}. Dále štítek dynaminový podle pravidla cmeAnalysis s~výchozími
parametry. V~každém snímku nafitujeme na pozici jamky skvrnku tvaru PSF ({e(meta['sigma_note'])})
a získáme amplitudu nad lokálním pozadím. Dráhu označíme za \\emph{{dynamin-pozitivní}}, když počet
snímků s~amplitudou významně nad 95.~percentilem pozadí filmu překročí binomický práh odpovídající
délce dráhy. Jde o test ,,s``, tzv. \\emph{{significantSlave}}, což je výchozí režim cmeAnalysis.
\\par
Pracujeme s~{meta['n_films']} filmy ,,lr registered`` (U2OS, 2\\,s/snímek, 79{{,}}1\\,nm/px)
a {e(fmt_n(M['n']))} dodanými drahami. Dráhy jsou předfiltrované již od dodavatele, a to na délku
přes 5 snímků, polohu uvnitř masky a úplnost v~záznamu. Žádný další délkový filtr neaplikujeme.
Pozadí filmů pro klasifikaci počítáme s~opravenými buněčnými maskami (,,masks fixed``; původní
masky byly poškozené interpolací při uložení, překryv s~opravenými je podle filmu 0{{,}}89 až
0{{,}}97). Poznamenejme, že ani jeden štítek není \\emph{{ground truth}} (skutečný stav).
Měříme tedy shodu dvou nedokonalých měření, ne správnost jednoho z~nich.

\\subsection*{{Pokrytí}}
{t_cov}
\\textbf{{Diskuze:}}

Z~tabulky~\\ref{{tab:cov}} plyne, že fit neselhal na žádném snímku a oba štítky má každá dodaná
dráha. Fit totiž selhává jen na okraji obrazu a dodané dráhy jsou od okraje odfiltrované.
Nevzniká tedy žádný dodatečný výběr, který by mohl korelovat s~délkou dráhy.

\\section*{{2\\; Výchozí binární call}}
Obrázek~\\ref{{fig:mosaic}} čteme takto. Sloupce dělí dráhy podle SI na abortivní a produktivní.
Řádky je dělí podle dynaminového štítku. Procenta udávají podíl ve sloupci, tedy jakou část
abortivních, resp. produktivních drah pravidlo označilo za pozitivní. První panel ukazuje všechny
dráhy dohromady. Další čtyři panely ukazují totéž po délkových pásmech.
{f_mosaic}
\\textbf{{Diskuze:}}

Z~obrázku~\\ref{{fig:mosaic}} plyne dvojí. Za prvé, dynamin-pozitivní je většina produktivních
drah, avšak současně i polovina abortivních. Za druhé, poolovaná matice naznačuje silnější
asociaci, než jaká platí uvnitř pásem. Oba štítky totiž vznikají operátorem typu ,,stalo se to
někdy za život``, a proto rostou s~délkou dráhy samy od sebe. Smícháním krátkých a dlouhých drah
pak vzniká zdánlivá asociace, tzv. Simpsonův jev. Po stratifikaci délkou zbývá odds ratio, jehož
interval spolehlivosti zahrnuje jedničku, viz popisek obrázku~\\ref{{fig:mosaic}}.
\\par
Poznamenejme ještě jednu věc. Výchozí pravidlo označí za pozitivní přes polovinu drah
(tabulka~\\ref{{tab:met}}). \\emph{{Box-mean}} call z~předchozí analýzy má z~konstrukce pozitivních
jen desetinu. Přímé srovnání $\\kappa$ mezi oběma pravidly by proto nebylo korektní. Srovnáváme je
až na stejném operačním bodě v~části~6.

\\section*{{3\\; Metriková tabulka výchozího callu}}
Tabulka~\\ref{{tab:met}} uvádí všechny běžně reportované metriky. Dynaminový štítek v~ní
hodnotíme jako ,,prediktor`` SI štítku; jde o konvenci zobrazení. Intervaly spolehlivosti
počítáme bootstrapem ({args.n_boot} tahů), ve kterém resamplujeme celé filmy. Dráhy stejného filmu
totiž nejsou nezávislé.
{t_met}
\\textbf{{Diskuze:}}

Z~tabulky~\\ref{{tab:met}} plyne trojí. Za prvé, shoda nad rámec náhody je nízká (Cohenovo
$\\kappa$). Za druhé, \\emph{{within-band}} AUC, tedy AUC počítaná jen mezi drahami podobné délky,
klesá téměř k~0{{,}}5. Rozdíl mezi poolovanou a \\emph{{within-band}} hodnotou říká, kolik zdánlivé
shody dodávala délka. Za třetí, samotná délka dráhy predikuje dynaminový štítek lépe než štítek
SI. Dynaminové pravidlo je tedy délkou tažené více než SI, které má ověřovat.

\\section*{{4\\; Závislost na délce dráhy}}
{t_band}
{f_len}
Obrázek~\\ref{{fig:pf}} ukazuje podíl snímků významně nad pozadím v~závislosti na délce dráhy,
zvlášť pro obě SI třídy. Svislé úsečky jsou střední chyby průměru.
{f_pf}
\\textbf{{Diskuze:}}

Pravidlo cmeAnalysis délku explicitně zohledňuje, binomický práh totiž s~délkou roste. Podíl
pozitivních drah přesto roste napříč pásmy (tabulka~\\ref{{tab:band}}). Z~obrázku~\\ref{{fig:pf}}
usuzujeme proč. S~délkou roste již pozitivita jednotlivých snímků a binomická korekce pak nemá co
korigovat. Produktivní dráhy leží o~několik procentních bodů výš, což odpovídá slabému
amplitudovému signálu z~části~6. Dominantní je ovšem společný růst obou tříd. Dlouhé dráhy tedy
leží v~místech s~více dynaminem. Může jít o vlastnost membránové oblasti. Současně může jít
o selekci trackování, dráhy v~jasných místech se totiž snáze trackují dlouho. Rozhodnout mezi
těmito vysvětleními z~těchto dat neumíme.

\\section*{{5\\; Rozdíly mezi filmy}}
{f_films}
\\textbf{{Diskuze:}}

Ve všech filmech označí pravidlo cmeAnalysis výrazně více drah než SI
(obrázek~\\ref{{fig:films}}). Rozptyl mezi filmy je současně značný. Právě proto počítáme všechny
intervaly spolehlivosti bootstrapem po filmech, ne po drahách.

\\section*{{6\\; Amplitudová větev na stejném operačním bodě}}
Binární call cmeAnalysis nemá použitelný rozsah operačních bodů, jak ukážeme v~části~9. Korektní
srovnání s~\\emph{{box-mean}} callem proto postavíme jinak. Nejprve vezmeme maximální amplitudu na
dráhu a práh položíme na její 90.~percentil. Tím dostaneme stejný podíl pozitivních, jaký má
\\emph{{box-mean}} call, a to desetinu drah. Dále hodnotíme amplitudu i spojitě, tedy bez prahu,
pomocí AUC. Aby srovnání nezáviselo na korpusu, spočítali jsme \\emph{{box-mean}} 5$\\times$5
vlastní implementací na stejných drahách a filmech, a to jako průměr 5$\\times$5\\,px dynaminového
kanálu dělený biexponenciálním fitem průměrů snímků buňky. Oběma způsobům odečtu říkáme dále
\\emph{{readouty}}; readout je způsob, jakým se z~obrazu získá číslo ,,kolik dynaminu je na
daném místě v~daném snímku`` (průměr okénka u~box-mean, amplituda PSF fitu u~cmeAnalysis).
{t_amp}
{f_amp}
{box_diskuze}
{agree_part}
\\section*{{8\\; Amplituda dynaminu podle SI třídy}}
{f_hist}
{t_hl}
\\textbf{{Diskuze:}}

Posun amplitud mezi třídami je prokazatelný, avšak malý, viz tabulka~\\ref{{tab:hl}}. Poolovaný
rozdíl mediánů je z~velké části opět efekt délky. Delší dráhy mají totiž vyšší maximum i vyšší
podíl SI+. Uvnitř pásem je posun řádově menší, ale ve všech pásmech kladný. To je stejný slabý
signál, jaký ukazuje \\emph{{within-band}} AUC v~části~6.

\\section*{{9\\; Citlivost na $\\alpha$}}
Pravidlo cmeAnalysis nevrací spojité skóre, ale přímo binární rozhodnutí. Jeho jediným parametrem
citlivosti je hladina $\\alpha$ statistického testu, výchozí hodnota je 0{{,}}05. Sweep přes
$\\alpha$ je tedy obdobou \\emph{{threshold-sensitivity}} analýzy.
{t_sw}
{f_al}
\\textbf{{Diskuze:}}

Křivky jsou téměř ploché. Amplitudy jsou totiž proti nejistotě fitu obrovské a p-hodnoty
jednotlivých snímků leží prakticky jen u~nuly, nebo u~jedničky. Posun $\\alpha$ přes dva řády
přeznačí jen zlomek snímků. Slabá shoda se SI tedy není důsledkem nevhodně zvolené citlivosti.
Současně z~toho plyne, že binární call nemá použitelný rozsah operačních bodů.

\\section*{{10\\; Shrnutí}}
\\begin{{enumerate}}
\\item Výchozí binární call cmeAnalysis se pro rozlišení produktivních drah nehodí. Označí přes
polovinu drah, jeho operační bod nelze hladinou $\\alpha$ posunout a délkou dráhy je tažen více
než SI (tabulka~\\ref{{tab:met}}, část~9).
\\item Amplituda použitelná je, avšak signál v~ní je slabý. Na spojité škále i na stejném
operačním bodě dává v~mezích intervalů spolehlivosti stejný výsledek jako \\emph{{box-mean}}
(tabulka~\\ref{{tab:amp}}). Oba \\emph{{readouty}} přitom měří stejnou veličinu (část~7).
\\item Posun amplitud mezi SI třídami je prokazatelný, ale malý (tabulka~\\ref{{tab:hl}}).
\\item Délková závislost vzniká již na úrovni jednotlivých snímků (obrázek~\\ref{{fig:pf}}).
Usuzujeme, že jde o vlastnost prostředí drah, ne o selhání binomické korekce.
\\item Ani jedna štítek není \\emph{{ground truth}}. Podle modelu skupiny může přibližně pětina
abortivních drah legitimně nést dynamin, a ani dokonalá měření by proto nedala shodu 100\\,\\%.
\\end{{enumerate}}

\\end{{document}}
"""


# =============================================================== main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measured", default=os.path.join(ROOT, "lr registered", "measured"))
    ap.add_argument("--out", default=os.path.join(ROOT, "lr registered", "comparison_report"))
    ap.add_argument("--si-threshold", type=float, default=0.7)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--n-boot-hl", type=int, default=200)
    ap.add_argument("--alphas", default="0.001,0.005,0.01,0.05,0.1")
    ap.add_argument("--calibration", default=os.path.join(ROOT, "lr registered", "psf_calibration_lr.json"))
    ap.add_argument("--skip-pdf", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_classified(args.measured, si_threshold=args.si_threshold)
    if df is None:
        sys.exit(f"zadne *-classified.csv v {args.measured}")

    peaks, box_frames = load_boxpeaks(args.measured)
    have_box = peaks is not None
    if have_box:
        df = df.merge(peaks, on=["film", "particle"], how="left")
        print(f"box-mean: {np.isfinite(df.box_peak).sum():,} / {len(df):,} drah ma box peak")
    else:
        print("VAROVANI: *-boxmean.csv nenalezeny (scripts/compute_boxmean.py) -- "
              "casti 6/7 budou bez box-mean vetve")

    cov = coverage_rows(args.measured)
    M = draw_metrics(df, have_box)
    M["cells"] = counts(df.si_pos.to_numpy(bool), df.cme_pos.to_numpy(bool))
    print(f"{df.film.nunique()} filmu, {fmt_n(M['n'])} drah; bootstrap {args.n_boot} tahu...", flush=True)
    CI = film_bootstrap_multi(df, have_box, n_boot=args.n_boot)

    print("Hodges-Lehmann po pasmech...", flush=True)
    hl_rows = hl_by_band(df, n_boot=args.n_boot_hl)
    ve = van_elteren(df)
    print(f"van Elteren: z={ve[0]:.2f} p={ve[1]:.2g}")

    sweep_rows = []
    for al in (float(x) for x in args.alphas.split(",")):
        suffix = "-classified.csv" if abs(al - 0.05) < 1e-12 else f"-classified-a{al:g}.csv"
        dfa = load_classified(args.measured, suffix=suffix, si_threshold=args.si_threshold)
        if dfa is None:
            print(f"  alpha {al:g}: soubory *{suffix} nenalezeny -- vynechavam", flush=True)
            continue
        ma = base_metrics(dfa)
        ma["alpha"] = al
        sweep_rows.append(ma)
    sweep_rows.sort(key=lambda r: r["alpha"])

    si_amp, no_amp = df[df.si_pos]["max_A"].dropna(), df[~df.si_pos]["max_A"].dropna()
    meta = {
        "date": _dt.date.today().isoformat(),
        "git": subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "?",
        "n_films": int(df.film.nunique()),
        "med_si": float(np.median(si_amp)), "med_no": float(np.median(no_amp)),
        "mw_p": float(mannwhitneyu(si_amp, no_amp, alternative="greater").pvalue),
        "rho_cme": float(spearmanr(df.track_len, df.cme_pos).correlation),
        "rho_si": float(spearmanr(df.track_len, df.si_pos).correlation),
        "sigma_note": "σ z kalibrace datasetu",
        "have_box": have_box, "rho_track": np.nan, "rho_frame": np.nan,
    }
    if meta["mw_p"] < 1e-300:
        meta["mw_p_md"], meta["mw_p_tex"] = "p < 10⁻³⁰⁰", r"$p<10^{-300}$"
    else:
        meta["mw_p_md"], meta["mw_p_tex"] = f"p = {meta['mw_p']:.2g}", f"$p={meta['mw_p']:.2g}$"
    if os.path.exists(args.calibration):
        import json as _json
        with open(args.calibration, encoding="utf-8") as fh:
            cal = _json.load(fh)
        try:
            meta["sigma_note"] = f"σ dynaminu {cz(cal['1']['sigma_clamped'])} px z kalibrace datasetu"
        except (KeyError, TypeError):
            pass

    fig_mosaic(df, os.path.join(args.out, "fig1_mosaic.png"))
    fig_length(df, M, os.path.join(args.out, "fig2_length.png"))
    fig_films(df, os.path.join(args.out, "fig3_films.png"))
    fig_amplitude(df, hl_rows, os.path.join(args.out, "fig4_amplitude.png"))
    if sweep_rows:
        fig_alpha(sweep_rows, os.path.join(args.out, "fig5_alpha.png"))
    fig_matched(M, CI, have_box, os.path.join(args.out, "fig6_matched.png"))
    if have_box:
        mfiles = sorted(glob.glob(os.path.join(args.measured, "*-lr-trajectories-dynamin.csv")))
        samples_c, samples_b = [], []
        rng = np.random.default_rng(0)
        for f in mfiles:
            film = int(re.search(r"_(\d+)-", os.path.basename(f)).group(1))
            dm = pd.read_csv(f, usecols=["particle", "frame", "dnm_A"])
            bx = box_frames[box_frames.film == film]
            j = dm.merge(bx, on=["particle", "frame"], how="inner").dropna()
            if len(j) > 8000:
                j = j.iloc[rng.choice(len(j), 8000, replace=False)]
            samples_c.append(j["dnm_A"].to_numpy())
            samples_b.append(j["box5_norm"].to_numpy())
        meta["rho_track"], meta["rho_frame"] = fig_agreement(
            df, np.concatenate(samples_c), np.concatenate(samples_b),
            os.path.join(args.out, "fig7_agreement.png"))
    fig_perframe(df, os.path.join(args.out, "fig8_perframe.png"))

    with open(os.path.join(args.out, "protokol.md"), "w", encoding="utf-8") as fh:
        fh.write(build_markdown(M, CI, df, sweep_rows, hl_rows, ve, cov, args, meta))
    with open(os.path.join(args.out, "protokol.tex"), "w", encoding="utf-8") as fh:
        fh.write(build_tex(M, CI, df, sweep_rows, hl_rows, ve, cov, args, meta))

    if not args.skip_pdf:
        if shutil.which("latexmk"):
            r = subprocess.run(["latexmk", "-xelatex", "-interaction=nonstopmode",
                                "-halt-on-error", "-cd", os.path.join(args.out, "protokol.tex")],
                               capture_output=True, text=True)
            if r.returncode == 0:
                subprocess.run(["latexmk", "-c", "-cd", os.path.join(args.out, "protokol.tex")],
                               capture_output=True, text=True)
                print("PDF: protokol.pdf")
            else:
                print("VAROVANI: xelatex selhal, PDF nevzniklo; posledni radky logu:")
                print("\n".join(r.stdout.splitlines()[-15:]))
        else:
            print("VAROVANI: latexmk neni k dispozici, PDF preskoceno")

    print(f"\nhotovo -> {args.out}/ (protokol.md, protokol.pdf, fig1..fig8)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
