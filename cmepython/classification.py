# -*- coding: utf-8 -*-
# CMEpython -- Python port mereni intenzity dynaminu z cmeAnalysis.
# Copyright (C) 2026 Ruslan Guliev
#
# Odvozeno z cmeAnalysis (DanuserLab, UT Southwestern), GPL-3.0.
# Tento program je svobodny software: muzete jej sirit a upravovat podle
# podminek GNU General Public License verze 3 nebo pozdejsi. Sirenu je
# BEZ JAKEKOLI ZARUKY. Viz soubor LICENSE.
"""Klasifikace dynamin-pozitivnich drah -- port runSlaveChannelClassification.m.

Aguetova statisticka definice (Aguet et al. 2013, Methods; zadani na ni
odkazuje jako na "druhy relevantni odstavec Methods"). Dva nezavisle testy
na trajektorii, oba binomicky korigovane na jeji delku:

  1. significant_master (:141-153) -- je pocet vyznamnych detekci ve slave
     kanale podel drahy vyssi, nez kolik by jich nahodne nasbirala draha
     teze delky? Ocekavanou nahodu dava `p_detection` -- empiricka mira
     vyznamnych pixelu na celem snimku, zmerena pres 16 snimku filmu.

  2. significant_slave (:155-173) -- je pocet bodu s amplitudou vyznamne
     nad 95. percentilem pozadi (`bg95`) vyssi nez ocekavanych 5 %
     falesnych pozitiv?

Obe globalni veliciny (p_detection, bg95) se meri na pixelech UVNITR bunky
a MIMO okoli CCS struktur -- viz `background_stats`.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom as _binom
from scipy.stats import norm as _norm
from scipy.stats import t as _student_t

from .background import build_ccp_mask, cell_mask_from_movie, filter_gaussian_fit_2d
from .slave_intensity import _mround

__all__ = ["background_stats", "classify_track", "classify_tracks"]


def background_stats(path, positions_by_frame, sigma_slave,
                     slave_channel=2, master_channel=0,
                     cellmask=None, n_frames=16, alpha=0.05):
    """Rozdeleni pozadi a mira falesnych detekci -- runSlaveChannelClassification.m:46-116.

    Parameters
    ----------
    path : cesta k filmu (TIFF)
    positions_by_frame : dict {frame: (ys, xs)} -- pozice CCS struktur
        (z trajektorii); okoli techto bodu se z pozadi vyradi
    sigma_slave : sirka PSF mericiho kanalu (urcuje polomer vyrazeni
        w = ceil(4*sigma), :56)
    cellmask : bool pole, nebo None -> spocita se z max-projekce master
        kanalu (getCellMask, :47)

    Returns
    -------
    dict: bg95, p_detection, cellmask, n_bg_pixels, per_frame (diagnostika)
    """
    import tifffile
    from .measure import movie_layout

    if cellmask is None:
        cellmask, _, _ = cell_mask_from_movie(path, master_channel)
    cellmask = np.asarray(cellmask, dtype=bool)

    w = int(np.ceil(4.0 * float(sigma_slave)))          # :56

    with tifffile.TiffFile(str(path)) as tf:
        T, C, Y, X, idx = movie_layout(tf)
        # :60 -- 16 snimku rovnomerne pres film (MATLAB round(linspace)).
        # ZADNA deduplikace: pro filmy s T < 16 MATLAB duplicitni snimky
        # zapocitava vicenasovne (v prumeru pDetection i v percentilu bg).
        frame_idx = np.round(np.linspace(0, T - 1, n_frames)).astype(int)

        bg_all = []
        p_det = []
        per_frame = []
        for t in frame_idx:
            frame = tf.pages[idx(int(t), slave_channel)].asarray().astype(np.float64)

            # :74-81 -- CCS maska dilatovana o w, pozadi = bunka minus CCS
            ys, xs = positions_by_frame.get(int(t), ((), ()))
            ccp = build_ccp_mask((Y, X), ys, xs, w)
            bg_mask = cellmask & ~ccp

            # :96-102. DVE PASTI:
            # (a) MATLAB vola filterGaussianFit2D(frame, sigma) BEZ predani
            #     Alpha -- kLevel je vzdy norminv(0.975) a uzivatelska alpha
            #     vstupuje jen do porovnani pval < alpha. Proto zde default.
            # (b) `sum(pval_Ar < Alpha) / sum(cellmask)` na 2D maticich NENI
            #     podil souctu: sum() vraci radkove vektory a `/` mezi nimi
            #     je mrdivide (nejmensi ctverce), tedy dot(a,b)/dot(b,b) se
            #     sloupcovymi soucty a, b. Overeno primo v MATLABu. Port je
            #     bit-verny VYKONANEMU kodu, ne komentovanemu zameru --
            #     rozdil je ~6 % relativne a posouva binomicky prah o 1
            #     u nekterych delek drah.
            A_est, _, pval = filter_gaussian_fit_2d(frame, sigma_slave)
            bg_all.append(A_est[bg_mask])
            a_cols = (pval < alpha).sum(axis=0).astype(float)
            b_cols = cellmask.sum(axis=0).astype(float)
            p = float(a_cols @ b_cols) / float(b_cols @ b_cols)
            p_det.append(p)
            per_frame.append(dict(frame=int(t), p_detection=p,
                                  n_bg=int(bg_mask.sum())))

    bg_all = np.concatenate(bg_all)
    return dict(
        bg95=float(np.percentile(bg_all, 95, method="hazen")),   # :116, prctile
        p_detection=float(np.mean(p_det)),                       # :106
        cellmask=cellmask,
        n_bg_pixels=int(bg_all.size),
        per_frame=per_frame,
    )


def classify_track(A, A_pstd, sigma_r, SE_sigma_r, hval_Ar,
                   bg95, p_detection,
                   alpha_sensitivity=0.05, amplitude_ratio=0.0,
                   master_max_A=None):
    """Oba testy pro jednu drahu -- runSlaveChannelClassification.m:125-174.

    Vstupy jsou vektory podel drahy (hodnoty z mereni intenzity; presne
    sloupce dnm_* z measured/*.csv). NaN body (okraj obrazu) se chovaji
    jako v MATLABu: nansum je ignoruje, delka drahy L je ale pocita.

    master_max_A : max amplitudy master kanalu na draze, pro podminku
        amplitudoveho pomeru (:151-153). Pri None se pouzije jen znamenko
        max slave amplitudy -- s vychozim amplitude_ratio=0 je to
        ekvivalentni (master max je vzdy kladny), pro nenulovy pomer je
        master_max_A povinny.
    """
    A = np.asarray(A, dtype=float)
    A_pstd = np.asarray(A_pstd, dtype=float)
    sigma_r = np.asarray(sigma_r, dtype=float)
    SE_sigma_r = np.asarray(SE_sigma_r, dtype=float)
    hval = np.asarray(hval_Ar, dtype=float)

    L = A.size                                           # :127

    # :135 -- npx zpetne z pomeru sigma_r/SE_sigma_r, MATLAB round()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio2 = (sigma_r / SE_sigma_r) ** 2
    npx = np.array([_mround(v / 2.0 + 1.0) if np.isfinite(v) else np.nan
                    for v in ratio2])

    # ---- test 1: pocet detekci vs nahoda (:145) ---------------------------
    n_detected = float(np.nansum(hval))
    thr_master = float(_binom.ppf(0.95, L, min(max(p_detection, 0.0), 1.0)))
    significant_master = n_detected > thr_master

    # :151-153 -- podminka amplitudoveho pomeru
    max_A = np.nanmax(A) if np.isfinite(A).any() else np.nan
    if master_max_A is not None:
        amp_ratio = max_A / master_max_A
    elif amplitude_ratio == 0.0:
        amp_ratio = max_A          # master max je kladny; rozhoduje znamenko
    else:
        raise ValueError("nenulovy amplitude_ratio vyzaduje master_max_A "
                         "(max amplitudy master kanalu na draze)")
    significant_master = bool(significant_master and (amp_ratio > amplitude_ratio))

    # ---- test 2: amplitudy vs 95. percentil pozadi (:158-173) -------------
    with np.errstate(divide="ignore", invalid="ignore"):
        SE_bg = bg95 / np.sqrt(2.0 * npx - 1.0)                    # :160
        df2 = (npx - 1.0) * (A_pstd ** 2 + SE_bg ** 2) ** 2 \
            / (A_pstd ** 4 + SE_bg ** 4)                           # :163
        scomb = np.sqrt((A_pstd ** 2 + SE_bg ** 2) / npx)          # :164
        T = (A - bg95) / scomb                                     # :165
        pval = _student_t.cdf(-T, df2)                             # :166
    hval_bg = pval < alpha_sensitivity                             # :167
    n_above = int(np.nansum(hval_bg))
    thr_slave = float(_binom.ppf(0.95, L, 0.05))                   # :173
    significant_slave = bool(n_above > thr_slave)

    return dict(
        significant_master=bool(significant_master),
        significant_slave=significant_slave,
        n_detected=int(n_detected), thr_master=thr_master,
        n_above_bg=n_above, thr_slave=thr_slave,
        track_len=int(L), max_A=float(max_A) if np.isfinite(max_A) else np.nan,
        significant_vs_background=hval_bg,                         # :168
    )


def classify_tracks(rows_by_particle, bg95, p_detection, **kw):
    """Klasifikace vsech drah filmu.

    rows_by_particle : dict {particle: dict s vektory A, A_pstd, sigma_r,
        SE_sigma_r, hval_Ar} -- napr. nactene z measured/*.csv
    Vraci dict {particle: vysledek classify_track} (bez per-bodoveho pole
    significant_vs_background, aby vystup zustal maly).
    """
    out = {}
    for pid, d in rows_by_particle.items():
        r = classify_track(d["A"], d["A_pstd"], d["sigma_r"],
                           d["SE_sigma_r"], d["hval_Ar"],
                           bg95, p_detection, **kw)
        r.pop("significant_vs_background")
        out[pid] = r
    return out
