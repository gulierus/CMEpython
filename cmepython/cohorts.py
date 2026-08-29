# -*- coding: utf-8 -*-
# CMEpython -- Python port mereni intenzity dynaminu z cmeAnalysis.
# Copyright (C) 2026 Ruslan Guliev
#
# Odvozeno z cmeAnalysis (DanuserLab, UT Southwestern), GPL-3.0.
# Tento program je svobodny software: muzete jej sirit a upravovat podle
# podminek GNU General Public License verze 3 nebo pozdejsi. Sirenu je
# BEZ JAKEKOLI ZARUKY. Viz soubor LICENSE.
"""Prubehy intenzity podle lifetime kohort -- port getIntensityCohorts.m
a statisticke casti plotIntensityCohorts.m.

Myslenka: drahy ruzne dlouhe se nedaji prumerovat primo. cmeAnalysis je
proto rozdeli do KOHORT podle zivotnosti (v sekundach), kazdou drahu
prevzorkuje na spolecnou delku sve kohorty a teprve pak prumeruje.

Konvence prevzate 1:1 z originalu:

  * hranice kohort  CohortBounds_s = [10 20 40 60 80 100 120] s
                    (getIntensityCohorts.m:30); zarazeni dolni mez
                    vcetne, horni mez bez; posledni horni mez se
                    rozsiri o jeden snimek (:70, :82)
  * delka kohorty   iLength = floor(mean(hranic)/framerate) + 2*b
                    (:64), b = pocet buffer snimku pred/po draze
                    (runTrackProcessing 'Buffer' = 5)
  * prevzorkovani   [start buffer, draha, end buffer] -> iLength bodu
                    kubickou interpolaci (binterp, :99-100)
  * osa casu        (-b : iLength-b-1) * framerate -- t = 0 je prvni
                    snimek drahy, buffer je v zapornem case (:67;
                    plotIntensityCohorts.m:103 pro Align='left')
  * statistika      prumer pres drahy v KAZDEM FILMU zvlast, pak prumer
                    a SEM PRES FILMY (plotIntensityCohorts.m:365-372:
                    SEM = nanstd(AMat)/sqrt(nd), nd = pocet filmu).
                    Alternativa 'pct': 25/50/75. percentil pres drahy
                    (:375).

Zavislosti: numpy, scipy.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

__all__ = [
    "DEFAULT_BOUNDS_S",
    "assign_cohorts",
    "cohort_lengths",
    "cohort_time_axis",
    "resample_track",
    "cohort_curves",
]

DEFAULT_BOUNDS_S = (10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0)


def assign_cohorts(lifetimes_s, bounds_s=DEFAULT_BOUNDS_S, framerate=2.0):
    """getIntensityCohorts.m:70-84. Vraci index kohorty 0..nc-1, nebo -1
    pro drahy mimo rozsah (kratsi nez prvni mez, delsi nez posledni)."""
    lt = np.asarray(lifetimes_s, dtype=float)
    b = np.asarray(bounds_s, dtype=float).copy()
    b[-1] += framerate                                  # :70
    idx = np.full(lt.shape, -1, dtype=int)
    for c in range(len(b) - 1):                         # :81-83
        idx[(b[c] <= lt) & (lt < b[c + 1])] = c
    return idx


def cohort_lengths(bounds_s=DEFAULT_BOUNDS_S, framerate=2.0, buffer_frames=5):
    """getIntensityCohorts.m:64 -- pocet bodu spolecne osy kazde kohorty."""
    b = np.asarray(bounds_s, dtype=float)
    mids = (b[:-1] + b[1:]) / 2.0
    return (np.floor(mids / framerate) + 2 * buffer_frames).astype(int)


def cohort_time_axis(n_points, buffer_frames=5, framerate=2.0, align="left"):
    """plotIntensityCohorts.m:102-105. 'left': t=0 v prvnim snimku drahy;
    'right': t=0 v poslednim snimku drahy (buffer po zaniku v kladnem case)."""
    if align == "left":
        return np.arange(-buffer_frames, n_points - buffer_frames) * framerate
    return np.arange(-n_points + buffer_frames + 1, buffer_frames + 1) * framerate


def resample_track(values, n_out):
    """binterp(A, linspace(1, L, n_out)) -- kubicka interpolace na n_out bodu.

    binterp je kubicky B-spline se zrcadlovym okrajem; zde CubicSpline
    (not-a-knot). Rozdil je jen v chovani na krajnich ~2 bodech, uvnitr
    je interpolant tentyz. NaN v hodnotach -> NaN vysledek (radek se pak
    ve statistice preskoci), stejne jako by dopadl MATLAB.
    """
    v = np.asarray(values, dtype=float)
    L = v.size
    if L == 0 or not np.isfinite(v).all():
        return np.full(int(n_out), np.nan)
    if L == 1:
        return np.full(int(n_out), v[0])
    x = np.arange(1, L + 1, dtype=float)
    xi = np.linspace(1.0, float(L), int(n_out))
    if L < 4:                                           # kubika potrebuje 4 body
        return np.interp(xi, x, v)
    return CubicSpline(x, v, bc_type="not-a-knot")(xi)


def cohort_curves(tracks, bounds_s=DEFAULT_BOUNDS_S, framerate=2.0,
                  buffer_frames=5, align="left"):
    """Prumerne prubehy po kohortach.

    Parameters
    ----------
    tracks : iterable dictu s klici
        'movie'   -- identifikator filmu (SEM se pocita pres filmy)
        'A'       -- pole intenzit podel drahy (jen snimky drahy)
        'sbA'     -- start buffer (delka buffer_frames) nebo None
        'ebA'     -- end buffer nebo None
        'lifetime_s' -- zivotnost v sekundach (pocet snimku * framerate)
    Chybi-li buffery (None), pouzije se buffer_frames=0 pro danou drahu
    POUZE pokud je buffer_frames == 0; jinak je draha vyrazena (NaN),
    aby vsechny drahy kohorty mely shodne zarovnani.

    Returns
    -------
    list (jedna polozka na kohortu) dictu:
        t, mean, sem, median, q25, q75, n_tracks, n_movies, bounds,
        per_movie (matice filmy x body), tracks (matice drahy x body)
    """
    bounds_s = tuple(float(x) for x in bounds_s)
    lengths = cohort_lengths(bounds_s, framerate, buffer_frames)
    nc = len(bounds_s) - 1
    tracks = list(tracks)
    cidx = assign_cohorts([t["lifetime_s"] for t in tracks], bounds_s, framerate)

    out = []
    for c in range(nc):
        n = int(lengths[c])
        rows, movies = [], []
        for t, ci in zip(tracks, cidx):
            if ci != c:
                continue
            A = np.asarray(t["A"], dtype=float)
            if buffer_frames:
                sb, eb = t.get("sbA"), t.get("ebA")
                if sb is None or eb is None:
                    continue
                full = np.concatenate([np.asarray(sb, float)[-buffer_frames:],
                                       A, np.asarray(eb, float)[:buffer_frames]])
            else:
                full = A
            rows.append(resample_track(full, n))            # :99-100
            movies.append(t["movie"])

        M = np.vstack(rows) if rows else np.empty((0, n))
        movies = np.asarray(movies)
        valid = np.isfinite(M).all(axis=1) if M.size else np.zeros(0, bool)
        M, movies = M[valid], movies[valid]

        # plotIntensityCohorts.m:365-372 -- prumer per film, pak pres filmy
        uniq = list(dict.fromkeys(movies.tolist()))
        per_movie = np.vstack([M[movies == m].mean(axis=0) for m in uniq]) \
            if uniq else np.empty((0, n))
        nd = per_movie.shape[0]
        with np.errstate(invalid="ignore"):
            mean = per_movie.mean(axis=0) if nd else np.full(n, np.nan)
            sem = (per_movie.std(axis=0, ddof=1) / np.sqrt(nd)) if nd > 1 \
                else np.full(n, np.nan)
            # :375 -- percentily pres DRAHY
            q25, med, q75 = (np.percentile(M, [25, 50, 75], axis=0, method="hazen")
                             if M.shape[0] else (np.full(n, np.nan),) * 3)

        out.append(dict(
            cohort=c, bounds=(bounds_s[c], bounds_s[c + 1]),
            t=cohort_time_axis(n, buffer_frames, framerate, align),
            mean=mean, sem=sem, median=med, q25=q25, q75=q75,
            n_tracks=int(M.shape[0]), n_movies=int(nd),
            per_movie=per_movie, tracks=M,
        ))
    return out
