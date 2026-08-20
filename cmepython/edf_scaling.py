# -*- coding: utf-8 -*-
# CMEpython -- Python port mereni intenzity dynaminu z cmeAnalysis.
# Copyright (C) 2026 Ruslan Guliev
#
# Odvozeno z cmeAnalysis (DanuserLab, UT Southwestern), GPL-3.0.
# Tento program je svobodny software: muzete jej sirit a upravovat podle
# podminek GNU General Public License verze 3 nebo pozdejsi. Sirenu je
# BEZ JAKEKOLI ZARUKY. Viz soubor LICENSE.
"""Skalovani amplitud napric filmy -- port scaleEDFs.m.

Medián amplitudy se mezi filmy lisi i nekolikanasobne (expozice, exprese,
bleaching). cmeAnalysis to pred poolovanim srovna tak, ze na kazdy film
najde multiplikativni faktor, ktery jeho empirickou distribucni funkci
(EDF) maximalnich amplitud nejlepe naskladane na referencni film.

Volane z getLifetimeData.m:223; vysledek se aplikuje na :234 jako
    A_skalovane = a[i] * A
tedy POUZE multiplikativne -- offset `c` se sice fituje, ale na amplitudy
se neaplikuje.

Pozor na dva detaily, ktere se snadno minou:

  * Reference 'med' se nevybira pres median EDF, ale pres median
    KVANTILOVE funkce: kazda EDF se interpoluje na spolecnou mrizku
    PRAVDEPODOBNOSTI (scaleEDFs.m:91, `interp1(F, x, fi)` -- vsimni si
    poradi argumentu), teprve pak se bere median a hleda se film s nejmensi
    sumou ctvercu odchylek (:95-96).
  * Rezidua se nuluji tam, kde kterakoli z EDF sedi na 0 nebo 1
    (scaleEDFs.m:234), jinak by ocasy distribuce prevalcovaly fit.

Zavislosti: numpy, scipy.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

__all__ = ["ecdf", "interp_edf", "scale_edfs", "apply_scaling"]


def ecdf(samples):
    """Empiricka distribucni funkce ve tvaru, jaky vraci MATLAB `ecdf`.

    Vraci (F, x), obe delky n+1, s vedoucim bodem F[0] = 0 a x[0] = x[1]
    -- proto se v `interp_edf` prvni prvek preskakuje.
    """
    s = np.sort(np.asarray(samples, dtype=float).ravel())
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("prazdny vzorek")
    F = np.arange(1, s.size + 1, dtype=float) / s.size
    return np.concatenate([[0.0], F]), np.concatenate([[s[0]], s])


def interp_edf(x_edf, f_edf, x):
    """scaleEDFs.m:237-241, vcetne okrajoveho chovani.

    Pod rozsahem dat vraci 0, nad rozsahem 1. Prvni bod je vzdy 0.
    """
    x = np.asarray(x, dtype=float)
    f = np.interp(x[1:], x_edf[1:], f_edf[1:], left=np.nan, right=np.nan)

    valid = np.nonzero(~np.isnan(f))[0]
    if valid.size == 0:
        return np.concatenate([[0.0], np.zeros_like(f)])
    f[:valid[0]] = 0.0                       # :239 -- pod rozsahem
    nan_after = np.nonzero(np.isnan(f))[0]
    if nan_after.size:
        f[nan_after[0]:] = 1.0               # :240 -- nad rozsahem
    return np.concatenate([[0.0], f])


def _select_reference(samples, F, x, mode):
    """scaleEDFs.m:84-99."""
    if mode == "max":
        return int(np.argmax([np.median(s) for s in samples]))
    if mode != "med":
        return int(mode)

    # :89-96 -- median KVANTILOVE funkce, ne EDF
    fi = np.arange(0.0, 1.0 + 1e-12, 0.001)
    M = np.vstack([np.interp(fi, F[i], x[i], left=np.nan, right=np.nan)
                   for i in range(len(samples))])
    median_edf = np.nanmedian(M, axis=0)
    J = np.nansum((M - median_edf[None, :]) ** 2, axis=1)
    return int(np.argmin(J))


def scale_edfs(samples, reference="med", ref_samples=None):
    """Najde multiplikativni faktor `a` pro kazdy vzorek.

    Parameters
    ----------
    samples : seznam 1D poli -- pro kazdy film jedna sada hodnot
        (v cmeAnalysis to jsou maximalni amplitudy na trajektorii,
         getLifetimeData.m:178)
    reference : 'med' (film nejblizsi medianove distribuci), 'max'
        (film s nejvyssim medianem) nebo index filmu
    ref_samples : volitelna externi referencni distribuce; pak se
        neodstranuje zadny film ze sady

    Returns
    -------
    a : pole faktoru, jeden na film (referencni film ma presne 1.0)
    c : pole offsetu z fitu -- cmeAnalysis je NEAPLIKUJE na amplitudy
    ref_idx : index referencniho filmu, nebo None pri `ref_samples`
    """
    samples = [np.asarray(s, dtype=float).ravel() for s in samples]
    samples = [s[np.isfinite(s)] for s in samples]
    nd = len(samples)

    if nd == 1 and ref_samples is None:       # :54-59
        return np.array([1.0]), np.array([0.0]), 0

    F, x = zip(*[ecdf(s) for s in samples])
    pooled = np.concatenate(samples)
    x0 = np.linspace(np.percentile(pooled, 1), np.percentile(pooled, 99), 1000)  # :79

    if ref_samples is None:
        ref_idx = _select_reference(samples, F, x, reference)
        ref_median = float(np.median(samples[ref_idx]))
        F_ref, x_ref = F[ref_idx], x[ref_idx]
        others = [i for i in range(nd) if i != ref_idx]
    else:
        ref = np.asarray(ref_samples, dtype=float).ravel()
        F_ref, x_ref = ecdf(ref)
        ref_median = float(np.median(ref))
        ref_idx = None
        others = list(range(nd))

    ref_edf = interp_edf(x_ref, F_ref, x0)

    a = np.ones(nd, dtype=float)
    c = np.zeros(nd, dtype=float)
    for i in others:
        med = float(np.median(samples[i]))
        a0 = ref_median / med if med != 0 else 1.0

        def cost(p, i=i):
            # :228-234
            if p[0] <= 0:
                return np.full(x0.size, 1e6)
            f_i = interp_edf(x[i], F[i], x0 / p[0])
            v = p[1] + (1.0 - p[1]) * f_i - ref_edf
            v[(f_i == 0) | (f_i == 1) | (ref_edf == 0) | (ref_edf == 1)] = 0.0
            return v

        # :120 -- meze [0 -1] az [Inf 1]
        try:
            sol = least_squares(cost, [max(a0, 1e-6), 0.0],
                                bounds=([1e-12, -1.0], [np.inf, 1.0]),
                                xtol=1e-6, ftol=1e-6, max_nfev=10000)
            a[i], c[i] = float(sol.x[0]), float(sol.x[1])
        except Exception:
            a[i], c[i] = a0, 0.0

    return a, c, ref_idx


def apply_scaling(values, a_i):
    """getLifetimeData.m:234 -- pouze multiplikativne, offset se neaplikuje."""
    return np.asarray(values, dtype=float) * float(a_i)
