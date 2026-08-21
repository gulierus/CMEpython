# -*- coding: utf-8 -*-
# CMEpython -- Python port mereni intenzity dynaminu z cmeAnalysis.
# Copyright (C) 2026 Ruslan Guliev
#
# Odvozeno z cmeAnalysis (DanuserLab, UT Southwestern), GPL-3.0.
# Tento program je svobodny software: muzete jej sirit a upravovat podle
# podminek GNU General Public License verze 3 nebo pozdejsi. Sirenu je
# BEZ JAKEKOLI ZARUKY. Viz soubor LICENSE.
"""Celoplosny fit a maska bunky -- porty filterGaussianFit2D.m a
maskFromFirstMode.m / getCellMask.m.

Oboji jsou vstupy klasifikace dynamin-pozitivnich drah
(runSlaveChannelClassification.m): z masky bunky a celoplosneho fitu se
stavi rozdeleni pozadi (bg95) a mira falesnych detekci (pDetection).

Na rozdil od fitu v `slave_intensity` tohle NENI iterativni optimalizace:
filterGaussianFit2D resi na kazdem pixelu 2-parametrovou LINEARNI ulohu
(amplituda + pozadi pri pevne pozici a sigme) v uzavrenem tvaru pres
konvoluce -- zadny MEX, zadny solver.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter, label
from scipy.signal import fftconvolve
from scipy.stats import norm as _norm
from scipy.stats import t as _student_t

__all__ = [
    "filter_gaussian_fit_2d",
    "mask_from_first_mode",
    "cell_mask_from_movie",
    "build_ccp_mask",
]


def filter_gaussian_fit_2d(img, sigma, alpha=0.05):
    """Port filterGaussianFit2D.m -- fit A a c v KAZDEM pixelu obrazu.

    V okne (2*ceil(4*sigma)+1)^2 kolem kazdeho pixelu resi linearni LSQ
    model A*g + c (gaussovka s pevnou sigmou a pevnym stredem v pixelu).
    Vse pres konvoluce, O(1) na pixel (filterGaussianFit2D.m:60-67).

    Returns
    -------
    (A_est, c_est, pval_Ar) -- tri pole tvaru img. pval_Ar je p-hodnota
    tehoz jednostranneho t-testu proti kLevel*sigma_res jako v detekci
    (:69-89); pval < alpha ~ "v tomto pixelu je vyznamny bodovy zdroj".
    """
    img = np.asarray(img, dtype=np.float64)
    w = int(np.ceil(4.0 * float(sigma)))

    # :48-55 -- kernel a jeho momenty
    x = np.arange(-w, w + 1, dtype=float)
    g = np.exp(-x ** 2 / (2.0 * float(sigma) ** 2))
    g2 = np.outer(g, g)
    n = g2.size
    gsum = g2.sum()
    g2sum = (g2 ** 2).sum()

    # C = inv(J'J) pro J = [g2(:), ones]; C[0,0] analyticky (:57-58)
    det = n * g2sum - gsum ** 2
    C00 = n / det

    # :60-63 -- zrcadlovy pad + 'valid' konvoluce.
    # POZOR NA NAZVOSLOVI: padarrayXT('symmetric') zrcadli BEZ duplikace
    # okrajoveho pixelu (padarrayXT.m:3 to rika vyslovne), coz je numpy
    # mode='reflect' -- NIKOLI numpy 'symmetric', ktery okraj duplikuje.
    # Zamena zpusobi rozdily az ~1e3 v A_est v pasu w pixelu u okraje
    # (zmereno proti MATLABu; uvnitr obrazu je vliv nulovy).
    # Kernely jsou symetricke, takze konvoluce == korelace.
    padded = np.pad(img, w, mode="reflect")
    box = np.ones((2 * w + 1, 2 * w + 1))
    fg = fftconvolve(padded, g2, mode="valid")
    fu = fftconvolve(padded, box, mode="valid")
    fu2 = fftconvolve(padded ** 2, box, mode="valid")

    # :65-66
    A_est = (fg - gsum * fu / n) / (g2sum - gsum ** 2 / n)
    c_est = (fu - A_est * gsum) / n

    # :71-75 -- RSS z konvoluci; zaporne hodnoty jsou roundoff
    f_c = fu2 - 2.0 * c_est * fu + n * c_est ** 2
    RSS = A_est ** 2 * g2sum - 2.0 * A_est * (fg - c_est * gsum) + f_c
    RSS[RSS < 0] = 0.0

    sigma_e2 = RSS / (n - 3)                    # :76
    sigma_A = np.sqrt(sigma_e2 * C00)           # :78
    sigma_res = np.sqrt(RSS / (n - 1))          # :81

    kLevel = _norm.ppf(1.0 - alpha / 2.0)       # :83

    # :85-89 -- Welchuv t-test proti kLevel*sigma_res, presne jako MATLAB
    SE_sigma_c = sigma_res / np.sqrt(2.0 * (n - 1)) * kLevel
    with np.errstate(divide="ignore", invalid="ignore"):
        df2 = (n - 1) * (sigma_A ** 2 + SE_sigma_c ** 2) ** 2 \
            / (sigma_A ** 4 + SE_sigma_c ** 4)
        scomb = np.sqrt((sigma_A ** 2 + SE_sigma_c ** 2) / n)
        T = (A_est - sigma_res * kLevel) / scomb
        pval = _student_t.cdf(-T, df2)

    return A_est, c_est, pval


def _ksdensity_100(v):
    """Aproximace MATLAB ksdensity(v, 'npoints', 100).

    MATLAB pouziva normalni kernel s robustni Silvermanovou sirkou
    (sigma z MAD) a mrizku 100 bodu pres rozsah dat rozsireny o ~3 sirky.
    Presna shoda neni garantovana -- vysledna maska se validuje proti
    MATLABu jako celek (Jaccard), viz matlab/export_mask_ref.m.
    """
    v = np.asarray(v, dtype=float)
    med = np.median(v)
    sig = np.median(np.abs(v - med)) / 0.6745
    if sig <= 0:
        sig = v.std(ddof=1)
    if sig <= 0:
        sig = 1.0
    bw = sig * (4.0 / (3.0 * v.size)) ** 0.2
    lo, hi = v.min() - 3 * bw, v.max() + 3 * bw
    xi = np.linspace(lo, hi, 100)
    # KDE po blocich, at se nealokuje v.size x 100 matice najednou
    f = np.zeros(100)
    for start in range(0, v.size, 200000):
        chunk = v[start:start + 200000]
        z = (xi[None, :] - chunk[:, None]) / bw
        f += np.exp(-0.5 * z ** 2).sum(axis=0)
    f /= (v.size * bw * np.sqrt(2 * np.pi))
    return f, xi


def _local_extrema(f, mode):
    """locmax1d/locmin1d(f, 3) -- MATLAB semantika vcetne plosin.

    MATLAB paduje (+inf pro max, 0 pro min), vezme klouzave okno 3 a vrati
    KAZDY index, kde se hodnota rovna okennimu extremu -- tedy NEostre
    porovnani, ktere vraci vsechny body plosiny. Na KDE, ktera mezi dvema
    vzdalenymi mody podtece presne na 0, by ostra verze zadne minimum
    nenasla a maska by tise degradovala na cely obraz.
    """
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    f = np.asarray(f, dtype=float)
    if mode == "max":
        w = maximum_filter1d(f, size=3, mode="constant", cval=np.inf)
        hit = f == w
        hit[0] = hit[-1] = False        # +inf pad: okraje nikdy nejsou max
    else:
        w = minimum_filter1d(f, size=3, mode="constant", cval=0.0)
        hit = f == w
    return np.nonzero(hit)[0].astype(int)


def mask_from_first_mode(img, mode_ratio=0.8, connect=True):
    """Port maskFromFirstMode.m -- prah z prvniho modu histogramu.

    Vyhlazeny obraz (sigma=5, :35), histogram orezany na 1.-99. percentil
    (:40-41), KDE (:43), prah = prvni lokalni minimum hustoty za prvnim
    lokalnim maximem (:53-58) -- prijme se jen kdyz hmota pod prahem
    nepresahne mode_ratio (:56). Pak nejvetsi 8-souvisla komponenta (:62-68).
    Pri selhani vraci masku pres cely obraz (:70-73), presne jako MATLAB.

    Pozn.: getCellMask.m:37 preda ModeRatio=0.8 (vlastni default
    maskFromFirstMode je 0.6) -- zde je default uz 0.8, protoze voláme
    vyhradne v roli getCellMask.
    """
    img = np.asarray(img, dtype=float)
    g = gaussian_filter(img, 5.0)               # filterGauss2D(img, 5)

    v = g.ravel()
    v = v[np.isfinite(v)]
    p1, p99 = np.percentile(v, [1, 99], method="hazen")   # MATLAB prctile
    v = v[(v >= p1) & (v <= p99)]

    f, xi = _ksdensity_100(v)
    lmax = _local_extrema(f, "max")
    lmin = _local_extrema(f, "min")
    dxi = xi[1] - xi[0]

    threshold = None
    if lmin.size and lmax.size:
        after = lmin[lmin > lmax[0]]
        if after.size and f[:after[0] + 1].sum() * dxi < mode_ratio:
            threshold = float(xi[after[0]])

    if threshold is None:
        return np.ones(img.shape, dtype=bool), None

    mask = g > threshold
    if connect:
        lab, ncomp = label(mask, structure=np.ones((3, 3)))   # 8-souvislost
        if ncomp:
            sizes = np.bincount(lab.ravel())[1:]
            # :67 -- ponechat vsechny komponenty o velikosti >= nejvetsi
            # (tj. nejvetsi vcetne pripadnych shodne velkych)
            keep = np.nonzero(sizes >= sizes.max())[0] + 1
            mask = np.isin(lab, keep)
    return mask, threshold


def cell_mask_from_movie(path, channel, movie_layout_fn=None):
    """Port getCellMask (Mode='maxproj', vychozi cesta cmeAnalysis).

    Max-projekce celeho stacku daneho kanalu (getCellMask.m:70), prah
    z prvniho modu, imfill (:116). Projekce se pocita inkrementalne --
    film se nedrzi v pameti.
    """
    import tifffile
    from .measure import movie_layout

    with tifffile.TiffFile(str(path)) as tf:
        T, C, Y, X, idx = movie_layout(tf)
        proj = None
        for t in range(T):
            fr = tf.pages[idx(t, channel)].asarray()
            proj = fr.copy() if proj is None else np.maximum(proj, fr)

    mask, threshold = mask_from_first_mode(proj.astype(float))
    mask = binary_fill_holes(mask)              # getCellMask.m:116
    return mask, proj, threshold


def build_ccp_mask(shape, ys, xs, radius):
    """Maska CCS oblasti: disky polomeru `radius` kolem zadanych pozic.

    VEDOMA ODCHYLKA od cmeAnalysis: original dilatuje detekcni masky
    z pointSourceDetection (dmasks.tif, runSlaveChannelClassification.m:75-80)
    diskem strel('disk', w). Detekcni masky nemame -- pozice z trajektorii
    ale pokryvaji tytez struktury, takze disk polomeru w kolem kazde pozice
    je funkcne ekvivalentni: ucel je vyloucit okoli CCS z rozdeleni pozadi.
    """
    mask = np.zeros(shape, dtype=bool)
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (yy ** 2 + xx ** 2) <= radius ** 2
    ny, nx = shape
    ys_i = np.rint(np.asarray(ys, dtype=float)).astype(int)
    xs_i = np.rint(np.asarray(xs, dtype=float)).astype(int)
    for y, x in zip(ys_i, xs_i):
        y0, y1 = max(y - r, 0), min(y + r + 1, ny)
        x0, x1 = max(x - r, 0), min(x + r + 1, nx)
        if y0 >= y1 or x0 >= x1:
            continue
        mask[y0:y1, x0:x1] |= disk[(y0 - y + r):(y1 - y + r),
                                   (x0 - x + r):(x1 - x + r)]
    return mask
