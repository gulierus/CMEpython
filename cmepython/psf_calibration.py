# -*- coding: utf-8 -*-
# CMEpython -- Python port mereni intenzity dynaminu z cmeAnalysis.
# Copyright (C) 2026 Ruslan Guliev
#
# Odvozeno z cmeAnalysis (DanuserLab, UT Southwestern), GPL-3.0.
# Tento program je svobodny software: muzete jej sirit a upravovat podle
# podminek GNU General Public License verze 3 nebo pozdejsi. Sirenu je
# BEZ JAKEKOLI ZARUKY. Viz soubor LICENSE.
"""Odhad sirky PSF z dat -- port getGaussianPSFsigmaFromData.m.

cmeAnalysis ve vychozim nastaveni ('GaussianPSF' == 'data', cmeAnalysis.m:85)
neodvozuje sigmu z optiky, ale meri ji primo z obrazu. Postup (radky odkazuji
do getGaussianPSFsigmaFromData.m):

  :66  prvni pruchod -- detekce s PEVNOU sigma = 1.5
  :67  druhy pruchod -- refit s VOLNOU sigmou (mod 'xyasc')
  :68  filtr        -- ~hval_AD & pval_Ar < 0.05
                       (rezidua musi projit testem normality A detekce
                        musi byt vyznamna)
  :71  sloucit sigmy ze VSECH obrazu dohromady
  :77  GMM s 1, 2 a 3 komponentami, vyber podle BIC
  :85  z vybraneho modelu vzit komponentu s NEJVYSSIM VRCHOLEM HUSTOTY
       (amp / (sqrt(2*pi) * svec)), NIKOLI s nejvetsi vahou

Pozn.: runDetection.m:90-92 vysledek jeste orizne zdola na 1.1 px. Clamp zde
NEni soucasti odhadu -- aplikuje se az v `apply_sigma_clamp`, aby bylo videt,
jestli vubec vystrelil.

Zavislosti: numpy, scipy, tifffile.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace, maximum_filter

from .slave_intensity import fit_gaussian_2d_point

__all__ = [
    "detect_candidates",
    "sigmas_from_frame",
    "estimate_psf_sigma",
    "apply_sigma_clamp",
    "fit_gmm_1d",
]

SIGMA_PROBE = 1.5          # getGaussianPSFsigmaFromData.m:66, pevna sigma 1. pruchodu
CLAMP_MIN = 1.1            # runDetection.m:92


# --------------------------------------------------------------------------
# 1D Gaussian mixture pres EM -- nahrada za gmdistribution.fit
# --------------------------------------------------------------------------

def fit_gmm_1d(x, n_components, n_iter=200, tol=1e-7, seed=0):
    """EM pro 1D smes gaussovek. Vraci (weights, mu, sigma, bic, loglik).

    Nahrazuje MATLAB gmdistribution.fit(svect', n) z
    getGaussianPSFsigmaFromData.m:79. Zamerne bez sklearn, aby balik zustal
    na numpy+scipy a byl spustitelny vsude.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 3 * n_components:
        return None

    rng = np.random.default_rng(seed)
    # inicializace kvantily -- deterministicke a stabilnejsi nez nahodne body
    q = np.linspace(0, 100, n_components + 2)[1:-1]
    mu = np.percentile(x, q).astype(float)
    mu = mu + rng.normal(0, 1e-6, n_components)     # rozbit pripadne shody
    sd = np.full(n_components, max(x.std(ddof=1), 1e-6))
    w = np.full(n_components, 1.0 / n_components)

    prev = -np.inf
    for _ in range(n_iter):
        # E-krok
        sd = np.maximum(sd, 1e-6)
        z = (x[:, None] - mu[None, :]) / sd[None, :]
        logp = -0.5 * z ** 2 - np.log(sd[None, :]) - 0.5 * np.log(2 * np.pi)
        logp = logp + np.log(np.maximum(w[None, :], 1e-300))
        m = logp.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(logp - m).sum(axis=1))
        resp = np.exp(logp - lse[:, None])

        ll = float(lse.sum())
        if np.abs(ll - prev) < tol * max(abs(ll), 1.0):
            prev = ll
            break
        prev = ll

        # M-krok
        nk = resp.sum(axis=0) + 1e-300
        w = nk / n
        mu = (resp * x[:, None]).sum(axis=0) / nk
        var = (resp * (x[:, None] - mu[None, :]) ** 2).sum(axis=0) / nk
        sd = np.sqrt(np.maximum(var, 1e-12))

    # BIC: k = (n_components-1) vah + n_components mu + n_components sigma
    k = 3 * n_components - 1
    bic = -2.0 * prev + k * np.log(n)
    return dict(weights=w, mu=mu, sigma=sd, bic=float(bic), loglik=float(prev))


# --------------------------------------------------------------------------
# Detekce kandidatu (zjednoduseny pointSourceDetection pro ucely kalibrace)
# --------------------------------------------------------------------------

def detect_candidates(img, sigma=SIGMA_PROBE, k=4.0, max_spots=600, border=None):
    """LoG + lokalni maxima + prah k nasobku sd sumu.

    Zjednodusena obdoba pointSourceDetection.m:133. Pro kalibraci staci --
    presny statisticky test se stejne aplikuje az na refitovane spoty.
    """
    a = np.asarray(img, dtype=float)
    log = -gaussian_laplace(a, sigma) * sigma ** 2
    mx = maximum_filter(log, size=2 * int(np.ceil(sigma)) + 1)
    resid = a - gaussian_filter(a, sigma)
    sd = np.median(np.abs(resid - np.median(resid))) * 1.4826
    peaks = (log == mx) & (log > k * max(sd, 1e-12))

    w = int(np.ceil(4 * sigma)) + 2 if border is None else int(border)
    peaks[:w, :] = False
    peaks[-w:, :] = False
    peaks[:, :w] = False
    peaks[:, -w:] = False

    ys, xs = np.nonzero(peaks)
    if ys.size > max_spots:
        order = np.argsort(log[ys, xs])[::-1][:max_spots]
        ys, xs = ys[order], xs[order]
    return ys, xs


def sigmas_from_frame(img, sigma_probe=SIGMA_PROBE, alpha=0.05, max_spots=600):
    """Sigmy z jednoho snimku, s filtrem `~hval_AD & pval_Ar < alpha`.

    Zrcadli getGaussianPSFsigmaFromData.m:66-69.
    """
    ys, xs = detect_candidates(img, sigma_probe, max_spots=max_spots)
    i_range = (float(np.min(img)), float(np.max(img)))
    out = []
    for y, x in zip(ys, xs):
        r = fit_gaussian_2d_point(img, float(x), float(y), sigma_probe,
                                  mode="xyasc", i_range=i_range, alpha=alpha)
        if not r["valid"]:
            continue
        s = r["s"]
        # :68 -- isPSF = ~hval_AD & pval_Ar < 0.05
        if np.isfinite(s) and (not r["hval_AD"]) and r["pval_Ar"] < alpha:
            out.append(float(s))
    return out


# --------------------------------------------------------------------------
# Hlavni vstup
# --------------------------------------------------------------------------

def estimate_psf_sigma(frames, sigma_probe=SIGMA_PROBE, alpha=0.05,
                       max_spots=600, return_detail=False):
    """Odhad sigma PSF ze seznamu snimku (2D numpy poli).

    `frames` musi pochazet ze VSECH filmu dane podminky dohromady --
    runDetection.m:66-80 vzorkuje ~40 snimku napric celym datasetem, ne
    z jednoho filmu. Sigma je jeden skalar na kanal pro cely dataset.
    """
    svect = []
    per_frame = []
    for img in frames:
        s = sigmas_from_frame(np.asarray(img, dtype=float),
                              sigma_probe=sigma_probe, alpha=alpha,
                              max_spots=max_spots)
        per_frame.append(len(s))
        svect.extend(s)

    svect = np.asarray(svect, dtype=float)
    if svect.size < 20:
        raise ValueError(f"prilis malo pouzitelnych spotu ({svect.size}); "
                         "getGaussianPSFsigmaFromData.m:118 v tomto pripade "
                         "spadne na prosty prumer")

    # :77-84 -- GMM s 1..3 komponentami, vyber podle BIC
    models = {}
    for n in (1, 2, 3):
        m = fit_gmm_1d(svect, n)
        if m is not None:
            models[n] = m
    best_n = min(models, key=lambda n: models[n]["bic"])
    m = models[best_n]

    # :85 -- komponenta s NEJVYSSIM VRCHOLEM HUSTOTY, ne s nejvetsi vahou.
    # MATLAB: [~,idx] = max(amp./(sqrt(2*pi)*svec))
    order = np.argsort(m["mu"])
    mu, sd, w = m["mu"][order], m["sigma"][order], m["weights"][order]
    peak = w / (np.sqrt(2 * np.pi) * np.maximum(sd, 1e-12))
    idx = int(np.argmax(peak))
    sigma = float(mu[idx])

    if not return_detail:
        return sigma
    return sigma, dict(
        sigma=sigma, n_spots=int(svect.size), n_components=best_n,
        bic={n: models[n]["bic"] for n in models},
        components=[dict(mu=float(mu[i]), sigma=float(sd[i]),
                         weight=float(w[i]), peak=float(peak[i]),
                         selected=(i == idx)) for i in range(len(mu))],
        median=float(np.median(svect)),
        iqr=(float(np.percentile(svect, 25)), float(np.percentile(svect, 75))),
        spots_per_frame=per_frame,
    )


def apply_sigma_clamp(sigma, minimum=CLAMP_MIN):
    """runDetection.m:90-92. Vraci (sigma_po_clampu, vystrelil_clamp).

    Originál na tomto miste jen vypise varovani do stderru a pokracuje --
    zde je to navratova hodnota, aby se to nedalo prehlednout.
    """
    fired = float(sigma) < minimum
    return (float(minimum) if fired else float(sigma)), fired
