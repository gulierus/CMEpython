# -*- coding: utf-8 -*-
# CMEpython -- Python port mereni intenzity dynaminu z cmeAnalysis.
# Copyright (C) 2026 Ruslan Guliev
#
# Odvozeno z cmeAnalysis (DanuserLab, UT Southwestern), GPL-3.0.
# Tento program je svobodny software: muzete jej sirit a upravovat podle
# podminek GNU General Public License verze 3 nebo pozdejsi. Sirenu je
# BEZ JAKEKOLI ZARUKY. Viz soubor LICENSE.
"""Testy klasifikace dynamin-pozitivnich drah a jejich vstupu.

Overuji chovani odvozene z runSlaveChannelClassification.m,
filterGaussianFit2D.m a maskFromFirstMode.m. Numerickou shodu
filterGaussianFit2D s MATLABem overuje zvlast
scripts/validate_classification.py (vyzaduje MATLAB).
"""

import numpy as np
import pytest

from cmepython.background import (
    build_ccp_mask,
    filter_gaussian_fit_2d,
    mask_from_first_mode,
)
from cmepython.classification import classify_track

SIGMA = 2.0


def synth(amplitudes_positions, size=192, bg=100.0, noise=5.0, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.full((size, size), bg)
    for A, y0, x0 in amplitudes_positions:
        img += A * np.exp(-(((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * SIGMA ** 2)))
    return img + rng.normal(0, noise, img.shape)


# ---------------------------------------------------------------- fit

def test_filter_fit_obnovi_amplitudu_a_pozadi():
    img = synth([(400.0, 96.0, 96.0)])
    A, c, pv = filter_gaussian_fit_2d(img, SIGMA)
    assert A[96, 96] == pytest.approx(400.0, rel=0.05)
    assert c[96, 96] == pytest.approx(100.0, abs=3.0)
    assert pv[96, 96] < 1e-10


def test_filter_fit_na_prazdnu_je_konzervativni():
    """Test proti kLevel*sigma_res: na cistem sumu skoro nic nevyznamne."""
    rng = np.random.default_rng(3)
    img = 100 + rng.normal(0, 5, (256, 256))
    _, _, pv = filter_gaussian_fit_2d(img, SIGMA)
    assert (pv < 0.05).mean() < 0.02


def test_filter_fit_rozmery_a_konecnost():
    img = synth([(300.0, 50.0, 120.0)])
    A, c, pv = filter_gaussian_fit_2d(img, SIGMA)
    assert A.shape == c.shape == pv.shape == img.shape
    assert np.isfinite(A).all() and np.isfinite(c).all()


# ---------------------------------------------------------------- maska

def test_maska_najde_jasnou_oblast():
    """Bimodalni obraz: bunka (jasna) vs pozadi -- maska pokryje bunku."""
    rng = np.random.default_rng(1)
    img = 100 + rng.normal(0, 3, (256, 256))
    img[60:200, 40:220] += 150            # "bunka"
    mask, thr = mask_from_first_mode(img)
    assert thr is not None
    inside = mask[80:180, 60:200].mean()
    outside = mask[:40, :].mean()
    assert inside > 0.95 and outside < 0.05


def test_maska_bez_modu_vrati_cely_obraz():
    """maskFromFirstMode.m:70-73: kdyz neni druhy mod, maska = ones."""
    rng = np.random.default_rng(2)
    img = 100 + rng.normal(0, 3, (128, 128))   # unimodalni
    mask, thr = mask_from_first_mode(img)
    assert thr is None
    assert mask.all()


def test_ccp_maska_kryje_disky():
    m = build_ccp_mask((64, 64), [32], [32], radius=5)
    assert m[32, 32] and m[32, 36] and not m[32, 39]
    # pozice u okraje nesmi spadnout
    m2 = build_ccp_mask((64, 64), [1, 63], [1, 63], radius=5)
    assert m2[0, 0] and m2[63, 63]


# ---------------------------------------------------------------- testy drah

def _track(A_vals, bg95=50.0, p_det=0.01, sigma_r=30.0, npx=529, **kw):
    L = len(A_vals)
    A = np.asarray(A_vals, dtype=float)
    SE = sigma_r / np.sqrt(2 * (npx - 1))
    return classify_track(
        A=A,
        A_pstd=np.full(L, 8.0),
        sigma_r=np.full(L, sigma_r),
        SE_sigma_r=np.full(L, SE),
        hval_Ar=(A > 100).astype(float),
        bg95=bg95, p_detection=p_det, **kw)


def test_silna_draha_je_pozitivni():
    r = _track([500.0] * 20)
    assert r["significant_master"] and r["significant_slave"]
    assert r["n_detected"] == 20 and r["n_above_bg"] == 20


def test_prazdna_draha_je_negativni():
    r = _track([2.0] * 20)
    assert not r["significant_master"] and not r["significant_slave"]


def test_binomicka_korekce_na_delku():
    """Prah binoinv(0.95, L, 0.05) roste s delkou drahy
    (runSlaveChannelClassification.m:173): 2 vyznamne body z 6 staci
    (prah 1), ale 3 ze 100 ne (prah ~9). Porovnani je OSTRE vetsi --
    binoinv(0.95, 6, 0.05) = 1, takze jediny bod (1 > 1) nestaci."""
    short2 = _track([500.0] * 2 + [2.0] * 4)     # 2/6 nad pozadim
    short1 = _track([500.0] + [2.0] * 5)         # 1/6 -- presne na prahu
    long = _track([500.0] * 3 + [2.0] * 97)      # 3/100 nad pozadim
    assert short2["significant_slave"]
    assert not short1["significant_slave"]       # 1 > 1 je False
    assert not long["significant_slave"]


def test_amplitudovy_pomer_blokuje_zaporne_drahy():
    """:151-153 -- s amplitude_ratio=0 propadne draha se zapornym max A."""
    r = _track([-5.0] * 20)
    assert not r["significant_master"]


def test_nenulovy_pomer_vyzaduje_master():
    with pytest.raises(ValueError, match="master_max_A"):
        _track([500.0] * 10, amplitude_ratio=0.3)
    r = _track([500.0] * 10, amplitude_ratio=0.3, master_max_A=1000.0)
    assert r["significant_master"]               # 0.5 > 0.3
    r2 = _track([500.0] * 10, amplitude_ratio=0.6, master_max_A=1000.0)
    assert not r2["significant_master"]          # 0.5 < 0.6


def test_nan_body_se_ignoruji_ale_delka_zustava():
    """NaN (okraj obrazu) nesmi shodit klasifikaci; L pocita i NaN body,
    presne jako MATLAB (L = numel(t), nansum pres hval)."""
    A = [500.0] * 10 + [np.nan] * 3
    L = len(A)
    r = classify_track(
        A=np.array(A), A_pstd=np.full(L, 8.0),
        sigma_r=np.full(L, 30.0),
        SE_sigma_r=np.full(L, 30.0 / np.sqrt(2 * 528)),
        hval_Ar=np.array([1.0] * 10 + [np.nan] * 3),
        bg95=50.0, p_detection=0.01)
    assert r["track_len"] == 13
    assert r["n_detected"] == 10
    assert r["significant_master"] and r["significant_slave"]


def test_vysoka_mira_falesnych_detekci_zvedne_prah():
    """Kdyz je p_detection velke, tentyz pocet detekci uz nestaci."""
    lo = _track([500.0] * 3 + [2.0] * 27, p_det=0.001)
    hi = _track([500.0] * 3 + [2.0] * 27, p_det=0.5)
    assert lo["significant_master"]
    assert not hi["significant_master"]


# ------------------------------------------------- background_stats end-to-end

def _write_movie(tmp_path, frames_by_channel):
    """Zapise maly TCYX hyperstack; frames_by_channel je seznam (T,Y,X) poli."""
    import tifffile
    stack = np.stack(frames_by_channel, axis=1)          # (T, C, Y, X)
    p = tmp_path / "movie.tif"
    tifffile.imwrite(p, np.clip(stack, 0, 65535).astype(np.uint16),
                     imagej=True, metadata={"axes": "TCYX"})
    return p


def test_background_stats_mrdivide_semantika(tmp_path):
    """p_detection musi byt dot(a,b)/dot(b,b) se sloupcovymi soucty --
    presne to, co VYKONAVA MATLAB `sum(pval<a)/sum(cellmask)` na 2D
    maticich (mrdivide mezi radkovymi vektory), overeno primo v MATLABu.
    Prosty podil souctu se lisi ~6 % a posouva binomicke prahy."""
    from cmepython.classification import background_stats
    from cmepython.background import filter_gaussian_fit_2d

    rng = np.random.default_rng(0)
    T, Y, X = 3, 96, 96
    master = np.stack([np.full((Y, X), 300.0) + rng.normal(0, 3, (Y, X))
                       for _ in range(T)])
    master[:, 20:80, 20:80] += 300           # "bunka" pro masku
    slave = np.stack([synth([(600.0, 40.0, 40.0)], size=Y, seed=t)
                      for t in range(T)])
    p = _write_movie(tmp_path, [master, slave])

    cellmask = np.zeros((Y, X), bool); cellmask[20:80, 20:80] = True
    stats = background_stats(p, {}, sigma_slave=SIGMA,
                             slave_channel=1, master_channel=0,
                             cellmask=cellmask, n_frames=T)

    # rucni vypocet mrdivide pomeru pro jeden snimek a prumer
    import tifffile
    with tifffile.TiffFile(p) as tf:
        expected = []
        for t in range(T):
            fr = tf.pages[t * 2 + 1].asarray().astype(float)
            _, _, pv = filter_gaussian_fit_2d(fr, SIGMA)
            a = (pv < 0.05).sum(axis=0).astype(float)
            b = cellmask.sum(axis=0).astype(float)
            expected.append((a @ b) / (b @ b))
    assert stats["p_detection"] == pytest.approx(np.mean(expected), rel=1e-12)


def test_background_stats_kratky_film_bez_deduplikace(tmp_path):
    """Pro T < 16 MATLAB duplicitni snimky zapocitava vicenasovne --
    per_frame musi mit n_frames zaznamu, ne pocet unikatnich snimku."""
    from cmepython.classification import background_stats
    rng = np.random.default_rng(5)
    T, Y, X = 4, 96, 96
    ch = np.stack([100 + rng.normal(0, 4, (Y, X)) for _ in range(T)])
    p = _write_movie(tmp_path, [ch, ch])
    cellmask = np.ones((Y, X), bool)
    stats = background_stats(p, {}, sigma_slave=SIGMA,
                             slave_channel=1, master_channel=0,
                             cellmask=cellmask, n_frames=16)
    assert len(stats["per_frame"]) == 16
