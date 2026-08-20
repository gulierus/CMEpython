# CMEpython -- Python port mereni intenzity dynaminu z cmeAnalysis.
# Copyright (C) 2026 Ruslan Guliev
#
# Odvozeno z cmeAnalysis (DanuserLab, UT Southwestern), GPL-3.0.
# Tento program je svobodny software: muzete jej sirit a upravovat podle
# podminek GNU General Public License verze 3 nebo pozdejsi. Sirenu je
# BEZ JAKEKOLI ZARUKY. Viz soubor LICENSE.
"""Testy pro cmepython.slave_intensity.

Ověřují chování, které je odvozené ze zdrojáku cmeAnalysis. NEověřují
numerickou shodu s MATLABem — na to je potřeba referenční výstup
z běžícího cmeAnalysis (viz README).
"""

import numpy as np
import pytest

from cmepython import dynamin_intensity, fit_gaussian_2d_point

SIGMA_S = 1.5
SIGMA_M = 1.3
BG = 100.0


def synth(amplitude, y0=32.0, x0=32.0, size=64, noise=5.0, seed=0):
    """Jeden gaussovský zdroj na konstantním pozadí + bílý šum."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    img = BG + amplitude * np.exp(-(((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * SIGMA_S ** 2)))
    return (img + rng.normal(0, noise, img.shape))[None, ...]


def measure(video, y=32.0, x=32.0):
    return dynamin_intensity(video, 0, y, x, sigma_slave=SIGMA_S, sigma_master=SIGMA_M)


@pytest.mark.parametrize("amplitude", [20.0, 50.0, 200.0, 1000.0])
def test_amplitude_je_obnovena(amplitude):
    """A odpovídá vložené amplitudě nad pozadím (fitGaussians2D.m:202)."""
    r = measure(synth(amplitude))
    assert r["A"] == pytest.approx(amplitude, rel=0.10, abs=5.0)


def test_pozadi_je_obnoveno():
    """c je fitované lokální pozadí, ne odečtené (fitGaussians2D.m:204)."""
    r = measure(synth(200.0))
    assert r["c"] == pytest.approx(BG, abs=2.0)


def test_bez_signalu_vraci_cislo_ne_chybu():
    """Klíčové chování: prázdné místo dá amplitudu ~0, ne selhání.

    Měření a rozhodnutí jsou v cmeAnalysis oddělené kroky — nevýznamnost
    se pozná z pval_Ar, ne z toho, že by fit spadl.
    """
    r = measure(synth(0.0))
    assert np.isfinite(r["A"])
    assert abs(r["A"]) < 20.0
    assert r["hval_Ar"] is False


def test_silny_signal_je_vyznamny():
    r = measure(synth(200.0))
    assert r["hval_Ar"] is True
    assert r["pval_Ar"] < 0.05


def test_lokalizace_dohledá_posunuty_zdroj():
    """Fit 'xyAc' smí posunout pozici; přijme se, když je blíž než
    3*sigma_master a dá vyšší amplitudu (runDetection.m:184)."""
    r = measure(synth(200.0, y0=34.0), y=32.0)
    assert r["A"] == pytest.approx(200.0, rel=0.15)
    assert r["y"] == pytest.approx(34.0, abs=0.5)


def test_prilis_vzdaleny_zdroj_se_neprijme():
    """Zdroj mimo povolený posun se nesmí 'přitáhnout' — amplituda
    na zadané pozici musí zůstat nízká."""
    r = measure(synth(200.0, y0=45.0), y=32.0)
    assert abs(r["A"]) < 50.0


def test_okraj_obrazu_vraci_nan():
    """Okno se nevejde do snímku -> NaN, nikdy ořez (fitGaussians2D.m:162)."""
    r = measure(synth(200.0), y=2.0, x=2.0)
    assert np.isnan(r["A"])
    assert r["valid"] is False


def test_sigma_se_nefituje():
    """Módy 'Ac' a 'xyAc' neobsahují 's' — šířka zůstává vstupní hodnotou."""
    r = measure(synth(200.0))
    assert r["s"] == pytest.approx(SIGMA_S)


def test_amplituda_skaluje_linearne():
    """Dvojnásobný signál -> dvojnásobná amplituda (žádná normalizace)."""
    a1 = measure(synth(100.0, seed=3))["A"]
    a2 = measure(synth(200.0, seed=3))["A"]
    assert a2 / a1 == pytest.approx(2.0, rel=0.10)


def test_mode_Ac_nehybe_pozici():
    """'Ac' má zamčenou pozici — x,y vracejí zaokrouhlený vstup."""
    r = fit_gaussian_2d_point(synth(200.0, y0=34.0)[0], 32.0, 32.0,
                              sigma=SIGMA_S, mode="Ac")
    assert r["x"] == pytest.approx(32.0)
    assert r["y"] == pytest.approx(32.0)
