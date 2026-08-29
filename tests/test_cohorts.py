# -*- coding: utf-8 -*-
# CMEpython -- Python port mereni intenzity dynaminu z cmeAnalysis.
# Copyright (C) 2026 Ruslan Guliev
#
# Odvozeno z cmeAnalysis (DanuserLab, UT Southwestern), GPL-3.0.
# Tento program je svobodny software: muzete jej sirit a upravovat podle
# podminek GNU General Public License verze 3 nebo pozdejsi. Sirenu je
# BEZ JAKEKOLI ZARUKY. Viz soubor LICENSE.
"""Testy kohortni analyzy -- chovani odvozene z getIntensityCohorts.m
a plotIntensityCohorts.m."""

import numpy as np
import pytest

from cmepython.cohorts import (
    DEFAULT_BOUNDS_S,
    assign_cohorts,
    cohort_curves,
    cohort_lengths,
    cohort_time_axis,
    resample_track,
)


def test_zarazeni_do_kohort_hranice():
    """:81-84 -- dolni mez vcetne, horni bez; posledni mez + framerate."""
    lt = [8, 10, 19.9, 20, 40, 119.9, 120, 121.9, 122, 300]
    c = assign_cohorts(lt, DEFAULT_BOUNDS_S, framerate=2.0)
    assert list(c) == [-1, 0, 0, 1, 2, 5, 5, 5, -1, -1]


def test_delky_kohort_odpovidaji_matlabu():
    """:64 -- floor(mean(bounds)/framerate) + 2b pro b=5, 2 s/snimek."""
    L = cohort_lengths(DEFAULT_BOUNDS_S, framerate=2.0, buffer_frames=5)
    # stredy 15,30,50,70,90,110 s -> 7,15,25,35,45,55 snimku + 10
    assert list(L) == [17, 25, 35, 45, 55, 65]


def test_osa_casu_left_a_right():
    t = cohort_time_axis(17, buffer_frames=5, framerate=2.0, align="left")
    assert t[0] == -10.0 and t[5] == 0.0 and t[-1] == 22.0
    r = cohort_time_axis(17, buffer_frames=5, framerate=2.0, align="right")
    assert r[-6] == 0.0 and r[-1] == 10.0


def test_prevzorkovani_zachova_linearni_prubeh():
    v = np.linspace(3.0, 9.0, 12)
    out = resample_track(v, 30)
    assert out.shape == (30,)
    assert np.allclose(out, np.linspace(3.0, 9.0, 30), atol=1e-9)
    assert out[0] == pytest.approx(3.0) and out[-1] == pytest.approx(9.0)


def test_prevzorkovani_nan_vrati_nan():
    assert np.isnan(resample_track([1.0, np.nan, 3.0, 4.0, 5.0], 8)).all()


def _tracks(rng, movie, n, length, amp, buffer_frames=5):
    out = []
    for _ in range(n):
        A = amp * np.sin(np.linspace(0, np.pi, length)) + rng.normal(0, 1, length)
        out.append(dict(movie=movie, A=A,
                        sbA=rng.normal(0, 1, buffer_frames),
                        ebA=rng.normal(0, 1, buffer_frames),
                        lifetime_s=length * 2.0))
    return out


def test_krivky_kohort_tvar_a_pocty():
    rng = np.random.default_rng(0)
    tr = _tracks(rng, "m1", 20, 12, 100.0) + _tracks(rng, "m2", 20, 12, 100.0) \
        + _tracks(rng, "m1", 10, 25, 50.0)
    res = cohort_curves(tr, DEFAULT_BOUNDS_S, framerate=2.0, buffer_frames=5)
    assert len(res) == 6
    c1 = res[1]                       # 20-40 s: delka 12 snimku = 24 s
    assert c1["n_tracks"] == 40 and c1["n_movies"] == 2
    assert c1["t"].shape == c1["mean"].shape == (25,)
    assert c1["mean"][5:-5].max() > 80                # vrchol sinusovky
    assert abs(c1["mean"][0]) < 5                     # buffer ~ 0
    c2 = res[2]                       # 40-60 s: 25 snimku = 50 s
    assert c2["n_tracks"] == 10 and c2["n_movies"] == 1
    assert np.isnan(c2["sem"]).all()                  # SEM pres 1 film = NaN
    assert res[0]["n_tracks"] == 0


def test_sem_je_pres_filmy_ne_drahy():
    """plotIntensityCohorts.m:370 -- SEM = std(per-film prumery)/sqrt(nd)."""
    rng = np.random.default_rng(1)
    tr = _tracks(rng, "a", 30, 12, 100.0) + _tracks(rng, "b", 30, 12, 100.0) \
        + _tracks(rng, "c", 30, 12, 100.0)
    res = cohort_curves(tr, DEFAULT_BOUNDS_S, 2.0, 5)[1]
    pm = res["per_movie"]
    expected = pm.std(axis=0, ddof=1) / np.sqrt(3)
    assert np.allclose(res["sem"], expected)
    assert res["per_movie"].shape == (3, 25)


def test_bez_bufferu_kdyz_buffer_frames_nula():
    rng = np.random.default_rng(2)
    tr = [dict(movie="m", A=np.ones(12) * 7.0, sbA=None, ebA=None, lifetime_s=24.0)]
    res = cohort_curves(tr, DEFAULT_BOUNDS_S, 2.0, buffer_frames=0)[1]
    assert res["n_tracks"] == 1 and res["t"][0] == 0.0
    assert np.allclose(res["mean"], 7.0)


def test_draha_bez_bufferu_je_vyrazena_kdyz_jsou_buffery_vyzadovany():
    tr = [dict(movie="m", A=np.ones(12), sbA=None, ebA=None, lifetime_s=24.0)]
    res = cohort_curves(tr, DEFAULT_BOUNDS_S, 2.0, buffer_frames=5)[1]
    assert res["n_tracks"] == 0
