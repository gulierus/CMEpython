# CMEpython -- Python port mereni intenzity dynaminu z cmeAnalysis.
# Copyright (C) 2026 Ruslan Guliev
#
# Odvozeno z cmeAnalysis (DanuserLab, UT Southwestern), GPL-3.0.
# Tento program je svobodny software: muzete jej sirit a upravovat podle
# podminek GNU General Public License verze 3 nebo pozdejsi. Sirenu je
# BEZ JAKEKOLI ZARUKY. Viz soubor LICENSE.
"""Testy davkove vrstvy a kalibrace sirky PSF.

Overuji chovani odvozene ze zdrojaku cmeAnalysis a vnitrni konzistenci
(davka == bod po bodu). NEoveruji numerickou shodu s MATLABem.
"""

import numpy as np
import pytest
import tifffile

from cmepython import (
    dynamin_intensity,
    measure_frame,
    measure_coords,
    measure_movie,
    estimate_psf_sigma,
    apply_sigma_clamp,
)
from cmepython.psf_calibration import detect_candidates
from cmepython.psf_calibration import fit_gmm_1d

SIGMA_S, SIGMA_M, BG = 2.0, 1.4, 100.0


def synth_frame(positions, amplitude=500.0, size=128, noise=8.0, seed=0, sigma=SIGMA_S):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.full((size, size), BG, dtype=float)
    for y0, x0 in positions:
        img += amplitude * np.exp(-(((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2)))
    return img + rng.normal(0, noise, img.shape)


def synth_movie(n_frames=4, size=128, seed=0, sigma=SIGMA_S):
    pos = [(32.0, 32.0), (64.0, 70.0), (96.0, 40.0)]
    return np.stack([synth_frame(pos, size=size, seed=seed + t, sigma=sigma)
                     for t in range(n_frames)]), pos


# ---------------------------------------------------------------- davka

def test_measure_frame_odpovida_bodovemu_volani():
    """Davkova cesta musi dat presne totez co volani bod po bodu."""
    img = synth_frame([(32.0, 32.0), (64.0, 70.0)])
    ys, xs = [32.0, 70.0], [32.0, 64.0]
    batch = measure_frame(img, ys, xs, SIGMA_S, SIGMA_M)
    for k, (y, x) in enumerate(zip(ys, xs)):
        ref = dynamin_intensity(img[None, ...], 0, y, x,
                                sigma_slave=SIGMA_S, sigma_master=SIGMA_M)
        assert batch["A"][k] == pytest.approx(ref["A"], rel=1e-12)
        assert batch["c"][k] == pytest.approx(ref["c"], rel=1e-12)
        assert bool(batch["hval_Ar"][k]) is bool(ref["hval_Ar"])


def test_measure_coords_seskupuje_po_snimcich():
    video, pos = synth_movie()
    coords = np.array([[t, y, x] for t in range(len(video)) for y, x in pos], float)
    out = measure_coords(video, coords, SIGMA_S, SIGMA_M)
    assert out["A"].shape == (len(coords),)
    assert np.isfinite(out["A"]).all()
    assert out["A"].min() > 100.0          # vsechny na realnych zdrojich


def test_measure_frame_vraci_pole_spravnych_typu():
    img = synth_frame([(32.0, 32.0)])
    out = measure_frame(img, [32.0], [32.0], SIGMA_S, SIGMA_M)
    assert out["hval_Ar"].dtype == bool
    assert out["npx"].dtype == np.int64
    assert out["A"].dtype == float


def test_measure_coords_odmitne_spatny_tvar():
    video, _ = synth_movie()
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        measure_coords(video, np.zeros((5, 2)), SIGMA_S, SIGMA_M)


def test_prazdny_seznam_souradnic():
    img = synth_frame([(32.0, 32.0)])
    out = measure_frame(img, [], [], SIGMA_S, SIGMA_M)
    assert out["A"].shape == (0,)


# ---------------------------------------------------------------- IO

def test_measure_movie_cte_spravny_kanal(tmp_path):
    """Trikanalovy hyperstack: mereni musi cist prave `slave_channel`."""
    video, pos = synth_movie(n_frames=3)
    T, Y, X = video.shape
    stack = np.zeros((T, 3, Y, X), dtype=np.uint16)
    rng = np.random.default_rng(11)
    # prazdne kanaly MUSI mit sum: pri konstantnim obrazu je iRange nulovy
    # a fitGaussians2D.m:198 (prm(3) < 2*diff(iRange)) vsechno zamitne -> NaN.
    stack[:, 0] = np.clip(BG + rng.normal(0, 8, (T, Y, X)), 0, 65535)
    stack[:, 1] = np.clip(BG + rng.normal(0, 8, (T, Y, X)), 0, 65535)
    stack[:, 2] = np.clip(video, 0, 65535)            # signal jen zde
    p = tmp_path / "movie.tif"
    tifffile.imwrite(p, stack, imagej=True, metadata={'axes': 'TCYX'})

    coords = np.array([[t, y, x] for t in range(T) for y, x in pos], float)
    on = measure_movie(p, coords, SIGMA_S, SIGMA_M, slave_channel=2)
    off = measure_movie(p, coords, SIGMA_S, SIGMA_M, slave_channel=0)
    assert np.nanmedian(on["A"]) > 100.0
    assert abs(np.nanmedian(off["A"])) < 20.0


def test_measure_movie_odmitne_kanal_mimo_rozsah(tmp_path):
    video, pos = synth_movie(n_frames=2)
    stack = np.clip(video, 0, 65535).astype(np.uint16)[:, None, :, :]
    p = tmp_path / "m.tif"
    tifffile.imwrite(p, stack, imagej=True, metadata={'axes': 'TCYX'})
    with pytest.raises(ValueError, match="mimo rozsah"):
        measure_movie(p, np.array([[0, 32.0, 32.0]]), SIGMA_S, SIGMA_M, slave_channel=5)


def test_measure_movie_odmitne_snimek_mimo_rozsah(tmp_path):
    video, _ = synth_movie(n_frames=2)
    stack = np.clip(video, 0, 65535).astype(np.uint16)[:, None, :, :]
    p = tmp_path / "m.tif"
    tifffile.imwrite(p, stack, imagej=True, metadata={'axes': 'TCYX'})
    with pytest.raises(ValueError, match="snimky mimo rozsah"):
        measure_movie(p, np.array([[99, 32.0, 32.0]]), SIGMA_S, SIGMA_M, slave_channel=0)


# ---------------------------------------------------------------- kalibrace

def test_estimate_psf_sigma_najde_vlozenou_sirku():
    """Kalibrace musi obnovit sirku, se kterou byly zdroje vygenerovany."""
    rng = np.random.default_rng(3)
    frames = []
    for t in range(6):
        pos = rng.uniform(20, 108, size=(40, 2))
        frames.append(synth_frame(pos, amplitude=900.0, noise=6.0,
                                  seed=100 + t, sigma=1.8))
    sigma = estimate_psf_sigma(frames)
    assert sigma == pytest.approx(1.8, abs=0.25)


def test_estimate_psf_sigma_odmitne_malo_spotu():
    frames = [np.full((64, 64), BG) for _ in range(2)]
    with pytest.raises(ValueError, match="prilis malo"):
        estimate_psf_sigma(frames)


def test_clamp_vraci_priznak():
    """runDetection.m:90-92 jen varuje do stderru; zde je to navratova hodnota."""
    assert apply_sigma_clamp(0.8) == (1.1, True)
    assert apply_sigma_clamp(2.5) == (2.5, False)
    assert apply_sigma_clamp(1.1) == (1.1, False)     # presne na prahu neclampuje


def test_gmm_rozdeli_dve_komponenty():
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.normal(1.2, 0.1, 600), rng.normal(2.6, 0.2, 400)])
    m1, m2 = fit_gmm_1d(x, 1), fit_gmm_1d(x, 2)
    assert m2["bic"] < m1["bic"]                       # dve komponenty vyhraji
    assert sorted(m2["mu"])[0] == pytest.approx(1.2, abs=0.15)
    assert sorted(m2["mu"])[1] == pytest.approx(2.6, abs=0.2)


def test_gmm_vraci_none_pri_malo_datech():
    assert fit_gmm_1d(np.array([1.0, 2.0]), 3) is None


def test_detect_candidates_najde_vlozene_zdroje():
    pos = [(32.0, 32.0), (64.0, 70.0), (96.0, 40.0)]
    img = synth_frame(pos, amplitude=900.0, noise=5.0)
    ys, xs = detect_candidates(img, SIGMA_S, k=5.0)
    for y0, x0 in pos:
        d = np.hypot(ys - y0, xs - x0).min()
        assert d < 2.0, f"zdroj ({y0}, {x0}) nenalezen"


# ---------------------------------------------------------------- EDF skalovani

def test_scale_edfs_obnovi_zname_faktory():
    """Tataz distribuce nasobena znamym faktorem -> skalovani ji srovna."""
    from cmepython import scale_edfs
    rng = np.random.default_rng(0)
    base = rng.lognormal(7.0, 0.6, 4000)
    samples = [base * f for f in (1.0, 2.0, 0.5, 3.0)]
    a, c, ref = scale_edfs(samples)
    meds = [np.median(s * ai) for s, ai in zip(samples, a)]
    assert np.std(meds) / np.mean(meds) < 0.05


def test_scale_edfs_referencni_film_ma_faktor_jedna():
    from cmepython import scale_edfs
    rng = np.random.default_rng(1)
    samples = [rng.lognormal(7.0, 0.5, 2000) * f for f in (1.0, 1.8, 0.6)]
    a, c, ref = scale_edfs(samples)
    assert a[ref] == 1.0


def test_scale_edfs_jediny_film():
    """scaleEDFs.m:54-59 -- jediny vzorek bez reference vraci a=1."""
    from cmepython import scale_edfs
    a, c, ref = scale_edfs([np.array([1.0, 2.0, 3.0])])
    assert a[0] == 1.0 and c[0] == 0.0 and ref == 0


def test_scale_edfs_reference_max():
    """'max' vybere film s nejvyssim medianem."""
    from cmepython import scale_edfs
    rng = np.random.default_rng(2)
    samples = [rng.lognormal(7.0, 0.4, 1500) * f for f in (1.0, 3.0, 0.5)]
    a, c, ref = scale_edfs(samples, reference="max")
    assert ref == 1


def test_interp_edf_okrajove_chovani():
    """scaleEDFs.m:237-241 -- pod rozsahem 0, nad rozsahem 1."""
    from cmepython.edf_scaling import ecdf, interp_edf
    F, x = ecdf(np.array([10.0, 20.0, 30.0]))
    f = interp_edf(x, F, np.array([0.0, 5.0, 15.0, 25.0, 100.0]))
    assert f[0] == 0.0
    assert f[1] == 0.0
    assert 0.0 < f[2] < 1.0
    assert f[-1] == 1.0


def test_apply_scaling_je_ciste_multiplikativni():
    """getLifetimeData.m:234 -- offset se na amplitudy neaplikuje."""
    from cmepython import apply_scaling
    v = np.array([100.0, 200.0])
    assert np.allclose(apply_scaling(v, 2.5), [250.0, 500.0])


# ------------------------------------------------- opravy z auditu portu

def test_ecdf_sluci_shodne_hodnoty():
    """MATLAB ecdf je Kaplan-Meier na RUZNYCH hodnotach -- duplicity slucuje.

    Bez toho ma interpolant v miste duplicit svisly usek navic a kvantilova
    funkce pouzita pri vyberu reference vraci jinou hodnotu.
    """
    from cmepython.edf_scaling import ecdf
    F, x = ecdf(np.array([1.0, 2.0, 2.0, 3.0]))
    assert len(x) == 4                      # 0 + tri RUZNE hodnoty
    assert np.allclose(x, [1.0, 1.0, 2.0, 3.0])
    assert np.allclose(F, [0.0, 0.25, 0.75, 1.0])


def test_interp_edf_same_nan_vraci_jednicky():
    """scaleEDFs.m:239-240 -- kdyz je vse NaN, MATLAB nastavi cele pole na 1."""
    from cmepython.edf_scaling import ecdf, interp_edf
    F, x = ecdf(np.array([10.0, 20.0]))
    f = interp_edf(x, F, np.array([1e6, 2e6, 3e6]))   # cely mimo rozsah nahoru
    assert f[0] == 0.0
    assert np.all(f[1:] == 1.0)


def test_gap_cesta_nema_npx_prah():
    """interpTrack (runTrackProcessing.m:879-926) zadny npx gate nema.

    Kdyz maska zakryje vetsinu okna, detekcni cesta vrati NaN, ale gap cesta
    musi vratit amplitudu -- MATLAB ji zapisuje bezpodminecne (:911).
    """
    from cmepython import dynamin_intensity_gap
    from cmepython.slave_intensity import fit_gaussian_2d_point
    img = synth_frame([(32.0, 32.0)], amplitude=800.0)
    # maska nechavajici jen par pixelu: vse krome uzkeho pruhu je "soused"
    labels = np.ones_like(img, dtype=int) * 2
    labels[30:34, 30:34] = 0                 # centralni komponenta = pozadi
    labels[32, 32] = 0

    det = fit_gaussian_2d_point(img, 32.0, 32.0, SIGMA_S, mode="xyAc",
                                labels=labels)
    gap = dynamin_intensity_gap(img[None, ...], 0, 32.0, 32.0,
                                sigma_slave=SIGMA_S, sigma_master=SIGMA_M,
                                A_init=700.0, c_init=BG, labels=labels)
    assert not det["valid"]                  # detekcni cesta: npx < 10 -> NaN
    assert np.isfinite(gap["A"])             # gap cesta: vzdy amplituda


def test_gap_cesta_nevraci_pole_ktera_interptrack_nedela():
    """interpTrack vraci deset poli; hval_Ar, mask_Ar, RSS, s, x_pstd, y_pstd
    mezi ne nepatri a v ProcessedTracks.mat zustavaji NaN."""
    from cmepython import dynamin_intensity_gap
    img = synth_frame([(32.0, 32.0)], amplitude=600.0)
    r = dynamin_intensity_gap(img[None, ...], 0, 32.0, 32.0,
                              sigma_slave=SIGMA_S, sigma_master=SIGMA_M,
                              A_init=500.0, c_init=BG)
    assert np.isfinite(r["A"]) and np.isfinite(r["pval_Ar"])
    for f in ("hval_Ar", "mask_Ar", "RSS", "s", "x_pstd", "y_pstd"):
        assert np.isnan(r[f]), f


def test_gmm_bic_z_finalnich_parametru():
    """Loglik se musi pocitat po posledni M-krok, jinak je BIC pri vycerpani
    iteraci nadhodnoceny a muze prehodit vyber poctu komponent."""
    from cmepython.psf_calibration import fit_gmm_1d
    rng = np.random.default_rng(9)
    x = np.concatenate([rng.normal(1.3, 0.12, 800), rng.normal(2.7, 0.25, 500)])
    slow = fit_gmm_1d(x, 2, n_iter=3)        # zamerne malo iteraci
    full = fit_gmm_1d(x, 2, n_iter=500)
    assert slow["converged"] is False
    assert full["converged"] is True
    # loglik odpovida vracenym parametrum, ne tem o krok zpet
    for m in (slow, full):
        z = (x[:, None] - m["mu"][None, :]) / m["sigma"][None, :]
        lp = (-0.5*z**2 - np.log(m["sigma"][None, :]) - 0.5*np.log(2*np.pi)
              + np.log(m["weights"][None, :]))
        mx = lp.max(axis=1, keepdims=True)
        ll = float((mx[:, 0] + np.log(np.exp(lp - mx).sum(axis=1))).sum())
        assert ll == pytest.approx(m["loglik"], rel=1e-9)


def test_movie_layout_je_verejny():
    """Skripty ho musi volat misto rucniho rozbaleni shape."""
    from cmepython import movie_layout
    assert callable(movie_layout)


def test_sigma_master_je_povinny():
    """runDetection.m:184 pouziva vzdy 3*sigma MASTER kanalu. Tichy fallback
    na sigma_slave by menil akceptacni polomer a prehazoval volbu fitu."""
    import inspect
    from cmepython import dynamin_intensity
    sig = inspect.signature(dynamin_intensity)
    assert sig.parameters["sigma_master"].default is inspect.Parameter.empty


def test_nan_souradnice_vrati_nan_nespadne():
    """MATLAB round(NaN) = NaN a bod se tise preskoci (fitGaussians2D.m:96,162).
    int(nan) v Pythonu vyhodi ValueError -- dosazitelne z gap cesty."""
    from cmepython.slave_intensity import fit_gaussian_2d_point
    img = synth_frame([(32.0, 32.0)])
    r = fit_gaussian_2d_point(img, np.nan, 32.0, SIGMA_S, mode="Ac")
    assert not r["valid"] and np.isnan(r["A"])
    r2 = fit_gaussian_2d_point(img, 32.0, np.nan, SIGMA_S, mode="Ac")
    assert not r2["valid"] and np.isnan(r2["A"])
