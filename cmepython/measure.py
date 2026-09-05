# -*- coding: utf-8 -*-
# CMEpython -- Python port mereni intenzity dynaminu z cmeAnalysis.
# Copyright (C) 2026 Ruslan Guliev
#
# Odvozeno z cmeAnalysis (DanuserLab, UT Southwestern), GPL-3.0.
# Tento program je svobodny software: muzete jej sirit a upravovat podle
# podminek GNU General Public License verze 3 nebo pozdejsi. Sirenu je
# BEZ JAKEKOLI ZARUKY. Viz soubor LICENSE.
"""Prakticka vrstva nad `slave_intensity` -- davkove mereni a cteni TIFF.

`slave_intensity` je verny port jednoho bodu (runDetection.m:180-190).
Tento modul k nemu pridava to, co v MATLABu obstarava smycka pres detekce
a `readtiff`, a co je potreba, aby se dalo merit na celem datasetu:

  * `measure_frame`   -- vsechny detekce v jednom snimku
  * `measure_coords`  -- libovolny seznam [frame, y, x] nad jednim filmem
  * `measure_movie`   -- totez, ale s lazy ctenim TIFF z disku

Zrychleni proti volani `dynamin_intensity` bod po bodu:
  - snimek se cte z disku jednou, ne pro kazdou detekci
  - `i_range` (min/max celeho snimku, fitGaussians2D.m:102) se pocita jednou
  - volitelne paralelizace pres procesy

Zavislosti: numpy, scipy, tifffile.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .slave_intensity import dynamin_intensity

__all__ = ["measure_frame", "measure_coords", "measure_movie",
           "movie_layout", "FIELDS"]

# Pole vracena `dynamin_intensity`, ktera davkujeme do poli.
FIELDS = ("A", "c", "x", "y", "s", "A_pstd", "c_pstd", "sigma_r",
          "SE_sigma_r", "RSS", "pval_Ar", "hval_Ar", "hval_AD",
          "mask_Ar", "npx")


def _empty(n):
    out = {}
    for f in FIELDS:
        if f in ("hval_Ar", "hval_AD"):
            out[f] = np.zeros(n, dtype=bool)
        elif f in ("npx", "mask_Ar"):
            out[f] = np.zeros(n, dtype=np.int64)
        else:
            out[f] = np.full(n, np.nan, dtype=float)
    out["valid"] = np.zeros(n, dtype=bool)
    return out


def measure_frame(img, ys, xs, sigma_slave, sigma_master,
                  localize=True, alpha=0.05, alpha_t=0.05, i_range=None):
    """Zmeri vsechny zadane pozice v JEDNOM snimku slave kanalu.

    `i_range` se spocte jednou pro cely snimek -- presne jak to dela
    fitGaussians2D.m:102, kde je to jedina globalni velicina uvnitr jinak
    lokalni rutiny.
    """
    img = np.asarray(img, dtype=float)
    ys = np.asarray(ys, dtype=float).ravel()
    xs = np.asarray(xs, dtype=float).ravel()
    if ys.size != xs.size:
        raise ValueError("ys a xs musi mit stejnou delku")

    if i_range is None:
        i_range = (float(np.nanmin(img)), float(np.nanmax(img)))

    video = img[None, ...]
    out = _empty(ys.size)
    for k in range(ys.size):
        r = dynamin_intensity(video, 0, ys[k], xs[k],
                              sigma_slave=sigma_slave, sigma_master=sigma_master,
                              localize=localize, alpha=alpha, alpha_t=alpha_t,
                              i_range=i_range)
        for f in FIELDS:
            out[f][k] = r[f]
        out["valid"][k] = r["valid"]
    return out


def measure_coords(video, coords, sigma_slave, sigma_master, **kw):
    """Zmeri seznam souradnic [frame, y, x] nad uz nactenym filmem.

    `video` je pole (T, Y, X) jednoho kanalu. Souradnice se seskupi po
    snimcich, takze se `i_range` pocita jednou na snimek.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords musi mit tvar (N, 3) -- [frame, y, x]")

    out = _empty(len(coords))
    frames = coords[:, 0].astype(int)
    for t in np.unique(frames):
        sel = np.nonzero(frames == t)[0]
        part = measure_frame(video[t], coords[sel, 1], coords[sel, 2],
                             sigma_slave, sigma_master, **kw)
        for f in FIELDS:
            out[f][sel] = part[f]
        out["valid"][sel] = part["valid"]
    return out


def movie_layout(tf):
    """Zjisti (T, C, Y, X) a mapovani (t, c) -> cislo stranky.

    tifffile hlasi ruzne poradi os podle toho, co soubor obsahuje a jaka
    metadata nese: 'TCYX' (realne hyperstacky), ale take 'ZCYX', 'ZYX',
    'CYX' nebo 'YX'. Rozbalovat shape natvrdo do ctyr promennych je chyba --
    jednokanalovy soubor ma jen tri osy.

    'Z' se bere jako casova osa, kdyz 'T' chybi: ImageJ uklada jednokanalovou
    casovou serii bez explicitnich metadat prave jako Z-stack.

    Pozn.: soubor s osami 'CYX' je principialne nejednoznacny -- z nej nelze
    poznat 3 snimky od 3 kanalu. Drzime se toho, co rika tifffile.
    """
    ser = tf.series[0]
    axes, shape = ser.axes, ser.shape
    dims = dict(zip(axes, shape))

    Y = dims.get("Y", shape[-2] if len(shape) >= 2 else 1)
    X = dims.get("X", shape[-1])
    C = dims.get("C", 1)
    # 'Z' nebo 'I' zastupuje cas, pokud 'T' neni pritomno: ImageJ uklada
    # jednokanalovou serii bez metadat jako Z-stack ci 'I' (image sequence)
    time_axis = "T" if "T" in dims else ("Z" if "Z" in dims
                                         else ("I" if "I" in dims else None))
    T = dims.get(time_axis, 1) if time_axis else 1

    lead = [a for a in axes if a not in ("Y", "X")]
    known = {time_axis, "C"} - {None}
    if any(a not in known for a in lead):
        raise ValueError(f"nepodporovane poradi os: {axes!r}")

    if lead == [time_axis, "C"]:
        def idx(t, c): return t * C + c
    elif lead == ["C", time_axis]:
        def idx(t, c): return c * T + t
    elif lead == [time_axis]:
        def idx(t, c): return t
    elif lead == ["C"]:
        def idx(t, c): return c
    elif not lead:
        def idx(t, c): return 0
    else:
        raise ValueError(f"nepodporovane poradi os: {axes!r}")

    return T, C, Y, X, idx


def _one_frame(args):
    """Worker pro ProcessPoolExecutor. Musi byt na urovni modulu, aby
    fungoval i pod `spawn` (Windows, macOS) -- ne jen pod `fork`."""
    (path, t, channel, ys, xs, sigma_slave, sigma_master, kw) = args
    import tifffile
    with tifffile.TiffFile(path) as tf:
        _, _, _, _, idx = movie_layout(tf)
        img = tf.pages[idx(t, channel)].asarray().astype(np.float64)
    return t, measure_frame(img, ys, xs, sigma_slave, sigma_master, **kw)


def measure_movie(path, coords, sigma_slave=None, sigma_master=None,
                  slave_channel=2, workers=None, validate=True,
                  master_channel=None, shift_warn_px=1.0, **kw):
    """Zmeri souradnice [frame, y, x] primo nad TIFF na disku.

    Snimky se ctou lazy, jeden po druhem -- 950 MB film se nikdy nedrzi
    cely v pameti. S `workers > 1` bezi snimky paralelne v procesech.

    Parameters
    ----------
    path : cesta k .tif (ImageJ hyperstack TCYX)
    coords : (N, 3) pole [frame, y, x]
    sigma_slave : sirka PSF mericiho kanalu v px, nebo None. Pri None se
        AUTOMATICKY odhadne z tohoto filmu (estimate_psf_sigma pres ~6
        snimku + clamp z runDetection.m:90-92) a vypise varovani s pouzitou
        hodnotou. Pro srovnatelnost napric filmy je spravnejsi kalibrovat
        pres CELY dataset (scripts/calibrate_dataset.py) a hodnotu predat.
    sigma_master : sirka PSF master kanalu (gate 3*sigma na
        runDetection.m:184), nebo None -- pak je nutne zadat
        `master_channel`, ze ktereho se odhadne.
    slave_channel : index kanalu, na kterem se meri.
    workers : None nebo 1 = serialne, 0 = vsechna jadra, >1 = tolik procesu.
        Pozn.: pri vice nez ~4 procesech se na strojich s vicevlaknovym BLAS
        casto projevi oversubscription a zrychleni klesa.
    validate : True = pred merenim zkontrolovat prvni pouzity snimek:
        SIM pattern (jednotlivy neprumerovany raw snimek) a -- pokud je
        zadan `master_channel` -- registraci kanalu. Nalezy jsou varovani,
        mereni pokracuje. Viz cmepython.validation.
    master_channel : index master kanalu; potreba pro kontrolu registrace
        a pro auto-odhad sigma_master.
    shift_warn_px : prah posunu kanalu pro varovani (px).
    """
    import tifffile

    path = str(Path(path))
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords musi mit tvar (N, 3) -- [frame, y, x]")

    with tifffile.TiffFile(path) as tf:
        T, C, Y, X, _ = movie_layout(tf)
    if not (0 <= slave_channel < C):
        raise ValueError(f"slave_channel {slave_channel} mimo rozsah (film ma {C} kanalu)")

    frames = coords[:, 0].astype(int)
    if frames.min() < 0 or frames.max() >= T:
        raise ValueError(f"snimky mimo rozsah 0..{T - 1}")

    # ---- auto-kalibrace sigmy (kdyz neni predana) --------------------------
    if sigma_slave is None or sigma_master is None:
        from .psf_calibration import estimate_psf_sigma, apply_sigma_clamp
        with tifffile.TiffFile(path) as tf:
            _, _, _, _, idx = movie_layout(tf)
            tcal = np.unique(np.linspace(0, T - 1, min(6, T)).astype(int))
            if sigma_slave is None:
                fr = [tf.pages[idx(t, slave_channel)].asarray().astype(np.float64)
                      for t in tcal]
                sigma_slave, fired = apply_sigma_clamp(estimate_psf_sigma(fr))
                warnings.warn(
                    f"sigma_slave odhadnuta z TOHOTO filmu: {sigma_slave:.4f} px"
                    + (" (clamp 1.1)" if fired else "")
                    + ". Pro srovnatelnost napric filmy kalibrujte pres cely "
                    "dataset (scripts/calibrate_dataset.py) a hodnotu predejte.",
                    RuntimeWarning, stacklevel=2)
            if sigma_master is None:
                if master_channel is None:
                    raise ValueError(
                        "sigma_master=None vyzaduje master_channel, ze ktereho "
                        "se da odhadnout -- gate na runDetection.m:184 pouziva "
                        "vzdy sigmu MASTER kanalu a tichy fallback neexistuje")
                fr = [tf.pages[idx(t, master_channel)].asarray().astype(np.float64)
                      for t in tcal]
                sigma_master, fired = apply_sigma_clamp(estimate_psf_sigma(fr))
                warnings.warn(
                    f"sigma_master odhadnuta z tohoto filmu: {sigma_master:.4f} px"
                    + (" (clamp 1.1)" if fired else ""),
                    RuntimeWarning, stacklevel=2)

    # ---- vstupni kontroly (SIM pattern, registrace) ------------------------
    if validate:
        from .validation import check_movie_frame
        t0 = int(frames.min())
        with tifffile.TiffFile(path) as tf:
            _, _, _, _, idx = movie_layout(tf)
            sl = tf.pages[idx(t0, slave_channel)].asarray().astype(np.float64)
            ma = None
            if master_channel is not None:
                ma = tf.pages[idx(t0, master_channel)].asarray().astype(np.float64)
        check_movie_frame(sl, ma, shift_warn_px=shift_warn_px,
                          context=Path(path).name)

    groups = [(t, np.nonzero(frames == t)[0]) for t in np.unique(frames)]
    out = _empty(len(coords))

    # POZOR: `workers == 0` znamena "vsechna jadra", NIKOLI "serialne".
    # Testovat `if not workers` by nulu chybne poslalo do serialni vetve.
    if workers is None or workers == 1:
        with tifffile.TiffFile(path) as tf:
            _, _, _, _, idx = movie_layout(tf)
            for t, sel in groups:
                img = tf.pages[idx(t, slave_channel)].asarray().astype(np.float64)
                part = measure_frame(img, coords[sel, 1], coords[sel, 2],
                                     sigma_slave, sigma_master, **kw)
                for f in FIELDS:
                    out[f][sel] = part[f]
                out["valid"][sel] = part["valid"]
        return out

    tasks = [(path, int(t), slave_channel, coords[sel, 1], coords[sel, 2],
              sigma_slave, sigma_master, kw) for t, sel in groups]
    index = {int(t): sel for t, sel in groups}

    # POZOR NA PRENOSITELNOST: macOS a Windows startuji procesy pres `spawn`,
    # ktery potrebuje importovatelny hlavni modul. Z interaktivni session,
    # heredocu nebo Jupyteru to spadne na BrokenProcessPool. Nechceme, aby
    # kvuli tomu selhalo cele mereni -- degradujeme na serialni beh.
    try:
        with ProcessPoolExecutor(max_workers=None if workers == 0 else workers) as ex:
            for t, part in ex.map(_one_frame, tasks):
                sel = index[int(t)]
                for f in FIELDS:
                    out[f][sel] = part[f]
                out["valid"][sel] = part["valid"]
        return out
    except Exception as exc:
        warnings.warn(
            f"paralelni beh selhal ({type(exc).__name__}), pokracuji serialne. "
            "Pod 'spawn' (macOS/Windows) musi byt volajici kod ve skriptu "
            "s `if __name__ == \'__main__\':`, ne v interaktivni session.",
            RuntimeWarning, stacklevel=2)
        return measure_movie(path, coords, sigma_slave, sigma_master,
                             slave_channel=slave_channel, workers=None, **kw)
