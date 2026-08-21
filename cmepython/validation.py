# -*- coding: utf-8 -*-
# CMEpython -- Python port mereni intenzity dynaminu z cmeAnalysis.
# Copyright (C) 2026 Ruslan Guliev
#
# Odvozeno z cmeAnalysis (DanuserLab, UT Southwestern), GPL-3.0.
# Tento program je svobodny software: muzete jej sirit a upravovat podle
# podminek GNU General Public License verze 3 nebo pozdejsi. Sirenu je
# BEZ JAKEKOLI ZARUKY. Viz soubor LICENSE.
"""Vstupni kontroly dat -- veci, ktere fit sam neodhali a tise by lhal.

Mereni predpoklada dve vlastnosti vstupu, ktere z jednoho okna pixelu
NELZE poznat, a jejich poruseni nevyvola zadnou chybu -- jen systematicky
spatne amplitudy:

  1. HOMOGENNI OSVETLENI. Jednotlivy raw TIRF-SIM snimek ma pres sebe
     periodicky iluminacni pattern; amplituda spotu pak zavisi na tom, kde
     vuci pruhum lezi. Merit se smi az na prumeru odpovidajici 9-tice.
     `detect_sim_pattern` pozna pruhy podle izolovanych piku ve 2D
     vykonovem spektru (prumerovani je vyrusi).

  2. REGISTROVANE KANALY. Slave se cte na souradnici mastera; posun
     kanalu ~1 px stoji u slabeho signalu desitky procent amplitudy
     (fixni 'Ac' fit posun nekompenzuje a lokalizovany 'xyAc' jen u
     silneho signalu). `estimate_channel_shift` posun zmeri FFT krizovou
     korelaci se subpixelovou interpolaci.

`measure_movie` oboji voli automaticky (parametr `validate`).
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = ["detect_sim_pattern", "estimate_channel_shift", "check_movie_frame"]


def _hann2d(shape):
    wy = np.hanning(shape[0])
    wx = np.hanning(shape[1])
    return np.outer(wy, wx)


def detect_sim_pattern(img, peak_ratio_threshold=10.0,
                       dc_exclude_frac=0.03):
    """Pozna periodicky (SIM) iluminacni pattern v jednom snimku.

    Princip: pruhy = temer bodove piky ve 2D vykonovem spektru. Pik se
    porovnava s medianem LOKALNIHO mezikruzi (5-10 px kolem kandidata),
    ne s radialnim prumerem -- upsamplovana data maji ctvercovou
    spektralni podporu a radialni baseline na nich kolabuje (zmereno:
    dava falesne pozitivy s pomerem >100 i na cistem prumeru).

    Pozn.: symetrie piku v +-k paru se testovat NEDA -- vykonove
    spektrum realneho obrazu je bodove symetricke vzdy (Hermitovska
    symetrie), takze nenese zadnou informaci.

    Returns
    -------
    dict: patterned (bool), peak_ratio (float),
    freq_cycles_per_px (float) -- frekvence nejsilnejsiho kandidata.
    """
    from scipy.ndimage import uniform_filter

    a = np.asarray(img, dtype=float)
    if a.ndim != 2 or min(a.shape) < 64:
        raise ValueError("ocekavam 2D snimek aspon 64x64")
    if not np.isfinite(a).all():
        a = np.where(np.isfinite(a), a, np.nanmedian(a))
    if a.std() == 0:
        return dict(patterned=False, peak_ratio=0.0, freq_cycles_per_px=0.0)

    a = (a - a.mean()) * _hann2d(a.shape)
    P = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
    # 3x3 vyhlazeni srazi exponencialni spicky sumu; delta-pik patternu prezije
    P = uniform_filter(P, size=3)

    ny, nx = P.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(yy - cy, xx - cx)
    core = r < max(4.0, dc_exclude_frac * min(ny, nx))

    Psearch = P.copy()
    Psearch[core] = -np.inf
    iy, ix = np.unravel_index(np.argmax(Psearch), P.shape)

    # lokalni pozadi: median mezikruzi 5-10 px kolem kandidata. Bodovy pik
    # ma mezikruzi na urovni pozadi (obrovsky pomer); bod na hrane
    # spektralni podpory ma pul mezikruzi na podobne urovni (pomer ~2-4).
    ly, lx = np.mgrid[-10:11, -10:11]
    ring = (np.hypot(ly, lx) >= 5) & (np.hypot(ly, lx) <= 10)
    y0, y1 = max(iy - 10, 0), min(iy + 11, ny)
    x0, x1 = max(ix - 10, 0), min(ix + 11, nx)
    patch = P[y0:y1, x0:x1]
    rpatch = ring[(y0 - iy + 10):(y1 - iy + 10), (x0 - ix + 10):(x1 - ix + 10)]
    local_bg = float(np.median(patch[rpatch])) if rpatch.any() else 0.0
    peak = float(P[iy, ix] / local_bg) if local_bg > 0 else 0.0

    freq = float(np.hypot(iy - cy, ix - cx) / min(ny, nx))
    return dict(patterned=bool(peak >= peak_ratio_threshold),
                peak_ratio=peak, freq_cycles_per_px=freq)


def estimate_channel_shift(img_master, img_slave, highpass_sigma=3.0,
                           max_shift=12):
    """Zmeri posun mezi dvema kanaly FFT krizovou korelaci.

    Vraci dict: dy, dx, magnitude (px), strength -- Pearsonova korelace
    v optimalnim zarovnani. Pri strength < ~0.05 je posun neurcitelny
    (kanaly nesdileji dost spolecne struktury) -- nerikat "registrovano",
    rikat "nelze overit".

    Pozn.: ruzne sirky PSF mezi kanaly mohou pik mirne vychylit; na
    tomto datasetu (sigma 1.4 vs 2.6 px) je chyba < ~1 px. Kontrola
    slouzi k odhaleni HRUBE neregistrovanych dat, ne k presne registraci.
    """
    from scipy.ndimage import gaussian_filter

    A = np.asarray(img_master, dtype=float)
    B = np.asarray(img_slave, dtype=float)
    if A.shape != B.shape:
        raise ValueError("kanaly maji ruzne rozmery")

    w = _hann2d(A.shape)
    a = (A - gaussian_filter(A, highpass_sigma)) * w
    b = (B - gaussian_filter(B, highpass_sigma)) * w
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)

    cc = np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))))
    cc = np.fft.fftshift(cc) / a.size
    cy, cx = np.array(cc.shape) // 2
    win = cc[cy - max_shift:cy + max_shift + 1, cx - max_shift:cx + max_shift + 1]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)

    def _sub(v, i):
        if 0 < i < len(v) - 1:
            d = (v[i - 1] - v[i + 1]) / (2 * (v[i - 1] - 2 * v[i] + v[i + 1]) + 1e-12)
            return i + float(np.clip(d, -0.5, 0.5))
        return float(i)

    dy = _sub(win[:, ix], iy) - max_shift
    dx = _sub(win[iy, :], ix) - max_shift
    return dict(dy=float(dy), dx=float(dx),
                magnitude=float(np.hypot(dy, dx)),
                strength=float(win[iy, ix]))


def check_movie_frame(slave_frame, master_frame=None, shift_warn_px=1.0,
                      warn=True, context=""):
    """Spusti obe kontroly na jednom snimku a vrati seznam nalezu.

    Vola se automaticky z `measure_movie` (parametr validate). Nalezy
    jsou varovani, ne chyby -- uzivatel muze mit duvod pokracovat.
    """
    findings = []

    sim = detect_sim_pattern(slave_frame)
    if sim["patterned"]:
        findings.append(
            f"SIM pattern v mericim kanale (pik {sim['peak_ratio']:.0f}x nad "
            f"lokalnim pozadim spektra, frekvence "
            f"{sim['freq_cycles_per_px']:.3f} cyklu/px). Vypada to na "
            "JEDNOTLIVY raw SIM snimek -- amplitudy budou zavisle na poloze "
            "vuci pruhum. Zprumerujte odpovidajici 9-tice pred merenim.")

    if master_frame is not None:
        sim_m = detect_sim_pattern(master_frame)
        if sim_m["patterned"]:
            findings.append(
                f"SIM pattern v master kanale (pik {sim_m['peak_ratio']:.0f}x)."
                " Detekce pozic na patternovanem snimku bude vychylena.")
        sh = estimate_channel_shift(master_frame, slave_frame)
        if sh["strength"] < 0.05:
            findings.append(
                f"Registraci kanalu nelze overit (korelace {sh['strength']:.3f}"
                " je prilis slaba -- kanaly nesdileji dost spolecnych struktur).")
        elif sh["magnitude"] > shift_warn_px:
            findings.append(
                f"Kanaly vypadaji NEREGISTROVANE: posun {sh['magnitude']:.2f} px"
                f" (dy={sh['dy']:+.2f}, dx={sh['dx']:+.2f}, korelace "
                f"{sh['strength']:.2f}). Fixni 'Ac' fit posun nekompenzuje;"
                " u slabeho signalu to stoji desitky procent amplitudy."
                " Registrujte data, nebo transformujte souradnice.")

    if warn:
        for f in findings:
            warnings.warn(("" if not context else context + ": ") + f,
                          RuntimeWarning, stacklevel=3)
    return findings
