# -*- coding: utf-8 -*-
# CMEpython -- Python port mereni intenzity dynaminu z cmeAnalysis.
# Copyright (C) 2026 Ruslan Guliev
#
# Odvozeno z cmeAnalysis (DanuserLab, UT Southwestern), GPL-3.0.
# Tento program je svobodny software: muzete jej sirit a upravovat podle
# podminek GNU General Public License verze 3 nebo pozdejsi. Sirenu je
# BEZ JAKEKOLI ZARUKY. Viz soubor LICENSE.
"""
cme_slave_intensity.py

Port detekcni faze mereni intenzity slave kanalu (dynamin) z cmeAnalysis.
Zavislosti: numpy, scipy.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t as _student_t
from scipy.stats import norm as _norm

__all__ = ["dynamin_intensity", "dynamin_intensity_gap", "fit_gaussian_2d_point"]


# --------------------------------------------------------------------------
# Pomocne funkce
# --------------------------------------------------------------------------

def _mround(v):
    """MATLAB round(): pulky se zaokrouhluji SMEREM OD NULY.

    np.round() pouziva bankerske zaokrouhlovani (round-half-to-even), takze
    napr. np.round(12.5) == 12 zatimco MATLAB round(12.5) == 13. Pro
    souradnici presne na .5 by se okno posunulo o pixel.
    Viz fitGaussians2D.m:96-97 (xi = round(x); yi = round(y);).
    """
    v = np.asarray(v, dtype=float)
    return np.sign(v) * np.floor(np.abs(v) + 0.5)


def _nan_result(npx=0):
    """Predalokovany 'neuspesny' vysledek.

    Zrcadli fitGaussians2D.m:108-131, kde jsou vsechna vystupni pole
    predinicializovana na NaN / False a bod, ktery neprojde nekterou z
    ctyr kontrol, si je ponecha.
    """
    return dict(
        x=np.nan, y=np.nan, A=np.nan, s=np.nan, c=np.nan,
        x_pstd=np.nan, y_pstd=np.nan, A_pstd=np.nan, s_pstd=np.nan, c_pstd=np.nan,
        sigma_r=np.nan, SE_sigma_r=np.nan, RSS=np.nan,
        pval_Ar=np.nan, hval_Ar=False, hval_AD=False, mask_Ar=0,
        npx=int(npx), mode="", valid=False,
    )



def _anderson_darling_normal(x, alpha=0.05):
    """Anderson-Darlinguv test normality s odhadnutymi mu/sigma.

    Vraci True, kdyz se normalita ZAMITA (tj. ekvivalent `hval_AD == 1`
    v fitGaussians2D.m:221).

    Kriticke hodnoty jsou tabulkove konstanty pro pripad, kdy jsou stredni
    hodnota i rozptyl odhadnuty z dat (Stephens 1974). Pocitame je zde primo
    misto volani scipy.stats.anderson, jehoz API se v SciPy 1.19 meni.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return False
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return False
    z = np.sort((x - x.mean()) / sd)
    cdf = _norm.cdf(z)
    eps = np.finfo(float).eps
    cdf = np.clip(cdf, eps, 1.0 - eps)
    i = np.arange(1, n + 1)
    A2 = -n - np.mean((2 * i - 1) * (np.log(cdf) + np.log1p(-cdf[::-1])))
    # korekce na male vzorky (Stephens 1974)
    A2_star = A2 * (1.0 + 0.75 / n + 2.25 / n ** 2)
    crit = {0.15: 0.576, 0.10: 0.656, 0.05: 0.787, 0.025: 0.918, 0.01: 1.092}
    key = min(crit, key=lambda a: abs(a - alpha))
    return bool(A2_star > crit[key])


def _gauss_kernel(w4, sigma):
    """Nenormalizovane Gaussovo jadro pres cele okno, peak == 1.

    MATLAB (fitGaussians2D.m:153-155):
        g = exp(-(-w4:w4).^2/(2*sigma_max^2));  g = g'*g;  g = g(:);
    coz je presne outer product == exp(-(x^2+y^2)/(2 sigma^2)).
    Pouziva se JEN pro mask_Ar (pocet pixelu, kde modelovy signal prevysi
    k*sigma_r), nikoli pro fit samotny.
    """
    v = np.exp(-np.arange(-w4, w4 + 1, dtype=float) ** 2 / (2.0 * sigma ** 2))
    return np.outer(v, v).ravel()


# --------------------------------------------------------------------------
# Jadro: jeden bod, jeden fit — port fitGaussians2D.m pro np == 1
# --------------------------------------------------------------------------

def fit_gaussian_2d_point(img, x, y, sigma, mode="Ac",
                          A_init=None, c_init=None,
                          labels=None, i_range=None,
                          alpha=0.05, alpha_t=0.05,
                          conf_radius=None, window_size=None,
                          min_npx=10):
    """Port fitGaussians2D.m (soubor cmeAnalysis/software/fitGaussians2D.m) pro
    jediny bod.

    REPRODUKUJE presne:
      * geometrii okna a mezikruzi   fitGaussians2D.m:134-155
      * border test (skip -> NaN)    fitGaussians2D.m:162
      * maskovani sousednich komponent a npx  :165-181
      * npx >= 10 gate               :183
      * A_init = max(window)-c_init  :187
      * akceptacni test              :198  (|dx|,|dy| < w2 a A < 2*diff(iRange))
      * prirazeni A = prm(3)         :202
      * rozprostreni prmStd pres estIdx  :104, :207-213
      * statistiky sigma_r, SE_sigma_r, df2, scomb, T, pval_Ar, hval_Ar, mask_Ar
                                     :215-237

    APROXIMUJE (viz komentare u konkretnich radku):
      * samotny LM solver: MATLAB pouziva kompilovany MEX fitGaussian2D
        (GSL, zdrojak NENI v repozitari) -> zde scipy.optimize.least_squares
      * normalizaci prmStd z kovariancni matice
      * definici res.std
      * res.hAD (Anderson-Darling) -- MEX doc stub mex/fitGaussian2D.m:24
        dokumentuje misto toho Kolmogorov-Smirnov a pole hAD vubec neuvadi

    Souradnicova konvence: 0-based numpy, y == radek, x == sloupec.
    img musi byt 2D float pole (v MATLABu se dela double(imread(...)),
    runDetection.m:175-177).
    """
    img = np.asarray(img, dtype=np.float64)
    ny, nx = img.shape

    kLevel = _norm.ppf(1.0 - alpha / 2.0)          # fitGaussians2D.m:100 -> ~1.959964

    # fitGaussians2D.m:102 -- JEDINA whole-frame velicina uvnitr jinak
    # ciste lokalni rutiny. Pouziva se pouze jako horni sanity mez na A.
    if i_range is None:
        i_range = (float(np.nanmin(img)), float(np.nanmax(img)))
    d_range = float(i_range[1] - i_range[0])

    # fitGaussians2D.m:134-142 -- sigma_max = max(sigma); zde je sigma skalar
    sigma_max = float(sigma)
    w2 = int(np.ceil(2.0 * sigma_max)) if conf_radius is None else int(conf_radius)
    w4 = int(np.ceil(4.0 * sigma_max)) if window_size is None else int(window_size)

    # MATLAB round(NaN) je NaN a border test na :162 pak vyhodnoti false,
    # takze se bod tise preskoci. int(nan) v Pythonu vyhodi ValueError.
    # Dosazitelne z gap cesty, kde x/y pochazi z interp1 podel tracku
    # a v NaN-oddelenem viacsegmentovem tracku muze byt NaN.
    if not (np.isfinite(x) and np.isfinite(y)):
        return _nan_result()

    xi = int(_mround(x))
    yi = int(_mround(y))

    # fitGaussians2D.m:162 -- border. MATLAB: xi>w4 && xi<=nx-w4 (1-based).
    # Prevedeno do 0-based je to presne "cele okno se vejde do snimku".
    # ZADNY padding, ZADNY oriz -- bod se cely preskoci a zustane NaN.
    if not (w4 <= xi <= nx - w4 - 1 and w4 <= yi <= ny - w4 - 1):
        return _nan_result()

    sl = (slice(yi - w4, yi + w4 + 1), slice(xi - w4, xi + w4 + 1))
    window = img[sl].copy()

    # fitGaussians2D.m:165-166 -- maskovani sousedu.
    # POZOR: na detekcni ceste slave kanalu se Mask NEPREDAVA
    # (runDetection.m:180 / :183), takze labels == zeros a nic se nemaskuje.
    # Aktivni je to jen v interpTrack (gap/buffer snimky).
    if labels is None:
        mask_window = np.zeros_like(window, dtype=np.int64)
    else:
        mask_window = np.asarray(labels, dtype=np.int64)[sl].copy()
        center_label = mask_window[w4, w4]
        mask_window[mask_window == center_label] = 0   # vlastni komponenta se NEmaskuje

    # fitGaussians2D.m:171-177 -- background init z mezikruzi [ceil(3s), ceil(4s)].
    # Pocita se PRED tim, nez se maskovane pixely nastavi na NaN.
    if c_init is None:
        yy, xx = np.mgrid[-w4:w4 + 1, -w4:w4 + 1].astype(float)
        r = np.sqrt(xx ** 2 + yy ** 2)
        cmask = ((r <= np.ceil(4.0 * sigma_max)) & (r >= np.ceil(3.0 * sigma_max)))
        cmask = cmask & (mask_window == 0)
        if not cmask.any():
            return _nan_result()
        c0 = float(np.mean(window[cmask]))
    else:
        c0 = float(c_init)

    # fitGaussians2D.m:180-181
    window[mask_window != 0] = np.nan
    finite = np.isfinite(window)
    npx = int(finite.sum())

    # fitGaussians2D.m:183. Plati jen pro DETEKCNI cestu -- interpTrack
    # (runTrackProcessing.m:879-926) zadny takovy prah nema, spocte npx
    # a rovnou fituje. Proto je to parametr, ne konstanta: gap/buffer cesta
    # predava min_npx=0, jinak by v oknech prekrytych sousedni detekci
    # vracela NaN tam, kde MATLAB vraci amplitudu.
    if npx < int(min_npx):
        return _nan_result(npx)

    # fitGaussians2D.m:186-190. MATLAB max(window(:)) ignoruje NaN -> nanmax.
    A0 = float(np.nanmax(window) - c0) if A_init is None else float(A_init)

    # Pocatecni pozice == sub-pixelovy zbytek master souradnice.
    # fitGaussians2D.m:192, pocatek soustavy je STRED okna (mex/fitGaussian2D.m:10).
    xp0 = float(x) - xi
    yp0 = float(y) - yi

    yy, xx = np.mgrid[-w4:w4 + 1, -w4:w4 + 1].astype(float)
    Xf, Yf, Df = xx[finite], yy[finite], window[finite]

    if mode == "Ac":
        # Volne jen A a c -> problem je LINEARNI. LM najde presne totez, co
        # uzavrena LS formule, takze shoda s MEXem je zde prakticky exaktni
        # (na rozdil od 'xyAc').
        def _resid(p):
            g = np.exp(-((Xf - xp0) ** 2 + (Yf - yp0) ** 2) / (2.0 * sigma_max ** 2))
            return p[0] * g + p[1] - Df
        p_init = np.array([A0, c0], dtype=float)
        est_idx = [2, 4]                       # regexpi('xyAsc','[Ac]') == [3 5] (1-based)
    elif mode == "xyAc":
        def _resid(p):
            g = np.exp(-((Xf - p[0]) ** 2 + (Yf - p[1]) ** 2) / (2.0 * sigma_max ** 2))
            return p[2] * g + p[3] - Df
        p_init = np.array([xp0, yp0, A0, c0], dtype=float)
        est_idx = [0, 1, 2, 4]                 # regexpi('xyAsc','[xyAc]') == [1 2 3 5]
    elif mode == "xyasc":
        # VOLNA sigma. Na slave kanalu se NIKDY nepouziva -- slouzi vyhradne
        # ke kalibraci sirky PSF, getGaussianPSFsigmaFromData.m:67.
        def _resid(p):
            s = p[3]
            if not (0.2 < s < 8.0):
                return np.full(Df.size, 1e7)
            g = np.exp(-((Xf - p[0]) ** 2 + (Yf - p[1]) ** 2) / (2.0 * s ** 2))
            return p[2] * g + p[4] - Df
        p_init = np.array([xp0, yp0, A0, sigma_max, c0], dtype=float)
        est_idx = [0, 1, 2, 3, 4]              # regexpi('xyAsc','[xyasc]') == vse
    else:
        raise ValueError("mode musi byt 'Ac', 'xyAc' nebo 'xyasc' "
                         "(na slave kanalu se pouzivaji jen prvni dva; "
                         "'s' se tam NIKDY nefituje -- runDetection.m:180/183)")

    # POZNAMKA K NUMERICKE SHODE: MEX je GSL Levenberg-Marquardt
    # (gsl_multifit_fdfsolver_lmsder). Jeho tolerance a max. pocet iteraci
    # jsou nedokumentovane (4. argument [maxIter eAbs eRel], mex/fitGaussian2D.m:15)
    # a zdrojovy kod v repozitari NENI. Zde volime prisne tolerance;
    # bit-for-bit shoda neni garantovana.
    try:
        sol = least_squares(_resid, p_init, method="lm",
                            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=2000)
    except Exception:
        return _nan_result(npx)

    p = sol.x
    if mode == "Ac":
        prm = np.array([xp0, yp0, p[0], sigma_max, p[1]], dtype=float)
    elif mode == "xyAc":
        prm = np.array([p[0], p[1], p[2], sigma_max, p[3]], dtype=float)
    else:                                      # xyasc -- sigma je fitovana
        prm = np.array([p[0], p[1], p[2], p[3], p[4]], dtype=float)

    dx, dy = prm[0], prm[1]

    # fitGaussians2D.m:198 -- JEDINY post-fit test v celem souboru.
    # dx a dy jsou omezeny NEZAVISLE PO OSACH vuci zaokrouhlenemu pixelu.
    # Neni tu zadna dolni mez na A (zaporna amplituda projde), zadny test
    # konvergence a zadny explicitni isnan(). NaN v prm ale zpusobi, ze vsechna
    # porovnani jsou False -> bod zustane NaN, coz je de facto NaN guard.
    # MATLAB testuje POUZE prm(1), prm(2) a prm(3). prm(4) (s, fixni) ani
    # prm(5) (c) se na finitnost nekontroluji -- bod s konecnou pozici
    # a amplitudou, ale NaN pozadim, MATLAB PRIJME a zapise c = NaN.
    # NaN porovnani jsou v Pythonu False stejne jako v MATLABu, takze
    # tri realne podminky staci a zadny extra isfinite guard netreba.
    if not (-w2 < dx < w2 and -w2 < dy < w2
            and prm[2] < 2.0 * d_range):
        return _nan_result(npx)

    residuals = sol.fun
    RSS = float(np.sum(residuals ** 2))
    n_free = len(p_init)

    # prmStd = sqrt(diag( RSS/dof * inv(J'J) )).
    #
    # ZMERENO proti MEXu, ne odhadnuto: 874 realnych oken, mody 'Ac' i 'xyAc',
    # pomer vysel konstantni (sd 7e-7) a odpovida dof = npx - n_free - 1
    # v obou modech. MEX tedy odecita jeden stupen volnosti navic proti
    # bezne konvenci npx - n_free. Bez teto opravy jsou A_pstd a vsechny
    # odvozene p-hodnoty systematicky vyssi o ~0.1 %.
    # (Pozn.: filterGaussianFit2D.m:74 pouziva RSS/(n-3) pri dvou volnych
    #  parametrech, coz je tataz konvence -- originál je v tom konzistentni.)
    dof = max(npx - n_free - 1, 1)
    try:
        J = sol.jac
        JTJ_inv = np.linalg.inv(J.T @ J)
        cov = JTJ_inv * (RSS / dof)
        prm_std_free = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        prm_std_free = np.full(n_free, np.nan)

    std_vect = np.zeros(5, dtype=float)         # fitGaussians2D.m:206-207
    std_vect[est_idx] = prm_std_free

    # APROXIMACE: res.std se pocita uvnitr MEXu. Doc stub rika "standard
    # deviation of the residuals" a zvlast uvadi res.mean, coz naznacuje
    # smerodatnou odchylku kolem prumeru. Volime ddof=1, konzistentne s
    # SE_sigma_r = std/sqrt(2*(npx-1)) na fitGaussians2D.m:218.
    # Protoze c je volny parametr, je prumer rezidui ~0 a volba ddof je
    # prakticky bezvyznamna.
    sigma_r = float(np.std(residuals, ddof=1))
    SE_sigma_r = sigma_r / np.sqrt(2.0 * (npx - 1))

    # APROXIMACE: res.hAD. Doc stub mex/fitGaussian2D.m:24 dokumentuje
    # Kolmogorov-Smirnov a pole hAD vubec neuvadi, presto ho volajici cte
    # (fitGaussians2D.m:221). Nazev napovida Anderson-Darling; hladina
    # vyznamnosti je NEZNAMA -- volime 5 %.
    # Pozn.: scipy >= 1.17 varuje, ze `anderson` bez `method` zmeni chovani
    # v 1.19. Kriticke hodnoty pro dist="norm" jsou tabulkove konstanty
    # (Stephens 1974), takze je pocitame primo -- nezavisle na verzi SciPy.
    # Tim je funkce odolna proti budoucim zmenam API i napric platformami.
    hval_AD = _anderson_darling_normal(residuals, alpha=0.05)

    # fitGaussians2D.m:225-237, doslovne prepsano.
    #
    # POZN. KE KORELOVANEMU SUMU: kdyz je kanal preskalovany nebo
    # zprumerovany (u tohoto datasetu je slave kanal 2x zvetseny), nejsou
    # sousedni pixely nezavisle -- namereno r(lag 1) = 0.66, 1D integral
    # autokorelace tau ~ 2.8 (2D integral ~ 20). Formalne to podhodnocuje
    # A_pstd nejmene ~1.7x.
    #
    # ZMERENO (a nezavisle overeno auditem, scripts/dataset_findings.py):
    # na prazdnych pozicich korekce npx -> npx/tau neprehodi ZADNE
    # rozhodnuti (p-hodnoty jsou tam 100% saturovane u 1); na signalnich
    # pozicich prehodi <1 % rozhodnuti pri tau=2.8 (~2 % pri tau=20) --
    # vyhradne hranicni pozitiva s p ~ 0.001-0.003. Duvod: statistika je
    # T = (A - kLevel*sigma_r)/scomb a ten odecet dominuje.
    #
    # Korekci proto zamerne NEZAVADIME -- praktic­ky nic nemeni a "spravna"
    # hodnota tau (1D vs 2D integral) je modelova volba, kterou data
    # nerozhodnou. Zacala by byt potreba, kdyby se test zmenil na prostou
    # nulovou hypotezu A > 0, kde uz body u prahu lezi.
    sigma_A = std_vect[2]
    A_est = prm[2]
    SE_r = SE_sigma_r * kLevel
    denom_df = sigma_A ** 4 + SE_r ** 4
    with np.errstate(divide="ignore", invalid="ignore"):
        df2 = (npx - 1) * (sigma_A ** 2 + SE_r ** 2) ** 2 / denom_df
        scomb = np.sqrt((sigma_A ** 2 + SE_r ** 2) / npx)
        T = (A_est - sigma_r * kLevel) / scomb
        pval_Ar = float(_student_t.cdf(-T, df2))
    hval_Ar = bool(pval_Ar < alpha_t)           # NaN < x je False, stejne jako v MATLABu

    g = _gauss_kernel(w4, sigma_max)
    mask_Ar = int(np.sum(A_est * g > sigma_r * kLevel))

    return dict(
        x=xi + dx, y=yi + dy,
        A=float(prm[2]), s=float(prm[3]), c=float(prm[4]),
        x_pstd=std_vect[0], y_pstd=std_vect[1], A_pstd=std_vect[2],
        s_pstd=std_vect[3], c_pstd=std_vect[4],
        sigma_r=sigma_r, SE_sigma_r=float(SE_sigma_r), RSS=RSS,
        pval_Ar=pval_Ar, hval_Ar=hval_Ar, hval_AD=hval_AD, mask_Ar=mask_Ar,
        npx=npx, mode=mode, valid=True,
    )


# --------------------------------------------------------------------------
# Verejne API: detekcni faze
# --------------------------------------------------------------------------

def dynamin_intensity(video, frame, y, x, sigma_slave, sigma_master,
                      localize=True, alpha=0.05, alpha_t=0.05, i_range=None):
    """Vrati intenzitu dynaminu (slave kanalu) pro souradnice [frame, y, x].

    ============================================================================
    CO PRESNE REPRODUKUJE
    ============================================================================
    Detekcni fazi cmeAnalysis, tj. runDetection.m:180-190. To je hodnota, ktera
    konci v detection_v2.mat a kterou runTrackProcessing.m:388 DOSLOVNE kopiruje
    do tracks(k).A(slaveCh, :) pro kazdy snimek, kde master detekce existuje.
    Pro drtivou vetsinu casovych bodu tracku je tohle "ta" dynaminova intenzita.

    Postup (runDetection.m:180-190):
      1. fit s ZAMCENOU pozici, mod 'Ac' -- volne jsou jen A a c, pozice je
         sub-pixelova master souradnice, sigma je fixni sigma_slave.
      2. fit s VOLNOU pozici, mod 'xyAc', seedovany master pozici a vysledkem
         kroku 1 (jeho A a c).
      3. vysledek kroku 2 se pouzije JEN kdyz se posunul mene nez
         3*sigma_MASTER (euklidovsky) A ZAROVEN dal VETSI amplitudu.
         Jinak plati krok 1.

    Vracena hodnota res["A"] == prm(3) == fitGaussians2D.m:202.
    Je to amplituda Gaussovky NAD lokalnim pozadim c, v surovych kamerovych
    jednotkach. NENI to integrovana intenzita (A*2*pi*s^2 se v celem balicku
    nevyskytuje), NENI to peak pixel a NENI to A+c.

    ============================================================================
    CO NEREPRODUKUJE / APROXIMUJE
    ============================================================================
    * MEX fitGaussian2D (GSL Levenberg-Marquardt) -- zdrojovy kod NENI v
      repozitari, jen binarky. Tolerance, max. iterace, normalizace prmStd,
      definice res.std a test za res.hAD jsou NEZJISTITELNE. Pro mod 'Ac' je
      problem linearni v A a c, takze A odpovida prakticky exaktne; pro 'xyAc'
      a pro A_pstd / sigma_r / pval_Ar / hval_AD ocekavejte male odchylky.
    * Gap a buffer snimky. Ty pocita jina kodova cesta (interpTrack,
      runTrackProcessing.m:879-926) s JINYM oknem (ceil(4*sigma_MASTER)),
      JINOU inicializaci (interpolovane A, c) a JINYM poradim fitu.
      Viz dynamin_intensity_gap().
    * Cross-movie EDF preskalovani a(i) (getLifetimeData.m:223,234), ktere je
      defaultne ZAPNUTE ve vsech cohortnich vystupech. Standalone funkce ho
      spocitat nemuze -- vyzaduje maximalni amplitudy vsech tracku vsech filmu
      podminky. Vracene hodnoty odpovidaji ProcessedTracks.mat, NE publikovanym
      cohortnim obrazkum.
    * Klasifikaci significantSlave / significantMaster / significantVsBackground
      (runSlaveChannelClassification.m:145,168,173). Ta potrebuje whole-movie
      pozadovou distribuci (bg95, pDetection z 16 snimku + cell mask).
      Lokalni nahradou je vracene pval_Ar / hval_Ar -- je to jiny, mene
      konzervativni test.
    * Skutecnost, ze cmeAnalysis pri NaN slave fitu SMAZE detekci ze VSECH
      kanalu (runDetection.m:192-195). Tato funkce jen vrati NaN.

    ============================================================================
    PARAMETRY
    ============================================================================
    video : array-like (T, Ny, Nx)
        Stack SLAVE (dynaminoveho) kanalu. Bude prevedeno na float64.
    frame : int          0-based index snimku
    y, x  : float        SUB-PIXELOVA master (klatrinova) pozice, 0-based,
                         y == radek, x == sloupec. Predani celociselneho pixelu
                         zmeni LM trajektorii a systematicky podhodnoti A
                         o jednotky procent.
    sigma_slave : float
        PSF sigma slave kanalu v pixelech. V cmeAnalysis je to defaultne
        multi-movie odhad (runDetection.m:79, ~40 snimku ze VSECH filmu
        podminky), clampnuty na >= 1.1 (runDetection.m:92). Pro shodu s
        existujici analyzou pouzijte frameInfo(1).s z detection_v2.mat.
    sigma_master : float
        PSF sigma MASTER kanalu; vstupuje POUZE do gate 3*sigma(mCh)
        na runDetection.m:184. POVINNY parametr -- zadny fallback na
        sigma_slave neexistuje, protoze by tise menil akceptacni polomer
        a prehazoval volbu mezi obema fity.
    localize : bool
        False vypne druhy ('xyAc') fit -- vrati se ciste pozicne zamcena
        amplituda.
    i_range : (min, max) or None
        Whole-frame rozsah slave snimku pro sanity gate na fitGaussians2D.m:198.
        Pri None se spocita z celeho snimku (tak to dela MATLAB,
        fitGaussians2D.m:102).

    ============================================================================
    NAVRAT
    ============================================================================
    dict s klici: A, c, x, y, s, A_pstd, c_pstd, x_pstd, y_pstd, sigma_r,
    SE_sigma_r, RSS, pval_Ar, hval_Ar, hval_AD, mask_Ar, npx, mode, valid.
    Pri selhani (okraj / npx<10 / neprijaty fit) je valid=False a vsechna
    ciselna pole jsou NaN -- presne jako fitGaussians2D.m:108-131.
    """
    img = np.asarray(video[frame], dtype=np.float64)

    if i_range is None:
        i_range = (float(np.nanmin(img)), float(np.nanmax(img)))

    # runDetection.m:180 -- FIT 1: pozice zamcena na master souradnici,
    # A=[] a c=[] => obe pocatecni hodnoty se odvodi z okna (mezikruzi / max).
    # Mask se NEPREDAVA -> zadne maskovani sousednich detekci.
    fixed = fit_gaussian_2d_point(img, x, y, sigma_slave, mode="Ac",
                                  A_init=None, c_init=None, labels=None,
                                  i_range=i_range, alpha=alpha, alpha_t=alpha_t)

    if not localize or not fixed["valid"]:
        return fixed

    # runDetection.m:183 -- FIT 2: pozice volna, seedovana master pozici
    # a vysledkem fitu 1.
    loc = fit_gaussian_2d_point(img, x, y, sigma_slave, mode="xyAc",
                                A_init=fixed["A"], c_init=fixed["c"], labels=None,
                                i_range=i_range, alpha=alpha, alpha_t=alpha_t)

    # runDetection.m:184 -- akceptacni pravidlo. POZOR na dve veci:
    #  (a) vzdalenost je EUKLIDOVSKA a mez je 3*sigma_MASTER, zatimco clamp
    #      uvnitr fitu (fitGaussians2D.m:198) je PO OSACH a pouziva
    #      ceil(2*sigma_SLAVE). Jsou to dve ruzna kriteria, nelze je slucovat.
    #  (b) v MATLABu je NaN > A vzdy false, takze neuspesny lokalizovany fit
    #      tise propadne na fit 1. Zde je to napsane explicitne.
    if not loc["valid"] or not np.isfinite(loc["A"]):
        return fixed

    dist = np.hypot(float(x) - loc["x"], float(y) - loc["y"])
    if dist < 3.0 * float(sigma_master) and loc["A"] > fixed["A"]:
        return loc
    return fixed


# --------------------------------------------------------------------------
# Verejne API: gap / buffer varianta (interpTrack)
# --------------------------------------------------------------------------


# interpTrack vraci JEN deset poli (runTrackProcessing.m:896-926):
# x, y, A, c, A_pstd, c_pstd, sigma_r, SE_sigma_r, hval_AD, pval_Ar.
# hval_Ar, mask_Ar, RSS, s, x_pstd a y_pstd tam nikdy nevzniknou a
# mergeStructs (:930-935) je proto nechava na NaN z predalokace (:341-343).
# Kdybychom je vraceli, volajici by pro gap a buffer snimky cetl True/False
# tam, kde ProcessedTracks.mat drzi NaN.
_NOT_IN_INTERPTRACK = ("hval_Ar", "mask_Ar", "RSS", "s", "x_pstd", "y_pstd")


def _interp_track_fields(res):
    for field in _NOT_IN_INTERPTRACK:
        res[field] = np.nan
    return res


def dynamin_intensity_gap(video, frame, y, x, sigma_slave, sigma_master,
                          A_init, c_init, labels=None, alpha=0.05):
    """Port interpTrack (runTrackProcessing.m:879-926) -- varianta pouzita
    VYHRADNE pro gap snimky a pro +-5 buffer snimku na zacatku/konci tracku
    (volani na runTrackProcessing.m:510-511, 526-527, 542-543).

    Rozdily oproti detekcni fazi, ktere je nutne dodrzet:
      * velikost okna a konfinacni polomer se odvozuji ze sigma_MASTER
        (runTrackProcessing.m:884-885), zatimco fitovana sirka PSF je
        sigma_slave (argument sigmaCh). Tato asymetrie je v puvodnim kodu
        skutecne pritomna.
      * poradi fitu je OPACNE: nejdriv 'xyAc', teprve pri uteku za w2
        fallback na 'Ac' (runTrackProcessing.m:896 / :905).
      * ZADNY NaN vystup -- ps.A = prm(3) se zapisuje bezpodminecne
        (runTrackProcessing.m:911). Neni tu ani border test, ani npx gate,
        ani amplitudovy sanity test; spoléha se na to, ze
        runTrackProcessing.m:406-407 uz zahodila tracky u okraje.
      * maskovani sousedu JE aktivni: labels = bwlabel masky MASTER detekci
        pro dany snimek (runTrackProcessing.m:485-489).
      * pocatecni A a c NEjsou z mezikruzi, ale z linearni interpolace podel
        tracku (runTrackProcessing.m:457-458) resp. z prvniho/posledniho
        snimku tracku.

    Pozn.: buffer snimky zadnou master detekci nemaji, takze do signatury
    f(video, frame, y, x) z principu nepatri. Tato funkce je tu pro uplnost.

    labels : 2D int array nebo None
        Oznacene souvisle komponenty masky master detekci pro dany snimek.
        None => zadne maskovani (odchylka od cmeAnalysis; sousedni CCP
        do ~4*sigma pak nadhodnoti A).
    """
    img = np.asarray(video[frame], dtype=np.float64)
    w2 = int(np.ceil(2.0 * float(sigma_master)))
    w4 = int(np.ceil(4.0 * float(sigma_master)))

    # i_range zamerne obrovske: interpTrack zadny amplitudovy sanity test nema.
    huge = (0.0, np.inf)

    res = fit_gaussian_2d_point(img, x, y, sigma_slave, mode="xyAc",
                                A_init=A_init, c_init=c_init, labels=labels,
                                i_range=huge, alpha=alpha,
                                conf_radius=w2, window_size=w4, min_npx=0)
    if res["valid"]:
        return _interp_track_fields(res)

    # fallback 'Ac' se stejnym seedem, pozice se vraci NEZMENENA (ps.x = x,
    # nikoli xi -- runTrackProcessing.m:906-907)
    res = fit_gaussian_2d_point(img, x, y, sigma_slave, mode="Ac",
                                A_init=A_init, c_init=c_init, labels=labels,
                                i_range=huge, alpha=alpha,
                                conf_radius=w2, window_size=w4, min_npx=0)
    if res["valid"]:
        res["x"], res["y"] = float(x), float(y)
    return _interp_track_fields(res)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Minimalni smoke test: syntetická Gaussovka na konstantnim pozadi + sum.
    rng = np.random.default_rng(0)
    sig, A_true, c_true = 1.5, 120.0, 300.0
    ny, nx, T = 64, 64, 3
    vid = np.full((T, ny, nx), c_true, dtype=float)
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    y0, x0 = 31.4, 30.7
    vid[1] += A_true * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sig ** 2))
    vid += rng.normal(0.0, 4.0, vid.shape)

    r = dynamin_intensity(vid, 1, y0, x0, sigma_slave=sig, sigma_master=sig)
    print(f"A={r['A']:.2f} (true {A_true})  c={r['c']:.2f} (true {c_true})")
    print(f"A_pstd={r['A_pstd']:.3f}  sigma_r={r['sigma_r']:.3f}  "
          f"pval_Ar={r['pval_Ar']:.3g}  hval_Ar={r['hval_Ar']}  "
          f"npx={r['npx']}  mode={r['mode']}")

    # Snimek bez signalu: fit NESELZE, jen vrati A ~ 0 a hval_Ar == False.
    r0 = dynamin_intensity(vid, 0, y0, x0, sigma_slave=sig, sigma_master=sig)
    print(f"bez signalu: A={r0['A']:.2f}  hval_Ar={r0['hval_Ar']}  valid={r0['valid']}")

    # U okraje: NaN, zadny padding.
    rb = dynamin_intensity(vid, 1, 2.0, 2.0, sigma_slave=sig, sigma_master=sig)
    print(f"u okraje: valid={rb['valid']}  A={rb['A']}")
