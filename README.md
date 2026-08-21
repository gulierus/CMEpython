# CMEpython

Python port měření intenzity dynaminu z [cmeAnalysis](https://github.com/DanuserLab/cmeAnalysis)
(DanuserLab, MATLAB).

Pro zadané souřadnice `[frame, y, x]` a TIRF video vrátí intenzitu dynaminu
spočtenou stejným postupem jako cmeAnalysis. Čistě numpy/scipy — žádná
kompilovaná rozšíření, žádný MATLAB, běží na Windows, Linuxu i obou
generacích Maců.

## Instalace

```bash
pip install -e .
```

Závislosti: `numpy`, `scipy`, `tifffile`. Pro testy `pip install -e ".[dev]"`.

## Rychlý start

```python
from cmepython import dynamin_intensity

r = dynamin_intensity(video, frame, y, x,
                      sigma_slave=2.6424,     # šířka PSF měřicího kanálu
                      sigma_master=1.4187)    # šířka PSF kanálu s detekcemi

r["A"]        # amplituda gaussovky nad lokálním pozadím — hledaná intenzita
r["c"]        # fitované lokální pozadí
r["A_pstd"]   # nejistota amplitudy
r["pval_Ar"]  # p-hodnota testu proti šumu
r["hval_Ar"]  # bool: signál je významný
```

Pro víc než pár bodů použij dávkovou variantu — čte snímky lazy z disku
a volitelně paralelizuje:

```python
from cmepython import measure_movie
import numpy as np

coords = np.array([[frame, y, x], ...])       # (N, 3)
res = measure_movie("film.tif", coords,
                    sigma_slave=2.6424, sigma_master=1.4187,
                    slave_channel=2, workers=0)   # 0 = všechna jádra
res["A"]      # pole intenzit, jeden prvek na řádek coords
```

> Pod `spawn` (macOS, Windows) musí být volání paralelní varianty ve skriptu
> s `if __name__ == "__main__":`. Z interaktivní session to spadne na
> serialní běh s varováním, ne s chybou.

## Celý postup

```
1. Kalibrace šířky PSF     scripts/calibrate_dataset.py
2. Měření na souřadnicích  scripts/measure_trajectories.py
3. Srovnání napříč filmy   cmepython.scale_edfs
```

**1. Šířka PSF** se odhaduje z dat, ne z optiky — stejně jako cmeAnalysis ve
výchozím nastavení. Vzorkuje ~40 snímků rozprostřených přes **všechny filmy
podmínky**, ne z jednoho filmu:

```bash
python3 scripts/calibrate_dataset.py "reconstructed registered"
```

**2. Měření** proběhne na souřadnicích z CSV s trajektoriemi; k původním
sloupcům přibudou hodnoty s prefixem `dnm_`:

```bash
python3 scripts/measure_trajectories.py
```

**3. Škálování napříč filmy** srovná systematické rozdíly mezi filmy
před poolováním:

```python
from cmepython import scale_edfs
a, c, ref = scale_edfs([max_amplitudy_filmu_1, max_amplitudy_filmu_2, ...])
```

> **Rozmyslete si, jestli ho chcete.** Na tomto datasetu je rozptyl mezi
> filmy převážně biologický, ne technický: dynaminové amplitudy kolísají
> 17,2× (CV 77 %), zatímco pozadí téhož kanálu jen 2,3× (CV ~30 %) a
> clathrin ~1,5–2,2× (CV 12–16 % podle snímku); korelace dynaminu s
> clathrinem je s n = 15 nerozlišitelná od nuly. Akviziční složka do ~2×
> se vyloučit nedá, ale 17× rozdíl nevysvětlí — u siRNA knockdownu se
> účinnost liší buňka od buňky. Škálování by tedy smazalo právě ten efekt,
> který se měří. Porovnejte výsledky s ním i bez něj.

## Co přesně reprodukuje

| Modul | Originál v cmeAnalysis |
|---|---|
| `slave_intensity.dynamin_intensity` | `runDetection.m:180-190` + `fitGaussians2D.m` |
| `slave_intensity.dynamin_intensity_gap` | `runTrackProcessing.m:879-926` (`interpTrack`) |
| `psf_calibration.estimate_psf_sigma` | `getGaussianPSFsigmaFromData.m` |
| `psf_calibration.apply_sigma_clamp` | `runDetection.m:90-92` |
| `edf_scaling.scale_edfs` | `scaleEDFs.m` |
| `measure.*` | smyčka přes detekce + `readtiff` |

Odkazy `soubor.m:řádek` v komentářích míří do `cmeAnalysis/software/`.
Upstream si naklonuj zvlášť, v repu není:

```bash
git clone https://github.com/DanuserLab/cmeAnalysis.git
```

## Data tohoto projektu

Filmy ve složce `reconstructed registered/` jsou **TIRF-SIM**: ch0 = clathrin
(SIM rekonstrukce, červený), ch1 = dynamin (SIM rekonstrukce, zelený),
**ch2 = zprůměrovaný raw dynamin TIRF, 2× upsamplovaný a registrovaný na
clathrin — na něm se měří**. Pixel: 79,1 nm v raw rozlišení, tedy
**39,55 nm/px** v gridu těchto souborů; NA = 1,5. Červený a zelený kanál
jdou jinou optickou dráhou (jiná PSF, jiná frekvence SIM patternu).

Naměřené σ tomu odpovídají: ch2 má FWHM ~246 nm proti difrakčnímu limitu
~172 nm (1,4× širší — průměrování 9 SIM snímků + interpolace při
upsamplingu; proto je správně σ odhadovat z dat, ne z optiky, což je
i výchozí chování cmeAnalysis). ch0/ch1 mají FWHM 132/82 nm, tedy pod
difrakčním limitem, jak má SIM rekonstrukce — **PSF model cmeAnalysis na
ně nepatří** a měřit se na nich nemá.

Interval mezi snímky zůstává nedohledaný (odhad z rozdělení životností:
~1,5–3 s).

## Co je dobré vědět

**Intenzita je amplituda gaussovky nad lokálním pozadím, v surových
kamerových jednotkách.** Není to integrovaná intenzita ani hodnota vrcholového
pixelu. Žádná normalizace se v per-detekčním výpočtu nedělá — cmeAnalysis
nemá ani korekci offsetu kamery, ani korekci bleachingu.

**Prázdné místo nevrací chybu, ale číslo blízko nule.** Měření a rozhodnutí,
jestli tam dynamin je, jsou oddělené kroky; nevýznamnost se pozná z
`pval_Ar`, ne z `NaN`. `NaN` znamená jen, že se okno nevešlo do snímku.

**Photobleaching je reálný, ale menší, než se zdá.** Po vyloučení drah
cenzurovaných na obou koncích filmu a kontrole délky klesá vrcholová
amplituda dynaminu postupně: ~12–17 % na 100 snímků pro dráhy ≥ 15 snímků,
~30 % pro krátké (10–15 snímků); pokles není soustředěný do začátku filmu.
Pozadí klesá typicky ~5 % (nejhorší film −9 %), celý snímek ~4 % — vyhasíná
vázaná frakce, volný pool se doplňuje difuzí. Původní odhady −44 % až −62 %
nafukovala hlavně levá cenzura: dráhy začínající ve snímku 0 jsou fragmenty
už existujících jasných struktur (medián max A ~8300 proti ~2800 u skutečných
zrodů). cmeAnalysis korekci nemá a tento port ji také nezavádí: spolehlivější
je zahrnout čas vzniku dráhy jako kovariátu než hodnoty upravovat a riskovat,
že se s bleachingem odečte i biologie.

**Test významnosti je konzervativní, ne liberální.** Netestuje „je tam
signál", ale „je tam signál silnější než 1,96 σ šumu". Na skutečně prázdném
pozadí vychází ~0,1–0,3 % falešně pozitivních při nominální hladině 5 %
(podle snímku a konstrukce nulové sady)
(horní mez ~1 %, pokud se do „prázdných" pozic připustí slabé zdroje pod
detekčním prahem — právě ty tvoří většinu zdánlivých falešných pozitiv).
Pro slabý dynamin je to záměrně přísné.

Související: měřicí kanál tohoto datasetu je 2× zvětšený, takže sousední
pixely nemají nezávislý šum (1D integrál autokorelace τ ≈ 2,8; 2D ≈ 20).
Korekce na efektivní počet pixelů nepřehodí na prázdných pozicích žádné
rozhodnutí a na signálních méně než 1 % (výhradně hraniční pozitiva) —
statistiku ovládá odečet 1,96 σ, ne jmenovatel. Reprodukce:
`scripts/dataset_findings.py`.

**Sigma se na slave kanálu nikdy nefituje.** Módy `Ac` a `xyAc` mají šířku
pevnou. Volný fit šířky (`xyasc`) slouží výhradně kalibraci.

**Numerická shoda s MATLABem je ověřená.** `fitGaussian2D` je v cmeAnalysis
zkompilovaná binárka bez zdrojáku (Levenberg–Marquardt z GSL), takže je
rekonstruovaná podle chování, ne přeložená. Shoda je ale změřená přímo proti
té binárce na 874 oknech z reálných dat: amplitudy, pozadí a rezidua sedí
na medián relativního rozdílu 10⁻⁸ až 10⁻¹⁶; lokalizace (dx, dy) na ~5×10⁻⁷
(max ~10⁻³ — poloha je nejhůř podmíněný parametr nelineárního fitu).
Srovnání módu `xyAc` pokrývá 567 z 874 oken; zbytek vyřadil akceptační test
(`fitGaussians2D.m:198`), který port zrcadlí:

```bash
matlab -batch export_fits                      # matlab/export_fits.m
python3 scripts/validate_against_matlab.py
```

Validace odhalila jednu skutečnou odchylku, která je teď opravená: MEX počítá
stupně volnosti jako `npx - n_free - 1`, tedy o jeden méně než běžná konvence.
Bez opravy byly `A_pstd` a všechny odvozené p-hodnoty systematicky vyšší
o 0,1 %.

## Testy

```bash
python3 -m pytest tests/ -q
```

Pro validaci proti MATLABu jsou potřeba toolboxy **Image Processing**
a **Statistics and Machine Learning**. Pokud je licence zahrnuje, ale nejsou
nainstalované, doinstalují se bez GUI:

```bash
curl -sL -o mpm https://www.mathworks.com/mpm/maca64/mpm && chmod +x mpm
./mpm install --destination=/Applications/MATLAB_R2026a.app --release=R2026a \
    --products Image_Processing_Toolbox Statistics_and_Machine_Learning_Toolbox
```

Testy ověřují chování odvozené ze zdrojáku cmeAnalysis a vnitřní konzistenci
(dávka == bod po bodu). Numerickou shodu s MATLABem ověřuje zvlášť
`scripts/validate_against_matlab.py` — vyžaduje nainstalovaný MATLAB.

## Licence

GPL-3.0-or-later. cmeAnalysis je GPL-3.0, takže odvozený port musí být také.
