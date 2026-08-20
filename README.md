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
(expozice, exprese, bleaching) před poolováním:

```python
from cmepython import scale_edfs
a, c, ref = scale_edfs([max_amplitudy_filmu_1, max_amplitudy_filmu_2, ...])
```

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

## Co je dobré vědět

**Intenzita je amplituda gaussovky nad lokálním pozadím, v surových
kamerových jednotkách.** Není to integrovaná intenzita ani hodnota vrcholového
pixelu. Žádná normalizace se v per-detekčním výpočtu nedělá — cmeAnalysis
nemá ani korekci offsetu kamery, ani korekci bleachingu.

**Prázdné místo nevrací chybu, ale číslo blízko nule.** Měření a rozhodnutí,
jestli tam dynamin je, jsou oddělené kroky; nevýznamnost se pozná z
`pval_Ar`, ne z `NaN`. `NaN` znamená jen, že se okno nevešlo do snímku.

**Sigma se na slave kanálu nikdy nefituje.** Módy `Ac` a `xyAc` mají šířku
pevnou. Volný fit šířky (`xyasc`) slouží výhradně kalibraci.

**Numerická shoda s MATLABem je ověřená.** `fitGaussian2D` je v cmeAnalysis
zkompilovaná binárka bez zdrojáku (Levenberg–Marquardt z GSL), takže je
rekonstruovaná podle chování, ne přeložená. Shoda je ale změřená přímo proti
té binárce na 874 oknech z reálných dat — medián relativního rozdílu vychází
10⁻⁸ až 10⁻¹⁶ pro všechny vracené veličiny v obou módech fitu:

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
