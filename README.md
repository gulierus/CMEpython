# CMEpython

Python port měření intenzity slave kanálu z balíku
[cmeAnalysis](https://github.com/DanuserLab/cmeAnalysis) (DanuserLab, MATLAB),
vyvinutý pro měření dynaminu na drahách klatrinových jamek.

Pro zadané souřadnice `[snímek, y, x]` a TIRF video vrátí intenzitu dynaminu
spočtenou stejným postupem jako cmeAnalysis: fit skvrnky tvaru PSF s pevnou
šířkou, výsledkem je amplituda nad lokálním pozadím, její nejistota
a p-hodnota. Čistě numpy/scipy — žádná kompilovaná rozšíření, žádný MATLAB;
běží na Windows, Linuxu i obou generacích Maců. Numerická shoda s originální
zkompilovanou binárkou je změřená (viz [Validace](#validace-proti-matlabu)).

**Obsah:** [Instalace](#instalace) ·
[Rychlý start](#rychlý-start) ·
[Formát vstupů](#formát-vstupních-dat) ·
[Pipeline krok za krokem](#pipeline-krok-za-krokem) ·
[Přehled skriptů](#přehled-skriptů) ·
[Jak port vznikal](#jak-port-vznikal-a-čím-se-liší-od-originálu) ·
[Dokumentace měření](#dokumentace-měření) ·
[Poznámky k našim datasetům](#poznámky-k-našim-datasetům) ·
[Co v repozitáři není](#co-v-repozitáři-záměrně-není) ·
[Testy](#testy) · [Licence](#licence)

## Instalace

```bash
git clone https://github.com/gulierus/CMEpython.git
cd CMEpython
pip install -e .
```

Jádro (`cmepython/`) potřebuje jen `numpy`, `scipy`, `tifffile`.
Analytické skripty (`scripts/compare_*`, `scripts/report_*`, kohorty) navíc
`pandas` a `matplotlib`:

```bash
pip install pandas matplotlib
```

PDF verze protokolů vyžaduje LaTeX (`latexmk` + `xelatex`); bez něj skripty
doběhnou s `--skip-pdf` nebo vypíší varování a nechají markdown a `.tex`.
Testy: `pip install -e ".[dev]"` a `pytest`.

## Rychlý start

Jeden bod:

```python
from cmepython import dynamin_intensity

r = dynamin_intensity(video, frame, y, x,
                      sigma_slave=1.4356,     # šířka PSF měřicího kanálu [px]
                      sigma_master=1.6424)    # šířka PSF kanálu s detekcemi [px]

r["A"]        # amplituda gaussovky nad lokálním pozadím — hledaná intenzita
r["c"]        # fitované lokální pozadí
r["A_pstd"]   # nejistota amplitudy
r["pval_Ar"]  # p-hodnota testu proti šumu
r["hval_Ar"]  # bool: signál je významný
```

Víc bodů najednou — čte snímky lazy z disku a volitelně paralelizuje:

```python
from cmepython import measure_movie
import numpy as np

coords = np.array([[frame, y, x], ...])       # (N, 3)
res = measure_movie("film.tif", coords,
                    sigma_slave=1.4356, sigma_master=1.6424,
                    slave_channel=1, master_channel=0,
                    workers=0)                # 0 = všechna jádra, None/1 = sériově
res["A"]      # pole intenzit, jeden prvek na řádek coords
```

> Pod `spawn` (macOS, Windows) musí být paralelní volání ve skriptu
> s `if __name__ == "__main__":`. Z interaktivní session spadne na sériový
> běh s varováním, ne s chybou.

## Formát vstupních dat

**Filmy:** vícestránkový TIFF. Osy se čtou z metadat; podporované rozložení
TCYX, ZCYX, CYX, ZYX i YX (`cmepython.measure.movie_layout`). Kanály se
adresují indexem (`slave_channel` = měřený kanál, `master_channel` = kanál,
ve kterém vznikly detekce).

**Trajektorie:** CSV s minimálně sloupci `particle` (id dráhy), `frame`
(index snímku, 0-based), `x`, `y` (pixely; `x` = sloupec, `y` = řádek,
stejná konvence jako v obrazových polích). Volitelné sloupce (`cls` = shape
index, `cluster`, …) projdou do výstupu beze změny.

**Párování film ↔ CSV:** skripty párují soubory podle čísla filmu v názvu
(regulární výraz `_(\d+)[-_]`), např. `..._16_LRR.tif` ↔
`..._16-lr-trajectories.csv`.

## Pipeline krok za krokem

Celý postup od syrových filmů k výsledkům. Každý skript má `--help`
se všemi volbami.

### 1. Kalibrace šířky PSF

Šířka σ se odhaduje z dat, ne z optiky — stejně jako výchozí chování
cmeAnalysis (`getGaussianPSFsigmaFromData.m`): detekce s pevnou σ, volný fit
`xyasc`, filtrace neúspěšných fitů, směs gaussovek 1–3 komponent podle BIC,
vybere se komponenta s nejvyšším vrcholem. Vzorkuje ~40 snímků napříč
**všemi filmy podmínky** (jedna σ na kanál na podmínku, jako cmeAnalysis):

```bash
python3 scripts/calibrate_dataset.py "lr registered" --out psf_calibration_lr.json
```

Výstupní JSON obsahuje σ pro každý kanál včetně diagnostiky (BIC, komponenty,
počty spotů). cmeAnalysis hodnoty pod 1,1 px zvedá na 1,1 (`runDetection.m:90`);
port to dělá také a hlásí to.

### 2. Měření intenzity na drahách

```bash
python3 scripts/measure_trajectories.py \
    --traj-dir "lr registered/trajectories" --movie-dir "lr registered" \
    --out-dir "lr registered/measured" \
    --sigma-slave 1.4356 --sigma-master 1.6424 \
    --slave-channel 1 --master-channel 0 --also-master --workers 0
```

Ke sloupcům vstupního CSV přibudou (prefix `dnm_` = slave/dynamin,
`clc_` = master/clathrin při `--also-master`):

| sloupec | význam |
|---|---|
| `dnm_A`, `dnm_c` | amplituda nad lokálním pozadím a pozadí [jednotky kamery] |
| `dnm_A_pstd` | nejistota amplitudy (šířená z fitu, stupně volnosti jako MEX) |
| `dnm_sigma_r` | směrodatná odchylka reziduí fitu |
| `dnm_pval`, `dnm_signif` | p-hodnota a verdikt testu signálu proti šumu (α = 0,05) |
| `dnm_x`, `dnm_y` | doladěná poloha (volný fit přijat při posunu < 3σ_master a vyšší A) |
| `dnm_valid` | fit uvnitř obrazu (`False` jen u okna přes okraj) |
| `track_len` | délka dráhy ve snímcích |

Prázdné místo **není chyba**: vrací A ≈ 0 a p ≈ 1. Měření a rozhodnutí
o pozitivitě jsou oddělené kroky.

Před měřením proběhnou **vstupní kontroly** (`validate=True`): detekce
SIM pruhů ve spektru (na jednotlivém raw TIRF-SIM snímku fit tiše lže;
řešení je zprůměrovat devítice) a kontrola registrace kanálů FFT křížovou
korelací (posun nad 1 px varuje). Nálezy jsou varování, ne chyby.

### 3. Klasifikace dynamin-pozitivních drah

Port Aguetovy statistické klasifikace (`runSlaveChannelClassification.m`).
Dva testy na dráhu, oba binomicky korigované na její délku: `significant_master`
(počet významných detekcí proti náhodě dané `p_detection` filmu)
a `significant_slave` (počet snímků s amplitudou nad 95. percentilem pozadí
filmu, t-test; výchozí režim „s" navazujících nástrojů cmeAnalysis).

```bash
python3 scripts/classify_trajectories.py \
    --measured "lr registered/measured" --movies "lr registered" \
    --masks "lr registered/masks" \
    --sigma-slave 1.4356 --slave-channel 1 --master-channel 0
# citlivost testu: --alpha 0.01   jiny vystup: --out-suffix "-classified-a0.01.csv"
```

Výstup na dráhu: `particle, track_len, n_detected, thr_master,
significant_master, n_above_bg, thr_slave, significant_slave, max_A,
max_master_A, max_si`. Pozadí filmu se počítá z míst uvnitř buněčné masky
dál než 4σ od všech detekcí; masky lze dodat (`--masks`), jinak se odhadnou
z maximální projekce.

> Výchozí cesty skriptu míří na první dataset; pro jiný **vždy** zadejte
> `--measured/--movies/--masks/--sigma-slave/--slave-channel/--master-channel`.

### 4. Průběhy intenzity podle lifetime kohort

Port `getIntensityCohorts.m` + statistik z `plotIntensityCohorts.m`: dráhy
se rozdělí podle životnosti (meze 10/20/40/60/80/100/120 s), převzorbují
kubicky na střední délku kohorty včetně 5 bufferových snímků před vznikem
a po zániku (měřeno gap cestou portu `interpTrack`), a průměrují — nejdřív
v každém filmu, pak přes filmy (SEM přes filmy, jako cmeAnalysis):

```bash
python3 scripts/cohort_analysis.py \
    --measured "lr registered/measured" --movies "lr registered" \
    --framerate 2 --sigma-slave 1.4356 --sigma-master 1.6424 \
    --slave-channel 1 --master-channel 0 --si-col cls --out out/lr_cohorts
```

Výstup: `cohort_curves.csv` a dva grafy (kohorty; produktivní vs. abortivní
podle `max SI > 0,7`).

### 5. Porovnání klasifikací a protokol (úloha B)

Tři skripty na sebe navazují:

```bash
python3 scripts/compare_classifications.py          # confusion matrix + JSON
python3 scripts/compute_boxmean.py                  # box-mean 5×5 readout na týchž drahách
python3 scripts/report_classification_comparison.py # kompletní protokol (md + tex + pdf)
```

Protokol obsahuje mozaiku confusion matic po délkových pásmech, kompletní
metrikovou tabulku s 95% intervaly bootstrapem po filmech, length-adjusted
OR (Mantel–Haenszel), within-band AUC, sweep přes α, Hodgesovy–Lehmannovy
posuny a van Elterenův test. Sweep přes α vyžaduje předpočítané klasifikace
(`--alpha A --out-suffix=-classified-aA.csv`, viz krok 3).

### 6. Korpusy pro detektor dynaminové pozitivity (úloha A)

`scripts/build_cme_corpus.py` převede naměřené dráhy do formátu korpusu
Shape2Fate (dvě varianty: surová amplituda a amplituda dělená biexponenciálním
fitem průměrů snímků buňky; konvence `intenzita = 1 + amplituda`, práh
`thresholds_q90.json`). Trénink samotný běží kódem spolupracujícího projektu
(větev `release/dynamin-confusion-v1` repozitáře Shape2Fate_Fake2Emulate)
na výpočetním clusteru; srovnávací protokol generuje
`scripts/report_detector_comparison.py`.

### 7. Validace proti MATLABu

```bash
matlab -batch export_fits                      # matlab/export_fits.m (jen jednou)
python3 scripts/validate_against_matlab.py
```

Vyžaduje MATLAB s toolboxy Image Processing a Statistics; referenční výstup
MEX binárky pro 874 oken je ale commitnutý (`matlab/mex_reference.mat`),
takže samotné srovnání běží i bez MATLABu.

## Přehled skriptů

| skript | účel |
|---|---|
| `calibrate_dataset.py` | odhad σ PSF z dat, celý dataset najednou |
| `measure_trajectories.py` | měření intenzity na drahách → `*-dynamin.csv` |
| `classify_trajectories.py` | Aguetova klasifikace drah → `*-classified.csv` |
| `cohort_analysis.py` | lifetime kohorty, křivky + grafy |
| `compare_classifications.py` | confusion matrix SI vs. cmeAnalysis |
| `compute_boxmean.py` | box-mean 5×5 readout na pozicích drah |
| `report_classification_comparison.py` | protokol úlohy B (md/tex/pdf) |
| `build_cme_corpus.py` | korpusy pro trénink detektoru (2 varianty) |
| `report_detector_comparison.py` | protokol úlohy A: srovnání detektorů |
| `validate_against_matlab.py` | numerické srovnání s MEX binárkou |
| `dataset_findings.py` | reprodukce datasetových nálezů (bleaching, FP rate) |
| `bench_measure.py` | benchmark měření |

## Jak port vznikal a čím se liší od originálu

Klíčová funkce cmeAnalysis `fitGaussian2D` existuje jen jako zkompilovaná
MEX binárka bez zdrojáku (Levenberg–Marquardt z GSL). Port ji proto
**rekonstruuje podle chování** a shodu měří přímo proti binárce na 874
oknech z reálných dat: amplitudy, pozadí a rezidua sedí na medián relativního
rozdílu 10⁻⁸ až 10⁻¹⁶, lokalizace na ~5×10⁻⁷ px (max ~10⁻³; poloha je nejhůř
podmíněný parametr). Akceptační test volného fitu propouští stejná okna
(567 z 874). Vyšší vrstvy (klasifikace, kohorty, kalibrace) jsou přepsané
řádek po řádku podle otevřených `.m` zdrojů; komentáře v kódu odkazují
`soubor.m:řádek`.

Poznatky, které při portování stály nejvíc času (a v kódu jsou zohledněné):

- **`padarrayXT('symmetric')` je numpy `'reflect'**, ne numpy `'symmetric'`
  — MATLAB zrcadlí bez duplikace okrajového pixelu. Záměna dá chyby ~10³
  v pásu u okraje.
- **MEX počítá stupně volnosti `npx − n_free − 1`**, o jeden méně než běžná
  konvence. Bez převzetí jsou `A_pstd` a všechny p-hodnoty o ~0,1 % vyšší.
- **`sum(pval<Alpha)/sum(cellmask)` na 2D maticích je v MATLABu maticové
  dělení** (mrdivide), ne podíl počtů. Port reprodukuje skutečně vykonávaný
  výpočet `p_detection`.
- **σ se na slave kanálu nikdy nefituje** — módy `Ac`/`xyAc` mají šířku
  pevnou; volný fit `xyasc` slouží jen kalibraci. σ < 1,1 px se zvedá na 1,1.
- **Poloha není striktně fixní**: po fitu s pevnou polohou se zkusí volný
  a přijme se při posunu < 3σ_master a vyšší amplitudě (`runDetection.m:180-190`).
- Jedna vědomá odchylka: CCS oblasti se z pozadí vylučují disky kolem pozic
  z trajektorií místo detekčních masek `dmasks.tif`, které bez běhu celé
  MATLAB pipeline neexistují.

Mapa modulů na originál:

| modul | originál v cmeAnalysis |
|---|---|
| `slave_intensity.dynamin_intensity` | `runDetection.m:180-190` + `fitGaussians2D.m` |
| `slave_intensity.dynamin_intensity_gap` | `runTrackProcessing.m:879-926` (`interpTrack`) |
| `psf_calibration.estimate_psf_sigma` | `getGaussianPSFsigmaFromData.m` |
| `classification.background_stats` / `classify_track` | `runSlaveChannelClassification.m` |
| `cohorts.cohort_curves` | `getIntensityCohorts.m` + `plotIntensityCohorts.m` |
| `background.filter_gaussian_fit_2d`, `mask_from_first_mode` | `filterGaussianFit2D.m`, maskovací větev |
| `edf_scaling.scale_edfs` | `scaleEDFs.m` |
| `validation.*` | vlastní vstupní kontroly (SIM pattern, registrace) |

Upstream si naklonuj zvlášť, v repozitáři není:

```bash
git clone https://github.com/DanuserLab/cmeAnalysis.git
```

## Dokumentace měření

Srozumitelný popis, co program s daty dělá — psaný i pro čtenáře bez
technického zázemí (vyhledávání blobu, fitování a co je výsledná intenzita,
selhání fitu, normalizace, globální statistiky):

- **[docs/how-dynamin-is-measured.md](docs/how-dynamin-is-measured.md)** — anglicky, markdown
- **[docs/jak-merime-dynamin.html](docs/jak-merime-dynamin.html)** — česky,
  stránka s diagramy (otevři v prohlížeči)

## Poznámky k našim datasetům

Hodnoty σ pro oba datasety projektu jsou commitnuté
(`psf_calibration.json`, `psf_calibration_lr.json`), aby čísla ve skriptech
nebyla magické konstanty.

**Dataset 1, `reconstructed registered/`** (TIRF-SIM, 1024², 39,55 nm/px):
ch0 = clathrin (SIM rekonstrukce), ch1 = dynamin (SIM rekonstrukce),
**ch2 = zprůměrovaný raw dynamin TIRF, 2× upsamplovaný, registrovaný — na něm
se měří** (σ 2,6424 px). SIM rekonstrukce mají FWHM pod difrakčním limitem
a PSF model na ně nepatří. Interval mezi snímky nedohledán (odhad 1,5–3 s).

**Dataset 2, `lr registered/`** (512², 79,1 nm/px, 2 s/snímek, NA 1,5):
ch0 = clathrin (σ 1,6424 px), ch1 = dynamin (σ 1,4356 px); oba kanály
registrované, dynamin je průměr 9 SIM expozic (simulovaný TIRF).

Ověřené vlastnosti měření na těchto datech (reprodukce
`scripts/dataset_findings.py`):

- **Rozptyl mezi filmy je převážně biologický** — dynaminové amplitudy
  kolísají 17×, pozadí ~2,3×, clathrin ~1,5–2×. EDF škálování
  (`scale_edfs`) by smazalo měřený efekt; používat vědomě.
- **Photobleaching**: ~12–17 % na 100 snímků pro dráhy ≥ 15 snímků; původní
  vyšší odhady nafukovala levá cenzura. Port korekci nezavádí (cmeAnalysis
  ji také nemá); vhodnější je čas vzniku dráhy jako kovariáta.
- **Test významnosti je konzervativní**: na prázdném pozadí ~0,1–0,3 %
  falešně pozitivních při nominálních 5 %. U 2× upsamplovaného kanálu není
  šum sousedních pixelů nezávislý; korekce na efektivní počet pixelů mění
  < 1 % rozhodnutí.

## Co v repozitáři záměrně není

Repozitář obsahuje jen kód a dokumentaci. Lokálně (mimo git) žijí:

| co | kde lokálně |
|---|---|
| filmy a masky | `reconstructed registered/`, `lr registered/` |
| trajektorie a všechna naměřená CSV | `lr registered/trajectories/`, `lr registered/measured/` |
| protokoly s výsledky (úlohy A i B) | `lr registered/comparison_report/`, `CME_for_Helios/report/` |
| kohortové výstupy | `lr registered/cohorts/`, `out/` |
| balík pro výpočetní cluster (korpusy + cizí kód) | `CME_for_Helios/` |
| klon cmeAnalysis a publikace | `cmeAnalysis/`, `Aguet13.pdf`, `mmc1.pdf` |
| velké validační reference | `matlab/filter_ref.mat` aj. (přegenerují se) |

## Testy

```bash
python3 -m pytest tests/ -q
```

Testy ověřují chování odvozené ze zdrojáku cmeAnalysis a vnitřní konzistenci
(dávka == bod po bodu). Numerickou shodu s MATLABem ověřuje zvlášť
`scripts/validate_against_matlab.py` (viz krok 7 pipeline).

## Licence

GPL-3.0-or-later. cmeAnalysis je GPL-3.0, odvozený port proto také.
