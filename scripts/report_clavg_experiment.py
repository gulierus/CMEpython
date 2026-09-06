#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protokol klatrinoveho experimentu podle zadani kolegu:
1) nasamplovana klatrinova intenzita (cmeAnalysis + box-mean),
2) prahova analyza samotne intenzity + replikace klasifikacniho
   experimentu se stitkem z dynaminove klasifikace.
Hlavni otazka: lze dynaminovou produktivitu predikovat jen z klatrinu?

Cisla prevzata z overenych behu (sample_clathrin_avg, clavg_threshold_analysis,
Helios joby 238740/238741); zde se nic nepocita znovu.

    python3 scripts/report_clavg_experiment.py
"""
from __future__ import annotations

import datetime as _dt
import os
import shutil
import subprocess
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_shape_story import md_to_tex  # noqa: E402  (prevod md -> tex)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Clathrin Analysis", "report")
ORANGE, TEAL, GREY, PURPLE, DARK = "#d95f02", "#1b9e77", "0.45", "#7570b3", "#333333"

RUN_AMP = os.path.join(ROOT, "CME_for_Helios", "runs", "clavg_amp_238740")
RUN_BOX = os.path.join(ROOT, "CME_for_Helios", "runs", "clavg_box_238741")

WB_BARS = [
    ("jas: nejlepší jedno číslo", 0.543, GREY),
    ("trénovaný model, box-mean", 0.595, TEAL),
    ("trénovaný model, amplituda", 0.657, ORANGE),
]
LEN_WB = 0.596   # baseline "jen delka" ve stejnych pasmech
LEN_POOLED = 0.810


def fig_summary(path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(WB_BARS))
    ax.bar(x, [v for _, v, _ in WB_BARS], 0.55,
           color=[c for _, _, c in WB_BARS])
    for xi, (_, v, _) in zip(x, WB_BARS):
        ax.text(xi, v + 0.005, f"{v:.3f}".replace("0.", "0,"), ha="center", fontsize=9)
    ax.axhline(0.5, color="0.6", lw=1, ls=":")
    ax.axhline(LEN_WB, color=PURPLE, lw=1.2, ls="--")
    ax.text(2.35, LEN_WB + 0.005, "jen délka života (0,596)", fontsize=8.5,
            color=PURPLE, ha="right",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    ax.set_xticks(x, [n for n, _, _ in WB_BARS], fontsize=9)
    ax.set_ylim(0.45, 0.70)
    ax.set_ylabel("AUC uvnitř délkových skupin")
    ax.set_title("Kolik dynaminové informace nese klatrin (po očištění o délku)",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


MD = """# Klatrinový experiment: predikce dynaminové produktivity z klatrinu

Datum {date} · CMEpython {git} · vygenerováno `scripts/report_clavg_experiment.py`
· zdrojové běhy: `sample_clathrin_avg.py`, `clavg_threshold_analysis.py`, Helios joby 238740/238741

## 1. Zadání a hlavní otázka

Podle zadání kolegů: (1) nasamplovat klatrinovou intenzitu pro stejné trajektorie
jamek metodou cmeAnalysis a pro srovnání i jednodušší metodou (průměr 5×5 s kompenzací
blednutí) a uložit jako samostatný výstup; (2) zopakovat klasifikační experiment,
kde vstupem je nově změřená klatrinová intenzita a referenčním štítkem dynaminová
klasifikace produktivity, včetně zjištění, jak dobře rozlišuje samotná intenzita
a jaký práh funguje nejlépe; rozhodovací práh volit Youdenovým J. Hlavní otázka:
**lze produktivitu určenou z dynaminu predikovat také jen z klatrinové intenzity, nebo dynamin nese informaci, kterou klatrin nezachytí?**

Odpověď dopředu: **dynamin nahradit klatrinem nejde.** Samotné klatrinové číslo
nerozliší nic, trénovaný model jen málo, a to ještě pouze s lepším ze dvou odečtů.

## 2. Data a štítek

Nová data: jednokanálové klatrinové filmy (jeden snímek = průměr devíti surových
SIM snímků), intenzity bez přeškálování. Ověřili jsme po pixelech, že starý
dvoukanálový dataset je z těchto dat vyroben lineárním roztažením po filmech
(faktory 43 až 149, shoda na zaokrouhlení); souřadnice dosavadních trajektorií
proto sedí beze změny. Nové jednotky jsou poprvé srovnatelné mezi filmy.

Štítek: dynamin+ / dynamin− podle výchozí klasifikace cmeAnalysis (test amplitudy
proti vlastnímu okolí na každém snímku + binomické pravidlo; opravené masky).
Z 33 750 drah je dynamin+ 56,8 %.

Jedno varování předem: podíl dynamin+ roste s délkou života jamky z 24 % na 95 %.
Smíchaná (pooled) čísla proto vycházejí vysoko u čehokoli, co s délkou souvisí.
Hodnotíme uvnitř čtyř délkových skupin a vždy srovnáváme s řádkem „jen délka";
ten má smíchaně 0,810 a uvnitř skupin 0,596 (skupiny jsou hrubé, zbytek délky
v nich zůstává). Model nese dynaminovou informaci teprve tehdy, když porazí 0,596.

## 3. Úkol 1: nasamplovaná klatrinová intenzita

Pro všech 702 630 snímků 33 750 drah v 15 filmech jsme spočítali obě metody:

- **Metoda A (cmeAnalysis):** amplituda 2D fitu tečky nad vlastním okolím,
   s nejistotou a testem významnosti na každém snímku. Šířka tečky kalibrována
   přímo na nových datech: 1,6423 px (stará klatrinová kalibrace 1,6424).
- **Metoda B (box-mean):** průměr 5×5 pixelů na pozici jamky, vydělený
   dvojexponenciálním fitem průměrů snímků (kompenzace blednutí). Fit
   zkonvergoval na všech 15 filmech.

Kontroly: zarovnání souřadnic na nová data má shodu R² = 1,00000 v každém filmu;
0 neplatných fitů. Obě metody spolu souvisejí, ale nejsou totéž (pořadová korelace
0,77): metoda A měří tečku nad okolím, metoda B sbírá i rozptýlený signál v okénku.

Výstup je uložen jako samostatný balíček `Clathrin Analysis/sampled/`
(15 CSV + `manifest.json` + README s popisem sloupců a jednotek) a je připraven
ke sdílení bez ohledu na výsledek klasifikace.

## 4. Úkol 2a: jak dobře rozlišuje samotná intenzita a jaký práh je nejlepší

Každou dráhu jsme zhustili do jednoho čísla (maximum nebo průměr přes život,
pro obě metody) a hledali nejlepší práh Youdenovým J proti dynaminovému štítku:

| skóre dráhy | smíchaně | uvnitř skupin | nejlepší práh | sens | spec |
|---|---|---|---|---|---|
| box-mean, maximum | 0,548 | 0,495 | 2,07 | 0,33 | 0,75 |
| box-mean, průměr | 0,533 | 0,509 | 1,98 | 0,21 | 0,85 |
| amplituda, maximum | 0,563 | 0,504 | 312 | 0,33 | 0,79 |
| amplituda, průměr | 0,574 | 0,543 | 158 | 0,44 | 0,69 |
| **jen délka života** | **0,810** | **0,596** | 14 snímků | 0,74 | 0,75 |

**Diskuze:** žádný klatrinový práh nefunguje. Všechna intenzitní čísla jsou hluboko
pod řádkem „jen délka" a po očištění o délku leží na minci (0,50 až 0,54).
Paradoxně nejlepší jednoduché pravidlo pro dynamin+ vůbec není o klatrinu:
„jamka žije aspoň 14 snímků" trefí sens 0,74 a spec 0,75. Jasnější jamka
neznamená jamku s dynaminem.

## 5. Úkol 2b: replikace klasifikačního experimentu

Trénovací kód kolegů jsme spustili beze změny řádku. Korpusy nesou klatrinovou
intenzitu (dvě varianty: amplituda a box-mean) a dynaminový štítek jsme vložili
formátovým trikem: sloupec `cls` obsahuje 1,0 pro dynamin+ a 0,0 pro dynamin−,
takže pravidlo „max(cls) > 0,7" vyrobí přesně dynaminový štítek. Stejné dělení
po filmech, stejné modely, práh Youdenovým J.

| model | smíchaně (amp / box) | uvnitř skupin (amp / box) |
|---|---|---|
| XGBoost | 0,802 / 0,783 | **0,657** / 0,595 |
| logistická regrese | 0,796 / 0,769 | 0,640 / 0,572 |
| síť (MLP) | 0,805 / 0,794 | 0,637 / 0,606 |
| jen výška vrcholu (bez učení) | 0,562 / 0,548 | 0,504 / 0,495 |
| zamíchané štítky (kontrola) | 0,542 / 0,524 | 0,519 / 0,507 |

U logistické regrese s Youdenovým prahem (amplituda: práh 0,503) vychází
sens 0,79 a spec 0,70; u box-mean (práh 0,497) sens 0,77 a spec 0,70.
Confusion matice všech modelů jsou na obrázku 1.

![mozaika](fig1_mosaic.png)

*Obrázek 1: Confusion matice všech modelů pro amplitudovou variantu (korpus bez
filtru, end-observed). Každý panel je jeden model: řádky skutečný štítek
(dynamin+ / dynamin−), sloupce verdikt modelu při Youdenově prahu. Čteme:
levý horní roh = správně zachycení dynamin+, pravý dolní = správně odmítnutí
dynamin−. Upozornění: práh se volí na týchž datech (in-sample), sens a spec
jsou proto horní odhady; a smíchaná čísla nafukuje délka.*

![souhrn](fig2_souhrn.png)

*Obrázek 2: Hlavní výsledek po očištění o délku. Šedý sloupec je nejlepší
jediné klatrinové číslo, zelený a oranžový trénované modely na dvou odečtech.
Čárkovaná fialová čára je laťka „jen délka života" (0,596), tečkovaná je mince.
Čteme: box-mean na laťku pouze dosáhne, amplituda ji přerůstá o 0,06.*

**Diskuze:** smíchaná čísla kolem 0,80 vypadají dobře, ale jsou to délková čísla
(řádek „jen délka" má 0,810 sám o sobě). Poctivé čtení je uvnitř skupin proti
laťce 0,596:
**box-mean varianta na laťku jen dosáhne (0,595), amplitudová ji přerůstá o 0,06 (0,657).**
V časovém průběhu klatrinu tedy nějaká dynaminová
informace je, ale malá, a s jednodušším odečtem se ztrácí úplně. Kontroly sedí:
zamíchané štítky dávají minci, jediné netrénované číslo také.

## 6. Odpověď na hlavní otázku

| směr predikce | uvnitř skupin (AUC) |
|---|---|
| klatrin → tvarový štítek (SI); dřívější kontrola | 0,775 |
| klatrin → dynamin (tento experiment, amplituda) | 0,657 |
| klatrin → dynamin (tento experiment, box-mean) | 0,595 |
| dynamin → SI (dřívější experimenty) | 0,585 |
| jedno klatrinové číslo → dynamin | 0,50 |

**Diskuze:** klatrin výborně vypovídá o tvaru vlastní struktury, ale o dynaminu
ví jen málo — a naopak. Každý kanál nese především svou vlastní informaci.
**Dynamin proto nelze nahradit klatrinovou intenzitou**; dynaminové měření
přináší informaci, kterou klatrinový signál nezachytí. To je odpověď na hlavní
otázku zadání.

## 7. Meze

Za prvé, Youdenův práh se volí na týchž datech, na kterých se hodnotí; sens
a spec jsou horní odhady. Za druhé, délkové skupiny jsou hrubé a zbytek délky
v nich zůstává; proto je laťkou 0,596, ne 0,5, a malé překročení laťky u box-mean
nelze číst jako signál. Za třetí, dynaminový štítek je naše výchozí klasifikace
cmeAnalysis; je dokumentovaná a reprodukovatelná, ale je to štítek, ne absolutní
pravda. Za čtvrté, jde o 15 filmů jedné buněčné linie a jedny podmínky snímání.

## 8. Shrnutí

1. Nasamplovaná klatrinová intenzita pro všechny dosavadní trajektorie je uložena
   a připravena ke sdílení (obě metody, kontroly zarovnání R² = 1, šířka tečky
   1,6423 px, blednutí zkonvergovalo všude).
2. Samotná klatrinová intenzita dynaminovou produktivitu nerozliší (uvnitř skupin
   0,50 až 0,54); žádný práh nefunguje, nejlepší jednoduché pravidlo je délka
   života ≥ 14 snímků.
3. Trénovaný model: smíchaně ~0,80 = délka; uvnitř skupin box-mean 0,595 (= laťka
   délky), amplituda 0,657 (+0,06 skutečné informace). Volba odečtu rozhoduje.
4. Odpověď na hlavní otázku: dynamin nahradit klatrinem nejde; dynamin nese
   informaci, kterou klatrinový signál nezachytí.
"""


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    shutil.copy(os.path.join(RUN_AMP, "dynamin_v2_confusion_none_endobs.png"),
                os.path.join(OUT, "fig1_mosaic.png"))
    fig_summary(os.path.join(OUT, "fig2_souhrn.png"))
    git = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip() or "?"
    md = MD.format(date=_dt.date.today().isoformat(), git=git)
    base = os.path.join(OUT, "protokol_klatrin_dynamin")
    with open(base + ".md", "w", encoding="utf-8") as fh:
        fh.write(md)
    with open(base + ".tex", "w", encoding="utf-8") as fh:
        fh.write(md_to_tex(md))
    if shutil.which("latexmk"):
        r = subprocess.run(["latexmk", "-xelatex", "-interaction=nonstopmode",
                            "-halt-on-error", "-cd", base + ".tex"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            subprocess.run(["latexmk", "-c", "-cd", base + ".tex"],
                           capture_output=True, text=True)
            print("PDF: protokol_klatrin_dynamin.pdf")
        else:
            print("VAROVANI: xelatex selhal; konec logu:")
            print("\n".join(r.stdout.splitlines()[-12:]))
    print(f"hotovo -> {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
