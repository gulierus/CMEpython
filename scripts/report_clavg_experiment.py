#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protokol klatrinoveho experimentu podle zadani kolegu (odpoved napred):
1) nasamplovana klatrinova intenzita (cmeAnalysis + box-mean),
2) prahova analyza samotne intenzity + replikace klasifikacniho
   experimentu se stitkem z dynaminove klasifikace.

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
ORANGE, TEAL, GREY, PURPLE = "#d95f02", "#1b9e77", "0.45", "#7570b3"

RUN_AMP = os.path.join(ROOT, "CME_for_Helios", "runs", "clavg_amp_238740")

WB_BARS = [
    ("jas: nejlepší jedno číslo", 0.543, GREY),
    ("trénovaný model, box-mean", 0.595, TEAL),
    ("trénovaný model, amplituda", 0.657, ORANGE),
]
LEN_WB = 0.596


def fig_summary(path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
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


MD = """# Klatrinový experiment: jde dynaminová produktivita předpovědět z klatrinu?

Datum {date} · CMEpython {git} · `scripts/report_clavg_experiment.py`

zdroje: `sample_clathrin_avg.py`, `clavg_threshold_analysis.py`, Helios joby 238740/238741

## 1. Zadání a odpověď

Zadání kolegů: (1) nasamplovat klatrinovou intenzitu pro stejné trajektorie
jamek metodou cmeAnalysis i jednodušší metodou (průměr 5×5 s kompenzací
blednutí) a uložit ke sdílení; (2) zopakovat klasifikační experiment
s klatrinovou intenzitou jako vstupem a s produktivitou (Fate) určenou
z dynaminu jako štítkem, včetně nejlepšího prahu samotné intenzity.
Hlavní otázka:
**nese dynamin informaci, kterou samotný klatrinový signál nezachytí?** Odpověď:

1. **Dynamin nahradit klatrinem nejde.** Nejlepší klatrinový model dává AUC
   po očištění o délku 0,657 proti 0,596 samotné délky života; jednodušší
   odečet nepřidává nic (0,595 až 0,606).
2. **Samotné klatrinové číslo nerozliší nic** (po očištění o délku AUC 0,50
   až 0,54); nejlepší jednoduché pravidlo pro dynamin+ je délka života
   aspoň 14 snímků, ne klatrin.
3. **Nasamplovaná data jsou uložena a připravena ke sdílení.**

## 2. Jak číst čísla

AUC je hra na dvojice: podíl dvojic dynamin+/dynamin−, ve kterých dal model
vyšší skóre té dynamin+ (0,5 = náhoda, 1,0 = vše správně). Podíl dynamin+
roste s délkou života z 24 % na 95 %; dvojice proto losujeme jen uvnitř čtyř
délkových skupin (6 až 9, 10 až 19, 20 až 39, 40 a více snímků) a výsledek
srovnáváme s řádkem **„jen délka"** (smíchaně 0,810; uvnitř skupin 0,596),
ne s 0,5. Štítek: dynamin+ podle výchozí klasifikace cmeAnalysis (test
amplitudy tečky proti místnímu pozadí na každém snímku + binomické pravidlo);
je to 56,8 % jamek a čteme ho jako produktivní (Fate). Každé skóre pochází
z modelu, který daný film při učení neviděl (5 skupin po celých filmech).

## 3. Úkol 1: nasamplovaná klatrinová intenzita

**Obě metody jsou spočítané pro všech 702 630 snímků 33 750 drah v 15 filmech a uložené jako sdílený balíček `Clathrin Analysis/sampled/`** (15 CSV,
`manifest.json`, `threshold_analysis.json`, README). Metoda A (cmeAnalysis):
amplituda 2D fitu tečky nad vlastním okolím s testem významnosti; šířka tečky
1,6423 px (stará kalibrace 1,6424). Metoda B (box-mean): průměr 5×5 pixelů
dělený dvojexponenciálním fitem průměrů snímků; fit prošel všude. Kontroly:
zarovnání R² = 1,00000 v každém filmu, 0 neplatných fitů; nová data bez
přeškálování (starý dataset = intenzity filmu krát konstanta 43 až 149).
Metody spolu korelují (pořadově 0,77), ale neměří stejnou věc.

## 4. Úkol 2a: samotná intenzita a nejlepší práh

Každou dráhu jsme zhustili do jednoho čísla a hledali práh Youdenovým J, tedy
s největším součtem sens (podíl zachycených dynamin+) a spec (podíl
odmítnutých dynamin−):

| skóre dráhy | smíchaně | uvnitř skupin | nejlepší práh | sens | spec |
|---|---|---|---|---|---|
| box-mean, maximum | 0,548 | 0,495 | 2,07 | 0,33 | 0,75 |
| box-mean, průměr | 0,533 | 0,509 | 1,98 | 0,21 | 0,85 |
| amplituda, maximum | 0,563 | 0,504 | 312 | 0,33 | 0,79 |
| amplituda, průměr | 0,574 | 0,543 | 158 | 0,44 | 0,69 |
| **jen délka života** | **0,810** | **0,596** | 14 snímků | 0,74 | 0,75 |

**Diskuze:** žádný klatrinový práh nefunguje; po očištění o délku leží
intenzitní čísla na náhodě. Nejlepší jednoduché pravidlo pro dynamin+ není
o klatrinu: „jamka žije aspoň 14 snímků" trefí sens 0,74 a spec 0,75. Prahy
box-mean jsou bezrozměrné, amplitudové v jednotkách intenzity; volí se na
stejných datech, sens a spec jsou proto horní odhady.

## 5. Úkol 2b: replikace klasifikačního experimentu

**Trénovaný model na amplitudě přerůstá řádek délky o 0,06; box-mean varianta zůstává na jeho úrovni.**
Trénovací kód kolegů běžel beze změny řádku; dynaminový štítek jsme vložili
přes sloupec `cls` (1,0/0,0), který si jejich kód čte pravidlem
„max(cls) > 0,7". Stejné dělení po filmech, práh Youdenovým J na skóre
modelu; confusion matice logistické regrese, včetně smíchaných statistik
a referenčního SI běhu vedle sebe, jsou na obrázku 2 na konci dokumentu.

| model | smíchaně (amp / box) | uvnitř skupin (amp / box) |
|---|---|---|
| XGBoost | 0,802 / 0,783 | **0,657** / 0,595 |
| logistická regrese | 0,796 / 0,769 | 0,640 / 0,572 |
| síť (MLP) | 0,805 / 0,794 | 0,637 / 0,606 |
| jen výška vrcholu (bez učení) | 0,562 / 0,548 | 0,504 / 0,495 |
| zamíchané štítky (kontrola) | 0,542 / 0,524 | 0,519 / 0,507 |

![souhrn](fig2_souhrn.png)

*Obrázek 1: Hlavní výsledek po očištění o délku. Šedý sloupec je nejlepší
jediné klatrinové číslo (0,543), čárkovaná fialová čára řádek „jen délka"
(0,596), tečkovaná náhoda (0,5).*

**Diskuze:** smíchaná čísla kolem 0,80 jsou délková (řádek „jen délka" má
0,810 sám). Uvnitř skupin box-mean zůstává na úrovni délky (0,595 až 0,606),
amplituda ji přerůstá o 0,06; volba odečtu rozhoduje. Kontroly sedí:
zamíchané štítky i netrénované číslo dávají hodnoty u náhody. Logistická
regrese s Youdenovým prahem: sens 0,79 a spec 0,70 (amplituda), sens 0,77
a spec 0,70 (box-mean).

## 6. Odpověď na hlavní otázku a meze

| směr predikce | uvnitř skupin (AUC) |
|---|---|
| klatrin → tvarový štítek (SI); kontrola, starší dataset | 0,775 |
| klatrin → dynamin (tento experiment, amplituda) | 0,657 |
| klatrin → dynamin (tento experiment, box-mean) | 0,595 |
| dynamin → SI (dřívější experimenty) | 0,585 |
| jedno klatrinové číslo → dynamin | 0,50 až 0,54 |

**Diskuze:** klatrin vypovídá o tvaru vlastní struktury, o dynaminu ví jen
málo, a naopak; každý kanál nese především svou vlastní informaci.
**Dynamin proto nelze nahradit klatrinovou intenzitou.** Meze: Youdenův práh
in-sample (sens a spec horní odhady); délkové skupiny jsou hrubé, proto se
srovnává s 0,596; štítek je naše klasifikace, ne absolutní pravda; 15 filmů
jedné buněčné linie.

![matice](fig_matice_logreg.png)

*Obrázek 2: Confusion matice logistické regrese, smíchaně (pooled) a po
délkových skupinách. Řádky obrázku jsou tři experimenty: klatrin→dynamin
s oběma odečty a pod nimi referenční dynamin→SI běh kolegů, takže klatrinové
a SI matice lze číst vedle sebe. V každé matici jsou řádky skutečný štítek
a sloupce verdikt; barva a procento udávají podíl v řádku. Pod maticí AUC,
práh (každý panel má vlastní, Youdenovo J na daném výběru, volený in-sample)
a sens/spec. Smíchané panely nafukuje délka; srovnává se po skupinách.*
"""


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts",
                    "fig_logreg_matrices.py"),
                    "--out", os.path.join(OUT, "fig_matice_logreg.png")],
                   check=True)
    fig_summary(os.path.join(OUT, "fig2_souhrn.png"))
    git = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip() or "?"
    md = MD.format(date=_dt.date.today().isoformat(), git=git)
    base = os.path.join(OUT, "protokol_klatrin_dynamin")
    with open(base + ".md", "w", encoding="utf-8") as fh:
        fh.write(md)
    tex = md_to_tex(md)
    tex = tex.replace("margin=2.2cm", "margin=1.9cm")
    tex = tex.replace("\\setlength{\\parskip}{5pt}",
                      "\\setlength{\\parskip}{3pt}\n"
                      "\\usepackage{titlesec}\n"
                      "\\titlespacing*{\\section}{0pt}{9pt}{4pt}")
    with open(base + ".tex", "w", encoding="utf-8") as fh:
        fh.write(tex)
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
