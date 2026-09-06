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


MD = """# Klatrinový experiment: jde dynaminová produktivita předpovědět z klatrinu?

Datum {date} · CMEpython {git} · vygenerováno `scripts/report_clavg_experiment.py`
· zdrojové běhy: `sample_clathrin_avg.py`, `clavg_threshold_analysis.py`, Helios joby 238740/238741

## 1. Zadání a odpověď

Zadání kolegů mělo dvě části. Za prvé: nasamplovat klatrinovou intenzitu pro
stejné trajektorie jamek jako dosud, metodou cmeAnalysis i jednodušší metodou
(průměr 5×5 s kompenzací blednutí), a uložit jako samostatný sdílený výstup.
Za druhé: zopakovat klasifikační experiment, tentokrát s klatrinovou intenzitou
jako vstupem a s produktivitou (Fate) určenou z dynaminu jako štítkem, včetně
otázky, jak dobře rozlišuje samotná intenzita a jaký práh funguje nejlépe
(práh volit Youdenovým J; vysvětlujeme v sekci 4). Hlavní otázka zadání:
**nese dynamin informaci, kterou samotný klatrinový signál nezachytí?**

Odpověď napřed, ve třech bodech:

1. **Dynamin nahradit klatrinem nejde.** Nejlepší klatrinový model nese jen
   malý kousek dynaminové informace: AUC po očištění o délku 0,657 proti
   0,596, kterých dosáhne samotná délka života (měřítko vysvětluje sekce 2).
   S jednodušším odečtem nezbývá nic (0,595 až 0,606, na úrovni délky).
2. **Samotné klatrinové číslo nerozliší nic.** Žádný práh nefunguje (po
   očištění o délku AUC 0,50 až 0,54). Nejlepší jednoduché pravidlo pro dynamin+
   není o klatrinu, ale o délce života (aspoň 14 snímků).
3. **Nasamplovaná data jsou uložena a připravena ke sdílení** bez ohledu na
   výsledek klasifikace; všechny kontroly prošly.

Zbytek dokumentu tyto věty dokládá. Než začneme, jedna sekce o tom, jak číst
čísla.

## 2. Jak číst čísla v tomto dokumentu

**Měřítko úspěchu.** Hra na dvojice: vylosujeme jednu dynamin+ a jednu dynamin−
jamku a ptáme se, jestli model dal vyšší skóre té dynamin+. Podíl správně
seřazených dvojic je AUC. Hodnota 0,5 znamená náhodu (model neví nic); 1,0
znamená, že model seřadí správně každou dvojici.

**Štítek.** Dynamin+ / dynamin− podle výchozí klasifikace cmeAnalysis
(s opravenými maskami). Na každém snímku se testuje, jestli je amplituda
dynaminu (jas tečky nad místním pozadím) prokazatelně nad šumem; takový
snímek je „významný". Binomické pravidlo pak rozhodne podle počtu
významných snímků; dynamin+ zhruba znamená víc významných snímků, než by
při dané délce dráhy dal samotný šum (podrobný popis je v protokolu
porovnání klasifikací, včetně opravených masek buňky). Jamku s prokázaným
dynaminem čteme jako produktivní (Fate); dál píšeme dynamin+. Z 33 750
jamek je dynamin+ 56,8 %.

**Losujeme jen stejně dlouhé dvojice.** Podíl dynamin+ roste s délkou života
jamky z 24 % na 95 %. Cokoli, co souvisí s délkou, proto vypadá ve smíchaných
(pooled) číslech skvěle. Dvojice tedy losujeme jen uvnitř délkových skupin.
Délku měříme v počtu snímků a skupiny jsou čtyři: 6 až 9, 10 až 19, 20 až 39
a 40 a více snímků. AUC spočítáme v každé skupině zvlášť a do tabulek dáváme
průměr vážený počtem dvojic. Ani skupiny délku neodstraní úplně. S hodnotou
0,5 proto výsledky nesrovnáváme; správné srovnání je řádek **„jen délka"**:
smíchaně dává 0,810 a uvnitř skupin 0,596. Model nese dynaminovou informaci
teprve tehdy, když je nad ním.

**Poctivost.** Filmů je 15 a rozdělili jsme je do pěti skupin. Model se vždy
učil na čtyřech skupinách a skóre počítal na páté. Každá jamka je tedy
hodnocena modelem, který její film při učení neviděl. Dělíme po celých
filmech, protože jamky z jednoho filmu jsou si podobné a model by jinak
poznával film, ne biologii.

## 3. Úkol 1: nasamplovaná klatrinová intenzita

**Obě metody jsou spočítané pro všech 702 630 snímků 33 750 drah a uložené jako sdílený balíček `Clathrin Analysis/sampled/`.** Nová data jsou jednokanálové klatrinové filmy (jeden snímek = průměr devíti
surových SIM snímků) s intenzitami bez přeškálování. Ověřili jsme po pixelech,
že starý dvoukanálový dataset vznikl z těchto dat vynásobením všech intenzit filmu jednou konstantou (43 až 149, pro každý film jinou); souřadnice dosavadních trajektorií proto sedí
beze změny a nové jednotky jsou poprvé srovnatelné mezi filmy.

Pro všech 702 630 snímků 33 750 drah v 15 filmech jsme spočítali obě metody:

- **Metoda A (cmeAnalysis):** amplituda 2D fitu tečky nad vlastním okolím,
   s nejistotou a testem významnosti na každém snímku. Šířka tečky kalibrována
   přímo na nových datech: 1,6423 px (stará klatrinová kalibrace 1,6424).
- **Metoda B (box-mean):** průměr 5×5 pixelů na pozici jamky, vydělený
   dvojexponenciálním fitem průměrů snímků (kompenzace blednutí). Fit
   zkonvergoval na všech 15 filmech.

Kontroly: zarovnání souřadnic na nová data má shodu R² = 1,00000 v každém
filmu; 0 neplatných fitů. Obě metody spolu souvisejí, ale neměří stejnou věc
(pořadová korelace 0,77): metoda A měří tečku nad okolím, metoda B sbírá
i rozptýlený signál v okénku.

Balíček obsahuje 15 CSV, `manifest.json`, `threshold_analysis.json` (zdroj
tabulky v sekci 4) a README s popisem sloupců a jednotek.

## 4. Úkol 2a: samotná intenzita a nejlepší práh

Každou dráhu jsme zhustili do jednoho čísla (maximum nebo průměr přes život,
pro obě metody) a hledali nejlepší práh Youdenovým J, tedy práh s největším součtem sens
a spec (sens = podíl správně zachycených dynamin+, spec = podíl správně
odmítnutých dynamin−):

| skóre dráhy | smíchaně | uvnitř skupin | nejlepší práh | sens | spec |
|---|---|---|---|---|---|
| box-mean, maximum | 0,548 | 0,495 | 2,07 | 0,33 | 0,75 |
| box-mean, průměr | 0,533 | 0,509 | 1,98 | 0,21 | 0,85 |
| amplituda, maximum | 0,563 | 0,504 | 312 | 0,33 | 0,79 |
| amplituda, průměr | 0,574 | 0,543 | 158 | 0,44 | 0,69 |
| **jen délka života** | **0,810** | **0,596** | 14 snímků | 0,74 | 0,75 |

Prahy box-mean jsou bezrozměrné (násobky průměru snímku po kompenzaci
blednutí), prahy amplitudy v jednotkách intenzity nového datasetu. Práh se
volí na stejných datech, sens a spec jsou proto horní odhady.

**Diskuze:** žádný klatrinový práh nefunguje. Všechna intenzitní čísla jsou
hluboko pod řádkem „jen délka" a po očištění o délku leží na náhodě. Nejlepší
jednoduché pravidlo pro dynamin+ není o klatrinu: „jamka žije aspoň 14 snímků"
trefí sens 0,74 a spec 0,75. Jasnější jamka neznamená jamku s dynaminem.

## 5. Úkol 2b: replikace klasifikačního experimentu

**Trénovaný model na amplitudě přerůstá řádek délky o 0,06; box-mean varianta zůstává na jeho úrovni.** Postup: trénovací kód kolegů jsme
spustili beze změny řádku. Korpusy nesou klatrinovou
intenzitu (dvě varianty: amplituda a box-mean) a dynaminový štítek jsme vložili
formátovým trikem: sloupec `cls` obsahuje 1,0 pro dynamin+ a 0,0 pro dynamin−. Pravidlo „max(cls) > 0,7", kterým si kód kolegů čte štítek, tak převezme náš dynaminový štítek bez jediné změny kódu. Stejné dělení po filmech, stejné modely, práh Youdenovým J. V tabulce jsou tři běžné učicí metody různé složitosti (stromy XGBoost, logistická regrese, malá neuronová síť); jejich detail není pro výsledek podstatný.

| model | smíchaně (amp / box) | uvnitř skupin (amp / box) |
|---|---|---|
| XGBoost | 0,802 / 0,783 | **0,657** / 0,595 |
| logistická regrese | 0,796 / 0,769 | 0,640 / 0,572 |
| síť (MLP) | 0,805 / 0,794 | 0,637 / 0,606 |
| jen výška vrcholu (bez učení) | 0,562 / 0,548 | 0,504 / 0,495 |
| zamíchané štítky (kontrola) | 0,542 / 0,524 | 0,519 / 0,507 |

U logistické regrese (u ní je práh na skóre nejčitelnější) vychází sens
0,79 a spec 0,70 (amplituda, práh 0,503); u box-mean sens 0,77 a spec 0,70
(práh 0,497). Práh se tu klade na výstupní skóre modelu mezi 0 a 1, ne na
intenzitu jako v sekci 4; hodnota kolem 0,5 je proto přirozená.
Confusion matice všech modelů jsou na obrázku 1, hlavní výsledek na obrázku 2.

![mozaika](fig1_mosaic.png)

*Obrázek 1: Confusion matice všech modelů pro amplitudovou variantu (jamky se zánikem uvnitř filmu, bez dalšího filtrování). Každý panel je jeden model: řádky skutečný štítek
(dynamin+ / dynamin−), sloupce verdikt modelu při Youdenově prahu. Čteme: levý
horní roh = správně zachycené dynamin+, pravý dolní = správně odmítnuté
dynamin−. Upozornění: práh se volí na stejných datech (in-sample), sens a spec
jsou proto horní odhady; a smíchaná čísla nafukuje délka.*

![souhrn](fig2_souhrn.png)

*Obrázek 2: Hlavní výsledek po očištění o délku. Šedý sloupec je nejlepší
jediné klatrinové číslo (amplituda, průměr přes život; 0,543), zelený a oranžový trénované modely na dvou odečtech.
Čárkovaná fialová čára je řádek „jen délka života" (0,596), tečkovaná náhoda (0,5).
Čteme: box-mean na řádek délky pouze dosáhne, amplituda ho přerůstá o 0,06.*

**Diskuze:** smíchaná čísla kolem 0,80 vypadají dobře, ale jsou to délková
čísla; řádek „jen délka" má 0,810 sám o sobě. Poctivé čtení je uvnitř skupin
proti řádku délky 0,596:
**box-mean varianta zůstává na jeho úrovni (0,595 až 0,606), amplitudová ho přerůstá o 0,06 (0,657).**
Síť dává u box-mean 0,606, o setinu nad řádkem délky; rozdíl této velikosti
nečteme jako signál (viz Meze). V časovém průběhu klatrinu tedy trocha
dynaminové informace je, ale malá, a s jednodušším odečtem se ztrácí. Kontroly sedí: zamíchané štítky dávají hodnoty u náhody (0,51 až 0,52;
odchylka od 0,5 je v mezích šumu), jediné netrénované číslo také.

## 6. Odpověď na hlavní otázku

| směr predikce | uvnitř skupin (AUC) |
|---|---|
| klatrin → tvarový štítek (SI); dřívější kontrola, starší dataset | 0,775 |
| klatrin → dynamin (tento experiment, amplituda) | 0,657 |
| klatrin → dynamin (tento experiment, box-mean) | 0,595 |
| dynamin → SI (dřívější experimenty) | 0,585 |
| jedno klatrinové číslo → dynamin | 0,50 až 0,54 |

*Zdroje: 0,775 = pozitivní kontrola na starším datasetu (joby 238690 a 238691);
0,585 = referenční běh kolegů, podrobně v dokumentu `interpretace_logreg.pdf`.*

**Diskuze:** klatrin výborně vypovídá o tvaru vlastní struktury, ale o dynaminu
ví jen málo, a naopak. Každý kanál nese především svou vlastní informaci.
**Dynamin proto nelze nahradit klatrinovou intenzitou**; dynaminové měření
přináší informaci, kterou klatrinový signál nezachytí. To je odpověď na hlavní
otázku zadání.

## 7. Meze

Za prvé, Youdenův práh se volí na stejných datech, na kterých se hodnotí; sens
a spec jsou horní odhady. Za druhé, délkové skupiny jsou hrubé a zbytek délky
v nich zůstává; proto se srovnává s hodnotou 0,596, ne s 0,5, a malé překročení u box-mean
nelze číst jako signál. Za třetí, dynaminový štítek je naše výchozí
klasifikace cmeAnalysis; je dokumentovaná a reprodukovatelná, ale je to štítek,
ne absolutní pravda. Za čtvrté, jde o 15 filmů jedné buněčné linie a jedny
podmínky snímání.

## 8. Shrnutí

1. Nasamplovaná klatrinová intenzita pro všechny dosavadní trajektorie je
   uložena a připravena ke sdílení (obě metody, zarovnání R² = 1, šířka tečky
   1,6423 px, blednutí zkonvergovalo všude).
2. Samotná klatrinová intenzita dynaminovou produktivitu nerozliší (uvnitř
   skupin 0,50 až 0,54); žádný práh nefunguje, nejlepší jednoduché pravidlo je
   délka života aspoň 14 snímků.
3. Trénovaný model: smíchaně ~0,80 = délka; uvnitř skupin box-mean 0,595 až 0,606
   (úroveň hodnoty jen z délky), amplituda 0,657 (+0,06 skutečné informace). Volba odečtu
   rozhoduje.
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
