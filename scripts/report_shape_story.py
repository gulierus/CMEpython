#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protokol "Nese prubeh dynaminu informaci o Fate jamky?" -- tri testy
hypotezy o tvaru krivky (tvar / mnozstvi / sit), odpoved napred.

Cisla jsou prevzata z overenych behu (audit, scripts/shape_phase1-3);
zadne se tu nepocita znovu.

    python3 scripts/report_shape_story.py
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import shutil
import subprocess
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "shape_report")
ORANGE, TEAL, GREY, PURPLE = "#d95f02", "#1b9e77", "0.45", "#7570b3"

# overena cisla (AUC uvnitr delkovych skupin; bezne stitky / vycistene stitky)
WB = {
    "jen délka":            (0.561, 0.622),
    "ruční čísla (GBM)":    (0.579, 0.670),
    "síť: jen dynamin":     (0.590, 0.684),
    "síť: jen klatrin":     (0.672, 0.762),
    "síť: oba kanály":      (0.687, 0.787),
}
REF_BOXMEAN = 0.585   # kolegova logisticka regrese na box-mean, bezne stitky


def fig_summary(path: str) -> None:
    labels = list(WB)
    plain = [WB[k][0] for k in labels]
    hard = [WB[k][1] for k in labels]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, vals, title in ((axes[0], plain, "běžné štítky (max SI > 0,7)"),
                            (axes[1], hard, "vyčištěné štítky (SI > 0,7 ve ≥3 snímcích)")):
        x = np.arange(len(labels))
        cols = [GREY, PURPLE, ORANGE, TEAL, "#333333"]
        ax.bar(x, vals, 0.6, color=cols)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.006, f"{v:.3f}".replace("0.", "0,"), ha="center", fontsize=8.5)
        ax.axhline(0.5, color="0.6", lw=1, ls=":")
        ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=8.5)
        ax.set_ylim(0.45, 0.83)
        ax.set_title(title, fontsize=10)
    axes[0].axhline(REF_BOXMEAN, color=PURPLE, lw=1, ls="--")
    axes[0].text(4.42, REF_BOXMEAN + 0.005, "referenční model kolegů (0,585)",
                 fontsize=7.5, color=PURPLE, ha="right",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    axes[0].set_ylabel("AUC uvnitř délkových skupin")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


MD = """# Nese průběh dynaminu informaci o Fate jamky?

Datum {date} · CMEpython {git} · vygenerováno `scripts/report_shape_story.py`
· zdrojová čísla: `scripts/shape_phase1.py`, `shape_phase2.py`, `shape_phase3_cnn.py`
(výsledky v `out/shape_phase1-3/`)

## 1. Otázka a odpověď

Kolegové vyslovili hypotézu, že produktivní a abortivní jamky mají každá svůj
typický průběh dynaminu. Fate jamky by pak šlo předpovědět modelem, který ten
tvar najde. Podnět dal audit sporných případů „Dynamin negative" (šest jamek
z prohlížeče kolegů). Z auditu zde přebíráme dvě zjištění: dynamin je u sporných
jamek prokazatelně přítomen, a polovina „produktivních" jamek stojí na jediném
snímku SI (míra tvaru klatrinové struktury; vysvětluje sekce 2). Hypotézu jsme ověřili třemi testy. Odpověď napřed, ve čtyřech bodech:

1. **Tvar křivky Fate nepředpovídá.** Model, který vidí jen tvar, je na úrovni
   náhody. Obě skupiny mají stejný průběh a liší se množstvím dynaminu.
2. **Informaci nese množství a významnost dynaminu**, hlavně ke konci života
   jamky. Vytěží ji jednoduchý model z devíti čísel. Neuronová síť k tomu
   přidává jen 0,01 až 0,014 bodu AUC (stupnici vysvětluje sekce 2).
3. **Hranici všech výsledků určuje kvalita Fate štítku.** Přísnější definice
   produktivní jamky (SI nad 0,7 aspoň ve třech snímcích) zvedá čísla o 0,06
   až 0,10, tedy víc než jakákoli výměna modelu.
4. **Klatrin do modelu Fate nepatří.** Vysoká čísla sítě s klatrinem neměří
   biologii: štítek vzniká ze SI a SI se počítá z klatrinového obrazu, takže
   model zčásti předpovídá veličinu, ze které štítek vznikl.

Zbytek dokumentu tyto body dokládá. Než začneme, jedna sekce o tom, jak číst
čísla.

## 2. Jak číst čísla v tomto dokumentu

**Co měříme.** Vstupem všech testů je amplituda dynaminu: jas fluorescenční
tečky nad místním pozadím, jak ho pro každý snímek spočítá detekční software
(cmeAnalysis). Významný snímek je snímek, ve kterém amplituda prošla
statistickým testem proti místnímu pozadí; signál je tam prokazatelně nad
šumem.

**Co je SI a odkud jsou štítky.** SI (Shape Index) je číslo spočítané z tvaru
klatrinové struktury v obraze. Hodnoty kolem 0,7 a výš odpovídají silně
zakřivené jamce těsně před odškrcením, a proto kolegové definují produktivní
jamku prahem SI nad 0,7. Používáme dva druhy štítků. Běžný štítek: produktivní
je jamka se SI nad 0,7 v kterémkoli snímku (33 750 jamek, 24 % produktivních).
Vyčištěný štítek: produktivní vyžaduje SI nad 0,7 aspoň ve třech snímcích,
abortivní max SI pod 0,65 a nejasné jamky mezi tím vynecháváme (zbývá 26 354,
14 % produktivních). Vyčištěná sada reaguje na zjištění auditu, že polovina
„produktivních" jamek stojí na jediném snímku. Čísla spočítaná na běžných a na
vyčištěných štítcích spolu nesrovnáváme; každý sloupec tabulek čteme zvlášť.

**Měřítko úspěchu.** Hra na dvojice: vylosujeme jednu produktivní a jednu
abortivní jamku a ptáme se, jestli model dal vyšší skóre té produktivní. Podíl
správně seřazených dvojic je AUC. Hodnota 0,5 znamená náhodu (model neví nic);
1,0 znamená, že model seřadí správně každou dvojici.

**Losujeme jen stejně dlouhé dvojice.** Delší jamky jsou produktivní mnohem
častěji. Model, který pozná jen délku života, proto vypadá dobře, i když
o dynaminu neví nic. Dvojice tedy losujeme jen uvnitř délkových skupin. Délku
měříme v počtu snímků a skupiny jsou čtyři: 6 až 9, 10 až 19, 20 až 39 a 40
a více snímků. AUC spočítáme v každé skupině zvlášť a do tabulek dáváme průměr
vážený počtem dvojic ve skupině. Ani skupiny ale délku neodstraní úplně.
S hodnotou 0,5 proto výsledky nesrovnáváme; správné srovnání je vždy řádek
**„jen délka"** ve stejné tabulce. Model nese dynaminovou informaci teprve
tehdy, když je nad ním.

**Poctivost.** Filmů je 15 a rozdělili jsme je do pěti skupin. Model se vždy
učil na čtyřech skupinách a skóre počítal na páté. Každá jamka je tedy hodnocena
modelem, který její film při učení neviděl. Dělíme po celých filmech, protože
jamky z jednoho filmu jsou si podobné a model by jinak poznával film, ne
biologii.

## 3. Test 1: rozhoduje tvar křivky?

Tvar jsme oddělili od množství. Každou křivku dynaminu jsme přepočítali na
její vlastní úroveň (průměr 0, rozptyl 1). Čas jamky jsme převedli na procenta
jejího života (0 % narození, 100 % zánik) a křivku odečetli ve 20 stejně
vzdálených bodech. V takové křivce zbývá jen průběh: kdy roste, kdy vrcholí,
kdy klesá. Test běžel na vyčištěných štítcích, protože dávají hypotéze tvaru
nejsilnější možnou šanci; s běžnými štítky by čísla byla jen nižší.

| model vidí | AUC uvnitř délkových skupin |
|---|---|
| jen délku | 0,622 |
| jen úroveň (průměr a maximum amplitudy) | 0,576 |
| **jen tvar** | **0,505** |
| všechno dohromady | 0,667 |

![test1](fig1_tvar.png)

*Obrázek 1: Vlevo průměrný tvar křivky dynaminu: oranžově produktivní, zeleně
abortivní jamky; pás zakrývá prostřední polovinu jamek. Uprostřed: histogram
toho, ve které části života jamky dynamin vrcholí. Vpravo: podíl jamek s daným
počtem významných snímků dynaminu v posledních deseti. Čteme: první dva panely
se překrývají (stejný tvar, stejné načasování), třetí se rozchází (jiné
množství).*

**Diskuze:** tvar sám o sobě je náhoda (0,505) a obrázek ukazuje proč. Obě
skupiny mají stejný průběh: pomalý nárůst, vrchol v poslední pětině života
(obrázek 1, prostřední panel), pád na konci. Co skupiny odlišuje, je množství:
aspoň pět významných snímků v posledních deseti má 76 % produktivních a 40 %
abortivních jamek. Hypotéza o typickém tvaru tedy neplatí. Na kohortových
obrázcích je rozdíl ve výšce křivek, ne v jejich průběhu.

## 4. Test 2: kolik informace nese množství?

Když ne tvar, pak množství. Model dostal devět srozumitelných čísel z měřené
amplitudy, ve třech skupinách:

- **kolik významných snímků:** celkový počet, podíl z života, počet
   v posledních deseti snímcích, nejdelší souvislý běh;
- **kolik amplitudy:** průměr, maximum, výška vrcholu nad vlastní klidovou
   hladinou jamky;
- **co se děje na konci:** změna průměru v posledních deseti snímcích proti
   zbytku života; a síla nejsilnějšího snímku (nejvyšší hodnota testu
   významnosti, která na rozdíl od maxima amplitudy neměří jas, ale
   průkaznost signálu).

| model | běžné štítky | vyčištěné štítky |
|---|---|---|
| jen délka | 0,561 | 0,622 |
| množství a významnost | 0,545 | **0,648** |
| obojí dohromady (GBM) | 0,579 | 0,670 |

**Diskuze:** dvě zprávy. S běžnými štítky se všechny dynaminové modely tlačí
kolem 0,55 až 0,58, stejně jako referenční model kolegů (0,585; jejich
logistická regrese na box-mean odečtu, převzato z jejich referenčního běhu;
podrobně v dokumentu `interpretace_logreg.pdf`). Dynaminový model je tu dokonce
pod řádkem délky (0,545 proti 0,561). To není vada měření, ale důsledek štítku:
na jamkách „produktivních" podle jediného snímku se model učí šum. S vyčištěnými
štítky množství dynaminu řádek délky poráží (0,648 proti 0,622) a kombinace
dává 0,670. To je jiná kombinace než 0,667 z testu 1; tam byl ve hře i tvar,
který nic nepřidal.

## 5. Test 3: najde neuronová síť něco navíc?

Nakonec jsme pustili malou konvoluční síť (typ neuronové sítě, který si vzory
v časové řadě hledá sám, bez ručně vybraných čísel). Síť čte posledních 40
snímků: amplitudu dynaminu, výsledek testu významnosti a amplitudu klatrinu.
U jamek kratších než 40 snímků chybějící začátek doplňujeme nulami a čtvrtý
vstup (maska) síti říká, které snímky jsou skutečně naměřené. A protože síť
dostala dva kanály, pustili jsme ji i s každým kanálem zvlášť:

| síť vidí | běžné štítky | vyčištěné štítky |
|---|---|---|
| oba kanály | 0,687 | 0,787 |
| jen klatrin | 0,672 | 0,762 |
| **jen dynamin** | **0,590** | **0,684** |

**Diskuze:** plná síť vypadá jako velký skok, ale rozklad po kanálech ukazuje,
odkud pochází: skoro celý z klatrinu. Proč to nic neznamená: štítek Fate vzniká
ze SI a SI se počítá z klatrinového obrazu. Síť, která vidí klatrin, tedy
zčásti předpovídá veličinu, ze které štítek vznikl. Její vysoké číslo neměří
biologii, ale tuto vazbu. Skoro stejné číslo (0,775) dala už naše dřívější
pozitivní kontrola metody: stejný trénink s klatrinem jako vstupem a SI štítkem
na starším datasetu (proto se liší od 0,762 v tabulce). Poctivé srovnání je
řádek „jen dynamin": 0,590 proti 0,579 a 0,684 proti 0,670 ručních čísel.
**Síť přidává 0,01 až 0,014; pro dynamin není potřeba.**

Obrázek 2 nepřidává nová čísla, jen staví všechny modely vedle sebe.

![souhrn](fig2_souhrn.png)

*Obrázek 2: AUC uvnitř délkových skupin pro všechny modely; vlevo běžné štítky,
vpravo vyčištěné. Šedý sloupec je řádek „jen délka", fialový ruční čísla,
oranžový síť jen s dynaminem, zelený síť jen s klatrinem, černý síť s oběma
kanály. Tečkovaná čára je náhoda (0,5), čárkovaná fialová vlevo referenční
model kolegů (0,585).*

**Diskuze:** dynaminové sloupce (fialový a oranžový) převyšují šedý řádek délky
jen mírně. Zelený a černý sloupec jsou nadsazené vazbou klatrin–štítek (viz
výše); platí to v obou panelech. A pravý panel je celý výš než levý; to je
zisk vyčištěného štítku.

## 6. Co doporučujeme pro Fate klasifikaci

1. **Zpevnit definici štítku:** produktivní = SI nad 0,7 aspoň ve třech
   snímcích. Nejlevnější krok s největším ziskem (+0,06 až +0,10 všude).
2. **Měřit dynamin jako množství a významnost** (test amplitudy proti vlastnímu
   okolí, snímek po snímku), ne modelovat tvar křivky; tam informace není.
3. **Stačí jednoduchý model** na devíti číslech; neuronová síť přidává nejvýš
   setinu a půl a hůř se vysvětluje.
4. **Nedávat klatrin do modelu Fate,** dokud je štítek vyroben z klatrinu:
   model pak zčásti předpovídá veličinu, ze které štítek vznikl, a čísla
   vypadají lépe, než jaká jsou.

## 7. Meze

Dvě meze čtení čísel jsme popsali už v sekci 2: délkové skupiny ruší vliv délky
jen zhruba, a čísla z běžných a vyčištěných štítků se nesrovnávají navzájem.
Dále: jde o jednu buněčnou linii a jedny podmínky snímání, 15 filmů. A síť jsme
zkoušeli v jedné malé podobě; větší síť by mohla čísla posunout o setiny, závěr
ale stojí na rozkladu po kanálech, ne na velikosti sítě.

## 8. Shrnutí

1. Tvar křivky dynaminu je pro Fate jamky náhoda (0,505). Obě skupiny mají
   stejný průběh a liší se množstvím.
2. Množství a významnost nesou skutečný signál: 0,648 proti řádku délky 0,622
   na vyčištěných štítcích; 76 % proti 40 % jamek s významným dynaminem na
   konci života.
3. S běžnými štítky končí všechny dynaminové modely u 0,55 až 0,59; hranici
   určuje kvalita štítku, ne výpočty.
4. Neuronová síť přidává pro dynamin 0,01 až 0,014; skok plné sítě nese
   klatrin, jehož vazba na štítek čísla nadsazuje.
5. Doporučení: štítek ≥3 snímky, dynamin měřit přímo, jednoduchý model,
   klatrin do modelu Fate nedávat.
"""


TEX_HEAD = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.2cm]{geometry}
\usepackage{fontspec}
\usepackage[czech]{babel}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage[hidelinks]{hyperref}
\raggedbottom
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\begin{document}
"""


def md_to_tex(md: str) -> str:
    """Prevod naseho md na tex: nadpisy, tabulky, obrazky, seznamy, tucne."""
    lines = md.splitlines()
    out = []
    title = lines[0][2:]
    out.append("{\\LARGE\\bfseries %s}\\\\[4pt]" % title)
    i = 1
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("## "):
            out.append("\\section*{%s}" % ln[3:])
        elif ln.startswith("!["):
            m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", ln)
            cap = ""
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].startswith("*Obrázek"):
                cap = lines[j].strip("*")
                while not lines[j].rstrip().endswith("*"):
                    j += 1
                    cap += " " + lines[j].strip("*")
                i = j
            cap = re.sub(r"^Obrázek \d+: ", "", cap)
            out.append("\\begin{figure}[H]\\centering"
                       "\\includegraphics[width=\\linewidth]{%s}"
                       "\\caption{%s}\\end{figure}" % (m.group(1), texify(cap)))
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                if not set("".join(cells)) <= set("-: "):
                    rows.append(cells)
                i += 1
            i -= 1
            ncol = len(rows[0])
            out.append("\\begin{table}[H]\\centering\\small")
            out.append("\\begin{tabular}{l" + "r" * (ncol - 1) + "}")
            out.append("\\toprule " + " & ".join(texify(c) for c in rows[0]) + " \\\\ \\midrule")
            for r in rows[1:]:
                out.append(" & ".join(texify(c) for c in r) + " \\\\")
            out.append("\\bottomrule\\end{tabular}\\end{table}")
        elif ln.startswith("*Obrázek"):
            pass  # zpracovano u obrazku
        elif re.match(r"^\d+\. ", ln) or ln.startswith("- "):
            env = "enumerate" if ln[0].isdigit() else "itemize"
            out.append("\\begin{%s}" % env)
            while i < len(lines) and (re.match(r"^\d+\. ", lines[i])
                                      or lines[i].startswith("- ")):
                item = re.sub(r"^(\d+\. |- )", "", lines[i])
                while i + 1 < len(lines) and lines[i + 1].startswith("   "):
                    i += 1
                    item += " " + lines[i].strip()
                out.append("\\item " + texify(item))
                i += 1
            i -= 1
            out.append("\\end{%s}" % env)
        else:
            out.append(texify(ln))
        i += 1
    return TEX_HEAD + "\n".join(out) + "\n\\end{document}\n"


def texify(s: str) -> str:
    s = (s.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
          .replace("×", "$\\times$").replace("≥", "$\\geq$").replace("_", "\\_"))
    s = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", s)
    s = re.sub(r"`([^`]+)`", r"\\texttt{\1}", s)
    s = s.replace("„", ",,").replace("“", "``").replace("”", "``")
    s = s.replace('"', "``")   # ASCII uvozovka je v czech babel aktivni znak
    return s


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    src = os.path.join(ROOT, "out", "shape_phase1", "shape_phase1.png")
    shutil.copy(src, os.path.join(OUT, "fig1_tvar.png"))
    fig_summary(os.path.join(OUT, "fig2_souhrn.png"))

    git = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip() or "?"
    md = MD.format(date=_dt.date.today().isoformat(), git=git)
    with open(os.path.join(OUT, "protokol_tvar.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    with open(os.path.join(OUT, "protokol_tvar.tex"), "w", encoding="utf-8") as fh:
        fh.write(md_to_tex(md))

    if shutil.which("latexmk"):
        r = subprocess.run(["latexmk", "-xelatex", "-interaction=nonstopmode",
                            "-halt-on-error", "-cd",
                            os.path.join(OUT, "protokol_tvar.tex")],
                           capture_output=True, text=True)
        if r.returncode == 0:
            subprocess.run(["latexmk", "-c", "-cd",
                            os.path.join(OUT, "protokol_tvar.tex")],
                           capture_output=True, text=True)
            print("PDF: protokol_tvar.pdf")
        else:
            print("VAROVANI: xelatex selhal; konec logu:")
            print("\n".join(r.stdout.splitlines()[-12:]))
    print(f"hotovo -> {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
