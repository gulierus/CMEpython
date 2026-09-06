#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protokol "Nese prubeh dynaminu informaci o osudu jamky?" -- cely pribeh:
pripady kolegu -> audit -> hypoteza tvaru -> tri patra testu -> zaver.

Cisla jsou prevzata z overenych behu (audit 28 agentu, scripts/shape_phase1-3);
zadne se tu nepocita znovu.

    python3 scripts/report_shape_story.py
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "shape_report")
ORANGE, TEAL, GREY, PURPLE = "#d95f02", "#1b9e77", "0.45", "#7570b3"

# overena cisla (within-band AUC; bezne stitky / vycistene stitky)
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
    axes[0].set_ylabel("rozlišení uvnitř délkových skupin (AUC)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


MD = """# Nese průběh dynaminu informaci o osudu jamky?

Datum {date} · CMEpython {git} · vygenerováno `scripts/report_shape_story.py`
· zdrojová čísla: audit (28 nezávisle ověřených výpočtů) a `scripts/shape_phase1-3.py`

## 1. Jak to začalo: případy kolegů a audit

Kolegové si v prohlížeči případů všimli jamek, které mají hezký kroužek, Shape Index
nad 0,7 a v obraze viditelný příchod dynaminu, a přesto nesou verdikt „Dynamin
negative". Ptali se, čím to je. Provedli jsme audit šesti ukázaných jamek a celého
řetězu výpočtu. Každé číslo auditu přepočítal nezávisle druhý výpočet.

První zjištění: verdikt nevydává měření, ale trénovaný model. Naše přímé měření
(jas tečky proti jejímu vlastnímu okolí, test na každém snímku) vidí dynamin u všech
šesti jamek naprosto jasně. Síla signálu je 18 až 52 násobků šumu. Ztrácí se až po
cestě, na čtyřech místech:

1. **Odečet.** Model nedostává jas tečky, ale průměr čtverečku 5×5 pixelů vydělený
   jasem celé buňky. Za „významný" se pak bere jen snímek nad společnou laťkou
   (horních 10 % hodnot celého souboru). Žádný ze šesti záblesků tu laťku nepřekročil.
   Celkově zůstává pod laťkou 62,5 % jamek, které podle měření dynamin mají.
2. **Body.** Model se učil sám, za co dávat body. Jediná spolehlivá kladná váha je
   trvale zvýšená hladina dynaminu. Za krátký záblesk na konci body nedává, spíš
   strhává.
3. **Cíl učení.** Model nebyl učen poznávat dynamin, ale štítek SI. Jeho „Dynamin
   negative" proto znamená „nevypadá to na SI-produktivní", ne „bez dynaminu".
4. **Rozhodovací čára.** Skóre všech šesti jamek leží těsně pod čarou 0,222, která
   hlavně kopíruje délku života jamky. Jedna z jamek by při jiném rozdělení filmů
   při tréninku vyšla kladně v 9 z 10 případů.

Vedle toho jsou tři z šesti jamek „produktivní" jen díky jedinému snímku SI těsně
nad 0,7. Posun mezi kamerami jsme vyloučili (nejvýše 0,065 pixelu).

**Diskuze:** z auditu vzešla otázka pro tento protokol. Kolegové vyslovili hypotézu,
že produktivní a abortivní jamky mají každá svůj typický průběh dynaminu, a že tedy
stačí lepší model, který ten tvar najde. Postavili jsme na to tři patra testů.

## 2. Data, štítky a způsob hodnocení

Pracujeme s 15 filmy U2OS (2 s na snímek) a 33 750 drahami jamek s délkou aspoň
6 snímků. Vstupem je amplituda dynaminu z cmeAnalysis: jas tečky proti jejímu
vlastnímu okolí, s testem významnosti na každém snímku. To je odečet, který v auditu
viděl všech šest sporných jamek správně.

Štítky bereme dvoje. **Běžné**: produktivní znamená SI nad 0,7 v kterémkoli snímku.
**Vyčištěné**: produktivní znamená SI nad 0,7 aspoň ve třech snímcích, abortivní
znamená max SI pod 0,65, a 7 396 nejasných drah mezi tím vynecháváme (zbývá 26 354).
Vyčištěné štítky reagují na zjištění auditu, že polovina „produktivních" drah stojí
na jediném snímku.

Hodnotíme hrou na dvojice: vylosujeme jednu produktivní a jednu abortivní dráhu
a ptáme se, jak často jim model dá skóre ve správném pořadí (AUC; 0,5 je mince,
1,0 je vždy správně). Losujeme jen dvojice **podobné délky** (čtyři délkové skupiny),
protože delší jamky jsou produktivní častěji a model by si jinak pomáhal délkou.
Ani skupiny délku neodstraní celou, a proto má každá tabulka řádek „jen délka".
Model má smysl teprve tehdy, když tento řádek porazí. Každé skóre pochází z modelu,
který dráhy z daného filmu při učení neviděl (dělení po filmech, 5 skupin).

## 3. Patro 1: má průběh zvláštní tvar?

Tvar jsme oddělili od množství: každou křivku dynaminu jsme přepočítali na její
vlastní úroveň (průměr 0, rozptyl 1) a na 20 bodů zlomku života. V takové křivce
zbývá jen průběh: kdy roste, kdy vrcholí, kdy klesá. Výška ani délka v ní nejsou.
Tři soupeři (vyčištěné štítky):

| model vidí | rozlišení uvnitř délkových skupin |
|---|---|
| A: jen délku | 0,622 |
| B: jen úroveň (průměr, maximum) | 0,576 |
| C: jen tvar | **0,505** |
| A+B+C dohromady | 0,667 |

![patro1](fig1_tvar.png)

*Obrázek 1: Vlevo průměrný tvar křivky dynaminu (oranžově produktivní, zeleně
abortivní; pás je rozpětí prostřední poloviny drah). Uprostřed rozdělení polohy
vrcholu dynaminu v životě. Vpravo podíl drah s daným počtem významných snímků
v posledních deseti. Čteme: levý a prostřední panel se skoro překrývají, pravý se
liší.*

**Diskuze:** tvar sám o sobě je mince (0,505). Obě skupiny mají stejný průběh:
pomalý nárůst, vrchol kolem 80 % života, pád na konci. Liší se v tom, **kolik**
dynaminu přišlo a jestli byl významný: aspoň 5 významných snímků v posledních deseti
má 76 % produktivních a 40 % abortivních drah. Hypotéza o zvláštním tvaru tedy
neplatí; na kohortových obrázcích je vidět rozdíl ve výšce křivek, ne v jejich
průběhu.

## 4. Patro 2: model z množství a významnosti

Postavili jsme model na devíti srozumitelných číslech z měřené amplitudy: počet
a podíl významných snímků, významné snímky v posledních deseti, nejdelší souvislý
běh významných snímků, průměr a maximum amplitudy, výška vrcholu nad vlastní
hladinou, změna na konci proti zbytku života a síla nejsilnějšího snímku.

| model | běžné štítky | vyčištěné štítky |
|---|---|---|
| jen délka | 0,561 | 0,622 |
| množství+významnost | 0,545 | 0,648 |
| obojí, nelineárně (GBM) | 0,579 | 0,670 |

**Diskuze:** dvě zprávy. S běžnými štítky se všechno mačká kolem 0,55 až 0,58,
stejně jako referenční model kolegů (0,585). Strop tu nedrží model ani měření, ale
štítek: na jednosnímkových špičkách se učit nedá. S vyčištěnými štítky množství
dynaminu poráží samotnou délku (0,648 proti 0,622) a kombinace dává 0,670.

## 5. Patro 3: síť a rozklad po kanálech

Nakonec jsme pustili malou konvoluční síť, která čte posledních 40 snímků surově,
bez ručních čísel: amplitudu dynaminu, výsledek testu, amplitudu klatrinu a masku
skutečných snímků. Učení běželo v 5 skupinách po filmech, třikrát s různým
náhodným začátkem, průměrováno. A pak totéž znovu jen s dynaminem a jen s klatrinem:

| síť vidí | běžné štítky | vyčištěné štítky |
|---|---|---|
| oba kanály | 0,687 | 0,787 |
| jen klatrin | 0,672 | 0,762 |
| **jen dynamin** | **0,590** | **0,684** |

**Diskuze:** plná síť vypadá jako velký objev, ale rozklad ukazuje, odkud skok je.
Skoro celý pochází z klatrinového kanálu. Ten ale předpovídá štítek, který je
z klatrinu vyroben (SI je tvar téže struktury), takže jde napůl o kruh; stejné
číslo dala už dřívější kontrola metody (0,775). Síť **jen s dynaminem** přidává
proti ručním číslům jediné procento (0,590 proti 0,579 a 0,684 proti 0,670).
Neuronka tedy pro dynamin není potřeba: informace je v množství, a to vytěží
jednoduchý model.

## 6. Souhrn čísel

![souhrn](fig2_souhrn.png)

*Obrázek 2: Rozlišení uvnitř délkových skupin pro všechny modely, vlevo běžné
štítky, vpravo vyčištěné. Šedý sloupec „jen délka" je laťka, kterou musí model
porazit. Tečkovaná čára je mince (0,5), čárkovaná fialová vlevo je referenční
model kolegů (0,585). Čteme: dynaminové modely (fialový, oranžový sloupec) laťku
porážejí jen mírně; velký skok síťových sloupců napravo nese klatrin.*

## 7. Co z toho plyne

1. **Průběh dynaminu nemá pro osud jamky zvláštní tvar.** Informace je v tom,
   kolik dynaminu přišlo a jestli byl statisticky významný, hlavně ke konci života.
2. **Tuhle informaci plně vytěží jednoduchý model na měřených amplitudách.**
   Neuronka přidává jediné procento; není potřeba.
3. **Největší dostupné zlepšení není ve výpočtech, ale ve štítku.** Definice
   „produktivní = SI nad 0,7 aspoň ve 3 snímcích" zvedá všechna čísla zhruba
   o 0,1. Doporučujeme ji kolegům jako první krok.
4. **Klatrinový kanál se do modelu osudu dávat nesmí**, pokud je štítek vyroben
   z klatrinu: model se pak učí kruhem a čísla vypadají lépe, než jaká jsou.
5. Pro tvrzení „dynamin ano/ne" u jednotlivé jamky zůstává nejlepší přímé měření
   s testem proti vlastnímu okolí, ne skóre modelu učeného na SI.

## 8. Meze

Za prvé, délkové skupiny ruší vliv délky jen zhruba; proto všude srovnáváme
s řádkem „jen délka" a ne s mincí. Za druhé, vyčištěné štítky dělají úlohu lehčí
(vynechávají nejasné dráhy); čísla mezi oběma režimy se proto nemají porovnávat
navzájem, jen uvnitř sloupce. Za třetí, jde o jednu buněčnou linii a jedny
podmínky snímání, 15 filmů. Za čtvrté, síť jsme zkoušeli v jedné malé podobě;
větší síť by mohla čísla posunout o setiny, směr závěrů ale drží rozklad po
kanálech, ne velikost sítě.

## 9. Shrnutí

1. Audit: „Dynamin negative" z prohlížeče případů vyrábí model učený na SI,
   ne měření; dynamin je u sporných jamek jasně přítomen.
2. Tvar křivky dynaminu je pro osud jamky mince (0,505); obě skupiny mají stejný
   průběh a liší se množstvím.
3. Množství a významnost dávají skutečný signál: 0,648 proti délce 0,622
   na vyčištěných štítcích.
4. S běžnými štítky vše končí u 0,55 až 0,59; strop drží kvalita štítku.
5. Síť jen s dynaminem přidává 0,01; skok plné sítě nese klatrin, který je se
   štítkem v kruhu.
6. Doporučení: zpevnit definici štítku (≥3 snímky), měřit dynamin přímo,
   nekombinovat klatrin se štítkem vyrobeným z klatrinu.
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
    """Prevod naseho md na tex: nadpisy, tabulky, obrazky, tucne, kurziva."""
    import re
    lines = md.splitlines()
    out = []
    i = 0
    # titul
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
    import re
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
