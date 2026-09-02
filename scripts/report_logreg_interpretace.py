# -*- coding: utf-8 -*-
"""Interpretace puvodni (referencni) analyzy logistickou regresi.

Cte VYHRADNE vysledky puvodniho behu (box-mean readout) z commitnuteho
dynamin_v2_confusion_sweep.json na vetvi release/dynamin-confusion-v1
a sestavi dokument (md + tex + pdf) ve stylu VU_RG: metodologie (na cem
se trenovalo, co se deje na pozadi), vysledky, interpretace a meze.

    python3 scripts/report_logreg_interpretace.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGREG = "logistic regression"
NULL = "permuted-label null (XGBoost)"
CONFIG_LABELS = {
    "none_interior": "bez filtru, interior", "none_endobs": "bez filtru, end-observed",
    "lt3_interior": "cluster < 3, interior", "lt3_endobs": "cluster < 3, end-observed",
    "lt2_interior": "cluster < 2, interior", "lt2_endobs": "cluster < 2, end-observed",
    "second5_interior": "2. soused ≥ 5 px, interior", "second5_endobs": "2. soused ≥ 5 px, end-observed",
}
BLUE, PURPLE, GREY = "#2b7bba", "#8e44ad", "0.45"


def fmt_n(v):
    return f"{int(v):,}".replace(",", " ")


def cz(v, nd=3):
    return f"{v:.{nd}f}".replace(".", ",")


def wb_auc(cfg, model):
    num = den = 0.0
    for b in cfg["bands"]:
        m = b["models"][model]
        n1 = b["n_productive"]; n0 = b["n"] - n1
        w = n1 * n0
        num += m["auc"] * w; den += w
    return num / den if den else np.nan


def rows_logreg(cfgs):
    rows = []
    for tag, lbl in CONFIG_LABELS.items():
        c = cfgs[tag]
        m = c["pooled"]["models"][LOGREG]
        wb = wb_auc(c, LOGREG)
        rows.append(dict(tag=tag, lbl=lbl, n=c["n_tracks"], prev=c["prevalence"],
                         auc=m["auc"], wb=wb, gap=m["auc"] - wb, thr=m["threshold"],
                         sens=m["sensitivity"], spec=m["specificity"],
                         null_auc=c["pooled"]["models"][NULL]["auc"],
                         null_wb=wb_auc(c, NULL),
                         rho=c["pooled"].get("rho_xgb_score_vs_length", np.nan)))
    return rows


def fig_overview(rows, out):
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r["auc"] for r in rows], 0.34, color=BLUE, label="pooled AUC")
    ax.bar(x + 0.18, [r["wb"] for r in rows], 0.34, color=PURPLE, label="within-band AUC")
    ax.plot(x, [r["null_wb"] for r in rows], "o--", color=GREY, ms=4,
            label="permutovaný null (within-band)")
    for xi, r in zip(x, rows):
        ax.text(xi - 0.18, r["auc"] + 0.005, cz(r["auc"]).replace("0,", ","), ha="center", fontsize=7.5)
        ax.text(xi + 0.18, r["wb"] + 0.005, cz(r["wb"]).replace("0,", ","), ha="center", fontsize=7.5)
    ax.axhline(0.5, color="0.6", lw=1, ls=":")
    ax.set_xticks(x, [r["lbl"] for r in rows], rotation=30, ha="right", fontsize=8.5)
    ax.set_ylim(0.45, 0.82)
    ax.set_ylabel("AUC")
    ax.set_title("Logistická regrese v referenčním běhu: pooled a within-band AUC po konfiguracích",
                 fontsize=10.5)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def md_table(rows_data, header):
    head = "| " + " | ".join(header) + " |"
    sep = "|" + "|".join("---" for _ in header) + "|"
    body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in rows_data)
    return f"{head}\n{sep}\n{body}"


def table_rows(rows):
    return [(r["lbl"], fmt_n(r["n"]), cz(100 * r["prev"], 1) + " %", cz(r["auc"]),
             cz(r["wb"]), "+" + cz(r["gap"], 2), cz(r["sens"], 2), cz(r["spec"], 2))
            for r in rows]


def build_markdown(rows, meta):
    t = md_table(table_rows(rows),
                 ["konfigurace", "n drah", "podíl SI+", "pooled AUC", "wb AUC", "gap",
                  "sens", "spec"])
    prim = next(r for r in rows if r["tag"] == "none_endobs")
    intr = next(r for r in rows if r["tag"] == "none_interior")
    return f"""# Interpretace referenční analýzy: logistická regrese na box-mean readoutu

Datum {meta['date']} · CMEpython {meta['git']} · vygenerováno `scripts/report_logreg_interpretace.py`
· zdroj čísel: `dynamin_v2_confusion_sweep.json` větve `release/dynamin-confusion-v1`

## 1. Co je předmětem dokumentu

Popisujeme a interpretujeme původní (referenční) analýzu, ve které se logistická regrese učí
predikovat SI nálepku dráhy z průběhu dynaminové intenzity. Nejprve vysvětlíme, na čem přesně
se trénovalo a co se děje na pozadí. Následně shrneme výsledky a jejich čtení. Naše opakování
téhož experimentu s intenzitou z cmeAnalysis zde záměrně neřešíme; je v samostatném protokolu.

## 2. Data a readout

Trénovalo se na korpusu 15 filmů U2OS (2 s/snímek, 79,1 nm/px). Každá dráha klatrinové jamky
nese nálepku: *produktivní* ⟺ max SI přes život > 0,7. Dynaminová intenzita dráhy je tzv.
*box-mean* readout, a to průměr 5×5 pixelů dynaminového kanálu na pozici jamky v každém
snímku, vydělený biexponenciálním fitem průměrů snímků dané buňky. Po tomto dělení je 1
rovna průměru buňky; pracuje se s *excessem* = hodnota − 1, takže 0 znamená „na úrovni
průměru buňky". Poznamenejme dvě vlastnosti volby. Za prvé, dělení průměrem buňky srovnává
filmy s různým jasem a odstraňuje blednutí. Za druhé, box-mean sbírá všechen signál v okénku,
tedy i difuzní membránový dynamin a příspěvky sousedních jamek; lokální pozadí se neodečítá.

Korpus se vyhodnocuje v osmi konfiguracích filtrů (tabulka 1): kombinace *cluster* filtru
(žádný, < 3, < 2), vzdálenostního filtru (2. nejbližší soused ≥ 5 px) a úplnosti dráhy
(*interior* = začátek i konec v záznamu; *end-observed* = navíc dráhy s useknutým začátkem).
Primární korpus analýzy je „bez filtru, end-observed" (n = {fmt_n(prim['n'])}).

## 3. Na čem přesně se model učí: featury

Z každé dráhy se počítá 27 vstupních čísel; informativních je 17, zbylých 10 (`pre_*`,
`post_*`) je v tomto korpusu identicky nulových (vzorky před vznikem a po zániku dráhy
korpus neobsahuje).

- **Terminální okno (10):** excess v posledních 20 s života, převzorkovaný v reálném čase na
  10 bodů; u drah kratších než okno se hodnoty před vznikem drží na první naměřené.
- **Střední fáze života (4):** průměr a maximum excessu před oknem, podíl „významných"
  snímků (excess > 90. percentil korpusu) a poměr maxima střední fáze k maximu okna;
  u drah ≤ 10 snímků jsou nulové.
- **Globální skaláry (3):** maximum excessu přes celý život, normalizovaná poloha vrcholu
  v životě a nejdelší souvislý běh významných snímků.

Délka dráhy mezi featurami záměrně **není**. Přesto ji modely ze vstupů rekonstruují
(Spearman ρ skóre–délka {cz(prim['rho'], 2)}, měřeno na XGBoost skóre téhož běhu),
hlavně přes nulové featury střední fáze
u krátkých drah; to je známý strukturální únik.

## 4. Co se děje na pozadí: trénink a vyhodnocení

Postup je pro každou konfiguraci stejný.

1. Sestaví se matice drahy × 27 featur a vektor SI nálepek.
2. Featury projdou mediánovou imputací a standardizací (odečtení průměru, dělení směrodatnou
   odchylkou).
3. Natrénuje se logistická regrese s mírnou L2 regularizací (C = 1, řešič lbfgs).
4. Vyhodnocení je *out-of-fold*: filmy se rozdělí do 5 skupin, natrénuje se 5 nezávislých
   modelů, každý na 12 filmech, a každá dráha dostane skóre od modelu, který její film
   neviděl. Grupování po filmech brání úniku „poznávání buňky".
5. Z agregovaných OOF skóre se zvolí práh maximalizací Youdenova J (sens + spec − 1)
   a spočítá confusion matice. Práh je volen na týchž skóre, tedy *in-sample*; sens a spec
   jsou proto mírně optimistické. AUC touto volbou dotčená není.
6. Jako kontroly běží permutovaný null (zamíchané nálepky; musí vyjít u 0,5) a netrénovaná
   skóre (výška peaku, peak − vlastní baseline).
7. Primární metrikou je *within-band* AUC, tedy AUC počítaná jen mezi drahami podobné délky
   (12 kvantilových strat, vážení počtem porovnatelných párů). Rozdíl pooled − within-band
   (gap) měří, kolik výkonu nese délka.

## 5. Výsledky

{t}

*Tabulka 1: Logistická regrese ve všech osmi konfiguracích referenčního běhu. Práh podle
Youdenova J na OOF skóre; gap = pooled − within-band AUC.*

![přehled](figL1_logreg_overview.png)

*Obrázek 1: Pooled a within-band AUC logistické regrese po konfiguracích; šedě within-band
AUC permutačního nullu. Tečkovaná čára je úroveň náhody.*

**Diskuze:**

Z tabulky 1 a obrázku 1 plyne čtvero. Za prvé, poolovaná AUC se drží kolem
{cz(intr['auc'], 2)} až {cz(prim['auc'], 2)}, avšak gap je všude +0,15 až +0,19; většinu
poolovaného výkonu tedy nese délka dráhy, kterou si model rekonstruuje ze vstupů. Za druhé,
délkově očištěný signál je malý, ale reálný: within-band AUC {cz(intr['wb'])} až
{cz(prim['wb'])} proti nullu na {cz(prim['null_wb'])}. Za třetí, výsledek je robustní vůči
filtrování; přísnější filtry within-band AUC spíše snižují (odstraňují neúměrně mnoho
produktivních drah), a proto se nefiltruje. Za čtvrté, end-observed korpus dává vyšší čísla
než interior; rozdíl jde z přidaných drah s useknutým začátkem (delší a jasnější), ne
z lepšího modelu.

## 6. Interpretace a meze

**Co čísla říkají.** Dynaminový průběh nese informaci o SI nálepce nad rámec délky dráhy,
avšak malou: dvě náhodně vybrané dráhy stejné délky, jedna produktivní a jedna abortivní,
seřadí model správně asi v 58 % případů (proti 50 % náhody). Sens {cz(prim['sens'], 2)}
a spec {cz(prim['spec'], 2)} u Youdenova prahu popisují tentýž slabý signál v řeči
confusion matice a kvůli in-sample volbě prahu jsou mírně optimistické.

**Co čísla neříkají.** Nejde o měřítko kvality SI ani o dynaminovou referenci pro článek.
Model je trénovaný na SI nálepkách; kdyby se jím SI ověřoval, byl by to kruh. Je to interní
diagnostika, kolik délkově nezávislé informace readout nese. Poolovaná AUC
({cz(prim['auc'], 2)}) se nemá číst jako výkon detektoru; obsahuje +{cz(prim['gap'], 2)}
příspěvku délky.

**Proč je signál tak malý.** Tři důvody se sčítají. Obě nálepky vznikají operátorem „stalo
se to někdy za život", a proto je délka dominantním společným faktorem. Dynamin je na
membráně i difuzně a box-mean jej sbírá včetně okolí, takže část signálu je kontext, ne
jamka. A reference sama je nedokonalá; podle modelu skupiny může přibližně pětina
abortivních drah dynamin legitimně nést, takže ani dokonalý klasifikátor by nedosáhl shody
100 %.

**Vztah k volbě modelu pro článek.** Logistická regrese byla zvolena, protože výkonem
odpovídá XGBoostu i MLP (rozdíly v setinách), má menší délkový únik a její koeficienty jsou
interpretovatelné. Naše nezávislá kontrola s intenzitou z cmeAnalysis dává na srovnatelném
korpusu tentýž obraz; podrobnosti v protokolu detektoru.

## 7. Shrnutí

1. Trénuje se na 27 featurách z průběhu box-mean excessu (okno posledních 20 s, souhrn
   střední fáze, tři skaláry); délka dráhy mezi featurami není, model si ji ale zrekonstruuje.
2. Vyhodnocení je out-of-fold s foldy po filmech; práh Youdenovým J (in-sample); kontrolou
   je permutovaný null a netrénovaná skóre.
3. Poolovaná AUC ≈ {cz(prim['auc'], 2)} je z většiny délka (gap +{cz(prim['gap'], 2)});
   délkově očištěný signál je within-band AUC ≈ {cz(prim['wb'], 2)}, malý, ale nad nullem.
4. Výsledek je robustní vůči filtrům; end-observed čísla zvedá složení korpusu, ne model.
5. Model je interní diagnostika readoutu, ne dynaminová reference pro Figure 3.
"""


TEX_HEAD = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.2cm]{geometry}
\usepackage{fontspec}
\usepackage[czech]{babel}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{float}
\usepackage[hidelinks]{hyperref}
\raggedbottom
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\begin{document}
"""


def tex_escape(s):
    return (s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
             .replace("#", r"\#").replace("κ", r"$\kappa$").replace("α", r"$\alpha$")
             .replace("ρ", r"$\rho$").replace("σ", r"$\sigma$").replace("×", r"$\times$")
             .replace("≥", r"$\geq$").replace("≈", r"$\approx$").replace("⟺", r"$\Leftrightarrow$")
             .replace("→", r"$\rightarrow$").replace("−", "--").replace("|", r"$\mid$"))


def tex_table(rows_data, colspec, header, caption, label):
    body = "\n".join(" & ".join(tex_escape(str(v)) for v in r) + r" \\" for r in rows_data)
    return (f"\\begin{{table}}[H]\\centering\\small\n"
            f"\\begin{{tabular}}{{{colspec}}}\n\\toprule {header} \\\\ \\midrule\n"
            f"{body}\n\\bottomrule\\end{{tabular}}\n"
            f"\\caption{{{tex_escape(caption)}}}\\label{{{label}}}\\end{{table}}")


def build_tex(rows, meta):
    e = tex_escape
    prim = next(r for r in rows if r["tag"] == "none_endobs")
    intr = next(r for r in rows if r["tag"] == "none_interior")
    t = tex_table(table_rows(rows), "p{4.4cm}rrrrrrr",
                  "konfigurace & n drah & podíl SI+ & pooled AUC & wb AUC & gap & sens & spec",
                  "Logistická regrese ve všech osmi konfiguracích referenčního běhu. Práh "
                  "podle Youdenova J na OOF skóre; gap = pooled -- within-band AUC.", "tab:lr")
    fig = ("\\begin{figure}[H]\\centering\\includegraphics[width=\\linewidth]{figL1_logreg_overview.png}"
           "\\caption{Pooled a within-band AUC logistické regrese po konfiguracích; šedě "
           "within-band AUC permutačního nullu. Tečkovaná čára je úroveň náhody.}"
           "\\label{fig:lr}\\end{figure}")
    return TEX_HEAD + f"""
{{\\LARGE\\bfseries Interpretace referenční analýzy:\\\\logistická regrese na box-mean readoutu}}\\\\[4pt]
{{\\small Datum {meta['date']} \\;·\\; CMEpython {meta['git']} \\;·\\; zdroj čísel:
\\texttt{{dynamin\\_v2\\_confusion\\_sweep.json}} větve \\texttt{{release/dynamin-confusion-v1}}}}

\\section*{{1\\; Co je předmětem dokumentu}}
Popisujeme a interpretujeme původní (referenční) analýzu, ve které se logistická regrese učí
predikovat SI nálepku dráhy z~průběhu dynaminové intenzity. Nejprve vysvětlíme, na čem přesně
se trénovalo a co se děje na pozadí. Následně shrneme výsledky a jejich čtení. Naše opakování
téhož experimentu s~intenzitou z~cmeAnalysis zde záměrně neřešíme; je v~samostatném protokolu.

\\section*{{2\\; Data a readout}}
Trénovalo se na korpusu 15 filmů U2OS (2\\,s/snímek, 79{{,}}1\\,nm/px). Každá dráha klatrinové
jamky nese nálepku: \\emph{{produktivní}} $\\Leftrightarrow$ max SI přes život $>$ 0{{,}}7.
Dynaminová intenzita dráhy je tzv. \\emph{{box-mean}} readout, a to průměr 5$\\times$5 pixelů
dynaminového kanálu na pozici jamky v~každém snímku, vydělený biexponenciálním fitem průměrů
snímků dané buňky. Po tomto dělení je 1 rovna průměru buňky; pracuje se s~\\emph{{excessem}}
= hodnota $-$ 1, takže 0 znamená ,,na úrovni průměru buňky``. Poznamenejme dvě vlastnosti
volby. Za prvé, dělení průměrem buňky srovnává filmy s~různým jasem a odstraňuje blednutí.
Za druhé, box-mean sbírá všechen signál v~okénku, tedy i difuzní membránový dynamin
a příspěvky sousedních jamek; lokální pozadí se neodečítá.
\\par
Korpus se vyhodnocuje v~osmi konfiguracích filtrů (tabulka~\\ref{{tab:lr}}): kombinace
\\emph{{cluster}} filtru (žádný, $<$ 3, $<$ 2), vzdálenostního filtru (2.~nejbližší soused
$\\geq$ 5\\,px) a úplnosti dráhy (\\emph{{interior}} = začátek i konec v~záznamu;
\\emph{{end-observed}} = navíc dráhy s~useknutým začátkem). Primární korpus analýzy je
,,bez filtru, end-observed`` (n = {e(fmt_n(prim['n']))}).

\\section*{{3\\; Na čem přesně se model učí: featury}}
Z~každé dráhy se počítá 27 vstupních čísel; informativních je 17, zbylých 10
(\\texttt{{pre\\_*}}, \\texttt{{post\\_*}}) je v~tomto korpusu identicky nulových (vzorky
před vznikem a po zániku dráhy korpus neobsahuje).
\\begin{{itemize}}
\\item Terminální okno (10): excess v~posledních 20\\,s života, převzorkovaný v~reálném čase
na 10 bodů; u~drah kratších než okno se hodnoty před vznikem drží na první naměřené.
\\item Střední fáze života (4): průměr a maximum excessu před oknem, podíl ,,významných``
snímků (excess $>$ 90.~percentil korpusu) a poměr maxima střední fáze k~maximu okna;
u~drah $\\leq$ 10 snímků jsou nulové.
\\item Globální skaláry (3): maximum excessu přes celý život, normalizovaná poloha vrcholu
v~životě a nejdelší souvislý běh významných snímků.
\\end{{itemize}}
Délka dráhy mezi featurami záměrně není. Přesto ji modely ze vstupů rekonstruují (Spearman
$\\rho$ skóre--délka {cz(prim['rho'], 2)}, měřeno na XGBoost skóre téhož běhu), hlavně přes
nulové featury střední fáze u~krátkých drah; to je známý strukturální únik.

\\section*{{4\\; Co se děje na pozadí: trénink a vyhodnocení}}
Postup je pro každou konfiguraci stejný.
\\begin{{enumerate}}
\\item Sestaví se matice drahy $\\times$ 27 featur a vektor SI nálepek.
\\item Featury projdou mediánovou imputací a standardizací (odečtení průměru, dělení
směrodatnou odchylkou).
\\item Natrénuje se logistická regrese s~mírnou L2 regularizací (C = 1, řešič lbfgs).
\\item Vyhodnocení je \\emph{{out-of-fold}}: filmy se rozdělí do 5 skupin, natrénuje se
5 nezávislých modelů, každý na 12 filmech, a každá dráha dostane skóre od modelu, který její
film neviděl. Grupování po filmech brání úniku ,,poznávání buňky``.
\\item Z~agregovaných OOF skóre se zvolí práh maximalizací Youdenova J (sens + spec $-$ 1)
a spočítá confusion matice. Práh je volen na týchž skóre, tedy \\emph{{in-sample}}; sens
a spec jsou proto mírně optimistické. AUC touto volbou dotčená není.
\\item Jako kontroly běží permutovaný null (zamíchané nálepky; musí vyjít u~0{{,}}5)
a netrénovaná skóre (výška peaku, peak $-$ vlastní baseline).
\\item Primární metrikou je \\emph{{within-band}} AUC, tedy AUC počítaná jen mezi drahami
podobné délky (12 kvantilových strat, vážení počtem porovnatelných párů). Rozdíl pooled $-$
within-band (gap) měří, kolik výkonu nese délka.
\\end{{enumerate}}

\\section*{{5\\; Výsledky}}
{t}
{fig}
\\textbf{{Diskuze:}}

Z~tabulky~\\ref{{tab:lr}} a obrázku~\\ref{{fig:lr}} plyne čtvero. Za prvé, poolovaná AUC se
drží kolem {cz(intr['auc'], 2)} až {cz(prim['auc'], 2)}, avšak gap je všude +0{{,}}15 až
+0{{,}}19; většinu poolovaného výkonu tedy nese délka dráhy, kterou si model rekonstruuje ze
vstupů. Za druhé, délkově očištěný signál je malý, ale reálný: within-band AUC
{cz(intr['wb'])} až {cz(prim['wb'])} proti nullu na {cz(prim['null_wb'])}. Za třetí,
výsledek je robustní vůči filtrování; přísnější filtry within-band AUC spíše snižují
(odstraňují neúměrně mnoho produktivních drah), a proto se nefiltruje. Za čtvrté,
end-observed korpus dává vyšší čísla než interior; rozdíl jde z~přidaných drah s~useknutým
začátkem (delší a jasnější), ne z~lepšího modelu.

\\section*{{6\\; Interpretace a meze}}
\\textbf{{Co čísla říkají.}} Dynaminový průběh nese informaci o~SI nálepce nad rámec délky
dráhy, avšak malou: dvě náhodně vybrané dráhy stejné délky, jedna produktivní a jedna
abortivní, seřadí model správně asi v~58\\,\\% případů (proti 50\\,\\% náhody). Sens
{cz(prim['sens'], 2)} a spec {cz(prim['spec'], 2)} u~Youdenova prahu popisují tentýž slabý
signál v~řeči confusion matice a kvůli in-sample volbě prahu jsou mírně optimistické.
\\par
\\textbf{{Co čísla neříkají.}} Nejde o~měřítko kvality SI ani o~dynaminovou referenci pro
článek. Model je trénovaný na SI nálepkách; kdyby se jím SI ověřoval, byl by to kruh. Je to
interní diagnostika, kolik délkově nezávislé informace readout nese. Poolovaná AUC
({cz(prim['auc'], 2)}) se nemá číst jako výkon detektoru; obsahuje +{cz(prim['gap'], 2)}
příspěvku délky.
\\par
\\textbf{{Proč je signál tak malý.}} Tři důvody se sčítají. Obě nálepky vznikají operátorem
,,stalo se to někdy za život``, a proto je délka dominantním společným faktorem. Dynamin je
na membráně i difuzně a box-mean jej sbírá včetně okolí, takže část signálu je kontext, ne
jamka. A reference sama je nedokonalá; podle modelu skupiny může přibližně pětina
abortivních drah dynamin legitimně nést, takže ani dokonalý klasifikátor by nedosáhl shody
100\\,\\%.
\\par
\\textbf{{Vztah k~volbě modelu pro článek.}} Logistická regrese byla zvolena, protože
výkonem odpovídá XGBoostu i MLP (rozdíly v~setinách), má menší délkový únik a její
koeficienty jsou interpretovatelné. Naše nezávislá kontrola s~intenzitou z~cmeAnalysis dává
na srovnatelném korpusu tentýž obraz; podrobnosti v~protokolu detektoru.

\\section*{{7\\; Shrnutí}}
\\begin{{enumerate}}
\\item Trénuje se na 27 featurách z~průběhu box-mean excessu (okno posledních 20\\,s, souhrn
střední fáze, tři skaláry); délka dráhy mezi featurami není, model si ji ale zrekonstruuje.
\\item Vyhodnocení je out-of-fold s~foldy po filmech; práh Youdenovým J (in-sample);
kontrolou je permutovaný null a netrénovaná skóre.
\\item Poolovaná AUC $\\approx$ {cz(prim['auc'], 2)} je z~většiny délka (gap
+{cz(prim['gap'], 2)}); délkově očištěný signál je within-band AUC $\\approx$
{cz(prim['wb'], 2)}, malý, ale nad nullem.
\\item Výsledek je robustní vůči filtrům; end-observed čísla zvedá složení korpusu, ne model.
\\item Model je interní diagnostika readoutu, ne dynaminová reference pro Figure~3.
\\end{{enumerate}}
\\end{{document}}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-json", default=os.path.join(
        ROOT, "external", "Shape2Fate_Fake2Emulate", "dynamin_v2_confusion_sweep.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "CME_for_Helios", "report"))
    ap.add_argument("--skip-pdf", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.sweep_json, encoding="utf-8") as fh:
        cfgs = json.load(fh)["configs"]
    rows = rows_logreg(cfgs)
    for r in rows:
        print(f"{r['lbl']:>28}: n={r['n']:>6,}  AUC {r['auc']:.3f}  wb {r['wb']:.3f}  "
              f"gap +{r['gap']:.2f}  sens {r['sens']:.2f} spec {r['spec']:.2f}")

    meta = {"date": _dt.date.today().isoformat(),
            "git": subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True).stdout.strip() or "?"}
    fig_overview(rows, os.path.join(args.out, "figL1_logreg_overview.png"))
    with open(os.path.join(args.out, "interpretace_logreg.md"), "w", encoding="utf-8") as fh:
        fh.write(build_markdown(rows, meta))
    with open(os.path.join(args.out, "interpretace_logreg.tex"), "w", encoding="utf-8") as fh:
        fh.write(build_tex(rows, meta))

    if not args.skip_pdf and shutil.which("latexmk"):
        r = subprocess.run(["latexmk", "-xelatex", "-interaction=nonstopmode",
                            "-halt-on-error", "-cd", os.path.join(args.out, "interpretace_logreg.tex")],
                           capture_output=True, text=True)
        if r.returncode == 0:
            subprocess.run(["latexmk", "-c", "-cd", os.path.join(args.out, "interpretace_logreg.tex")],
                           capture_output=True, text=True)
            print("PDF: interpretace_logreg.pdf")
        else:
            print("VAROVANI: xelatex selhal; konec logu:")
            print("\n".join(r.stdout.splitlines()[-12:]))
    print(f"hotovo -> {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
