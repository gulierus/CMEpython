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


SCALAR_LABELS = [
    ("mid_mean", "průměr před oknem"),
    ("mid_max", "maximum před oknem"),
    ("mid_frac_sig", "podíl významných před oknem"),
    ("mid_to_win_ratio", "poměr maxim před/okno"),
    ("peak_snr", "maximum přes život"),
    ("phi_peak", "poloha vrcholu v životě"),
    ("longest_run", "nejdelší běh významných"),
]


def fig_coefs(coefs_json, out):
    """Koeficienty rovnocenneho behu (cmeAnalysis amplituda, bez filtru interior):
    vlevo souhrnne vstupy s 95% CI po pasmech, vpravo profil vah terminalniho okna."""
    with open(coefs_json, encoding="utf-8") as fh:
        d = json.load(fh)
    names = d["_meta"]["features"]
    bands = d["configs"]["none_interior"]["bands"]
    band_lbl = [b["band"] for b in bands]
    cmap = plt.get_cmap("viridis")
    cols = [cmap(0.1 + 0.8 * i / max(len(bands) - 1, 1)) for i in range(len(bands))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    for bi, (blk, col) in enumerate(zip(bands, cols)):
        c = np.asarray(blk["coef"], float)
        lo = np.asarray(blk["boot_lo"], float)
        hi = np.asarray(blk["boot_hi"], float)
        for fi, (key, _) in enumerate(SCALAR_LABELS):
            k = names.index(key)
            y = fi + (bi - 1.5) * 0.17
            ax1.errorbar(c[k], y, xerr=[[c[k] - lo[k]], [hi[k] - c[k]]],
                         fmt="o", ms=3.5, color=col, capsize=2, lw=1)
        w = [names.index(f"win_{i}") for i in range(10)]
        t = np.arange(10) * 2.0 - 18.0
        ax2.fill_between(t, lo[w], hi[w], color=col, alpha=0.15, linewidth=0)
        ax2.plot(t, c[w], "o-", ms=3, color=col, label=f"{blk['band']} snímků")
    ax1.axvline(0, color="0.5", lw=1)
    ax1.set_yticks(range(len(SCALAR_LABELS)), [l for _, l in SCALAR_LABELS], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("koeficient [log-odds na 1 SD]")
    ax1.set_title("Souhrnné vstupy", fontsize=10.5)
    ax2.axhline(0, color="0.5", lw=1)
    ax2.set_xlabel("čas před koncem dráhy [s]")
    ax2.set_ylabel("koeficient [log-odds na 1 SD]")
    ax2.set_title("Váhy bodů terminálního okna", fontsize=10.5)
    ax2.legend(fontsize=8, title="délkové pásmo", title_fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_sigmoid(coefs_json, out):
    """Klasicky pohled: sigmoida P(produktivni) podel nejsilnejsiho vstupu
    (prumer pred oknem), ostatni vstupy drzene na prumeru (0 SD)."""
    with open(coefs_json, encoding="utf-8") as fh:
        d = json.load(fh)
    names = d["_meta"]["features"]
    k = names.index("mid_mean")
    bands = d["configs"]["none_interior"]["bands"]
    cmap = plt.get_cmap("viridis")
    cols = [cmap(0.1 + 0.8 * i / max(len(bands) - 1, 1)) for i in range(len(bands))]
    x = np.linspace(-2.0, 3.0, 200)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for blk, col in zip(bands, cols):
        b0 = float(blk["intercept"])
        beta = float(np.asarray(blk["coef"], float)[k])
        p = 1.0 / (1.0 + np.exp(-(b0 + beta * x)))
        ax.plot(x, p, color=col, lw=1.8, label=f"{blk['band']} snímků")
        prev = blk["n_productive"] / blk["n"]
        ax.axhline(prev, color=col, lw=0.7, ls=":", alpha=0.6)
    ax.axvline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("průměr amplitudy před oknem [směrodatné odchylky od průměru]")
    ax.set_ylabel("P(produktivní)")
    ax.set_ylim(0, 1)
    ax.set_title("Model podél nejsilnějšího vstupu; ostatní vstupy na průměru", fontsize=10.5)
    ax.legend(fontsize=8, title="délkové pásmo", title_fontsize=8)
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
predikovat SI štítek dráhy z průběhu dynaminové intenzity. Nejprve vysvětlíme, na čem přesně
se trénovalo a co se děje na pozadí. Následně shrneme výsledky a jejich čtení. Naše opakování
téhož experimentu s intenzitou z cmeAnalysis zde záměrně neřešíme; je v samostatném protokolu.

## 2. Data a readout

Trénovalo se na korpusu 15 filmů U2OS (2 s/snímek, 79,1 nm/px). Každá dráha klatrinové jamky
nese štítek: *produktivní* ⟺ max SI přes život > 0,7. Dynaminová intenzita dráhy je tzv.
*box-mean* readout, a to průměr 5×5 pixelů dynaminového kanálu na pozici jamky v každém
snímku, vydělený biexponenciálním fitem průměrů snímků dané buňky. Slovem *readout*
(odečet) obecně označujeme způsob, jakým se z obrazu získá číslo „kolik dynaminu je na
daném místě v daném snímku"; box-mean je jedna z možností, amplituda PSF fitu
z cmeAnalysis jiná. Po tomto dělení je 1
rovna průměru buňky; pracuje se s *excessem* = hodnota − 1, takže 0 znamená „na úrovni
průměru buňky". Poznamenejme dvě vlastnosti volby. Za prvé, dělení průměrem buňky srovnává
filmy s různým jasem a odstraňuje blednutí. Za druhé, box-mean sbírá všechen signál v okénku,
tedy i difuzní membránový dynamin a příspěvky sousedních jamek; lokální pozadí se neodečítá.
Příklad pro představu: okénko má průměrný jas 1 300 a průměr buňky v tom čase je 1 000;
hodnota readoutu je 1,3 a excess 0,3, tedy „o 30 % nad průměrem buňky".

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

Postup je pro každou konfiguraci stejný; projděme ho krok za krokem.

**Vstupní tabulka.** Představme si tabulku: jeden řádek je jedna dráha, 27 sloupců jsou
čísla z části 3 a vedle nich stojí štítek produktivní/abortivní. Nic jiného model nevidí;
neví, ze kterého filmu dráha pochází, a délku dráhy dostává jen nepřímo.

**Srovnání měřítek (standardizace).** Sloupce mají různé jednotky a rozsahy; průměrný
excess je řádově desetina, nejdelší běh významných snímků desítky. Aby byly souměřitelné,
každý sloupec se přepočte: odečte se jeho průměr a vydělí se směrodatnou odchylkou.
Hodnota +1 pak vždy znamená „o jednu typickou odchylku nad průměrem", ať jde o amplitudu,
podíl, nebo počet. Případné chybějící hodnoty se předtím doplní mediánem sloupce.

**Model.** Logistická regrese je vážený součet. Každé z 27 čísel vynásobí svou vahou,
výsledky sečte a součet převede na pravděpodobnost mezi 0 a 1, že dráha je produktivní.
Trénink hledá váhy, se kterými tyto pravděpodobnosti nejlépe sedí na skutečné štítky.
Mírná regularizace (L2, C = 1) drží váhy malé, aby model nesázel příliš na jednotlivé
sloupce.

**Poctivé vyhodnocení (out-of-fold).** Kdyby se model hodnotil na drahách, na kterých se
učil, vyšla by čísla nadhodnocená. Filmy se proto rozdělí do 5 skupin a natrénuje se
5 nezávislých modelů, každý na 12 filmech. Každá dráha pak dostane skóre od toho modelu,
který její film při tréninku neviděl. Dělí se po celých filmech, ne po drahách; dráhy
z téže buňky jsou si totiž podobné a model by se jinak naučil poznávat buňku místo
biologie.

**Práh a confusion matice.** Skóre je číslo mezi 0 a 1; aby vznikla tabulka
správně/špatně, je třeba zvolit hranici. Youdenovo J volí hranici tam, kde je součet
sensitivity a specificity nejvyšší. Hranice se ovšem vybírá na týchž skóre, která se pak
hodnotí (*in-sample*); sens a spec jsou proto mírně nadhodnocené. AUC žádnou hranici
nepotřebuje, a nadhodnocená tedy není.

**Kontroly.** Vedle modelu běží permutovaný null: tytéž featury, ale náhodně zamíchané
štítky. Musí vyjít u 0,5; kdyby ne, je v postupu únik. A netrénovaná skóre (výška peaku,
peak − vlastní baseline) říkají, co zvládne jedno číslo bez jakéhokoli učení.

**Co je AUC, pooled a within-band.** AUC odpovídá na otázku: vyberme náhodně jednu
produktivní a jednu abortivní dráhu; jak často jim model dá skóre ve správném pořadí?
Hodnota 0,5 znamená házení mincí, 1,0 vždy správně. *Pooled* AUC losuje dvojice ze všech
drah bez ohledu na délku; klidně tedy porovná produktivní dráhu žijící 100 s s abortivní
žijící 12 s. Dlouhé dráhy jsou ale mnohem častěji produktivní, takže i model, který se
naučí jen odhadovat délku, vyhraje většinu takových nesourodých dvojic, aniž by o dynaminu
věděl cokoli. Pooled číslo proto míchá dynaminový signál s „umím poznat délku".
*Within-band* AUC losuje dvojice pouze mezi drahami podobné délky (12 kvantilových strat,
vážení počtem porovnatelných párů); otázka je pak férová, protože oběma drahám ve dvojici
délka pomoci nemůže. Rozdíl pooled − within-band (gap) říká, kolik výkonu nesla délka.
Na číslech primárního korpusu: pooled {cz(prim['auc'], 2)}, within-band
{cz(prim['wb'], 2)}, gap +{cz(prim['gap'], 2)}; ze zdánlivého výkonu tedy
{cz(prim['gap'], 2)} nese délka a nad náhodou zbývá {cz(prim['wb'] - 0.5, 2)} skutečné
dynaminové informace.

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

## 6. Čeho se regrese drží: koeficienty

Přirozená otázka zní, které vstupy model táhnou. U referenčního běhu na ni nelze odpovědět
přímo: větev s výsledky obsahuje výkonnostní čísla, ale hodnoty naučených vah k box-mean
modelu neukládá. K dispozici je ovšem rovnocenná analýza z opakování téhož experimentu
s amplitudou cmeAnalysis, tedy stejné featury, stejný trénink a detektor s výkonem shodným
v setinách. Její koeficienty (standardizované, fitované po délkových pásmech, s 95%
intervaly bootstrapem po filmech) ukazují jednoznačný vzor. Nejsilnější a stabilně kladný
je průměr amplitudy přes část života před terminálním oknem; v pásmu 10–19 snímků jeho
maximum. Váhy jednotlivých bodů terminálního okna jsou malé a většinou s intervalem přes
nulu.
{meta['coef_fig_md']}
**Diskuze:**

Dá se říct, že regrese rozpoznává produktivní dráhy podle trvale vyššího dynaminu během
života, ne podle výrazné události na konci; kdyby scission nesl specifický podpis, ležely
by váhy na konci okna. Přenos tohoto čtení na referenční box-mean model je úsudek, byť
podložený shodou výkonu i vstupů; přímé potvrzení by vyžadovalo doběhnout koeficientový
režim na referenčním korpusu. Podrobný návod, jak standardizované koeficienty číst, uvádí
protokol detektoru.

## 7. Interpretace a meze

**Co čísla říkají.** Dynaminový průběh nese informaci o SI štítku nad rámec délky dráhy,
avšak malou: dvě náhodně vybrané dráhy stejné délky, jedna produktivní a jedna abortivní,
seřadí model správně asi v 58 % případů (proti 50 % náhody). Sens {cz(prim['sens'], 2)}
a spec {cz(prim['spec'], 2)} u Youdenova prahu popisují tentýž slabý signál v řeči
confusion matice a kvůli in-sample volbě prahu jsou mírně optimistické.

**Co čísla neříkají.** Nejde o měřítko kvality SI ani o dynaminovou referenci pro článek.
Model je trénovaný na SI štítcích; kdyby se jím SI ověřoval, byl by to kruh. Je to interní
diagnostika, kolik délkově nezávislé informace readout nese. Poolovaná AUC
({cz(prim['auc'], 2)}) se nemá číst jako výkon detektoru; obsahuje +{cz(prim['gap'], 2)}
příspěvku délky.

**Proč je signál tak malý.** Tři důvody se sčítají. Oba štítky vznikají operátorem „stalo
se to někdy za život", a proto je délka dominantním společným faktorem. Dynamin je na
membráně i difuzně a box-mean jej sbírá včetně okolí, takže část signálu je kontext, ne
jamka. A reference sama je nedokonalá; podle modelu skupiny může přibližně pětina
abortivních drah dynamin legitimně nést, takže ani dokonalý klasifikátor by nedosáhl shody
100 %.

**Vztah k volbě modelu pro článek.** Logistická regrese byla zvolena, protože výkonem
odpovídá XGBoostu i MLP (rozdíly v setinách), má menší délkový únik a její koeficienty jsou
interpretovatelné. Naše nezávislá kontrola s intenzitou z cmeAnalysis dává na srovnatelném
korpusu tentýž obraz; podrobnosti v protokolu detektoru.

## 8. Shrnutí

1. Trénuje se na 27 featurách z průběhu box-mean excessu (okno posledních 20 s, souhrn
   střední fáze, tři skaláry); délka dráhy mezi featurami není, model si ji ale zrekonstruuje.
2. Vyhodnocení je out-of-fold s foldy po filmech; práh Youdenovým J (in-sample); kontrolou
   je permutovaný null a netrénovaná skóre.
3. Poolovaná AUC ≈ {cz(prim['auc'], 2)} je z většiny délka (gap +{cz(prim['gap'], 2)});
   délkově očištěný signál je within-band AUC ≈ {cz(prim['wb'], 2)}, malý, ale nad nullem.
4. Výsledek je robustní vůči filtrům; end-observed čísla zvedá složení korpusu, ne model.
5. Koeficienty rovnocenné analýzy ukazují, že regrese stojí na trvale zvýšeném dynaminu
   během života, ne na terminální události; referenční běh vlastní váhy neukládá.
6. Model je interní diagnostika readoutu, ne dynaminová reference pro Figure 3.
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
predikovat SI štítek dráhy z~průběhu dynaminové intenzity. Nejprve vysvětlíme, na čem přesně
se trénovalo a co se děje na pozadí. Následně shrneme výsledky a jejich čtení. Naše opakování
téhož experimentu s~intenzitou z~cmeAnalysis zde záměrně neřešíme; je v~samostatném protokolu.

\\section*{{2\\; Data a readout}}
Trénovalo se na korpusu 15 filmů U2OS (2\\,s/snímek, 79{{,}}1\\,nm/px). Každá dráha klatrinové
jamky nese štítek: \\emph{{produktivní}} $\\Leftrightarrow$ max SI přes život $>$ 0{{,}}7.
Dynaminová intenzita dráhy je tzv. \\emph{{box-mean}} readout, a to průměr 5$\\times$5 pixelů
dynaminového kanálu na pozici jamky v~každém snímku, vydělený biexponenciálním fitem průměrů
snímků dané buňky. Slovem \\emph{{readout}} (odečet) obecně označujeme způsob, jakým se
z~obrazu získá číslo ,,kolik dynaminu je na daném místě v~daném snímku``; box-mean je jedna
z~možností, amplituda PSF fitu z~cmeAnalysis jiná. Po tomto dělení je 1 rovna průměru
buňky; pracuje se s~\\emph{{excessem}}
= hodnota $-$ 1, takže 0 znamená ,,na úrovni průměru buňky``. Poznamenejme dvě vlastnosti
volby. Za prvé, dělení průměrem buňky srovnává filmy s~různým jasem a odstraňuje blednutí.
Za druhé, box-mean sbírá všechen signál v~okénku, tedy i difuzní membránový dynamin
a příspěvky sousedních jamek; lokální pozadí se neodečítá. Příklad pro představu: okénko má
průměrný jas 1\\,300 a průměr buňky v~tom čase je 1\\,000; hodnota readoutu je 1{{,}}3
a excess 0{{,}}3, tedy ,,o~30\\,\\% nad průměrem buňky``.
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
Postup je pro každou konfiguraci stejný; projděme ho krok za krokem.
\\par
\\textbf{{Vstupní tabulka.}} Představme si tabulku: jeden řádek je jedna dráha, 27 sloupců
jsou čísla z~části~3 a vedle nich stojí štítek produktivní/abortivní. Nic jiného model
nevidí; neví, ze kterého filmu dráha pochází, a délku dráhy dostává jen nepřímo.
\\par
\\textbf{{Srovnání měřítek (standardizace).}} Sloupce mají různé jednotky a rozsahy;
průměrný excess je řádově desetina, nejdelší běh významných snímků desítky. Aby byly
souměřitelné, každý sloupec se přepočte: odečte se jeho průměr a vydělí se směrodatnou
odchylkou. Hodnota +1 pak vždy znamená ,,o~jednu typickou odchylku nad průměrem``, ať jde
o~amplitudu, podíl, nebo počet. Případné chybějící hodnoty se předtím doplní mediánem
sloupce.
\\par
\\textbf{{Model.}} Logistická regrese je vážený součet. Každé z~27 čísel vynásobí svou
vahou, výsledky sečte a součet převede na pravděpodobnost mezi 0 a 1, že dráha je
produktivní. Trénink hledá váhy, se kterými tyto pravděpodobnosti nejlépe sedí na skutečné
štítky. Mírná regularizace (L2, C = 1) drží váhy malé, aby model nesázel příliš na
jednotlivé sloupce.
\\par
\\textbf{{Poctivé vyhodnocení (out-of-fold).}} Kdyby se model hodnotil na drahách, na
kterých se učil, vyšla by čísla nadhodnocená. Filmy se proto rozdělí do 5 skupin a natrénuje
se 5 nezávislých modelů, každý na 12 filmech. Každá dráha pak dostane skóre od toho modelu,
který její film při tréninku neviděl. Dělí se po celých filmech, ne po drahách; dráhy
z~téže buňky jsou si totiž podobné a model by se jinak naučil poznávat buňku místo
biologie.
\\par
\\textbf{{Práh a confusion matice.}} Skóre je číslo mezi 0 a 1; aby vznikla tabulka
správně/špatně, je třeba zvolit hranici. Youdenovo J volí hranici tam, kde je součet
sensitivity a specificity nejvyšší. Hranice se ovšem vybírá na týchž skóre, která se pak
hodnotí (\\emph{{in-sample}}); sens a spec jsou proto mírně nadhodnocené. AUC žádnou hranici
nepotřebuje, a nadhodnocená tedy není.
\\par
\\textbf{{Kontroly.}} Vedle modelu běží permutovaný null: tytéž featury, ale náhodně
zamíchané štítky. Musí vyjít u~0{{,}}5; kdyby ne, je v~postupu únik. A netrénovaná skóre
(výška peaku, peak $-$ vlastní baseline) říkají, co zvládne jedno číslo bez jakéhokoli
učení.
\\par
\\textbf{{Co je AUC, pooled a within-band.}} AUC odpovídá na otázku: vyberme náhodně jednu
produktivní a jednu abortivní dráhu; jak často jim model dá skóre ve správném pořadí?
Hodnota 0{{,}}5 znamená házení mincí, 1{{,}}0 vždy správně. \\emph{{Pooled}} AUC losuje
dvojice ze všech drah bez ohledu na délku; klidně tedy porovná produktivní dráhu žijící
100\\,s s~abortivní žijící 12\\,s. Dlouhé dráhy jsou ale mnohem častěji produktivní, takže
i model, který se naučí jen odhadovat délku, vyhraje většinu takových nesourodých dvojic,
aniž by o~dynaminu věděl cokoli. Pooled číslo proto míchá dynaminový signál s~,,umím poznat
délku``. \\emph{{Within-band}} AUC losuje dvojice pouze mezi drahami podobné délky
(12 kvantilových strat, vážení počtem porovnatelných párů); otázka je pak férová, protože
oběma drahám ve dvojici délka pomoci nemůže. Rozdíl pooled $-$ within-band (gap) říká,
kolik výkonu nesla délka. Na číslech primárního korpusu: pooled {cz(prim['auc'], 2)},
within-band {cz(prim['wb'], 2)}, gap +{cz(prim['gap'], 2)}; ze zdánlivého výkonu tedy
{cz(prim['gap'], 2)} nese délka a nad náhodou zbývá {cz(prim['wb'] - 0.5, 2)} skutečné
dynaminové informace.

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

\\section*{{6\\; Čeho se regrese drží: koeficienty}}
Přirozená otázka zní, které vstupy model táhnou. U~referenčního běhu na ni nelze odpovědět
přímo: větev s~výsledky obsahuje výkonnostní čísla, ale hodnoty naučených vah k~box-mean
modelu neukládá. K~dispozici je ovšem rovnocenná analýza z~opakování téhož experimentu
s~amplitudou cmeAnalysis, tedy stejné featury, stejný trénink a detektor s~výkonem shodným
v~setinách. Její koeficienty (standardizované, fitované po délkových pásmech, s~95\\,\\%
intervaly bootstrapem po filmech) ukazují jednoznačný vzor. Nejsilnější a stabilně kladný
je průměr amplitudy přes část života před terminálním oknem; v~pásmu 10--19 snímků jeho
maximum. Váhy jednotlivých bodů terminálního okna jsou malé a většinou s~intervalem přes
nulu.
{meta['coef_fig_tex']}
\\textbf{{Diskuze:}}

Dá se říct, že regrese rozpoznává produktivní dráhy podle trvale vyššího dynaminu během
života, ne podle výrazné události na konci; kdyby scission nesl specifický podpis, ležely
by váhy na konci okna. Přenos tohoto čtení na referenční box-mean model je úsudek, byť
podložený shodou výkonu i vstupů; přímé potvrzení by vyžadovalo doběhnout koeficientový
režim na referenčním korpusu. Podrobný návod, jak standardizované koeficienty číst, uvádí
protokol detektoru.

\\section*{{7\\; Interpretace a meze}}
\\textbf{{Co čísla říkají.}} Dynaminový průběh nese informaci o~SI štítku nad rámec délky
dráhy, avšak malou: dvě náhodně vybrané dráhy stejné délky, jedna produktivní a jedna
abortivní, seřadí model správně asi v~58\\,\\% případů (proti 50\\,\\% náhody). Sens
{cz(prim['sens'], 2)} a spec {cz(prim['spec'], 2)} u~Youdenova prahu popisují tentýž slabý
signál v~řeči confusion matice a kvůli in-sample volbě prahu jsou mírně optimistické.
\\par
\\textbf{{Co čísla neříkají.}} Nejde o~měřítko kvality SI ani o~dynaminovou referenci pro
článek. Model je trénovaný na SI štítcích; kdyby se jím SI ověřoval, byl by to kruh. Je to
interní diagnostika, kolik délkově nezávislé informace readout nese. Poolovaná AUC
({cz(prim['auc'], 2)}) se nemá číst jako výkon detektoru; obsahuje +{cz(prim['gap'], 2)}
příspěvku délky.
\\par
\\textbf{{Proč je signál tak malý.}} Tři důvody se sčítají. Oba štítky vznikají operátorem
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

\\section*{{8\\; Shrnutí}}
\\begin{{enumerate}}
\\item Trénuje se na 27 featurách z~průběhu box-mean excessu (okno posledních 20\\,s, souhrn
střední fáze, tři skaláry); délka dráhy mezi featurami není, model si ji ale zrekonstruuje.
\\item Vyhodnocení je out-of-fold s~foldy po filmech; práh Youdenovým J (in-sample);
kontrolou je permutovaný null a netrénovaná skóre.
\\item Poolovaná AUC $\\approx$ {cz(prim['auc'], 2)} je z~většiny délka (gap
+{cz(prim['gap'], 2)}); délkově očištěný signál je within-band AUC $\\approx$
{cz(prim['wb'], 2)}, malý, ale nad nullem.
\\item Výsledek je robustní vůči filtrům; end-observed čísla zvedá složení korpusu, ne model.
\\item Koeficienty rovnocenné analýzy ukazují, že regrese stojí na trvale zvýšeném dynaminu
během života, ne na terminální události; referenční běh vlastní váhy neukládá.
\\item Model je interní diagnostika readoutu, ne dynaminová reference pro Figure~3.
\\end{{enumerate}}
\\end{{document}}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-json", default=os.path.join(
        ROOT, "external", "Shape2Fate_Fake2Emulate", "dynamin_v2_confusion_sweep.json"))
    ap.add_argument("--coefs-json", default=os.path.join(
        ROOT, "CME_for_Helios", "runs", "raw_238490", "dynamin_v2_logreg_coefs.json"))
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
    meta["coef_fig_md"] = meta["coef_fig_tex"] = ""
    if os.path.exists(args.coefs_json):
        fig_coefs(args.coefs_json, os.path.join(args.out, "figL2_logreg_coefs.png"))
        cap = ("Koeficienty rovnocenného běhu na amplitudě cmeAnalysis (korpus bez filtru, "
               "interior; referenční běh váhy neukládá). Vlevo souhrnné vstupy s 95% "
               "intervaly bootstrapem po filmech, barvy rozlišují délková pásma; hodnoty "
               "vpravo od nuly táhnou k „produktivní\". Vpravo váhy jednotlivých bodů "
               "terminálního okna s pásy 95% intervalů.")
        fig_sigmoid(args.coefs_json, os.path.join(args.out, "figL3_logreg_sigmoid.png"))
        cap3 = ("Klasický pohled na tutéž regresi: pravděpodobnost „produktivní\" podél "
                "nejsilnějšího vstupu (průměr amplitudy před oknem), ostatní vstupy držené "
                "na průměru. Plné čáry jsou esovky modelu po délkových pásmech, tečkované "
                "vodorovné čáry podíl produktivních v pásmu; kde esovka protíná svislou "
                "tečkovanou čáru (průměrná dráha), model vrací přibližně tento podíl. "
                "Vícerozměrný model esovku má, jen ji lze kreslit vždy pro jeden vstup.")
        meta["coef_fig_md"] = (f"\n![koeficienty]({'figL2_logreg_coefs.png'})\n\n"
                               f"*Obrázek 2: {cap}*\n"
                               f"\n![sigmoida]({'figL3_logreg_sigmoid.png'})\n\n"
                               f"*Obrázek 3: {cap3}*\n")
        meta["coef_fig_tex"] = (
            "\\begin{figure}[H]\\centering"
            "\\includegraphics[width=\\linewidth]{figL2_logreg_coefs.png}"
            "\\caption{Koeficienty rovnocenného běhu na amplitudě cmeAnalysis (korpus bez "
            "filtru, interior; referenční běh váhy neukládá). Vlevo souhrnné vstupy s~95\\,\\% "
            "intervaly bootstrapem po filmech, barvy rozlišují délková pásma; hodnoty vpravo "
            "od nuly táhnou k~,,produktivní``. Vpravo váhy jednotlivých bodů terminálního "
            "okna s~pásy 95\\,\\% intervalů.}\\label{fig:coef}\\end{figure}\n"
            "\\begin{figure}[H]\\centering"
            "\\includegraphics[width=0.72\\linewidth]{figL3_logreg_sigmoid.png}"
            "\\caption{Klasický pohled na tutéž regresi: pravděpodobnost ,,produktivní`` "
            "podél nejsilnějšího vstupu (průměr amplitudy před oknem), ostatní vstupy držené "
            "na průměru. Plné čáry jsou esovky modelu po délkových pásmech, tečkované "
            "vodorovné čáry podíl produktivních v~pásmu; kde esovka protíná svislou "
            "tečkovanou čáru (průměrná dráha), model vrací přibližně tento podíl. "
            "Vícerozměrný model esovku má, jen ji lze kreslit vždy pro jeden vstup.}"
            "\\label{fig:sig}\\end{figure}\n\\par")
    else:
        print(f"VAROVANI: {args.coefs_json} nenalezen -- obrazek koeficientu vynechan")
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
