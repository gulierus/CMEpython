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


def fig_features(out):
    """Schema vstupu modelu na smyslene draze: okno, stredni faze, prah,
    vrchol, beh vyznamnych snimku."""
    dt = 2.0
    n = 25                                   # 25 snimku = 50 s
    t = np.arange(n) * dt
    a = (0.15 + 0.55 * np.exp(-0.5 * ((t - 16) / 6) ** 2)
         + 1.00 * np.exp(-0.5 * ((t - 36) / 5) ** 2)
         + 0.05 * np.sin(t / 3.0))
    thr = 0.45
    win0 = n - 10                            # poslednich 10 snimku = 20 s
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax.axvspan(t[win0] - dt / 2, t[-1] + dt / 2, color="#2b7bba", alpha=0.10)
    ax.axvspan(t[0] - dt / 2, t[win0] - dt / 2, color="#e07a29", alpha=0.08)
    ax.plot(t, a, "-", color="0.35", lw=1.2)
    ax.plot(t[win0:], a[win0:], "o", ms=5, color="#2b7bba")
    ax.plot(t[:win0], a[:win0], ".", ms=4, color="#e07a29")
    # prah vyznamnosti
    ax.axhline(thr, color="0.5", lw=1, ls="--")
    ax.text(t[0], thr + 0.03, "práh významnosti (q90 korpusu)", fontsize=8, color="0.35")
    # prumer stredni faze
    mid_mean = a[:win0].mean()
    ax.hlines(mid_mean, t[0], t[win0 - 1], color="#b3541e", lw=1.6, ls=":")
    ax.annotate("průměr před oknem", (t[3], mid_mean), (t[1], mid_mean + 0.22),
                fontsize=8, color="#b3541e", arrowprops=dict(arrowstyle="->", color="#b3541e", lw=0.8))
    # maximum stredni faze
    km = int(np.argmax(a[:win0]))
    ax.plot(t[km], a[km], "^", ms=9, color="#b3541e")
    ax.annotate("maximum před oknem", (t[km], a[km]), (t[km] - 12, a[km] + 0.18),
                fontsize=8, color="#b3541e", arrowprops=dict(arrowstyle="->", color="#b3541e", lw=0.8))
    # globalni vrchol
    kp = int(np.argmax(a))
    ax.plot(t[kp], a[kp], "*", ms=14, color="#7d3c98")
    ax.annotate("maximum přes život", (t[kp] + 0.5, a[kp] - 0.02), (t[kp] + 5, a[kp] - 0.25),
                fontsize=8, color="#7d3c98", arrowprops=dict(arrowstyle="->", color="#7d3c98", lw=0.8))
    ax.annotate("", (t[kp], 0.02), (t[0], 0.02),
                arrowprops=dict(arrowstyle="->", color="#7d3c98", lw=0.9))
    ax.text(t[kp] / 2, 0.05, f"poloha vrcholu v životě = {t[kp] / t[-1]:.0%}",
            fontsize=8, color="#7d3c98", ha="center")
    # nejdelsi beh vyznamnych
    sig = a > thr
    best_len, best_s, cur_s = 0, 0, None
    for i, v in enumerate(np.append(sig, False)):
        if v and cur_s is None:
            cur_s = i
        if not v and cur_s is not None:
            if i - cur_s > best_len:
                best_len, best_s = i - cur_s, cur_s
            cur_s = None
    y_br = thr - 0.10
    ax.annotate("", (t[best_s + best_len - 1], y_br), (t[best_s], y_br),
                arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.9))
    ax.text((t[best_s] + t[best_s + best_len - 1]) / 2, y_br - 0.09,
            "nejdelší běh významných snímků", fontsize=8, color="0.3", ha="center")
    # popisky pasem
    ax.text((t[0] + t[win0]) / 2, 1.13, "střední fáze → 3 souhrnná čísla",
            fontsize=9, color="#b3541e", ha="center")
    ax.text((t[win0] + t[-1]) / 2, 1.13, "terminální okno → 10 bodů\n(win_0 … win_9)",
            fontsize=9, color="#2b7bba", ha="center")
    ax.set_xlabel("čas od vzniku dráhy [s]")
    ax.set_ylabel("excess dynaminu")
    ax.set_ylim(-0.05, 1.28)
    ax.set_title("Vstupy modelu na příkladu jedné dráhy", fontsize=10.5)
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

Vykládáme referenční analýzu. Logistická regrese se v ní učí poznat SI štítek dráhy
z průběhu dynaminové intenzity. Výklad jde od dat přes trénink k výsledkům a jejich mezím.
Naše opakování téhož experimentu s intenzitou z cmeAnalysis má samostatný protokol a zde
je neřešíme.

## 2. Data a readout

Začněme tím, co je jedna položka dat. Je to dráha klatrinové jamky v jednom z 15 filmů
U2OS (2 s na snímek, 79,1 nm na pixel). Každá dráha nese štítek. Produktivní je tehdy,
když její Shape Index někdy za život překročí 0,7. Jinak je abortivní.

K dráze patří průběh dynaminové intenzity, jedno číslo na snímek. Dodává je tzv.
*box-mean* readout. Slovo *readout* (odečet) obecně znamená předpis, jak z obrazu získat
číslo „kolik dynaminu je na tomto místě v tomto snímku". Box-mean je předpis nejprostší
možný. Vezmi průměr okénka 5×5 pixelů na pozici jamky a vyděl ho jasem celé buňky v tom
čase. Jas buňky se přitom bere jako hladký (biexponenciální) fit průměrů
snímků, aby ho nerozkmitával šum. Po dělení znamená 1 přesně průměr buňky. Pracuje se s *excessem*, tedy hodnotou
minus 1. Příklad: okénko má průměrný jas 1 300 a buňka 1 000. Readout je 1,3, excess 0,3.
Čteme: o 30 % nad průměrem buňky.

Dvě vlastnosti této volby si zapamatujme. Dělení jasem buňky srovnává filmy s různým jasem
a ruší blednutí. To je silná stránka. A průměr okénka sbírá všechen signál uvnitř, tedy
i difuzní dynamin membrány a příspěvky sousedních jamek. Lokální pozadí se neodečítá.

Analýza běží v osmi konfiguracích filtrů (tabulka 1). Kombinují se *cluster* filtr
(žádný, < 3, < 2), vzdálenostní filtr (2. nejbližší soused ≥ 5 px) a úplnost dráhy.
*Interior* znamená začátek i konec v záznamu. *End-observed* přidává dráhy s useknutým
začátkem. Primární korpus analýzy je „bez filtru, end-observed" (n = {fmt_n(prim['n'])}).

## 3. Na čem přesně se model učí: featury

Featura je jedno číslo spočítané z průběhu dráhy. Dohromady jich je 27 a tvoří jeden řádek
vstupní tabulky. Nejlépe se chápou na příkladu. Obrázek 1 ukazuje smyšlenou dráhu žijící
50 s. Její život dělíme na dvě části. Posledních 20 s je terminální okno, modrý pás.
Všechno před tím je střední fáze, oranžový pás.

![featury](figL0_features.png)

*Obrázek 1: Vstupy modelu na příkladu jedné dráhy. Modrý pás je terminální okno. Jeho
průběh model vidí bod po bodu, deset teček win_0 až win_9. Oranžový pás je střední fáze.
Tu model vidí jen jako tři souhrny: průměr (tečkovaná úsečka), maximum (trojúhelník)
a podíl snímků nad prahem významnosti. Práh je čárkovaná vodorovná čára, 90. percentil
excessu přes celý korpus. Hvězda je maximum přes celý život. Fialová šipka dole ukazuje
polohu vrcholu v životě. Svorka vyznačuje nejdelší nepřerušený běh nadprahových snímků.*

Po skupinách:

- **Terminální okno, 10 čísel.** Průběh excessu v posledních 20 s, převzorkovaný na
  10 bodů po 2 s. Jediná část života, kterou model vidí bod po bodu. U drah kratších než
  okno se body před vznikem drží na první naměřené hodnotě.
- **Souhrny střední fáze, 4 čísla.** Průměr, maximum, podíl významných snímků a poměr
  maxima střední fáze k maximu okna. U drah do 10 snímků střední fáze neexistuje. Všechna
  čtyři čísla jsou pak nula a právě tudy model pozná krátkou dráhu.
- **Skaláry přes celý život, 3 čísla.** Maximum excessu. Poloha vrcholu v životě, kde 0 je
  vznik a 1 zánik. Nejdelší souvislý běh významných snímků.
- **Mrtvé vstupy, 10 čísel.** Sloty pre a post jsou vyhrazené pro vzorky před vznikem
  a po zániku dráhy. Korpus je neobsahuje. Jsou proto vždy nula a model je ignoruje.

Délka dráhy mezi featurami záměrně není. Přesto si ji modely ze vstupů zrekonstruují,
a to čtyřmi cestami. Za prvé, dráha do 10 snímků nemá střední fázi, takže její čtyři
souhrny jsou přesně nula. Čtyři nuly vedle sebe jsou spolehlivý prapor krátké dráhy.
Za druhé, u dráhy kratší než okno se body před vznikem drží na první naměřené hodnotě.
Počet stejných bodů na začátku okna tak skoro přímo odečítá, kolik snímků dráze chybí.
Za třetí, nejdelší běh významných snímků nemůže být delší než dráha sama. Horní mez této
featury je délka. A za čtvrté, maxima rostou s počtem pokusů. Maximum z 60 snímků je
i u čistého šumu vyšší než maximum z 8 snímků. Model tyto stopy složí dohromady a délku
fakticky zná (Spearman ρ skóre a délky {cz(prim['rho'], 2)}, měřeno na XGBoost skóre
téhož běhu). To je známý strukturální únik. Právě jeho výhodu ruší within-band
vyhodnocení v části 4.

## 4. Co se děje na pozadí: trénink a vyhodnocení

Postup je pro každou konfiguraci stejný. Projděme ho v pořadí, v jakém běží.

**Vstupní tabulka.** Všechno, co model kdy uvidí, je jedna tabulka. Řádek je dráha.
Sloupců je 27, jsou to čísla z části 3, a vedle nich stojí štítek. Nic víc. Model neví,
z jakého filmu dráha pochází, a délku dostává jen nepřímo.

**Srovnání měřítek (standardizace).** Sloupce mají různé jednotky a rozsahy. Průměrný
excess je řádově desetina, nejdelší běh významných snímků desítky. Proto se každý sloupec
přepočte: odečte se jeho průměr a vydělí se směrodatnou odchylkou. Hodnota +1 pak vždy
znamená totéž, o jednu typickou odchylku nad průměrem. Chybějící hodnoty se předtím doplní
mediánem sloupce.

**Model.** Logistická regrese je nejprostší myslitelný model: vážený součet. Každé z 27
čísel vynásobí svou vahou, sečte a součet převede na pravděpodobnost mezi 0 a 1. Trénink
jen hledá váhy, se kterými pravděpodobnosti nejlépe sedí na skutečné štítky. Mírná
regularizace (L2, C = 1) drží váhy malé, aby model nesázel příliš na jediný sloupec.

**Poctivé vyhodnocení (out-of-fold).** Zkouška z látky, kterou student právě viděl, nic
neměří. Proto se filmy rozdělí do 5 skupin a natrénuje se 5 nezávislých modelů, každý na
12 filmech. Každá dráha pak dostane skóre od modelu, který její film nikdy neviděl. Dělí
se po celých filmech, ne po drahách. Dráhy z jedné buňky jsou si totiž podobné a model by
se jinak naučil poznávat buňku, ne biologii.

**Práh a confusion matice.** Skóre je číslo mezi 0 a 1. Tabulka správně/špatně potřebuje
hranici. Youdenovo J ji položí tam, kde je součet sensitivity a specificity nejvyšší.
Jedna poctivá poznámka: hranice se vybírá na týchž skóre, která potom hodnotíme
(*in-sample*). Sens a spec proto vycházejí o něco lichotivěji, než by vyšly na nových
datech. AUC hranici nepotřebuje a tato výtka se jí netýká.

**Kontroly.** Takové analýze se věří až po kontrolách. První je permutovaný null: tytéž
featury, náhodně zamíchané štítky. Musí vyjít u 0,5. Kdyby ne, je v postupu únik. Druhou
kontrolou jsou netrénovaná skóre, výška peaku a peak minus vlastní baseline. Ta říkají,
co dokáže jediné číslo bez jakéhokoli učení.

**Co je AUC, pooled a within-band.** AUC je jednoduchá hra. Vylosujeme jednu produktivní
a jednu abortivní dráhu a zeptáme se, jestli jim model dal skóre ve správném pořadí.
Podíl správných pořadí je AUC. Hodnota 0,5 je mince, 1,0 neomylnost. *Pooled* AUC losuje
ze všech drah bez ohledu na délku. A přesně tady vzniká nafouknutí. Kdo hází kostkou
dvacetkrát, hodí šestku téměř jistě. Kdo jen třikrát, spíše ne. Oba štítky jsou soutěže
typu „povedlo se to někdy za život", takže s délkou rostou samy od sebe. Podíl SI+ stoupá
napříč délkovými pásmy z 9 % na 68 % a podíl dynamin+ z 24 % na 95 % (tytéž filmy,
protokol porovnání klasifikací). Model, který umí jen odhadnout délku, proto vyhrává
většinu nesourodých dvojic, aniž o dynaminu cokoli ví. *Within-band* AUC tu výhodu ruší.
Losuje jen dvojice drah podobné délky (4 délková pásma: 6–9, 10–19, 20–39 a 40+ snímků, vážení počtem dvojic).
A *gap* je prosté odečtení: pooled minus within-band. Je to přesně ten kus výkonu, který
zmizí, jakmile modelu délku vezmeme. Na číslech primárního korpusu: pooled
{cz(prim['auc'], 2)}, within-band {cz(prim['wb'], 2)}, gap +{cz(prim['gap'], 2)}.
Z výkonu {cz(prim['auc'], 2)} tedy {cz(prim['gap'], 2)} dodala délka. Nad náhodou zbývá
{cz(prim['wb'] - 0.5, 2)} skutečné dynaminové informace.

## 5. Výsledky

{t}

*Tabulka 1: Logistická regrese ve všech osmi konfiguracích referenčního běhu. Práh podle
Youdenova J na OOF skóre. Gap = pooled − within-band AUC.*

![přehled](figL1_logreg_overview.png)

*Obrázek 2: Pooled a within-band AUC logistické regrese po konfiguracích. Šedě within-band
AUC permutačního nullu. Tečkovaná čára je úroveň náhody.*

**Diskuze:**

Z tabulky 1 a obrázku 2 si odnesme čtyři věci. Za prvé, poolovaná AUC vypadá slušně, drží
se kolem {cz(intr['auc'], 2)} až {cz(prim['auc'], 2)}. Jenže gap je všude +0,15 až +0,20.
Většinu toho výkonu tedy nese délka. Za druhé,
délkově očištěný signál existuje, ale jen u nefiltrovaných korpusů. Jejich within-band
AUC {cz(intr['wb'])} až {cz(prim['wb'])} leží nad nullem na {cz(prim['null_wb'])}.
Za třetí, filtrování škodí. Přísnější filtry odstraňují neúměrně mnoho produktivních
drah a within-band AUC srážejí až na úroveň nullu (cluster < 2: 0,49). Proto se
nefiltruje. A za čtvrté, end-observed čísla jsou vyšší než interior. Vyšší
je ale korpus, ne model. Přidané dráhy s useknutým začátkem jsou delší a jasnější.

## 6. Čeho se regrese drží: koeficienty

Přirozená otázka zní, které vstupy model táhnou. U referenčního běhu na ni nelze odpovědět
přímo. Větev s výsledky obsahuje výkonnostní čísla, ale hodnoty naučených vah k box-mean
modelu neukládá. Máme ovšem rovnocennou analýzu z opakování téhož experimentu s amplitudou
cmeAnalysis. Stejné featury, stejný trénink, výkon shodný v setinách. Její koeficienty
ukazují jednoznačný vzor. Nejsilnější a stabilně kladný je průměr amplitudy přes část
života před terminálním oknem. V pásmu 10–19 snímků jeho maximum. Váhy jednotlivých bodů
terminálního okna jsou malé a většinou s intervalem přes nulu.
{meta['coef_fig_md']}
**Diskuze:**

Dá se říct, že regrese rozpoznává produktivní dráhy podle trvale vyššího dynaminu během
života, ne podle výrazné události na konci. Kdyby scission nesl specifický podpis, ležely
by váhy na konci okna. Přenos tohoto čtení na referenční box-mean model je úsudek, byť
podložený shodou výkonu i vstupů. Přímé potvrzení by vyžadovalo doběhnout koeficientový
režim na referenčním korpusu. Podrobný návod, jak standardizované koeficienty číst, uvádí
protokol detektoru.

## 7. Interpretace a meze

**Co čísla říkají.** Představme si dvě dráhy stejné délky, jednu produktivní a jednu
abortivní. Model je seřadí správně asi v 58 % případů. Náhoda by dala 50 %. To je celý
dynaminový signál: reálný, ale malý. Sens {cz(prim['sens'], 2)} a spec
{cz(prim['spec'], 2)} u Youdenova prahu jsou týž signál v řeči confusion matice, mírně
nadhodnocený volbou prahu.

**Co čísla neříkají.** Nejsou to známky kvality SI ani dynaminová reference pro článek.
Model je trénovaný na SI štítcích. Kdyby se jím SI ověřoval, ověřovali bychom kruhem. Je
to interní diagnostika, kolik délkově nezávislé informace readout nese. A poolovaná AUC
({cz(prim['auc'], 2)}) se nemá citovat jako výkon detektoru. Obsahuje +{cz(prim['gap'], 2)}
příspěvku délky.

**Proč je signál tak malý.** Tři důvody, a sčítají se. Oba štítky vznikají operátorem
„stalo se to někdy za život", takže délka je dominantní společný faktor. Dynamin je na
membráně i difuzně a box-mean jej sbírá včetně okolí. Část signálu je proto kontext, ne
jamka. A reference sama je nedokonalá. Podle modelu skupiny může přibližně pětina
abortivních drah dynamin legitimně nést. Ani dokonalý klasifikátor by proto nedosáhl
shody 100 %.

**Vztah k volbě modelu pro článek.** Logistická regrese byla zvolena, protože výkonem
odpovídá XGBoostu i MLP (rozdíly v setinách), má menší délkový únik a její koeficienty
jsou interpretovatelné. Naše nezávislá kontrola s intenzitou z cmeAnalysis dává na
srovnatelném korpusu tentýž obraz. Podrobnosti uvádí protokol detektoru.

## 8. Shrnutí

1. Trénuje se na 27 featurách z průběhu box-mean excessu (okno posledních 20 s, souhrn
   střední fáze, tři skaláry). Délka dráhy mezi featurami není, model si ji ale zrekonstruuje.
2. Vyhodnocení je out-of-fold s foldy po filmech. Práh se volí Youdenovým J, in-sample.
   Kontrolou je permutovaný null a netrénovaná skóre.
3. Poolovaná AUC ≈ {cz(prim['auc'], 2)} je z většiny délka (gap +{cz(prim['gap'], 2)}).
   Délkově očištěný signál je within-band AUC ≈ {cz(prim['wb'], 2)}, malý, ale nad nullem.
4. Filtry výsledek nezlepšují, přísné ho srážejí k nullu. End-observed čísla zvedá
   složení korpusu, ne model.
5. Koeficienty rovnocenné analýzy ukazují, že regrese stojí na trvale zvýšeném dynaminu
   během života, ne na terminální události. Referenční běh vlastní váhy neukládá.
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
                  "podle Youdenova J na OOF skóre. Gap = pooled -- within-band AUC.", "tab:lr")
    fig = ("\\begin{figure}[H]\\centering\\includegraphics[width=\\linewidth]{figL1_logreg_overview.png}"
           "\\caption{Pooled a within-band AUC logistické regrese po konfiguracích. Šedě "
           "within-band AUC permutačního nullu. Tečkovaná čára je úroveň náhody.}"
           "\\label{fig:lr}\\end{figure}")
    return TEX_HEAD + f"""
{{\\LARGE\\bfseries Interpretace referenční analýzy:\\\\logistická regrese na box-mean readoutu}}\\\\[4pt]
{{\\small Datum {meta['date']} \;·\; CMEpython {meta['git']} \\\\ zdroj čísel:
\\texttt{{dynamin\\_v2\\_confusion\\_sweep.json}} větve \\texttt{{release/dynamin-confusion-v1}}}}

\\section*{{1\; Co je předmětem dokumentu}}
Vykládáme referenční analýzu. Logistická regrese se v~ní učí poznat SI štítek dráhy
z~průběhu dynaminové intenzity. Výklad jde od dat přes trénink k~výsledkům a jejich mezím.
Naše opakování téhož experimentu s~intenzitou z~cmeAnalysis má samostatný protokol a zde
je neřešíme.

\\section*{{2\; Data a readout}}
Začněme tím, co je jedna položka dat. Je to dráha klatrinové jamky v~jednom z~15 filmů
U2OS (2\\,s na snímek, 79{{,}}1\\,nm na pixel). Každá dráha nese štítek. \\emph{{Produktivní}}
je tehdy, když její Shape Index někdy za život překročí 0{{,}}7. Jinak je \\emph{{abortivní}}.
\\par
K~dráze patří průběh dynaminové intenzity, jedno číslo na snímek. Dodává je tzv.
\\emph{{box-mean}} readout. Slovo \\emph{{readout}} (odečet) obecně znamená předpis, jak
z~obrazu získat číslo ,,kolik dynaminu je na tomto místě v~tomto snímku``. Box-mean je
předpis nejprostší možný. Vezmi průměr okénka 5$\\times$5 pixelů na pozici jamky a vyděl
ho jasem celé buňky v~tom čase. Jas buňky se přitom bere jako hladký (biexponenciální) fit
průměrů snímků, aby ho nerozkmitával šum. Po dělení znamená 1 přesně průměr buňky. Pracuje
se s~\\emph{{excessem}}, tedy hodnotou minus 1. Příklad: okénko má průměrný jas 1\\,300
a buňka 1\\,000. Readout je 1{{,}}3, excess 0{{,}}3. Čteme: o~30\\,\\% nad průměrem buňky.
\\par
Dvě vlastnosti této volby si zapamatujme. Dělení jasem buňky srovnává filmy s~různým jasem
a ruší blednutí. To je silná stránka. A průměr okénka sbírá všechen signál uvnitř, tedy
i difuzní dynamin membrány a příspěvky sousedních jamek. Lokální pozadí se neodečítá.
\\par
Analýza běží v~osmi konfiguracích filtrů (tabulka~\\ref{{tab:lr}}). Kombinují se
\\emph{{cluster}} filtr (žádný, $<$ 3, $<$ 2), vzdálenostní filtr (2.~nejbližší soused
$\\geq$ 5\\,px) a úplnost dráhy. \\emph{{Interior}} znamená začátek i konec v~záznamu.
\\emph{{End-observed}} přidává dráhy s~useknutým začátkem. Primární korpus analýzy je
,,bez filtru, end-observed`` (n = {e(fmt_n(prim['n']))}).

\\section*{{3\; Na čem přesně se model učí: featury}}
Featura je jedno číslo spočítané z~průběhu dráhy. Dohromady jich je 27 a tvoří jeden řádek
vstupní tabulky. Nejlépe se chápou na příkladu. Obrázek~\\ref{{fig:feat}} ukazuje smyšlenou
dráhu žijící 50\\,s. Její život dělíme na dvě části. Posledních 20\\,s je terminální okno,
modrý pás. Všechno před tím je střední fáze, oranžový pás.
\\begin{{figure}}[H]\\centering
\\includegraphics[width=\\linewidth]{{figL0_features.png}}
\\caption{{Vstupy modelu na příkladu jedné dráhy. Modrý pás je terminální okno. Jeho průběh
model vidí bod po bodu, deset teček win\\_0 až win\\_9. Oranžový pás je střední fáze. Tu
model vidí jen jako tři souhrny: průměr (tečkovaná úsečka), maximum (trojúhelník) a podíl
snímků nad prahem významnosti. Práh je čárkovaná vodorovná čára, 90.~percentil excessu přes
celý korpus. Hvězda je maximum přes celý život. Fialová šipka dole ukazuje polohu vrcholu
v~životě. Svorka vyznačuje nejdelší nepřerušený běh nadprahových snímků.}}
\\label{{fig:feat}}\\end{{figure}}
Po skupinách:
\\begin{{itemize}}
\\item Terminální okno, 10 čísel. Průběh excessu v~posledních 20\\,s, převzorkovaný na
10 bodů po 2\\,s. Jediná část života, kterou model vidí bod po bodu. U~drah kratších než
okno se body před vznikem drží na první naměřené hodnotě.
\\item Souhrny střední fáze, 4 čísla. Průměr, maximum, podíl významných snímků a poměr
maxima střední fáze k~maximu okna. U~drah do 10 snímků střední fáze neexistuje. Všechna
čtyři čísla jsou pak nula a právě tudy model pozná krátkou dráhu.
\\item Skaláry přes celý život, 3 čísla. Maximum excessu. Poloha vrcholu v~životě, kde 0 je
vznik a 1 zánik. Nejdelší souvislý běh významných snímků.
\\item Mrtvé vstupy, 10 čísel. Sloty pre a post jsou vyhrazené pro vzorky před vznikem
a po zániku dráhy. Korpus je neobsahuje. Jsou proto vždy nula a model je ignoruje.
\\end{{itemize}}
Délka dráhy mezi featurami záměrně není. Přesto si ji modely ze vstupů zrekonstruují,
a to čtyřmi cestami. Za prvé, dráha do 10 snímků nemá střední fázi, takže její čtyři
souhrny jsou přesně nula. Čtyři nuly vedle sebe jsou spolehlivý prapor krátké dráhy.
Za druhé, u~dráhy kratší než okno se body před vznikem drží na první naměřené hodnotě.
Počet stejných bodů na začátku okna tak skoro přímo odečítá, kolik snímků dráze chybí.
Za třetí, nejdelší běh významných snímků nemůže být delší než dráha sama. Horní mez této
featury je délka. A za čtvrté, maxima rostou s~počtem pokusů. Maximum z~60 snímků je
i u~čistého šumu vyšší než maximum z~8 snímků. Model tyto stopy složí dohromady a délku
fakticky zná (Spearman $\\rho$ skóre a délky {cz(prim['rho'], 2)}, měřeno na XGBoost
skóre téhož běhu). To je známý strukturální únik. Právě jeho výhodu ruší within-band
vyhodnocení v~části~4.

\\section*{{4\; Co se děje na pozadí: trénink a vyhodnocení}}
Postup je pro každou konfiguraci stejný. Projděme ho v~pořadí, v~jakém běží.
\\par
\\textbf{{Vstupní tabulka.}} Všechno, co model kdy uvidí, je jedna tabulka. Řádek je dráha.
Sloupců je 27, jsou to čísla z~části~3, a vedle nich stojí štítek. Nic víc. Model neví,
z~jakého filmu dráha pochází, a délku dostává jen nepřímo.
\\par
\\textbf{{Srovnání měřítek (standardizace).}} Sloupce mají různé jednotky a rozsahy.
Průměrný excess je řádově desetina, nejdelší běh významných snímků desítky. Proto se každý
sloupec přepočte: odečte se jeho průměr a vydělí se směrodatnou odchylkou. Hodnota +1 pak
vždy znamená totéž, o~jednu typickou odchylku nad průměrem. Chybějící hodnoty se předtím
doplní mediánem sloupce.
\\par
\\textbf{{Model.}} Logistická regrese je nejprostší myslitelný model: vážený součet. Každé
z~27 čísel vynásobí svou vahou, sečte a součet převede na pravděpodobnost mezi 0 a 1.
Trénink jen hledá váhy, se kterými pravděpodobnosti nejlépe sedí na skutečné štítky. Mírná
regularizace (L2, C = 1) drží váhy malé, aby model nesázel příliš na jediný sloupec.
\\par
\\textbf{{Poctivé vyhodnocení (out-of-fold).}} Zkouška z~látky, kterou student právě viděl,
nic neměří. Proto se filmy rozdělí do 5 skupin a natrénuje se 5 nezávislých modelů, každý
na 12 filmech. Každá dráha pak dostane skóre od modelu, který její film nikdy neviděl.
Dělí se po celých filmech, ne po drahách. Dráhy z~jedné buňky jsou si totiž podobné
a model by se jinak naučil poznávat buňku, ne biologii.
\\par
\\textbf{{Práh a confusion matice.}} Skóre je číslo mezi 0 a 1. Tabulka správně/špatně
potřebuje hranici. Youdenovo J ji položí tam, kde je součet sensitivity a specificity
nejvyšší. Jedna poctivá poznámka: hranice se vybírá na týchž skóre, která potom hodnotíme
(\\emph{{in-sample}}). Sens a spec proto vycházejí o~něco lichotivěji, než by vyšly na
nových datech. AUC hranici nepotřebuje a tato výtka se jí netýká.
\\par
\\textbf{{Kontroly.}} Takové analýze se věří až po kontrolách. První je permutovaný null:
tytéž featury, náhodně zamíchané štítky. Musí vyjít u~0{{,}}5. Kdyby ne, je v~postupu únik.
Druhou kontrolou jsou netrénovaná skóre, výška peaku a peak minus vlastní baseline. Ta
říkají, co dokáže jediné číslo bez jakéhokoli učení.
\\par
\\textbf{{Co je AUC, pooled a within-band.}} AUC je jednoduchá hra. Vylosujeme jednu
produktivní a jednu abortivní dráhu a zeptáme se, jestli jim model dal skóre ve správném
pořadí. Podíl správných pořadí je AUC. Hodnota 0{{,}}5 je mince, 1{{,}}0 neomylnost.
\\emph{{Pooled}} AUC losuje ze všech drah bez ohledu na délku. A přesně tady vzniká
nafouknutí. Kdo hází kostkou dvacetkrát, hodí šestku téměř jistě. Kdo jen třikrát, spíše
ne. Oba štítky jsou soutěže typu ,,povedlo se to někdy za život``, takže s~délkou rostou
samy od sebe. Podíl SI+ stoupá napříč délkovými pásmy z~9\\,\\% na 68\\,\\% a podíl
dynamin+ z~24\\,\\% na 95\\,\\% (tytéž filmy, protokol porovnání klasifikací). Model,
který umí jen odhadnout délku, proto vyhrává většinu nesourodých dvojic, aniž o~dynaminu
cokoli ví. \\emph{{Within-band}} AUC tu výhodu ruší. Losuje jen dvojice drah podobné délky
(4 délková pásma: 6--9, 10--19, 20--39 a 40+ snímků, vážení počtem dvojic). A \\emph{{gap}} je prosté odečtení: pooled
minus within-band. Je to přesně ten kus výkonu, který zmizí, jakmile modelu délku vezmeme.
Na číslech primárního korpusu: pooled {cz(prim['auc'], 2)}, within-band {cz(prim['wb'], 2)},
gap +{cz(prim['gap'], 2)}. Z~výkonu {cz(prim['auc'], 2)} tedy {cz(prim['gap'], 2)} dodala
délka. Nad náhodou zbývá {cz(prim['wb'] - 0.5, 2)} skutečné dynaminové informace.

\\section*{{5\; Výsledky}}
{t}
{fig}
\\textbf{{Diskuze:}}

Z~tabulky~\\ref{{tab:lr}} a obrázku~\\ref{{fig:lr}} si odnesme čtyři věci. Za prvé,
poolovaná AUC vypadá slušně, drží se kolem {cz(intr['auc'], 2)} až {cz(prim['auc'], 2)}.
Jenže gap je všude +0{{,}}15 až +0{{,}}20. Většinu toho výkonu tedy nese délka. Za druhé, délkově očištěný signál existuje, ale jen u nefiltrovaných korpusů. Jejich
within-band AUC {cz(intr['wb'])} až {cz(prim['wb'])} leží nad nullem na
{cz(prim['null_wb'])}. Za třetí, filtrování škodí. Přísnější filtry odstraňují
neúměrně mnoho produktivních drah a within-band AUC srážejí až na úroveň nullu
(cluster $<$ 2: 0{{,}}49). Proto se nefiltruje. A za čtvrté, end-observed čísla jsou vyšší
než interior. Vyšší je ale korpus, ne model. Přidané dráhy s~useknutým začátkem jsou delší
a jasnější.

\\section*{{6\; Čeho se regrese drží: koeficienty}}
Přirozená otázka zní, které vstupy model táhnou. U~referenčního běhu na ni nelze odpovědět
přímo. Větev s~výsledky obsahuje výkonnostní čísla, ale hodnoty naučených vah k~box-mean
modelu neukládá. Máme ovšem rovnocennou analýzu z~opakování téhož experimentu s~amplitudou
cmeAnalysis. Stejné featury, stejný trénink, výkon shodný v~setinách. Její koeficienty
(standardizované, fitované po délkových pásmech, s~95\\,\\% intervaly bootstrapem po
filmech) ukazují jednoznačný vzor. Nejsilnější a stabilně kladný je průměr amplitudy přes
část života před terminálním oknem. V~pásmu 10--19 snímků jeho maximum. Váhy jednotlivých
bodů terminálního okna jsou malé a většinou s~intervalem přes nulu.
{meta['coef_fig_tex']}
\\textbf{{Diskuze:}}

Dá se říct, že regrese rozpoznává produktivní dráhy podle trvale vyššího dynaminu během
života, ne podle výrazné události na konci. Kdyby scission nesl specifický podpis, ležely
by váhy na konci okna. Přenos tohoto čtení na referenční box-mean model je úsudek, byť
podložený shodou výkonu i vstupů. Přímé potvrzení by vyžadovalo doběhnout koeficientový
režim na referenčním korpusu. Podrobný návod, jak standardizované koeficienty číst, uvádí
protokol detektoru.

\\section*{{7\; Interpretace a meze}}
\\textbf{{Co čísla říkají.}} Představme si dvě dráhy stejné délky, jednu produktivní
a jednu abortivní. Model je seřadí správně asi v~58\\,\\% případů. Náhoda by dala
50\\,\\%. To je celý dynaminový signál: reálný, ale malý. Sens {cz(prim['sens'], 2)}
a spec {cz(prim['spec'], 2)} u~Youdenova prahu jsou týž signál v~řeči confusion matice,
mírně nadhodnocený volbou prahu.
\\par
\\textbf{{Co čísla neříkají.}} Nejsou to známky kvality SI ani dynaminová reference pro
článek. Model je trénovaný na SI štítcích. Kdyby se jím SI ověřoval, ověřovali bychom
kruhem. Je to interní diagnostika, kolik délkově nezávislé informace readout nese.
A poolovaná AUC ({cz(prim['auc'], 2)}) se nemá citovat jako výkon detektoru. Obsahuje
+{cz(prim['gap'], 2)} příspěvku délky.
\\par
\\textbf{{Proč je signál tak malý.}} Tři důvody, a sčítají se. Oba štítky vznikají
operátorem ,,stalo se to někdy za život``, takže délka je dominantní společný faktor.
Dynamin je na membráně i difuzně a box-mean jej sbírá včetně okolí. Část signálu je proto
kontext, ne jamka. A reference sama je nedokonalá. Podle modelu skupiny může přibližně
pětina abortivních drah dynamin legitimně nést. Ani dokonalý klasifikátor by proto
nedosáhl shody 100\\,\\%.
\\par
\\textbf{{Vztah k~volbě modelu pro článek.}} Logistická regrese byla zvolena, protože
výkonem odpovídá XGBoostu i MLP (rozdíly v~setinách), má menší délkový únik a její
koeficienty jsou interpretovatelné. Naše nezávislá kontrola s~intenzitou z~cmeAnalysis
dává na srovnatelném korpusu tentýž obraz. Podrobnosti uvádí protokol detektoru.

\\section*{{8\; Shrnutí}}
\\begin{{enumerate}}
\\item Trénuje se na 27 featurách z~průběhu box-mean excessu (okno posledních 20\\,s,
souhrn střední fáze, tři skaláry). Délka dráhy mezi featurami není, model si ji ale
zrekonstruuje.
\\item Vyhodnocení je out-of-fold s~foldy po filmech. Práh se volí Youdenovým J,
in-sample. Kontrolou je permutovaný null a netrénovaná skóre.
\\item Poolovaná AUC $\\approx$ {cz(prim['auc'], 2)} je z~většiny délka (gap
+{cz(prim['gap'], 2)}). Délkově očištěný signál je within-band AUC $\\approx$
{cz(prim['wb'], 2)}, malý, ale nad nullem.
\\item Filtry výsledek nezlepšují, přísné ho srážejí k~nullu. End-observed čísla zvedá
složení korpusu, ne model.
\\item Koeficienty rovnocenné analýzy ukazují, že regrese stojí na trvale zvýšeném dynaminu
během života, ne na terminální události. Referenční běh vlastní váhy neukládá.
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
    fig_features(os.path.join(args.out, "figL0_features.png"))
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
        guide2 = (
            "Obrázek 3 čteme takto. Model dává každé dráze skóre: každý vstup vynásobí "
            "svou vahou a výsledky sečte. Kladná váha táhne dráhu k „produktivní\", "
            "záporná k „abortivní\". Levý panel ukazuje váhy souhrnných vstupů. Každý "
            "řádek je jeden vstup, každá tečka jeho váha v jednom délkovém pásmu (barvy), "
            "vousy kolem tečky jsou 95% interval a svislá čára je nula. Tečka vpravo od "
            "nuly znamená „táhne k produktivní\". Když vous protíná nulu, není jistý ani "
            "směr a takovou váhu nevykládáme. Pravý panel ukazuje totéž pro deset bodů "
            "posledních 20 s života. Osa x je čas před koncem dráhy. Kdyby model poznával "
            "produktivní dráhy podle události těsně před koncem, vyletěla by zde některá "
            "čára u pravého okraje nahoru.")
        guide3 = (
            "Obrázek 4 je myšlenkový pokus s týmž modelem. Vezměme průměrnou dráhu "
            "a otáčejme jediným knoflíkem, a to průměrem amplitudy před oknem; ostatní "
            "vstupy držíme na průměru. Křivka ukazuje, jakou pravděpodobnost "
            "„produktivní\" model dráze přisoudí. Vodorovné tečkované čáry jsou podíl "
            "produktivních v pásmu. Svislá tečkovaná čára označuje průměrnou dráhu. "
            "Ploché křivky krátkých pásem říkají, že tam tento vstup nerozhoduje, ve "
            "shodě s levým panelem obrázku 3.")
        meta["coef_fig_md"] = (f"\n{guide2}\n"
                               f"\n![koeficienty]({'figL2_logreg_coefs.png'})\n\n"
                               f"*Obrázek 3: {cap}*\n"
                               f"\n{guide3}\n"
                               f"\n![sigmoida]({'figL3_logreg_sigmoid.png'})\n\n"
                               f"*Obrázek 4: {cap3}*\n")
        meta["coef_fig_tex"] = (
            "Obrázek~\\ref{fig:coef} čteme takto. Model dává každé dráze skóre: každý "
            "vstup vynásobí svou vahou a výsledky sečte. Kladná váha táhne dráhu "
            "k~,,produktivní``, záporná k~,,abortivní``. Levý panel ukazuje váhy "
            "souhrnných vstupů. Každý řádek je jeden vstup, každá tečka jeho váha "
            "v~jednom délkovém pásmu (barvy), vousy kolem tečky jsou 95\\,\\% interval "
            "a svislá čára je nula. Tečka vpravo od nuly znamená ,,táhne "
            "k~produktivní``. Když vous protíná nulu, není jistý ani směr a takovou váhu "
            "nevykládáme. Pravý panel ukazuje totéž pro deset bodů posledních 20\\,s "
            "života. Osa x je čas před koncem dráhy. Kdyby model poznával produktivní "
            "dráhy podle události těsně před koncem, vyletěla by zde některá čára "
            "u~pravého okraje nahoru.\n"
            "\\begin{figure}[H]\\centering"
            "\\includegraphics[width=\\linewidth]{figL2_logreg_coefs.png}"
            "\\caption{Koeficienty rovnocenného běhu na amplitudě cmeAnalysis (korpus bez "
            "filtru, interior; referenční běh váhy neukládá). Vlevo souhrnné vstupy s~95\\,\\% "
            "intervaly bootstrapem po filmech, barvy rozlišují délková pásma; hodnoty vpravo "
            "od nuly táhnou k~,,produktivní``. Vpravo váhy jednotlivých bodů terminálního "
            "okna s~pásy 95\\,\\% intervalů.}\\label{fig:coef}\\end{figure}\n"
            "Obrázek~\\ref{fig:sig} je myšlenkový pokus s~týmž modelem. Vezměme průměrnou "
            "dráhu a otáčejme jediným knoflíkem, a to průměrem amplitudy před oknem; "
            "ostatní vstupy držíme na průměru. Křivka ukazuje, jakou pravděpodobnost "
            ",,produktivní`` model dráze přisoudí. Vodorovné tečkované čáry jsou podíl "
            "produktivních v~pásmu. Svislá tečkovaná čára označuje průměrnou dráhu. "
            "Ploché křivky krátkých pásem říkají, že tam tento vstup nerozhoduje, ve "
            "shodě s~levým panelem obrázku~\\ref{fig:coef}.\n"
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
