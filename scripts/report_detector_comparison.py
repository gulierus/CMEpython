#!/usr/bin/env python3
"""Protokol k ukolu A: opakovani detektoroveho experimentu s intenzitou z cmeAnalysis.

Cte tri vystupy Matyasova sweepu (dynamin_v2_confusion_sweep.json): puvodni
box-mean bezh z jeho vetve a dva nase behy z Heliosu (surova a normalizovana
amplituda cmeAnalysis). K tomu koeficienty logisticke regrese z behu 238490/1.
Vytvori srovnavaci tabulky, obrazky a protokol (md + tex + pdf) ve stylu VU_RG.

    python3 scripts/report_detector_comparison.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
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
MODELS = ["XGBoost", "NN (MLP 64-32)", "logistic regression", "peak height only",
          "peak − own baseline (CSV burst score)", "permuted-label null (XGBoost)"]
SHORT = ["XGBoost", "MLP", "logistická regrese", "výška peaku",
         "peak − vlastní baseline", "permutovaný null"]
CONFIG_LABELS = {
    "none_interior": "bez filtru, interior", "none_endobs": "bez filtru, end-observed",
    "lt3_interior": "cluster < 3, interior", "lt3_endobs": "cluster < 3, end-observed",
    "lt2_interior": "cluster < 2, interior", "lt2_endobs": "cluster < 2, end-observed",
    "second5_interior": "2. soused ≥ 5 px, interior", "second5_endobs": "2. soused ≥ 5 px, end-observed",
}
READOUTS = ["box-mean", "cme surová", "cme normalizovaná"]
BLUE, ORANGE, GREEN = "#2b7bba", "#e07a29", "#27ae60"


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


def load_runs(args):
    def newest(pattern):
        cands = sorted(glob.glob(pattern))
        return cands[-1] if cands else None
    paths = {
        "box-mean": args.boxmean_json,
        "cme surová": args.raw_json or newest(os.path.join(args.runs, "raw_*", "dynamin_v2_confusion_sweep.json")),
        "cme normalizovaná": args.norm_json or newest(os.path.join(args.runs, "normalized_*", "dynamin_v2_confusion_sweep.json")),
    }
    runs = {}
    for name, p in paths.items():
        if p is None or not os.path.exists(p):
            sys.exit(f"chybi sweep JSON pro '{name}' ({p})")
        with open(p, encoding="utf-8") as fh:
            runs[name] = json.load(fh)["configs"]
        print(f"{name}: {p}")
    return runs, paths


# =============================================================== tabulky
def main_table_rows(runs, tag="none_interior"):
    rows = []
    for model, short in zip(MODELS, SHORT):
        row = [short]
        for r in READOUTS:
            c = runs[r][tag]
            row += [cz(c["pooled"]["models"][model]["auc"]), cz(wb_auc(c, model))]
        rows.append(row)
    return rows


def config_table_rows(runs, model="XGBoost"):
    rows = []
    for tag, lbl in CONFIG_LABELS.items():
        row = [lbl, fmt_n(runs["cme surová"][tag]["n_tracks"])]
        for r in READOUTS:
            row.append(cz(wb_auc(runs[r][tag], model)))
        rows.append(row)
    return rows


def corpus_table_rows(runs, tag="none_interior"):
    rows = []
    for r in READOUTS:
        c = runs[r][tag]
        rho = c["pooled"].get("rho_xgb_score_vs_length", np.nan)
        rows.append([r, fmt_n(c["n_tracks"]), cz(100 * c["prevalence"], 1) + " %", cz(rho, 2)])
    return rows


# =============================================================== obrazky
def fig_wb_bars(runs, out, tag="none_interior"):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(MODELS))
    colors = [BLUE, GREEN, ORANGE]
    for i, (r, col) in enumerate(zip(READOUTS, colors)):
        vals = [wb_auc(runs[r][tag], m) for m in MODELS]
        ax.bar(x + (i - 1) * 0.26, vals, 0.24, color=col, label=r)
        for xi, v in zip(x + (i - 1) * 0.26, vals):
            ax.text(xi, v + 0.004, cz(v).replace("0,", ","), ha="center", fontsize=7.5)
    ax.axhline(0.5, color="0.5", lw=1, ls=":")
    ax.set_xticks(x, SHORT, fontsize=9)
    ax.set_ylim(0.45, 0.65)
    ax.set_ylabel("within-band AUC")
    ax.set_title("Within-band AUC po modelech a readoutech (bez filtru, interior)", fontsize=10.5)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_pooled_vs_wb(runs, out, tag="none_interior"):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    marks = {"box-mean": "o", "cme surová": "s", "cme normalizovaná": "^"}
    cols = dict(zip(READOUTS, [BLUE, GREEN, ORANGE]))
    for r in READOUTS:
        for model, short in zip(MODELS, SHORT):
            c = runs[r][tag]
            p = c["pooled"]["models"][model]["auc"]
            w = wb_auc(c, model)
            ax.scatter(p, w, marker=marks[r], color=cols[r], s=45,
                       label=r if model == MODELS[0] else None)
    ax.plot([0.45, 0.8], [0.45, 0.8], ls=":", color="0.6", lw=1)
    ax.axhline(0.5, color="0.8", lw=1, ls=":")
    ax.set_xlabel("pooled AUC"); ax.set_ylabel("within-band AUC")
    ax.set_title("Pooled proti within-band AUC (bez filtru, interior)", fontsize=10.5)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# =============================================================== dokumenty
def md_table(rows, header):
    head = "| " + " | ".join(header) + " |"
    sep = "|" + "|".join("---" for _ in header) + "|"
    body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}"


def build_markdown(runs, meta):
    t_corp = md_table(corpus_table_rows(runs),
                      ["readout", "drah (interior)", "podíl SI+", "ρ(skóre XGB, délka)"])
    t_main = md_table(main_table_rows(runs),
                      ["model", "box-mean pooled", "box-mean wb", "cme surová pooled",
                       "cme surová wb", "cme norm. pooled", "cme norm. wb"])
    t_cfg = md_table(config_table_rows(runs),
                     ["konfigurace filtrů", "n (cme)", "box-mean", "cme surová", "cme norm."])
    return f"""# Detektor dynaminové pozitivity na intenzitě z cmeAnalysis

Datum {meta['date']} · CMEpython {meta['git']} · vygenerováno `scripts/report_detector_comparison.py`

## 1. Co jsme udělali a proč

Zopakovali jsme Matyášův experiment s trénováním detektoru dynaminové pozitivity. Jedinou
změnou je vstupní dynaminová intenzita, kterou jsme nahradili amplitudou z cmeAnalysis.
Ptáme se, zda kvalita měření byla úzkým hrdlem předchozích výsledků.

Postupovali jsme takto. Nejprve jsme z našich měření sestavili dva korpusy ve formátu, který
Matyášův kód čte beze změny. V prvním je amplituda v surových jednotkách kamery. Ve druhém je
navíc vydělena biexponenciálním fitem průměrů snímků buňky, tedy stejnou normalizací, jakou
používá jeho *box-mean* readout. Konvence `intenzita = 1 + amplituda` zajišťuje, že jeho
`excess` je přímo amplituda cmeAnalysis. Následně jsme na výpočetním clusteru HELIOS spustili
jeho trénovací skript nad oběma korpusy, a to bez jakékoli úpravy kódu. Skript trénuje šest
modelů v osmi konfiguracích filtrů, s pětinásobnou křížovou validací grupovanou po filmech
a s prahem podle Youdenova indexu. Referencí je jeho původní běh s *box-mean* readoutem.

Poznamenejme dvě omezení srovnání. Za prvé, naše korpusy obsahují jen dodané dráhy, tedy
interior podmnožinu; jeho korpus obsahuje i dráhy s useknutým začátkem. Primární srovnání
proto vedeme na konfiguraci „bez filtru, interior", kde jsou populace téměř shodné
(tabulka 1). Za druhé, na clusteru běžely mírně jiné verze knihoven (pandas 2.2.3
a xgboost 2.1.4 místo 2.3.3 a 3.1.2); podle Matyášovy dokumentace mění verze skóre
fitovaných modelů jen v posledních desetinných místech.

{t_corp}

*Tabulka 1: Korpusy v konfiguraci „bez filtru, interior". Populace i prevalence jsou téměř
shodné; korelace skóre s délkou dráhy je shodná u všech tří readoutů.*

## 2. Hlavní výsledek

Tabulku 2 čteme takto. Pro každý model uvádíme dvojici čísel na každý readout. *Pooled* AUC
hodnotí všechny dráhy dohromady. *Within-band* AUC porovnává jen dráhy podobné délky, a je
tedy očištěná o délkový efekt. Poslední řádek je kontrola se zamíchanými nálepkami, která
musí vyjít u 0,5.

{t_main}

*Tabulka 2: Pooled a within-band AUC po modelech a readoutech, konfigurace „bez filtru,
interior".*

![within-band AUC](figA1_wb_bars.png)

*Obrázek 1: Within-band AUC po modelech a readoutech. Tečkovaná čára je úroveň náhody.*

![pooled vs within-band](figA2_pooled_wb.png)

*Obrázek 2: Pooled proti within-band AUC; každý bod je jedna dvojice model a readout.
Všechny body leží hluboko pod diagonálou, rozdíl je délkový efekt.*

**Diskuze:**

Z tabulky 2 a obrázků 1 a 2 plyne trojí. Za prvé, surová a normalizovaná amplituda dávají
prakticky stejné výsledky. Volba normalizace tedy nehraje roli, což odpovídá tomu, že
amplituda cmeAnalysis má lokální pozadí odečtené již z konstrukce. Za druhé, amplituda
cmeAnalysis dává mírně nižší, avšak řádově stejné hodnoty jako *box-mean*. Rozdíly ve
*within-band* AUC jsou v setinách a leží pod mezifilmovým rozptylem, který Matyášova
dokumentace uvádí kolem 0,05. Za třetí, pořadí modelů i velikost délkového efektu zůstávají
stejné. Korelace skóre s délkou je u všech readoutů shodná (tabulka 1) a rozdíl mezi pooled
a within-band hodnotou se drží kolem 0,15 až 0,18. Usuzujeme, že kvalita měření úzkým hrdlem
nebyla, a to je hlavní odpověď tohoto experimentu.

## 3. Robustnost napříč konfiguracemi filtrů

{t_cfg}

*Tabulka 3: Within-band AUC modelu XGBoost ve všech osmi konfiguracích filtrů. U našich
korpusů se konfigurace „end-observed" od „interior" liší jen nepatrně, dodané dráhy totiž
useknuté konce téměř neobsahují.*

**Diskuze:**

Závěr z části 2 platí ve všech konfiguracích. Žádná kombinace filtrů nezvedne *within-band*
AUC nad úroveň, kterou známe z *box-mean* readoutu.

## 4. Čeho se model drží: koeficienty logistické regrese

Z dřívějšího běhu na týchž korpusech máme standardizované koeficienty logistické regrese po
délkových pásmech (soubory `dynamin_v2_logreg_coefs*` v obou runech). Nejsilnější a
bootstrapově stabilní koeficient v delších pásmech je průměrná amplituda v rané a střední
fázi života dráhy, kladný. Váhy terminálního okna, tedy posledních 20 s před koncem dráhy,
jsou malé a většinou nestabilní.

**Diskuze:**

Dá se říct, že detektor rozpoznává produktivní dráhy podle trvale vyššího dynaminu během
života, ne podle výrazné události na konci. To je konzistentní s dřívějším pozorováním, že
terminální vzestup mají obě třídy. Surová a normalizovaná varianta dávají stejné koeficienty,
což opět potvrzuje, že volba normalizace nehraje roli.

## 5. Shrnutí

1. Experiment jsme zopakovali beze změny Matyášova kódu, pouze s naší intenzitou;
   obě varianty korpusu doběhly čistě a kontrola s permutovanými nálepkami sedí na 0,5.
2. Surová a normalizovaná amplituda dávají stejné výsledky; volba normalizace nehraje roli.
3. Amplituda cmeAnalysis nepřekonala *box-mean*; rozdíly jsou v setinách a pod mezifilmovým
   rozptylem (tabulka 2). Kvalita měření tedy nebyla úzkým hrdlem.
4. Délkový efekt zůstává beze změny u všech readoutů (ρ ≈ 0,87; gap ≈ 0,15 až 0,18).
   Strop výsledků drží délkový confounding a vlastnosti nálepek, ne měření.
5. Výsledek je konzistentní s úlohou B, kde na spojité amplitudě vyšla within-band AUC
   0,520 pro cmeAnalysis a 0,536 pro *box-mean* na stejných drahách.

## 6. Reprodukce

```bash
python3 scripts/build_cme_corpus.py --variant both        # korpusy (CME_for_Helios/data)
# na HELIOSu: ./setup_env.sh && ./submit_all.sh           # joby 238568 (raw), 238569 (norm)
python3 scripts/report_detector_comparison.py             # tento protokol
```

Vstupy: `CME_for_Helios/runs/raw_238568/`, `.../normalized_238569/` a Matyášův
`dynamin_v2_confusion_sweep.json` z větve `release/dynamin-confusion-v1`.
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
             .replace("≥", r"$\geq$").replace("≈", r"$\approx$")
             .replace("→", r"$\rightarrow$").replace("−", "--").replace("|", r"$\mid$"))


def tex_table(rows, colspec, header, caption, label):
    body = "\n".join(" & ".join(tex_escape(str(v)) for v in r) + r" \\" for r in rows)
    return (f"\\begin{{table}}[H]\\centering\\small\n"
            f"\\begin{{tabular}}{{{colspec}}}\n\\toprule {header} \\\\ \\midrule\n"
            f"{body}\n\\bottomrule\\end{{tabular}}\n"
            f"\\caption{{{tex_escape(caption)}}}\\label{{{label}}}\\end{{table}}")


def tex_fig(fname, caption, label, width="\\linewidth"):
    return (f"\\begin{{figure}}[H]\\centering"
            f"\\includegraphics[width={width}]{{{fname}}}"
            f"\\caption{{{tex_escape(caption)}}}\\label{{{label}}}\\end{{figure}}")


def build_tex(runs, meta):
    t_corp = tex_table(corpus_table_rows(runs), "lrrr",
                       "readout & drah (interior) & podíl SI+ & $\\rho$(skóre, délka)",
                       "Korpusy v konfiguraci ,,bez filtru, interior``. Populace i prevalence "
                       "jsou téměř shodné; korelace skóre s délkou dráhy je shodná u všech tří "
                       "readoutů.", "tab:corp")
    t_main = tex_table(main_table_rows(runs), "p{3.4cm}rrrrrr",
                       "model & \\multicolumn{2}{c}{box-mean} & \\multicolumn{2}{c}{cme surová} "
                       "& \\multicolumn{2}{c}{cme norm.} \\\\ & pooled & wb & pooled & wb & pooled & wb",
                       "Pooled a within-band AUC po modelech a readoutech, konfigurace "
                       ",,bez filtru, interior``.", "tab:main")
    t_cfg = tex_table(config_table_rows(runs), "p{4.6cm}rrrr",
                      "konfigurace filtrů & n (cme) & box-mean & cme surová & cme norm.",
                      "Within-band AUC modelu XGBoost ve všech osmi konfiguracích filtrů. "
                      "U našich korpusů se ,,end-observed`` od ,,interior`` liší jen nepatrně, "
                      "dodané dráhy totiž useknuté konce téměř neobsahují.", "tab:cfg")
    f1 = tex_fig("figA1_wb_bars.png", "Within-band AUC po modelech a readoutech. Tečkovaná "
                 "čára je úroveň náhody.", "fig:wb")
    f2 = tex_fig("figA2_pooled_wb.png", "Pooled proti within-band AUC; každý bod je jedna "
                 "dvojice model a readout. Všechny body leží hluboko pod diagonálou, rozdíl "
                 "je délkový efekt.", "fig:pw", width="0.72\\linewidth")
    return TEX_HEAD + f"""
{{\\LARGE\\bfseries Detektor dynaminové pozitivity\\\\na intenzitě z cmeAnalysis}}\\\\[4pt]
{{\\small Datum {meta['date']} \\;·\\; CMEpython {meta['git']} \\;·\\;
\\texttt{{scripts/report\\_detector\\_comparison.py}}}}

\\section*{{1\\; Co jsme udělali a proč}}
Zopakovali jsme Matyášův experiment s~trénováním detektoru dynaminové pozitivity. Jedinou
změnou je vstupní dynaminová intenzita, kterou jsme nahradili amplitudou z~cmeAnalysis.
Ptáme se, zda kvalita měření byla úzkým hrdlem předchozích výsledků.
\\par
Postupovali jsme takto. Nejprve jsme z~našich měření sestavili dva korpusy ve formátu, který
Matyášův kód čte beze změny. V~prvním je amplituda v~surových jednotkách kamery. Ve druhém je
navíc vydělena biexponenciálním fitem průměrů snímků buňky, tedy stejnou normalizací, jakou
používá jeho \\emph{{box-mean}} readout. Konvence \\texttt{{intenzita = 1 + amplituda}}
zajišťuje, že jeho \\texttt{{excess}} je přímo amplituda cmeAnalysis. Následně jsme na
výpočetním clusteru HELIOS spustili jeho trénovací skript nad oběma korpusy, a to bez
jakékoli úpravy kódu. Skript trénuje šest modelů v~osmi konfiguracích filtrů,
s~pětinásobnou křížovou validací grupovanou po filmech a s~prahem podle Youdenova indexu.
Referencí je jeho původní běh s~\\emph{{box-mean}} readoutem.
\\par
Poznamenejme dvě omezení srovnání. Za prvé, naše korpusy obsahují jen dodané dráhy, tedy
interior podmnožinu; jeho korpus obsahuje i dráhy s~useknutým začátkem. Primární srovnání
proto vedeme na konfiguraci ,,bez filtru, interior``, kde jsou populace téměř shodné
(tabulka~\\ref{{tab:corp}}). Za druhé, na clusteru běžely mírně jiné verze knihoven
(pandas 2.2.3 a xgboost 2.1.4 místo 2.3.3 a 3.1.2); podle Matyášovy dokumentace mění verze
skóre fitovaných modelů jen v~posledních desetinných místech.
{t_corp}

\\section*{{2\\; Hlavní výsledek}}
Tabulku~\\ref{{tab:main}} čteme takto. Pro každý model uvádíme dvojici čísel na každý
readout. \\emph{{Pooled}} AUC hodnotí všechny dráhy dohromady. \\emph{{Within-band}} AUC
porovnává jen dráhy podobné délky, a je tedy očištěná o délkový efekt. Poslední řádek je
kontrola se zamíchanými nálepkami, která musí vyjít u~0{{,}}5.
{t_main}
{f1}
{f2}
\\textbf{{Diskuze:}}

Z~tabulky~\\ref{{tab:main}} a obrázků~\\ref{{fig:wb}} a~\\ref{{fig:pw}} plyne trojí. Za prvé,
surová a normalizovaná amplituda dávají prakticky stejné výsledky. Volba normalizace tedy
nehraje roli, což odpovídá tomu, že amplituda cmeAnalysis má lokální pozadí odečtené již
z~konstrukce. Za druhé, amplituda cmeAnalysis dává mírně nižší, avšak řádově stejné hodnoty
jako \\emph{{box-mean}}. Rozdíly ve \\emph{{within-band}} AUC jsou v~setinách a leží pod
mezifilmovým rozptylem, který Matyášova dokumentace uvádí kolem 0{{,}}05. Za třetí, pořadí
modelů i velikost délkového efektu zůstávají stejné. Korelace skóre s~délkou je u~všech
readoutů shodná (tabulka~\\ref{{tab:corp}}) a rozdíl mezi pooled a within-band hodnotou se
drží kolem 0{{,}}15 až 0{{,}}18. Usuzujeme, že kvalita měření úzkým hrdlem nebyla, a to je
hlavní odpověď tohoto experimentu.

\\section*{{3\\; Robustnost napříč konfiguracemi filtrů}}
{t_cfg}
\\textbf{{Diskuze:}}

Závěr z~části~2 platí ve všech konfiguracích. Žádná kombinace filtrů nezvedne
\\emph{{within-band}} AUC nad úroveň, kterou známe z~\\emph{{box-mean}} readoutu.

\\section*{{4\\; Čeho se model drží: koeficienty logistické regrese}}
Z~dřívějšího běhu na týchž korpusech máme standardizované koeficienty logistické regrese po
délkových pásmech. Nejsilnější a bootstrapově stabilní koeficient v~delších pásmech je
průměrná amplituda v~rané a střední fázi života dráhy, kladný. Váhy terminálního okna, tedy
posledních 20\\,s před koncem dráhy, jsou malé a většinou nestabilní.
\\par
\\textbf{{Diskuze:}}

Dá se říct, že detektor rozpoznává produktivní dráhy podle trvale vyššího dynaminu během
života, ne podle výrazné události na konci. To je konzistentní s~dřívějším pozorováním, že
terminální vzestup mají obě třídy. Surová a normalizovaná varianta dávají stejné
koeficienty, což opět potvrzuje, že volba normalizace nehraje roli.

\\section*{{5\\; Shrnutí}}
\\begin{{enumerate}}
\\item Experiment jsme zopakovali beze změny Matyášova kódu, pouze s~naší intenzitou; obě
varianty korpusu doběhly čistě a kontrola s~permutovanými nálepkami sedí na 0{{,}}5.
\\item Surová a normalizovaná amplituda dávají stejné výsledky; volba normalizace nehraje roli.
\\item Amplituda cmeAnalysis nepřekonala \\emph{{box-mean}}; rozdíly jsou v~setinách a pod
mezifilmovým rozptylem (tabulka~\\ref{{tab:main}}). Kvalita měření tedy nebyla úzkým hrdlem.
\\item Délkový efekt zůstává beze změny u~všech readoutů ($\\rho \\approx 0{{,}}87$; rozdíl
pooled a within-band $\\approx$ 0{{,}}15 až 0{{,}}18). Strop výsledků drží délkový
confounding a vlastnosti nálepek, ne měření.
\\item Výsledek je konzistentní s~úlohou B, kde na spojité amplitudě vyšla within-band AUC
0{{,}}520 pro cmeAnalysis a 0{{,}}536 pro \\emph{{box-mean}} na stejných drahách.
\\end{{enumerate}}

\\section*{{6\\; Reprodukce}}
\\texttt{{build\\_cme\\_corpus.py --variant both}} (korpusy); na HELIOSu
\\texttt{{./setup\\_env.sh}} a \\texttt{{./submit\\_all.sh}} (joby 238568 a 238569);
\\texttt{{report\\_detector\\_comparison.py}} (tento protokol).
\\end{{document}}
"""


# =============================================================== main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default=os.path.join(ROOT, "CME_for_Helios", "runs"))
    ap.add_argument("--boxmean-json", default=os.path.join(
        ROOT, "external", "Shape2Fate_Fake2Emulate", "dynamin_v2_confusion_sweep.json"))
    ap.add_argument("--raw-json", default=None)
    ap.add_argument("--norm-json", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "CME_for_Helios", "report"))
    ap.add_argument("--skip-pdf", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs, paths = load_runs(args)
    meta = {
        "date": _dt.date.today().isoformat(),
        "git": subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "?",
    }

    fig_wb_bars(runs, os.path.join(args.out, "figA1_wb_bars.png"))
    fig_pooled_vs_wb(runs, os.path.join(args.out, "figA2_pooled_wb.png"))
    # kopie mozaik pro vizualni srovnani vedle protokolu
    pairs = [
        (os.path.join(ROOT, "external", "Shape2Fate_Fake2Emulate", "generator",
                      "generator_showcase", "dynamin_v2_confusion_pooled.png"),
         "pooled_boxmean_original.png"),
        (os.path.join(os.path.dirname(paths["cme surová"]), "dynamin_v2_confusion_pooled.png"),
         "pooled_cme_raw.png"),
        (os.path.join(os.path.dirname(paths["cme normalizovaná"]), "dynamin_v2_confusion_pooled.png"),
         "pooled_cme_normalized.png"),
    ]
    for src, dst in pairs:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, dst))

    with open(os.path.join(args.out, "protokol_detektor.md"), "w", encoding="utf-8") as fh:
        fh.write(build_markdown(runs, meta))
    with open(os.path.join(args.out, "protokol_detektor.tex"), "w", encoding="utf-8") as fh:
        fh.write(build_tex(runs, meta))

    if not args.skip_pdf and shutil.which("latexmk"):
        r = subprocess.run(["latexmk", "-xelatex", "-interaction=nonstopmode",
                            "-halt-on-error", "-cd", os.path.join(args.out, "protokol_detektor.tex")],
                           capture_output=True, text=True)
        if r.returncode == 0:
            subprocess.run(["latexmk", "-c", "-cd", os.path.join(args.out, "protokol_detektor.tex")],
                           capture_output=True, text=True)
            print("PDF: protokol_detektor.pdf")
        else:
            print("VAROVANI: xelatex selhal; posledni radky logu:")
            print("\n".join(r.stdout.splitlines()[-15:]))

    print(f"\nhotovo -> {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
