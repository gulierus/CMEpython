#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment dynamin -> klatrin: v kolegove puvodnim experimentu se
nahradi SI stitek stitkem odvozenym z klatrinove intenzity; vstupni
dynaminove featury, filtry, foldy i logisticka regrese zustavaji bit po
bitu jeho (stitek vznika z df.cls pravidlem max(cls) > 0.7, takze staci
prepsat sloupec cls hodnotami 1.0/0.0).

Stitky z klatrinu (per-track max amplitudy z AVG datasetu):
  prev  prah srovnany na prevalenci SI v none_endobs (stejny pocet
        pozitivnich jako SI -> primo srovnatelne confusion matice)
  gmm   prah z pruseciku dvou komponent GMM na log10(max amplitudy)
+ sweep prahu pres percentily 50-95 (none_endobs).

Vystup: Clathrin Analysis/report/dyn_to_clc_results.json

    python3 scripts/dyn_to_clc_experiment.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "external", "Shape2Fate_Fake2Emulate")
CORPUS = os.path.join(CODE, "data", "dynamin_u2os", "processed_v2")
PT_CSV = os.path.join(ROOT, "Clathrin Analysis", "his_corpus_clc", "per_track.csv")
OUT_JSON = os.path.join(ROOT, "Clathrin Analysis", "report", "dyn_to_clc_results.json")
SEED = 0

os.environ["DYNAMIN_V2_DIR"] = CORPUS
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "scripts"))

from dynamin.features import ARM_END_ANCHORED, FeatureSpec, build_features  # noqa: E402
from dynamin.report import oof_probabilities  # noqa: E402
from dynamin.splits import within_experiment_film_grouped  # noqa: E402
from dynamin_terminal_score import BANDS, N_WINDOW, WINDOW_S, band_label, band_of  # noqa: E402
from dynamin_terminal_score_figure import make_logreg  # noqa: E402
from dynamin_v2_terminal_score_figure import build_v2_timecourses, load_v2_frame  # noqa: E402
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

CFGS = {
    "none_endobs": dict(cluster_filter="none", completeness="end_observed", distance_filter=None),
    "none_interior": dict(cluster_filter="none", completeness="interior", distance_filter=None),
    "lt3_endobs": dict(cluster_filter="lt3", completeness="end_observed", distance_filter=None),
    "lt3_interior": dict(cluster_filter="lt3", completeness="interior", distance_filter=None),
    "lt2_endobs": dict(cluster_filter="lt2", completeness="end_observed", distance_filter=None),
    "lt2_interior": dict(cluster_filter="lt2", completeness="interior", distance_filter=None),
    "second5_endobs": dict(cluster_filter="none", completeness="end_observed", distance_filter=5.0),
    "second5_interior": dict(cluster_filter="none", completeness="interior", distance_filter=5.0),
}


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    j = 1 + int(np.argmax((tpr - fpr)[1:]))
    t = float(thr[j])
    pred = s >= t
    tp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum()); tn = int((~pred & (y == 0)).sum())
    return dict(threshold=round(t, 6), tp=tp, fn=fn, fp=fp, tn=tn,
                sensitivity=round(tp / max(tp + fn, 1), 4),
                specificity=round(tn / max(tn + fp, 1), 4))


def evaluate(X, y, film, band, folds):
    score = oof_probabilities(X, y, folds, make_model=lambda: make_logreg(SEED), seed=SEED)
    ok = np.isfinite(score)
    y, score, band = y[ok], score[ok], band[ok]
    pooled = dict(n=int(len(y)), prevalence=round(float(y.mean()), 4),
                  auc=round(float(roc_auc_score(y, score)), 4), **youden(y, score))
    bands = []
    num = den = 0.0
    for lo, hi in BANDS:
        lab = band_label(lo, hi)
        sel = band == lab
        yb, sb = y[sel], score[sel]
        n1 = int(yb.sum()); n0 = int(sel.sum()) - n1
        a = float(roc_auc_score(yb, sb)) if n1 and n0 else float("nan")
        rec = dict(band=lab, n=int(sel.sum()), n_pos=n1,
                   prevalence=round(n1 / max(sel.sum(), 1), 4),
                   auc=round(a, 4), **(youden(yb, sb) if n1 and n0 else {}))
        bands.append(rec)
        if np.isfinite(a):
            num += a * n1 * n0; den += n1 * n0
    return dict(pooled=pooled, wb_auc=round(num / den, 4), bands=bands)


def main() -> int:
    pt = pd.read_csv(PT_CSV)
    amp = {(int(r.film), int(r.particle)): r.amp_max for r in pt.itertuples()}

    df = load_v2_frame()
    films = sorted(df.film.unique().tolist())
    with open(os.path.join(CORPUS, "thresholds_q90.json")) as fh:
        sig_threshold = float(json.load(fh)["5x5"])
    spec = FeatureSpec(arm=ARM_END_ANCHORED, window_s=WINDOW_S,
                       n_window_points=N_WINDOW, include_lifetime=False,
                       min_life_frames=6)

    # featury a SI stitky jednou pro kazdou konfiguraci (labels meni jen y)
    built = {}
    for tag, kw in CFGS.items():
        tcs, labels, film_of = build_v2_timecourses(
            df, films, readout="5x5", si_thr=0.7, sig_threshold=sig_threshold, **kw)
        X, names, meta = build_features(tcs, spec)
        kept = [i for i, tc in enumerate(tcs) if tc.lifetime_frames >= 6]
        y_si = labels[kept]
        film = film_of[kept]
        tid = meta["track_id"]
        L = meta["lifetime_frames"]
        band = np.array([band_of(int(l)) for l in L])
        folds, _ = within_experiment_film_grouped(meta["experiment"],
                                                  film.astype(object), 5, seed=SEED)
        a = np.array([amp.get((int(f), int(t)), np.nan) for f, t in zip(film, tid)])
        n_miss = int(np.isnan(a).sum())
        built[tag] = dict(X=X, y_si=np.asarray(y_si), film=film, tid=tid,
                          band=band, folds=folds, amp=np.nan_to_num(a, nan=-np.inf),
                          n_miss=n_miss)
        print(f"[{tag}] n={len(y_si):,}  SI+ {np.mean(y_si)*100:.1f} %  "
              f"chybi amp {n_miss}", flush=True)

    # ---- definice prahu na none_endobs ----
    b0 = built["none_endobs"]
    s_max = b0["amp"]
    n_pos_si = int(b0["y_si"].sum())
    thr_prev = float(np.sort(s_max)[::-1][n_pos_si - 1])   # presne stejny pocet +
    finite = s_max[np.isfinite(s_max) & (s_max > 0)]
    lg = np.log10(finite)
    gm = GaussianMixture(2, random_state=0).fit(lg.reshape(-1, 1))
    grid = np.linspace(lg.min(), lg.max(), 2000)
    post = gm.predict_proba(grid.reshape(-1, 1))
    hi_comp = int(np.argmax(gm.means_.ravel()))
    lo_m, hi_m = sorted(gm.means_.ravel())
    inside = (grid > lo_m) & (grid < hi_m)
    cross = grid[inside][np.argmin(np.abs(post[inside, hi_comp] - 0.5))]
    thr_gmm = float(10 ** cross)
    print(f"\nprahy: prevalencni {thr_prev:.1f}  |  GMM {thr_gmm:.1f}  "
          f"(GMM stredy 10^{lo_m:.2f}={10**lo_m:.0f}, 10^{hi_m:.2f}={10**hi_m:.0f})",
          flush=True)

    results = dict(_meta=dict(
        label_source="max clc_A per track (AVG dataset, sigma 1.6423)",
        thr_prev=round(thr_prev, 2), thr_gmm=round(thr_gmm, 2),
        gmm_means=[round(float(10 ** lo_m), 1), round(float(10 ** hi_m), 1)],
        si_prevalence_none_endobs=round(float(b0["y_si"].mean()), 4)))

    for name, thr in (("prev", thr_prev), ("gmm", thr_gmm)):
        results[name] = dict(threshold=round(thr, 2), configs={})
        for tag, b in built.items():
            y = (b["amp"] >= thr).astype(int)
            rec = evaluate(b["X"], y, b["film"], b["band"], b["folds"])
            rec["kappa_vs_SI"] = round(float(cohen_kappa_score(b["y_si"], y)), 4)
            results[name]["configs"][tag] = rec
            p = rec["pooled"]
            print(f"[{name}] {tag:>18}: prev {p['prevalence']:.3f}  AUC {p['auc']:.3f}  "
                  f"wb {rec['wb_auc']:.3f}  sens {p['sensitivity']:.2f} "
                  f"spec {p['specificity']:.2f}  kappa(SI) {rec['kappa_vs_SI']:.2f}",
                  flush=True)

    # ---- sweep prahu (none_endobs) ----
    sweep = []
    for q in range(50, 96, 5):
        thr = float(np.percentile(s_max[np.isfinite(s_max)], q))
        y = (b0["amp"] >= thr).astype(int)
        rec = evaluate(b0["X"], y, b0["film"], b0["band"], b0["folds"])
        p = rec["pooled"]
        sweep.append(dict(pctl=q, threshold=round(thr, 1),
                          prevalence=p["prevalence"], auc=p["auc"],
                          wb_auc=rec["wb_auc"], sensitivity=p["sensitivity"],
                          specificity=p["specificity"]))
        print(f"[sweep] pctl {q}: thr {thr:8.1f}  prev {p['prevalence']:.3f}  "
              f"AUC {p['auc']:.3f}  wb {rec['wb_auc']:.3f}", flush=True)
    results["sweep_none_endobs"] = sweep

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nJSON -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
