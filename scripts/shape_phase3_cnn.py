#!/usr/bin/env python3
"""Patro 3 (zaverecna kontrola): mala 1D konvolucni sit nad surovymi stopami.

Otazka: najde sit v kombinaci vysky a tvaru neco, co 9 rucnich cisel
(patro 2) nezachytilo? Vstup na drahu = posledni T=40 snimku, 4 kanaly:
  dyn_snr  amplituda dynaminu / jeji nejistota (A/sigma, jednotky sigma)
  dyn_sig  vysledek lokalniho testu (0/1)
  clc_snr  totez pro klatrin
  mask     1 = skutecny snimek, 0 = vypln pred vznikem
Sit: Conv1d(4->16,k5) -> ReLU -> Conv1d(16->32,k5) -> ReLU -> global
max+mean pool -> FC -> 1. ~20k parametru.

Vyhodnoceni jako patra 1-2: OOF s foldy po filmech, AUC pooled +
within-band, oba rezimy stitku. PARALELNE: kazdy fold vlastni proces
(--workers, default 5), v procesu 2 torch vlakna; 3 seedy prumerovane.

    python3 scripts/shape_phase3_cnn.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANDS = [(6, 9), (10, 19), (20, 39), (40, 10 ** 9)]
T = 40
SEEDS = (0, 1, 2)


def auc_rank(score, y):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(score).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wb_auc(score, y, tlen):
    num = den = 0.0; per = []
    for lo, hi in BANDS:
        m = (tlen >= lo) & (tlen <= hi)
        a = auc_rank(score[m], y[m])
        n1 = int(y[m].sum()); w = n1 * (int(m.sum()) - n1)
        per.append(a)
        if np.isfinite(a):
            num += a * w; den += w
    return (num / den if den else np.nan), per


def load_data():
    seqs, meta = [], []
    for p in sorted(glob.glob(os.path.join(ROOT, "lr registered", "measured",
                                           "*-lr-trajectories-dynamin.csv"))):
        movie = int(os.path.basename(p).split("_DNMsiRNA_")[1].split("-")[0])
        d = pd.read_csv(p, usecols=["particle", "frame", "cls", "dnm_A",
                                    "dnm_A_pstd", "dnm_signif",
                                    "clc_A", "clc_A_pstd"])
        for pid, t in d.groupby("particle"):
            t = t.sort_values("frame")
            L = len(t)
            if L < 6:
                continue
            def snr(a, s):
                a = a.to_numpy(float); s = s.to_numpy(float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    v = np.where(s > 0, a / s, 0.0)
                return np.clip(np.nan_to_num(v), -5.0, 50.0) / 10.0
            ch = np.zeros((4, T), np.float32)
            k = min(L, T)
            ch[0, T - k:] = snr(t.dnm_A, t.dnm_A_pstd)[-k:]
            ch[1, T - k:] = t.dnm_signif.to_numpy(float)[-k:]
            ch[2, T - k:] = snr(t.clc_A, t.clc_A_pstd)[-k:]
            ch[3, T - k:] = 1.0
            cls = t.cls.to_numpy(float)
            seqs.append(ch)
            meta.append((movie, int(pid), L, int(np.nanmax(cls) > 0.7),
                         1 if (cls > 0.7).sum() >= 3
                         else (0 if np.nanmax(cls) < 0.65 else -1)))
    X = np.stack(seqs)
    m = pd.DataFrame(meta, columns=["movie", "particle", "track_len",
                                    "si_plain", "si_hard"])
    return X, m


def train_fold(args):
    """Jeden fold: natrenuje sit pro kazdy seed a vrati prumerne OOF skore."""
    X, y, tr_idx, te_idx, fold_id = args
    n_ch = X.shape[1]
    import torch
    import torch.nn as nn
    torch.set_num_threads(2)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv1d(n_ch, 16, 5, padding=2)
            self.c2 = nn.Conv1d(16, 32, 5, padding=2)
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            h = torch.relu(self.c1(x))
            h = torch.relu(self.c2(h))
            g = torch.cat([h.max(dim=2).values, h.mean(dim=2)], dim=1)
            return self.fc(g).squeeze(1)

    Xtr = torch.from_numpy(X[tr_idx]); ytr = torch.from_numpy(y[tr_idx].astype(np.float32))
    Xte = torch.from_numpy(X[te_idx])
    pos_w = torch.tensor((len(ytr) - ytr.sum()) / max(ytr.sum(), 1.0))
    scores = np.zeros(len(te_idx))
    for seed in SEEDS:
        torch.manual_seed(seed)
        net = Net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        n = len(ytr)
        for epoch in range(20):
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                b = perm[i:i + 256]
                opt.zero_grad()
                loss = loss_fn(net(Xtr[b]), ytr[b])
                loss.backward()
                opt.step()
        with torch.no_grad():
            scores += torch.sigmoid(net(Xte)).numpy()
    return fold_id, te_idx, scores / len(SEEDS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--channels", choices=["all", "dyn", "clc"], default="all",
                    help="all = dyn+clc; dyn = jen dynamin+maska; clc = jen klatrin+maska")
    args = ap.parse_args()
    from sklearn.model_selection import GroupKFold

    X, m = load_data()
    CH = {"all": [0, 1, 2, 3], "dyn": [0, 1, 3], "clc": [2, 3]}[args.channels]
    X = np.ascontiguousarray(X[:, CH, :])
    print(f"kanaly: {args.channels} -> {CH}", flush=True)
    print(f"drah: {len(m):,}, tensor {X.shape}", flush=True)
    out = {"channels": args.channels}
    for label_name, ycol in (("bezne stitky", "si_plain"),
                             ("zpevnene stitky", "si_hard")):
        keep = (m[ycol] >= 0).to_numpy()
        Xs, ms = X[keep], m[keep].reset_index(drop=True)
        y = ms[ycol].to_numpy(int)
        folds = list(GroupKFold(5).split(Xs, y, ms.movie.to_numpy()))
        jobs = [(Xs, y, tr, te, i) for i, (tr, te) in enumerate(folds)]
        scores = np.full(len(y), np.nan)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for fold_id, te_idx, s in pool.map(train_fold, jobs):
                scores[te_idx] = s
                print(f"  [{label_name}] fold {fold_id} hotov", flush=True)
        pooled = auc_rank(scores, y)
        wb, per = wb_auc(scores, y, ms.track_len.to_numpy(int))
        out[ycol] = dict(pooled=pooled, wb=wb, bands=per)
        print(f"{label_name}: n={len(y):,}  pooled {pooled:.3f}  wb {wb:.3f}  "
              f"pasma " + "  ".join(f"{a:.3f}" for a in per), flush=True)

    print("\nsrovnani wb: bezne stitky - delka 0.561, GBM D+A 0.579; "
          "zpevnene - delka 0.622, GBM D+A 0.670")
    od = os.path.join(ROOT, "out", "shape_phase3")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, f"results_{args.channels}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"JSON -> {od}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
