# CMEpython -- Python port mereni intenzity dynaminu z cmeAnalysis.
# Copyright (C) 2026 Ruslan Guliev
#
# Odvozeno z cmeAnalysis (DanuserLab, UT Southwestern), GPL-3.0.
# Tento program je svobodny software: muzete jej sirit a upravovat podle
# podminek GNU General Public License verze 3 nebo pozdejsi. Sirenu je
# BEZ JAKEKOLI ZARUKY. Viz soubor LICENSE.
import os, sys, time
import numpy as np, tifffile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmepython import measure_movie, movie_layout
from cmepython.psf_calibration import detect_candidates

P = "reconstructed registered/U2OS_DYNAMIN_MSTAYGOLD_GREEN_SNAP_CLC_RED_DNMsiRNA_16_RR.tif"
S_SLAVE, S_MASTER = 2.6424, 1.4187

def main():
    print(f"CPU jader: {os.cpu_count()}\n")
    with tifffile.TiffFile(P) as tf:
        T, C, Y, X, page = movie_layout(tf)
        coords = []
        for t in range(0, 60, 2):
            ys, xs = detect_candidates(tf.pages[page(t, 0)].asarray(), S_MASTER, k=5.0, max_spots=250)
            coords += [[t, y, x] for y, x in zip(ys, xs)]
    coords = np.array(coords, float)
    print(f"{len(coords)} detekci pres {len(np.unique(coords[:,0]))} snimku")
    print(f"{'rezim':>15}{'cas [s]':>10}{'ms/det':>10}{'zrychleni':>12}{'shoda':>12}")
    print("-" * 59)
    t0 = time.time(); r0 = measure_movie(P, coords, S_SLAVE, S_MASTER, slave_channel=2)
    ts = time.time() - t0
    print(f"{'serialne':>15}{ts:>10.2f}{1000*ts/len(coords):>10.2f}{'1.0x':>12}{'-':>12}")
    for w in (4, 0):
        t0 = time.time(); r = measure_movie(P, coords, S_SLAVE, S_MASTER, slave_channel=2, workers=w)
        tp = time.time() - t0
        d = np.nanmax(np.abs(r0['A'] - r['A']))
        print(f"{(str(w)+' workeru' if w else 'vsechna jadra'):>15}{tp:>10.2f}"
              f"{1000*tp/len(coords):>10.2f}{ts/tp:>11.1f}x{d:>12.1e}")
    tot = 23 * 151 * 250
    print(f"\ncely dataset (23 x 151 x ~250 = {tot:,} detekci):")
    print(f"  serialne  ~{tot*ts/len(coords)/3600:.1f} h")

if __name__ == "__main__":
    main()
