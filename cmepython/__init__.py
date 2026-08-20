# CMEpython -- Python port mereni intenzity dynaminu z cmeAnalysis.
# Copyright (C) 2026 Ruslan Guliev
#
# Odvozeno z cmeAnalysis (DanuserLab, UT Southwestern), GPL-3.0.
# Tento program je svobodny software: muzete jej sirit a upravovat podle
# podminek GNU General Public License verze 3 nebo pozdejsi. Sirenu je
# BEZ JAKEKOLI ZARUKY. Viz soubor LICENSE.
"""CMEpython — Python port of the cmeAnalysis (DanuserLab, MATLAB) CME pipeline.

Runs on all platforms: pure numpy/scipy, no compiled extensions, no MATLAB.
"""

__version__ = "0.1.0"

from .slave_intensity import (
    dynamin_intensity,
    dynamin_intensity_gap,
    fit_gaussian_2d_point,
)
from .measure import measure_frame, measure_coords, measure_movie
from .psf_calibration import estimate_psf_sigma, apply_sigma_clamp, detect_candidates
from .edf_scaling import scale_edfs, apply_scaling, ecdf

__all__ = [
    # verny port jednoho bodu
    "dynamin_intensity", "dynamin_intensity_gap", "fit_gaussian_2d_point",
    # davkove mereni a IO
    "measure_frame", "measure_coords", "measure_movie",
    # kalibrace sirky PSF
    "estimate_psf_sigma", "apply_sigma_clamp", "detect_candidates",
    # skalovani napric filmy
    "scale_edfs", "apply_scaling", "ecdf",
]
