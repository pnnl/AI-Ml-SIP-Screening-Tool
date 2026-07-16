# ------------------------------------------------------------------------------
# Main QC workflow
# ------------------------------------------------------------------------------
"""
This cell runs the full SIP QC workflow.

Main steps:
1. Read the SIP CSV files.
2. Build one feature row per spectrum.
3. Train the CatBoost model on labeled spectra.
4. Predict classes for all spectra.
5. Apply the QC rules to assign Good, Review, or Bad.
6. Save the output tables, figures, and gallery files.
"""

from __future__ import annotations

from pathlib import Path
import re
import datetime
import math
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from catboost import CatBoostClassifier, Pool

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor


# ------------------------------------------------------------------------------
# PowerPoint gallery style
# ------------------------------------------------------------------------------

PPT_THEME = {
    "good": {
        "accent": RGBColor(132, 176, 138),
        "accent_dark": RGBColor(74, 124, 79),
    },
    "review": {
        "accent": RGBColor(238, 208, 139),
        "accent_dark": RGBColor(154, 116, 34),
    },
    "bad": {
        "accent": RGBColor(240, 191, 187),
        "accent_dark": RGBColor(156, 73, 73),
    },
    "default": {
        "accent": RGBColor(190, 198, 210),
        "accent_dark": RGBColor(90, 105, 130),
    },
}

def _ppt_theme(class_name: str) -> dict:
    return PPT_THEME.get(str(class_name).lower(), PPT_THEME["default"])

def _apply_font(paragraph, size_pt: float, *, bold: bool = False,
                color=None, name: str = "Arial", align=None) -> None:
    if align is not None:
        paragraph.alignment = align
    paragraph.font.name = name
    paragraph.font.size = Pt(size_pt)
    paragraph.font.bold = bold
    if color is not None:
        paragraph.font.color.rgb = color

# ==============================================================================
# Settings to edit
# ==============================================================================

BASE_DIR = Path(".").resolve()

# Field folders (required)
FOLDERS = {
    "good": BASE_DIR / "good data",
    "bad":  BASE_DIR / "bad data",
    # Optional:
    "all":  BASE_DIR / "all data",
}

# Optional HPE base
HPE_BASE = BASE_DIR / "Raw SIP Data" / "HPE SIP Data"
HPE_RECURSIVE = True
WRITE_DISCOVERY_AUDIT = True

# Output folders
OUT_DIR = BASE_DIR / "Final Results"
FIG_DIR = OUT_DIR / "figs"
SPEC_DIR = OUT_DIR / "spectra_pngs"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)
SPEC_DIR.mkdir(exist_ok=True)

# PowerPoint gallery layout
GALLERY_COLS = 3
GALLERY_ROWS = 2
SPECTRA_PER_SLIDE = GALLERY_COLS * GALLERY_ROWS

# Review logic around CatBoost threshold (optional)
NEAR_THRESHOLD_BAND = 0.05     # if |p_good - thr| < band => REVIEW candidate
CAUTIOUS_THRESHOLD = 0.85      # if max prob < this => REVIEW candidate

# KK screening settings
USE_KK_GATE = True

# REVIEW-first: KK triggers send spectra to REVIEW (not BAD) unless enabled below
KK_HARD_BAD_ENABLED = False

# Minimum labeled GOOD spectra needed to calibrate thresholds robustly
KK_MIN_GOOD_FOR_CAL = 20

# REVIEW thresholds derived from labeled GOOD distribution
KK_Q_REVIEW = 0.98          # chi2/rms above this quantile => KK_REVIEW
KK_MU_Q_REVIEW = 0.02       # mu below this quantile => KK_REVIEW

# EXTREME thresholds (only used if KK_HARD_BAD_ENABLED=True)
KK_Q_BAD = 0.995            # chi2/rms above this quantile => candidate KK_BAD
KK_MU_Q_BAD = 0.005         # mu below this quantile => KK_BAD

# For KK_BAD classification: require BOTH (chi2 & rms) extreme, OR mu extremely low
KK_BAD_REQUIRE_BOTH_CHI_AND_RMS = True


# QC rule settings
# Δphase thresholds in Phase_mrad jump between neighboring points.
# We now allow different thresholds by frequency band:
LOWFREQ_RELAX_HZ = 0.1

DPHASE_THR_LOW_MRAD = 450.0        # relaxed threshold for <0.1 Hz region
DPHASE_THR_MID_MRAD = 250.0        # default threshold for most mid-range
DPHASE_THR_HF_MRAD  = 320.0        # slightly more forgiving in 1k–10k to avoid HF noise over-penalizing

# "Few isolated spikes" allowance: if spikes are NOT neighboring and count <= this -> treat as review, not jagged-run
MAX_ISOLATED_SPIKES_ALLOWED = 2    # "a few" isolated spikes

# How many consecutive big jumps constitutes a jagged RUN trigger
JAGGED_RUN_LEN = 2                 # >1 neighboring instances

# High phase review threshold vs reference (robust):
HIGH_PHASE_MULT = 1.30             # notably larger than typical => REVIEW
HIGH_PHASE_REF_Q = 0.95            # reference quantile (per similarity group)

# Similarity grouping for high-phase reference.
# Use column_number when available; also include exp_group so "similar" is tighter for HPE data.
HIGH_PHASE_GROUP_COLS = ["source", "column_number", "exp_group"]

# Matplotlib
PLOT_FIGSIZE = (3.10, 2.20)
PLOT_DPI = 300

CAPTION_FONT_UID = 7.2
CAPTION_FONT_META = 6.2

FORCE_STRETCH_TO_CELL = True
