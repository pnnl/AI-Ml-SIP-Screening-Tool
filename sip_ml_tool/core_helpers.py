# Extracted from SIP_ML_Tool.ipynb main workflow cell.
# Small parsing, frequency-band, peak, and rule-flag helpers.

def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def parse_filename(path: Path):
    """
    Parse field filenames like: 20240212_col103-1_1mMNaCl_static.csv
    """
    name = path.stem
    m = re.search(r"([0-9]{8}).*->(col[0-9]+(->:-[0-9]+)->)", name, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse filename into date & column ID: {path.name}")

    date_str = m.group(1)
    col_raw = m.group(2).lower()
    col_dash = col_raw.replace("col", "")
    col_num = int(col_dash.split("-")[0])

    return {
        "file_date": pd.to_datetime(date_str, format="%Y%m%d"),
        "column_name": col_raw,
        "column_number": col_num,
        "column_number_dash": col_dash,
        "original_file_name": path.name,
    }


def find_header_line(path: Path, max_lines: int = 80):
    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if i > max_lines:
                break
            if "Frequency" in line and "Phase" in line:
                return i
    return None


def read_header_date(path: Path):
    """
    HPE header Date,DD/MM/YYYY,
    """
    date_str = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("Date,"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    date_str = parts[1]
                break
    if date_str is None:
        return pd.NaT
    return pd.to_datetime(date_str, format="%d/%m/%Y", errors="coerce")


def band_mask(freq, fmin, fmax):
    return (freq >= fmin) & (freq <= fmax)


def phase_jaggedness(phase_vals):
    phase_vals = np.asarray(phase_vals)
    if len(phase_vals) < 2:
        return np.nan
    diffs = np.diff(phase_vals)
    return float(np.nanmax(np.abs(diffs)))


def slope_vs_logf(freq, y):
    freq = np.asarray(freq)
    y = np.asarray(y)
    mask = (freq > 0) & np.isfinite(freq) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xf = np.log10(freq[mask])
    yf = y[mask]
    A = np.vstack([xf, np.ones_like(xf)]).T
    m, _ = np.linalg.lstsq(A, yf, rcond=None)[0]
    return float(m)


def pick_peak(g: pd.DataFrame):
    """
    Peak-picking per UniqueID, using Phase_mrad > 0.
    """
    g = g[g["Phase_mrad"] > 0].copy()
    if g.empty:
        return np.nan, np.nan

    col = int(g["column_number"].iloc[0])

    if col == 111:
        g = g[(g["Freq"] >= 0.1) & (g["Freq"] <= 100.0)]
    elif col == 103:
        g = g[(g["Freq"] > 100.0) & (g["Freq"] <= 9000.0)]
    else:
        g = g[g["Freq"] > 0.0]

    if g.empty:
        return np.nan, np.nan

    idx = g["Phase_mrad"].idxmax()
    return float(g.loc[idx, "Freq"]), float(g.loc[idx, "Phase_mrad"])


# ==============================================================================
# RULES (computed from raw curve)
# ==============================================================================

def _has_run(mask_bool, run_len=2) -> bool:
    """True if there is a contiguous run of True values of length >= run_len."""
    count = 0
    for v in mask_bool:
        if bool(v):
            count += 1
            if count >= run_len:
                return True
        else:
            count = 0
    return False


def _count_isolated_true(mask_bool) -> int:
    """
    Counts True values that are NOT adjacent to another True.
    Example: [T,F,T] => 2 isolated
             [T,T,F] => 0 isolated (it's a run)
             [F,T,T,F,T] => 1 isolated (last T), first TT is a run
    """
    mask = np.asarray(mask_bool, dtype=bool)
    if mask.size == 0:
        return 0
    isolated = 0
    for i, v in enumerate(mask):
        if not v:
            continue
        left = mask[i - 1] if i - 1 >= 0 else False
        right = mask[i + 1] if i + 1 < mask.size else False
        if (not left) and (not right):
            isolated += 1
    return isolated


def compute_rule_flags_from_group(
    g: pd.DataFrame,
    lowfreq_relax_hz: float = LOWFREQ_RELAX_HZ,
    dphase_thr_low_mrad: float = DPHASE_THR_LOW_MRAD,
    dphase_thr_mid_mrad: float = DPHASE_THR_MID_MRAD,
    dphase_thr_hf_mrad: float = DPHASE_THR_HF_MRAD,
    jagged_run_len: int = JAGGED_RUN_LEN,
    max_isolated_spikes_allowed: int = MAX_ISOLATED_SPIKES_ALLOWED,
):
    """
    Computes rule flags on a single spectrum group (one UniqueID).
    Uses Phase_mrad (your sign convention).
    """
    g = g.sort_values("Freq")
    f = g["Freq"].to_numpy()
    ph = g["Phase_mrad"].to_numpy()

    out = {
        "rule_neg_lt_1k": False,
        "rule_neg_1k_10k": False,
        "rule_jagged_run": False,
        "rule_isolated_spikes": 0,
        "rule_isolated_spikes_allowed": False,
        "rule_big_jump_count": 0,
        "max_phase_gt_1k_mrad": float(np.nanmax(ph)) if len(ph) else np.nan,
    }

    if len(f) < 3:
        return out

    # Negative phase constraints (basic, later replaced by tolerant flags in patch v2)
    out["rule_neg_lt_1k"] = bool(np.any((f < 1000.0) & (ph < 0.0)))
    out["rule_neg_1k_10k"] = bool(np.any((f >= 1000.0) & (f <= 10000.0) & (ph < 0.0)))

    # Δphase between adjacent points
    dph = np.diff(ph)
    mid_f = (f[:-1] + f[1:]) / 2.0

    # Band-specific thresholds
    thr = np.full_like(mid_f, fill_value=dphase_thr_mid_mrad, dtype=float)
    thr[mid_f < lowfreq_relax_hz] = dphase_thr_low_mrad
    thr[(mid_f >= 1000.0) & (mid_f <= 10000.0)] = dphase_thr_hf_mrad

    big_jump = np.abs(dph) >= thr
    out["rule_big_jump_count"] = int(np.sum(big_jump))

    # Jagged RUN: needs >1 neighboring big jumps
    out["rule_jagged_run"] = bool(_has_run(big_jump, run_len=jagged_run_len))

    # Isolated spikes: singular OR a few non-neighboring spikes
    isolated_cnt = _count_isolated_true(big_jump)
    out["rule_isolated_spikes"] = int(isolated_cnt)
    out["rule_isolated_spikes_allowed"] = bool((isolated_cnt > 0) and (isolated_cnt <= max_isolated_spikes_allowed))

    # Max phase >1k for high-phase review rule
    out["max_phase_gt_1k_mrad"] = float(np.nanmax(ph[f >= 1000.0])) if np.any(f >= 1000.0) else float(np.nan)
    out["max_abs_phase_mrad"] = float(np.nanmax(np.abs(ph))) if len(ph) else float(np.nan)
    out["max_abs_phase_gt_1k_mrad"] = float(np.nanmax(np.abs(ph[f >= 1000.0]))) if np.any(f >= 1000.0) else float(np.nan)

    return out
