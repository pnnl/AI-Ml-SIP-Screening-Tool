# Extracted from SIP_ML_Tool.ipynb main workflow cell.
# Shape/QC feature augmentation and intermediate override updates.

NEG_TOL_LT1K_MRAD = 1.5
NEG_TOL_1K_10K_MRAD = 1.5
NEG_COUNT_MIN_LT1K = 2
NEG_COUNT_MIN_1K_10K = 2
NEG_RUN_MIN = 2
NEG_DEEP_LT1K_MRAD = -5.0

OSC_DELTA_EPS_MRAD = 1.0

# Absolute phase plausibility gates (mrad)
PHASE_ABS_REVIEW_MRAD = 400.0
PHASE_ABS_BAD_MRAD = 800.0

RESID_SMOOTH_WIN = 5
RESID_RMS_FRAC_THR = 0.025
FLIP_FRAC_THR = 0.42
MIDBAND_FLIP_FRAC_THR = 0.38

TAIL_FREQ_MIN_HZ = 1000.0
TAIL_LAST_N_MIN = 8
TAIL_FLIPS_MIN = 4
TAIL_SPIKE_RATIO_THR = 10.0
TAIL_RMS_FRAC_THR = 0.060

SMOOTH_RESCUE_RESID_RMS_MAX = 0.012
SMOOTH_RESCUE_FLIP_FRAC_MAX = 0.18
SMOOTH_RESCUE_PB_MIN = 0.90
SMOOTH_RESCUE_TO_GOOD = False

HF_NEG_ONLY_KEEP_GOOD_PG = 0.985
HF_NEG_ONLY_KEEP_BAD_PB = 0.985


def _max_run_len(mask_bool) -> int:
    count = 0
    best = 0
    for v in np.asarray(mask_bool, dtype=bool):
        if v:
            count += 1
            best = max(best, count)
        else:
            count = 0
    return int(best)


def _sign_flip_count(arr, eps=0.0) -> int:
    a = np.asarray(arr, dtype=float)
    s = np.zeros_like(a, dtype=int)
    s[a > eps] = 1
    s[a < -eps] = -1
    s = s[s != 0]
    if s.size < 2:
        return 0
    return int(np.sum(s[1:] != s[:-1]))


def _rolling_median_smooth(y, win=5):
    yy = pd.Series(np.asarray(y, dtype=float))
    med = yy.rolling(window=win, center=True, min_periods=1).median()
    sm = med.rolling(window=3, center=True, min_periods=1).mean()
    return sm.to_numpy(dtype=float)


def _safe_phase_range(y, floor=1.0):
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return float(floor)
    r = float(np.nanmax(y) - np.nanmin(y))
    return max(r, float(floor))


def _compute_shape_qc_metrics_from_group(g: pd.DataFrame) -> dict:
    g = g.sort_values("Freq")
    f = g["Freq"].to_numpy(dtype=float)
    ph = g["Phase_mrad"].to_numpy(dtype=float)

    out = {
        "neg_lt1k_count": 0,
        "neg_1k_10k_count": 0,
        "neg_lt1k_run": 0,
        "neg_1k_10k_run": 0,
        "deep_neg_lt1k": False,
        "rule_neg_lt_1k_tolerant": False,
        "rule_neg_1k_10k_tolerant": False,
        "resid_rms_mrad": 0.0,
        "resid_rms_frac": 0.0,
        "flip_count": 0,
        "flip_frac": 0.0,
        "flip_count_mid": 0,
        "flip_frac_mid": 0.0,
        "rule_oscillatory_jagged": False,
        "tail_resid_rms_frac": 0.0,
        "tail_flip_count": 0,
        "tail_spike_ratio": 0.0,
        "tail_max_jump_mrad": 0.0,
        "rule_tail_instability": False,
        "rule_smooth_candidate": False,
    }

    if f.size < 4 or ph.size < 4:
        return out

    # Tolerant negative-phase rules (deadband + persistence)
    mask_lt1k = (f < 1000.0)
    mask_1k_10k = (f >= 1000.0) & (f <= 10000.0)

    neg_lt1k = mask_lt1k & (ph < NEG_TOL_LT1K_MRAD)
    neg_1k_10k = mask_1k_10k & (ph < NEG_TOL_1K_10K_MRAD)

    neg_lt1k_count = int(np.sum(neg_lt1k))
    neg_1k_10k_count = int(np.sum(neg_1k_10k))
    neg_lt1k_run = _max_run_len(neg_lt1k)
    neg_1k_10k_run = _max_run_len(neg_1k_10k)

    deep_neg_lt1k = False
    if np.any(mask_lt1k):
        deep_neg_lt1k = bool(np.nanmin(ph[mask_lt1k]) <= NEG_DEEP_LT1K_MRAD)

    out.update({
        "neg_lt1k_count": neg_lt1k_count,
        "neg_1k_10k_count": neg_1k_10k_count,
        "neg_lt1k_run": neg_lt1k_run,
        "neg_1k_10k_run": neg_1k_10k_run,
        "deep_neg_lt1k": deep_neg_lt1k,
        "rule_neg_lt_1k_tolerant": bool(
            deep_neg_lt1k or
            (neg_lt1k_count >= NEG_COUNT_MIN_LT1K) or
            (neg_lt1k_run >= NEG_RUN_MIN)
        ),
        "rule_neg_1k_10k_tolerant": bool(
            (neg_1k_10k_count >= NEG_COUNT_MIN_1K_10K) or
            (neg_1k_10k_run >= NEG_RUN_MIN)
        ),
    })

    # Whole-curve oscillation / jaggedness on residuals + sign flips
    y_smooth = _rolling_median_smooth(ph, win=RESID_SMOOTH_WIN)
    resid = ph - y_smooth
    phase_range = _safe_phase_range(ph)

    resid_rms = float(np.sqrt(np.mean(np.square(resid)))) if resid.size else 0.0
    resid_rms_frac = resid_rms / phase_range

    dph = np.diff(ph)
    flip_count = _sign_flip_count(dph, eps=OSC_DELTA_EPS_MRAD)
    valid_steps = int(np.sum(np.abs(dph) > OSC_DELTA_EPS_MRAD))
    flip_frac = (flip_count / max(valid_steps - 1, 1)) if valid_steps >= 2 else 0.0

    mid_mask = (f >= 1.0) & (f <= 1000.0)
    if np.sum(mid_mask) >= 4:
        dph_mid = np.diff(ph[mid_mask])
        flip_count_mid = _sign_flip_count(dph_mid, eps=OSC_DELTA_EPS_MRAD)
        valid_steps_mid = int(np.sum(np.abs(dph_mid) > OSC_DELTA_EPS_MRAD))
        flip_frac_mid = (flip_count_mid / max(valid_steps_mid - 1, 1)) if valid_steps_mid >= 2 else 0.0
    else:
        flip_count_mid = 0
        flip_frac_mid = 0.0

    rule_osc = bool(
        (resid_rms_frac >= RESID_RMS_FRAC_THR and flip_frac >= FLIP_FRAC_THR) or
        (flip_frac_mid >= MIDBAND_FLIP_FRAC_THR and resid_rms_frac >= (0.7 * RESID_RMS_FRAC_THR))
    )

    out.update({
        "resid_rms_mrad": resid_rms,
        "resid_rms_frac": resid_rms_frac,
        "flip_count": int(flip_count),
        "flip_frac": float(flip_frac),
        "flip_count_mid": int(flip_count_mid),
        "flip_frac_mid": float(flip_frac_mid),
        "rule_oscillatory_jagged": rule_osc,
    })

    # HF tail instability
    tail_mask = (f >= TAIL_FREQ_MIN_HZ)
    if np.sum(tail_mask) < 4:
        tail_mask = np.zeros_like(f, dtype=bool)
        tail_mask[max(0, len(f) - TAIL_LAST_N_MIN):] = True

    if np.sum(tail_mask) >= 4:
        ph_tail = ph[tail_mask]
        y_tail_smooth = _rolling_median_smooth(ph_tail, win=min(5, max(3, len(ph_tail)//2*2+1)))
        resid_tail = ph_tail - y_tail_smooth
        dph_tail = np.diff(ph_tail)

        tail_resid_rms = float(np.sqrt(np.mean(np.square(resid_tail)))) if resid_tail.size else 0.0
        tail_resid_rms_frac = tail_resid_rms / phase_range
        tail_flip_count = _sign_flip_count(dph_tail, eps=OSC_DELTA_EPS_MRAD)
        tail_max_jump = float(np.max(np.abs(dph_tail))) if dph_tail.size else 0.0

        base_mask = (f >= 1.0) & (f <= 1000.0)
        if np.sum(base_mask) >= 4:
            base_d = np.diff(ph[base_mask])
        else:
            base_d = dph
        base_jump_scale = float(np.median(np.abs(base_d))) if base_d.size else 1.0
        base_jump_scale = max(base_jump_scale, 1.0)
        tail_spike_ratio = tail_max_jump / base_jump_scale
    else:
        tail_resid_rms_frac = 0.0
        tail_flip_count = 0
        tail_spike_ratio = 0.0
        tail_max_jump = 0.0

    rule_tail = bool(
        (tail_flip_count >= TAIL_FLIPS_MIN and tail_spike_ratio >= TAIL_SPIKE_RATIO_THR) or
        (tail_resid_rms_frac >= TAIL_RMS_FRAC_THR and tail_flip_count >= max(2, TAIL_FLIPS_MIN - 1))
    )

    out.update({
        "tail_resid_rms_frac": float(tail_resid_rms_frac),
        "tail_flip_count": int(tail_flip_count),
        "tail_spike_ratio": float(tail_spike_ratio),
        "tail_max_jump_mrad": float(tail_max_jump),
        "rule_tail_instability": rule_tail,
    })

    out["rule_smooth_candidate"] = bool(
        (resid_rms_frac <= SMOOTH_RESCUE_RESID_RMS_MAX) and
        (flip_frac <= SMOOTH_RESCUE_FLIP_FRAC_MAX) and
        (not rule_osc) and
        (not rule_tail)
    )

    return out


# Preserve originals (idempotent patch)
if "_orig_build_feature_table" not in globals():
    _orig_build_feature_table = build_feature_table
if "_orig_apply_override" not in globals():
    _orig_apply_override = apply_override


def build_feature_table(raw: pd.DataFrame):
    """
    Wrapper around original build_feature_table():
    - keeps all original features
    - adds shape-QC metrics from raw spectra groups
    - replaces brittle negative-phase flags with tolerant versions
    """
    features = _orig_build_feature_table(raw)

    # Compute extra metrics directly from raw grouped spectra
    extra_rows = []
    for uid, g in raw.groupby("UniqueID", sort=False):
        m = _compute_shape_qc_metrics_from_group(g)
        m["UniqueID"] = uid
        extra_rows.append(m)
    extra = pd.DataFrame(extra_rows)

    if not extra.empty:
        features = features.merge(extra, on="UniqueID", how="left")

        # Fill missing numerics/bools
        for c in features.columns:
            if c.startswith("rule_") and features[c].dtype != object:
                features[c] = features[c].fillna(0)

        # Replace old brittle negatives with tolerant versions when available
        if "rule_neg_lt_1k_tolerant" in features.columns:
            features["rule_neg_lt_1k"] = features["rule_neg_lt_1k_tolerant"].fillna(False).astype(int)
        if "rule_neg_1k_10k_tolerant" in features.columns:
            features["rule_neg_1k_10k"] = features["rule_neg_1k_10k_tolerant"].fillna(False).astype(int)

    return features






def apply_override(features: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper around original apply_override():
    - preserves original model + rule features
    - keeps GOOD reserved for clean + trusted spectra
    - uses REVIEW for clean-but-weak / ambiguous / provenance-like spectra
    - reserves BAD for structurally unstable / nonphysical spectra
    - adds audit reason columns
    """
    out = _orig_apply_override(features.copy())

    # Ensure expected columns exist
    for c, default in [
        ("pred_final", None),
        ("pred_binary", None),
        ("prob_good", np.nan),
        ("prob_bad", np.nan),
        ("prob_max", np.nan),
        ("rule_neg_lt_1k", 0),
        ("rule_neg_1k_10k", 0),
        ("rule_jagged_run", 0),
        ("rule_big_jump_count", 0),
        ("rule_isolated_spikes", 0),
        ("rule_isolated_spikes_allowed", 0),
        ("rule_high_phase_gt_1k", 0),
        ("rule_combo_bad", 0),
        ("rule_near_thr", 0),
        ("rule_low_conf", 0),
        ("rule_oscillatory_jagged", 0),
        ("rule_tail_instability", 0),
        ("rule_smooth_candidate", 0),
        ("deep_neg_lt1k", False),
        ("max_phase_gt_1k_mrad", np.nan),
        ("max_abs_phase_mrad", np.nan),
        ("max_abs_phase_gt_1k_mrad", np.nan),
        ("tail_resid_rms_frac", np.nan),
        ("tail_flip_count", 0),
        ("tail_spike_ratio", np.nan),
        ("kk_status", "KK_NA"),
        ("kk_flag_review", 0),
        ("kk_flag_bad", 0),
    ]:
        if c not in out.columns:
            out[c] = default

    if "pred_final_base" not in out.columns:
        out["pred_final_base"] = out["pred_final"].astype(str)

    # Tunables: bias BAD toward true structural failure only
    GOOD_STRONG_PG = 0.997
    GOOD_TAIL_ONLY_PG = 0.999
    GOOD_NEG_ONLY_PG = 0.998
    GOOD_MODEL_ONLY_PG = 0.996

    MODEL_ONLY_BAD_GOOD_MAXABS_1K = 140.0
    MODEL_ONLY_BAD_REVIEW_MAXABS_1K = 520.0

    NEG_LT1K_GOOD_MAXABS_1K = 100.0
    NEG_LT1K_REVIEW_MAXABS_1K = 280.0

    HF_NEG_GOOD_MAXABS_1K = 35.0
    HF_NEG_REVIEW_MAXABS_1K = 160.0

    TAIL_GOOD_MAXABS_1K = 120.0
    TAIL_REVIEW_MAXABS_1K = 520.0

    TAIL_BAD_FLIPS_MIN = 5
    TAIL_BAD_SPIKE_RATIO_MIN = 12.0
    TAIL_BAD_RMS_FRAC_MIN = 0.09
    BIGJUMPS_BAD_MIN = 7
    ABS_BAD_MRAD = max(PHASE_ABS_BAD_MRAD, 1000.0)

    reasons = []
    new_final = []

    for r in out.itertuples(index=False):
        base_final = str(getattr(r, "pred_final"))
        pred_binary = str(getattr(r, "pred_binary"))
        pg = float(getattr(r, "prob_good", np.nan))
        pb = float(getattr(r, "prob_bad", np.nan))

        # Base/legacy rules
        rule_neg_lt = bool(getattr(r, "rule_neg_lt_1k", 0))
        rule_neg_hf = bool(getattr(r, "rule_neg_1k_10k", 0))
        rule_jrun = bool(getattr(r, "rule_jagged_run", 0))
        rule_bigj = int(getattr(r, "rule_big_jump_count", 0) or 0)
        rule_iso_allowed = bool(getattr(r, "rule_isolated_spikes_allowed", 0))
        rule_high = bool(getattr(r, "rule_high_phase_gt_1k", 0))
        rule_combo = bool(getattr(r, "rule_combo_bad", 0))
        rule_near = bool(getattr(r, "rule_near_thr", 0))
        rule_lowc = bool(getattr(r, "rule_low_conf", 0))

        # New shape rules
        osc_jagged = bool(getattr(r, "rule_oscillatory_jagged", 0))
        tail_instab = bool(getattr(r, "rule_tail_instability", 0))
        smooth_candidate = bool(getattr(r, "rule_smooth_candidate", 0))
        deep_neg_lt1k = bool(getattr(r, "deep_neg_lt1k", False))

        # KK flags
        kk_status = str(getattr(r, "kk_status", "KK_NA"))
        kk_ok = (kk_status == "KK_OK")
        kk_missing = (kk_status == "KK_NA")
        kk_bad_like = kk_status in ("KK_REVIEW", "KK_BAD")
        kk_any = kk_bad_like or bool(getattr(r, "kk_flag_review", 0)) or bool(getattr(r, "kk_flag_bad", 0))

        # Amplitude metrics
        maxabs_1k = float(getattr(r, "max_abs_phase_gt_1k_mrad", np.nan))
        abs_extreme = bool(np.isfinite(maxabs_1k) and (maxabs_1k >= ABS_BAD_MRAD))
        abs_review = bool(np.isfinite(maxabs_1k) and (maxabs_1k >= PHASE_ABS_REVIEW_MRAD))

        # Tail metrics
        tail_spike_ratio = float(getattr(r, "tail_spike_ratio", np.nan))
        tail_flip_count = int(getattr(r, "tail_flip_count", 0) or 0)
        tail_rms_frac = float(getattr(r, "tail_resid_rms_frac", np.nan))

        tail_spike_ok = (not np.isfinite(tail_spike_ratio)) or (tail_spike_ratio < 7.0)
        tail_rms_ok = (not np.isfinite(tail_rms_frac)) or (tail_rms_frac < 0.05)

        clean_tail_metrics = (tail_flip_count <= 2) and tail_spike_ok and tail_rms_ok
        clean_shape = bool(
            (not abs_review) and (not osc_jagged) and (not rule_jrun) and
            (rule_bigj <= 2) and clean_tail_metrics
        )
        soft_clean_shape = bool(
            (not abs_review) and (not osc_jagged) and (not rule_jrun) and
            (rule_bigj <= 3) and
            (tail_flip_count <= 3) and
            ((not np.isfinite(tail_spike_ratio)) or (tail_spike_ratio < 9.0)) and
            ((not np.isfinite(tail_rms_frac)) or (tail_rms_frac < 0.07))
        )

        severe_tail = bool(
            tail_instab and (
                (osc_jagged and (tail_flip_count >= 3 or rule_bigj >= 3)) or
                (tail_flip_count >= TAIL_BAD_FLIPS_MIN) or
                (np.isfinite(tail_spike_ratio) and tail_spike_ratio >= TAIL_BAD_SPIKE_RATIO_MIN) or
                (np.isfinite(tail_rms_frac) and tail_rms_frac >= TAIL_BAD_RMS_FRAC_MIN) or
                (rule_bigj >= BIGJUMPS_BAD_MIN)
            )
        )
        mild_tail = bool(
            tail_instab and (not severe_tail) and (not osc_jagged) and (not rule_jrun) and
            (not rule_neg_lt) and (not rule_neg_hf) and
            (tail_flip_count <= 2) and
            ((not np.isfinite(tail_spike_ratio)) or (tail_spike_ratio < 7.0)) and
            ((not np.isfinite(tail_rms_frac)) or (tail_rms_frac < 0.05)) and
            (rule_bigj <= 2)
        )

        oscillatory_struct = bool(
            osc_jagged and (
                tail_instab or rule_jrun or rule_bigj >= 4 or tail_flip_count >= 3
            )
        )

        sustained_hf_negative = bool(
            rule_neg_hf and (
                osc_jagged or tail_instab or rule_bigj >= 3 or
                (np.isfinite(maxabs_1k) and maxabs_1k > HF_NEG_REVIEW_MAXABS_1K)
            )
        )

        severe_neg_lt = bool(rule_neg_lt and (deep_neg_lt1k or rule_neg_hf or oscillatory_struct or severe_tail or kk_bad_like))
        neg_lt_only = bool(
            rule_neg_lt and not any([
                rule_neg_hf, rule_jrun, tail_instab, osc_jagged, rule_iso_allowed,
                rule_high, rule_combo, rule_near, rule_lowc, kk_bad_like, abs_review
            ])
        )
        hf_neg_only = bool(
            rule_neg_hf and not any([
                rule_neg_lt, rule_jrun, rule_iso_allowed, rule_high, rule_combo,
                rule_near, rule_lowc, osc_jagged, tail_instab, kk_bad_like, abs_review
            ])
        )
        no_rule_now = not any([
            rule_neg_lt, rule_neg_hf, rule_jrun, rule_iso_allowed, rule_high,
            rule_combo, rule_near, rule_lowc, kk_any, tail_instab, osc_jagged
        ])
        tail_only = bool(
            tail_instab and not any([
                rule_neg_lt, rule_neg_hf, rule_jrun, rule_iso_allowed, rule_high,
                rule_combo, rule_near, rule_lowc, osc_jagged, kk_bad_like, abs_review
            ])
        )

        # BAD should mean true structural instability / nonphysical behavior
        structural_bad = bool(
            severe_tail or
            oscillatory_struct or
            (rule_combo and (tail_instab or osc_jagged or kk_bad_like)) or
            (rule_neg_lt and rule_neg_hf and (tail_instab or osc_jagged or kk_bad_like)) or
            (rule_neg_hf and kk_missing and (tail_instab or osc_jagged or rule_bigj >= 3) and (np.isfinite(maxabs_1k) and maxabs_1k >= 120.0)) or
            (kk_bad_like and (tail_instab or osc_jagged or rule_neg_hf)) or
            (abs_extreme and (tail_instab or osc_jagged or kk_bad_like or rule_bigj >= 4))
        )

        # GOOD should stay clean; smooth weak or provenance-like should drop to REVIEW
        risky_for_good = bool(
            structural_bad or
            oscillatory_struct or
            (osc_jagged and tail_instab) or
            (tail_instab and not mild_tail) or
            (rule_neg_hf and not (kk_ok and clean_shape and np.isfinite(maxabs_1k) and maxabs_1k <= HF_NEG_GOOD_MAXABS_1K and tail_flip_count <= 1)) or
            (rule_neg_lt and not neg_lt_only) or
            kk_bad_like or rule_jrun or abs_review or rule_high or rule_iso_allowed or rule_near or rule_lowc
        )

        final = base_final
        reason = "base"

        # 0) Hard BAD only for real structural evidence
        if structural_bad:
            final = "bad"
            if abs_extreme and (tail_instab or osc_jagged or kk_bad_like):
                reason = "abs+instability(BAD)"
            elif rule_neg_hf and kk_missing and (tail_instab or osc_jagged):
                reason = "hf-neg+KKNA(BAD)"
            else:
                reason = "structural-unstable(BAD)"

        else:
            # 1) Clean up GOOD bucket first
            if final == "good":
                if risky_for_good:
                    final = "review"
                    reason = "good-veto(REV)"
                elif hf_neg_only:
                    if kk_ok and clean_shape and np.isfinite(maxabs_1k) and (maxabs_1k <= HF_NEG_GOOD_MAXABS_1K) and (tail_flip_count <= 1) and ((not np.isnan(pg)) and (pg >= 0.9995)):
                        final = "good"
                        reason = "hf-neg-only_keep_good"
                    else:
                        final = "review"
                        reason = "hf-neg-only_review"
                elif neg_lt_only and kk_ok and soft_clean_shape:
                    if np.isfinite(maxabs_1k) and (maxabs_1k <= NEG_LT1K_GOOD_MAXABS_1K) and ((not np.isnan(pg)) and (pg >= GOOD_NEG_ONLY_PG)):
                        final = "good"
                        reason = "negative-baseline_rescue(GOOD)"
                    else:
                        final = "review"
                        reason = "negative-baseline_rescue(REV)"
                elif tail_only:
                    if kk_ok and mild_tail and np.isfinite(maxabs_1k) and (maxabs_1k <= TAIL_GOOD_MAXABS_1K) and ((not np.isnan(pg)) and (pg >= GOOD_TAIL_ONLY_PG)):
                        final = "good"
                        reason = "tail-only_keep_good"
                    else:
                        final = "review"
                        reason = "tail-only_review"

            # 2) If model predicts GOOD and nothing structural is wrong
            if pred_binary == "good" and final == base_final and final != "good":
                if risky_for_good:
                    final = "review"
                    reason = "good-soft-review"
                else:
                    final = "good"
                    reason = "pred-good"

            # 3) If model predicts BAD but there is no structural evidence, use REVIEW by default
            if pred_binary == "bad" and final not in ("good", "bad"):
                if no_rule_now and kk_ok and soft_clean_shape:
                    if np.isfinite(maxabs_1k) and (maxabs_1k <= MODEL_ONLY_BAD_GOOD_MAXABS_1K) and ((not np.isnan(pg)) and (pg >= GOOD_MODEL_ONLY_PG)):
                        final = "good"
                        reason = "model-only_bad_rescue(GOOD)"
                    else:
                        final = "review"
                        reason = "model-only_bad_rescue(REV)"
                elif tail_only and kk_ok and soft_clean_shape:
                    if mild_tail and np.isfinite(maxabs_1k) and (maxabs_1k <= TAIL_GOOD_MAXABS_1K) and ((not np.isnan(pg)) and (pg >= GOOD_TAIL_ONLY_PG)):
                        final = "good"
                        reason = "tail-only_rescue(GOOD)"
                    else:
                        final = "review"
                        reason = "tail-only_rescue(REV)"
                elif neg_lt_only and kk_ok and soft_clean_shape:
                    if np.isfinite(maxabs_1k) and (maxabs_1k <= NEG_LT1K_GOOD_MAXABS_1K) and ((not np.isnan(pg)) and (pg >= GOOD_NEG_ONLY_PG)):
                        final = "good"
                        reason = "negative-baseline_rescue(GOOD)"
                    else:
                        final = "review"
                        reason = "negative-baseline_rescue(REV)"
                elif hf_neg_only:
                    if kk_ok and clean_shape and np.isfinite(maxabs_1k) and (maxabs_1k <= HF_NEG_GOOD_MAXABS_1K) and (tail_flip_count <= 1) and ((not np.isnan(pg)) and (pg >= 0.9995)):
                        final = "good"
                        reason = "hf-neg-only_keep_good"
                    else:
                        final = "review"
                        reason = "hf-neg-only_review"
                else:
                    final = "review"
                    reason = "soft-bad_to_review"

            # 4) REVIEW cases: promote to GOOD only if the only issue is clearly benign
            if final == "review" and kk_ok and soft_clean_shape and (not structural_bad):
                if no_rule_now and ((not np.isnan(pg)) and (pg >= GOOD_STRONG_PG)) and np.isfinite(maxabs_1k) and (maxabs_1k <= MODEL_ONLY_BAD_GOOD_MAXABS_1K):
                    final = "good"
                    reason = "review-clean_rescue(GOOD)"
                elif tail_only and mild_tail and ((not np.isnan(pg)) and (pg >= GOOD_TAIL_ONLY_PG)) and np.isfinite(maxabs_1k) and (maxabs_1k <= TAIL_GOOD_MAXABS_1K):
                    final = "good"
                    reason = "tail-review_rescue(GOOD)"
                elif neg_lt_only and ((not np.isnan(pg)) and (pg >= GOOD_NEG_ONLY_PG)) and np.isfinite(maxabs_1k) and (maxabs_1k <= NEG_LT1K_GOOD_MAXABS_1K):
                    final = "good"
                    reason = "negative-baseline_rescue(GOOD)"

            # 5) Last guardrail: if still BAD without structural evidence, soften to REVIEW
            if final == "bad" and (not structural_bad):
                final = "review"
                reason = "bad-softened(REV)"

        new_final.append(final)
        reasons.append(reason)

    out["pred_final"] = new_final
    out["rule_patch_v3_reason"] = reasons
    out["rule_patch_v3_applied"] = (out["rule_patch_v3_reason"] != "base").astype(int)
    out["rule_patch_v2_reason"] = out["rule_patch_v3_reason"]
    out["rule_patch_v2_applied"] = out["rule_patch_v3_applied"]

    return out




def apply_override(features: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper around original apply_override():
    - preserves original model + rule features
    - keeps GOOD reserved for clean + trusted spectra
    - routes clean-but-weak / provenance-like spectra to REVIEW
    - reserves BAD for structurally unstable / nonphysical spectra
    - adds audit reason columns
    """
    out = _orig_apply_override(features.copy())

    # Ensure expected columns exist
    for c, default in [
        ("pred_final", None),
        ("pred_binary", None),
        ("prob_good", np.nan),
        ("prob_bad", np.nan),
        ("prob_max", np.nan),
        ("rule_neg_lt_1k", 0),
        ("rule_neg_1k_10k", 0),
        ("rule_jagged_run", 0),
        ("rule_big_jump_count", 0),
        ("rule_isolated_spikes", 0),
        ("rule_isolated_spikes_allowed", 0),
        ("rule_high_phase_gt_1k", 0),
        ("rule_combo_bad", 0),
        ("rule_near_thr", 0),
        ("rule_low_conf", 0),
        ("rule_oscillatory_jagged", 0),
        ("rule_tail_instability", 0),
        ("rule_smooth_candidate", 0),
        ("deep_neg_lt1k", False),
        ("max_phase_gt_1k_mrad", np.nan),
        ("max_abs_phase_mrad", np.nan),
        ("max_abs_phase_gt_1k_mrad", np.nan),
        ("tail_resid_rms_frac", np.nan),
        ("tail_flip_count", 0),
        ("tail_spike_ratio", np.nan),
        ("kk_status", "KK_NA"),
        ("kk_flag_review", 0),
        ("kk_flag_bad", 0),
    ]:
        if c not in out.columns:
            out[c] = default

    if "pred_final_base" not in out.columns:
        out["pred_final_base"] = out["pred_final"].astype(str)

    # Tunables: GOOD = trusted, REVIEW = inspect, BAD = structural / nonphysical
    GOOD_STRONG_PG = 0.998
    GOOD_TAIL_ONLY_PG = 0.999
    GOOD_NEG_ONLY_PG = 0.9985

    NEG_LT1K_GOOD_MAXABS_1K = 90.0
    HF_NEG_GOOD_MAXABS_1K = 25.0
    TAIL_GOOD_MAXABS_1K = 110.0

    TAIL_STRUCT_FLIPS_MIN = 4
    TAIL_STRUCT_SPIKE_RATIO_MIN = 10.0
    TAIL_STRUCT_RMS_FRAC_MIN = 0.08
    BIGJUMPS_STRUCT_MIN = 6
    ABS_BAD_MRAD = max(PHASE_ABS_BAD_MRAD, 1000.0)

    reasons = []
    new_final = []

    for r in out.itertuples(index=False):
        base_final = str(getattr(r, "pred_final"))
        pred_binary = str(getattr(r, "pred_binary"))
        pg = float(getattr(r, "prob_good", np.nan))
        pb = float(getattr(r, "prob_bad", np.nan))

        # Base/legacy rules
        rule_neg_lt = bool(getattr(r, "rule_neg_lt_1k", 0))
        rule_neg_hf = bool(getattr(r, "rule_neg_1k_10k", 0))
        rule_jrun = bool(getattr(r, "rule_jagged_run", 0))
        rule_bigj = int(getattr(r, "rule_big_jump_count", 0) or 0)
        rule_iso_allowed = bool(getattr(r, "rule_isolated_spikes_allowed", 0))
        rule_high = bool(getattr(r, "rule_high_phase_gt_1k", 0))
        rule_combo = bool(getattr(r, "rule_combo_bad", 0))
        rule_near = bool(getattr(r, "rule_near_thr", 0))
        rule_lowc = bool(getattr(r, "rule_low_conf", 0))

        # New shape rules
        osc_jagged = bool(getattr(r, "rule_oscillatory_jagged", 0))
        tail_instab = bool(getattr(r, "rule_tail_instability", 0))
        smooth_candidate = bool(getattr(r, "rule_smooth_candidate", 0))
        deep_neg_lt1k = bool(getattr(r, "deep_neg_lt1k", False))

        # KK flags
        kk_status = str(getattr(r, "kk_status", "KK_NA"))
        kk_ok = (kk_status == "KK_OK")
        kk_missing = (kk_status == "KK_NA")
        kk_review_like = (kk_status in ("KK_REVIEW", "KK_BAD"))
        kk_any = kk_review_like or bool(getattr(r, "kk_flag_review", 0)) or bool(getattr(r, "kk_flag_bad", 0))

        # Amplitude metrics
        maxabs_1k = float(getattr(r, "max_abs_phase_gt_1k_mrad", np.nan))
        abs_extreme = bool(np.isfinite(maxabs_1k) and (maxabs_1k >= ABS_BAD_MRAD))
        abs_review = bool(np.isfinite(maxabs_1k) and (maxabs_1k >= PHASE_ABS_REVIEW_MRAD))

        # Tail metrics
        tail_spike_ratio = float(getattr(r, "tail_spike_ratio", np.nan))
        tail_flip_count = int(getattr(r, "tail_flip_count", 0) or 0)
        tail_rms_frac = float(getattr(r, "tail_resid_rms_frac", np.nan))

        tail_spike_mild = (not np.isfinite(tail_spike_ratio)) or (tail_spike_ratio < 6.0)
        tail_rms_mild = (not np.isfinite(tail_rms_frac)) or (tail_rms_frac < 0.05)

        clean_shape = bool(
            (not abs_review) and (not osc_jagged) and (not rule_jrun) and
            (rule_bigj <= 2) and (tail_flip_count <= 2) and tail_spike_mild and tail_rms_mild
        )
        soft_clean_shape = bool(
            (not abs_review) and (not osc_jagged) and (not rule_jrun) and
            (rule_bigj <= 3) and (tail_flip_count <= 3) and
            ((not np.isfinite(tail_spike_ratio)) or (tail_spike_ratio < 8.0)) and
            ((not np.isfinite(tail_rms_frac)) or (tail_rms_frac < 0.07))
        )

        # Tail regimes: distinguish smooth tail curvature from real instability
        severe_tail = bool(
            tail_instab and (
                (rule_bigj >= BIGJUMPS_STRUCT_MIN) or
                (tail_flip_count >= TAIL_STRUCT_FLIPS_MIN and (osc_jagged or rule_jrun or (np.isfinite(tail_spike_ratio) and tail_spike_ratio >= 6.0))) or
                (np.isfinite(tail_spike_ratio) and tail_spike_ratio >= TAIL_STRUCT_SPIKE_RATIO_MIN) or
                (np.isfinite(tail_rms_frac) and tail_rms_frac >= TAIL_STRUCT_RMS_FRAC_MIN and (osc_jagged or tail_flip_count >= 3 or rule_jrun))
            )
        )
        mild_tail = bool(
            tail_instab and (not severe_tail) and (not osc_jagged) and (not rule_jrun) and
            (tail_flip_count <= 2) and
            ((not np.isfinite(tail_spike_ratio)) or (tail_spike_ratio < 6.0)) and
            ((not np.isfinite(tail_rms_frac)) or (tail_rms_frac < 0.05)) and
            (rule_bigj <= 2)
        )

        oscillatory_struct = bool(
            osc_jagged and (
                rule_jrun or rule_bigj >= 4 or tail_flip_count >= 3 or
                (tail_instab and ((np.isfinite(tail_spike_ratio) and tail_spike_ratio >= 5.0) or (np.isfinite(tail_rms_frac) and tail_rms_frac >= 0.05)))
            )
        )

        hf_negative_struct = bool(
            rule_neg_hf and (
                oscillatory_struct or severe_tail or kk_review_like or
                (kk_missing and (tail_instab or rule_bigj >= 3))
            ) and np.isfinite(maxabs_1k) and (maxabs_1k >= 80.0)
        )
        severe_neg_lt = bool(rule_neg_lt and (deep_neg_lt1k or (rule_neg_hf and (oscillatory_struct or severe_tail or kk_review_like))))

        neg_lt_only = bool(
            rule_neg_lt and not any([
                rule_neg_hf, rule_jrun, tail_instab, osc_jagged, rule_iso_allowed,
                rule_high, rule_combo, rule_near, rule_lowc, kk_review_like, abs_review
            ])
        )
        hf_neg_only = bool(
            rule_neg_hf and not any([
                rule_neg_lt, rule_jrun, rule_iso_allowed, rule_high, rule_combo,
                rule_near, rule_lowc, osc_jagged, tail_instab, kk_review_like, abs_review
            ])
        )
        no_rule_now = not any([
            rule_neg_lt, rule_neg_hf, rule_jrun, rule_iso_allowed, rule_high,
            rule_combo, rule_near, rule_lowc, kk_any, tail_instab, osc_jagged
        ])
        tail_only = bool(
            tail_instab and not any([
                rule_neg_lt, rule_neg_hf, rule_jrun, rule_iso_allowed, rule_high,
                rule_combo, rule_near, rule_lowc, osc_jagged, kk_review_like, abs_review
            ])
        )

        # BAD must mean structural instability / nonphysical behavior, not merely smooth provenance-like shape.
        structural_bad = bool(
            abs_extreme or
            severe_tail or
            oscillatory_struct or
            severe_neg_lt or
            hf_negative_struct or
            (rule_combo and (osc_jagged or severe_tail or kk_review_like or rule_jrun)) or
            (kk_review_like and (osc_jagged or severe_tail or (rule_neg_hf and tail_instab)))
        )

        # GOOD should stay reserved for clean + trusted spectra.
        risky_for_good = bool(
            structural_bad or
            oscillatory_struct or
            (osc_jagged and tail_instab) or
            (tail_instab and not mild_tail) or
            (rule_neg_hf and not (kk_ok and clean_shape and np.isfinite(maxabs_1k) and maxabs_1k <= HF_NEG_GOOD_MAXABS_1K and tail_flip_count <= 1)) or
            (rule_neg_lt and not neg_lt_only) or
            kk_review_like or rule_jrun or abs_review or rule_high or rule_iso_allowed or rule_near or rule_lowc
        )

        final = base_final
        reason = "base"

        # 0) Hard BAD only for real structural evidence
        if structural_bad:
            final = "bad"
            if abs_extreme:
                reason = "abs+instability(BAD)"
            elif hf_negative_struct and kk_missing:
                reason = "hf-neg+KKNA(BAD)"
            else:
                reason = "structural-unstable(BAD)"
        else:
            # 1) Clean up GOOD bucket first
            if final == "good":
                if risky_for_good:
                    final = "review"
                    reason = "good-veto(REV)"
                elif hf_neg_only:
                    if kk_ok and clean_shape and np.isfinite(maxabs_1k) and (maxabs_1k <= HF_NEG_GOOD_MAXABS_1K) and (tail_flip_count <= 1) and ((not np.isnan(pg)) and (pg >= 0.9995)):
                        final = "good"
                        reason = "hf-neg-only_keep_good"
                    else:
                        final = "review"
                        reason = "hf-neg-only_review"
                elif neg_lt_only and kk_ok and soft_clean_shape:
                    if np.isfinite(maxabs_1k) and (maxabs_1k <= NEG_LT1K_GOOD_MAXABS_1K) and ((not np.isnan(pg)) and (pg >= GOOD_NEG_ONLY_PG)):
                        final = "good"
                        reason = "negative-baseline_rescue(GOOD)"
                    else:
                        final = "review"
                        reason = "negative-baseline_rescue(REV)"
                elif tail_only:
                    if kk_ok and mild_tail and np.isfinite(maxabs_1k) and (maxabs_1k <= TAIL_GOOD_MAXABS_1K) and ((not np.isnan(pg)) and (pg >= GOOD_TAIL_ONLY_PG)):
                        final = "good"
                        reason = "tail-only_keep_good"
                    else:
                        final = "review"
                        reason = "tail-only_review"

            # 2) If model predicts GOOD and nothing structural is wrong
            if pred_binary == "good" and final == base_final and final != "good":
                if risky_for_good:
                    final = "review"
                    reason = "good-soft-review"
                else:
                    final = "good"
                    reason = "pred-good"

            # 3) If model predicts BAD but there is no structural evidence, REVIEW by default
            if pred_binary == "bad" and final not in ("good", "bad"):
                if no_rule_now and kk_ok and soft_clean_shape:
                    final = "review"
                    reason = "model-only_bad_rescue(REV)"
                elif tail_only and kk_ok and soft_clean_shape:
                    final = "review"
                    reason = "tail-only_rescue(REV)"
                elif neg_lt_only and kk_ok and soft_clean_shape:
                    final = "review"
                    reason = "negative-baseline_rescue(REV)"
                elif hf_neg_only and kk_ok and clean_shape:
                    final = "review"
                    reason = "hf-neg-only_review"
                else:
                    final = "review"
                    reason = "soft-bad_to_review"

            # 4) REVIEW cases: only promote to GOOD if the only issue is clearly benign
            if final == "review" and kk_ok and soft_clean_shape and (not structural_bad):
                if no_rule_now and ((not np.isnan(pg)) and (pg >= GOOD_STRONG_PG)) and np.isfinite(maxabs_1k) and (maxabs_1k <= 120.0):
                    final = "good"
                    reason = "review-clean_rescue(GOOD)"
                elif tail_only and mild_tail and ((not np.isnan(pg)) and (pg >= GOOD_TAIL_ONLY_PG)) and np.isfinite(maxabs_1k) and (maxabs_1k <= TAIL_GOOD_MAXABS_1K):
                    final = "good"
                    reason = "tail-review_rescue(GOOD)"
                elif neg_lt_only and ((not np.isnan(pg)) and (pg >= GOOD_NEG_ONLY_PG)) and np.isfinite(maxabs_1k) and (maxabs_1k <= NEG_LT1K_GOOD_MAXABS_1K):
                    final = "good"
                    reason = "negative-baseline_rescue(GOOD)"

            # 5) Last guardrail: if still BAD without structural evidence, soften to REVIEW
            if final == "bad" and (not structural_bad):
                final = "review"
                reason = "bad-softened(REV)"

        new_final.append(final)
        reasons.append(reason)

    out["pred_final"] = new_final
    out["rule_patch_v3_reason"] = reasons
    out["rule_patch_v3_applied"] = (out["rule_patch_v3_reason"] != "base").astype(int)
    out["rule_patch_v2_reason"] = out["rule_patch_v3_reason"]
    out["rule_patch_v2_applied"] = out["rule_patch_v3_applied"]

    return out
