# Extracted from SIP_ML_Tool.ipynb main workflow cell.
# V2.6.1-V2.6.6 physics-rule update stack.

if "_v261_base_apply_override" not in globals():
    _v261_base_apply_override = apply_override

V261_NEG_LT1K_RESCUE_PG = 0.995
V261_NEG_LT1K_RESCUE_MAXABS_1K = 140.0
V261_NEG_LT1K_RESCUE_MAX_RUN = 2
V261_NEG_LT1K_RESCUE_MAX_COUNT = 2
V261_NEG_LT1K_RESCUE_MAX_BIGJUMPS = 2
V261_NEG_LT1K_RESCUE_MAX_TAIL_FLIPS = 2
V261_NEG_LT1K_RESCUE_MAX_TAIL_SPIKE = 7.0
V261_NEG_LT1K_RESCUE_MAX_TAIL_RMS_FRAC = 0.05

def apply_override(features: pd.DataFrame) -> pd.DataFrame:
    out = _v261_base_apply_override(features.copy())

    # Ensure fields exist for the targeted recovery rule.
    defaults = {
        "pred_final": None,
        "pred_binary": None,
        "prob_good": np.nan,
        "prob_bad": np.nan,
        "kk_status": "KK_NA",
        "rule_patch_v26_reason": "",
        "rule_patch_v2_reason": "",
        "rule_patch_v3_reason": "",
        "rule_neg_lt_1k": 0,
        "rule_neg_1k_10k": 0,
        "rule_combo_bad": 0,
        "rule_jagged_run": 0,
        "rule_big_jump_count": 0,
        "rule_isolated_spikes_allowed": 0,
        "rule_high_phase_gt_1k": 0,
        "rule_oscillatory_jagged": 0,
        "rule_tail_instability": 0,
        "rule_smooth_candidate": 0,
        "deep_neg_lt1k": False,
        "neg_lt1k_run": 0,
        "neg_lt1k_count": 0,
        "tail_flip_count": 0,
        "tail_spike_ratio": np.nan,
        "tail_resid_rms_frac": np.nan,
        "max_abs_phase_gt_1k_mrad": np.nan,
    }
    for c, default in defaults.items():
        if c not in out.columns:
            out[c] = default

    tail_spike_ok = out["tail_spike_ratio"].isna() | (out["tail_spike_ratio"].astype(float) <= V261_NEG_LT1K_RESCUE_MAX_TAIL_SPIKE)
    tail_rms_ok = out["tail_resid_rms_frac"].isna() | (out["tail_resid_rms_frac"].astype(float) <= V261_NEG_LT1K_RESCUE_MAX_TAIL_RMS_FRAC)
    maxabs_ok = out["max_abs_phase_gt_1k_mrad"].isna() | (out["max_abs_phase_gt_1k_mrad"].astype(float) <= V261_NEG_LT1K_RESCUE_MAXABS_1K)

    rescue_neg_lt1k = (
        out["pred_final"].astype(str).eq("bad") &
        out["rule_patch_v26_reason"].astype(str).eq("nonphysical_negative_phase_lt_1k") &
        (out["prob_good"].astype(float) >= V261_NEG_LT1K_RESCUE_PG) &
        out["kk_status"].astype(str).eq("KK_OK") &
        out["rule_neg_lt_1k"].fillna(0).astype(int).eq(1) &
        out["rule_neg_1k_10k"].fillna(0).astype(int).eq(0) &
        out["rule_combo_bad"].fillna(0).astype(int).eq(0) &
        out["rule_jagged_run"].fillna(0).astype(int).eq(0) &
        out["rule_big_jump_count"].fillna(0).astype(int).le(V261_NEG_LT1K_RESCUE_MAX_BIGJUMPS) &
        out["rule_isolated_spikes_allowed"].fillna(0).astype(int).eq(0) &
        out["rule_high_phase_gt_1k"].fillna(0).astype(int).eq(0) &
        out["rule_oscillatory_jagged"].fillna(0).astype(int).eq(0) &
        out["rule_tail_instability"].fillna(0).astype(int).eq(0) &
        out["rule_smooth_candidate"].fillna(0).astype(int).eq(1) &
        (~out["deep_neg_lt1k"].fillna(False).astype(bool)) &
        out["neg_lt1k_run"].fillna(0).astype(int).le(V261_NEG_LT1K_RESCUE_MAX_RUN) &
        out["neg_lt1k_count"].fillna(0).astype(int).le(V261_NEG_LT1K_RESCUE_MAX_COUNT) &
        out["tail_flip_count"].fillna(0).astype(int).le(V261_NEG_LT1K_RESCUE_MAX_TAIL_FLIPS) &
        tail_spike_ok &
        tail_rms_ok &
        maxabs_ok
    )

    if rescue_neg_lt1k.any():
        out.loc[rescue_neg_lt1k, "pred_final"] = "review"
        out.loc[rescue_neg_lt1k, "rule_patch_v26_reason"] = "mild_negative_lt_1k_softened_to_review_v261"
        out.loc[rescue_neg_lt1k, "rule_patch_v26_applied"] = 1
        out.loc[rescue_neg_lt1k, "rule_patch_v3_reason"] = out.loc[rescue_neg_lt1k, "rule_patch_v26_reason"]
        out.loc[rescue_neg_lt1k, "rule_patch_v3_applied"] = 1
        out.loc[rescue_neg_lt1k, "rule_patch_v2_reason"] = out.loc[rescue_neg_lt1k, "rule_patch_v26_reason"]
        out.loc[rescue_neg_lt1k, "rule_patch_v2_applied"] = 1

    out["qc_version"] = "V2.6.1_targeted_recovery"
    return out




# ============================== V2.6.2 NEGATIVE-SEVERITY PATCH ==============================
# Purpose:
#   Make the sub-1 kHz negative-phase rule physically graded instead of nearly binary.
#   We keep the working V2.6.1 GOOD behavior and only soften a subset of BAD calls
#   when the negative-phase evidence is mild/moderate rather than clearly nonphysical.
#
# Scientific intent:
#   - GOOD bucket should stay strict
#   - REVIEW should absorb salvageable "mild negative below 1 kHz" spectra
#   - BAD should remain reserved for persistent/deep/multi-feature failures
#
# This patch does NOT redesign CatBoost or calibration. It only improves the
# final SIP-specific decision layer using metrics the notebook already computes.

if "_v262_prev_apply_override" not in globals():
    _v262_prev_apply_override = apply_override

V262_NEG_REVIEW_PG_HIGH = 0.85
V262_NEG_REVIEW_PG_MID  = 0.65
V262_NEG_COUNT_SOFT_MAX = 4
V262_NEG_RUN_SOFT_MAX   = 3
V262_TAIL_SPIKE_SOFT_MAX = 6.0
V262_TAIL_RMS_SOFT_MAX   = 0.08

def _classify_neg_lt1k_severity_row(r) -> tuple[int, str]:
    """Return (severity_score, severity_label) for sub-1 kHz negative-phase behavior."""
    neg_count = int(getattr(r, "neg_lt1k_count", 0) or 0)
    neg_run = int(getattr(r, "neg_lt1k_run", 0) or 0)
    deep_neg = bool(getattr(r, "deep_neg_lt1k", False))
    neg_hf = bool(getattr(r, "rule_neg_1k_10k", 0))
    combo_bad = bool(getattr(r, "rule_combo_bad", 0))
    jrun = bool(getattr(r, "rule_jagged_run", 0))
    osc = bool(getattr(r, "rule_oscillatory_jagged", 0))
    tail = bool(getattr(r, "rule_tail_instability", 0))
    high_phase = bool(getattr(r, "rule_high_phase_gt_1k", 0))
    kk_status = str(getattr(r, "kk_status", "KK_NA"))

    tail_flip_count = int(getattr(r, "tail_flip_count", 0) or 0)
    tail_spike_ratio = float(getattr(r, "tail_spike_ratio", np.nan))
    tail_rms_frac = float(getattr(r, "tail_resid_rms_frac", np.nan))

    score = 0

    # Core sub-1 kHz negative persistence/depth
    if neg_count >= 2:
        score += 1
    if neg_count >= 4:
        score += 1
    if neg_run >= 2:
        score += 1
    if neg_run >= 4:
        score += 1
    if deep_neg:
        score += 2

    # Escalators from other suspicious SIP structure
    if neg_hf:
        score += 2
    if combo_bad:
        score += 2
    if jrun:
        score += 1
    if osc:
        score += 1
    if tail:
        score += 2
    if tail_flip_count >= 4:
        score += 1
    if np.isfinite(tail_spike_ratio) and tail_spike_ratio >= V262_TAIL_SPIKE_SOFT_MAX:
        score += 1
    if np.isfinite(tail_rms_frac) and tail_rms_frac >= V262_TAIL_RMS_SOFT_MAX:
        score += 1
    if high_phase:
        score += 1

    # KK consistency influences severity but does not dominate alone
    if kk_status == "KK_REVIEW":
        score += 1
    elif kk_status == "KK_BAD":
        score += 3

    if score <= 2:
        label = "mild"
    elif score <= 5:
        label = "moderate"
    else:
        label = "severe"
    return score, label

def apply_override(features: pd.DataFrame) -> pd.DataFrame:
    out = _v262_prev_apply_override(features.copy())

    defaults = {
        "pred_final": None,
        "pred_binary": None,
        "prob_good": np.nan,
        "prob_bad": np.nan,
        "kk_status": "KK_NA",
        "rule_patch_v26_reason": "",
        "rule_patch_v2_reason": "",
        "rule_patch_v3_reason": "",
        "rule_neg_lt_1k": 0,
        "rule_neg_1k_10k": 0,
        "rule_combo_bad": 0,
        "rule_jagged_run": 0,
        "rule_high_phase_gt_1k": 0,
        "rule_oscillatory_jagged": 0,
        "rule_tail_instability": 0,
        "rule_smooth_candidate": 0,
        "neg_lt1k_run": 0,
        "neg_lt1k_count": 0,
        "deep_neg_lt1k": False,
        "tail_flip_count": 0,
        "tail_spike_ratio": np.nan,
        "tail_resid_rms_frac": np.nan,
    }
    for c, default in defaults.items():
        if c not in out.columns:
            out[c] = default

    sev = out.apply(_classify_neg_lt1k_severity_row, axis=1, result_type="expand")
    out["neg_lt1k_severity_score"] = sev[0].astype(int)
    out["neg_lt1k_severity_label"] = sev[1].astype(str)

    kk_ok_like = out["kk_status"].astype(str).isin(["KK_OK", "KK_NA"])
    kk_review_like = out["kk_status"].astype(str).eq("KK_REVIEW")

    # Candidate BAD calls produced primarily by the sub-1 kHz negative rule
    candidate_neg_bad = (
        out["pred_final"].astype(str).eq("bad") &
        out["rule_patch_v26_reason"].astype(str).eq("nonphysical_negative_phase_lt_1k")
    )

    # Hard blockers: these remain BAD because they are persistent/deep or tied
    # to other structural issues that make the spectrum hard to defend.
    hard_block = (
        out["deep_neg_lt1k"].fillna(False).astype(bool) |
        out["rule_neg_1k_10k"].fillna(0).astype(int).eq(1) |
        out["rule_combo_bad"].fillna(0).astype(int).eq(1) |
        out["rule_jagged_run"].fillna(0).astype(int).eq(1) |
        out["rule_oscillatory_jagged"].fillna(0).astype(int).eq(1) |
        out["rule_tail_instability"].fillna(0).astype(int).eq(1) |
        out["rule_high_phase_gt_1k"].fillna(0).astype(int).eq(1) |
        out["kk_status"].astype(str).eq("KK_BAD") |
        out["neg_lt1k_count"].fillna(0).astype(int).gt(V262_NEG_COUNT_SOFT_MAX) |
        out["neg_lt1k_run"].fillna(0).astype(int).gt(V262_NEG_RUN_SOFT_MAX) |
        (out["neg_lt1k_severity_label"].astype(str).eq("severe"))
    )

    # Soft-rescue tier A:
    # Strong model support + KK_OK/NA + no hard blocker + mild/moderate severity.
    rescue_review_high = (
        candidate_neg_bad &
        (~hard_block) &
        kk_ok_like &
        (out["prob_good"].astype(float) >= V262_NEG_REVIEW_PG_HIGH) &
        out["neg_lt1k_severity_score"].astype(int).le(5)
    )

    # Soft-rescue tier B:
    # Moderately good model support, but only for very smooth / low-severity cases.
    rescue_review_mid = (
        candidate_neg_bad &
        (~hard_block) &
        kk_ok_like &
        (out["prob_good"].astype(float) >= V262_NEG_REVIEW_PG_MID) &
        out["rule_smooth_candidate"].fillna(0).astype(int).eq(1) &
        out["neg_lt1k_severity_score"].astype(int).le(3) &
        (out["tail_flip_count"].fillna(0).astype(int) <= 3) &
        (out["tail_spike_ratio"].isna() | (out["tail_spike_ratio"].astype(float) < V262_TAIL_SPIKE_SOFT_MAX)) &
        (out["tail_resid_rms_frac"].isna() | (out["tail_resid_rms_frac"].astype(float) < V262_TAIL_RMS_SOFT_MAX))
    )

    rescue_review = rescue_review_high | rescue_review_mid

    if rescue_review.any():
        mild_mask = rescue_review & out["neg_lt1k_severity_label"].astype(str).eq("mild")
        moderate_mask = rescue_review & out["neg_lt1k_severity_label"].astype(str).eq("moderate")

        out.loc[rescue_review, "pred_final"] = "review"
        out.loc[mild_mask, "rule_patch_v26_reason"] = "mild_negative_lt_1k_softened_to_review_v262"
        out.loc[moderate_mask, "rule_patch_v26_reason"] = "moderate_negative_lt_1k_softened_to_review_v262"
        # fallback for any rescued row with unexpected label
        fallback_mask = rescue_review & ~mild_mask & ~moderate_mask
        out.loc[fallback_mask, "rule_patch_v26_reason"] = "negative_lt_1k_softened_to_review_v262"

        out.loc[rescue_review, "rule_patch_v26_applied"] = 1
        out.loc[rescue_review, "rule_patch_v3_reason"] = out.loc[rescue_review, "rule_patch_v26_reason"]
        out.loc[rescue_review, "rule_patch_v3_applied"] = 1
        out.loc[rescue_review, "rule_patch_v2_reason"] = out.loc[rescue_review, "rule_patch_v26_reason"]
        out.loc[rescue_review, "rule_patch_v2_applied"] = 1

    out["qc_version"] = "V2.6.2_neg_severity"
    return out



# ==============================================================================
# PATCH V2.6.3 — Physics admissibility BAD gate + trusted-GOOD Isolation Forest
# ==============================================================================
"""
Goal:
- Keep CatBoost as the core supervised engine.
- Preserve the current GOOD and REVIEW behavior as much as possible.
- Improve the BAD category by requiring stronger physics-based evidence for hard rejection.
- Add a one-class "trusted GOOD manifold" check using Isolation Forest, used only to soften
  a subset of current BAD calls into REVIEW when the spectrum still looks like the clean GOOD family.
"""

from sklearn.ensemble import IsolationForest

# --- V2.6.3 tunables ---
V263_IFOREST_MIN_TRAIN = 12
V263_IFOREST_CONTAM = 0.10
V263_IFOREST_Q_FAR = 0.08

V263_PHYS_MODERATE_MAX = 5
V263_PHYS_MILD_MAX = 3

V263_RESCUE_PG_LOW = 0.30
V263_RESCUE_PG_MID = 0.45
V263_RESCUE_PG_HIGH = 0.70

V263_MAX_ISSUES_FOR_RESCUE = 2
V263_MAX_ISSUES_FOR_RESCUE_MID = 3

def _v263_numeric_issue_count_row(r) -> int:
    count = 0
    bits = [
        bool(getattr(r, "rule_neg_lt_1k", 0)),
        bool(getattr(r, "rule_neg_1k_10k", 0)),
        bool(getattr(r, "rule_combo_bad", 0)),
        bool(getattr(r, "rule_jagged_run", 0)),
        bool(getattr(r, "rule_oscillatory_jagged", 0)),
        bool(getattr(r, "rule_tail_instability", 0)),
        bool(getattr(r, "rule_high_phase_gt_1k", 0)),
        str(getattr(r, "kk_status", "KK_NA")) == "KK_REVIEW",
        str(getattr(r, "kk_status", "KK_NA")) == "KK_BAD",
    ]
    count += int(sum(bits))
    maxabs_1k = float(getattr(r, "max_abs_phase_gt_1k_mrad", np.nan))
    if np.isfinite(maxabs_1k) and maxabs_1k >= PHASE_ABS_REVIEW_MRAD:
        count += 1
    return count

def _v263_physics_inadmissibility_row(r):
    neg_count = int(getattr(r, "neg_lt1k_count", 0) or 0)
    neg_run = int(getattr(r, "neg_lt1k_run", 0) or 0)
    deep_neg = bool(getattr(r, "deep_neg_lt1k", False))
    neg_hf = bool(getattr(r, "rule_neg_1k_10k", 0))
    combo_bad = bool(getattr(r, "rule_combo_bad", 0))
    jrun = bool(getattr(r, "rule_jagged_run", 0))
    osc = bool(getattr(r, "rule_oscillatory_jagged", 0))
    tail = bool(getattr(r, "rule_tail_instability", 0))
    high_phase = bool(getattr(r, "rule_high_phase_gt_1k", 0))
    kk_status = str(getattr(r, "kk_status", "KK_NA"))
    tail_flip_count = int(getattr(r, "tail_flip_count", 0) or 0)
    tail_spike_ratio = float(getattr(r, "tail_spike_ratio", np.nan))
    tail_rms_frac = float(getattr(r, "tail_resid_rms_frac", np.nan))
    maxabs_1k = float(getattr(r, "max_abs_phase_gt_1k_mrad", np.nan))

    score = 0.0
    if neg_count >= 1:
        score += 0.5
    if neg_count >= 3:
        score += 1.0
    if neg_count >= 5:
        score += 1.0
    if neg_run >= 2:
        score += 1.0
    if neg_run >= 4:
        score += 1.0
    if deep_neg:
        score += 2.0

    if neg_hf:
        score += 2.0
    if combo_bad:
        score += 2.0
    if jrun:
        score += 1.0
    if osc:
        score += 1.0
    if tail:
        score += 1.0
    if high_phase:
        score += 1.0

    if tail_flip_count >= 4:
        score += 1.0
    if np.isfinite(tail_spike_ratio) and tail_spike_ratio >= 8.0:
        score += 1.0
    if np.isfinite(tail_spike_ratio) and tail_spike_ratio >= 12.0:
        score += 1.0
    if np.isfinite(tail_rms_frac) and tail_rms_frac >= 0.08:
        score += 1.0
    if np.isfinite(tail_rms_frac) and tail_rms_frac >= 0.12:
        score += 1.0

    if np.isfinite(maxabs_1k) and maxabs_1k >= PHASE_ABS_REVIEW_MRAD:
        score += 1.0
    if np.isfinite(maxabs_1k) and maxabs_1k >= PHASE_ABS_BAD_MRAD:
        score += 2.0

    if kk_status == "KK_REVIEW":
        score += 1.0
    elif kk_status == "KK_BAD":
        score += 3.0

    if score <= V263_PHYS_MILD_MAX:
        label = "mild"
    elif score <= V263_PHYS_MODERATE_MAX:
        label = "moderate"
    else:
        label = "severe"
    return float(score), label

def _v263_good_manifold_features(df: pd.DataFrame):
    candidate_cols = [
        "freq_min","freq_max","n_points","log_freq_range",
        "phase_min","phase_max","phase_mean","phase_std","phase_frac_negative",
        "realcond_mean","realcond_std","imagcond_mean","imagcond_std",
        "imp_mean","imp_std","imp_range","peakFreq","peakPhase_mrad",
        "frac_pos_low","frac_pos_mid","frac_pos_high",
        "phase_jag_all","phase_jag_mid",
        "slope_mag_all","slope_mag_low","slope_mag_mid","slope_mag_high",
        "max_phase_gt_1k_mrad","max_abs_phase_mrad","max_abs_phase_gt_1k_mrad",
        "neg_lt1k_count","neg_1k_10k_count","neg_lt1k_run","neg_1k_10k_run",
        "resid_rms_frac","flip_frac","flip_frac_mid",
        "tail_resid_rms_frac","tail_flip_count","tail_spike_ratio","tail_max_jump_mrad",
        "kk_pseudo_chi2","kk_mu","kk_resid_rms","kk_num_rc",
    ]
    use_cols = [c for c in candidate_cols if c in df.columns]
    if not use_cols:
        return None, []
    X = df[use_cols].copy()
    for c in use_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    med = X.median(numeric_only=True)
    X = X.fillna(med).fillna(0.0)
    return X.astype(float), use_cols

def _v263_fit_good_iforest(df: pd.DataFrame):
    train_good = pd.DataFrame()
    if "true_label" in df.columns:
        train_good = df[df["true_label"].astype(str).eq("good")].copy()
    if len(train_good) < V263_IFOREST_MIN_TRAIN:
        train_good = df[
            df["pred_final"].astype(str).eq("good") &
            df["kk_status"].astype(str).isin(["KK_OK", "KK_NA"]) &
            (pd.to_numeric(df["prob_good"], errors="coerce") >= 0.99)
        ].copy()
    if len(train_good) < V263_IFOREST_MIN_TRAIN:
        return None, None, None, []

    X_train, use_cols = _v263_good_manifold_features(train_good)
    if X_train is None or len(use_cols) < 5 or len(X_train) < V263_IFOREST_MIN_TRAIN:
        return None, None, None, []

    iso = IsolationForest(
        n_estimators=300,
        contamination=V263_IFOREST_CONTAM,
        random_state=42,
        bootstrap=False,
    )
    iso.fit(X_train)
    train_scores = iso.score_samples(X_train)
    far_thr = float(np.quantile(train_scores, V263_IFOREST_Q_FAR))
    return iso, far_thr, len(X_train), use_cols

if "_v263_prev_apply_override" not in globals():
    _v263_prev_apply_override = apply_override

def apply_override(features: pd.DataFrame) -> pd.DataFrame:
    out = _v263_prev_apply_override(features.copy())

    defaults = {
        "pred_final": None,
        "pred_binary": None,
        "prob_good": np.nan,
        "prob_bad": np.nan,
        "prob_max": np.nan,
        "kk_status": "KK_NA",
        "rule_patch_v26_reason": "",
        "rule_patch_v2_reason": "",
        "rule_patch_v3_reason": "",
        "rule_neg_lt_1k": 0,
        "rule_neg_1k_10k": 0,
        "rule_combo_bad": 0,
        "rule_jagged_run": 0,
        "rule_isolated_spikes_allowed": 0,
        "rule_high_phase_gt_1k": 0,
        "rule_oscillatory_jagged": 0,
        "rule_tail_instability": 0,
        "rule_smooth_candidate": 0,
        "neg_lt1k_run": 0,
        "neg_lt1k_count": 0,
        "deep_neg_lt1k": False,
        "tail_flip_count": 0,
        "tail_spike_ratio": np.nan,
        "tail_resid_rms_frac": np.nan,
        "max_abs_phase_gt_1k_mrad": np.nan,
    }
    for c, default in defaults.items():
        if c not in out.columns:
            out[c] = default

    phys = out.apply(_v263_physics_inadmissibility_row, axis=1, result_type="expand")
    out["physics_inadmissibility_score"] = pd.to_numeric(phys[0], errors="coerce").fillna(0.0)
    out["physics_inadmissibility_label"] = phys[1].astype(str)
    out["physics_issue_count"] = out.apply(_v263_numeric_issue_count_row, axis=1).astype(int)

    iso, far_thr, n_train_good, manifold_cols = _v263_fit_good_iforest(out)
    out["good_manifold_score"] = np.nan
    out["good_manifold_far"] = False
    out["good_manifold_train_n"] = int(n_train_good or 0)

    if iso is not None:
        X_all, _ = _v263_good_manifold_features(out)
        all_scores = iso.score_samples(X_all)
        out["good_manifold_score"] = all_scores
        out["good_manifold_far"] = out["good_manifold_score"].astype(float) < float(far_thr)

    kk_status_s = out["kk_status"].astype(str)
    maxabs_1k = pd.to_numeric(out["max_abs_phase_gt_1k_mrad"], errors="coerce")
    hard_bad = (
        out["physics_inadmissibility_label"].astype(str).eq("severe") |
        kk_status_s.eq("KK_BAD") |
        out["rule_combo_bad"].fillna(0).astype(int).eq(1) |
        (
            out["deep_neg_lt1k"].fillna(False).astype(bool) &
            out["rule_neg_1k_10k"].fillna(0).astype(int).eq(1)
        ) |
        (
            out["rule_tail_instability"].fillna(0).astype(int).eq(1) &
            out["rule_oscillatory_jagged"].fillna(0).astype(int).eq(1) &
            out["rule_high_phase_gt_1k"].fillna(0).astype(int).eq(1)
        ) |
        (maxabs_1k >= PHASE_ABS_BAD_MRAD)
    )

    candidate_bad = out["pred_final"].astype(str).eq("bad")
    kk_ok_like = kk_status_s.isin(["KK_OK", "KK_NA"])
    manifold_ok = (~out["good_manifold_far"].fillna(False).astype(bool)) | out["good_manifold_score"].isna()

    rescue_review = (
        candidate_bad &
        (~hard_bad) &
        kk_ok_like &
        manifold_ok &
        (
            (
                (out["physics_inadmissibility_score"].astype(float) <= V263_PHYS_MILD_MAX) &
                (out["physics_issue_count"].astype(int) <= V263_MAX_ISSUES_FOR_RESCUE_MID) &
                (pd.to_numeric(out["prob_good"], errors="coerce") >= V263_RESCUE_PG_MID)
            ) |
            (
                (out["physics_inadmissibility_score"].astype(float) <= V263_PHYS_MODERATE_MAX) &
                (out["physics_issue_count"].astype(int) <= V263_MAX_ISSUES_FOR_RESCUE) &
                (pd.to_numeric(out["prob_good"], errors="coerce") >= V263_RESCUE_PG_HIGH)
            ) |
            (
                out["rule_patch_v26_reason"].astype(str).eq("nonphysical_negative_phase_lt_1k") &
                out["rule_smooth_candidate"].fillna(0).astype(int).eq(1) &
                (out["physics_inadmissibility_score"].astype(float) <= V263_PHYS_MILD_MAX) &
                (pd.to_numeric(out["prob_good"], errors="coerce") >= V263_RESCUE_PG_LOW)
            )
        )
    )

    if rescue_review.any():
        out.loc[rescue_review, "pred_final"] = "review"
        out.loc[rescue_review, "rule_patch_v26_reason"] = np.where(
            out.loc[rescue_review, "good_manifold_score"].notna(),
            "physics_plus_good_manifold_softened_to_review_v263",
            "physics_softened_to_review_v263"
        )
        out.loc[rescue_review, "rule_patch_v26_applied"] = 1
        out.loc[rescue_review, "rule_patch_v3_reason"] = out.loc[rescue_review, "rule_patch_v26_reason"]
        out.loc[rescue_review, "rule_patch_v3_applied"] = 1
        out.loc[rescue_review, "rule_patch_v2_reason"] = out.loc[rescue_review, "rule_patch_v26_reason"]
        out.loc[rescue_review, "rule_patch_v2_applied"] = 1

    out["bad_gate_v263_hard_bad"] = hard_bad.astype(int)
    out["bad_gate_v263_rescued_to_review"] = rescue_review.astype(int)

    print(
        f"[INFO] V2.6.3 BAD adjudicator | trained good-manifold n={int(n_train_good or 0)} "
        f"| rescued BAD->REVIEW={int(rescue_review.sum())} "
        f"| remaining BAD={int((out['pred_final'].astype(str) == 'bad').sum())}"
    )

    out["qc_version"] = "V2.6.3_physics_iforest_badgate"
    return out


# ==============================================================================
# PATCH V2.6.4 — Rescue isolated neg<1k(BAD) + KK_OK cases into REVIEW
# ==============================================================================
"""
Goal:
- Preserve V2.6.3 behavior that already improved BAD->REVIEW rescue.
- Specifically target the remaining BAD spectra that are still being rejected
  almost entirely because of isolated sub-1 kHz negative phase.
- Do NOT touch GOOD or broad REVIEW logic.
- Do NOT change CatBoost.
"""

if "_v264_base_apply_override" not in globals():
    _v264_base_apply_override = apply_override

V264_RESCUE_PG_MIN = 0.20
V264_RESCUE_PG_HIGH = 0.50
V264_MAX_NEG_LT1K_COUNT = 3
V264_MAX_NEG_LT1K_RUN = 3
V264_MAX_PHYS_SCORE = 4.5
V264_MAX_ISSUES = 2
V264_MAXABS_1K_REVIEW = max(float(PHASE_ABS_REVIEW_MRAD), 250.0)

def apply_override(features: pd.DataFrame) -> pd.DataFrame:
    out = _v264_base_apply_override(features.copy())

    # Ensure required columns exist
    defaults = {
        "pred_final": None,
        "kk_status": "KK_NA",
        "rule_patch_v26_reason": "",
        "rule_neg_lt_1k": 0,
        "rule_neg_1k_10k": 0,
        "rule_combo_bad": 0,
        "rule_jagged_run": 0,
        "rule_oscillatory_jagged": 0,
        "rule_tail_instability": 0,
        "rule_high_phase_gt_1k": 0,
        "rule_isolated_spikes_allowed": 0,
        "rule_smooth_candidate": 0,
        "prob_good": np.nan,
        "physics_inadmissibility_score": np.nan,
        "physics_issue_count": 0,
        "neg_lt1k_count": 0,
        "neg_lt1k_run": 0,
        "deep_neg_lt1k": False,
        "max_abs_phase_gt_1k_mrad": np.nan,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    kk_ok = out["kk_status"].astype(str).eq("KK_OK")
    still_bad = out["pred_final"].astype(str).eq("bad")

    isolated_neg_reason = out["rule_patch_v26_reason"].astype(str).eq("nonphysical_negative_phase_lt_1k")
    no_midband_neg = out["rule_neg_1k_10k"].fillna(0).astype(int).eq(0)
    no_combo = out["rule_combo_bad"].fillna(0).astype(int).eq(0)
    no_jrun = out["rule_jagged_run"].fillna(0).astype(int).eq(0)
    no_osc = out["rule_oscillatory_jagged"].fillna(0).astype(int).eq(0)
    no_tail = out["rule_tail_instability"].fillna(0).astype(int).eq(0)
    no_high_phase = out["rule_high_phase_gt_1k"].fillna(0).astype(int).eq(0)
    no_iso_spikes = out["rule_isolated_spikes_allowed"].fillna(0).astype(int).eq(0)

    mild_neg_extent = (
        pd.to_numeric(out["neg_lt1k_count"], errors="coerce").fillna(0).astype(float) <= V264_MAX_NEG_LT1K_COUNT
    ) & (
        pd.to_numeric(out["neg_lt1k_run"], errors="coerce").fillna(0).astype(float) <= V264_MAX_NEG_LT1K_RUN
    ) & (
        ~out["deep_neg_lt1k"].fillna(False).astype(bool)
    )

    mild_phys = (
        pd.to_numeric(out["physics_inadmissibility_score"], errors="coerce").fillna(999.0).astype(float) <= V264_MAX_PHYS_SCORE
    ) & (
        pd.to_numeric(out["physics_issue_count"], errors="coerce").fillna(999).astype(float) <= V264_MAX_ISSUES
    )

    bounded_amp = (
        pd.to_numeric(out["max_abs_phase_gt_1k_mrad"], errors="coerce").fillna(0.0).astype(float) <= V264_MAXABS_1K_REVIEW
    )

    smooth_or_highpg = (
        out["rule_smooth_candidate"].fillna(0).astype(int).eq(1)
    ) | (
        pd.to_numeric(out["prob_good"], errors="coerce").fillna(0.0).astype(float) >= V264_RESCUE_PG_HIGH
    )

    pg_ok = pd.to_numeric(out["prob_good"], errors="coerce").fillna(0.0).astype(float) >= V264_RESCUE_PG_MIN

    rescue_review = (
        still_bad &
        kk_ok &
        isolated_neg_reason &
        no_midband_neg &
        no_combo &
        no_jrun &
        no_osc &
        no_tail &
        no_high_phase &
        no_iso_spikes &
        mild_neg_extent &
        mild_phys &
        bounded_amp &
        smooth_or_highpg &
        pg_ok
    )

    if rescue_review.any():
        out.loc[rescue_review, "pred_final"] = "review"
        out.loc[rescue_review, "rule_patch_v26_reason"] = "isolated_neg_lt1k_softened_to_review_v264"
        out.loc[rescue_review, "rule_patch_v26_applied"] = 1
        out.loc[rescue_review, "rule_patch_v3_reason"] = out.loc[rescue_review, "rule_patch_v26_reason"]
        out.loc[rescue_review, "rule_patch_v3_applied"] = 1
        out.loc[rescue_review, "rule_patch_v2_reason"] = out.loc[rescue_review, "rule_patch_v26_reason"]
        out.loc[rescue_review, "rule_patch_v2_applied"] = 1

    out["bad_gate_v264_rescued_to_review"] = rescue_review.astype(int)

    print(
        f"[INFO] V2.6.4 isolated neg<1k rescue | rescued BAD->REVIEW={int(rescue_review.sum())} "
        f"| remaining BAD={int((out['pred_final'].astype(str) == 'bad').sum())}"
    )

    out["qc_version"] = "V2.6.4_isolated_neglt1k_rescue"
    return out



# ==============================================================================
# PATCH V2.6.5 — Compound neg<1k adjudicator for remaining BAD cases
# ==============================================================================
"""
Goal:
- Preserve the V2.6.3 / V2.6.4 gains: GOOD stays strict, REVIEW stays useful.
- Target the remaining BAD spectra that are still dominated by sub-1 kHz negative phase,
  but now allow a *small* number of mild companion warnings to be softened into REVIEW.
- Keep truly multi-failure / inadmissible negative-phase spectra in BAD.
"""

if "_v265_base_apply_override" not in globals():
    _v265_base_apply_override = apply_override

V265_PG_MIN = 0.25
V265_PG_STRONG = 0.80
V265_NEG_COUNT_MAX = 6
V265_NEG_RUN_MAX = 4
V265_PHYS_SCORE_MAX = 6.5
V265_PHYS_SCORE_STRONG = 5.0
V265_MAX_ISSUES = 4
V265_MAX_ISSUES_STRONG = 3
V265_MAXABS_1K_REVIEW = max(float(PHASE_ABS_REVIEW_MRAD), 320.0)

def apply_override(features: pd.DataFrame) -> pd.DataFrame:
    out = _v265_base_apply_override(features.copy())

    defaults = {
        "pred_final": None,
        "kk_status": "KK_NA",
        "rule_patch_v26_reason": "",
        "rule_neg_lt_1k": 0,
        "rule_neg_1k_10k": 0,
        "rule_combo_bad": 0,
        "rule_jagged_run": 0,
        "rule_oscillatory_jagged": 0,
        "rule_tail_instability": 0,
        "rule_high_phase_gt_1k": 0,
        "rule_isolated_spikes_allowed": 0,
        "prob_good": np.nan,
        "physics_inadmissibility_score": np.nan,
        "physics_inadmissibility_label": "mild",
        "physics_issue_count": 0,
        "neg_lt1k_count": 0,
        "neg_lt1k_run": 0,
        "deep_neg_lt1k": False,
        "good_manifold_far": False,
        "good_manifold_score": np.nan,
        "max_abs_phase_gt_1k_mrad": np.nan,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    still_bad = out["pred_final"].astype(str).eq("bad")
    kk_ok = out["kk_status"].astype(str).eq("KK_OK")
    reason_s = out["rule_patch_v26_reason"].astype(str)
    neg_reason_family = reason_s.str.contains("negative_phase_lt_1k", regex=False) | out["rule_neg_lt_1k"].fillna(0).astype(int).eq(1)

    midband_neg = out["rule_neg_1k_10k"].fillna(0).astype(int).eq(1)
    combo_bad = out["rule_combo_bad"].fillna(0).astype(int).eq(1)
    jagged = out["rule_jagged_run"].fillna(0).astype(int).eq(1)
    osc = out["rule_oscillatory_jagged"].fillna(0).astype(int).eq(1)
    rough = jagged | osc
    tail = out["rule_tail_instability"].fillna(0).astype(int).eq(1)
    high_phase = out["rule_high_phase_gt_1k"].fillna(0).astype(int).eq(1)
    iso_spikes = out["rule_isolated_spikes_allowed"].fillna(0).astype(int).eq(1)

    deep_neg = out["deep_neg_lt1k"].fillna(False).astype(bool)
    neg_count = pd.to_numeric(out["neg_lt1k_count"], errors="coerce").fillna(0.0).astype(float)
    neg_run = pd.to_numeric(out["neg_lt1k_run"], errors="coerce").fillna(0.0).astype(float)
    phys_score = pd.to_numeric(out["physics_inadmissibility_score"], errors="coerce").fillna(999.0).astype(float)
    issue_count = pd.to_numeric(out["physics_issue_count"], errors="coerce").fillna(999.0).astype(float)
    prob_good = pd.to_numeric(out["prob_good"], errors="coerce").fillna(0.0).astype(float)
    maxabs_1k = pd.to_numeric(out["max_abs_phase_gt_1k_mrad"], errors="coerce").fillna(np.inf).astype(float)
    manifold_ok = (~out["good_manifold_far"].fillna(False).astype(bool)) | out["good_manifold_score"].isna()

    companion_count = (
        midband_neg.astype(int)
        + rough.astype(int)
        + tail.astype(int)
        + high_phase.astype(int)
        + iso_spikes.astype(int)
    )

    severe_compound = (
        combo_bad
        | deep_neg
        | (maxabs_1k >= float(PHASE_ABS_BAD_MRAD))
        | out["physics_inadmissibility_label"].astype(str).eq("severe")
        | (companion_count >= 3)
        | ((midband_neg & rough))
        | ((midband_neg & tail))
        | ((rough & tail & high_phase))
    )

    bounded_neg_extent = (
        (neg_count <= V265_NEG_COUNT_MAX) &
        (neg_run <= V265_NEG_RUN_MAX)
    )
    bounded_phys = (
        (phys_score <= V265_PHYS_SCORE_MAX) &
        (issue_count <= V265_MAX_ISSUES)
    )
    bounded_amp = maxabs_1k <= V265_MAXABS_1K_REVIEW

    mild_companion_path = (
        (companion_count <= 1) &
        bounded_neg_extent &
        bounded_phys &
        bounded_amp &
        (prob_good >= V265_PG_MIN) &
        manifold_ok
    )

    strong_model_path = (
        (companion_count <= 2) &
        (phys_score <= V265_PHYS_SCORE_STRONG) &
        (issue_count <= V265_MAX_ISSUES_STRONG) &
        bounded_amp &
        (prob_good >= V265_PG_STRONG) &
        manifold_ok
    )

    rescue_review = (
        still_bad &
        kk_ok &
        neg_reason_family &
        (~severe_compound) &
        (mild_companion_path | strong_model_path)
    )

    if rescue_review.any():
        out.loc[rescue_review, "pred_final"] = "review"
        out.loc[rescue_review, "rule_patch_v26_reason"] = "compound_neg_lt1k_mild_companions_softened_to_review_v265"
        out.loc[rescue_review, "rule_patch_v26_applied"] = 1
        out.loc[rescue_review, "rule_patch_v3_reason"] = out.loc[rescue_review, "rule_patch_v26_reason"]
        out.loc[rescue_review, "rule_patch_v3_applied"] = 1
        out.loc[rescue_review, "rule_patch_v2_reason"] = out.loc[rescue_review, "rule_patch_v26_reason"]
        out.loc[rescue_review, "rule_patch_v2_applied"] = 1

    out["bad_gate_v265_companion_count"] = companion_count.astype(int)
    out["bad_gate_v265_severe_compound"] = severe_compound.astype(int)
    out["bad_gate_v265_rescued_to_review"] = rescue_review.astype(int)

    print(
        f"[INFO] V2.6.5 compound neg<1k adjudicator | rescued BAD->REVIEW={int(rescue_review.sum())} "
        f"| remaining BAD={int((out['pred_final'].astype(str) == 'bad').sum())}"
    )

    out["qc_version"] = "V2.6.5_compound_neglt1k_adjudicator"
    return out



# ==============================================================================
# PATCH V2.6.6 — BAD severity score + hidden BAD subtypes + rescue-to-review gate
# ==============================================================================
"""
Goal:
- Preserve the current GOOD / REVIEW behavior as much as possible.
- Make BAD more physically selective by grading reject evidence with a composite severity score.
- Keep clearly compound / nonphysical spectra in BAD.
- Soften weaker KK_OK BAD calls into REVIEW when the evidence is weak-to-moderate rather than truly inadmissible.
"""

if "_v266_prev_apply_override" not in globals():
    _v266_prev_apply_override = apply_override

# --- V2.6.6 tunables ---
V266_REVIEW_SCORE_MAX_ISO = 4
V266_REVIEW_SCORE_MAX_COMP = 6
V266_REVIEW_PHYS_MAX = 5
V266_REVIEW_PG_MIN_ISO = 0.55
V266_REVIEW_PG_MIN_COMP = 0.72
V266_REVIEW_COMPANION_MAX_ISO = 1
V266_REVIEW_COMPANION_MAX_COMP = 2
V266_REVIEW_NEGCOUNT_MAX_ISO = 4
V266_REVIEW_NEGRUN_MAX_ISO = 3
V266_REVIEW_MAXABS_1K = 180.0
V266_REVIEW_TAIL_SPIKE_MAX = 7.0
V266_REVIEW_TAIL_RMS_MAX = 0.09
V266_REVIEW_TAIL_FLIPS_MAX = 4

def _v266_bool(v) -> bool:
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass
    return bool(v)

def _v266_bad_severity_triplet_row(r):
    """
    Returns:
      score (float): composite reject evidence score
      label (str): low / moderate / high
      subtype (str): bad_nonphysical / bad_artifact / bad_low_information
    """
    neg_lt1k = _v266_bool(getattr(r, "rule_neg_lt_1k", 0))
    midband_neg = _v266_bool(getattr(r, "rule_neg_1k_10k", 0))
    combo_bad = _v266_bool(getattr(r, "rule_combo_bad", 0))
    jagged = _v266_bool(getattr(r, "rule_jagged_run", 0))
    osc = _v266_bool(getattr(r, "rule_oscillatory_jagged", 0))
    tail = _v266_bool(getattr(r, "rule_tail_instability", 0))
    high_phase = _v266_bool(getattr(r, "rule_high_phase_gt_1k", 0))
    iso_spikes = _v266_bool(getattr(r, "rule_isolated_spikes_allowed", 0))
    deep_neg = _v266_bool(getattr(r, "deep_neg_lt1k", False))

    neg_count = float(pd.to_numeric(getattr(r, "neg_lt1k_count", 0), errors="coerce") or 0.0)
    neg_run = float(pd.to_numeric(getattr(r, "neg_lt1k_run", 0), errors="coerce") or 0.0)
    tail_flips = float(pd.to_numeric(getattr(r, "tail_flip_count", 0), errors="coerce") or 0.0)

    tail_spike = pd.to_numeric(getattr(r, "tail_spike_ratio", np.nan), errors="coerce")
    tail_rms = pd.to_numeric(getattr(r, "tail_resid_rms_frac", np.nan), errors="coerce")
    maxabs_1k = pd.to_numeric(getattr(r, "max_abs_phase_gt_1k_mrad", np.nan), errors="coerce")
    prob_good = pd.to_numeric(getattr(r, "prob_good", np.nan), errors="coerce")
    phys_score = pd.to_numeric(getattr(r, "physics_inadmissibility_score", np.nan), errors="coerce")
    kk_mu = pd.to_numeric(getattr(r, "kk_mu", np.nan), errors="coerce")
    kk_chi2 = pd.to_numeric(getattr(r, "kk_pseudo_chi2", np.nan), errors="coerce")
    kk_status = str(getattr(r, "kk_status", "KK_NA"))

    score = 0.0

    # Core negative-phase evidence
    if neg_lt1k:
        score += 1.0
    if neg_count >= 2:
        score += 1.0
    if neg_count >= 4:
        score += 1.0
    if neg_run >= 2:
        score += 1.0
    if neg_run >= 4:
        score += 1.0
    if deep_neg:
        score += 2.0

    # Companion spectral issues
    if midband_neg:
        score += 2.0
    if combo_bad:
        score += 2.0
    if jagged:
        score += 1.0
    if osc:
        score += 1.0
    if tail:
        score += 1.5
    if high_phase:
        score += 1.0
    if iso_spikes:
        score += 0.5

    # Continuous/semi-continuous cues
    if np.isfinite(maxabs_1k) and maxabs_1k >= 120.0:
        score += 0.5
    if np.isfinite(maxabs_1k) and maxabs_1k >= 220.0:
        score += 1.0
    if tail_flips >= 3:
        score += 0.5
    if tail_flips >= 5:
        score += 0.5
    if np.isfinite(tail_spike) and tail_spike >= 5.0:
        score += 0.5
    if np.isfinite(tail_spike) and tail_spike >= 8.0:
        score += 0.5
    if np.isfinite(tail_rms) and tail_rms >= 0.05:
        score += 0.5
    if np.isfinite(tail_rms) and tail_rms >= 0.10:
        score += 0.5

    # Physics consistency
    if kk_status == "KK_REVIEW":
        score += 1.0
    elif kk_status == "KK_BAD":
        score += 3.0
    if np.isfinite(phys_score) and phys_score >= 4:
        score += 1.0
    if np.isfinite(phys_score) and phys_score >= 6:
        score += 1.0
    if np.isfinite(kk_chi2) and kk_chi2 >= 0.20:
        score += 0.5
    if np.isfinite(kk_chi2) and kk_chi2 >= 0.50:
        score += 0.5
    if np.isfinite(kk_mu) and kk_mu >= 0.45:
        score += 0.5

    # Model disagreement slightly softens otherwise plausible curves
    if np.isfinite(prob_good) and prob_good < 0.20:
        score += 0.5
    if np.isfinite(prob_good) and prob_good < 0.05:
        score += 0.5

    if score <= 3.5:
        label = "low"
    elif score <= 6.5:
        label = "moderate"
    else:
        label = "high"

    # Hidden internal subtype
    if (
        combo_bad
        or (midband_neg and (jagged or osc or tail))
        or kk_status == "KK_BAD"
        or (np.isfinite(phys_score) and phys_score >= 6)
        or deep_neg
    ):
        subtype = "bad_nonphysical"
    elif (
        (tail or high_phase or iso_spikes or osc or jagged)
        and not deep_neg
        and not combo_bad
        and not midband_neg
    ):
        subtype = "bad_artifact"
    else:
        subtype = "bad_low_information"

    return float(score), label, subtype


def apply_override(features: pd.DataFrame) -> pd.DataFrame:
    out = _v266_prev_apply_override(features.copy())

    defaults = {
        "pred_final": None,
        "prob_good": np.nan,
        "kk_status": "KK_NA",
        "kk_mu": np.nan,
        "kk_pseudo_chi2": np.nan,
        "rule_patch_v26_reason": "",
        "rule_patch_v2_reason": "",
        "rule_patch_v3_reason": "",
        "rule_neg_lt_1k": 0,
        "rule_neg_1k_10k": 0,
        "rule_combo_bad": 0,
        "rule_jagged_run": 0,
        "rule_high_phase_gt_1k": 0,
        "rule_oscillatory_jagged": 0,
        "rule_tail_instability": 0,
        "rule_isolated_spikes_allowed": 0,
        "deep_neg_lt1k": False,
        "neg_lt1k_count": 0,
        "neg_lt1k_run": 0,
        "tail_flip_count": 0,
        "tail_spike_ratio": np.nan,
        "tail_resid_rms_frac": np.nan,
        "max_abs_phase_gt_1k_mrad": np.nan,
        "physics_inadmissibility_score": np.nan,
        "physics_inadmissibility_label": "unknown",
        "physics_issue_count": 0,
        "good_manifold_far": False,
        "good_manifold_score": np.nan,
    }
    for c, default in defaults.items():
        if c not in out.columns:
            out[c] = default

    trip = out.apply(_v266_bad_severity_triplet_row, axis=1, result_type="expand")
    out["bad_severity_score"] = pd.to_numeric(trip[0], errors="coerce").fillna(0.0)
    out["bad_severity_label"] = trip[1].astype(str)
    out["bad_internal_subtype"] = trip[2].astype(str)

    # Companion count for rescue adjudication
    midband_neg = out["rule_neg_1k_10k"].fillna(0).astype(int).eq(1)
    combo_bad = out["rule_combo_bad"].fillna(0).astype(int).eq(1)
    jagged = out["rule_jagged_run"].fillna(0).astype(int).eq(1)
    osc = out["rule_oscillatory_jagged"].fillna(0).astype(int).eq(1)
    tail = out["rule_tail_instability"].fillna(0).astype(int).eq(1)
    high_phase = out["rule_high_phase_gt_1k"].fillna(0).astype(int).eq(1)
    iso_spikes = out["rule_isolated_spikes_allowed"].fillna(0).astype(int).eq(1)
    deep_neg = out["deep_neg_lt1k"].fillna(False).astype(bool)

    still_bad = out["pred_final"].astype(str).eq("bad")
    kk_ok = out["kk_status"].astype(str).eq("KK_OK")
    prob_good = pd.to_numeric(out["prob_good"], errors="coerce").fillna(0.0).astype(float)
    phys_score = pd.to_numeric(out["physics_inadmissibility_score"], errors="coerce").fillna(999.0).astype(float)
    issue_count = pd.to_numeric(out["physics_issue_count"], errors="coerce").fillna(999.0).astype(float)
    neg_count = pd.to_numeric(out["neg_lt1k_count"], errors="coerce").fillna(0.0).astype(float)
    neg_run = pd.to_numeric(out["neg_lt1k_run"], errors="coerce").fillna(0.0).astype(float)
    maxabs_1k = pd.to_numeric(out["max_abs_phase_gt_1k_mrad"], errors="coerce").fillna(np.inf).astype(float)
    tail_flips = pd.to_numeric(out["tail_flip_count"], errors="coerce").fillna(0.0).astype(float)
    tail_spike = pd.to_numeric(out["tail_spike_ratio"], errors="coerce")
    tail_rms = pd.to_numeric(out["tail_resid_rms_frac"], errors="coerce")
    manifold_ok = (~out["good_manifold_far"].fillna(False).astype(bool)) | out["good_manifold_score"].isna()

    companion_count = (
        midband_neg.astype(int)
        + combo_bad.astype(int)
        + jagged.astype(int)
        + osc.astype(int)
        + tail.astype(int)
        + high_phase.astype(int)
        + iso_spikes.astype(int)
    )

    neg_reason_family = (
        out["rule_patch_v26_reason"].astype(str).str.contains("negative_phase_lt_1k", regex=False)
        | out["rule_patch_v26_reason"].astype(str).str.contains("neg_lt1k", regex=False)
        | out["rule_neg_lt_1k"].fillna(0).astype(int).eq(1)
    )

    tail_not_severe = (
        tail_flips.le(V266_REVIEW_TAIL_FLIPS_MAX)
        & (tail_spike.isna() | tail_spike.le(V266_REVIEW_TAIL_SPIKE_MAX))
        & (tail_rms.isna() | tail_rms.le(V266_REVIEW_TAIL_RMS_MAX))
    )

    isolated_neg_rescue = (
        still_bad
        & kk_ok
        & neg_reason_family
        & (~deep_neg)
        & (~midband_neg)
        & (~combo_bad)
        & companion_count.le(V266_REVIEW_COMPANION_MAX_ISO)
        & neg_count.le(V266_REVIEW_NEGCOUNT_MAX_ISO)
        & neg_run.le(V266_REVIEW_NEGRUN_MAX_ISO)
        & tail_not_severe
        & maxabs_1k.le(V266_REVIEW_MAXABS_1K)
        & phys_score.le(V266_REVIEW_PHYS_MAX)
        & out["bad_severity_score"].le(V266_REVIEW_SCORE_MAX_ISO)
        & prob_good.ge(V266_REVIEW_PG_MIN_ISO)
        & (manifold_ok | prob_good.ge(0.80))
    )

    compound_mild_rescue = (
        still_bad
        & kk_ok
        & neg_reason_family
        & (~deep_neg)
        & (~combo_bad)
        & companion_count.le(V266_REVIEW_COMPANION_MAX_COMP)
        & tail_not_severe
        & maxabs_1k.le(V266_REVIEW_MAXABS_1K)
        & phys_score.le(V266_REVIEW_PHYS_MAX)
        & issue_count.le(4)
        & out["bad_severity_score"].le(V266_REVIEW_SCORE_MAX_COMP)
        & out["bad_severity_label"].astype(str).isin(["low", "moderate"])
        & prob_good.ge(V266_REVIEW_PG_MIN_COMP)
        & manifold_ok
    )

    artifact_rescue = (
        still_bad
        & kk_ok
        & out["bad_internal_subtype"].astype(str).eq("bad_artifact")
        & out["bad_severity_label"].astype(str).isin(["low", "moderate"])
        & phys_score.le(4)
        & issue_count.le(3)
        & tail_not_severe
        & prob_good.ge(0.68)
        & manifold_ok
    )

    rescue_review = isolated_neg_rescue | compound_mild_rescue | artifact_rescue

    if rescue_review.any():
        out.loc[rescue_review, "pred_final"] = "review"
        out.loc[isolated_neg_rescue, "rule_patch_v26_reason"] = "bad_severity_isolated_neg_softened_to_review_v266"
        out.loc[compound_mild_rescue & ~isolated_neg_rescue, "rule_patch_v26_reason"] = "bad_severity_compound_mild_softened_to_review_v266"
        out.loc[artifact_rescue & ~(isolated_neg_rescue | compound_mild_rescue), "rule_patch_v26_reason"] = "bad_severity_artifact_softened_to_review_v266"
        out.loc[rescue_review, "rule_patch_v26_applied"] = 1
        out.loc[rescue_review, "rule_patch_v3_reason"] = out.loc[rescue_review, "rule_patch_v26_reason"]
        out.loc[rescue_review, "rule_patch_v3_applied"] = 1
        out.loc[rescue_review, "rule_patch_v2_reason"] = out.loc[rescue_review, "rule_patch_v26_reason"]
        out.loc[rescue_review, "rule_patch_v2_applied"] = 1

    out["bad_gate_v266_companion_count"] = companion_count.astype(int)
    out["bad_gate_v266_rescued_to_review"] = rescue_review.astype(int)

    # Compact BAD audit summary for iteration
    bad_mask = out["pred_final"].astype(str).eq("bad")
    if bad_mask.any():
        audit = (
            out.loc[bad_mask, ["bad_internal_subtype", "bad_severity_label"]]
            .value_counts(dropna=False)
            .rename("n")
            .reset_index()
            .sort_values(["bad_internal_subtype", "bad_severity_label"])
        )
        print("[INFO] V2.6.6 BAD audit summary:")
        for _, rr in audit.iterrows():
            print(f"    - {rr['bad_internal_subtype']} | {rr['bad_severity_label']}: {int(rr['n'])}")

    print(
        f"[INFO] V2.6.6 BAD severity gate | rescued BAD->REVIEW={int(rescue_review.sum())} "
        f"| remaining BAD={int((out['pred_final'].astype(str) == 'bad').sum())}"
    )

    out["qc_version"] = "V2.6.6_bad_severity_score"
    return out
