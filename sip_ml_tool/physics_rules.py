# Extracted from SIP_ML_Tool.ipynb main workflow cell.
# Initial physics-guided Good/Review/Bad override logic.

def apply_override(features: pd.DataFrame) -> pd.DataFrame:
    """
    FINAL label in {good, bad, review} based on:
      - CatBoost base + confidence
      - raw-curve rules
      - KK/log-F consistency gate (REVIEW-first, but promotes *severe* HF instability to BAD)
    """

    # --- TUNABLES (refined based on observed "bad-in-review" failure mode) ---
    # Promote obviously broken HF tails to BAD when they are also jagged/unstable.
    HF_SEVERE_ABS_MAXPHASE_BAD = 400.0   # mrad, absolute hard floor (very conservative)
    HF_SEVERE_REF_MULT_BAD = 3.0         # relative to ref_high_phase (per group)
    HF_SEVERE_BIGJUMPS_BAD = 6           # across curve; typically huge for oscillatory tails
    HF_SEVERE_TAIL_SPIKE_RATIO_BAD = 8.0 # if tail_spike_ratio available

    # Reduce false REVIEW when only KK/high-phase triggered but curve is smooth + model is very confident GOOD.
    KEEP_GOOD_IF_ONLY_KK_PG = 0.99
    KEEP_GOOD_IF_ONLY_HIGH_PHASE_PG = 0.99
    LOWAMP_MAXABS_1K_MRAD = 120.0      # mrad, smooth low-amplitude curves above 1 kHz are usually OK
    MODEL_STRONG_BAD_PB = 0.97        # if model p(bad) exceeds this, route rescued cases to REVIEW instead of GOOD

    # Build reference for "high phase at >1k compared to similar datasets"
    group_cols = [c for c in HIGH_PHASE_GROUP_COLS if c in features.columns]
    ref = (
        features.dropna(subset=["max_phase_gt_1k_mrad"])
        .groupby(group_cols)["max_phase_gt_1k_mrad"]
        .quantile(HIGH_PHASE_REF_Q)
        .reset_index()
        .rename(columns={"max_phase_gt_1k_mrad": "ref_high_phase"})
    )
    features = features.merge(ref, on=group_cols, how="left")

    features["rule_high_phase_gt_1k"] = (
        features["max_phase_gt_1k_mrad"] > (HIGH_PHASE_MULT * features["ref_high_phase"])
    ).fillna(False)

    # Combo penalty: jagged run + negative in 1k–10k
    features["rule_combo_bad"] = (
        (features["rule_jagged_run"] == 1) & (features["rule_neg_1k_10k"] == 1)
    )

    # Review candidates from confidence
    best_t = float(features["good_threshold_used"].dropna().iloc[0]) if "good_threshold_used" in features.columns else 0.5
    features["rule_near_thr"] = (np.abs(features["prob_good"] - best_t) < NEAR_THRESHOLD_BAND)
    features["rule_low_conf"] = (features["prob_max"] < CAUTIOUS_THRESHOLD)

    # ---------------- KK / LOG-F gate (derived from labeled GOOD) ----------------
    features["kk_flag_review"] = False
    features["kk_flag_bad"] = False
    features["kk_status"] = "KK_NA"

    if USE_KK_GATE and all(c in features.columns for c in ["kk_pseudo_chi2", "kk_mu", "kk_resid_rms"]):
        # Prefer calibration from labeled GOOD
        good_ref = features[(features.get("true_label", "") == "good")].dropna(subset=["kk_pseudo_chi2", "kk_mu", "kk_resid_rms"])
        use_ref = good_ref if len(good_ref) >= KK_MIN_GOOD_FOR_CAL else features.dropna(subset=["kk_pseudo_chi2", "kk_mu", "kk_resid_rms"])

        if not use_ref.empty:
            # thresholds
            chi_review = float(use_ref["kk_pseudo_chi2"].quantile(KK_Q_REVIEW))
            rms_review = float(use_ref["kk_resid_rms"].quantile(KK_Q_REVIEW))
            mu_review  = float(use_ref["kk_mu"].quantile(KK_MU_Q_REVIEW))

            chi_bad = float(use_ref["kk_pseudo_chi2"].quantile(KK_Q_BAD))
            rms_bad = float(use_ref["kk_resid_rms"].quantile(KK_Q_BAD))
            mu_bad  = float(use_ref["kk_mu"].quantile(KK_MU_Q_BAD))

            # store for audit
            features["kk_thr_chi_review"] = chi_review
            features["kk_thr_rms_review"] = rms_review
            features["kk_thr_mu_review"]  = mu_review
            features["kk_thr_chi_bad"]    = chi_bad
            features["kk_thr_rms_bad"]    = rms_bad
            features["kk_thr_mu_bad"]     = mu_bad

            chi = features["kk_pseudo_chi2"]
            rms = features["kk_resid_rms"]
            mu  = features["kk_mu"]

            # KK_BAD definition (extreme)
            if KK_BAD_REQUIRE_BOTH_CHI_AND_RMS:
                kk_bad = ((chi > chi_bad) & (rms > rms_bad)) | (mu < mu_bad)
            else:
                kk_bad = (chi > chi_bad) | (rms > rms_bad) | (mu < mu_bad)
            features["kk_flag_bad"] = kk_bad.fillna(False)

            # KK_REVIEW (includes moderate tail, excludes KK_BAD)
            kk_review = ((chi > chi_review) | (rms > rms_review) | (mu < mu_review)) & (~features["kk_flag_bad"])
            features["kk_flag_review"] = kk_review.fillna(False)

            # KK status label (independent of whether we hard-bad)
            features["kk_status"] = np.where(
                features["kk_flag_bad"], "KK_BAD",
                np.where(features["kk_flag_review"], "KK_REVIEW", "KK_OK")
            )

            ref_name = "labeled GOOD" if len(good_ref) >= KK_MIN_GOOD_FOR_CAL else "ALL rows (fallback)"
            print(f"[INFO] KK gate calibrated from {ref_name}: "
                  f"REVIEW chi>{chi_review:.3g} rms>{rms_review:.3g} mu<{mu_review:.3g} | "
                  f"BAD chi>{chi_bad:.3g} rms>{rms_bad:.3g} mu<{mu_bad:.3g} | "
                  f"hard_bad={'ON' if KK_HARD_BAD_ENABLED else 'OFF'}")

    # ---------------- FINAL DECISION ----------------
    final = []
    for r in features.itertuples(index=False):

        kk_status = str(getattr(r, "kk_status", "KK_NA"))
        pg = float(getattr(r, "prob_good", np.nan))
        pb = float(getattr(r, "prob_bad", np.nan))

        # Common rule flags / metrics
        rule_neg_lt = (getattr(r, "rule_neg_lt_1k") == 1)
        rule_neg_hf = (getattr(r, "rule_neg_1k_10k") == 1)
        rule_jrun   = (getattr(r, "rule_jagged_run") == 1)
        iso_allowed = (getattr(r, "rule_isolated_spikes_allowed") == 1)
        big_jumps   = int(getattr(r, "rule_big_jump_count", 0) or 0)

        maxph_1k = float(getattr(r, "max_phase_gt_1k_mrad", np.nan))
        maxabs = float(getattr(r, "max_abs_phase_mrad", np.nan))
        maxabs_1k = float(getattr(r, "max_abs_phase_gt_1k_mrad", np.nan))
        abs_extreme = bool(np.isfinite(maxabs_1k) and (maxabs_1k >= PHASE_ABS_BAD_MRAD))
        abs_review = bool(np.isfinite(maxabs_1k) and (maxabs_1k >= PHASE_ABS_REVIEW_MRAD))
        refhp    = float(getattr(r, "ref_high_phase", np.nan))
        high_phase = bool(getattr(r, "rule_high_phase_gt_1k", False))

        # Absolute phase plausibility: extremely large phase at high frequency is almost certainly bad
        if abs_extreme:
            final.append("bad")
            continue


        # New shape rules (if present)
        tail_instab = bool(getattr(r, "rule_tail_instability", False))
        osc_jagged  = bool(getattr(r, "rule_oscillatory_jagged", False))
        smooth_candidate = bool(getattr(r, "rule_smooth_candidate", False))
        tail_spike_ratio = float(getattr(r, "tail_spike_ratio", np.nan))
        tail_flip_count = int(getattr(r, "tail_flip_count", 0) or 0)
        tail_rms_frac = float(getattr(r, "tail_resid_rms_frac", np.nan))

        # Benign tail: smooth HF upturn can look "tail-unstable" for small phase-range spectra.
        # If model is very confident GOOD and KK is OK, do not force REVIEW on mild tail metrics.
        tail_benign_patch = False
        try:
            tail_benign_patch = bool(
                tail_instab and (kk_status == "KK_OK") and (pg >= 0.99) and (not abs_review) and
                (np.isfinite(tail_spike_ratio) and tail_spike_ratio < 12.0) and
                (tail_flip_count <= 4) and
                (np.isfinite(tail_rms_frac) and tail_rms_frac < 0.08)
            )
        except Exception:
            tail_benign_patch = False

        if tail_benign_patch:
            tail_instab = False


        # 0) Optional KK extreme -> BAD (disabled by default)
        if KK_HARD_BAD_ENABLED and kk_status == "KK_BAD":
            final.append("bad")
            continue

        # 1) Hard BAD overrides
        if rule_neg_lt:
            final.append("bad")
            continue
        if bool(getattr(r, "rule_combo_bad", False)):
            final.append("bad")
            continue

        # 1b) Promote *severe* HF instability to BAD (fixes "bad-in-review" oscillatory blowups)
        # Condition combines a large HF magnitude (absolute or relative) WITH instability evidence.
        hf_mag_severe = False
        if np.isfinite(maxph_1k):
            rel_ok = np.isfinite(refhp) and (maxph_1k > (HF_SEVERE_REF_MULT_BAD * refhp))
            abs_ok = (maxph_1k >= HF_SEVERE_ABS_MAXPHASE_BAD)
            hf_mag_severe = bool(abs_ok or rel_ok or abs_review)

        hf_instability_severe = bool(
            (rule_jrun and big_jumps >= 2) or
            (big_jumps >= HF_SEVERE_BIGJUMPS_BAD) or
            tail_instab or
            osc_jagged or
            (np.isfinite(tail_spike_ratio) and tail_spike_ratio >= HF_SEVERE_TAIL_SPIKE_RATIO_BAD)
        )

        if hf_mag_severe and hf_instability_severe:
            final.append("bad")
            continue

        # Review candidates from confidence
        near_thr = bool(getattr(r, "rule_near_thr", False))
        low_conf = bool(getattr(r, "rule_low_conf", False))

        # 2) Decide whether KK/high-phase should force REVIEW:
        # If KK/high-phase is the ONLY issue, and the curve is smooth + model is very confident GOOD, keep GOOD.
        kk_review_like = (kk_status in ("KK_REVIEW", "KK_BAD"))
        other_rule_like = any([rule_neg_hf, rule_jrun, iso_allowed, high_phase, bool(getattr(r, "rule_combo_bad", False)), near_thr, low_conf])

        kk_only = kk_review_like and (not other_rule_like)
        high_only = high_phase and (not any([rule_neg_hf, rule_jrun, iso_allowed, near_thr, low_conf, kk_review_like]))

        very_conf_good = (not np.isnan(pg)) and (pg >= KEEP_GOOD_IF_ONLY_KK_PG)

        # If only KK and smooth + very confident GOOD, do NOT route to review.
        if kk_only and smooth_candidate and very_conf_good:
            final.append(str(getattr(r, "pred_binary")))
            continue

        # If only high-phase and smooth + very confident GOOD, do NOT route to review.
        if high_only and smooth_candidate and (not np.isnan(pg)) and (pg >= KEEP_GOOD_IF_ONLY_HIGH_PHASE_PG):
            final.append(str(getattr(r, "pred_binary")))
            continue

        # 2b) Prevent "model-only BAD" for smooth, KK-consistent, low-amplitude spectra.
        # These often appear as clean, gently varying curves but may be over-associated with BAD in training labels.
        pred_bin = str(getattr(r, "pred_binary"))
        if pred_bin == "bad":
            kk_ok = (kk_status == "KK_OK")
            combo_bad = bool(getattr(r, "rule_combo_bad", False))
            no_other_flags = not any([
                rule_neg_lt, rule_neg_hf, rule_jrun, iso_allowed, high_phase,
                combo_bad, near_thr, low_conf, abs_review, tail_instab, osc_jagged
            ])
            low_amp = bool(np.isfinite(maxabs_1k) and (maxabs_1k < LOWAMP_MAXABS_1K_MRAD))
            few_bigjumps = (big_jumps <= 2)

            if kk_ok and no_other_flags and low_amp and few_bigjumps:
                # If the model is extremely confident BAD, route to REVIEW; otherwise accept as GOOD.
                if (not np.isnan(pb)) and (pb >= MODEL_STRONG_BAD_PB):
                    final.append("review")
                else:
                    final.append("good")
                continue

        # 3) REVIEW triggers (include KK)
        review_triggers = (
            kk_review_like or
            rule_jrun or
            iso_allowed or
            rule_neg_hf or
            bool(getattr(r, "rule_high_phase_gt_1k", False)) or
            near_thr or
            low_conf or
            abs_review
        )

        if review_triggers:
            final.append("review")
        else:
            final.append(str(getattr(r, "pred_binary")))

    features["pred_final"] = final
    return features
