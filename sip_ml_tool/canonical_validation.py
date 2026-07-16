# Extracted from SIP_ML_Tool.ipynb main workflow cell.
# V2.6 canonical grouped validation, thresholds, and model patch.

# ==============================================================================
# V2.6 CANONICAL SIP-QC PATCH
# ------------------------------------------------------------------------------
# This block is the authoritative active path for SIP QC in this notebook.
# It intentionally:
#   1) trains on spectrum-derived features only (no provenance leakage),
#   2) validates with grouped splits,
#   3) uses conservative GOOD thresholds,
#   4) reserves BAD for structurally unstable / nonphysical spectra,
#   5) routes everything ambiguous to REVIEW.
# ==============================================================================

from sklearn.model_selection import GroupShuffleSplit


SIP_QC_METADATA_EXCLUDE = {
    "UniqueID",
    "true_label",
    "file_date",
    "source",
    "exp_group",
    "column_number",
    "cv_group",
    "qc_version",
    "pred_binary",
    "pred_final",
    "prob_good",
    "prob_bad",
    "prob_max",
    "good_threshold_used",
    "bad_threshold_used",
    "review_band_low",
    "review_band_high",
    "cautious_threshold",
    "rule_near_thr",
    "rule_low_conf",
    "rule_combo_bad",
    "rule_high_phase_gt_1k",
    "ref_high_phase",
    "kk_flag_review",
    "kk_flag_bad",
    "kk_status",
    "threshold_objective",
}

GOOD_THRESHOLD_PRECISION_TARGET = 0.97
GOOD_THRESHOLD_GRID = np.linspace(0.50, 0.995, 100)
BAD_THRESHOLD_MARGIN = 0.20
GROUP_SPLIT_TEST_SIZE = 0.20
GROUP_SPLIT_MAX_TRIES = 24


def _safe_text_key(v):
    if pd.isna(v):
        return "na"
    return str(v)


def _make_cv_group_from_features(features: pd.DataFrame) -> pd.Series:
    src = features["source"].astype(str) if "source" in features.columns else pd.Series(["field"] * len(features), index=features.index)
    exp = features["exp_group"].astype(str) if "exp_group" in features.columns else pd.Series(["none"] * len(features), index=features.index)
    col = features["column_number"].astype(str) if "column_number" in features.columns else pd.Series(["-1"] * len(features), index=features.index)
    date = features["file_date"].astype(str) if "file_date" in features.columns else pd.Series(["nodate"] * len(features), index=features.index)
    return src.fillna("field") + "|" + exp.fillna("none") + "|col=" + col.fillna("-1") + "|date=" + date.fillna("nodate")


def _get_ml_feature_columns(features: pd.DataFrame):
    cols = []
    for c in features.columns:
        if c in SIP_QC_METADATA_EXCLUDE:
            continue
        if pd.api.types.is_bool_dtype(features[c]) or pd.api.types.is_numeric_dtype(features[c]):
            cols.append(c)
    return cols


def _prepare_ml_matrix(features: pd.DataFrame, feature_names=None) -> pd.DataFrame:
    if feature_names is None:
        feature_names = _get_ml_feature_columns(features)
    X = features[feature_names].copy()
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(int)
    return X


def _grouped_train_val_split(train_df: pd.DataFrame, group_col: str = "cv_group", test_size: float = GROUP_SPLIT_TEST_SIZE):
    groups = train_df[group_col].fillna("ungrouped").astype(str).values
    y = train_df["true_label"].astype(str).values

    best = None
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Grouped validation requires at least 2 distinct cv_group values.")

    for seed in range(GROUP_SPLIT_MAX_TRIES):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42 + seed)
        train_idx, val_idx = next(gss.split(train_df, y=y, groups=groups))
        y_train = y[train_idx]
        y_val = y[val_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue
        best = (train_idx, val_idx)
        break

    if best is None:
        raise ValueError(
            "Unable to create a grouped train/validation split containing both classes. "
            "Check cv_group granularity or labeled data balance."
        )
    return best


def _binary_metrics_from_probs(y_true, p_good, threshold):
    y_true = np.asarray(y_true).astype(str)
    p_good = np.asarray(p_good, dtype=float)
    pred = np.where(p_good >= threshold, "good", "bad")

    tp_good = int(np.sum((pred == "good") & (y_true == "good")))
    fp_good = int(np.sum((pred == "good") & (y_true == "bad")))
    fn_good = int(np.sum((pred == "bad") & (y_true == "good")))
    tn_good = int(np.sum((pred == "bad") & (y_true == "bad")))

    precision_good = tp_good / max(tp_good + fp_good, 1)
    recall_good = tp_good / max(tp_good + fn_good, 1)
    recall_bad = tn_good / max(tn_good + fp_good, 1)
    bad_escape_rate = fp_good / max(np.sum(y_true == "bad"), 1)
    accuracy = (tp_good + tn_good) / max(len(y_true), 1)
    balanced_accuracy = 0.5 * (recall_good + recall_bad)

    return {
        "threshold": float(threshold),
        "precision_good": float(precision_good),
        "recall_good": float(recall_good),
        "recall_bad": float(recall_bad),
        "bad_escape_rate": float(bad_escape_rate),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "tp_good": tp_good,
        "fp_good": fp_good,
        "fn_good": fn_good,
        "tn_good": tn_good,
    }


def _pick_good_threshold(y_true, p_good, precision_target: float = GOOD_THRESHOLD_PRECISION_TARGET):
    rows = [_binary_metrics_from_probs(y_true, p_good, t) for t in GOOD_THRESHOLD_GRID]
    thr_df = pd.DataFrame(rows)

    meets = thr_df[thr_df["precision_good"] >= precision_target].copy()
    if not meets.empty:
        meets = meets.sort_values(
            ["bad_escape_rate", "recall_good", "balanced_accuracy", "threshold"],
            ascending=[True, False, False, True],
        )
        chosen = meets.iloc[0].to_dict()
        chosen["selection_mode"] = f"precision>={precision_target:.2f}"
    else:
        fallback = thr_df.sort_values(
            ["precision_good", "bad_escape_rate", "recall_good", "balanced_accuracy", "threshold"],
            ascending=[False, True, False, False, True],
        ).iloc[0].to_dict()
        chosen = dict(fallback)
        chosen["selection_mode"] = "best_available_precision"

    chosen["bad_threshold"] = float(max(0.05, chosen["threshold"] - BAD_THRESHOLD_MARGIN))
    chosen["review_band_low"] = float(max(0.0, chosen["threshold"] - NEAR_THRESHOLD_BAND))
    chosen["review_band_high"] = float(min(1.0, chosen["threshold"] + NEAR_THRESHOLD_BAND))
    return chosen, thr_df


def build_feature_table(raw: pd.DataFrame):
    """
    Canonical V2.6 feature builder.
    Keeps the original per-spectrum engineered features, adds shape-QC metrics,
    replaces brittle negative-phase rules with tolerant versions, and adds a
    grouped-validation key.
    """
    features = _orig_build_feature_table(raw)

    extra_rows = []
    for uid, g in raw.groupby("UniqueID", sort=False):
        m = _compute_shape_qc_metrics_from_group(g)
        m["UniqueID"] = uid
        extra_rows.append(m)
    extra = pd.DataFrame(extra_rows)

    if not extra.empty:
        features = features.merge(extra, on="UniqueID", how="left")

    for c in features.columns:
        if c.startswith("rule_") and pd.api.types.is_numeric_dtype(features[c]):
            features[c] = features[c].fillna(0)

    if "rule_neg_lt_1k_tolerant" in features.columns:
        features["rule_neg_lt_1k"] = features["rule_neg_lt_1k_tolerant"].fillna(False).astype(int)
    if "rule_neg_1k_10k_tolerant" in features.columns:
        features["rule_neg_1k_10k"] = features["rule_neg_1k_10k_tolerant"].fillna(False).astype(int)

    features["cv_group"] = _make_cv_group_from_features(features)
    features["qc_version"] = "V2.6_canonical"

    return features


def train_catboost(features: pd.DataFrame):
    """
    Train the SIP QC classifier on spectrum-derived features only.
    Provenance-like columns are excluded to avoid leakage.
    Validation is grouped so the score better reflects generalization to new runs.
    """
    train_df = features.dropna(subset=["true_label"]).copy()
    train_df["true_label"] = train_df["true_label"].astype(str)

    print("\n[INFO] Training samples:", len(train_df))
    print(train_df["true_label"].value_counts(dropna=False))

    if "cv_group" not in train_df.columns:
        train_df["cv_group"] = _make_cv_group_from_features(train_df)

    feature_names = _get_ml_feature_columns(train_df)
    if not feature_names:
        raise ValueError("No numeric spectrum-derived ML features were found after leakage exclusion.")

    excluded_now = sorted([c for c in train_df.columns if c not in feature_names])
    print("\n[INFO] Excluding non-ML / provenance columns from training:")
    print(excluded_now)

    X = _prepare_ml_matrix(train_df, feature_names)
    y = train_df["true_label"].astype(str)

    train_idx, val_idx = _grouped_train_val_split(train_df, group_col="cv_group", test_size=GROUP_SPLIT_TEST_SIZE)

    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_val = y.iloc[val_idx].copy()

    print(f"\n[INFO] Grouped split: train={len(X_train)} | val={len(X_val)} | "
          f"unique groups train={train_df.iloc[train_idx]['cv_group'].nunique()} | "
          f"val={train_df.iloc[val_idx]['cv_group'].nunique()}")

    class_names = ["bad", "good"]
    n_bad = max(int(np.sum(y_train == "bad")), 1)
    n_good = max(int(np.sum(y_train == "good")), 1)

    # Favor protecting the GOOD bucket from bad spectra by upweighting BAD.
    bad_weight = max(1.0, len(y_train) / (2.0 * n_bad))
    good_weight = 1.0

    train_pool = Pool(X_train, y_train, feature_names=feature_names)
    val_pool = Pool(X_val, y_val, feature_names=feature_names)

    param_grid = [
        {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 5, "iterations": 2500},
        {"depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 5, "iterations": 2500},
        {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 4, "iterations": 2200},
        {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5, "iterations": 1800},
    ]

    best_model = None
    best_params = None
    best_threshold = None
    best_threshold_df = None
    best_score = -np.inf
    best_metrics = None

    for params in param_grid:
        print("\n[INFO] Trying params:", params)
        model_tmp = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            iterations=params["iterations"],
            l2_leaf_reg=params["l2_leaf_reg"],
            random_seed=42,
            verbose=False,
            class_names=class_names,
            class_weights=[bad_weight, good_weight],
        )
        model_tmp.fit(train_pool, eval_set=val_pool)

        prob_val = model_tmp.predict_proba(val_pool)
        classes_model = list(model_tmp.classes_)
        idx_good = classes_model.index("good")
        p_good_val = prob_val[:, idx_good]

        threshold_info, threshold_df = _pick_good_threshold(y_val.values, p_good_val)
        metrics = _binary_metrics_from_probs(y_val.values, p_good_val, threshold_info["threshold"])

        # Conservative SIP-QC objective: minimize BAD escaping into GOOD,
        # then recover as many genuine GOOD spectra as safely possible.
        score = (
            3.0 * metrics["precision_good"]
            - 2.5 * metrics["bad_escape_rate"]
            + 0.5 * metrics["recall_good"]
            + 0.25 * metrics["balanced_accuracy"]
        )

        print("[INFO] Validation metrics @ chosen threshold:")
        print({
            "threshold": round(metrics["threshold"], 4),
            "precision_good": round(metrics["precision_good"], 4),
            "recall_good": round(metrics["recall_good"], 4),
            "recall_bad": round(metrics["recall_bad"], 4),
            "bad_escape_rate": round(metrics["bad_escape_rate"], 4),
            "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
            "selection_mode": threshold_info["selection_mode"],
        })

        if score > best_score:
            best_score = score
            best_model = model_tmp
            best_params = params
            best_threshold = dict(threshold_info)
            best_threshold_df = threshold_df.copy()
            best_metrics = dict(metrics)

    print("\n[INFO] Best grouped-validation score:", round(best_score, 4))
    print("[INFO] Best params:", best_params)
    print("[INFO] Best threshold package:", {
        "good_threshold": round(best_threshold["threshold"], 4),
        "bad_threshold": round(best_threshold["bad_threshold"], 4),
        "selection_mode": best_threshold["selection_mode"],
    })

    full_X = _prepare_ml_matrix(train_df, feature_names)
    full_y = train_df["true_label"].astype(str)
    full_pool = Pool(full_X, full_y, feature_names=feature_names)

    final_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        depth=best_params["depth"],
        learning_rate=best_params["learning_rate"],
        iterations=best_params["iterations"],
        l2_leaf_reg=best_params["l2_leaf_reg"],
        random_seed=42,
        verbose=False,
        class_names=["bad", "good"],
        class_weights=[max(1.0, len(full_y) / (2.0 * max(int(np.sum(full_y == "bad")), 1))), 1.0],
    )
    final_model.fit(full_pool)

    threshold_payload = {
        "good_threshold": float(best_threshold["threshold"]),
        "bad_threshold": float(best_threshold["bad_threshold"]),
        "review_band_low": float(best_threshold["review_band_low"]),
        "review_band_high": float(best_threshold["review_band_high"]),
        "selection_mode": best_threshold["selection_mode"],
        "precision_target": float(GOOD_THRESHOLD_PRECISION_TARGET),
        "threshold_objective": "protect_good_bucket_from_bad_escape",
        "validation_metrics": best_metrics,
        "threshold_grid_table": best_threshold_df,
    }

    return final_model, feature_names, [], threshold_payload


def apply_catboost(model, features: pd.DataFrame, feature_names, cat_features_idx, threshold_payload):
    X_all = _prepare_ml_matrix(features, feature_names)
    pool_all = Pool(X_all, feature_names=feature_names)

    prob_all = model.predict_proba(pool_all)
    classes_model = list(model.classes_)
    idx_good = classes_model.index("good")
    idx_bad = classes_model.index("bad")

    p_good_all = prob_all[:, idx_good]
    p_bad_all = prob_all[:, idx_bad]
    maxp_all = prob_all.max(axis=1)

    if isinstance(threshold_payload, dict):
        good_t = float(threshold_payload.get("good_threshold", 0.5))
        bad_t = float(threshold_payload.get("bad_threshold", max(0.05, good_t - BAD_THRESHOLD_MARGIN)))
        review_low = float(threshold_payload.get("review_band_low", max(0.0, good_t - NEAR_THRESHOLD_BAND)))
        review_high = float(threshold_payload.get("review_band_high", min(1.0, good_t + NEAR_THRESHOLD_BAND)))
        threshold_objective = str(threshold_payload.get("threshold_objective", "protect_good_bucket_from_bad_escape"))
    else:
        good_t = float(threshold_payload)
        bad_t = float(max(0.05, good_t - BAD_THRESHOLD_MARGIN))
        review_low = float(max(0.0, good_t - NEAR_THRESHOLD_BAND))
        review_high = float(min(1.0, good_t + NEAR_THRESHOLD_BAND))
        threshold_objective = "protect_good_bucket_from_bad_escape"

    pred_binary = np.where(p_good_all >= good_t, "good", "bad")

    out = features.copy()
    out["pred_binary"] = pred_binary
    out["prob_good"] = p_good_all
    out["prob_bad"] = p_bad_all
    out["prob_max"] = maxp_all
    out["good_threshold_used"] = float(good_t)
    out["bad_threshold_used"] = float(bad_t)
    out["review_band_low"] = float(review_low)
    out["review_band_high"] = float(review_high)
    out["cautious_threshold"] = float(CAUTIOUS_THRESHOLD)
    out["threshold_objective"] = threshold_objective
    out["rule_near_thr"] = ((out["prob_good"] >= review_low) & (out["prob_good"] <= review_high)).astype(int)
    out["rule_low_conf"] = (out["prob_max"] < CAUTIOUS_THRESHOLD).astype(int)

    return out


def apply_override(features: pd.DataFrame) -> pd.DataFrame:
    """
    Canonical V2.6 triage for SIP QC.
    GOOD = clean + confident + physically plausible
    BAD  = structurally unstable / nonphysical
    REVIEW = everything ambiguous or borderline
    """
    out = features.copy()

    for c, default in [
        ("rule_neg_lt_1k", 0),
        ("rule_neg_1k_10k", 0),
        ("rule_jagged_run", 0),
        ("rule_big_jump_count", 0),
        ("rule_isolated_spikes", 0),
        ("rule_isolated_spikes_allowed", 0),
        ("rule_oscillatory_jagged", 0),
        ("rule_tail_instability", 0),
        ("rule_smooth_candidate", 0),
        ("max_phase_gt_1k_mrad", np.nan),
        ("max_abs_phase_mrad", np.nan),
        ("max_abs_phase_gt_1k_mrad", np.nan),
        ("tail_spike_ratio", np.nan),
        ("tail_flip_count", 0),
        ("tail_resid_rms_frac", np.nan),
        ("kk_pseudo_chi2", np.nan),
        ("kk_mu", np.nan),
        ("kk_resid_rms", np.nan),
        ("prob_good", np.nan),
        ("prob_bad", np.nan),
        ("prob_max", np.nan),
        ("pred_binary", None),
    ]:
        if c not in out.columns:
            out[c] = default

    # Reference curve for high-phase at >1 kHz, calibrated from labeled GOOD when possible.
    group_cols = [c for c in HIGH_PHASE_GROUP_COLS if c in out.columns]
    if group_cols and "max_phase_gt_1k_mrad" in out.columns:
        ref_source = out[out.get("true_label", "").astype(str) == "good"].copy()
        if ref_source["max_phase_gt_1k_mrad"].dropna().empty:
            ref_source = out.copy()
        ref = (
            ref_source.dropna(subset=["max_phase_gt_1k_mrad"])
            .groupby(group_cols)["max_phase_gt_1k_mrad"]
            .quantile(HIGH_PHASE_REF_Q)
            .reset_index()
            .rename(columns={"max_phase_gt_1k_mrad": "ref_high_phase"})
        )
        out = out.drop(columns=["ref_high_phase"], errors="ignore").merge(ref, on=group_cols, how="left")
    else:
        out["ref_high_phase"] = np.nan

    out["rule_high_phase_gt_1k"] = (
        out["max_phase_gt_1k_mrad"] > (HIGH_PHASE_MULT * out["ref_high_phase"])
    ).fillna(False).astype(int)
    out["rule_combo_bad"] = ((out["rule_jagged_run"] == 1) & (out["rule_neg_1k_10k"] == 1)).astype(int)

    # KK gate, calibrated primarily from labeled GOOD.
    out["kk_flag_review"] = False
    out["kk_flag_bad"] = False
    out["kk_status"] = "KK_NA"

    if USE_KK_GATE and all(c in out.columns for c in ["kk_pseudo_chi2", "kk_mu", "kk_resid_rms"]):
        good_ref = out[(out.get("true_label", "").astype(str) == "good")].dropna(subset=["kk_pseudo_chi2", "kk_mu", "kk_resid_rms"])
        use_ref = good_ref if len(good_ref) >= KK_MIN_GOOD_FOR_CAL else out.dropna(subset=["kk_pseudo_chi2", "kk_mu", "kk_resid_rms"])

        if not use_ref.empty:
            chi_review = float(use_ref["kk_pseudo_chi2"].quantile(KK_Q_REVIEW))
            rms_review = float(use_ref["kk_resid_rms"].quantile(KK_Q_REVIEW))
            mu_review = float(use_ref["kk_mu"].quantile(KK_MU_Q_REVIEW))

            chi_bad = float(use_ref["kk_pseudo_chi2"].quantile(KK_Q_BAD))
            rms_bad = float(use_ref["kk_resid_rms"].quantile(KK_Q_BAD))
            mu_bad = float(use_ref["kk_mu"].quantile(KK_MU_Q_BAD))

            out["kk_thr_chi_review"] = chi_review
            out["kk_thr_rms_review"] = rms_review
            out["kk_thr_mu_review"] = mu_review
            out["kk_thr_chi_bad"] = chi_bad
            out["kk_thr_rms_bad"] = rms_bad
            out["kk_thr_mu_bad"] = mu_bad

            chi = out["kk_pseudo_chi2"]
            rms = out["kk_resid_rms"]
            mu = out["kk_mu"]

            kk_bad = (((chi > chi_bad) & (rms > rms_bad)) | (mu < mu_bad)).fillna(False)
            kk_review = (((chi > chi_review) | (rms > rms_review) | (mu < mu_review)) & (~kk_bad)).fillna(False)

            out["kk_flag_bad"] = kk_bad
            out["kk_flag_review"] = kk_review
            out["kk_status"] = np.where(
                out["kk_flag_bad"], "KK_BAD",
                np.where(out["kk_flag_review"], "KK_REVIEW", "KK_OK")
            )

            ref_name = "labeled GOOD" if len(good_ref) >= KK_MIN_GOOD_FOR_CAL else "ALL rows (fallback)"
            print(f"[INFO] KK gate calibrated from {ref_name}: "
                  f"REVIEW chi>{chi_review:.3g} rms>{rms_review:.3g} mu<{mu_review:.3g} | "
                  f"BAD chi>{chi_bad:.3g} rms>{rms_bad:.3g} mu<{mu_bad:.3g}")

    final = []
    reasons = []

    good_t = float(out["good_threshold_used"].dropna().iloc[0]) if "good_threshold_used" in out.columns and out["good_threshold_used"].notna().any() else 0.5
    cautious_t = float(out["cautious_threshold"].dropna().iloc[0]) if "cautious_threshold" in out.columns and out["cautious_threshold"].notna().any() else CAUTIOUS_THRESHOLD

    for r in out.itertuples(index=False):
        pg = float(getattr(r, "prob_good", np.nan))
        pb = float(getattr(r, "prob_bad", np.nan))
        pred_binary = str(getattr(r, "pred_binary", "bad"))
        kk_status = str(getattr(r, "kk_status", "KK_NA"))

        rule_neg_lt = bool(getattr(r, "rule_neg_lt_1k", 0))
        rule_neg_hf = bool(getattr(r, "rule_neg_1k_10k", 0))
        rule_jrun = bool(getattr(r, "rule_jagged_run", 0))
        rule_combo = bool(getattr(r, "rule_combo_bad", 0))
        rule_iso_allowed = bool(getattr(r, "rule_isolated_spikes_allowed", 0))
        rule_high_phase = bool(getattr(r, "rule_high_phase_gt_1k", 0))
        rule_near_thr = bool(getattr(r, "rule_near_thr", 0))
        rule_low_conf = bool(getattr(r, "rule_low_conf", 0))
        rule_osc = bool(getattr(r, "rule_oscillatory_jagged", 0))
        rule_tail = bool(getattr(r, "rule_tail_instability", 0))
        smooth_candidate = bool(getattr(r, "rule_smooth_candidate", 0))

        big_jumps = int(getattr(r, "rule_big_jump_count", 0) or 0)
        tail_flip_count = int(getattr(r, "tail_flip_count", 0) or 0)
        tail_spike_ratio = float(getattr(r, "tail_spike_ratio", np.nan))
        tail_rms_frac = float(getattr(r, "tail_resid_rms_frac", np.nan))
        maxabs_1k = float(getattr(r, "max_abs_phase_gt_1k_mrad", np.nan))
        abs_review = bool(np.isfinite(maxabs_1k) and (maxabs_1k >= PHASE_ABS_REVIEW_MRAD))
        abs_extreme = bool(np.isfinite(maxabs_1k) and (maxabs_1k >= PHASE_ABS_BAD_MRAD))

        kk_bad = (kk_status == "KK_BAD")
        kk_review = (kk_status == "KK_REVIEW")
        kk_ok = (kk_status == "KK_OK")

        severe_tail = bool(
            rule_tail and (
                rule_osc or
                big_jumps >= 4 or
                tail_flip_count >= 4 or
                (np.isfinite(tail_spike_ratio) and tail_spike_ratio >= 8.0) or
                (np.isfinite(tail_rms_frac) and tail_rms_frac >= 0.10)
            )
        )

        structural_bad = any([
            rule_neg_lt,
            rule_combo,
            abs_extreme,
            severe_tail,
            kk_bad and (rule_tail or rule_osc or rule_jrun),
        ])

        structural_warning = any([
            structural_bad,
            rule_neg_hf,
            rule_jrun,
            rule_iso_allowed,
            rule_high_phase,
            rule_near_thr,
            rule_low_conf,
            rule_osc,
            rule_tail,
            kk_review,
            abs_review,
        ])

        clean_good_candidate = all([
            pred_binary == "good",
            kk_ok or kk_status == "KK_NA",
            not structural_warning,
            np.isfinite(pg) and (pg >= good_t),
            np.isfinite(getattr(r, "prob_max", np.nan)) and (float(getattr(r, "prob_max", np.nan)) >= cautious_t),
        ])

        if structural_bad:
            final.append("bad")
            if rule_neg_lt:
                reasons.append("nonphysical_negative_phase_lt_1k")
            elif rule_combo:
                reasons.append("jagged_plus_negative_1k_10k")
            elif abs_extreme:
                reasons.append("extreme_high_frequency_phase")
            elif severe_tail:
                reasons.append("severe_tail_instability")
            else:
                reasons.append("kk_bad_plus_structural_instability")
            continue

        if clean_good_candidate:
            final.append("good")
            reasons.append("clean_confident_physically_plausible")
            continue

        # Conservative SIP QC rule:
        # any ambiguity stays in REVIEW, including model-only BAD with no structural failure.
        if pred_binary == "good":
            if rule_near_thr:
                final.append("review"); reasons.append("good_near_threshold")
            elif rule_low_conf:
                final.append("review"); reasons.append("good_low_confidence")
            elif kk_review:
                final.append("review"); reasons.append("good_but_kk_review")
            elif structural_warning:
                final.append("review"); reasons.append("good_with_shape_or_phase_warning")
            else:
                final.append("review"); reasons.append("good_not_clean_enough")
        else:
            if kk_ok and smooth_candidate and (not structural_warning):
                final.append("review"); reasons.append("model_only_bad_softened_to_review")
            elif kk_review or structural_warning:
                final.append("review"); reasons.append("bad_with_moderate_warning_review")
            else:
                final.append("review"); reasons.append("uncertain_bad_review")

    out["pred_final"] = final
    out["rule_patch_v26_reason"] = reasons
    out["rule_patch_v26_applied"] = 1
    out["rule_patch_v3_reason"] = out["rule_patch_v26_reason"]
    out["rule_patch_v3_applied"] = out["rule_patch_v26_applied"]
    out["rule_patch_v2_reason"] = out["rule_patch_v26_reason"]
    out["rule_patch_v2_applied"] = out["rule_patch_v26_applied"]

    return out
