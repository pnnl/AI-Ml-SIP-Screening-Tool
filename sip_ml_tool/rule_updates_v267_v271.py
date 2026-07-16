# Extracted from SIP_ML_Tool.ipynb main workflow cell.
# V2.6.7-V2.7.1 final base workflow update stack.

# ==============================================================================
# V2.6.7 - Make BAD severity authoritative for weak KK_OK neg<1k cases
# ------------------------------------------------------------------------------
# Motivation:
# - Prior versions computed a useful BAD severity score, but the old binary
#   neg<1k BAD family could still dominate the final label.
# - This patch gives the severity framework explicit authority over the weakest
#   remaining KK_OK sub-1 kHz negative cases.

if "_v267_prev_apply_override" not in globals():
    _v267_prev_apply_override = apply_override

V267_LOW_SEVERITY_MAX = 3.5
V267_MODERATE_SEVERITY_MAX = 4.5
V267_NEGCOUNT_MAX_LOW = 3
V267_NEGRUN_MAX_LOW = 2
V267_NEGCOUNT_MAX_MOD = 4
V267_NEGRUN_MAX_MOD = 3
V267_MAXABS_1K_LOW = 220.0
V267_MAXABS_1K_MOD = 260.0
V267_PHYS_MAX_LOW = 4.0
V267_PHYS_MAX_MOD = 5.0
V267_COMPANION_MAX_LOW = 1
V267_COMPANION_MAX_MOD = 2
V267_PG_MIN_MOD = 0.45


def apply_override(features: pd.DataFrame) -> pd.DataFrame:
    out = _v267_prev_apply_override(features.copy())

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
        "deep_neg_lt1k": False,
        "neg_lt1k_count": 0,
        "neg_lt1k_run": 0,
        "max_abs_phase_gt_1k_mrad": np.nan,
        "physics_inadmissibility_score": np.nan,
        "physics_issue_count": np.nan,
        "prob_good": np.nan,
        "bad_severity_score": np.nan,
        "bad_severity_label": "",
        "bad_internal_subtype": "",
        "bad_gate_v266_companion_count": np.nan,
        "tail_flip_count": np.nan,
        "tail_spike_ratio": np.nan,
        "tail_resid_rms_frac": np.nan,
    }
    for c, default in defaults.items():
        if c not in out.columns:
            out[c] = default

    still_bad = out["pred_final"].astype(str).eq("bad")
    kk_ok = out["kk_status"].astype(str).eq("KK_OK")
    neg_reason_family = (
        out["rule_patch_v26_reason"].astype(str).str.contains("negative_phase_lt_1k", regex=False)
        | out["rule_patch_v26_reason"].astype(str).str.contains("neg_lt1k", regex=False)
        | out["rule_neg_lt_1k"].fillna(0).astype(int).eq(1)
    )

    midband_neg = out["rule_neg_1k_10k"].fillna(0).astype(int).eq(1)
    combo_bad = out["rule_combo_bad"].fillna(0).astype(int).eq(1)
    jagged = out["rule_jagged_run"].fillna(0).astype(int).eq(1)
    osc = out["rule_oscillatory_jagged"].fillna(0).astype(int).eq(1)
    tail = out["rule_tail_instability"].fillna(0).astype(int).eq(1)
    high_phase = out["rule_high_phase_gt_1k"].fillna(0).astype(int).eq(1)
    iso_spikes = out["rule_isolated_spikes_allowed"].fillna(0).astype(int).eq(1)
    deep_neg = out["deep_neg_lt1k"].fillna(False).astype(bool)

    companion_count = pd.to_numeric(out["bad_gate_v266_companion_count"], errors="coerce")
    if companion_count.isna().all():
        companion_count = (
            midband_neg.astype(int)
            + combo_bad.astype(int)
            + jagged.astype(int)
            + osc.astype(int)
            + tail.astype(int)
            + high_phase.astype(int)
            + iso_spikes.astype(int)
        ).astype(float)
    else:
        companion_count = companion_count.fillna(0.0).astype(float)

    neg_count = pd.to_numeric(out["neg_lt1k_count"], errors="coerce").fillna(0.0).astype(float)
    neg_run = pd.to_numeric(out["neg_lt1k_run"], errors="coerce").fillna(0.0).astype(float)
    maxabs_1k = pd.to_numeric(out["max_abs_phase_gt_1k_mrad"], errors="coerce").fillna(np.inf).astype(float)
    phys_score = pd.to_numeric(out["physics_inadmissibility_score"], errors="coerce").fillna(999.0).astype(float)
    prob_good = pd.to_numeric(out["prob_good"], errors="coerce").fillna(0.0).astype(float)
    bad_sev = pd.to_numeric(out["bad_severity_score"], errors="coerce").fillna(999.0).astype(float)
    bad_label = out["bad_severity_label"].astype(str)
    subtype = out["bad_internal_subtype"].astype(str)
    tail_flips = pd.to_numeric(out["tail_flip_count"], errors="coerce").fillna(0.0).astype(float)
    tail_spike = pd.to_numeric(out["tail_spike_ratio"], errors="coerce")
    tail_rms = pd.to_numeric(out["tail_resid_rms_frac"], errors="coerce")

    tail_not_severe = (
        tail_flips.le(6)
        & (tail_spike.isna() | tail_spike.le(8.0))
        & (tail_rms.isna() | tail_rms.le(0.10))
    )

    authoritative_low = (
        still_bad
        & kk_ok
        & neg_reason_family
        & (~deep_neg)
        & (~midband_neg)
        & (~combo_bad)
        & tail_not_severe
        & companion_count.le(V267_COMPANION_MAX_LOW)
        & neg_count.le(V267_NEGCOUNT_MAX_LOW)
        & neg_run.le(V267_NEGRUN_MAX_LOW)
        & maxabs_1k.le(V267_MAXABS_1K_LOW)
        & phys_score.le(V267_PHYS_MAX_LOW)
        & bad_sev.le(V267_LOW_SEVERITY_MAX)
        & bad_label.eq("low")
        & subtype.ne("bad_nonphysical")
    )

    authoritative_moderate = (
        still_bad
        & kk_ok
        & neg_reason_family
        & (~deep_neg)
        & (~combo_bad)
        & tail_not_severe
        & companion_count.le(V267_COMPANION_MAX_MOD)
        & neg_count.le(V267_NEGCOUNT_MAX_MOD)
        & neg_run.le(V267_NEGRUN_MAX_MOD)
        & maxabs_1k.le(V267_MAXABS_1K_MOD)
        & phys_score.le(V267_PHYS_MAX_MOD)
        & bad_sev.le(V267_MODERATE_SEVERITY_MAX)
        & bad_label.isin(["low", "moderate"])
        & subtype.isin(["bad_artifact", "bad_low_information"])
        & prob_good.ge(V267_PG_MIN_MOD)
    )

    rescue_review = authoritative_low | authoritative_moderate
    if rescue_review.any():
        out.loc[rescue_review, "pred_final"] = "review"
        out.loc[authoritative_low, "rule_patch_v26_reason"] = "v267_authoritative_low_severity_neglt1k_to_review"
        out.loc[authoritative_moderate & ~authoritative_low, "rule_patch_v26_reason"] = "v267_authoritative_moderate_neglt1k_to_review"
        out.loc[rescue_review, "rule_patch_v26_applied"] = 1
        out.loc[rescue_review, "rule_patch_v3_reason"] = out.loc[rescue_review, "rule_patch_v26_reason"]
        out.loc[rescue_review, "rule_patch_v3_applied"] = 1
        out.loc[rescue_review, "rule_patch_v2_reason"] = out.loc[rescue_review, "rule_patch_v26_reason"]
        out.loc[rescue_review, "rule_patch_v2_applied"] = 1

    print(
        f"[INFO] V2.6.7 authoritative BAD gate | rescued BAD->REVIEW={int(rescue_review.sum())} "
        f"| remaining BAD={int((out['pred_final'].astype(str) == 'bad').sum())}"
    )
    out["qc_version"] = "V2.6.7_authoritative_badgate"
    return out



# ==============================================================================
# PATCH V2.7.1 — Delivery refinement: evidence-based BAD gate, stronger SIP features,
# grouped CV summaries, and compact run report
# ==============================================================================

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if "_v271_prev_build_feature_table" not in globals():
    _v271_prev_build_feature_table = build_feature_table
if "_v271_prev_train_catboost" not in globals():
    _v271_prev_train_catboost = train_catboost
if "_v271_prev_apply_override" not in globals():
    _v271_prev_apply_override = apply_override
if "_v271_prev_main" not in globals():
    _v271_prev_main = main

V271_MAX_CV_SPLITS = 5
V271_MIN_VALID_FOLDS = 2
V271_PARAM_GRID = [
    {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 5, "iterations": 2200},
    {"depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 5, "iterations": 2200},
    {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 4, "iterations": 1800},
]

V271_LOW_BAD_SCORE_MAX = 2.8
V271_MOD_BAD_SCORE_MAX = 5.0
V271_RESCUE_PROB_GOOD_MOD = 0.18
V271_REVIEW_HIGH_PG = 0.85

def _v271_safe_float(v, default=np.nan):
    try:
        if pd.isna(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)

def _v271_max_run(mask) -> int:
    best = 0
    cur = 0
    for val in np.asarray(mask, dtype=bool):
        if val:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)

def _v271_negative_area_logf(freq_hz, phase_mrad, lo=None, hi=None) -> float:
    f = np.asarray(freq_hz, dtype=float)
    ph = np.asarray(phase_mrad, dtype=float)
    mask = np.isfinite(f) & np.isfinite(ph) & (f > 0)
    if lo is not None:
        mask &= (f >= float(lo))
    if hi is not None:
        mask &= (f <= float(hi))
    if np.sum(mask) < 2:
        return 0.0
    x = np.log10(f[mask])
    y = np.clip(-ph[mask], 0.0, None)
    if x.size < 2:
        return 0.0
    return float(np.trapezoid(y, x))

def _v271_curvature_metric(freq_hz, phase_mrad, lo=None, hi=None) -> float:
    f = np.asarray(freq_hz, dtype=float)
    ph = np.asarray(phase_mrad, dtype=float)
    mask = np.isfinite(f) & np.isfinite(ph) & (f > 0)
    if lo is not None:
        mask &= (f >= float(lo))
    if hi is not None:
        mask &= (f <= float(hi))
    if np.sum(mask) < 4:
        return np.nan
    x = np.log10(f[mask])
    y = ph[mask]
    dx = np.diff(x)
    if dx.size < 2 or np.any(dx == 0):
        return np.nan
    d1 = np.diff(y) / dx
    dx2 = np.diff(x[1:])
    if dx2.size == 0 or np.any(dx2 == 0):
        return np.nan
    d2 = np.diff(d1) / dx2
    return float(np.nanmean(np.abs(d2))) if d2.size else np.nan

def _v271_total_variation_norm(phase_mrad) -> float:
    ph = np.asarray(phase_mrad, dtype=float)
    ph = ph[np.isfinite(ph)]
    if ph.size < 2:
        return np.nan
    tv = float(np.sum(np.abs(np.diff(ph))))
    rng = max(float(np.nanmax(ph) - np.nanmin(ph)), 1.0)
    return float(tv / rng)

def _v271_augmented_cv_group(raw_group: pd.DataFrame) -> str:
    src = str(raw_group["source"].iloc[0]) if "source" in raw_group.columns else "field"
    exp = str(raw_group["exp_group"].iloc[0]) if "exp_group" in raw_group.columns else "none"
    col = str(raw_group["column_number_dash"].iloc[0]) if "column_number_dash" in raw_group.columns else "NA"
    if src.upper() == "HPE":
        return f"HPE|{exp}"
    return f"FIELD|{col}"

def build_feature_table(raw: pd.DataFrame):
    """
    V2.7.1 feature builder.
    Keeps the current canonical feature table, then augments it with a few
    higher-value SIP features that help split BAD into nonphysical vs artifact-like
    vs low-information cases.
    """
    features = _v271_prev_build_feature_table(raw)

    extra_rows = []
    for uid, g in raw.groupby("UniqueID", sort=False):
        g = g.sort_values("Freq")
        f = pd.to_numeric(g["Freq"], errors="coerce").to_numpy(dtype=float)
        ph_mrad = pd.to_numeric(g["Phase_mrad"], errors="coerce").to_numpy(dtype=float)

        low_mask = np.isfinite(f) & np.isfinite(ph_mrad) & (f > 0) & (f < 1000.0)
        mid_mask = np.isfinite(f) & np.isfinite(ph_mrad) & (f >= 1000.0) & (f <= 10000.0)

        neg_low = low_mask & (ph_mrad < 0.0)
        neg_mid = mid_mask & (ph_mrad < 0.0)

        extra_rows.append({
            "UniqueID": uid,
            "cv_group": _v271_augmented_cv_group(g),
            "neg_lt1k_area_logf": _v271_negative_area_logf(f, ph_mrad, lo=None, hi=1000.0),
            "neg_lt1k_frac": float(np.mean(neg_low)) if low_mask.any() else 0.0,
            "neg_lt1k_min_mrad": float(np.nanmin(ph_mrad[low_mask])) if low_mask.any() else np.nan,
            "neg_midband_frac": float(np.mean(neg_mid)) if mid_mask.any() else 0.0,
            "neg_midband_run": _v271_max_run(neg_mid),
            "curvature_low": _v271_curvature_metric(f, ph_mrad, lo=None, hi=10.0),
            "curvature_mid": _v271_curvature_metric(f, ph_mrad, lo=10.0, hi=1000.0),
            "curvature_high": _v271_curvature_metric(f, ph_mrad, lo=1000.0, hi=None),
            "phase_tv_norm": _v271_total_variation_norm(ph_mrad),
        })

    extra = pd.DataFrame(extra_rows)
    if not extra.empty:
        for col in [c for c in extra.columns if c != "UniqueID"]:
            if col in features.columns:
                features = features.drop(columns=[col])
        features = features.merge(extra, on="UniqueID", how="left")

    features["qc_version"] = "V2.7.1_delivery_refine"
    return features

def _v271_training_feature_columns(train_df: pd.DataFrame):
    exclude = set(SIP_QC_METADATA_EXCLUDE) | {
        "column_number_dash",
        "bad_internal_subtype",
        "bad_severity_label",
        "bad_evidence_label",
        "bad_evidence_subtype",
        "rule_patch_v26_reason",
        "rule_patch_v2_reason",
        "rule_patch_v3_reason",
        "pulled_from",
        "raw_path",
        "source_raw",
        "exp_group_raw",
    }
    cols = []
    for c in train_df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_bool_dtype(train_df[c]) or pd.api.types.is_numeric_dtype(train_df[c]):
            cols.append(c)
    return cols

def _v271_grouped_cv_splits(train_df: pd.DataFrame):
    groups = train_df["cv_group"].astype(str)
    n_groups = groups.nunique()
    if n_groups < 2:
        return []
    n_splits = min(V271_MAX_CV_SPLITS, n_groups)
    if n_splits < 2:
        return []
    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    X_dummy = np.zeros((len(train_df), 1))
    y = train_df["true_label"].astype(str).values
    for tr_idx, va_idx in gkf.split(X_dummy, y, groups):
        y_tr = pd.Series(y[tr_idx])
        y_va = pd.Series(y[va_idx])
        if y_tr.nunique() < 2 or y_va.nunique() < 2:
            continue
        splits.append((tr_idx, va_idx))
    return splits

def train_catboost(features: pd.DataFrame):
    """
    V2.7.1 training:
    - trains on spectrum-derived features only
    - uses grouped cross-validation summaries (by cv_group)
    - chooses threshold from grouped out-of-fold probabilities
    - writes fold metrics + threshold curve for auditability
    """
    train_df = features.dropna(subset=["true_label"]).copy()
    train_df["true_label"] = train_df["true_label"].astype(str)

    if "cv_group" not in train_df.columns:
        train_df["cv_group"] = _make_cv_group_from_features(train_df)

    print("\n[INFO] Training samples:", len(train_df))
    print(train_df["true_label"].value_counts(dropna=False))
    print(f"[INFO] Training groups: {train_df['cv_group'].nunique()}")

    feature_names = _v271_training_feature_columns(train_df)
    if not feature_names:
        raise ValueError("No numeric spectrum-derived ML features were found after leakage exclusion.")

    excluded_now = sorted([c for c in train_df.columns if c not in feature_names])
    print("\n[INFO] Excluding non-ML / provenance columns from training:")
    print(excluded_now)

    X = _prepare_ml_matrix(train_df, feature_names)
    y = train_df["true_label"].astype(str)
    groups = train_df["cv_group"].astype(str)

    splits = _v271_grouped_cv_splits(train_df)
    if len(splits) < V271_MIN_VALID_FOLDS:
        print("[WARN] Not enough valid grouped CV folds; falling back to prior grouped trainer.")
        return _v271_prev_train_catboost(features)

    class_names = ["bad", "good"]
    cv_rows = []
    best = None

    for params in V271_PARAM_GRID:
        oof_prob_good = pd.Series(np.nan, index=train_df.index, dtype=float)
        fold_metrics = []

        for fold_id, (tr_idx, va_idx) in enumerate(splits, start=1):
            X_tr = X.iloc[tr_idx]
            X_va = X.iloc[va_idx]
            y_tr = y.iloc[tr_idx]
            y_va = y.iloc[va_idx]

            n_bad = max(int(np.sum(y_tr == "bad")), 1)
            bad_weight = max(1.0, len(y_tr) / (2.0 * n_bad))

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
                class_weights=[bad_weight, 1.0],
            )
            train_pool = Pool(X_tr, y_tr, feature_names=feature_names)
            val_pool = Pool(X_va, y_va, feature_names=feature_names)
            model_tmp.fit(train_pool, eval_set=val_pool)

            prob_val = model_tmp.predict_proba(val_pool)
            classes_model = list(model_tmp.classes_)
            idx_good = classes_model.index("good")
            p_good = prob_val[:, idx_good]
            oof_prob_good.iloc[va_idx] = p_good

            default_pred = np.where(p_good >= 0.5, "good", "bad")
            fold_metrics.append({
                "param_set": str(params),
                "fold": fold_id,
                "n_train": len(tr_idx),
                "n_val": len(va_idx),
                "groups_train": int(groups.iloc[tr_idx].nunique()),
                "groups_val": int(groups.iloc[va_idx].nunique()),
                "accuracy_default_t05": float(accuracy_score(y_va, default_pred)),
                "precision_good_t05": float(precision_score(y_va, default_pred, pos_label="good", zero_division=0)),
                "recall_good_t05": float(recall_score(y_va, default_pred, pos_label="good", zero_division=0)),
                "f1_macro_t05": float(f1_score(y_va, default_pred, average="macro", zero_division=0)),
            })

        valid_mask = oof_prob_good.notna()
        if valid_mask.sum() == 0:
            continue

        chosen, thr_df = _pick_good_threshold(
            y.loc[valid_mask].astype(str).values,
            oof_prob_good.loc[valid_mask].astype(float).values,
        )

        param_cv = pd.DataFrame(fold_metrics)
        mean_macro_f1 = float(param_cv["f1_macro_t05"].mean()) if not param_cv.empty else -np.inf
        selection_score = float(chosen.get("precision_good", 0.0)) + float(chosen.get("balanced_accuracy", 0.0))

        cv_rows.extend(fold_metrics)
        if (best is None) or (selection_score > best["selection_score"]):
            best = {
                "params": params,
                "threshold": float(chosen["threshold"]),
                "threshold_info": chosen,
                "threshold_curve": thr_df.copy(),
                "oof_prob_good": oof_prob_good.copy(),
                "mean_macro_f1": mean_macro_f1,
                "selection_score": selection_score,
            }

    if best is None:
        print("[WARN] Grouped CV did not yield a valid configuration; falling back to prior grouped trainer.")
        return _v271_prev_train_catboost(features)

    cv_summary_df = pd.DataFrame(cv_rows)
    cv_summary_path = OUT_DIR / "sip_training_grouped_cv_summary.csv"
    cv_summary_df.to_csv(cv_summary_path, index=False)

    thr_df = best["threshold_curve"].copy()
    thr_path = OUT_DIR / "sip_training_threshold_curve_grouped_oof.csv"
    thr_df.to_csv(thr_path, index=False)

    summary_txt = OUT_DIR / "sip_training_grouped_cv_report.txt"
    with open(summary_txt, "w", encoding="utf-8") as fh:
        fh.write("SIP grouped CV training report\n")
        fh.write(f"Training rows: {len(train_df)}\n")
        fh.write(f"Training groups: {train_df['cv_group'].nunique()}\n")
        fh.write(f"Chosen params: {best['params']}\n")
        fh.write(f"Chosen GOOD threshold: {best['threshold']:.3f}\n")
        fh.write(f"Selection mode: {best['threshold_info'].get('selection_mode', 'na')}\n")
        fh.write(f"Grouped OOF precision_good: {best['threshold_info'].get('precision_good', np.nan):.3f}\n")
        fh.write(f"Grouped OOF recall_good: {best['threshold_info'].get('recall_good', np.nan):.3f}\n")
        fh.write(f"Grouped OOF balanced_accuracy: {best['threshold_info'].get('balanced_accuracy', np.nan):.3f}\n")
        fh.write(f"Grouped OOF bad_escape_rate: {best['threshold_info'].get('bad_escape_rate', np.nan):.3f}\n")

    print(f"[OK] Saved grouped CV summary: {cv_summary_path}")
    print(f"[OK] Saved grouped threshold curve: {thr_path}")
    print(f"[OK] Saved grouped CV report: {summary_txt}")

    full_X = X.copy()
    full_y = y.copy()
    n_bad_full = max(int(np.sum(full_y == "bad")), 1)
    bad_weight_full = max(1.0, len(full_y) / (2.0 * n_bad_full))

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        depth=best["params"]["depth"],
        learning_rate=best["params"]["learning_rate"],
        iterations=best["params"]["iterations"],
        l2_leaf_reg=best["params"]["l2_leaf_reg"],
        random_seed=42,
        verbose=False,
        class_names=class_names,
        class_weights=[bad_weight_full, 1.0],
    )
    full_pool = Pool(full_X, full_y, feature_names=feature_names)
    model.fit(full_pool)

    globals()["_V271_TRAINING_REPORT"] = {
        "cv_summary_path": str(cv_summary_path),
        "threshold_curve_path": str(thr_path),
        "report_txt_path": str(summary_txt),
        "chosen_params": best["params"],
        "chosen_threshold": float(best["threshold"]),
        "selection_mode": best["threshold_info"].get("selection_mode", "na"),
    }

    return model, feature_names, [], float(best["threshold"])

def _v271_bad_evidence_triplet_row(r):
    kk_status = str(getattr(r, "kk_status", "KK_NA"))
    prob_good = _v271_safe_float(getattr(r, "prob_good", np.nan))
    phys_score = _v271_safe_float(getattr(r, "physics_inadmissibility_score", np.nan))
    kk_chi2 = _v271_safe_float(getattr(r, "kk_pseudo_chi2", np.nan))
    kk_mu = _v271_safe_float(getattr(r, "kk_mu", np.nan))

    neg_area = max(0.0, _v271_safe_float(getattr(r, "neg_lt1k_area_logf", 0.0), default=0.0))
    neg_frac = max(0.0, _v271_safe_float(getattr(r, "neg_lt1k_frac", 0.0), default=0.0))
    neg_min = _v271_safe_float(getattr(r, "neg_lt1k_min_mrad", np.nan))
    neg_run = int(_v271_safe_float(getattr(r, "neg_lt1k_run", 0), default=0.0))
    neg_mid_frac = max(0.0, _v271_safe_float(getattr(r, "neg_midband_frac", 0.0), default=0.0))
    neg_mid_run = int(_v271_safe_float(getattr(r, "neg_midband_run", 0), default=0.0))

    curvature_low = _v271_safe_float(getattr(r, "curvature_low", np.nan))
    curvature_mid = _v271_safe_float(getattr(r, "curvature_mid", np.nan))
    curvature_high = _v271_safe_float(getattr(r, "curvature_high", np.nan))
    phase_tv_norm = _v271_safe_float(getattr(r, "phase_tv_norm", np.nan))

    combo_bad = int(_v271_safe_float(getattr(r, "rule_combo_bad", 0), default=0.0)) == 1
    midband_neg_flag = int(_v271_safe_float(getattr(r, "rule_neg_1k_10k", 0), default=0.0)) == 1
    jagged = int(_v271_safe_float(getattr(r, "rule_jagged_run", 0), default=0.0)) == 1
    osc = int(_v271_safe_float(getattr(r, "rule_oscillatory_jagged", 0), default=0.0)) == 1
    tail = int(_v271_safe_float(getattr(r, "rule_tail_instability", 0), default=0.0)) == 1
    high_phase = int(_v271_safe_float(getattr(r, "rule_high_phase_gt_1k", 0), default=0.0)) == 1
    iso_spikes = int(_v271_safe_float(getattr(r, "rule_isolated_spikes_allowed", 0), default=0.0)) == 1
    deep_neg = bool(getattr(r, "deep_neg_lt1k", False))

    score = 0.0

    # Low-frequency negative-phase evidence
    if neg_frac > 0:
        score += 0.6
    if neg_frac >= 0.15:
        score += 0.6
    if neg_frac >= 0.40:
        score += 0.8

    if neg_run >= 2:
        score += 0.6
    if neg_run >= 4:
        score += 0.8

    if np.isfinite(neg_min) and neg_min <= -5.0:
        score += 0.6
    if np.isfinite(neg_min) and neg_min <= -20.0:
        score += 0.8

    if neg_area >= 25.0:
        score += 0.5
    if neg_area >= 75.0:
        score += 0.7
    if neg_area >= 150.0:
        score += 0.8

    # Mid-band negative phase is more concerning
    if midband_neg_flag or (neg_mid_frac > 0.0):
        score += 1.0
    if neg_mid_run >= 2:
        score += 0.8

    # Structural roughness / artifact evidence
    if jagged:
        score += 0.6
    if osc:
        score += 0.8
    if tail:
        score += 0.8
    if high_phase:
        score += 0.6
    if iso_spikes:
        score += 0.3
    if combo_bad:
        score += 1.2

    # Curve-shape severity
    for val in (curvature_low, curvature_mid, curvature_high):
        if np.isfinite(val) and val >= 100.0:
            score += 0.3
        if np.isfinite(val) and val >= 250.0:
            score += 0.3
    if np.isfinite(phase_tv_norm) and phase_tv_norm >= 3.0:
        score += 0.4
    if np.isfinite(phase_tv_norm) and phase_tv_norm >= 5.0:
        score += 0.5

    # Physics / KK consistency
    if kk_status == "KK_REVIEW":
        score += 0.8
    elif kk_status == "KK_BAD":
        score += 1.8
    if np.isfinite(phys_score) and phys_score >= 4.0:
        score += 0.8
    if np.isfinite(phys_score) and phys_score >= 6.0:
        score += 1.0
    if np.isfinite(kk_chi2) and kk_chi2 >= 0.20:
        score += 0.4
    if np.isfinite(kk_chi2) and kk_chi2 >= 0.50:
        score += 0.5
    if np.isfinite(kk_mu) and kk_mu >= 0.45:
        score += 0.4

    # A strong GOOD disagreement slightly softens borderline spectra
    if np.isfinite(prob_good) and prob_good >= 0.95:
        score -= 0.4
    if np.isfinite(prob_good) and prob_good >= 0.99:
        score -= 0.3

    score = float(max(score, 0.0))

    if score <= V271_LOW_BAD_SCORE_MAX:
        label = "low"
    elif score <= V271_MOD_BAD_SCORE_MAX:
        label = "moderate"
    else:
        label = "high"

    if (
        combo_bad
        or (deep_neg and (midband_neg_flag or neg_mid_run >= 1))
        or kk_status == "KK_BAD"
        or score >= 6.0
    ):
        subtype = "bad_nonphysical"
    elif (
        (tail or high_phase or iso_spikes or osc or jagged)
        and not combo_bad
        and not deep_neg
        and not midband_neg_flag
        and neg_mid_run == 0
    ):
        subtype = "bad_artifact"
    else:
        subtype = "bad_low_information"

    return float(score), label, subtype

def apply_override(features: pd.DataFrame) -> pd.DataFrame:
    """
    V2.7.1 final adjudicator.
    Keeps the existing GOOD and REVIEW behavior largely intact, but makes the
    BAD side evidence-based and authoritative for weak/moderate cases.
    """
    out = _v271_prev_apply_override(features.copy())

    defaults = {
        "pred_final": None,
        "prob_good": np.nan,
        "kk_status": "KK_NA",
        "rule_patch_v26_reason": "",
        "rule_patch_v2_reason": "",
        "rule_patch_v3_reason": "",
        "rule_neg_1k_10k": 0,
        "rule_combo_bad": 0,
        "rule_jagged_run": 0,
        "rule_oscillatory_jagged": 0,
        "rule_tail_instability": 0,
        "rule_high_phase_gt_1k": 0,
        "rule_isolated_spikes_allowed": 0,
        "rule_smooth_candidate": 0,
        "deep_neg_lt1k": False,
        "good_manifold_far": False,
        "good_manifold_score": np.nan,
        "physics_inadmissibility_score": np.nan,
        "neg_lt1k_area_logf": 0.0,
        "neg_lt1k_frac": 0.0,
        "neg_lt1k_min_mrad": np.nan,
        "neg_midband_frac": 0.0,
        "neg_midband_run": 0,
        "curvature_low": np.nan,
        "curvature_mid": np.nan,
        "curvature_high": np.nan,
        "phase_tv_norm": np.nan,
    }
    for c, default in defaults.items():
        if c not in out.columns:
            out[c] = default

    trip = out.apply(_v271_bad_evidence_triplet_row, axis=1, result_type="expand")
    out["bad_evidence_score"] = pd.to_numeric(trip[0], errors="coerce").fillna(0.0)
    out["bad_evidence_label"] = trip[1].astype(str)
    out["bad_internal_subtype"] = trip[2].astype(str)

    still_bad = out["pred_final"].astype(str).eq("bad")
    kk_ok_like = out["kk_status"].astype(str).isin(["KK_OK", "KK_NA"])
    prob_good = pd.to_numeric(out["prob_good"], errors="coerce").fillna(0.0).astype(float)

    midband_neg = out["rule_neg_1k_10k"].fillna(0).astype(int).eq(1) | (pd.to_numeric(out["neg_midband_frac"], errors="coerce").fillna(0.0) > 0)
    combo_bad = out["rule_combo_bad"].fillna(0).astype(int).eq(1)
    jagged = out["rule_jagged_run"].fillna(0).astype(int).eq(1)
    osc = out["rule_oscillatory_jagged"].fillna(0).astype(int).eq(1)
    tail = out["rule_tail_instability"].fillna(0).astype(int).eq(1)
    high_phase = out["rule_high_phase_gt_1k"].fillna(0).astype(int).eq(1)
    iso_spikes = out["rule_isolated_spikes_allowed"].fillna(0).astype(int).eq(1)
    deep_neg = out["deep_neg_lt1k"].fillna(False).astype(bool)
    smooth_candidate = out["rule_smooth_candidate"].fillna(0).astype(int).eq(1)
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

    evidence_score = pd.to_numeric(out["bad_evidence_score"], errors="coerce").fillna(999.0)
    evidence_label = out["bad_evidence_label"].astype(str)
    subtype = out["bad_internal_subtype"].astype(str)

    hard_nonphysical = (
        subtype.eq("bad_nonphysical")
        | out["kk_status"].astype(str).eq("KK_BAD")
        | combo_bad
        | (deep_neg & midband_neg)
        | evidence_label.eq("high")
    )

    weak_bad = (
        still_bad &
        kk_ok_like &
        (~hard_nonphysical) &
        evidence_label.eq("low") &
        (companion_count <= 1)
    )

    moderate_artifact_like = (
        still_bad &
        kk_ok_like &
        (~hard_nonphysical) &
        evidence_label.eq("moderate") &
        subtype.isin(["bad_artifact", "bad_low_information"]) &
        (companion_count <= 2) &
        (
            (prob_good >= V271_RESCUE_PROB_GOOD_MOD) |
            smooth_candidate |
            manifold_ok
        )
    )

    rescue_review = weak_bad | moderate_artifact_like

    if rescue_review.any():
        low_mask = rescue_review & evidence_label.eq("low")
        mod_mask = rescue_review & evidence_label.eq("moderate")

        out.loc[rescue_review, "pred_final"] = "review"
        out.loc[low_mask, "rule_patch_v26_reason"] = "v271_evidence_low_bad_to_review"
        out.loc[mod_mask, "rule_patch_v26_reason"] = "v271_evidence_moderate_bad_to_review"
        other_mask = rescue_review & ~(low_mask | mod_mask)
        out.loc[other_mask, "rule_patch_v26_reason"] = "v271_evidence_bad_to_review"

        out.loc[rescue_review, "rule_patch_v26_applied"] = 1
        out.loc[rescue_review, "rule_patch_v3_reason"] = out.loc[rescue_review, "rule_patch_v26_reason"]
        out.loc[rescue_review, "rule_patch_v3_applied"] = 1
        out.loc[rescue_review, "rule_patch_v2_reason"] = out.loc[rescue_review, "rule_patch_v26_reason"]
        out.loc[rescue_review, "rule_patch_v2_applied"] = 1

    print(
        f"[INFO] V2.7.1 evidence-based BAD gate | rescued BAD->REVIEW={int(rescue_review.sum())} "
        f"| remaining BAD={int((out['pred_final'].astype(str) == 'bad').sum())}"
    )

    out["qc_version"] = "V2.7.1_delivery_refine"
    return out

def _v271_write_run_report(features: pd.DataFrame, audit_df: pd.DataFrame | None = None):
    class_counts = (
        features["pred_final"].astype(str).value_counts(dropna=False)
        .rename_axis("pred_final")
        .reset_index(name="n")
    )
    class_counts_path = OUT_DIR / "sip_run_class_counts.csv"
    class_counts.to_csv(class_counts_path, index=False)

    bad_only = features[features["pred_final"].astype(str) == "bad"].copy()
    review_only = features[features["pred_final"].astype(str) == "review"].copy()

    bad_reason_counts = (
        bad_only.groupby(
            ["rule_patch_v26_reason", "bad_evidence_label", "bad_internal_subtype", "kk_status"],
            dropna=False,
        )
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    bad_reason_path = OUT_DIR / "sip_bad_reason_summary.csv"
    bad_reason_counts.to_csv(bad_reason_path, index=False)

    weak_bad = bad_only[
        bad_only["kk_status"].astype(str).isin(["KK_OK", "KK_NA"]) &
        bad_only["bad_evidence_label"].astype(str).isin(["low", "moderate"])
    ].copy()
    weak_bad["disagreement_bucket"] = "bad_weak_or_borderline"

    review_high_pg = review_only[
        pd.to_numeric(review_only.get("prob_good", np.nan), errors="coerce") >= V271_REVIEW_HIGH_PG
    ].copy()
    review_high_pg["disagreement_bucket"] = "review_high_prob_good"

    disagreement = pd.concat([weak_bad, review_high_pg], ignore_index=True, sort=False)
    disagreement_path = OUT_DIR / "sip_disagreement_candidates.csv"
    disagreement.to_csv(disagreement_path, index=False)

    report_txt = OUT_DIR / "sip_run_report.txt"
    with open(report_txt, "w", encoding="utf-8") as fh:
        fh.write("SIP QC run report\n")
        fh.write("=" * 60 + "\n")
        for _, row in class_counts.iterrows():
            fh.write(f"{row['pred_final']}: {int(row['n'])}\n")
        fh.write("\nTop BAD families:\n")
        for _, row in bad_reason_counts.head(15).iterrows():
            fh.write(
                f"- reason={row['rule_patch_v26_reason']} | evidence={row['bad_evidence_label']} "
                f"| subtype={row['bad_internal_subtype']} | kk={row['kk_status']} | n={int(row['n'])}\n"
            )
        fh.write(f"\nWeak/borderline BAD rows: {len(weak_bad)}\n")
        fh.write(f"High-pG REVIEW rows (p_good >= {V271_REVIEW_HIGH_PG:.2f}): {len(review_high_pg)}\n")
        if audit_df is not None and not audit_df.empty:
            fh.write(f"\nFiles discovered: {len(audit_df)}\n")
            fh.write(f"Files loaded: {int((audit_df['status'] == 'loaded').sum())}\n")
            fh.write(f"Files skipped: {int((audit_df['status'] == 'skipped').sum())}\n")
            used_train_col = (
                'used_for_training_actual' if 'used_for_training_actual' in audit_df.columns
                else ('used_for_training' if 'used_for_training' in audit_df.columns else None)
            )
            used_train_n = 0
            if used_train_col is not None:
                used_train_series = pd.to_numeric(audit_df[used_train_col], errors='coerce')
                if hasattr(used_train_series, 'fillna'):
                    used_train_n = int(used_train_series.fillna(0).sum())
            fh.write(f"Files used for training: {used_train_n}\n")

    print(f"[OK] Saved class counts: {class_counts_path}")
    print(f"[OK] Saved BAD reason summary: {bad_reason_path}")
    print(f"[OK] Saved disagreement candidates: {disagreement_path}")
    print(f"[OK] Saved run report: {report_txt}")

def main():
    print(f"[INFO] BASE_DIR: {BASE_DIR}")
    print("[INFO] Loading raw CSVs (field + optional HPE)...")
    raw, discovery_audit = load_all_raw_data()
    print(f"[INFO] Raw rows loaded: {len(raw):,}")
    print(f"[INFO] UniqueIDs in raw: {raw['UniqueID'].nunique():,}")

    print("[INFO] Building feature table...")
    features = build_feature_table(raw)
    feat_path = OUT_DIR / "sip_features_catboost_with_rule_feats.csv"
    features.to_csv(feat_path, index=False)
    print(f"[OK] Saved feature table: {feat_path}")

    audit_enriched = None
    if WRITE_DISCOVERY_AUDIT:
        audit_enriched = enrich_discovery_audit(discovery_audit, raw, features)
        audit_path = OUT_DIR / "sip_file_discovery_audit.csv"
        audit_enriched.to_csv(audit_path, index=False)
        print(f"[OK] Saved discovery audit: {audit_path}")

    print("[INFO] Training CatBoost...")
    model, feature_names, cat_features_idx, best_t = train_catboost(features)

    print("[INFO] Predicting on ALL UniqueIDs...")
    features = apply_catboost(model, features, feature_names, cat_features_idx, best_t)

    print("[INFO] Applying override layer...")
    features = apply_override(features)

    pred_path = OUT_DIR / "sip_quality_pred_FINAL_catboost.csv"
    features.to_csv(pred_path, index=False)
    print(f"[OK] Saved final predictions: {pred_path}")

    _v271_write_run_report(features, audit_enriched)

    print("[INFO] Saving sanity figures...")
    cm_path, thr_path = save_confusion_and_threshold(features, FIG_DIR)

    print("[INFO] Building UID meta...")
    uid_meta = build_uid_meta(raw)

    outputs = []
    for cls in ["good", "bad", "review"]:
        print(f"[INFO] Generating spectra PNGs for FINAL {cls.upper()}...")
        sub = build_all_spectrum_pngs(raw, features, cls, SPEC_DIR, uid_meta)
        pptx_path = OUT_DIR / f"SIP_QC_FINAL_{cls.upper()}_ALL_SPECTRA.pptx"
        print(f"[INFO] Building PPTX for FINAL {cls.upper()}...")
        build_gallery_deck(cls, sub, cm_path, thr_path, pptx_path)
        outputs.append(pptx_path)

    print("\n[DONE]")
    for p in outputs:
        print(f"  - {p}")
    print(f"  PNGs: {SPEC_DIR}")
    print(f"  FIGS: {FIG_DIR}")
