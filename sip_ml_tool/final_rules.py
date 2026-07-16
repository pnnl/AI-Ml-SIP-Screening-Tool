"""Final rule/reporting section loaded by SIP_ML_Tool.ipynb.

This file is executed inside the notebook namespace because the final
section intentionally overrides functions defined in the main workflow cell.
"""

# ------------------------------------------------------------------------------
# Final rules and reporting
# ------------------------------------------------------------------------------
# This cell adds the final behavior to the base workflow.
# It sets the grouped validation, final QC thresholds, output files, and review queue.

import time
from contextlib import contextmanager

from sklearn.model_selection import LeaveOneGroupOut

QC_VERSION = "final_practical"
FINAL_PRECISION_TARGET = 0.92
FINAL_REVIEW_HIGH_PG = 0.90
FINAL_BUILD_GALLERIES = True
FINAL_AUTO_TUNE_TRIAGE = True
FINAL_REVIEW_TO_GOOD_MIN_POINTS = 10
FINAL_REVIEW_TO_GOOD_REQUIRE_CLEAN_FLAGS = True
FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE = True
FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE_MIN_PROB_GOOD = 0.999
FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE_MAX_KK_PSEUDO_CHI2 = 0.01
FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE_MAX_PHASE_JAG_ALL = 0.02
FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE_MAX_TAIL_MAX_JUMP_MRAD = 20.0
FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE_MAX_ABS_PHASE_MRAD = 100.0
FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE_MAX_FLAG_COUNT = 1
FINAL_FORCE_BAD_UIDS = {
    "PolyphosphatePost_012521": "expert_review_force_bad",
    "ColF QR 0.01B1S1 Perch": "expert_review_force_bad",
    "ColF QR 0.1F1S3 Perch": "expert_review_force_bad",
}
FINAL_ENABLE_CONFLICT_CARVEOUT = True
FINAL_CONFLICT_SUBTYPES = ["bad_nonphysical", "bad_low_information"]
FINAL_CONFLICT_RESTORE_MAX_ABS_PHASE_MRAD = 15.0
FINAL_CONFLICT_RESTORE_MAX_KK_PSEUDO_CHI2 = 0.01
FINAL_CONFLICT_RESTORE_MAX_PHASE_JAG_ALL = 0.02
FINAL_CONFLICT_REVIEW_MAX_ABS_PHASE_MRAD = 500.0
FINAL_CONFLICT_REVIEW_MAX_KK_PSEUDO_CHI2 = 10.0
FINAL_CONFLICT_REVIEW_MAX_TAIL_MAX_JUMP_MRAD = 800.0
FINAL_PARAM_GRID = [
    {"depth": 3, "learning_rate": 0.03, "l2_leaf_reg": 6, "iterations": 1800},
    {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 5, "iterations": 2200},
    {"depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 5, "iterations": 2200},
    {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 4, "iterations": 1800},
]

# Output folders
OUT_DIR = BASE_DIR / "Final Results"
FIG_DIR = OUT_DIR / "figs"
SPEC_DIR = OUT_DIR / "spectra_pngs"
for _p in (OUT_DIR, FIG_DIR, SPEC_DIR):
    _p.mkdir(exist_ok=True)
STAGED_ARTIFACT_DIR = OUT_DIR / "Validation" / "model_artifacts"


def _progress_elapsed(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} hr {minutes} min {secs} sec"
    if minutes:
        return f"{minutes} min {secs} sec"
    return f"{secs} sec"


def _progress_banner(title: str) -> float:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    return time.perf_counter()


@contextmanager
def _progress_step(stage: str, step: int, total: int, title: str):
    print(f"\n[{stage} {step}/{total}] {title}")
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"[DONE] {title} ({_progress_elapsed(time.perf_counter() - start)})")


def _progress_detail(message: str) -> None:
    print(f"    {message}")

# Keep a copy of the base functions before replacing them
if "_final_base_train_catboost" not in globals():
    _final_base_train_catboost = train_catboost
if "_final_base_apply_catboost" not in globals():
    _final_base_apply_catboost = apply_catboost
if "_final_base_apply_override" not in globals():
    _final_base_apply_override = apply_override
if "_final_base_write_run_report" not in globals():
    _final_base_write_run_report = _v271_write_run_report


# Grouped validation helpers
def _final_grouped_cv_splits(train_df):
    groups = train_df["cv_group"].astype(str)
    y = train_df["true_label"].astype(str).values
    logo = LeaveOneGroupOut()
    X_dummy = np.zeros((len(train_df), 1))
    splits = []
    for fold_id, (tr_idx, va_idx) in enumerate(logo.split(X_dummy, y, groups), start=1):
        if pd.Series(y[tr_idx]).nunique() < 2:
            continue
        holdout_group = "|".join(sorted(pd.Series(groups.iloc[va_idx]).astype(str).unique().tolist()))
        splits.append((fold_id, holdout_group, tr_idx, va_idx))
    if len(splits) >= 2:
        return splits, "LeaveOneGroupOut"
    legacy = []
    for fold_id, (tr_idx, va_idx) in enumerate(_v271_grouped_cv_splits(train_df), start=1):
        holdout_group = "|".join(sorted(pd.Series(groups.iloc[va_idx]).astype(str).unique().tolist()))
        legacy.append((fold_id, holdout_group, tr_idx, va_idx))
    return legacy, "GroupedKFoldFallback"


# Train the CatBoost model with grouped validation
def train_catboost(features):
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
    print("\n[INFO] Excluding non-ML / provenance columns from training:")
    print(sorted([c for c in train_df.columns if c not in feature_names]))
    X = _prepare_ml_matrix(train_df, feature_names)
    y = train_df["true_label"].astype(str)
    groups = train_df["cv_group"].astype(str)
    splits, split_mode = _final_grouped_cv_splits(train_df)
    if len(splits) < 2:
        print("[WARN] Not enough valid grouped folds; falling back to prior trainer.")
        return _final_base_train_catboost(features)

    best = None
    all_cv_rows = []
    for params in FINAL_PARAM_GRID:
        oof_prob_good = pd.Series(np.nan, index=train_df.index, dtype=float)
        oof_holdout = pd.Series(pd.NA, index=train_df.index, dtype="object")
        fold_rows = []
        for fold_id, holdout_group, tr_idx, va_idx in splits:
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
                class_names=["bad", "good"],
                class_weights=[bad_weight, 1.0],
                allow_writing_files=False,
            )
            train_pool = Pool(X.iloc[tr_idx], y_tr, feature_names=feature_names)
            val_pool = Pool(X.iloc[va_idx], y_va, feature_names=feature_names)
            model_tmp.fit(train_pool, eval_set=val_pool)
            p_good = model_tmp.predict_proba(val_pool)[:, list(model_tmp.classes_).index("good")]
            oof_prob_good.iloc[va_idx] = p_good
            oof_holdout.iloc[va_idx] = holdout_group
            m05 = _binary_metrics_from_probs(y_va.astype(str).values, p_good, 0.5)
            fold_rows.append({
                "param_set": str(params),
                "fold": fold_id,
                "split_mode": split_mode,
                "holdout_group": holdout_group,
                "n_train": len(tr_idx),
                "n_val": len(va_idx),
                "groups_train": int(groups.iloc[tr_idx].nunique()),
                "groups_val": int(groups.iloc[va_idx].nunique()),
                "val_classes": "|".join(sorted(pd.Series(y_va).astype(str).unique().tolist())),
                "val_has_both_classes": bool(pd.Series(y_va).nunique() >= 2),
                "precision_good_t05": float(m05["precision_good"]),
                "recall_good_t05": float(m05["recall_good"]),
                "balanced_accuracy_t05": float(m05["balanced_accuracy"]),
                "bad_escape_rate_t05": float(m05["bad_escape_rate"]),
            })
        valid = oof_prob_good.notna()
        if valid.sum() == 0:
            continue
        chosen, thr_df = _pick_good_threshold(
            y.loc[valid].astype(str).values,
            oof_prob_good.loc[valid].astype(float).values,
            precision_target=FINAL_PRECISION_TARGET,
        )
        score = 3.0 * float(chosen["precision_good"]) - 2.5 * float(chosen["bad_escape_rate"]) + 1.5 * float(chosen["recall_good"]) + 0.5 * float(chosen["balanced_accuracy"])
        all_cv_rows.extend(fold_rows)
        if best is None or score > best["score"]:
            best = {
                "params": params,
                "score": score,
                "threshold": float(chosen["threshold"]),
                "threshold_info": chosen,
                "threshold_curve": thr_df.copy(),
                "fold_rows": list(fold_rows),
                "oof_prob_good": oof_prob_good.copy(),
                "oof_holdout": oof_holdout.copy(),
                "split_mode": split_mode,
            }
    if best is None:
        print("[WARN] final grouped CV did not yield a valid configuration; falling back to prior trainer.")
        return _final_base_train_catboost(features)

    cv_summary = pd.DataFrame(all_cv_rows)
    cv_summary_path = OUT_DIR / "sip_training_grouped_cv_summary.csv"
    cv_summary.to_csv(cv_summary_path, index=False)
    thr_path = OUT_DIR / "sip_training_threshold_curve_grouped_oof.csv"
    best["threshold_curve"].to_csv(thr_path, index=False)
    oof_df = train_df[["UniqueID", "true_label", "cv_group"]].copy()
    oof_df["holdout_group"] = best["oof_holdout"].astype("object")
    oof_df["oof_prob_good"] = pd.to_numeric(best["oof_prob_good"], errors="coerce")
    oof_df["oof_pred_binary"] = np.where(oof_df["oof_prob_good"] >= best["threshold"], "good", "bad")
    oof_path = OUT_DIR / "sip_training_grouped_oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    group_rows = []
    for holdout_group, sub in oof_df.groupby("holdout_group", dropna=False):
        m = _binary_metrics_from_probs(sub["true_label"].astype(str).values, pd.to_numeric(sub["oof_prob_good"], errors="coerce").fillna(0.0).values, best["threshold"])
        group_rows.append({
            "holdout_group": holdout_group,
            "n": len(sub),
            "good_n": int(np.sum(sub["true_label"].astype(str) == "good")),
            "bad_n": int(np.sum(sub["true_label"].astype(str) == "bad")),
            "precision_good": float(m["precision_good"]),
            "recall_good": float(m["recall_good"]),
            "balanced_accuracy": float(m["balanced_accuracy"]),
            "bad_escape_rate": float(m["bad_escape_rate"]),
        })
    group_report_path = OUT_DIR / "sip_training_group_holdout_report.csv"
    pd.DataFrame(group_rows).to_csv(group_report_path, index=False)
    summary_path = OUT_DIR / "sip_training_grouped_cv_report.txt"
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("SIP grouped CV training report - final practical\n")
        fh.write(f"Training rows: {len(train_df)}\n")
        fh.write(f"Training groups: {train_df['cv_group'].nunique()}\n")
        fh.write(f"Validation mode: {best['split_mode']}\n")
        fh.write(f"Chosen params: {best['params']}\n")
        fh.write(f"Chosen GOOD threshold: {best['threshold']:.3f}\n")
        fh.write(f"Selection mode: {best['threshold_info'].get('selection_mode', 'na')}\n")
        fh.write(f"Grouped OOF precision_good: {best['threshold_info'].get('precision_good', np.nan):.3f}\n")
        fh.write(f"Grouped OOF recall_good: {best['threshold_info'].get('recall_good', np.nan):.3f}\n")
        fh.write(f"Grouped OOF balanced_accuracy: {best['threshold_info'].get('balanced_accuracy', np.nan):.3f}\n")
        fh.write(f"Grouped OOF bad_escape_rate: {best['threshold_info'].get('bad_escape_rate', np.nan):.3f}\n")
    print(f"[OK] Saved grouped CV summary: {cv_summary_path}")
    print(f"[OK] Saved grouped threshold curve: {thr_path}")
    print(f"[OK] Saved grouped OOF predictions: {oof_path}")
    print(f"[OK] Saved holdout group report: {group_report_path}")
    print(f"[OK] Saved grouped CV report: {summary_path}")
    n_bad_full = max(int(np.sum(y == "bad")), 1)
    bad_weight_full = max(1.0, len(y) / (2.0 * n_bad_full))
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        depth=best["params"]["depth"],
        learning_rate=best["params"]["learning_rate"],
        iterations=best["params"]["iterations"],
        l2_leaf_reg=best["params"]["l2_leaf_reg"],
        random_seed=42,
        verbose=False,
        class_names=["bad", "good"],
        class_weights=[bad_weight_full, 1.0],
        allow_writing_files=False,
    )
    model.fit(Pool(X, y, feature_names=feature_names))
    globals()["_FINAL_TRAINING_REPORT"] = {
        "training_rows": int(len(train_df)),
        "training_groups": int(train_df["cv_group"].nunique()),
        "chosen_threshold": float(best["threshold"]),
        "precision_good": float(best["threshold_info"].get("precision_good", np.nan)),
        "recall_good": float(best["threshold_info"].get("recall_good", np.nan)),
        "balanced_accuracy": float(best["threshold_info"].get("balanced_accuracy", np.nan)),
        "bad_escape_rate": float(best["threshold_info"].get("bad_escape_rate", np.nan)),
        "validation_mode": best["split_mode"],
    }
    return model, feature_names, [], {
        "good_threshold": float(best["threshold"]),
        "bad_threshold": float(best["threshold_info"].get("bad_threshold", max(0.05, best["threshold"] - BAD_THRESHOLD_MARGIN))),
        "review_band_low": float(best["threshold_info"].get("review_band_low", max(0.0, best["threshold"] - NEAR_THRESHOLD_BAND))),
        "review_band_high": float(best["threshold_info"].get("review_band_high", min(1.0, best["threshold"] + NEAR_THRESHOLD_BAND))),
        "threshold_objective": str(best["threshold_info"].get("selection_mode", "final_grouped_cv")),
    }


# Apply the trained model to all spectra
def apply_catboost(model, features, feature_names, cat_features_idx, threshold_payload):
    out = _final_base_apply_catboost(model, features, feature_names, cat_features_idx, threshold_payload)
    prob_good = pd.to_numeric(out["prob_good"], errors="coerce").fillna(0.0)
    prob_bad = pd.to_numeric(out["prob_bad"], errors="coerce").fillna(0.0)
    prob_max = pd.to_numeric(out["prob_max"], errors="coerce").fillna(0.0)
    out["prob_gap"] = (prob_good - prob_bad).abs()
    out["model_confidence_tier"] = np.select(
        [prob_max >= 0.995, prob_max >= 0.97, prob_max >= 0.90, prob_max >= 0.75],
        ["very_high", "high", "medium", "low"],
        default="very_low",
    )
    return out


# Apply the tuned Review and Bad rules
def _final_apply_tuned_triage_policy(out: pd.DataFrame, policy: dict | None):
    if not policy:
        out["triage_policy_reason"] = "base"
        out["triage_policy_applied"] = 0
        out["triage_policy_name"] = "base_default"
        return out
    pred = out.get("pred_final", pd.Series(["review"] * len(out), index=out.index)).astype(str)
    uid = out.get("UniqueID", pd.Series([""] * len(out), index=out.index)).astype(str)
    pred_binary = out.get("pred_binary", pd.Series(["bad"] * len(out), index=out.index)).astype(str)
    prob_good = pd.to_numeric(out.get("prob_good", np.nan), errors="coerce").fillna(0.0)
    n_points = pd.to_numeric(out.get("n_points", np.nan), errors="coerce").fillna(0.0)
    kk_status = out.get("kk_status", pd.Series(["KK_NA"] * len(out), index=out.index)).astype(str)
    evidence = out.get("bad_evidence_label", pd.Series([""] * len(out), index=out.index)).astype(str)
    subtype = out.get("bad_internal_subtype", pd.Series([""] * len(out), index=out.index)).astype(str)
    neg_lt1k = out.get("rule_neg_lt_1k_tolerant", pd.Series([False] * len(out), index=out.index)).fillna(False).astype(bool)
    neg_1k_10k = out.get("rule_neg_1k_10k_tolerant", pd.Series([False] * len(out), index=out.index)).fillna(False).astype(bool)
    tail_bad = out.get("rule_tail_instability", pd.Series([False] * len(out), index=out.index)).fillna(False).astype(bool)
    osc_bad = out.get("rule_oscillatory_jagged", pd.Series([False] * len(out), index=out.index)).fillna(False).astype(bool)
    combo_bad = out.get("rule_combo_bad", pd.Series([False] * len(out), index=out.index)).fillna(False).astype(bool)
    review_to_good_clean = ~(neg_lt1k | neg_1k_10k | tail_bad | osc_bad | combo_bad)
    flag_count = neg_lt1k.astype(int) + neg_1k_10k.astype(int) + tail_bad.astype(int) + osc_bad.astype(int)
    max_abs_phase = pd.to_numeric(out.get("max_abs_phase_mrad", np.nan), errors="coerce")
    kk_chi2 = pd.to_numeric(out.get("kk_pseudo_chi2", np.nan), errors="coerce")
    phase_jag = pd.to_numeric(out.get("phase_jag_all", np.nan), errors="coerce")
    tail_jump = pd.to_numeric(out.get("tail_max_jump_mrad", np.nan), errors="coerce")
    reason = pd.Series("base", index=out.index, dtype="object")
    mild_subtypes = subtype.isin(policy.get("mild_subtypes", ["bad_artifact", "bad_low_information"]))
    mild_evidence = evidence.isin(policy.get("mild_evidence", ["low", "moderate"]))
    restore_conflict_good = pd.Series(False, index=out.index)
    downgrade_conflict_review = pd.Series(False, index=out.index)
    if bool(policy.get("enable_conflict_carveout", False)) and bool(globals().get("FINAL_ENABLE_CONFLICT_CARVEOUT", True)):
        conflict_subtypes = subtype.isin(policy.get("conflict_subtypes", globals().get("FINAL_CONFLICT_SUBTYPES", ["bad_nonphysical", "bad_low_information"])))
        conflict_mask = (
            pred.eq("bad")
            & pred_binary.eq("good")
            & kk_status.eq("KK_OK")
            & evidence.eq("high")
            & conflict_subtypes
        )
        restore_conflict_good = (
            conflict_mask
            & (max_abs_phase <= float(policy.get("conflict_restore_max_abs_phase_mrad", globals().get("FINAL_CONFLICT_RESTORE_MAX_ABS_PHASE_MRAD", 15.0))))
            & (kk_chi2 <= float(policy.get("conflict_restore_max_kk_pseudo_chi2", globals().get("FINAL_CONFLICT_RESTORE_MAX_KK_PSEUDO_CHI2", 0.01))))
            & (phase_jag <= float(policy.get("conflict_restore_max_phase_jag_all", globals().get("FINAL_CONFLICT_RESTORE_MAX_PHASE_JAG_ALL", 0.02))))
        )
        downgrade_conflict_review = (
            conflict_mask
            & ~restore_conflict_good
            & (max_abs_phase <= float(policy.get("conflict_review_max_abs_phase_mrad", globals().get("FINAL_CONFLICT_REVIEW_MAX_ABS_PHASE_MRAD", 500.0))))
            & (kk_chi2 <= float(policy.get("conflict_review_max_kk_pseudo_chi2", globals().get("FINAL_CONFLICT_REVIEW_MAX_KK_PSEUDO_CHI2", 10.0))))
            & (tail_jump <= float(policy.get("conflict_review_max_tail_max_jump_mrad", globals().get("FINAL_CONFLICT_REVIEW_MAX_TAIL_MAX_JUMP_MRAD", 800.0))))
        )
    review_to_good = (
        pred.eq("review")
        & pred_binary.eq("good")
        & kk_status.isin(policy.get("review_to_good_kk", ["KK_OK"]))
        & (n_points >= float(policy.get("review_to_good_min_points", globals().get("FINAL_REVIEW_TO_GOOD_MIN_POINTS", 10))))
        & (prob_good >= float(policy.get("review_to_good_pg", 1.01)))
    )
    if bool(policy.get("review_to_good_require_clean_flags", globals().get("FINAL_REVIEW_TO_GOOD_REQUIRE_CLEAN_FLAGS", True))):
        review_to_good = review_to_good & review_to_good_clean
    if bool(policy.get("review_to_good_require_mild", True)):
        review_to_good = review_to_good & mild_subtypes & mild_evidence
    rescue_review_leans_good = (
        bool(policy.get("review_model_leans_good_rescue", globals().get("FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE", True)))
        & pred.eq("review")
        & pred_binary.eq("good")
        & kk_status.eq("KK_OK")
        & (n_points >= float(policy.get("review_model_leans_good_rescue_min_points", globals().get("FINAL_REVIEW_TO_GOOD_MIN_POINTS", 10))))
        & (prob_good >= float(policy.get("review_model_leans_good_rescue_pg", globals().get("FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE_MIN_PROB_GOOD", 0.999))))
        & (kk_chi2 <= float(policy.get("review_model_leans_good_rescue_max_kk_pseudo_chi2", globals().get("FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE_MAX_KK_PSEUDO_CHI2", 0.01))))
        & (phase_jag <= float(policy.get("review_model_leans_good_rescue_max_phase_jag_all", globals().get("FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE_MAX_PHASE_JAG_ALL", 0.02))))
        & (tail_jump <= float(policy.get("review_model_leans_good_rescue_max_tail_max_jump_mrad", globals().get("FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE_MAX_TAIL_MAX_JUMP_MRAD", 20.0))))
        & (max_abs_phase <= float(policy.get("review_model_leans_good_rescue_max_abs_phase_mrad", globals().get("FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE_MAX_ABS_PHASE_MRAD", 100.0))))
        & (flag_count <= int(policy.get("review_model_leans_good_rescue_max_flag_count", globals().get("FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE_MAX_FLAG_COUNT", 1))))
        & ~osc_bad
        & ~combo_bad
        & ~review_to_good
    )
    bad_to_review = (
        pred.eq("bad")
        & pred_binary.eq("good")
        & kk_status.isin(policy.get("bad_to_review_kk", ["KK_OK", "KK_REVIEW"]))
        & mild_subtypes
        & mild_evidence
        & (prob_good >= float(policy.get("bad_to_review_pg", 1.01)))
    )
    out.loc[review_to_good | rescue_review_leans_good, "pred_final"] = "good"
    out.loc[bad_to_review, "pred_final"] = "review"
    out.loc[restore_conflict_good, "pred_final"] = "review"
    out.loc[downgrade_conflict_review, "pred_final"] = "review"
    force_bad_map = policy.get("force_bad_uids", globals().get("FINAL_FORCE_BAD_UIDS", {})) or {}
    force_bad_mask = uid.isin(list(force_bad_map.keys())) if len(force_bad_map) else pd.Series(False, index=out.index)
    out.loc[force_bad_mask, "pred_final"] = "bad"
    reason.loc[review_to_good] = f"policy_review_to_good_pg>={float(policy.get('review_to_good_pg', np.nan)):.3f}"
    reason.loc[rescue_review_leans_good] = f"policy_review_model_leans_good_rescue_pg>={float(policy.get('review_model_leans_good_rescue_pg', globals().get('FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE_MIN_PROB_GOOD', 0.999))):.3f}"
    reason.loc[bad_to_review] = f"policy_bad_to_review_pg>={float(policy.get('bad_to_review_pg', np.nan)):.3f}"
    reason.loc[restore_conflict_good] = "policy_conflict_restore_to_review"
    reason.loc[downgrade_conflict_review] = "policy_conflict_bad_to_review"
    if len(force_bad_map):
        for _uid, _reason in force_bad_map.items():
            reason.loc[uid.eq(str(_uid))] = str(_reason)
    out["triage_policy_reason"] = reason.astype(str)
    out["triage_policy_applied"] = reason.ne("base").astype(int)
    out["triage_policy_name"] = str(policy.get("policy_name", "physics_aware_triage"))
    return out


# Apply the final QC rules
def apply_override(features):
    out = _final_base_apply_override(features.copy())
    out = _final_apply_tuned_triage_policy(out, globals().get("_FINAL_TRIAGE_POLICY", None))
    pred = out["pred_final"].astype(str)
    prob_good = pd.to_numeric(out.get("prob_good", np.nan), errors="coerce").fillna(0.0)
    prob_bad = pd.to_numeric(out.get("prob_bad", np.nan), errors="coerce").fillna(0.0)
    prob_gap = pd.to_numeric(out.get("prob_gap", (prob_good - prob_bad).abs()), errors="coerce").fillna(0.0)
    kk_status = out.get("kk_status", pd.Series(["KK_NA"] * len(out), index=out.index)).astype(str)
    evidence = out.get("bad_evidence_label", pd.Series([""] * len(out), index=out.index)).astype(str)
    subtype = out.get("bad_internal_subtype", pd.Series([""] * len(out), index=out.index)).astype(str)
    good_mask = pred.eq("good")
    bad_mask = pred.eq("bad")
    review_mask = pred.eq("review")
    accept_clean = good_mask & kk_status.eq("KK_OK") & (prob_good >= 0.97) & (prob_gap >= 0.75)
    reject_strong = bad_mask & (evidence.eq("high") | kk_status.eq("KK_BAD") | (prob_bad >= 0.85))
    review_high = review_mask & (evidence.eq("high") | kk_status.eq("KK_BAD") | prob_good.between(0.45, 0.55) | (prob_gap <= 0.10))
    review_likely_good = review_mask & ~review_high & (prob_good >= FINAL_REVIEW_HIGH_PG)
    out["recommended_action"] = "manual_review"
    out.loc[accept_clean, "recommended_action"] = "accept"
    out.loc[good_mask & ~accept_clean, "recommended_action"] = "accept_with_spot_check"
    out.loc[reject_strong, "recommended_action"] = "reject"
    out.loc[bad_mask & ~reject_strong, "recommended_action"] = "review_before_reject"
    out.loc[review_high, "recommended_action"] = "manual_review_high_priority"
    out.loc[review_likely_good, "recommended_action"] = "manual_review_likely_good"
    out["review_priority"] = "none"
    out.loc[review_mask, "review_priority"] = "medium"
    out.loc[review_high | (bad_mask & ~reject_strong), "review_priority"] = "high"
    out.loc[review_likely_good, "review_priority"] = "low"
    out["decision_confidence"] = np.select(
        [accept_clean | reject_strong, (good_mask & (prob_good >= 0.93)) | (bad_mask & evidence.eq("high")), review_mask & ((prob_good >= 0.80) | (prob_bad >= 0.80) | (prob_gap >= 0.40))],
        ["very_high", "high", "medium"],
        default="low",
    )
    out["attention_rank"] = 0.0
    out.loc[review_mask, "attention_rank"] = 100.0
    out.loc[review_high, "attention_rank"] = 140.0
    out.loc[bad_mask & ~reject_strong, "attention_rank"] = 130.0
    out.loc[good_mask & ~accept_clean, "attention_rank"] = 20.0
    out["attention_rank"] = out["attention_rank"] + (1.0 - prob_gap.clip(0.0, 1.0)) * 10.0
    out["qc_summary_reason"] = np.where(
        pred.eq("bad"),
        subtype.astype(str) + ";" + evidence.astype(str) + ";" + kk_status.astype(str),
        np.where(pred.eq("good"), "good_adjudicated", np.where(prob_good >= FINAL_REVIEW_HIGH_PG, "review_model_leans_good", "review_borderline")),
    )
    policy_applied = out.get("triage_policy_applied", pd.Series([0] * len(out), index=out.index)).fillna(0).astype(int).eq(1)
    out.loc[policy_applied, "qc_summary_reason"] = out.loc[policy_applied, "triage_policy_reason"].astype(str)
    out["qc_version"] = QC_VERSION
    return out


# Build QC metrics and model comparison tables
def _final_uid_map(audit_df):
    if audit_df is None or audit_df.empty or "uniqueid_list" not in audit_df.columns:
        return pd.DataFrame(columns=["UniqueID", "folder_label_discovered", "category_folder_discovered", "file_name_discovered", "raw_path_discovered", "used_for_training_actual"])
    out = audit_df.copy()
    out["UniqueID"] = out["uniqueid_list"].astype(str)
    out = out.rename(columns={
        "folder_label": "folder_label_discovered",
        "category_folder": "category_folder_discovered",
        "file_name": "file_name_discovered",
        "raw_path": "raw_path_discovered",
    })
    return out[[c for c in ["UniqueID", "folder_label_discovered", "category_folder_discovered", "file_name_discovered", "raw_path_discovered", "used_for_training_actual"] if c in out.columns]].drop_duplicates("UniqueID")


def _final_qc_label_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    good_true = y_true == "good"
    bad_true = y_true == "bad"
    accept_mask = y_pred == "good"
    reject_mask = y_pred == "bad"
    review_mask = y_pred == "review"
    n_all = int(len(y_true))
    n_good = int(np.sum(good_true))
    n_bad = int(np.sum(bad_true))
    tp_good = int(np.sum(good_true & accept_mask))
    fp_good = int(np.sum(bad_true & accept_mask))
    tp_bad = int(np.sum(bad_true & reject_mask))
    fp_bad = int(np.sum(good_true & reject_mask))
    precision_good = tp_good / max(int(np.sum(accept_mask)), 1)
    precision_bad = tp_bad / int(np.sum(reject_mask)) if int(np.sum(reject_mask)) > 0 else np.nan
    recall_good = tp_good / max(n_good, 1)
    bad_escape_rate = fp_good / max(n_bad, 1)
    bad_capture_rate = 1.0 - bad_escape_rate if n_bad > 0 else np.nan
    good_reject_rate = fp_bad / max(n_good, 1)
    good_review_rate = float(np.sum(good_true & review_mask)) / max(n_good, 1)
    bad_review_rate = float(np.sum(bad_true & review_mask)) / max(n_bad, 1)
    resolved_mask = accept_mask | reject_mask
    resolved_accuracy = float(np.mean(y_true[resolved_mask] == y_pred[resolved_mask])) if np.any(resolved_mask) else np.nan
    return {
        "accuracy": float(np.mean(y_true == y_pred)) if n_all else np.nan,
        "balanced_accuracy": float(0.5 * (recall_good + bad_capture_rate)) if n_good > 0 and n_bad > 0 else np.nan,
        "precision_good": float(precision_good),
        "recall_good": float(recall_good),
        "bad_escape_rate": float(bad_escape_rate),
        "bad_capture_rate": float(bad_capture_rate) if np.isfinite(bad_capture_rate) else np.nan,
        "auto_accept_rate": float(np.mean(accept_mask)) if n_all else np.nan,
        "auto_accept_precision": float(precision_good),
        "auto_reject_rate": float(np.mean(reject_mask)) if n_all else np.nan,
        "auto_reject_precision": float(precision_bad) if np.isfinite(precision_bad) else np.nan,
        "good_reject_rate": float(good_reject_rate),
        "good_review_rate": float(good_review_rate),
        "bad_review_rate": float(bad_review_rate),
        "review_rate": float(np.mean(review_mask)) if n_all else np.nan,
        "resolved_rate": float(np.mean(resolved_mask)) if n_all else np.nan,
        "resolved_accuracy": float(resolved_accuracy) if np.isfinite(resolved_accuracy) else np.nan,
        "n_labeled": int(n_all),
        "n_good_labeled": int(n_good),
        "n_bad_labeled": int(n_bad),
    }


def _final_qc_metrics_by_group(comp_df, pred_col, model_name):
    rows = []
    if "holdout_group" not in comp_df.columns:
        return pd.DataFrame()
    for holdout_group, g in comp_df.groupby("holdout_group", dropna=False):
        metrics = _final_qc_label_metrics(g["true_label"].astype(str).values, g[pred_col].astype(str).values)
        metrics.update({
            "model": str(model_name),
            "holdout_group": str(holdout_group),
            "n": int(len(g)),
            "good_n": int((g["true_label"].astype(str) == "good").sum()),
            "bad_n": int((g["true_label"].astype(str) == "bad").sum()),
        })
        rows.append(metrics)
    return pd.DataFrame(rows)


def _final_score_triage_candidate(metrics):
    auto_accept_precision = float(metrics.get("auto_accept_precision", np.nan))
    bad_escape_rate = float(metrics.get("bad_escape_rate", np.nan))
    if (not np.isfinite(auto_accept_precision)) or (not np.isfinite(bad_escape_rate)):
        return -np.inf
    if bad_escape_rate > 0.0 + 1e-12:
        return -np.inf
    if auto_accept_precision < 0.999999:
        return -np.inf
    resolved_accuracy = float(metrics.get("resolved_accuracy", np.nan)) if np.isfinite(metrics.get("resolved_accuracy", np.nan)) else 0.0
    balanced_accuracy = float(metrics.get("balanced_accuracy", np.nan)) if np.isfinite(metrics.get("balanced_accuracy", np.nan)) else 0.0
    review_rate = float(metrics.get("review_rate", np.nan)) if np.isfinite(metrics.get("review_rate", np.nan)) else 1.0
    good_reject_rate = float(metrics.get("good_reject_rate", np.nan)) if np.isfinite(metrics.get("good_reject_rate", np.nan)) else 1.0
    auto_reject_precision = float(metrics.get("auto_reject_precision", np.nan)) if np.isfinite(metrics.get("auto_reject_precision", np.nan)) else 0.0
    recall_good = float(metrics.get("recall_good", np.nan)) if np.isfinite(metrics.get("recall_good", np.nan)) else 0.0
    return (
        3.0 * resolved_accuracy
        + 0.8 * balanced_accuracy
        + 0.5 * recall_good
        + 0.4 * auto_reject_precision
        - 1.2 * review_rate
        - 1.5 * good_reject_rate
    )


def _final_select_triage_policy(comp_base):
    search_rows = []
    default_metrics = _final_qc_label_metrics(comp_base["true_label"].astype(str).values, comp_base["pred_final"].astype(str).values)
    default_score = _final_score_triage_candidate(default_metrics)
    search_rows.append({
        "policy_name": "base_default",
        "review_to_good_pg": np.nan,
        "bad_to_review_pg": np.nan,
        "allow_kk_review_rescue": False,
        "score": default_score,
        **default_metrics,
    })
    best_policy = None
    best_metrics = default_metrics
    best_score = default_score
    for enable_conflict_carveout in [False, True]:
        for review_to_good_pg in [0.78, 0.80, 0.85, 0.90, 0.95]:
            for bad_to_review_pg in [0.75, 0.80, 0.85, 0.90]:
                for allow_kk_review_rescue in [False, True]:
                    for review_to_good_require_mild in [False, True]:
                        policy = {
                            "policy_name": f"physics_aware_triage_rtg{review_to_good_pg:.2f}_btr{bad_to_review_pg:.2f}_{'kkr' if allow_kk_review_rescue else 'kko'}_{'mild' if review_to_good_require_mild else 'broad'}_{'cc' if enable_conflict_carveout else 'nocc'}",
                            "review_to_good_pg": float(review_to_good_pg),
                            "bad_to_review_pg": float(bad_to_review_pg),
                            "review_to_good_kk": ["KK_OK", "KK_REVIEW"] if allow_kk_review_rescue else ["KK_OK"],
                            "review_to_good_require_mild": bool(review_to_good_require_mild),
                            "review_to_good_min_points": float(globals().get("FINAL_REVIEW_TO_GOOD_MIN_POINTS", 10)),
                            "review_to_good_require_clean_flags": bool(globals().get("FINAL_REVIEW_TO_GOOD_REQUIRE_CLEAN_FLAGS", True)),
                            "review_model_leans_good_rescue": bool(globals().get("FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE", True)),
                            "review_model_leans_good_rescue_pg": float(globals().get("FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE_MIN_PROB_GOOD", 0.999)),
                            "review_model_leans_good_rescue_min_points": float(globals().get("FINAL_REVIEW_TO_GOOD_MIN_POINTS", 10)),
                            "review_model_leans_good_rescue_max_kk_pseudo_chi2": float(globals().get("FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE_MAX_KK_PSEUDO_CHI2", 0.01)),
                            "review_model_leans_good_rescue_max_phase_jag_all": float(globals().get("FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE_MAX_PHASE_JAG_ALL", 0.02)),
                            "review_model_leans_good_rescue_max_tail_max_jump_mrad": float(globals().get("FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE_MAX_TAIL_MAX_JUMP_MRAD", 20.0)),
                            "review_model_leans_good_rescue_max_abs_phase_mrad": float(globals().get("FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE_MAX_ABS_PHASE_MRAD", 100.0)),
                            "review_model_leans_good_rescue_max_flag_count": int(globals().get("FINAL_REVIEW_MODEL_LEANS_GOOD_RESCUE_MAX_FLAG_COUNT", 1)),
                            "bad_to_review_kk": ["KK_OK", "KK_REVIEW"],
                            "enable_conflict_carveout": bool(enable_conflict_carveout),
                            "force_bad_uids": dict(globals().get("FINAL_FORCE_BAD_UIDS", {})),
                            "mild_subtypes": ["bad_artifact", "bad_low_information"],
                            "mild_evidence": ["low", "moderate"],
                            "conflict_subtypes": list(globals().get("FINAL_CONFLICT_SUBTYPES", ["bad_nonphysical", "bad_low_information"])),
                            "conflict_restore_max_abs_phase_mrad": float(globals().get("FINAL_CONFLICT_RESTORE_MAX_ABS_PHASE_MRAD", 15.0)),
                            "conflict_restore_max_kk_pseudo_chi2": float(globals().get("FINAL_CONFLICT_RESTORE_MAX_KK_PSEUDO_CHI2", 0.01)),
                            "conflict_restore_max_phase_jag_all": float(globals().get("FINAL_CONFLICT_RESTORE_MAX_PHASE_JAG_ALL", 0.02)),
                            "conflict_review_max_abs_phase_mrad": float(globals().get("FINAL_CONFLICT_REVIEW_MAX_ABS_PHASE_MRAD", 500.0)),
                            "conflict_review_max_kk_pseudo_chi2": float(globals().get("FINAL_CONFLICT_REVIEW_MAX_KK_PSEUDO_CHI2", 10.0)),
                            "conflict_review_max_tail_max_jump_mrad": float(globals().get("FINAL_CONFLICT_REVIEW_MAX_TAIL_MAX_JUMP_MRAD", 800.0)),
                        }
                        tuned = _final_apply_tuned_triage_policy(comp_base.copy(), policy)
                        metrics = _final_qc_label_metrics(tuned["true_label"].astype(str).values, tuned["pred_final"].astype(str).values)
                        score = _final_score_triage_candidate(metrics)
                        search_rows.append({
                            "policy_name": policy["policy_name"],
                            "review_to_good_pg": policy["review_to_good_pg"],
                            "bad_to_review_pg": policy["bad_to_review_pg"],
                            "allow_kk_review_rescue": allow_kk_review_rescue,
                            "enable_conflict_carveout": enable_conflict_carveout,
                            "review_to_good_require_mild": review_to_good_require_mild,
                            "score": score,
                            **metrics,
                        })
                        if score > best_score + 1e-12:
                            best_policy = policy
                            best_metrics = metrics
                            best_score = score
    search_df = pd.DataFrame(search_rows).sort_values("score", ascending=False)
    search_df.to_csv(OUT_DIR / "sip_triage_policy_search.csv", index=False)
    selected_rows = [{"selected": int(best_policy is not None), "score": best_score, **best_metrics}]
    if best_policy is None:
        selected_rows[0]["policy_name"] = "base_default"
        globals()["_FINAL_TRIAGE_POLICY"] = None
    else:
        selected_rows[0].update(best_policy)
        globals()["_FINAL_TRIAGE_POLICY"] = best_policy
    pd.DataFrame(selected_rows).to_csv(OUT_DIR / "sip_triage_policy_selected.csv", index=False)
    return globals().get("_FINAL_TRIAGE_POLICY", None), best_metrics, default_metrics


def _final_build_model_comparison(features):
    labeled = features.dropna(subset=["true_label"]).copy()
    oof_path = OUT_DIR / "sip_training_grouped_oof_predictions.csv"
    if labeled.empty or not oof_path.exists():
        return None
    oof = pd.read_csv(oof_path)
    keep = [c for c in ["UniqueID", "holdout_group", "oof_prob_good", "oof_pred_binary"] if c in oof.columns]
    comp = labeled.merge(oof[keep], on="UniqueID", how="inner")
    if comp.empty:
        return None
    meta = globals().get("_FINAL_TRAINING_REPORT", {})
    good_t = float(meta.get("chosen_threshold", 0.5))
    comp["prob_good"] = pd.to_numeric(comp["oof_prob_good"], errors="coerce").fillna(0.0)
    comp["prob_bad"] = 1.0 - comp["prob_good"]
    comp["prob_max"] = comp[["prob_good", "prob_bad"]].max(axis=1)
    comp["pred_binary"] = np.where(comp["prob_good"] >= good_t, "good", "bad")
    comp["good_threshold_used"] = float(good_t)
    comp["bad_threshold_used"] = float(max(0.05, good_t - BAD_THRESHOLD_MARGIN))
    comp["review_band_low"] = float(max(0.0, good_t - NEAR_THRESHOLD_BAND))
    comp["review_band_high"] = float(min(1.0, good_t + NEAR_THRESHOLD_BAND))
    comp["cautious_threshold"] = float(CAUTIOUS_THRESHOLD)
    comp["threshold_objective"] = "final_oof_model_comparison"
    comp["rule_near_thr"] = ((comp["prob_good"] >= comp["review_band_low"]) & (comp["prob_good"] <= comp["review_band_high"])).astype(int)
    comp["rule_low_conf"] = (comp["prob_max"] < CAUTIOUS_THRESHOLD).astype(int)
    comp["prob_gap"] = (comp["prob_good"] - comp["prob_bad"]).abs()
    import io
    from contextlib import redirect_stdout
    globals()["_FINAL_TRIAGE_POLICY"] = None
    with io.StringIO() as _buf, redirect_stdout(_buf):
        comp_base = _final_base_apply_override(comp.copy())
    selected_policy = None
    if FINAL_AUTO_TUNE_TRIAGE:
        selected_policy, _, _ = _final_select_triage_policy(comp_base.copy())
    if selected_policy:
        with io.StringIO() as _buf, redirect_stdout(_buf):
            comp_final = apply_override(comp.copy())
    else:
        with io.StringIO() as _buf, redirect_stdout(_buf):
            comp_final = apply_override(comp.copy())
    base_metrics = _final_qc_label_metrics(comp["true_label"].astype(str).values, comp["pred_binary"].astype(str).values)
    final_metrics = _final_qc_label_metrics(comp_final["true_label"].astype(str).values, comp_final["pred_final"].astype(str).values)
    cmp_df = pd.DataFrame([
        {"model": "catboost_only_oof", **base_metrics},
        {"model": "physics_aware_triage_oof", **final_metrics},
    ])
    metric_order = [
        "model", "n_labeled", "n_good_labeled", "n_bad_labeled", "auto_accept_precision", "bad_escape_rate",
        "auto_reject_precision", "good_reject_rate", "review_rate", "resolved_rate", "resolved_accuracy",
        "auto_accept_rate", "auto_reject_rate", "good_review_rate", "bad_review_rate", "precision_good",
        "recall_good", "bad_capture_rate", "balanced_accuracy", "accuracy",
    ]
    cmp_df = cmp_df[[c for c in metric_order if c in cmp_df.columns]]
    cmp_path = OUT_DIR / "sip_model_comparison.csv"
    cmp_df.to_csv(cmp_path, index=False)
    cmp_df.to_csv(OUT_DIR / "sip_qc_scorecard.csv", index=False)
    by_group = pd.concat([
        _final_qc_metrics_by_group(comp, "pred_binary", "catboost_only_oof"),
        _final_qc_metrics_by_group(comp_final, "pred_final", "physics_aware_triage_oof"),
    ], ignore_index=True, sort=False)
    if not by_group.empty:
        by_group.to_csv(OUT_DIR / "sip_model_comparison_by_group.csv", index=False)
        by_group.to_csv(OUT_DIR / "sip_qc_scorecard_by_group.csv", index=False)
    detail_cols = [c for c in ["UniqueID", "true_label", "holdout_group", "oof_prob_good", "pred_binary", "pred_final", "recommended_action", "review_priority", "qc_summary_reason", "bad_evidence_label", "bad_internal_subtype", "kk_status", "triage_policy_reason", "triage_policy_name"] if c in comp_final.columns]
    comp_final[detail_cols].to_csv(OUT_DIR / "sip_model_comparison_detail.csv", index=False)
    display_cols = [c for c in [
        "model", "auto_accept_precision", "bad_escape_rate", "auto_reject_precision", "good_reject_rate",
        "review_rate", "resolved_rate", "resolved_accuracy", "balanced_accuracy", "accuracy"
    ] if c in cmp_df.columns]
    print("[INFO] Upfront QC validation summary (OOF labeled rows):")
    print(cmp_df[display_cols].to_string(index=False))
    if selected_policy:
        print(f"[INFO] Selected physics-aware triage policy: {selected_policy['policy_name']}")
    else:
        print("[INFO] Selected physics-aware triage policy: base_default")
    return cmp_df


# Write the summary report files
def _v271_write_run_report(features, audit_df=None):
    _final_base_write_run_report(features, audit_df)
    action_cols = [c for c in ["UniqueID", "pred_final", "recommended_action", "review_priority", "decision_confidence", "attention_rank", "qc_summary_reason", "prob_good", "prob_bad", "prob_gap", "kk_status", "bad_evidence_label", "bad_internal_subtype", "folder_label_discovered", "category_folder_discovered", "source", "exp_group", "file_name_discovered", "raw_path_discovered", "true_label"] if c in features.columns]
    action = features[action_cols].copy().sort_values([c for c in ["attention_rank", "prob_bad", "prob_good"] if c in action_cols], ascending=False)
    action_path = OUT_DIR / "sip_action_queue.csv"
    action.to_csv(action_path, index=False)
    review_path = OUT_DIR / "sip_review_queue.csv"
    action[action["pred_final"].astype(str).eq("review")].to_csv(review_path, index=False)
    train_meta = globals().get("_FINAL_TRAINING_REPORT", {})
    summary = pd.DataFrame([{
        "qc_version": QC_VERSION,
        "prediction_rows": int(len(features)),
        "training_rows": int(train_meta.get("training_rows", features["true_label"].notna().sum() if "true_label" in features.columns else 0)),
        "training_groups": int(train_meta.get("training_groups", features["cv_group"].dropna().astype(str).nunique() if "cv_group" in features.columns else 0)),
        "chosen_good_threshold": float(train_meta.get("chosen_threshold", np.nan)),
        "precision_good": float(train_meta.get("precision_good", np.nan)),
        "recall_good": float(train_meta.get("recall_good", np.nan)),
        "balanced_accuracy": float(train_meta.get("balanced_accuracy", np.nan)),
        "bad_escape_rate": float(train_meta.get("bad_escape_rate", np.nan)),
        "validation_mode": str(train_meta.get("validation_mode", "na")),
        "good_n": int((features["pred_final"].astype(str) == "good").sum()),
        "review_n": int((features["pred_final"].astype(str) == "review").sum()),
        "bad_n": int((features["pred_final"].astype(str) == "bad").sum()),
    }])
    cmp_path = OUT_DIR / "sip_model_comparison.csv"
    if cmp_path.exists():
        cmp_df = pd.read_csv(cmp_path)
        for _, row in cmp_df.iterrows():
            prefix = str(row.get("model", "model"))
            for col in [
                "accuracy", "balanced_accuracy", "precision_good", "recall_good", "bad_escape_rate",
                "bad_capture_rate", "auto_accept_rate", "auto_accept_precision", "auto_reject_rate",
                "auto_reject_precision", "good_reject_rate", "good_review_rate", "bad_review_rate",
                "review_rate", "resolved_rate", "resolved_accuracy", "n_labeled", "n_good_labeled", "n_bad_labeled"
            ]:
                if col in row and pd.notna(row[col]):
                    summary.loc[0, f"{prefix}_{col}"] = row[col]
    by_group_path = OUT_DIR / "sip_qc_scorecard_by_group.csv"
    if by_group_path.exists():
        by_group_df = pd.read_csv(by_group_path)
        for model_name, g in by_group_df.groupby("model", dropna=False):
            prefix = str(model_name)
            ba = pd.to_numeric(g.get("balanced_accuracy", np.nan), errors="coerce")
            be = pd.to_numeric(g.get("bad_escape_rate", np.nan), errors="coerce")
            ra = pd.to_numeric(g.get("resolved_accuracy", np.nan), errors="coerce")
            if ba.notna().any():
                summary.loc[0, f"{prefix}_worst_holdout_balanced_accuracy"] = float(ba.min())
            if be.notna().any():
                summary.loc[0, f"{prefix}_worst_holdout_bad_escape_rate"] = float(be.max())
            if ra.notna().any():
                summary.loc[0, f"{prefix}_worst_holdout_resolved_accuracy"] = float(ra.min())
            summary.loc[0, f"{prefix}_holdout_groups_scored"] = int(g['holdout_group'].astype(str).nunique())
    triage_policy_path = OUT_DIR / "sip_triage_policy_selected.csv"
    if triage_policy_path.exists():
        triage_policy_df = pd.read_csv(triage_policy_path)
        if not triage_policy_df.empty:
            triage_row = triage_policy_df.iloc[0]
            for col in ["policy_name", "review_to_good_pg", "bad_to_review_pg", "review_to_good_require_mild"]:
                if col in triage_row and pd.notna(triage_row[col]):
                    summary.loc[0, f"triage_{col}"] = triage_row[col]
    summary_path = OUT_DIR / "sip_validation_snapshot.csv"
    summary.to_csv(summary_path, index=False)
    if audit_df is not None and not audit_df.empty:
        coverage = audit_df.groupby(["source_bucket", "folder_label", "status"], dropna=False).size().reset_index(name="n")
        coverage.to_csv(OUT_DIR / "sip_source_coverage_summary.csv", index=False)
    report_md = OUT_DIR / "sip_run_report.md"
    lines = [
        "# SIP QC run report - final practical",
        "",
        f"- QC version: `{QC_VERSION}`",
        f"- Validation mode: `{summary.iloc[0]['validation_mode']}`",
        f"- Training rows: `{int(summary.iloc[0]['training_rows'])}`",
        f"- Training groups: `{int(summary.iloc[0]['training_groups'])}`",
        f"- Chosen GOOD threshold: `{summary.iloc[0]['chosen_good_threshold']}`",
        f"- Selected triage policy: `{summary.iloc[0].get('triage_policy_name', 'base_default')}`",
        f"- Review-to-good threshold: `{summary.iloc[0].get('triage_review_to_good_pg', np.nan)}` | bad-to-review threshold: `{summary.iloc[0].get('triage_bad_to_review_pg', np.nan)}`",
        f"- Grouped OOF precision_good: `{summary.iloc[0]['precision_good']}`",
        f"- Grouped OOF recall_good: `{summary.iloc[0]['recall_good']}`",
        f"- Grouped OOF balanced_accuracy: `{summary.iloc[0]['balanced_accuracy']}`",
        f"- Grouped OOF bad_escape_rate: `{summary.iloc[0]['bad_escape_rate']}`",
        "",
        "## Final class counts",
        "",
        f"- good: `{int(summary.iloc[0]['good_n'])}`",
        f"- review: `{int(summary.iloc[0]['review_n'])}`",
        f"- bad: `{int(summary.iloc[0]['bad_n'])}`",
    ]
    comparison_models = [
        ("catboost_only_oof", "CatBoost-only with final SIP features (OOF)"),
        ("physics_aware_triage_oof", "Full physics-aware triage (OOF)"),
    ]
    available_models = [item for item in comparison_models if f"{item[0]}_accuracy" in summary.columns]
    if available_models:
        lines.extend([
            "",
            "## QC proof scorecard",
            "",
            "- Primary QC metrics are `auto_accept_precision`, `bad_escape_rate`, `auto_reject_precision`, `review_rate`, and `resolved_accuracy`.",
            "- Raw `accuracy` is secondary because `review` is an abstain/manual-review state, not a normal class.",
            "",
        ])
        for prefix, label in available_models:
            lines.extend([
                f"### {label}",
                "",
                f"- auto_accept_precision: `{summary.iloc[0][f'{prefix}_auto_accept_precision']}`",
                f"- bad_escape_rate: `{summary.iloc[0][f'{prefix}_bad_escape_rate']}`",
                f"- auto_reject_precision: `{summary.iloc[0][f'{prefix}_auto_reject_precision']}`",
                f"- good_reject_rate: `{summary.iloc[0][f'{prefix}_good_reject_rate']}`",
                f"- review_rate: `{summary.iloc[0][f'{prefix}_review_rate']}`",
                f"- resolved_rate: `{summary.iloc[0][f'{prefix}_resolved_rate']}`",
                f"- resolved_accuracy: `{summary.iloc[0][f'{prefix}_resolved_accuracy']}`",
                f"- recall_good: `{summary.iloc[0][f'{prefix}_recall_good']}`",
                f"- balanced_accuracy: `{summary.iloc[0][f'{prefix}_balanced_accuracy']}`",
                f"- accuracy: `{summary.iloc[0][f'{prefix}_accuracy']}`",
                f"- worst_holdout_balanced_accuracy: `{summary.iloc[0].get(f'{prefix}_worst_holdout_balanced_accuracy', np.nan)}`",
                f"- worst_holdout_bad_escape_rate: `{summary.iloc[0].get(f'{prefix}_worst_holdout_bad_escape_rate', np.nan)}`",
                f"- holdout_groups_scored: `{summary.iloc[0].get(f'{prefix}_holdout_groups_scored', np.nan)}`",
                "",
            ])
        if all(f"{m[0]}_resolved_accuracy" in summary.columns for m in comparison_models):
            triage_prefix = "physics_aware_triage_oof"
            base_prefix = "catboost_only_oof"
            delta_good_reject = float(summary.iloc[0][f'{base_prefix}_good_reject_rate']) - float(summary.iloc[0][f'{triage_prefix}_good_reject_rate'])
            delta_review = float(summary.iloc[0][f'{triage_prefix}_review_rate']) - float(summary.iloc[0][f'{base_prefix}_review_rate'])
            delta_resolved_acc = float(summary.iloc[0][f'{triage_prefix}_resolved_accuracy']) - float(summary.iloc[0][f'{base_prefix}_resolved_accuracy'])
            lines.extend([
                "## Comparison takeaway",
                "",
                f"- Full physics-aware triage reduces wrong hard auto-rejections by `{delta_good_reject}` versus CatBoost-only.",
                f"- Full physics-aware triage changes resolved_accuracy by `{delta_resolved_acc}` and review_rate by `{delta_review}` versus CatBoost-only.",
                "",
            ])
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Saved action queue: {action_path}")
    print(f"[OK] Saved review queue: {review_path}")
    print(f"[OK] Saved validation snapshot: {summary_path}")
    print(f"[OK] Saved practical markdown run report: {report_md}")


# Build and organize the final reader files
def _final_move_path(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists() and dst.is_file():
            dst.unlink()
        dst.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            _final_move_path(child, dst / child.name)
        shutil.rmtree(src, ignore_errors=True)
        return
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst, ignore_errors=True)
        else:
            dst.unlink()
    shutil.copy2(str(src), str(dst))
    src.unlink(missing_ok=True)


def _final_reader_queue_view(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "pred_final": "final_class",
        "exp_group": "experiment_group",
        "folder_label_discovered": "discovered_folder",
        "file_name_discovered": "source_file",
        "raw_path_discovered": "raw_file_path",
    }
    preferred = [
        "UniqueID", "pred_final", "recommended_action", "review_priority", "decision_confidence",
        "attention_rank", "prob_good", "prob_bad", "prob_gap", "kk_status", "qc_summary_reason",
        "bad_evidence_label", "folder_label_discovered", "exp_group", "file_name_discovered", "raw_path_discovered"
    ]
    use_cols = [c for c in preferred if c in df.columns]
    out = df[use_cols].rename(columns=rename_map).copy()
    if "final_class" in out.columns:
        class_order = pd.CategoricalDtype(["good", "review", "bad"], ordered=True)
        out["final_class"] = out["final_class"].astype(class_order)
        sort_cols = [c for c in ["final_class", "attention_rank", "prob_good"] if c in out.columns]
        sort_dir = [True, False, False][:len(sort_cols)]
        out = out.sort_values(sort_cols, ascending=sort_dir)
        out["final_class"] = out["final_class"].astype(str)
    return out.reset_index(drop=True)


def _final_build_reader_deliverables() -> None:
    model_label_map = {
        "catboost_only_oof": "CatBoost-only with final SIP features",
        "physics_aware_triage_oof": "Full physics-aware triage",
    }
    score_src = OUT_DIR / "sip_qc_scorecard.csv"
    if score_src.exists():
        score = pd.read_csv(score_src)
        if "model" in score.columns:
            score["model"] = score["model"].map(model_label_map).fillna(score["model"])
        keep = [
            "model", "n_labeled", "auto_accept_precision", "bad_escape_rate", "auto_reject_precision",
            "good_reject_rate", "review_rate", "resolved_rate", "resolved_accuracy",
            "precision_good", "recall_good", "balanced_accuracy", "accuracy"
        ]
        score = score[[c for c in keep if c in score.columns]].copy()
        score.to_csv(OUT_DIR / "03_QC_Proof_Scorecard.csv", index=False)

    counts_src = OUT_DIR / "sip_run_class_counts.csv"
    if counts_src.exists():
        counts = pd.read_csv(counts_src).rename(columns={"pred_final": "final_class", "n": "count"})
        if "final_class" in counts.columns:
            class_order = pd.CategoricalDtype(["good", "review", "bad"], ordered=True)
            counts["final_class"] = counts["final_class"].astype(class_order)
            counts = counts.sort_values("final_class")
            counts["final_class"] = counts["final_class"].astype(str)
        counts.to_csv(OUT_DIR / "04_Final_Class_Counts.csv", index=False)

    action_src = OUT_DIR / "sip_action_queue.csv"
    if action_src.exists():
        action = pd.read_csv(action_src)
        _final_reader_queue_view(action).to_csv(OUT_DIR / "05_Final_Decision_Queue.csv", index=False)

    review_src = OUT_DIR / "sip_review_queue.csv"
    if review_src.exists():
        review = pd.read_csv(review_src)
        _final_reader_queue_view(review).to_csv(OUT_DIR / "06_Manual_Review_Queue.csv", index=False)

    pred_src = OUT_DIR / "sip_quality_pred_FINAL_catboost.csv"
    if pred_src.exists():
        pred = pd.read_csv(pred_src).rename(columns={"pred_final": "final_class"})
        pred.to_csv(OUT_DIR / "07_All_Spectra_Detailed_Predictions.csv", index=False)


def _final_write_output_index(out_dir: Path) -> None:
    scorecard_path = out_dir / "03_QC_Proof_Scorecard.csv"
    counts_path = out_dir / "04_Final_Class_Counts.csv"
    lines = [
        "# SIP ML output guide",
        "",
        "Open these first:",
        "- `02_QC_Summary_Report.md`: concise narrative summary of the run and the QC proof metrics.",
        "- `03_QC_Proof_Scorecard.csv`: the main side-by-side proof table comparing CatBoost-only vs full physics-aware triage.",
        "- `04_Final_Class_Counts.csv`: final totals for good / review / bad.",
        "- `05_Final_Decision_Queue.csv`: operational queue for accept / review / reject decisions.",
        "- `06_Manual_Review_Queue.csv`: only the spectra that still need manual review.",
        "- `07_Gallery_GOOD.pptx`, `08_Gallery_REVIEW.pptx`, `09_Gallery_BAD.pptx`: slide galleries for visual inspection.",
        "",
        "Top-level files are the reader-facing deliverables only.",
        "Everything else was moved into `Validation` so the delivery folder stays clean.",
        "That support folder is not required for a normal reader review; it is only for technical traceability.",
    ]
    try:
        if scorecard_path.exists():
            score = pd.read_csv(scorecard_path)
            for _, row in score.iterrows():
                lines.extend([
                    "",
                    f"## {row['model']}",
                    f"- auto_accept_precision: `{row.get('auto_accept_precision', np.nan)}`",
                    f"- bad_escape_rate: `{row.get('bad_escape_rate', np.nan)}`",
                    f"- review_rate: `{row.get('review_rate', np.nan)}`",
                    f"- resolved_accuracy: `{row.get('resolved_accuracy', np.nan)}`",
                    f"- balanced_accuracy: `{row.get('balanced_accuracy', np.nan)}`",
                ])
        if counts_path.exists():
            counts = pd.read_csv(counts_path)
            lines.extend(["", "## Final class counts"])
            for _, row in counts.iterrows():
                lines.append(f"- {row['final_class']}: `{int(row['count'])}`")
    except Exception as exc:
        lines.extend(["", f"Index note: could not load summary metrics automatically ({exc})."])
    (out_dir / "01_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _final_organize_outputs() -> None:
    _final_build_reader_deliverables()
    renamed_top_level = {
        "sip_run_report.md": "02_QC_Summary_Report.md",
        "SIP_QC_FINAL_GOOD_ALL_SPECTRA.pptx": "07_Gallery_GOOD.pptx",
        "SIP_QC_FINAL_REVIEW_ALL_SPECTRA.pptx": "08_Gallery_REVIEW.pptx",
        "SIP_QC_FINAL_BAD_ALL_SPECTRA.pptx": "09_Gallery_BAD.pptx",
    }
    supporting_validation = {
        "sip_action_queue.csv": "Validation/sip_action_queue.csv",
        "sip_bad_reason_summary.csv": "Validation/sip_bad_reason_summary.csv",
        "sip_disagreement_candidates.csv": "Validation/sip_disagreement_candidates.csv",
        "sip_features_catboost_with_rule_feats.csv": "Validation/sip_features_catboost_with_rule_feats.csv",
        "sip_file_discovery_audit.csv": "Validation/sip_file_discovery_audit.csv",
        "sip_model_comparison.csv": "Validation/sip_model_comparison.csv",
        "sip_model_comparison_by_group.csv": "Validation/sip_model_comparison_by_group.csv",
        "sip_model_comparison_detail.csv": "Validation/sip_model_comparison_detail.csv",
        "sip_qc_scorecard.csv": "Validation/sip_qc_scorecard.csv",
        "sip_qc_scorecard_by_group.csv": "Validation/sip_qc_scorecard_by_group.csv",
        "sip_quality_pred_FINAL_catboost.csv": "Validation/sip_quality_pred_FINAL_catboost.csv",
        "sip_review_queue.csv": "Validation/sip_review_queue.csv",
        "sip_run_class_counts.csv": "Validation/sip_run_class_counts.csv",
        "sip_run_report.txt": "Validation/sip_run_report_legacy.txt",
        "sip_source_coverage_summary.csv": "Validation/sip_source_coverage_summary.csv",
        "sip_training_group_holdout_report.csv": "Validation/sip_training_group_holdout_report.csv",
        "sip_training_grouped_cv_report.txt": "Validation/sip_training_grouped_cv_report.txt",
        "sip_training_grouped_cv_summary.csv": "Validation/sip_training_grouped_cv_summary.csv",
        "sip_training_grouped_oof_predictions.csv": "Validation/sip_training_grouped_oof_predictions.csv",
        "sip_training_threshold_curve_grouped_oof.csv": "Validation/sip_training_threshold_curve_grouped_oof.csv",
        "sip_triage_policy_search.csv": "Validation/sip_triage_policy_search.csv",
        "sip_triage_policy_selected.csv": "Validation/sip_triage_policy_selected.csv",
        "sip_validation_snapshot.csv": "Validation/sip_validation_snapshot.csv",
        "07_All_Spectra_Detailed_Predictions.csv": "Validation/07_All_Spectra_Detailed_Predictions.csv",
        "11_Override_Audit_Summary.md": "Validation/override_audit/11_Override_Audit_Summary.md",
        "12_Override_Audit_Candidates.csv": "Validation/override_audit/12_Override_Audit_Candidates.csv",
        "13_Override_Audit_Gallery.pptx": "Validation/override_audit/13_Override_Audit_Gallery.pptx",
        "14_Override_Audit_AI_FirstPass.csv": "Validation/override_audit/14_Override_Audit_AI_FirstPass.csv",
        "15_Override_Audit_Findings.md": "Validation/override_audit/15_Override_Audit_Findings.md",
        "16_Override_Audit_Prototype_Changes.csv": "Validation/override_audit/16_Override_Audit_Prototype_Changes.csv",
        "supporting_validation": "Validation",
    }
    supporting_dirs = {
        "figs": "Validation/assets/figs",
        "spectra_pngs": "Validation/assets/spectra_pngs",
        "ppt_preview": "Validation/assets/ppt_preview",
        "ppt_preview_refined": "Validation/assets/ppt_preview_refined",
        "supporting_assets": "Validation/assets",
    }
    legacy_outputs = {
        "SIP_QC_FINAL_GOOD_ALL_SPECTRA.pptx": "Validation/archive/SIP_QC_FINAL_GOOD_ALL_SPECTRA_pre_final.pptx",
        "SIP_QC_FINAL_BAD_ALL_SPECTRA.pptx": "Validation/archive/SIP_QC_FINAL_BAD_ALL_SPECTRA_pre_final.pptx",
        "SIP_QC_FINAL_REVIEW_ALL_SPECTRA.pptx": "Validation/archive/SIP_QC_FINAL_REVIEW_ALL_SPECTRA_pre_final.pptx",
        "archive_legacy_outputs": "Validation/archive",
    }
    for mapping in [renamed_top_level, supporting_validation, supporting_dirs, legacy_outputs]:
        for src_name, dst_rel in mapping.items():
            _final_move_path(OUT_DIR / src_name, OUT_DIR / dst_rel)
    stale_top_level = [
        "01_README_OUTPUTS.md",
        "02_QC_Summary_Report_FINAL.md",
        "03_QC_Proof_Scorecard_FINAL.csv",
        "04_Final_Class_Counts_FINAL.csv",
        "05_Final_Decision_Queue_FINAL.csv",
        "06_Manual_Review_Queue_FINAL.csv",
        "07_Gallery_GOOD_FINAL.pptx",
        "08_Gallery_REVIEW_FINAL.pptx",
        "09_Gallery_BAD_FINAL.pptx",
    ]
    for stale_name in stale_top_level:
        stale_path = OUT_DIR / stale_name
        if stale_path.exists():
            if stale_path.is_dir():
                shutil.rmtree(stale_path, ignore_errors=True)
            else:
                stale_path.unlink(missing_ok=True)
    _final_write_output_index(OUT_DIR)


def _staged_json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _staged_save_training_artifacts(
    model,
    feature_names,
    cat_features_idx,
    threshold_payload,
    features,
    audit_enriched=None,
) -> dict:
    import json

    STAGED_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = STAGED_ARTIFACT_DIR / "catboost_model.cbm"
    model.save_model(str(model_path))

    feature_cache_path = STAGED_ARTIFACT_DIR / "features_unpredicted.pkl"
    features.to_pickle(feature_cache_path)

    audit_cache_path = None
    if audit_enriched is not None and not audit_enriched.empty:
        audit_cache_path = STAGED_ARTIFACT_DIR / "discovery_audit_enriched.pkl"
        audit_enriched.to_pickle(audit_cache_path)

    metadata = {
        "model_path": str(model_path),
        "feature_cache_path": str(feature_cache_path),
        "audit_cache_path": str(audit_cache_path) if audit_cache_path else None,
        "feature_names": list(feature_names),
        "cat_features_idx": list(cat_features_idx),
        "threshold_payload": threshold_payload,
        "training_report": globals().get("_FINAL_TRAINING_REPORT", {}),
        "triage_policy": globals().get("_FINAL_TRIAGE_POLICY", None),
    }
    metadata_path = STAGED_ARTIFACT_DIR / "training_artifacts.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=_staged_json_default), encoding="utf-8")
    print(f"[OK] Saved staged model artifact: {model_path}")
    print(f"[OK] Saved staged workflow metadata: {metadata_path}")
    return metadata


def _staged_load_training_artifacts() -> tuple:
    import json

    metadata_path = STAGED_ARTIFACT_DIR / "training_artifacts.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Training artifacts were not found: {metadata_path}. Run run_training_stage() first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = CatBoostClassifier()
    model.load_model(metadata["model_path"])
    globals()["_FINAL_TRAINING_REPORT"] = metadata.get("training_report", {})
    globals()["_FINAL_TRIAGE_POLICY"] = metadata.get("triage_policy", None)
    return (
        model,
        metadata["feature_names"],
        metadata.get("cat_features_idx", []),
        metadata["threshold_payload"],
        metadata,
    )


def run_training_stage() -> dict:
    """Run data loading, feature building, grouped validation, and model training."""
    stage_start = _progress_banner("SIP ML TOOL | TRAINING AND VALIDATION STAGE")
    print(f"[INFO] BASE_DIR: {BASE_DIR}")
    print(f"[INFO] QC_VERSION: {QC_VERSION}")
    print(f"[INFO] Output directory: {OUT_DIR}")

    with _progress_step("TRAINING", 1, 5, "Loading raw SIP files"):
        _progress_detail("Scanning Good, Bad, Cautious, and Unknown data folders.")
        raw, discovery_audit = load_all_raw_data()
    if raw.empty:
        raise ValueError("No SIP spectra were loaded. Check the configured data folders before running the final workflow.")

    unique_raw = int(raw["UniqueID"].nunique())
    _progress_detail(f"Loaded {len(raw):,} measurement rows across {unique_raw:,} unique spectra.")
    if "source" in raw.columns:
        source_counts = raw.groupby("source").size().rename("n")
        _progress_detail("Source coverage:")
        print(source_counts.to_string())

    with _progress_step("TRAINING", 2, 5, "Building spectrum-level feature table"):
        _progress_detail("Converting each multi-frequency spectrum into one feature record.")
        features = build_feature_table(raw)
        feat_path = OUT_DIR / "sip_features_catboost_with_rule_feats.csv"
        features.to_csv(feat_path, index=False)
        _progress_detail(f"Saved feature table: {feat_path}")

    audit_enriched = None
    with _progress_step("TRAINING", 3, 5, "Preparing training labels and discovery audit"):
        labeled_n = int(features["true_label"].notna().sum() if "true_label" in features.columns else 0)
        good_n = int((features["true_label"].astype(str) == "good").sum() if "true_label" in features.columns else 0)
        bad_n = int((features["true_label"].astype(str) == "bad").sum() if "true_label" in features.columns else 0)
        _progress_detail(f"Feature records: {len(features):,}")
        _progress_detail(f"Labeled spectra: {labeled_n:,} ({good_n:,} Good, {bad_n:,} Bad)")
        if "cv_group" in features.columns:
            group_n = int(features.loc[features["true_label"].notna(), "cv_group"].dropna().astype(str).nunique())
            _progress_detail(f"Grouped-validation holdout groups: {group_n}")
        if WRITE_DISCOVERY_AUDIT:
            audit_enriched = enrich_discovery_audit(discovery_audit, raw, features)
            audit_path = OUT_DIR / "sip_file_discovery_audit.csv"
            audit_enriched.to_csv(audit_path, index=False)
            _progress_detail(f"Saved discovery audit: {audit_path}")
        else:
            _progress_detail("Discovery audit disabled.")

    with _progress_step("TRAINING", 4, 5, "Training CatBoost and running grouped validation"):
        _progress_detail("Testing candidate CatBoost settings using grouped out-of-fold validation.")
        model, feature_names, cat_features_idx, threshold_payload = train_catboost(features)
        train_meta = globals().get("_FINAL_TRAINING_REPORT", {})
        if train_meta:
            _progress_detail(f"Selected Good threshold: {train_meta.get('chosen_threshold', 'n/a')}")
            _progress_detail(f"Validation mode: {train_meta.get('validation_mode', 'n/a')}")

    with _progress_step("TRAINING", 5, 5, "Selecting physics-guided triage policy and saving model artifacts"):
        _progress_detail("Comparing CatBoost-only decisions against the physics-guided workflow.")
        _final_build_model_comparison(features)
        selected_policy = globals().get("_FINAL_TRIAGE_POLICY", None)
        if selected_policy:
            _progress_detail(f"Selected triage policy: {selected_policy.get('policy_name', 'selected_policy')}")
        else:
            _progress_detail("Selected triage policy: base_default")

        metadata = _staged_save_training_artifacts(
            model,
            feature_names,
            cat_features_idx,
            threshold_payload,
            features,
            audit_enriched,
        )

    print(f"\n[DONE] Training/validation stage complete ({_progress_elapsed(time.perf_counter() - stage_start)})")
    return {
        "raw_rows": int(len(raw)),
        "feature_rows": int(len(features)),
        "labeled_rows": int(features["true_label"].notna().sum() if "true_label" in features.columns else 0),
        "artifacts": metadata,
    }


def run_prediction_stage(use_cached_features: bool = True, build_galleries: bool | None = None) -> dict:
    """Load the saved model and run prediction, physics triage, and reporting."""
    if build_galleries is None:
        build_galleries = bool(FINAL_BUILD_GALLERIES)

    stage_start = _progress_banner("SIP ML TOOL | PREDICTION AND REPORTING STAGE")
    print(f"[INFO] BASE_DIR: {BASE_DIR}")
    print(f"[INFO] QC_VERSION: {QC_VERSION}")
    print(f"[INFO] Output directory: {OUT_DIR}")

    with _progress_step("PREDICTION", 1, 6, "Loading saved model and workflow settings"):
        _progress_detail("Loading CatBoost model, selected thresholds, and triage-policy settings.")
        model, feature_names, cat_features_idx, threshold_payload, metadata = _staged_load_training_artifacts()
        _progress_detail(f"Model artifact: {metadata.get('model_path', 'n/a')}")

    raw = None
    audit_enriched = None
    feature_cache = Path(metadata.get("feature_cache_path", ""))
    audit_cache = Path(metadata.get("audit_cache_path", "")) if metadata.get("audit_cache_path") else None

    with _progress_step("PREDICTION", 2, 6, "Loading or rebuilding spectrum features"):
        if use_cached_features and feature_cache.exists():
            _progress_detail("Using cached feature table from the training stage.")
            features = pd.read_pickle(feature_cache)
            if audit_cache and audit_cache.exists():
                audit_enriched = pd.read_pickle(audit_cache)
        else:
            _progress_detail("Cached features unavailable or disabled; rebuilding from raw SIP files.")
            raw, discovery_audit = load_all_raw_data()
            if raw.empty:
                raise ValueError("No SIP spectra were loaded. Check the configured data folders before running prediction.")
            features = build_feature_table(raw)
            if WRITE_DISCOVERY_AUDIT:
                audit_enriched = enrich_discovery_audit(discovery_audit, raw, features)
        _progress_detail(f"Feature records ready for prediction: {len(features):,}")

    with _progress_step("PREDICTION", 3, 6, "Applying CatBoost probabilities to all spectra"):
        _progress_detail("Generating Good/Bad probabilities before physics-guided screening.")
        features = apply_catboost(model, features, feature_names, cat_features_idx, threshold_payload)

    with _progress_step("PREDICTION", 4, 6, "Applying physics-guided screening"):
        _progress_detail("Combining model probabilities with spectral-shape and KK consistency checks.")
        features = apply_override(features)
        class_counts = features["pred_final"].astype(str).value_counts().reindex(["good", "review", "bad"], fill_value=0)
        _progress_detail(
            f"Final classes: {int(class_counts['good'])} Good, "
            f"{int(class_counts['review'])} Review, {int(class_counts['bad'])} Bad"
        )

    if audit_enriched is not None and not audit_enriched.empty:
        audit_map = _final_uid_map(audit_enriched)
        if not audit_map.empty:
            features = features.merge(audit_map, on="UniqueID", how="left")

    with _progress_step("PREDICTION", 5, 6, "Writing prediction tables, scorecards, and summary reports"):
        pred_path = OUT_DIR / "sip_quality_pred_FINAL_catboost.csv"
        features.to_csv(pred_path, index=False)
        _progress_detail(f"Saved final predictions: {pred_path}")

        _v271_write_run_report(features, audit_enriched)

    with _progress_step("PREDICTION", 6, 6, "Generating figures, galleries, and reader-facing outputs"):
        _progress_detail("Saving sanity-check figures.")
        cm_path, thr_path = save_confusion_and_threshold(features, FIG_DIR)

        if build_galleries:
            if raw is None:
                _progress_detail("Reloading raw spectra for gallery generation.")
                raw, _ = load_all_raw_data()
            _progress_detail("Building UID metadata for spectrum galleries.")
            uid_meta = build_uid_meta(raw)
            for cls in ["good", "bad", "review"]:
                _progress_detail(f"Generating spectra PNGs and PowerPoint gallery for final {cls.upper()} spectra.")
                sub = build_all_spectrum_pngs(raw, features, cls, SPEC_DIR, uid_meta)
                pptx_path = OUT_DIR / f"SIP_QC_FINAL_{cls.upper()}_ALL_SPECTRA.pptx"
                build_gallery_deck(cls, sub, cm_path, thr_path, pptx_path)
        else:
            _progress_detail("Gallery generation skipped for this prediction run.")

        _progress_detail("Organizing output directory for reader-facing delivery.")
        _final_organize_outputs()

    print(f"\n[DONE] Prediction/reporting stage complete ({_progress_elapsed(time.perf_counter() - stage_start)})")
    return {
        "prediction_rows": int(len(features)),
        "good_n": int((features["pred_final"].astype(str) == "good").sum()),
        "review_n": int((features["pred_final"].astype(str) == "review").sum()),
        "bad_n": int((features["pred_final"].astype(str) == "bad").sum()),
    }


def run_staged_workflow(build_galleries: bool | None = None) -> dict:
    """Run the official workflow as separate training and prediction stages."""
    overall_start = _progress_banner("SIP ML TOOL | FULL QUALITY-CONTROL WORKFLOW")
    _progress_detail("This run will train/validate the model, apply final predictions, and write outputs.")
    training = run_training_stage()
    prediction = run_prediction_stage(use_cached_features=True, build_galleries=build_galleries)
    print("\n" + "=" * 78)
    print("SIP ML TOOL | WORKFLOW COMPLETE")
    print("=" * 78)
    _progress_detail(f"Total elapsed time: {_progress_elapsed(time.perf_counter() - overall_start)}")
    _progress_detail(
        f"Final classes: {prediction['good_n']} Good, "
        f"{prediction['review_n']} Review, {prediction['bad_n']} Bad"
    )
    _progress_detail(f"Reader-facing outputs: {OUT_DIR}")
    return {"training": training, "prediction": prediction}


# Run the final workflow
def main():
    """Run the full official workflow as Training followed by Prediction."""
    return run_staged_workflow(build_galleries=FINAL_BUILD_GALLERIES)
