# Extracted from SIP_ML_Tool.ipynb main workflow cell.
# Original base main workflow before later QC patches.

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
