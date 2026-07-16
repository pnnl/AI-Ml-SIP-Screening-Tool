# Extracted from SIP_ML_Tool.ipynb main workflow cell.
# Initial CatBoost training and prediction helpers.

def train_catboost(features: pd.DataFrame):
    train_df = features.dropna(subset=["true_label"]).copy()
    print("\nTraining samples:", len(train_df))
    print(train_df["true_label"].value_counts())

    drop_cols = ["true_label", "UniqueID", "file_date"]
    X = train_df.drop(columns=drop_cols)
    y = train_df["true_label"].astype(str)

    feature_names = X.columns.tolist()

    cat_feature_names = ["source", "exp_group"]
    cat_features_idx = [i for i, col in enumerate(feature_names) if col in cat_feature_names]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_pool = Pool(X_train, y_train, feature_names=feature_names, cat_features=cat_features_idx)
    val_pool = Pool(X_val, y_val, feature_names=feature_names, cat_features=cat_features_idx)

    param_grid = [
        {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 5, "iterations": 3000},
        {"depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 5, "iterations": 3000},
        {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3, "iterations": 2500},
        {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5, "iterations": 2000},
        {"depth": 7, "learning_rate": 0.03, "l2_leaf_reg": 7, "iterations": 2500},
    ]

    best_model = None
    best_params = None
    best_val_acc = -np.inf

    class_weight_bad = float(len(y) / (2 * (y == "bad").sum()))

    for params in param_grid:
        print("\nTrying params:", params)
        model_tmp = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Accuracy",
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            iterations=params["iterations"],
            l2_leaf_reg=params["l2_leaf_reg"],
            random_seed=42,
            verbose=False,
            class_weights=[1.0, class_weight_bad],
        )
        model_tmp.fit(train_pool, eval_set=val_pool)
        val_pred = model_tmp.predict(val_pool)
        acc = (val_pred == y_val.values).mean()
        print("  Validation accuracy:", acc)

        if acc > best_val_acc:
            best_val_acc = acc
            best_model = model_tmp
            best_params = params

    print("\nBest validation accuracy from grid:", best_val_acc)
    print("Best params:", best_params)

    model = best_model

    # Threshold tuning on validation
    val_prob = model.predict_proba(val_pool)
    classes_model = list(model.classes_)
    idx_good = classes_model.index("good")
    p_good_val = val_prob[:, idx_good]

    best_t = 0.5
    best_acc = 0.0
    for t in np.linspace(0.1, 0.9, 161):
        y_pred_t = np.where(p_good_val >= t, "good", "bad")
        acc_t = (y_pred_t == y_val.values).mean()
        if acc_t > best_acc:
            best_acc = acc_t
            best_t = t

    print(f"Best GOOD prob threshold on validation: {best_t:.3f}")
    print(f"Tuned validation accuracy: {best_acc:.3f}")

    # Retrain on all labeled
    full_pool = Pool(
        train_df.drop(columns=["true_label", "UniqueID", "file_date"]),
        y,
        feature_names=X.columns.tolist(),
        cat_features=cat_features_idx,
    )
    model.fit(full_pool)

    return model, X.columns.tolist(), cat_features_idx, best_t


def apply_catboost(model, features: pd.DataFrame, feature_names, cat_features_idx, best_t: float):
    drop_cols = ["true_label", "UniqueID", "file_date"]
    X_all = features.drop(columns=drop_cols)

    pool_all = Pool(X_all, feature_names=feature_names, cat_features=cat_features_idx)

    prob_all = model.predict_proba(pool_all)
    classes_model = list(model.classes_)
    idx_good = classes_model.index("good")
    idx_bad  = classes_model.index("bad")

    p_good_all = prob_all[:, idx_good]
    p_bad_all  = prob_all[:, idx_bad]
    maxp_all   = prob_all.max(axis=1)

    pred_binary = np.where(p_good_all >= best_t, "good", "bad")

    features["pred_binary"] = pred_binary
    features["prob_good"] = p_good_all
    features["prob_bad"] = p_bad_all
    features["prob_max"] = maxp_all
    features["good_threshold_used"] = float(best_t)
    features["cautious_threshold"] = float(CAUTIOUS_THRESHOLD)

    return features
