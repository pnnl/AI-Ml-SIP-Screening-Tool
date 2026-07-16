# Extracted from SIP_ML_Tool.ipynb main workflow cell.
# CSV readers, discovery audit, raw data loading, and UID metadata.

def load_field_csv(path: Path, folder_label: str):
    hdr = find_header_line(path)
    if hdr is None:
        return None, "no_header_detected"

    try:
        df = pd.read_csv(path, skiprows=hdr, engine="python")
    except Exception as e:
        print(f"[WARN] Failed field read {path.name}: {e}")
        return None, f"read_error: {e}"
    if df.empty:
        return None, "empty_after_read"

    freq_cols  = [c for c in df.columns if "Freq" in str(c) and "Hz" in str(c)]
    phase_cols = [c for c in df.columns if "Phase" in str(c) and "rad" in str(c)]
    imp_cols   = [c for c in df.columns if "Impedance" in str(c)]

    if not freq_cols or not phase_cols or not imp_cols:
        return None, "missing_required_columns"

    df = df.rename(columns={
        freq_cols[0]:  "Frequency_Hz",
        phase_cols[0]: "Phase_rad",
        imp_cols[0]:   "Impedance_Ohms",
    })

    meta = parse_filename(path)
    for k, v in meta.items():
        df[k] = v

    df["folder_label"] = folder_label
    df["source"] = "field"
    df["exp_group"] = "none"
    df["raw_path"] = str(path)
    df["pulled_from"] = f"{folder_label} data"

    return df, "loaded"


def load_hpe_csv(path: Path, folder_label: str, exp_group: str, category_folder: str):
    hdr = find_header_line(path)
    if hdr is None:
        return None, "no_header_detected"

    try:
        df = pd.read_csv(path, skiprows=hdr, engine="python")
    except Exception as e:
        print(f"[WARN] Failed HPE read {path.name}: {e}")
        return None, f"read_error: {e}"
    if df.empty:
        return None, "empty_after_read"

    freq_cols  = [c for c in df.columns if "Freq" in str(c) and "Hz" in str(c)]
    phase_cols = [c for c in df.columns if "Phase" in str(c) and "rad" in str(c)]
    imp_cols   = [c for c in df.columns if "Impedance" in str(c)]

    if not freq_cols or not phase_cols or not imp_cols:
        return None, "missing_required_columns"

    df = df.rename(columns={
        freq_cols[0]:  "Frequency_Hz",
        phase_cols[0]: "Phase_rad",
        imp_cols[0]:   "Impedance_Ohms",
    })

    file_date = read_header_date(path)

    df["file_date"]          = file_date
    df["column_name"]        = path.stem.lower()
    df["column_number"]      = -1
    df["column_number_dash"] = "NA"
    df["original_file_name"] = path.name
    df["folder_label"]       = folder_label
    df["source"]             = "HPE"
    df["exp_group"]          = exp_group
    df["raw_path"]           = str(path)
    df["pulled_from"]        = f"HPE/{category_folder}/{exp_group}"

    return df, "loaded"


def _make_audit_row(path: Path, *, source_bucket: str, folder_label: str, category_folder: str,
                    exp_group: str, status: str, reason: str, intended_role: str):
    try:
        rel_path = str(path.relative_to(BASE_DIR))
    except Exception:
        rel_path = str(path)
    return {
        "raw_path": str(path),
        "relative_path": rel_path,
        "source_bucket": source_bucket,
        "folder_label": folder_label,
        "category_folder": category_folder,
        "exp_group": exp_group,
        "status": status,
        "reason": reason,
        "intended_role": intended_role,
        "file_name": path.name,
    }


def _print_discovery_counts(audit_df: pd.DataFrame):
    if audit_df.empty:
        print("[WARN] Discovery audit is empty.")
        return

    print("[INFO] Discovery counts by source/folder/status:")
    summary = (
        audit_df.groupby(["source_bucket", "folder_label", "status"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["source_bucket", "folder_label", "status"])
    )
    print(summary.to_string(index=False))

    loaded = audit_df[audit_df["status"] == "loaded"].copy()
    if not loaded.empty:
        print("[INFO] Loaded file counts by source/folder:")
        loaded_summary = (
            loaded.groupby(["source_bucket", "folder_label"], dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(["source_bucket", "folder_label"])
        )
        print(loaded_summary.to_string(index=False))


def enrich_discovery_audit(audit_df: pd.DataFrame, raw: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    audit_df = audit_df.copy()
    if audit_df.empty:
        return audit_df

    raw_map = raw[["raw_path", "UniqueID", "source", "folder_label", "exp_group", "pulled_from"]].drop_duplicates()
    feat_map = features[["UniqueID", "true_label"]].drop_duplicates()
    raw_map = raw_map.merge(feat_map, on="UniqueID", how="left")

    path_summary = (
        raw_map.groupby("raw_path", as_index=False)
        .agg(
            loaded_uniqueids=("UniqueID", "nunique"),
            uniqueid_list=("UniqueID", lambda s: " | ".join(sorted({str(x) for x in s.dropna()}))),
            assigned_true_labels=("true_label", lambda s: " | ".join(sorted({str(x) for x in s.dropna()}))),
            used_for_training_actual=("true_label", lambda s: bool(s.isin(["good", "bad"]).any())),
            pulled_from_loaded=("pulled_from", lambda s: s.iloc[0] if len(s) else ""),
            source_loaded=("source", lambda s: s.iloc[0] if len(s) else ""),
            exp_group_loaded=("exp_group", lambda s: s.iloc[0] if len(s) else ""),
        )
    )

    out = audit_df.merge(path_summary, on="raw_path", how="left")
    out["discovered"] = True
    out["loaded"] = out["status"].eq("loaded")
    out["used_for_training_actual"] = out["used_for_training_actual"].where(
        out["used_for_training_actual"].notna(), False
    ).astype(bool)

    for col in ["uniqueid_list", "assigned_true_labels", "pulled_from_loaded", "source_loaded", "exp_group_loaded"]:
        if col in out.columns:
            out[col] = out[col].fillna("")

    return out


def load_all_raw_data():
    all_dfs = []
    audit_rows = []

    # Field
    for label, folder in FOLDERS.items():
        intended_role = "train" if label in {"good", "bad"} else "score_only"
        if not folder.exists():
            print(f"[WARN] Field folder missing: {folder}")
            continue

        field_files = sorted(folder.glob("*.csv"))
        print(f"[INFO] Field discovery | {label}: {len(field_files)} top-level CSV files found")
        for fpath in field_files:
            df_tmp, reason = load_field_csv(fpath, label)
            status = "loaded" if df_tmp is not None else "skipped"
            audit_rows.append(_make_audit_row(
                fpath,
                source_bucket="field",
                folder_label=label,
                category_folder=f"{label} data",
                exp_group="none",
                status=status,
                reason=reason,
                intended_role=intended_role,
            ))
            if df_tmp is not None:
                all_dfs.append(df_tmp)

    # HPE (optional)
    HPE_LABEL_MAP = {
        "Good data":     "good",
        "Bad data":      "bad",
        "Cautious data": "cautious",
        "Unknown data":  "unknown",
    }

    if HPE_BASE.exists():
        for category_folder, label in HPE_LABEL_MAP.items():
            intended_role = "train" if label in {"good", "bad"} else "score_only"
            cat_path = HPE_BASE / category_folder
            if not cat_path.exists():
                continue
            cat_short = category_folder.replace(" data", "")

            if HPE_RECURSIVE:
                hpe_files = sorted([p for p in cat_path.rglob("*.csv") if p.is_file()])
                mode_txt = "recursive"
            else:
                hpe_files = []
                for sub in cat_path.iterdir():
                    if not sub.is_dir():
                        continue
                    hpe_files.extend(sorted(sub.glob("*.csv")))
                hpe_files = sorted(hpe_files)
                mode_txt = "one-level"

            print(f"[INFO] HPE discovery | {category_folder}: {len(hpe_files)} CSV files found ({mode_txt})")

            for fpath in hpe_files:
                try:
                    rel_parent = fpath.parent.relative_to(cat_path)
                    exp_group = str(rel_parent).replace("\\", "/")
                except Exception:
                    exp_group = fpath.parent.name
                if exp_group in ("", "."):
                    exp_group = "__root__"

                df_tmp, reason = load_hpe_csv(fpath, label, exp_group, category_folder=cat_short)
                status = "loaded" if df_tmp is not None else "skipped"
                audit_rows.append(_make_audit_row(
                    fpath,
                    source_bucket="HPE",
                    folder_label=label,
                    category_folder=category_folder,
                    exp_group=exp_group,
                    status=status,
                    reason=reason,
                    intended_role=intended_role,
                ))
                if df_tmp is not None:
                    all_dfs.append(df_tmp)

    if not all_dfs:
        raise RuntimeError("No CSVs loaded. Check BASE_DIR/folders and header parsing.")

    audit_df = pd.DataFrame(audit_rows)
    _print_discovery_counts(audit_df)

    data = pd.concat(all_dfs, ignore_index=True)

    # Numeric cleaning
    data["Freq"] = pd.to_numeric(data["Frequency_Hz"], errors="coerce")
    data["Phase_rad"] = pd.to_numeric(data["Phase_rad"], errors="coerce")
    data["Impedance_Ohms"] = pd.to_numeric(data["Impedance_Ohms"], errors="coerce")

    if "Magnitude[ratio]" in data.columns:
        data["Magnitude_ratio"] = pd.to_numeric(data["Magnitude[ratio]"], errors="coerce")
    else:
        data["Magnitude_ratio"] = np.nan

    data = data.dropna(subset=["Freq", "Phase_rad", "Impedance_Ohms"])
    data["file_date"] = pd.to_datetime(data["file_date"], errors="coerce")

    # day index (field-like)
    data["day"] = data.groupby("column_name")["file_date"].transform(lambda s: (s - s.min()).dt.days)

    # UniqueID
    data["UniqueID"] = data["column_number_dash"].astype(str) + "_" + data["day"].astype(str)
    mask_hpe = data["source"].astype(str).str.upper() == "HPE"
    if mask_hpe.any():
        data.loc[mask_hpe, "UniqueID"] = data.loc[mask_hpe, "original_file_name"].str.replace(".csv", "", regex=False)

    # Phase_mrad sign convention
    data["Phase_mrad"] = -1000.0 * data["Phase_rad"]

    # Conductivity components
    K_GEOM = 290.0
    rho = K_GEOM * data["Impedance_Ohms"]
    sigma = (1.0 / rho) * 1000.0  # mS/m
    data["RealCond"] = sigma * np.cos(data["Phase_rad"])
    data["ImagCond"] = sigma * np.sin(-1.0 * data["Phase_rad"])

    return data, audit_df


def build_uid_meta(raw_data: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_data.groupby("UniqueID", as_index=False)
        .agg(
            pulled_from=("pulled_from", lambda s: s.iloc[0]),
            source_raw=("source", lambda s: s.iloc[0]),
            exp_group_raw=("exp_group", lambda s: s.iloc[0]),
            raw_path=("raw_path", lambda s: s.iloc[0]),
        )
    )
