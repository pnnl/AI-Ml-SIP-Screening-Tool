# Extracted from SIP_ML_Tool.ipynb main workflow cell.
# Initial spectrum feature-table construction.

def build_feature_table(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for uid, g in raw.groupby("UniqueID"):
        g = g.sort_values("Freq")

        col = int(g["column_number"].iloc[0]) if len(g) else -1
        date = g["file_date"].min()

        labels_here = [str(x) for x in g["folder_label"].dropna().unique().tolist()]
        if "good" in labels_here:
            label = "good"
        elif "bad" in labels_here:
            label = "bad"
        else:
            label = None

        freq  = g["Freq"].values
        phase = g["Phase_rad"].values
        mag   = g["Magnitude_ratio"].values

        # Global stats
        freq_min = float(np.min(freq))
        freq_max = float(np.max(freq))
        phase_min = float(np.min(phase))
        phase_max = float(np.max(phase))
        phase_mean = float(np.mean(phase))
        phase_frac_neg = float((phase < 0).mean())

        real_mean = float(g["RealCond"].mean())
        imag_mean = float(g["ImagCond"].mean())
        imp_mean  = float(g["Impedance_Ohms"].mean())
        imp_std   = float(g["Impedance_Ohms"].std())

        n_points = int(len(freq))
        log_freq_range = float(np.log10(freq_max / freq_min)) if freq_min > 0 else np.nan
        phase_std = float(np.std(phase))
        real_std = float(np.std(g["RealCond"].values))
        imag_std = float(np.std(g["ImagCond"].values))
        imp_range = float(g["Impedance_Ohms"].max() - g["Impedance_Ohms"].min())

        src = g["source"].iloc[0] if "source" in g.columns else "field"
        exp = g["exp_group"].iloc[0] if "exp_group" in g.columns else "none"

        low_mask  = band_mask(freq,  0.1,   10.0)
        mid_mask  = band_mask(freq, 10.0,  100.0)
        high_mask = band_mask(freq, 100.0, 10000.0)

        def frac_positive_phase(mask):
            if mask.sum() == 0:
                return np.nan
            return float((phase[mask] > 0).mean())

        frac_pos_low  = frac_positive_phase(low_mask)
        frac_pos_mid  = frac_positive_phase(mid_mask)
        frac_pos_high = frac_positive_phase(high_mask)

        jag_all = phase_jaggedness(phase)
        jag_mid = phase_jaggedness(phase[mid_mask]) if mid_mask.sum() > 0 else np.nan

        slope_mag_all  = slope_vs_logf(freq, mag)
        slope_mag_low  = slope_vs_logf(freq[low_mask],  mag[low_mask])
        slope_mag_mid  = slope_vs_logf(freq[mid_mask],  mag[mid_mask])
        slope_mag_high = slope_vs_logf(freq[high_mask], mag[high_mask])

        peakF, peakP = pick_peak(g)

        # Rule flags from raw curve (updated)
        flags = compute_rule_flags_from_group(g)
        kk = compute_kk_features(freq, g["Impedance_Ohms"].values, phase, max_freq=3000.0)

        rows.append(
            {
                "UniqueID": uid,
                "column_number": col,
                "file_date": date,

                "freq_min": freq_min,
                "freq_max": freq_max,
                "n_points": n_points,
                "log_freq_range": log_freq_range,

                "phase_min": phase_min,
                "phase_max": phase_max,
                "phase_mean": phase_mean,
                "phase_std": phase_std,
                "phase_frac_negative": phase_frac_neg,

                "realcond_mean": real_mean,
                "realcond_std": real_std,
                "imagcond_mean": imag_mean,
                "imagcond_std": imag_std,
                "imp_mean": imp_mean,
                "imp_std": imp_std,
                "imp_range": imp_range,

                "peakFreq": peakF,
                "peakPhase_mrad": peakP,

                "frac_pos_low":  frac_pos_low,
                "frac_pos_mid":  frac_pos_mid,
                "frac_pos_high": frac_pos_high,
                "phase_jag_all": jag_all,
                "phase_jag_mid": jag_mid,
                "slope_mag_all":  slope_mag_all,
                "slope_mag_low":  slope_mag_low,
                "slope_mag_mid":  slope_mag_mid,
                "slope_mag_high": slope_mag_high,

                "source": src,
                "exp_group": exp,

                "true_label": label,

                # Rule-derived features (updated)
                "rule_neg_lt_1k": int(flags["rule_neg_lt_1k"]),
                "rule_neg_1k_10k": int(flags["rule_neg_1k_10k"]),
                "rule_jagged_run": int(flags["rule_jagged_run"]),
                "rule_big_jump_count": int(flags["rule_big_jump_count"]),
                "rule_isolated_spikes": int(flags["rule_isolated_spikes"]),
                "rule_isolated_spikes_allowed": int(flags["rule_isolated_spikes_allowed"]),
                "max_phase_gt_1k_mrad": float(flags["max_phase_gt_1k_mrad"]),
                "max_abs_phase_mrad": float(flags.get("max_abs_phase_mrad", np.nan)),
                "max_abs_phase_gt_1k_mrad": float(flags.get("max_abs_phase_gt_1k_mrad", np.nan)),
                "kk_num_rc": int(kk.get("num_rc", -1)) if pd.notna(kk.get("num_rc", np.nan)) else -1,
                "kk_pseudo_chi2": float(kk.get("pseudo_chi2", np.nan)),
                "kk_mu": float(kk.get("mu", np.nan)),
                "kk_resid_rms": float(kk.get("resid_rms", np.nan)),
                "kk_resid_max": float(kk.get("resid_max", np.nan)),
            }
        )

    return pd.DataFrame(rows)
