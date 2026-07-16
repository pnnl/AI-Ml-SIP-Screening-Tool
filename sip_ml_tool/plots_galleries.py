# Extracted from SIP_ML_Tool.ipynb main workflow cell.
# Confusion plots, spectrum PNGs, and PowerPoint gallery creation.

def save_confusion_and_threshold(features: pd.DataFrame, out_dir: Path):
    cm_path = out_dir / "confusion_true_label.png"
    thr_path = out_dir / "threshold_curve_true_label.png"

    labeled = features[features.get("true_label", "").isin(["good", "bad"])].copy() if "true_label" in features.columns else features.iloc[0:0].copy()
    if labeled.empty:
        plt.figure(figsize=(4, 4)); plt.text(0.1, 0.5, "No true_label rows found"); plt.axis("off")
        plt.savefig(cm_path, dpi=220); plt.close()
        plt.figure(figsize=(4, 4)); plt.text(0.1, 0.5, "No true_label rows found"); plt.axis("off")
        plt.savefig(thr_path, dpi=220); plt.close()
        return cm_path, thr_path

    best_t = float(features["good_threshold_used"].dropna().iloc[0]) if "good_threshold_used" in features.columns else 0.5
    labeled["pred_from_thr"] = np.where(labeled["prob_good"] >= best_t, "good", "bad")

    cm = confusion_matrix(labeled["true_label"], labeled["pred_from_thr"], labels=["good", "bad"])

    plt.figure(figsize=(4.2, 4.2))
    plt.imshow(cm)
    plt.xticks([0, 1], ["good", "bad"])
    plt.yticks([0, 1], ["good", "bad"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.title("Confusion matrix (true_label rows)")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=220)
    plt.close()

    thresholds = np.linspace(0.1, 0.9, 161)
    acc = []
    for t in thresholds:
        pred_t = np.where(labeled["prob_good"].values >= t, "good", "bad")
        acc.append((pred_t == labeled["true_label"].values).mean())

    plt.figure(figsize=(5.2, 4.0))
    plt.plot(thresholds, acc, marker="o", markersize=2, linewidth=1)
    plt.axvline(best_t, linestyle="--")
    plt.xlabel("GOOD probability threshold")
    plt.ylabel("Accuracy (true_label rows)")
    plt.title("Threshold tuning curve (true_label rows)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(thr_path, dpi=220)
    plt.close()

    return cm_path, thr_path


# ==============================================================================
# SPECTRA PNGS
# ==============================================================================

def plot_spectrum_png(g: pd.DataFrame, outpath: Path, kk_line=None) -> None:
    g = g.sort_values("Freq")
    x = g["Freq"].values
    y = g["Phase_mrad"].values

    plt.figure(figsize=PLOT_FIGSIZE)
    plt.semilogx(x, y, linewidth=1.1)

    plt.xlabel("Frequency (Hz)", fontsize=7, labelpad=1)
    plt.ylabel("Phase (mrad)", fontsize=7, labelpad=1)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.grid(True, which="both", ls="--", alpha=0.18)

    if kk_line:
        ax = plt.gca()
        ax.text(
            0.01, 0.99, kk_line,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=6
        )

    plt.tight_layout(pad=0.2)
    plt.savefig(outpath, dpi=PLOT_DPI)
    plt.close()


def build_all_spectrum_pngs(raw_data: pd.DataFrame, features: pd.DataFrame, class_name: str, out_dir: Path, uid_meta: pd.DataFrame):
    sub = features[features["pred_final"] == class_name].copy()
    if sub.empty:
        return sub

    for stale_png in out_dir.glob(f"{class_name}__*.png"):
        stale_png.unlink(missing_ok=True)

    sub = sub.merge(uid_meta, on="UniqueID", how="left")
    groups = raw_data.groupby("UniqueID")

    png_paths = []
    missing = 0

    def _fmt(x, fmt):
        try:
            return f"{float(x):{fmt}}" if np.isfinite(x) else "NA"
        except Exception:
            return "NA"

    for r in sub.itertuples(index=False):
        uid = r.UniqueID
        png = out_dir / f"{class_name}__{safe_filename(uid)}.png"

        kk_status = getattr(r, "kk_status", "KK_NA")
        kk_mu  = getattr(r, "kk_mu", np.nan)
        kk_chi = getattr(r, "kk_pseudo_chi2", np.nan)
        kk_rc  = getattr(r, "kk_num_rc", np.nan)

        try:
            if (not np.isfinite(kk_mu)) or (not np.isfinite(kk_chi)) or (not np.isfinite(kk_rc)) or (int(kk_rc) < 0):
                kk_status = "KK_NA"
        except Exception:
            kk_status = "KK_NA"

        kk_line = f"{kk_status} | μ={_fmt(kk_mu, '.2f')} χ²={_fmt(kk_chi, '.2g')} RC={int(kk_rc) if np.isfinite(kk_rc) else 'NA'}"

        if not png.exists():
            if uid not in groups.indices:
                missing += 1
                png_paths.append("")
                continue
            plot_spectrum_png(groups.get_group(uid), png, kk_line=kk_line)

        png_paths.append(str(png))

    if missing > 0:
        print(f"[WARN] Missing raw curves for {missing} UniqueIDs in class '{class_name}'")

    sub["png_path"] = png_paths
    return sub

    sub = sub.merge(uid_meta, on="UniqueID", how="left")
    groups = raw_data.groupby("UniqueID")

    png_paths = []
    missing = 0

    for r in sub.itertuples(index=False):
        uid = r.UniqueID
        png = out_dir / f"{class_name}__{safe_filename(uid)}.png"

        if not png.exists():
            if uid not in groups.indices:
                missing += 1
                png_paths.append("")
                continue
            plot_spectrum_png(groups.get_group(uid), png)

        png_paths.append(str(png))

    if missing > 0:
        print(f"[WARN] Missing raw curves for {missing} UniqueIDs in class '{class_name}'")

    sub["png_path"] = png_paths
    return sub


# ==============================================================================
# PPT HELPERS
# ==============================================================================

def ppt_add_title_top(slide, text: str, class_name: str = "default") -> None:
    theme = _ppt_theme(class_name)

    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.35), Inches(0.10), Inches(12.55), Inches(0.045)
    )
    accent.fill.solid()
    accent.fill.fore_color.rgb = theme["accent"]
    accent.line.fill.background()

    box = slide.shapes.add_textbox(Inches(0.35), Inches(0.17), Inches(12.2), Inches(0.32))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    _apply_font(p, 18, bold=True, color=theme["accent_dark"], align=PP_ALIGN.LEFT)

    divider = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.35), Inches(0.51), Inches(12.55), Inches(0.012)
    )
    divider.fill.solid()
    divider.fill.fore_color.rgb = RGBColor(221, 224, 228)
    divider.line.fill.background()

def build_gallery_deck(
    class_name: str,
    sub_with_pngs: pd.DataFrame,
    cm_path: Path,
    thr_path: Path,
    out_pptx: Path
) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    today = datetime.date.today().strftime("%Y-%m-%d")

    total = len(sub_with_pngs)
    if total == 0:
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        slide.shapes.title.text = f"SIP QC – FINAL {class_name.upper()}"
        slide.placeholders[1].text = "No items found for this class."
        prs.save(out_pptx)
        return

    # Sort by relevance
    if class_name == "good":
        sub_with_pngs = sub_with_pngs.sort_values("prob_good", ascending=False)
    elif class_name == "bad":
        sub_with_pngs = sub_with_pngs.sort_values("prob_bad", ascending=False)
    else:
        sub_with_pngs = sub_with_pngs.sort_values("prob_max", ascending=True)

    best_t = float(sub_with_pngs["good_threshold_used"].dropna().iloc[0]) if "good_threshold_used" in sub_with_pngs.columns else 0.5


    # Title slide (reference template matched to the provided GOOD cover)
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    theme = _ppt_theme(class_name)

    card_left = Inches(0.20)
    card_top = Inches(0.72)
    card_w = Inches(12.93)
    card_h = Inches(3.03)

    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, card_left, card_top, card_w, card_h)
    card.fill.solid()
    card.fill.fore_color.rgb = RGBColor(250, 250, 250)
    card.line.color.rgb = RGBColor(205, 210, 215)
    card.line.width = Pt(1.0)

    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, card_left, card_top, card_w, Inches(0.045))
    accent.fill.solid()
    accent.fill.fore_color.rgb = theme["accent"]
    accent.line.fill.background()

    title_box = slide.shapes.add_textbox(Inches(0.45), Inches(1.06), Inches(12.2), Inches(0.48))
    tf = title_box.text_frame
    tf.clear()
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = f"SIP QC – FINAL {class_name.upper()}"
    _apply_font(p, 21, bold=True, color=theme["accent_dark"], align=PP_ALIGN.CENTER)

    summary_box = slide.shapes.add_textbox(Inches(1.55), Inches(1.54), Inches(10.2), Inches(0.278))
    tf = summary_box.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = (
        f"SIP ML Tool   •   CatBoost v4 + rules + KK/log-F   •   Generated {today}   •   "
        f"n={total}   •   p(good) threshold={best_t:.3f}"
    )
    _apply_font(p, 8.8, color=RGBColor(70, 70, 70), align=PP_ALIGN.CENTER)

    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.45), Inches(1.88), Inches(12.2), Inches(0.015))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(215, 218, 222)
    line.line.fill.background()

    hdr = slide.shapes.add_textbox(Inches(0.45), Inches(1.94), Inches(12.2), Inches(0.24))
    tfh = hdr.text_frame
    tfh.clear()
    ph = tfh.paragraphs[0]
    ph.text = "Caption legend"
    _apply_font(ph, 11.5, bold=True, color=RGBColor(45, 45, 45), align=PP_ALIGN.CENTER)

    left = slide.shapes.add_textbox(Inches(1.55), Inches(2.224), Inches(4.55), Inches(0.776))
    tfl = left.text_frame
    tfl.clear()
    tfl.word_wrap = True
    bullets_l = [
        "ID line: UniqueID / spectrum name",
        "from=… : source bucket (Field/HPE + subfolder)",
        "pG / pB : model probabilities (GOOD / BAD)",
        "max : model confidence = max(pG, pB)",
    ]
    for j, b in enumerate(bullets_l):
        pp = tfl.paragraphs[0] if j == 0 else tfl.add_paragraph()
        pp.text = b
        _apply_font(pp, 9.4, color=RGBColor(40, 40, 40), align=PP_ALIGN.LEFT)
        pp.space_after = Pt(0.8)

    right = slide.shapes.add_textbox(Inches(7.05), Inches(2.218), Inches(4.95), Inches(0.776))
    tfr = right.text_frame
    tfr.clear()
    tfr.word_wrap = True
    bullets_r = [
        "rule tags: why the spectrum moved to REVIEW / BAD",
        "KK_OK / REVIEW / BAD : KK/log-F consistency status",
        "μ : fit-quality indicator (lower is worse)",
        "χ² : pseudo chi-square  •  RC : selected RC count",
    ]
    for j, b in enumerate(bullets_r):
        pp = tfr.paragraphs[0] if j == 0 else tfr.add_paragraph()
        pp.text = b
        _apply_font(pp, 9.4, color=RGBColor(40, 40, 40), align=PP_ALIGN.LEFT)
        pp.space_after = Pt(0.8)

    note = slide.shapes.add_textbox(Inches(1.65), Inches(3.115), Inches(10.0), Inches(0.278))
    tfn = note.text_frame
    tfn.clear()
    pn = tfn.paragraphs[0]
    pn.text = "Use the caption and curve together: GOOD = trusted, REVIEW = inspect, BAD = structurally unstable / nonphysical."
    _apply_font(pn, 8.2, color=RGBColor(85, 85, 85), align=PP_ALIGN.CENTER)

    # Sanity slide
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)
    ppt_add_title_top(slide, "Model sanity-check (true_label rows)", class_name)
    slide.shapes.add_picture(str(cm_path), Inches(0.8), Inches(0.85), height=Inches(4.3))
    slide.shapes.add_picture(str(thr_path), Inches(6.7), Inches(0.85), height=Inches(4.3))

    # Gallery slides
    n_slides = math.ceil(total / SPECTRA_PER_SLIDE)

    margin_left = 0.25
    margin_top = 0.55
    bottom_margin = 0.20

    slide_w = 13.33
    slide_h = 7.5

    cell_w = (slide_w - 2 * margin_left) / GALLERY_COLS
    cell_h = (slide_h - margin_top - bottom_margin) / GALLERY_ROWS

    cap_h = 0.84
    img_pad_x = 0.05
    img_pad_y = cap_h + 0.05

    img_w = cell_w - 2 * img_pad_x
    img_h = cell_h - img_pad_y - 0.05

    rows = sub_with_pngs.to_dict(orient="records")

    for sidx in range(n_slides):
        slide = prs.slides.add_slide(blank)

        start = sidx * SPECTRA_PER_SLIDE
        end = min((sidx + 1) * SPECTRA_PER_SLIDE, total)
        ppt_add_title_top(slide, f"{class_name.upper()} spectra gallery ({start+1}–{end} of {total})", class_name)

        chunk = rows[start:end]
        for i, row in enumerate(chunk):
            rr = i // GALLERY_COLS
            cc = i % GALLERY_COLS

            left = margin_left + cc * cell_w
            top = margin_top + rr * cell_h

            uid = row.get("UniqueID", "")
            pg = row.get("prob_good", np.nan)
            pb = row.get("prob_bad", np.nan)
            pm = row.get("prob_max", np.nan)
            pulled = row.get("pulled_from", "UNKNOWN")

            # rule summary for caption
            r_neg_lt_1k = row.get("rule_neg_lt_1k", 0)
            r_neg_1k_10k = row.get("rule_neg_1k_10k", 0)
            r_jrun = row.get("rule_jagged_run", 0)
            r_iso_allowed = row.get("rule_isolated_spikes_allowed", 0)
            r_iso_cnt = row.get("rule_isolated_spikes", 0)
            r_high = row.get("rule_high_phase_gt_1k", False)
            r_combo = row.get("rule_combo_bad", False)

            # patch v2 flags (if present)
            osc = row.get("rule_oscillatory_jagged", 0)
            tail = row.get("rule_tail_instability", 0)
            patch_reason = row.get("rule_patch_v2_reason", "")

            rule_bits = []
            if r_neg_lt_1k: rule_bits.append("neg<1k(BAD)")
            if r_neg_1k_10k: rule_bits.append("neg1k-10k(REV)")
            if r_jrun: rule_bits.append("dP-run(REV)")
            if r_iso_allowed: rule_bits.append(f"isol-spikes={r_iso_cnt}(REV)")
            if r_high: rule_bits.append("highP>1k(REV)")
            if r_combo: rule_bits.append("COMBO(BAD)")
            if osc: rule_bits.append("osc(REV)")
            if tail: rule_bits.append("tail(REV/BAD)")
            if patch_reason and patch_reason != "base": rule_bits.append(f"patch={patch_reason}")
            rule_txt = " | ".join(rule_bits) if rule_bits else "no-rule"

            def _caption_clip(text, limit):
                text = re.sub(r"\s+", " ", str(text or "")).strip()
                return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"

            pulled_short = _caption_clip(pulled, 58)
            rule_short = _caption_clip(rule_txt, 78)

            # Caption box
            cap = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(cell_w), Inches(cap_h))
            tf = cap.text_frame
            tf.clear()
            tf.word_wrap = True
            tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
            tf.margin_left = Inches(0.02)
            tf.margin_right = Inches(0.02)
            tf.margin_top = Inches(0.01)
            tf.margin_bottom = Inches(0.01)

            p1 = tf.paragraphs[0]
            p1.text = uid
            _apply_font(p1, CAPTION_FONT_UID, bold=True, color=RGBColor(35, 35, 35), align=PP_ALIGN.LEFT)

            p2 = tf.add_paragraph()
            p2.text = f"from={pulled_short} | pG={pg:.3f} pB={pb:.3f} max={pm:.3f}"
            _apply_font(p2, CAPTION_FONT_META, color=RGBColor(70, 70, 70), align=PP_ALIGN.LEFT)

            p3 = tf.add_paragraph()
            p3.text = rule_short
            _apply_font(p3, CAPTION_FONT_META, color=RGBColor(85, 85, 85), align=PP_ALIGN.LEFT)

            def _fmt(x, fmt):
                try:
                    return f"{float(x):{fmt}}" if np.isfinite(x) else "NA"
                except Exception:
                    return "NA"

            kk_status = row.get("kk_status", "KK_NA")
            kk_mu  = row.get("kk_mu", np.nan)
            kk_chi = row.get("kk_pseudo_chi2", np.nan)
            kk_rc  = row.get("kk_num_rc", np.nan)

            try:
                if (not np.isfinite(kk_mu)) or (not np.isfinite(kk_chi)) or (not np.isfinite(kk_rc)) or (int(kk_rc) < 0):
                    kk_status = "KK_NA"
            except Exception:
                kk_status = "KK_NA"

            p4 = tf.add_paragraph()
            p4.text = f"{kk_status} | μ={_fmt(kk_mu, '.2f')} χ²={_fmt(kk_chi, '.2g')} RC={int(kk_rc) if np.isfinite(kk_rc) else 'NA'}"
            _apply_font(p4, CAPTION_FONT_META, color=RGBColor(85, 85, 85), align=PP_ALIGN.LEFT)

            # Image
            png_path = row.get("png_path", "")
            img_left = Inches(left + img_pad_x)
            img_top = Inches(top + img_pad_y)

            if png_path and Path(png_path).exists():
                if FORCE_STRETCH_TO_CELL:
                    slide.shapes.add_picture(
                        png_path, img_left, img_top,
                        width=Inches(img_w), height=Inches(img_h)
                    )
                else:
                    slide.shapes.add_picture(png_path, img_left, img_top, width=Inches(img_w))
            else:
                rect = slide.shapes.add_shape(1, img_left, img_top, Inches(img_w), Inches(img_h))
                rect.fill.solid()
                rect.fill.fore_color.rgb = RGBColor(242, 242, 242)
                rect.line.color.rgb = RGBColor(180, 180, 180)
                rtf = rect.text_frame
                rtf.text = "Missing raw curve\n(no CSV match found)"
                rtf.paragraphs[0].font.size = Pt(10)

    prs.save(out_pptx)
    print(f"[OK] Saved: {out_pptx}")
