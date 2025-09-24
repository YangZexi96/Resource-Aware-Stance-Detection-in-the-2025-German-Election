#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_dataset_summary_hardcoded.py
---------------------------------
Hardcoded version: works with semicolon-separated CSVs.
Generates dataset summary with tweet counts, label distribution,
Target mismatch and Ironie for test set, and token length stats.
"""

import os
import pandas as pd
import numpy as np

# ============================
# Hardcoded paths (adjust if needed)
# ============================
TOTAL_PATH = r"E:\Final_Github_1\MasterThesis_final\data\processed\annotation\annotation_dataset\annotation_dataset_labeled_with_name2.csv"
SPLITS_ROOT = r"E:\Final_Github_1\MasterThesis_final\data\processed\training and test\final"
OUT_PREFIX = r"E:\Final_Github_1\MasterThesis_final\data\processed\training and test\final\dataset_summary_table"
# ============================


GER2ENG = {"Zustimmung": "Favor", "Ablehnung": "Against", "Neutral": "Neutral",
           "Favor": "Favor", "Against": "Against"}

def safe_read_csv(path: str) -> pd.DataFrame:
    """CSV einlesen mit Semikolon-Trennung (für deine Files)."""
    return pd.read_csv(path, encoding="utf-8-sig", sep=";", on_bad_lines="skip")

def normalize_label(x):
    if pd.isna(x):
        return x
    return GER2ENG.get(str(x).strip(), str(x).strip())

def count_tokens(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(text.strip().split())

def class_counts_and_perc(df: pd.DataFrame, label_col: str):
    tmp = df[label_col].map(normalize_label)
    counts = tmp.value_counts(dropna=False)
    total = len(df)
    out = {}
    for k in ["Favor","Against","Neutral"]:
        c = int(counts.get(k, 0))
        p = (c / total * 100.0) if total else 0.0
        out[k] = (c, p)
    return out

def length_stats(df: pd.DataFrame):
    lens = df["text"].astype(str).apply(count_tokens)
    return float(np.mean(lens)), int(np.median(lens))

def format_count_pct(count: int, pct: float, places: int = 1) -> str:
    return f"{count:,} ({pct:.{places}f} %)".replace(",", ".")

def compute_target_mismatch(test_joined: pd.DataFrame):
    def norm(x): return "" if pd.isna(x) else str(x).strip().casefold()
    name_split  = test_joined["name"].apply(norm) if "name" in test_joined.columns else pd.Series([""]*len(test_joined))
    name_total  = test_joined["Name"].apply(norm)  if "Name" in test_joined.columns else pd.Series([""]*len(test_joined))
    party_total = test_joined["party"].apply(norm) if "party" in test_joined.columns else pd.Series([""]*len(test_joined))
    handle_tot  = test_joined["handle"].apply(norm) if "handle" in test_joined.columns else pd.Series([""]*len(test_joined))
    match = (name_split == name_total) | (name_split == party_total) | (name_split == handle_tot)
    mismatches = int((~match).sum())
    pct = (mismatches / max(len(test_joined),1)) * 100.0
    return mismatches, pct

def build_table(total_df, train, val, test):
    # ---------- Label column detection ----------
    label_col = None
    for cand in ["Stance", "stance", "label", "Label"]:
        if cand in total_df.columns:
            label_col = cand
            break
    if label_col is None:
        raise KeyError(f"Total dataset lacks a stance/label column. Found columns: {list(total_df.columns)}")

    total_df["__label__"] = total_df[label_col].map(normalize_label)

    # ---------- Counts ----------
    n_total, n_train, n_val, n_test = len(total_df), len(train), len(val), len(test)
    total_cc, train_cc, val_cc, test_cc = (
        class_counts_and_perc(total_df, "__label__"),
        class_counts_and_perc(train, "label"),
        class_counts_and_perc(val, "label"),
        class_counts_and_perc(test, "label"),
    )

    # ---------- Join test with total for Ironie & mismatch ----------
    test_joined = test.merge(total_df.drop_duplicates(subset=["text"]),
                             on="text", how="left", suffixes=("", "_total"))

    irony_true, irony_pct = 0, 0.0
    if "Ironie" in test_joined.columns:
        s = test_joined["Ironie"].astype(str).str.strip().str.lower()
        irony_true = int(s.isin({"true","1","yes"}).sum())
        irony_pct = (irony_true / max(len(test_joined),1)) * 100.0

    mismatches, mismatch_pct = compute_target_mismatch(test_joined)

    # ---------- Length stats ----------
    avg_total, med_total = length_stats(total_df)
    avg_train, med_train = length_stats(train)
    avg_val,   med_val   = length_stats(val)
    avg_test,  med_test  = length_stats(test)

    # ---------- Rows ----------
    rows = []
    rows.append(["Number of tweets",
                 format_count_pct(n_total, 100.0, 0),
                 format_count_pct(n_train, (n_train/max(n_total,1))*100.0, 0),
                 format_count_pct(n_val,   (n_val  /max(n_total,1))*100.0, 0),
                 format_count_pct(n_test,  (n_test /max(n_total,1))*100.0, 0)])
    def row_for(de, en):
        tot, trn, vl, tst = total_cc[en], train_cc[en], val_cc[en], test_cc[en]
        rows.append([f"{de} ({en})",
                     format_count_pct(tot[0], tot[1]),
                     format_count_pct(trn[0], trn[1]),
                     format_count_pct(vl[0],  vl[1]),
                     format_count_pct(tst[0], tst[1])])
    row_for("Zustimmung","Favor")
    row_for("Ablehnung","Against")
    row_for("Neutral","Neutral")

    rows.append(["Target mismatch","–","–","–", format_count_pct(mismatches, mismatch_pct)])
    rows.append(["Ironie","–","–","–", format_count_pct(irony_true, irony_pct)])

    rows.append(["Avg. tweet length (tokens)",
                 f"{avg_total:.1f} (–)", f"{avg_train:.1f} (–)", f"{avg_val:.1f} (–)", f"{avg_test:.1f} (–)"])
    rows.append(["Median tweet length (tokens)",
                 f"{med_total:d} (–)", f"{med_train:d} (–)", f"{med_val:d} (–)", f"{med_test:d} (–)"])
    return pd.DataFrame(rows, columns=["Metric","Total","Train","Validation","Test"])

def main():
    # ---------- Read total ----------
    total_df = safe_read_csv(TOTAL_PATH)
    if "text" not in total_df.columns and "Text" in total_df.columns:
        total_df.rename(columns={"Text":"text"}, inplace=True)

    # ---------- Read splits ----------
    train = safe_read_csv(os.path.join(SPLITS_ROOT, "train.csv"))
    val   = safe_read_csv(os.path.join(SPLITS_ROOT, "val.csv"))
    test  = safe_read_csv(os.path.join(SPLITS_ROOT, "test.csv"))
    for df in (train,val,test):
        if "text" not in df.columns and "Text" in df.columns:
            df.rename(columns={"Text":"text"}, inplace=True)
        if "label" not in df.columns and "Label" in df.columns:
            df.rename(columns={"Label":"label"}, inplace=True)
        if "name" not in df.columns and "Name" in df.columns:
            df.rename(columns={"Name":"name"}, inplace=True)

    # ---------- Build table ----------
    table = build_table(total_df, train, val, test)

    # ---------- Save ----------
    md_path = f"{OUT_PREFIX}.md"
    csv_path = f"{OUT_PREFIX}.csv"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(table.to_markdown(index=False))
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(table.to_markdown(index=False))
    print(f"\nSaved: {md_path}\nSaved: {csv_path}")

if __name__ == "__main__":
    main()
