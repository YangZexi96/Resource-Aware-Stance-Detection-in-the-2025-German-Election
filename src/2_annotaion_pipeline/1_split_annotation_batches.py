#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_annotation_batches.py

Creates a balanced and deduplicated annotation dataset for manual labeling.

Steps:
1) Loads the completed annotation_results.csv
2) Removes duplicates by tweet_id
3) Samples exactly TARGET_PER examples of each sentiment
4) Shuffles the combined set
5) Writes one full CSV with all samples (label column empty)
6) Splits into 18 batches of 500 rows
"""

import sys
from pathlib import Path
import os
import pandas as pd

# ─── Bootstrap src/ into import path ────────────────────────────────────────────
HERE    = Path(__file__).resolve()
PROJECT = HERE.parents[1]
SRC_DIR = PROJECT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ─── Configuration ─────────────────────────────────────────────────────────────
INPUT_RESULTS = r"E:\Final_Github_1\MasterThesis_final\data\processed\annotation\annotation_results.csv"
OUTPUT_DIR    = r"E:\Final_Github_1\MasterThesis_final\data\processed\annotation\annotation_dataset"

TARGET_PER  = 1500          # Number of examples per sentiment
CHUNK_SIZE  = 500           # Each batch will contain 500 samples
NUM_CHUNKS  = 9           # Total batches = 9000 / 500
LABEL_COL   = 'gbert_label'
SCORE_COL   = 'gbert_score'

def main():
    # 1) Load all predictions
    df = pd.read_csv(INPUT_RESULTS, sep=';', encoding='utf-8')

    # 2) Remove duplicates by tweet_id
    df = df.drop_duplicates(subset="tweet_id", keep="first")

    # 3) Sample exactly TARGET_PER of each label
    samples = []
    for label in ['positive', 'neutral', 'negative']:
        subset = df[df[LABEL_COL] == label]
        if len(subset) < TARGET_PER:
            raise ValueError(f"Not enough '{label}' examples: found {len(subset)}, need {TARGET_PER}")
        sampled = subset.sample(n=TARGET_PER, random_state=42)
        samples.append(sampled)

    # 4) Combine and shuffle
    balanced = pd.concat(samples).sample(frac=1, random_state=42).reset_index(drop=True)

    # 5) Ensure tweet_id uniqueness
    if balanced["tweet_id"].duplicated().any():
        raise ValueError("❌ Duplicate tweet_id detected after sampling.")
    assert len(balanced["tweet_id"].unique()) == len(balanced), "❌ Tweet ID collision after sampling."

    # 6) Prepare final DataFrame
    full_df = (
        balanced
        .drop(columns=[LABEL_COL, SCORE_COL], errors='ignore')
        .rename(columns={'tweet_text': 'text', 'candidate': 'handle'})
    )

    required_cols = [
        'text', 'handle', 'party', 'timestamp',
        'quoted_tweet_text', 'quoted_tweet_author', 'tweet_id'
    ]
    for col in required_cols:
        if col not in full_df.columns:
            raise KeyError(f"Required column '{col}' not found in the input CSV.")

    # Add empty 'label' column after 'text'
    full_df.insert(1, 'label', '')

    # Reorder columns
    full_df = full_df[
        ['text', 'label', 'handle', 'party', 'timestamp',
         'quoted_tweet_text', 'quoted_tweet_author', 'tweet_id']
    ]

    # 7) Write full dataset
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_path = os.path.join(OUTPUT_DIR, "annotation_dataset_all.csv")
    full_df.to_csv(full_path, sep=';', index=False, encoding='utf-8-sig')
    print(f"✅ Wrote full dataset of {len(full_df)} rows to {full_path}")

    # 8) Split into NUM_CHUNKS batches
    total = len(full_df)
    expected_total = NUM_CHUNKS * CHUNK_SIZE
    if total != expected_total:
        raise ValueError(f"Expected {expected_total} rows, found {total} — mismatch.")

    for i in range(NUM_CHUNKS):
        start = i * CHUNK_SIZE
        batch_df = full_df.iloc[start:start + CHUNK_SIZE]
        csv_path = os.path.join(OUTPUT_DIR, f"annotation_dataset_{i+1}.csv")
        batch_df.to_csv(csv_path, sep=';', index=False, encoding='utf-8-sig')
        print(f"📦 Wrote batch {i+1} with {len(batch_df)} rows to {csv_path}")

    print(f"🎉 Done: created full dataset + {NUM_CHUNKS} batches in '{OUTPUT_DIR}' with no duplicates.")

if __name__ == "__main__":
    main()
