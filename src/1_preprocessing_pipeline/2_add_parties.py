#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_map_fixed_encoding.py

1) Load the merged master CSV (augmentation master) with UTF-8 or UTF-8-SIG,
2) Clean whitespace (preserving emojis),
3) Generate stable tweet_id’s,
4) Load candidate→party mapping from raw list,
5) Map each candidate to their party,
6) Write a UTF-8-SIG CSV for the mapped dataset.
"""

import sys
from pathlib import Path

# ─── Bootstrap src/ into import path ────────────────────────────────────────────
HERE    = Path(__file__).resolve()
PROJECT = HERE.parents[2]        # …/src/annotation_pipeline → MasterThesis_final
SRC_DIR = PROJECT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os
import re
import uuid
import pandas as pd

from utils.config import BASE_DIR, EXTERNAL_DIR

# ─── Paths ──────────────────────────────────────────────────────────────────────
# Master CSV produced by build_master.py
MASTER_INPUT = os.path.join(
    BASE_DIR, 'data', 'processed', 'annotation', 'augmentation', 'augmentation_master.csv'
)
# Candidate→party mapping maintained in raw/list
MAPPING_CSV  = os.path.join(EXTERNAL_DIR, 'candidate_to_party_mapping.csv')
# Output of this step (mapped file)
OUTPUT_FILE  = os.path.join(
    BASE_DIR, 'data', 'processed', 'annotation', 'augmentation', 'augmentation_mapped.csv'
)

# ─── Helpers ───────────────────────────────────────────────────────────────────
def clean_whitespace(s: str) -> str:
    """Clean up whitespace (preserve emojis)."""
    if not isinstance(s, str):
        return ""
    t = re.sub(r"[\r\n\t]+", " ", s)
    return re.sub(r" {2,}", " ", t).strip()

def generate_id(row) -> str:
    """Generate stable UUID5 from row index and username."""
    base = f"{row.name}-{row.get('username','')}"
    return uuid.uuid5(uuid.NAMESPACE_OID, base).hex

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1) Read master CSV (try UTF-8-SIG, then UTF-8)
    try:
        df = pd.read_csv(MASTER_INPUT, sep=";", engine="python",
                         on_bad_lines="skip", encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(MASTER_INPUT, sep=";", engine="python",
                         on_bad_lines="skip", encoding="utf-8")

    # 2) Clean all string columns
    text_cols = df.select_dtypes(include='object').columns
    for col in text_cols:
        df[col] = df[col].fillna("").apply(clean_whitespace)

    # 3) Generate stable tweet_id
    df['tweet_id'] = df.apply(generate_id, axis=1)

    # 4) Load candidate→party mapping (semicolon or comma)
    try:
        party_df = pd.read_csv(MAPPING_CSV, sep=";", dtype=str,
                               on_bad_lines="skip", encoding="utf-8")
    except Exception:
        party_df = pd.read_csv(MAPPING_CSV, sep=",", dtype=str,
                               on_bad_lines="skip", encoding="utf-8")

    # Expect columns 'username' and 'party'
    if not {'username','party'} <= set(party_df.columns):
        raise KeyError(f"Mapping file must have 'username' and 'party' columns: {MAPPING_CSV}")

    party_map = party_df.set_index('username')['party'].to_dict()

    # 5) Map candidate→party
    df['party'] = df['candidate'].map(party_map).fillna('UNKNOWN')

    # 6) Select and write output
    keep = ['tweet_id', 'username', 'candidate', 'party',
            'timestamp', 'tweet_text', 'quoted_tweet_text', 'quoted_tweet_author']
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    out.to_csv(OUTPUT_FILE, sep=";", index=False, encoding="utf-8-sig")
    print(f"Preprocessed {len(out)} tweets → {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
