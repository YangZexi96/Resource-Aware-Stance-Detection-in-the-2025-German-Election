import sys
from pathlib import Path

# ─── Bootstrap src/ into import path ────────────────────────────────────────────
HERE    = Path(__file__).resolve()
PROJECT = HERE.parents[2]        # …/src/1_preprocessing_pipeline → MasterThesis_final
SRC_DIR = PROJECT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import re
import os
import pandas as pd
from transformers import pipeline
from pathlib import Path as _Path

from utils.config import BASE_DIR, TWEETS_DIR

# ─── Directories ───────────────────────────────────────────────────────────────
# Raw tweets for augmentation
RAW_ROOT     = os.path.join(TWEETS_DIR, 'augmentation_dataset')
# Where cleaned output goes
PROC_ROOT    = os.path.join(
    BASE_DIR, 'data', 'processed', 'annotation', 'augmentation', 'cleaned'
)
# Subfolder for dropped rows
DELETED_ROOT = os.path.join(PROC_ROOT, '_deleted')

# ─── Constants ─────────────────────────────────────────────────────────────────
SEP            = ';'
THRESHOLD      = 0.8
MENTION_REGEX  = re.compile(r"@\w+")

# ─── Helper: strip Twitter mentions ─────────────────────────────────────────────
def strip_mentions(text: str) -> str:
    return MENTION_REGEX.sub('', text)

# ─── Initialize HF LID pipeline ────────────────────────────────────────────────
hf_pipe = pipeline(
    "text-classification",
    model="papluca/xlm-roberta-base-language-detection",
    tokenizer="cardiffnlp/twitter-xlm-roberta-base",
    top_k=None,
    truncation=True,
    max_length=128,
    batch_size=64,
    device=-1
)

def detect_german_batch(texts: list[str]) -> list[bool]:
    cleaned = [strip_mentions(t) for t in texts]
    outputs = hf_pipe(cleaned)
    results = []
    for preds in outputs:
        is_de = any(
            p["label"].lower() == "de" and p["score"] >= THRESHOLD
            for p in preds
        )
        results.append(is_de)
    return results

# ─── Main processing ───────────────────────────────────────────────────────────
def main():
    for src in _Path(RAW_ROOT).rglob("*.csv"):
        # Determine output paths
        rel      = src.relative_to(RAW_ROOT)
        out_clean = _Path(PROC_ROOT) / rel
        if out_clean.exists():
            print(f"{src} → already processed, skipping")
            continue

        # Load raw tweets
        df_orig = pd.read_csv(src, sep=SEP, encoding="utf-8",
                              on_bad_lines="skip",
                              dtype={"tweet_text": str, "username": str})
        n_initial = len(df_orig)

        # A: drop null or too-short tweets
        mask_valid = df_orig["tweet_text"].notna() & (df_orig["tweet_text"].str.len() >= 5)
        dfA        = df_orig[mask_valid]
        short_df   = df_orig[~mask_valid]

        # B: drop duplicates on username + text
        dfB        = dfA.drop_duplicates(subset=["username", "tweet_text"])
        dup_idxs   = dfA.index.difference(dfB.index)
        dup_df     = dfA.loc[dup_idxs]

        # C: detect German language
        unique_texts = dfB["tweet_text"].unique().tolist()
        flags        = detect_german_batch(unique_texts)
        german_map   = dict(zip(unique_texts, flags))

        mask_de   = dfB["tweet_text"].map(german_map)
        final_df  = dfB[mask_de].copy()
        nonde_df  = dfB[~mask_de]

        n_final = len(final_df)

        # Save cleaned tweets
        out_clean.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(out_clean, index=False, sep=SEP, encoding="utf-8")

        # Save dropped subsets by reason
        for reason, df_drop in [
            ("short_text", short_df),
            ("duplicates", dup_df),
            ("non_german", nonde_df),
        ]:
            out_del = _Path(DELETED_ROOT) / reason / rel
            out_del.parent.mkdir(parents=True, exist_ok=True)
            df_drop.to_csv(out_del, index=False, sep=SEP, encoding="utf-8")

        # Log summary
        print(
            f"{src} → kept {n_final:,}/{n_initial:,}   "
            f"short {len(short_df):,}   dup {len(dup_df):,}   non_de {len(nonde_df):,}"
        )

if __name__ == "__main__":
    main()
