#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample_for_annotation.py

Incrementally evaluates mirror_prepped.csv with
a German sentiment model, logging every prediction as it goes,
and stopping once 1500 of each positive/neutral/negative have been collected.

Progress is saved to annotation_results.csv in OUTPUT_DIR,
flushing every 50 tweets, so you can safely restart.
"""

import sys
from pathlib import Path

# ─── Bootstrap src/ into import path ────────────────────────────────────────────
HERE    = Path(__file__).resolve()
PROJECT = HERE.parents[1]        # …/src → MasterThesis_final
SRC_DIR = PROJECT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os
import csv
import logging
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline
)

from utils.config import BASE_DIR

# ─── Paths ──────────────────────────────────────────────────────────────────────
# Input: cleaned & mapped tweets for mirror pipeline
INPUT_FILE   = os.path.join(
    BASE_DIR,
    'data', 'processed', 'preprocessing', 'mirror', 'mirror_prepped.csv'
)
# Output directory for annotation results
OUTPUT_DIR   = os.path.join(
    BASE_DIR,
    'data', 'processed', 'annotation'
)
# HuggingFace token for private models
KEY_PATH     = os.path.join(
    BASE_DIR,
    'keys', 'huggingface_key.txt'
)
# Results CSV
RESULTS_CSV  = os.path.join(OUTPUT_DIR, "annotation_results.csv")

# ─── Sampling configuration ─────────────────────────────────────────────────────
MODEL_NAME     = "oliverguhr/german-sentiment-bert"
BATCH_SIZE     = 64
TARGET_PER     = 3000
LABEL_ORDER    = ["positive", "neutral", "negative"]
FLUSH_INTERVAL = 50  # flush every N tweets

# ─── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def load_token(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_pipeline(model_name: str, token: str) -> TextClassificationPipeline:
    logger.info("Loading model '%s'", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=token)
    device    = 0 if torch.cuda.is_available() else -1
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=BATCH_SIZE,
        return_all_scores=False
    )
    logger.info("Model loaded on %s", "GPU" if device>=0 else "CPU")
    return pipe

def load_existing_counts(results_path: str) -> dict:
    """Count how many of each label we already have."""
    if not os.path.exists(results_path):
        return {lab: 0 for lab in LABEL_ORDER}
    df = pd.read_csv(results_path, sep=';', usecols=['gbert_label'], encoding='utf-8-sig')
    counts = df['gbert_label'].value_counts().to_dict()
    return {lab: counts.get(lab, 0) for lab in LABEL_ORDER}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("Loading tweets from %s", INPUT_FILE)
    try:
        df = pd.read_csv(INPUT_FILE, sep=';', encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_FILE, sep=';', encoding='utf-8')
    logger.info("Total tweets available: %d", len(df))

    # shuffle so sampling is random
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # load HF pipeline
    hf_token = load_token(KEY_PATH)
    pipe     = load_pipeline(MODEL_NAME, hf_token)

    # resume previous counts if any
    counts = load_existing_counts(RESULTS_CSV)
    logger.info("Starting with counts: %s", counts)

    needs_header = not os.path.exists(RESULTS_CSV)
    fout = open(RESULTS_CSV, 'a', newline='', encoding='utf-8-sig')
    writer = csv.writer(fout, delimiter=';')
    if needs_header:
        writer.writerow([
            'tweet_id','username','candidate','party','timestamp',
            'tweet_text','gbert_label','gbert_score'
        ])

    logged = 0
    total = len(df)
    logger.info("Sampling until each label has %d examples", TARGET_PER)

    for start in tqdm(range(0, total, BATCH_SIZE), desc="Sampling"):
        batch_texts = df['tweet_text'].iloc[start:start+BATCH_SIZE].fillna('').tolist()
        batch_rows  = df.iloc[start:start+BATCH_SIZE]

        preds = pipe(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=512
        )
        for i, pred in enumerate(preds):
            if all(counts[l] >= TARGET_PER for l in LABEL_ORDER):
                break
            lab = pred['label']
            row = batch_rows.iloc[i]
            writer.writerow([
                row['tweet_id'],
                row['username'],
                row['candidate'],
                row['party'],
                row['timestamp'],
                row['tweet_text'],
                lab,
                f"{pred['score']:.4f}"
            ])
            counts[lab] += 1
            logged += 1
            if counts[lab] == TARGET_PER:
                logger.info("Reached %d samples for %s", TARGET_PER, lab)
            if logged % FLUSH_INTERVAL == 0:
                fout.flush()

        if all(counts[l] >= TARGET_PER for l in LABEL_ORDER):
            logger.info("All targets reached: %s", counts)
            break

    fout.flush()
    fout.close()
    logger.info("Done sampling. Final counts: %s", counts)
    logger.info("Results saved to %s", RESULTS_CSV)

if __name__ == "__main__":
    main()
