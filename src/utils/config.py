# src/utils/config.py

import os

# ─── Path Configuration ────────────────────────────────────────────────────────

BASE_DIR     = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))

LIST_DIR     = os.path.join(BASE_DIR, 'data', 'raw', 'list')
EXTERNAL_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'external')
TWEETS_DIR   = os.path.join(BASE_DIR, 'data', 'raw', 'tweets')

# ─── Scraper Configuration ─────────────────────────────────────────────────────

NITTER_BASE  = "http://217.154.241.195:8935"
