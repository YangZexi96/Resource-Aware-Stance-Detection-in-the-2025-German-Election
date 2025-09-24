import sys
from pathlib import Path

# ─── Bootstrap src/ into import path ────────────────────────────────────────────
HERE    = Path(__file__).resolve()
PROJECT = HERE.parents[2]        # …/src/3_count_hashtags.py → MasterThesis/
SRC_DIR = PROJECT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os
import pandas as pd
from collections import Counter
from utils.config import BASE_DIR, TWEETS_DIR

# ─── Input & Output directories ────────────────────────────────────────────────
RAW_MIRROR_DIR = os.path.join(TWEETS_DIR, 'mirror_dataset')
OUTPUT_DIR     = os.path.join(BASE_DIR, 'data', 'processed' , 'augmentation', 'hashtags')

# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    # aggregate counters per month
    counts_by_month: dict[str, Counter] = {}

    # iterate over each handle folder
    for handle in os.listdir(RAW_MIRROR_DIR):
        handle_path = os.path.join(RAW_MIRROR_DIR, handle)
        if not os.path.isdir(handle_path):
            continue

        # process each day's CSV for this handle
        for fname in os.listdir(handle_path):
            if not fname.endswith('.csv'):
                continue
            # derive month (YYYY-MM) from filename like "2024-09-04_handle.csv"
            month = fname[:7]
            counts_by_month.setdefault(month, Counter())

            fpath = os.path.join(handle_path, fname)
            try:
                df = pd.read_csv(fpath, sep=';', engine='python')
            except Exception as e:
                print(f"Warning: could not read {fpath}: {e}")
                continue

            if 'hashtags' not in df.columns:
                continue

            # count each hashtag occurrence
            for cell in df['hashtags'].dropna():
                # split on semicolon, strip whitespace
                tags = [t.strip() for t in cell.split(';') if t.strip()]
                for tag in tags:
                    counts_by_month[month][tag] += 1

    # ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # write monthly CSVs
    for month, counter in sorted(counts_by_month.items()):
        out_path = os.path.join(OUTPUT_DIR, f'hashtag_counts_{month}.csv')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('hashtag;count\n')
            for tag, count in counter.most_common():
                f.write(f'{tag};{count}\n')
        print(f"Wrote {len(counter)} hashtags for {month} to {out_path}")

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

# ─── Bootstrap src/ into import path ────────────────────────────────────────────
HERE    = Path(__file__).resolve()
PROJECT = HERE.parents[2]        # …/src/3_count_hashtags.py → MasterThesis/
SRC_DIR = PROJECT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os
import pandas as pd
from collections import Counter
from utils.config import BASE_DIR, TWEETS_DIR

# ─── Input & Output directories ────────────────────────────────────────────────
RAW_MIRROR_DIR = os.path.join(TWEETS_DIR, 'mirror_dataset')
OUTPUT_DIR     = os.path.join(
    BASE_DIR,
    'data', 'processed', 'augmentation', 'hashtags'
)

# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    # aggregate counters per month
    counts_by_month: dict[str, Counter] = {}

    # iterate over each handle folder
    for handle in os.listdir(RAW_MIRROR_DIR):
        handle_path = os.path.join(RAW_MIRROR_DIR, handle)
        if not os.path.isdir(handle_path):
            continue

        # process each day's CSV for this handle
        for fname in os.listdir(handle_path):
            if not fname.endswith('.csv'):
                continue
            # derive month (YYYY-MM) from filename like "2024-09-04_handle.csv"
            month = fname[:7]
            counts_by_month.setdefault(month, Counter())

            fpath = os.path.join(handle_path, fname)
            try:
                df = pd.read_csv(fpath, sep=';', engine='python')
            except Exception as e:
                print(f"Warning: could not read {fpath}: {e}")
                continue

            if 'hashtags' not in df.columns:
                continue

            # count each hashtag occurrence
            for cell in df['hashtags'].dropna():
                # split on semicolon, strip whitespace
                tags = [t.strip() for t in cell.split(';') if t.strip()]
                for tag in tags:
                    counts_by_month[month][tag] += 1

    # ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # write monthly CSVs
    for month, counter in sorted(counts_by_month.items()):
        out_path = os.path.join(OUTPUT_DIR, f'hashtag_counts_{month}.csv')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('hashtag;count\n')
            for tag, count in counter.most_common():
                f.write(f'{tag};{count}\n')
        print(f"Wrote {len(counter)} hashtags for {month} to {out_path}")

if __name__ == '__main__':
    main()
