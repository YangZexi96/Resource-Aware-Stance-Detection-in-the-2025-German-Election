import sys
from pathlib import Path

# ─── Bootstrap src/ into import path ────────────────────────────────────────────
HERE    = Path(__file__).resolve()
PROJECT = HERE.parents[2]        # …/src/0_scraping_pipeline → MasterThesis_final
SRC_DIR = PROJECT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os
import time
import logging
import pandas as pd
import calendar
import re
from datetime import datetime, date, timedelta
from gdeltdoc import GdeltDoc, Filters
from requests.exceptions import HTTPError

from utils.config import BASE_DIR

# ─── Configuration ─────────────────────────────────────────────────────────────
END_DATE       = '2024-10-31'
DOMAINS        = [
    'tagesschau.de', 'heute.de', 'deutschlandfunk.de',
    'sueddeutsche.de', 'faz.net', 'spiegel.de',
    'zeit.de', 'handelsblatt.com', 'welt.de', 'tagesspiegel.de'
]

# where your per-month keyword files live
KEYWORD_DIR    = os.path.join(
    BASE_DIR, 'data', 'processed', 'augmentation', 'keywords'
)
# where to write daily counts per keyword
OUTPUT_DIR     = os.path.join(
    BASE_DIR, 'data', 'processed', 'augmentation', 'gdelt_counts', 'daily_keyword_count'
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# initialize GDELT client
gd = GdeltDoc()

def count_for_day_kw_dom(date_obj: date, keyword: str, domain: str, retries: int = 5) -> int:
    """
    Try up to `retries` times. On HTTP 429, back off exponentially in minutes.
    Returns 0 after final failure.
    """
    dt_start = datetime(date_obj.year, date_obj.month, date_obj.day, 0, 0, 0)
    dt_end   = datetime(date_obj.year, date_obj.month, date_obj.day, 23, 59, 59)
    for attempt in range(retries):
        try:
            filt = Filters(
                keyword    = keyword,
                start_date = dt_start,
                end_date   = dt_end,
                domain     = domain
            )
            df = gd.article_search(filt)
            count = len(df)
            logging.info(f"Domain {domain!r}, keyword {keyword!r}, date {date_obj}: {count}")
            return count
        except HTTPError as e:
            if e.response.status_code == 429:
                wait_min = 2 ** attempt
                logging.warning(
                    f"429 rate limit for {domain!r}, {keyword!r}, {date_obj}: "
                    f"waiting {wait_min} min (attempt {attempt+1}/{retries})"
                )
                time.sleep(wait_min * 60)
                continue
            logging.error(f"HTTP error {e} for {domain!r}, {keyword!r}, {date_obj}")
            return 0
        except ValueError as e:
            msg = str(e)
            if 'too short' in msg:
                logging.warning(f"Keyword {keyword!r} too short for {date_obj}, skipping")
                return 0
            raise
    logging.error(
        f"After {retries} attempts: {domain!r}, {keyword!r}, {date_obj} → returning 0"
    )
    return 0

def main():
    # ask for start date
    start_input = input("Start processing from date (YYYY-MM-DD): ").strip()
    try:
        global_start = datetime.strptime(start_input, '%Y-%m-%d').date()
    except ValueError:
        logging.error(f"Invalid date {start_input!r}, must be YYYY-MM-DD")
        return

    global_end = datetime.strptime(END_DATE, '%Y-%m-%d').date()

    # list keyword files
    files = sorted(f for f in os.listdir(KEYWORD_DIR) if f.endswith('.txt'))
    logging.info(f"Found {len(files)} keyword file(s)")
    if not files:
        logging.error(f"No files in {KEYWORD_DIR}")
        return

    for filename in files:
        m = re.search(r'(\d{4}-\d{2})', filename)
        if not m:
            logging.warning(f"Skipping {filename}: no YYYY-MM found")
            continue
        ym = m.group(1)
        year, month = map(int, ym.split('-'))

        month_first = date(year, month, 1)
        month_last  = date(year, month, calendar.monthrange(year, month)[1])

        if month_last < global_start or month_first > global_end:
            logging.info(f"Skipping {ym}: outside {global_start}–{global_end}")
            continue

        start_date = max(month_first, global_start)
        end_date   = min(month_last,  global_end)

        # load keywords
        path = os.path.join(KEYWORD_DIR, filename)
        with open(path, encoding='utf-8') as f:
            keywords = [l.strip() for l in f if l.strip()]

        # prepare output CSV
        output_csv = os.path.join(OUTPUT_DIR, f"{ym}.csv")

        # track processed dates
        processed = set()
        if os.path.exists(output_csv):
            old = pd.read_csv(output_csv, sep=';')
            old['date'] = pd.to_datetime(old['date'], format='%Y-%m-%d', errors='coerce')
            processed = set(old['date'].dt.strftime('%Y-%m-%d'))
            logging.info(f"{len(processed)} days already in {output_csv}")

        # iterate days
        total_days = (end_date - start_date).days + 1
        for i in range(total_days):
            dt = start_date + timedelta(days=i)
            ds = dt.strftime('%Y-%m-%d')
            if ds in processed:
                logging.info(f"Skipping {ds} for {ym}")
                continue

            logging.info(f"Processing {ds} for {ym}")
            row = {'date': ds}
            for kw in keywords:
                total = sum(
                    count_for_day_kw_dom(dt, kw, dom) or 0
                    for dom in DOMAINS
                )
                row[kw] = total
                time.sleep(1)
            row['total'] = sum(row[k] for k in keywords)

            # append or create
            if processed:
                df_old = pd.read_csv(output_csv, sep=';')
                df_save = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
            else:
                df_save = pd.DataFrame([row])

            df_save.to_csv(output_csv, sep=';', index=False)
            logging.info(f"Saved data for {ds} in {output_csv}")
            processed.add(ds)

    logging.info("All months processed")

if __name__ == "__main__":
    main()
