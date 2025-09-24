
import sys
from pathlib import Path

# ─── Bootstrap src/ into import path ────────────────────────────────────────────
HERE    = Path(__file__).resolve()
PROJECT = HERE.parents[2]        # up from …/src/0_scraping_pipeline → MasterThesis_final
SRC_DIR = PROJECT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.config import LIST_DIR, EXTERNAL_DIR, NITTER_BASE


# ─── File paths ────────────────────────────────────────────────────────────────
CANDIDATES_IN  = os.path.join(EXTERNAL_DIR, 'candidates.csv')
PARTIES_IN     = os.path.join(EXTERNAL_DIR, 'parties.csv')
CANDIDATES_OUT = os.path.join(LIST_DIR, 'candidates_with_followers.csv')
PARTIES_OUT    = os.path.join(LIST_DIR, 'parties_with_followers.csv')

# ─── Filter settings ───────────────────────────────────────────────────────────
PARTIES_OF_INTEREST = [
    'AfD', 'CDU', 'SPD', 'B90/GRÜNE', 'FDP', 'LINKE', 'CSU', 'BSW'
]

# ─── Concurrency ───────────────────────────────────────────────────────────────
BATCH_SIZE  = 200
MAX_WORKERS = 3

def extract_handle_from_url(url):
    """
    Given a profile URL or XID, return only the Twitter handle.
    """
    if pd.isna(url) or not url:
        return None
    return url.rstrip('/').split('/')[-1]

def scrape_followers(session, handle):
    """
    Scrape the follower count for a single handle from Nitter.
    Returns an integer or None if unavailable.
    """
    if not handle:
        return None

    try:
        r = session.get(f"{NITTER_BASE}/{handle}", timeout=10)
        r.raise_for_status()
    except requests.RequestException:
        return None

    soup = BeautifulSoup(r.text, 'lxml')
    li   = soup.find('li', class_='followers')
    span = li.find('span', class_='profile-stat-num') if li else None
    if not span:
        return None

    txt = span.get_text(strip=True).replace('\u202f', '').replace('\xa0', '')
    if txt.endswith('K'):
        return int(float(txt[:-1]) * 1_000)
    if txt.endswith('M'):
        return int(float(txt[:-1]) * 1_000_000)
    try:
        return int(txt.replace(',', ''))
    except ValueError:
        return None

def process_list(name, infile, outfile, session):
    """
    Read infile from LIST_DIR, filter by PARTIES_OF_INTEREST,
    scrape follower counts, and write outfile.
    """
    print(f"\n=== Processing {name} ===")
    df = pd.read_csv(os.path.join(LIST_DIR, infile), sep=None, engine='python')

    # keep only the parties you care about
    if 'T_Partei' in df.columns:
        df = df[df['T_Partei'].isin(PARTIES_OF_INTEREST)].reset_index(drop=True)
        print(f"→ {len(df)} rows after filtering by {PARTIES_OF_INTEREST}")
    else:
        print("→ Warning: 'T_Partei' column not found, skipping filter")

    df['handle'] = df['SM_XURL'].fillna(df.get('SM_XID', '')).apply(extract_handle_from_url)
    df['Follower_Count'] = None

    # batch-scrape
    total = len(df)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = list(range(start, end))

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(scrape_followers, session, df.at[i, 'handle']): i
                for i in batch
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                df.at[idx, 'Follower_Count'] = fut.result()
                print(f"{idx} | {df.at[idx,'handle']}: {df.at[idx,'Follower_Count']}")

        # checkpoint
        df.to_csv(os.path.join(LIST_DIR, outfile), sep=';', index=False)
        print(f"→ checkpoint saved through row {end-1}")

    print(f"✅ Finished {name}! Output: {os.path.join(LIST_DIR, outfile)}")

def main():
    with requests.Session() as session:
        process_list('candidates', 'candidates.csv', 'candidates_with_followers.csv', session)
        process_list('parties',   'parties.csv',    'parties_with_followers.csv',    session)

if __name__ == '__main__':
    main()
