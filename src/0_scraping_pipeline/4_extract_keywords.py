import sys
from pathlib import Path

# ─── Bootstrap src/ into import path ────────────────────────────────────────────
HERE    = Path(__file__).resolve()
PROJECT = HERE.parents[1]          # …/src → MasterThesis_final
SRC_DIR = PROJECT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os
import pandas as pd
import calendar
import string
import logging
from datetime import datetime
from gdeltdoc import GdeltDoc, Filters
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.config import BASE_DIR, TWEETS_DIR

# ─── Dynamic directories ───────────────────────────────────────────────────────
# Where monthly hashtag counts live
MONTHLY_COUNTS_DIR    = os.path.join(BASE_DIR, 'data', 'test', 'augmentation', 'hashtags')
# Where extracted keywords should be written
KEYWORDS_DIR          = os.path.join(BASE_DIR, 'data', 'test', 'augmentation', 'keywords')

MONTHLY_FILE_TEMPLATE = 'hashtag_counts_{month}.csv'

# ─── Keyword extraction settings ───────────────────────────────────────────────
MIN_KEYWORD_LEN       = 5
INITIAL_SEED_COUNT    = 25
SEED_INCREMENT        = 10
MAX_SEED_COUNT        = 200
TOP_N_FINAL           = 10

# ─── GDELT settings ────────────────────────────────────────────────────────────
MAX_ARTICLES          = 1_000_000
DOMAINS = [
    'tagesschau.de',
    'heute.de',
    'deutschlandfunk.de',
    'sueddeutsche.de',
    'faz.net',
    'spiegel.de',
    'zeit.de',
    'handelsblatt.com',
    'welt.de',
    'tagesspiegel.de'
]

# ─── TF-IDF parameters ──────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES   = 500
MAX_NGRAM_LENGTH     = 2

gd = GdeltDoc()

def load_seeds(month: str, top_n: int) -> list[str]:
    path = os.path.join(
        MONTHLY_COUNTS_DIR,
        MONTHLY_FILE_TEMPLATE.format(month=month)
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file for {month}: {path}")
    df = pd.read_csv(path, sep=';')
    tag_col = 'hashtag' if 'hashtag' in df.columns else df.columns[0]
    cnt_col = 'count'   if 'count'   in df.columns else df.select_dtypes(include='number').columns[0]

    df['term'] = df[tag_col].astype(str).str.lstrip('#').str.lower()
    df = df[df['term'].str.len() >= MIN_KEYWORD_LEN]

    exclude = {'gruene', 'grüne', 'grünen', 'linke'}
    df = df[~df['term'].isin(exclude)]

    return df.nlargest(top_n, cnt_col)['term'].tolist()

def fetch_corpus(start_str: str, end_str: str) -> list[str]:
    dt_start = datetime.strptime(start_str, '%Y%m%d')
    dt_end   = datetime.strptime(end_str,   '%Y%m%d')
    texts = []
    per_domain_max = max(1, MAX_ARTICLES // len(DOMAINS))

    for domain in DOMAINS:
        filt = Filters(
            start_date = dt_start,
            end_date   = dt_end,
            language   = "german",
            country    = "germany",
            domain     = domain
        )
        try:
            df = gd.article_search(filt)
        except Exception as e:
            logging.warning(f"Fetch for domain {domain} failed: {e}")
            continue
        if df.empty:
            continue
        df = df.head(per_domain_max)
        if 'title' in df.columns:
            texts.extend(df['title'].dropna().astype(str).tolist())
        else:
            texts.extend(df.iloc[:,0].dropna().astype(str).tolist())

    return texts

def preprocess_texts(texts: list[str]) -> list[str]:
    table = str.maketrans('', '', string.punctuation)
    return [t.lower().translate(table).strip() for t in texts]

def extract_tfidf_terms(texts: list[str]) -> set[str]:
    vec = TfidfVectorizer(
        ngram_range=(1, MAX_NGRAM_LENGTH),
        max_features=TFIDF_MAX_FEATURES
    )
    vec.fit(texts)
    return set(vec.get_feature_names_out())

def month_bounds(month: str) -> tuple[str, str]:
    y, m = map(int, month.split('-'))
    first = datetime(y, m, 1).strftime('%Y%m%d')
    last  = datetime(y, m, calendar.monthrange(y, m)[1]).strftime('%Y%m%d')
    return first, last

def main_for_month(month: str):
    start_str, end_str = month_bounds(month)
    seed_count = INITIAL_SEED_COUNT
    validated: list[str] = []

    while len(validated) < TOP_N_FINAL:
        seeds = load_seeds(month, seed_count)
        print(f"[{month}] Testing with {seed_count} seeds…")

        corpus = fetch_corpus(start_str, end_str)
        if not corpus:
            print(f"[{month}] No corpus found, stopping early.")
            break

        clean = preprocess_texts(corpus)
        tfidf_terms = extract_tfidf_terms(clean)

        new_matches = [kw for kw in seeds if kw in tfidf_terms and kw not in validated]
        slots = TOP_N_FINAL - len(validated)
        validated.extend(new_matches[:slots])

        if len(validated) >= TOP_N_FINAL:
            break

        seed_count += SEED_INCREMENT
        if seed_count > MAX_SEED_COUNT:
            print(f"[{month}] Seed limit reached, keeping existing matches.")
            break

    os.makedirs(KEYWORDS_DIR, exist_ok=True)
    out_path = os.path.join(KEYWORDS_DIR, f"keywords_{month}.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        for kw in validated:
            f.write(kw + '\n')
    print(f"[{month}] Wrote {len(validated)} keywords: {validated}")

def main_all_months():
    files = os.listdir(MONTHLY_COUNTS_DIR)
    months = sorted(
        fname[len('hashtag_counts_'):-4]
        for fname in files
        if fname.startswith('hashtag_counts_') and fname.endswith('.csv')
    )
    for month in months:
        main_for_month(month)

if __name__ == '__main__':
    main_all_months()
