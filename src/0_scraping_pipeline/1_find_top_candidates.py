import sys
from pathlib import Path

# ─── Bootstrap src/ into import path ────────────────────────────────────────────
HERE    = Path(__file__).resolve()
PROJECT = HERE.parents[2]        # …/src/0_scraping_pipeline → MasterThesis_final
SRC_DIR = PROJECT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os
import pandas as pd

from utils.config import LIST_DIR

# ─── File paths ────────────────────────────────────────────────────────────────
CANDIDATES_FOLLOWERS = os.path.join(LIST_DIR, 'candidates_with_followers.csv')
PARTIES_FOLLOWERS    = os.path.join(LIST_DIR, 'parties_with_followers.csv')
TOP10_PER_PARTY_OUT  = os.path.join(LIST_DIR, 'top10_per_party.csv')
TOP3_PER_PARTY_OUT   = os.path.join(LIST_DIR, 'top3_per_party.csv')

# ─── Column definitions ─────────────────────────────────────────────────────────
PARTY_COL    = 'T_Partei'
FOLLOWER_COL = 'Follower_Count'
HANDLE_COL   = 'handle'

def compute_top10_per_party():
    """
    Load candidate follower counts, compute top 10 handles per party,
    and write to TOP10_PER_PARTY_OUT.
    """
    df = pd.read_csv(CANDIDATES_FOLLOWERS, sep=';', engine='python')
    # ensure columns exist
    for col in (PARTY_COL, FOLLOWER_COL, HANDLE_COL):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in {CANDIDATES_FOLLOWERS}")

    df = df.dropna(subset=[PARTY_COL, FOLLOWER_COL, HANDLE_COL])
    df[FOLLOWER_COL] = df[FOLLOWER_COL].astype(int)

    # sort by party & followers desc
    df_sorted = df.sort_values([PARTY_COL, FOLLOWER_COL],
                               ascending=[True, False])

    # take top 10 per party
    top10 = df_sorted.groupby(PARTY_COL).head(10).copy()
    top10['Rank'] = (
        top10.groupby(PARTY_COL)[FOLLOWER_COL]
             .rank(method='first', ascending=False)
             .astype(int)
    )

    # reorder columns: party,rank,handle,followers,...
    cols = [PARTY_COL, 'Rank', HANDLE_COL, FOLLOWER_COL] + \
           [c for c in top10.columns if c not in (PARTY_COL, 'Rank', HANDLE_COL, FOLLOWER_COL)]
    top10 = top10[cols]

    top10.to_csv(TOP10_PER_PARTY_OUT, sep=';', index=False)
    print(f"Top 10 candidate accounts per party written to {TOP10_PER_PARTY_OUT}")

def compute_top3_party_accounts():
    """
    Load party follower counts, compute top 3 party accounts per party (T_Partei),
    and write to TOP3_PER_PARTY_OUT.
    """
    df = pd.read_csv(PARTIES_FOLLOWERS, sep=';', engine='python')
    for col in (PARTY_COL, FOLLOWER_COL, HANDLE_COL):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in {PARTIES_FOLLOWERS}")

    df = df.dropna(subset=[PARTY_COL, FOLLOWER_COL, HANDLE_COL])
    df[FOLLOWER_COL] = df[FOLLOWER_COL].astype(int)

    # sort by party & followers desc
    df_sorted = df.sort_values([PARTY_COL, FOLLOWER_COL],
                               ascending=[True, False])

    # take top 3 accounts per party
    top3 = df_sorted.groupby(PARTY_COL).head(3).copy()
    top3['Rank'] = (
        top3.groupby(PARTY_COL)[FOLLOWER_COL]
            .rank(method='first', ascending=False)
            .astype(int)
    )

    cols = [PARTY_COL, 'Rank', HANDLE_COL, FOLLOWER_COL] + \
           [c for c in top3.columns if c not in (PARTY_COL, 'Rank', HANDLE_COL, FOLLOWER_COL)]
    top3 = top3[cols]

    top3.to_csv(TOP3_PER_PARTY_OUT, sep=';', index=False)
    print(f"Top 3 party accounts per party written to {TOP3_PER_PARTY_OUT}")

def main():
    compute_top10_per_party()
    compute_top3_party_accounts()

if __name__ == '__main__':
    main()
