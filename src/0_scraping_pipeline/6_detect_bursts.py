#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

# ─── Bootstrap src/ into import path ────────────────────────────────────────────
HERE    = Path(__file__).resolve()
PROJECT = HERE.parents[2]        # …/src/6_detect_bursts.py → MasterThesis_final
SRC_DIR = PROJECT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json
import math
import glob
from pathlib import Path as _Path

import numpy as np
import pandas as pd

from utils.config import BASE_DIR

# ─── Parameters ────────────────────────────────────────────────────────────────
S_FACTOR  = 1.3   # burst amplification factor s (>1)
GAMMA     = 1     # transition cost multiplier γ (>=0)
EXCLUDE   = [
    '2024-09-04', '2024-09-21',
    '2024-10-01', '2024-10-24',
    '2024-11-08', '2024-11-09',
    '2024-12-05', '2024-12-08',
    '2025-01-04', '2025-01-24',
    '2025-02-22', '2025-02-24',
    '2025-03-18', '2025-03-29'
]

# ─── Directories ───────────────────────────────────────────────────────────────
# Input: monthly GDELT daily counts per keyword
INPUT_DIR  = _Path(BASE_DIR) / 'data' / 'processed' / 'augmentation' / 'gdelt_counts' / 'daily_keyword_count'
# Output: burst detection results
OUTPUT_DIR = _Path(BASE_DIR) / 'data' / 'processed' / 'augmentation' / 'burst_results'

# create output root if needed
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

summary_records = []

def kleinberg_burst_detection(df: pd.DataFrame, s: float, gamma: float) -> dict:
    # 1. sort by date
    df = df.sort_values('date').reset_index(drop=True)
    dates  = df['date'].tolist()
    events = df['total'].values
    N      = len(events)

    # 2. global mean rate μ
    mu = events.sum() / N

    # 3. determine max burst level k_max
    k_max = int(np.ceil(np.log(events.max() / mu) / np.log(s)))
    states = list(range(k_max + 1))

    # 4. rates r_k
    rates = {k: mu * (s ** k) for k in states}

    # 5. event-cost matrix (-ln Poisson)
    cost_event = np.zeros((N, len(states)))
    for t, x in enumerate(events):
        for k in states:
            r = rates[k]
            cost_event[t, k] = -(x * math.log(r) - r - math.lgamma(x + 1))

    # 6. transition-cost matrix τ(i→j)
    tau = np.zeros((len(states), len(states)))
    for i in states:
        for j in states:
            tau[i, j] = 0 if j >= i else gamma * (i - j)

    # 7. dynamic programming (Viterbi)
    C    = np.full((N, len(states)), np.inf)
    prev = np.zeros((N, len(states)), dtype=int)

    # initialization t=0
    for k in states:
        C[0, k]    = cost_event[0, k] + tau[0, k]
        prev[0, k] = 0

    # recurrence
    for t in range(1, N):
        for j in states:
            costs  = [C[t-1, i] + tau[i, j] for i in states]
            i_min  = int(np.argmin(costs))
            C[t, j] = cost_event[t, j] + costs[i_min]
            prev[t, j] = i_min

    # backtracking
    state_seq = [0] * N
    k_end     = int(np.argmin(C[-1, :]))
    state_seq[-1] = k_end
    for t in range(N-1, 0, -1):
        state_seq[t-1] = prev[t, state_seq[t]]

    # find the most intense burst day
    max_state  = max(state_seq)
    candidates = [t for t, k in enumerate(state_seq) if k == max_state]
    burst_idx  = max(candidates, key=lambda t: events[t])
    burst_day  = dates[burst_idx]
    burst_state = max_state

    return {
        'dates':          dates,
        'rates':          rates,
        'cost_event':     cost_event,
        'tau':            tau,
        'dp_cost':        C,
        'state_sequence': state_seq,
        'burst_day':      burst_day,
        'burst_state':    burst_state
    }

# ─── Main process ──────────────────────────────────────────────────────────────
for file_path in glob.glob(str(INPUT_DIR / '*.csv')):
    df_in    = pd.read_csv(file_path, sep=';')
    df_clean = df_in[~df_in['date'].isin(EXCLUDE)].copy()

    result = kleinberg_burst_detection(df_clean, S_FACTOR, GAMMA)

    basename = _Path(file_path).stem
    out_dir  = OUTPUT_DIR / basename
    out_dir.mkdir(parents=True, exist_ok=True)

    # save intermediate outputs
    pd.DataFrame.from_dict({
        'state': list(result['rates'].keys()),
        'rate':  list(result['rates'].values())
    }).to_csv(out_dir / 'rates.csv', sep=';', index=False)

    pd.DataFrame(
        result['cost_event'],
        index=result['dates'],
        columns=[f'k={k}' for k in result['rates'].keys()]
    ).to_csv(out_dir / 'cost_event.csv', sep=';')

    pd.DataFrame(
        result['tau'],
        index=[f'i={i}' for i in result['rates'].keys()],
        columns=[f'j={j}' for j in result['rates'].keys()]
    ).to_csv(out_dir / 'tau.csv', sep=';')

    pd.DataFrame(
        result['dp_cost'],
        index=result['dates'],
        columns=[f'k={k}' for k in result['rates'].keys()]
    ).to_csv(out_dir / 'dp_cost.csv', sep=';')

    pd.DataFrame({
        'date':  result['dates'],
        'state': result['state_sequence']
    }).to_csv(out_dir / 'state_sequence.csv', sep=';', index=False)

    # summary JSON
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as fp:
        json.dump({
            'burst_day':   result['burst_day'],
            'burst_state': int(result['burst_state'])
        }, fp, ensure_ascii=False, indent=2)

    summary_records.append({
        'file':        basename,
        'burst_day':   result['burst_day'],
        'burst_state': int(result['burst_state'])
    })

# save overall summary
pd.DataFrame(summary_records).to_csv(
    OUTPUT_DIR / 'summary_all_months.csv',
    sep=';',
    index=False
)

print("Burst detection complete. Results in", OUTPUT_DIR)
