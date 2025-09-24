import sys
from pathlib import Path

# ─── Bootstrap src/ into import path ────────────────────────────────────────────
HERE    = Path(__file__).resolve()
PROJECT = HERE.parents[1]        # …/src/assemble_master.py → MasterThesis_final
SRC_DIR = PROJECT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import glob
import os
import pandas as pd

from utils.config import BASE_DIR

# ─── Configuration ─────────────────────────────────────────────────────────────
# Input: cleaned annotation CSVs for augmentation
INPUT_DIR   = os.path.join(
    BASE_DIR,
    'data', 'processed', 'annotation', 'augmentation', 'cleaned'
)
# Output: assembled master file
OUTPUT_FILE = os.path.join(
    BASE_DIR,
    'data', 'processed', 'annotation', 'augmentation', 'augmentation_master.csv'
)

# ─── Supported readers ─────────────────────────────────────────────────────────
READERS = {
    ".csv": lambda fn: pd.read_csv(
        fn,
        sep=";",                # semicolon delimiter
        engine="python",
        on_bad_lines="skip",
        encoding="utf-8"
    ),
    ".xls": pd.read_excel,
    ".xlsx": pd.read_excel,
}

def assemble_master(input_dir: str, output_file: str):
    # 1) collect all files recursively, skipping any '_deleted' folders
    pattern = os.path.join(input_dir, "**", "*")
    files = glob.glob(pattern, recursive=True)
    dataframes = []
    for fn in files:
        parts = Path(fn).parts
        if os.path.isdir(fn) or "_deleted" in parts:
            continue
        if os.path.abspath(fn) == os.path.abspath(output_file):
            continue

        ext = Path(fn).suffix.lower()
        reader = READERS.get(ext)
        if not reader:
            continue

        try:
            df = reader(fn)
        except Exception as e:
            print(f"Failed to read {fn}: {e}")
            continue

        # tag each row with its candidate or handle (parent folder name)
        df["candidate"] = parts[-2] if len(parts) >= 2 else ""
        dataframes.append(df)

    if not dataframes:
        print("No valid files found to assemble.")
        return

    # 2) concatenate all into one master DataFrame
    master = pd.concat(dataframes, ignore_index=True)

    # 3) write out semicolon-delimited CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    master.to_csv(output_file, sep=";", index=False, encoding="utf-8")
    print(f"Assembled {len(master)} rows into {output_file}")

if __name__ == "__main__":
    assemble_master(INPUT_DIR, OUTPUT_FILE)
