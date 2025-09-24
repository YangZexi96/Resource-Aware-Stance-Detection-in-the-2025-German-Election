# bulk_transform.py
import pandas as pd
from pathlib import Path

# ==== Paths ====
CANDIDATE_MAP_PATH = r"E:\Final_Github_1\MasterThesis_final\data\raw\external\candidates.csv"
MASTER_DATASET_PATH = r"E:\Final_Github_1\MasterThesis_final\data\processed\preprocessing\mirror\mirror_prepped.csv"

# ==== Load data ====
df_master = pd.read_csv(MASTER_DATASET_PATH, sep=";", encoding="utf-8-sig")
df_map = pd.read_csv(CANDIDATE_MAP_PATH, sep=",", encoding="utf-8-sig")

# ==== Extract handle from SM_XURL ====
df_map["handle"] = df_map["SM_XURL"].str.extract(r"https://www\.x\.com/([^/]+)")
df_map["handle"] = df_map["handle"].str.strip().str.lower()

# ==== Prepare mapping dict: handle -> Name ====
map_dict = dict(zip(df_map["handle"], df_map["Name"]))

# ==== Create 'name' column in master dataset ====
df_master["name"] = df_master["candidate"].str.strip().str.lower().map(map_dict)

# Optional: fallback to original candidate if mapping missing
df_master["name"] = df_master.apply(
    lambda row: row["name"] if pd.notna(row["name"]) else row["candidate"],
    axis=1
)

# ==== Keep only required columns ====
df_final = df_master.rename(columns={
    "tweet_id": "ID",
    "tweet_text": "text"
})[["ID", "name", "text"]]

# ==== Save to same folder ====
master_path = Path(MASTER_DATASET_PATH)
output_path = master_path.parent / f"{master_path.stem}_for_inference.csv"

df_final.to_csv(output_path, sep=";", encoding="utf-8-sig", index=False)

print(f"✅ Saved transformed dataset to {output_path}")
