import os
import time
import logging
import json
from pathlib import Path
import pandas as pd
import openai
import re
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

# Few-Shot oder Zero-Shot
FEW_SHOT = True  # True für Few-Shot
MODEL_ID = "gpt-4.1"

# Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# OpenAI Key aus Datei laden
KEY_PATH = Path(r"E:\Final_Github_1\MasterThesis_final\keys\openai_key.txt")
with open(KEY_PATH, "r", encoding="utf-8") as f:
    openai.api_key = f.read().strip()

BASE_DIR = Path(r"E:\Final_Github_1\MasterThesis_final\data\processed\training and test")
INPUT_CSV = BASE_DIR / "test.csv"

# Output-Pfade
mode_name = "few" if FEW_SHOT else "zero"
OUT_PREDS = BASE_DIR / f"{MODEL_ID}_{mode_name}_preds.csv"
OUT_METRICS = BASE_DIR / f"{MODEL_ID}_{mode_name}_metrics.json"

# Few-Shot Beispiele
FIXED_FEWSHOT = r"""
1: Tweet: "Polizeiliche Schnellgerichte für Haftstrafen bei Angriffen auf Einsatzkräfte – absolut richtig!"
Bewerte die Haltung des folgenden Tweets gegenüber "CDU/CSU"?
Antwort: Zu

2: Tweet: "Sie/Ihr versteht es nicht und werdet es niemals verstehen."
Bewerte die Haltung des folgenden Tweets gegenüber "Lang, Ricarda"?
Antwort: Ab

3: Tweet: "Energiesektor: #Subventionen im #Energiesektor müssen gezielt und nachhaltig eingesetzt werden."
Bewerte die Haltung des folgenden Tweets gegenüber "Söder, Markus"?
Antwort: Ne
""".strip()

# Label Mapping
LABEL_TOKEN_MAP = {"Zustimmung": "Zu", "Ablehnung": "Ab", "Neutral": "Ne"}
SHORT_TO_LABEL = {v.lower(): k for k, v in LABEL_TOKEN_MAP.items()}
SHORTS = list(LABEL_TOKEN_MAP.values())

# Prompt-Builder
def build_prompt(name: str, text: str, few_shot: bool) -> str:
    header = (
        "### Aufgabe\n"
        f"Bewerte die Haltung des folgenden Tweets gegenüber \"{name}\". "
        "Berücksichtige Wortlaut, Untertöne, Ironie und politische Anspielungen.\n\n"
        f"Tweet: {text}\n\n"
        "### Antwortmöglichkeiten:\n"
        "• Zustimmung: Der Tweet äußert sich explizit oder implizit positiv oder unterstützend über das Ziel.\n"
        "• Ablehnung: Der Tweet äußert sich explizit oder implizit negativ oder kritisch über das Ziel.\n"
        "• Neutral: Der Tweet ist sachlich, ambivalent oder zeigt keine erkennbare Haltung.\n"
    )
    examples = f"\n### Beispiele\n{FIXED_FEWSHOT}\n" if few_shot else ""
    tail = (
        "\n### Ausgabeformat (Kurzform):\n"
        "Gib genau eines der folgenden Kürzel zurück (ohne Anführungszeichen, ohne Punkt):\n"
        "Zu\n"
        "Ab\n"
        "Ne"
    )
    return header + examples + tail

def build_messages(name: str, text: str, few_shot: bool):
    system_msg = {
        "role": "system",
        "content": (
            "Du bist ein Stance-Klassifizierer für politische Tweets. "
            "Kategorisiere die Haltung als genau eine der drei Klassen: Zustimmung (Zu), Ablehnung (Ab) oder Neutral (Ne)."
        )
    }
    user_msg = {"role": "user", "content": build_prompt(name, text, few_shot)}
    return [system_msg, user_msg]

# Parser
CLEAN_RE = re.compile(r'[\"\'\.\,\:\;\!\?\-\—\–\(\)\[\]\{\}]')
def parse_label(raw_output: str) -> str:
    if not isinstance(raw_output, str):
        return "NONE"

    t = raw_output.strip().lower()
    t = CLEAN_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()

    if t in SHORT_TO_LABEL:
        return SHORT_TO_LABEL[t]
    return "NONE"

# Daten laden
df = pd.read_csv(INPUT_CSV, sep=";")
if not {"name", "text", "label"}.issubset(df.columns):
    raise ValueError(f"{INPUT_CSV} muss die Spalten name, text, label enthalten")

df["pred"] = ""

# Inferenz
start_time = time.time()
for idx, row in df.iterrows():
    messages = build_messages(str(row["name"]).strip(), str(row["text"]).strip(), FEW_SHOT)
    try:
        resp = openai.ChatCompletion.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=5,
            temperature=0.0,
            top_p=1.0
        )
        raw = resp.choices[0].message.content or ""
        df.at[idx, "pred"] = parse_label(raw)
    except Exception as e:
        logging.error(f"API-Fehler bei Zeile {idx}: {e}")
        df.at[idx, "pred"] = "NONE"

    time.sleep(0.2)

duration = time.time() - start_time

# Metriken
labels = list(LABEL_TOKEN_MAP.keys())
report = classification_report(df["label"], df["pred"], labels=labels, output_dict=True)
conf_mat = confusion_matrix(df["label"], df["pred"], labels=labels)
per_class_f1 = f1_score(df["label"], df["pred"], labels=labels, average=None)

metrics = {
    "model": MODEL_ID,
    "mode": mode_name,
    "accuracy": accuracy_score(df["label"], df["pred"]),
    "macro_f1": f1_score(df["label"], df["pred"], average="macro"),
    "weighted_f1": f1_score(df["label"], df["pred"], average="weighted"),
    "precision_macro": precision_score(df["label"], df["pred"], average="macro"),
    "recall_macro": recall_score(df["label"], df["pred"], average="macro"),
    "per_class_f1": dict(zip(labels, per_class_f1.tolist())),
    "confusion_matrix": conf_mat.tolist(),
    "inference_time_per_1000": (duration / max(len(df), 1)) * 1000
}

df.to_csv(OUT_PREDS, sep=";", index=False, encoding="utf-8-sig")
with open(OUT_METRICS, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print("\nClassification Report:")
print(classification_report(df["label"], df["pred"], labels=labels))
print(f"\nVorhersagen gespeichert unter: {OUT_PREDS}")
print(f"Metriken gespeichert unter:   {OUT_METRICS}")
