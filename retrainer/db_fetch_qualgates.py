import os
import psycopg2
import pandas as pd
from langdetect import detect, LangDetectException
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------- CONFIG ----------------
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "ai_text_feedback"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
}

MIN_LENGTH = 5
MAX_LENGTH = 1000

# ---------------- HELPER FUNCTIONS ----------------
def get_conn():
    return psycopg2.connect(**DB_CONFIG)

def is_english(text):
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False

def is_consistent(row):
    """
    Remove feedback that contradicts model_score badly.
    (Protects regression from noisy labels)
    """
    label = row["user_feedback"]
    score = row["model_score"]

    if label == "human" and score > 0.7:
        return False
    if label == "ai" and score < 0.3:
        return False
    return True

# ---------------- FETCH DATA ----------------
conn = get_conn()
df = pd.read_sql("SELECT * FROM feedback_events", conn)
conn.close()

# ---------------- DATA QUALITY GATES ----------------

# 1️⃣ Deduplication (keep latest feedback per text)
df = df.sort_values("event_timestamp")
df = df.drop_duplicates(subset="text_hash", keep="last")

# 2️⃣ Language filtering
df = df[df["text"].apply(is_english)]

# 3️⃣ Length filtering (TEXT, not feedback label)
df = df[df["text"].str.len().between(MIN_LENGTH, MAX_LENGTH)]

# 4️⃣ Remove inconsistent feedback
df = df[df.apply(is_consistent, axis=1)]

# 5️⃣ Safety check: target_score must exist
df = df[df["target_score"].notna()]

# ---------------- FINAL DATASET ----------------
print("✅ Clean data ready for retraining")
print(df[["text", "model_score", "user_feedback", "target_score"]].head())

# Save for training
df.to_csv("clean_feedback.csv", index=False)
