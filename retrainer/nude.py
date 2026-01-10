import os
import psycopg2
import pandas as pd
from langdetect import detect, LangDetectException
from dotenv import load_dotenv

# ---------------- LOAD ENV ----------------
# Only loads .env locally; ignored in GitHub Actions
load_dotenv()

# ---------------- CONFIG ----------------
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),  # default 5432 if not set
}

if not all(DB_CONFIG.values()):
    raise ValueError(
        "Database config incomplete. Make sure DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT are set."
    )

MIN_LENGTH = 5
MAX_LENGTH = 1000

FEEDBACK_TO_SCORE = {
    "human": 0.0,
    "human_edited_ai": 0.5,
    "ai": 1.0
}

# ---------------- HELPER FUNCTIONS ----------------
def get_conn():
    """Connect to PostgreSQL using DB_CONFIG"""
    return psycopg2.connect(**DB_CONFIG)

def is_english(text):
    """Return True if text is English"""
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False

def is_consistent(row):
    """Remove feedback that contradicts model_score badly"""
    label = row["user_feedback"]
    score = row["model_score"]

    if label == "human" and score > 0.7:
        return False
    if label == "ai" and score < 0.3:
        return False
    return True

# ---------------- FETCH DATA ----------------
print("🌐 Connecting to database...")
conn = get_conn()
df = pd.read_sql("SELECT * FROM feedback_events", conn)
conn.close()
print(f"📦 Fetched {len(df)} rows from DB")

# ---------------- DATA QUALITY GATES ----------------
# 1️⃣ Deduplication (keep latest feedback per text)
df = df.sort_values("event_timestamp")
df = df.drop_duplicates(subset="text_hash", keep="last")

# 2️⃣ Language filtering
df = df[df["text"].apply(is_english)]

# 3️⃣ Length filtering
df = df[df["text"].str.len().between(MIN_LENGTH, MAX_LENGTH)]

# 4️⃣ Remove inconsistent feedback
df = df[df.apply(is_consistent, axis=1)]

# 5️⃣ Map user_feedback to numeric regression target if not already
if "target_score" not in df.columns or df["target_score"].isnull().any():
    df["target_score"] = df["user_feedback"].map(FEEDBACK_TO_SCORE)

# ---------------- FINAL DATASET ----------------
print("✅ Clean data ready for retraining")
print(df[["text", "model_score", "user_feedback", "target_score"]].head())

# Save to CSV for training
df.to_csv("clean_feedback.csv", index=False)
print(f"💾 Saved clean data to clean_feedback.csv ({len(df)} rows)")
