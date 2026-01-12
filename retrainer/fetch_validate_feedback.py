import os
import sys
import psycopg2
import pandas as pd
from datetime import datetime, timezone
from langdetect import detect, LangDetectException
from dotenv import load_dotenv

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

# Database config
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "ai_text_feedback"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
}

# Data quality thresholds
MIN_LENGTH = 5
MAX_LENGTH = 1000
MIN_SAMPLES = 100
MAX_DUPLICATE_RATIO = 0.05
MAX_SINGLE_CLASS_RATIO = 0.85
MAX_MEAN_SCORE_DRIFT = 0.2
MAX_AVG_AGE_DAYS = 30
MAX_FEEDBACK_MISMATCH = 0.1

# Output paths
OUTPUT_PATH = "clean_feedback.csv"
HISTORICAL_MEAN_PATH = "previous_mean.txt"

# Required columns
REQUIRED_COLUMNS = {
    "text",
    "text_hash",
    "user_feedback",
    "model_score",
    "target_score",
    "event_timestamp",
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def fail(msg):
    """Exit with failure message"""
    print(f"❌ VALIDATION FAILED: {msg}")
    sys.exit(1)

def warn(msg):
    """Print warning message"""
    print(f"⚠️  WARNING: {msg}")

def get_db_connection():
    """Create database connection"""
    return psycopg2.connect(**DB_CONFIG)

def is_english(text):
    """Check if text is in English"""
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False

def is_consistent(row):
    """
    Check if feedback is consistent with model score.
    Removes contradictory labels that would hurt training.
    """
    label = row["user_feedback"]
    score = row["model_score"]
    
    # Human feedback but high AI score
    if label == "human" and score > 0.7:
        return False
    # AI feedback but low AI score
    if label == "ai" and score < 0.3:
        return False
    return True

# ============================================================
# STEP 1: FETCH DATA FROM DATABASE
# ============================================================

print("=" * 60)
print("STEP 1: FETCHING DATA FROM DATABASE")
print("=" * 60)

try:
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM feedback_events", conn)
    conn.close()
    print(f"✅ Fetched {len(df)} rows from database")
except Exception as e:
    fail(f"Database fetch failed: {e}")

# ============================================================
# STEP 2: DATA CLEANING
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: DATA CLEANING")
print("=" * 60)

initial_count = len(df)

# 1. Deduplication (keep latest feedback per text)
print("\n🔹 Removing duplicates...")
df = df.sort_values("event_timestamp")
before_dedup = len(df)
df = df.drop_duplicates(subset="text_hash", keep="last")
print(f"   Removed {before_dedup - len(df)} duplicates")

# 2. Language filtering
print("\n🔹 Filtering non-English texts...")
before_lang = len(df)
df = df[df["text"].apply(is_english)]
print(f"   Removed {before_lang - len(df)} non-English texts")

# 3. Length filtering
print("\n🔹 Applying length filters ({MIN_LENGTH}-{MAX_LENGTH} chars)...")
before_length = len(df)
df = df[df["text"].str.len().between(MIN_LENGTH, MAX_LENGTH)]
print(f"   Removed {before_length - len(df)} texts outside length range")

# 4. Remove inconsistent feedback
print("\n🔹 Removing inconsistent feedback...")
before_consistency = len(df)
df = df[df.apply(is_consistent, axis=1)]
print(f"   Removed {before_consistency - len(df)} inconsistent labels")

# 5. Remove rows with missing target_score
print("\n🔹 Removing rows with missing target_score...")
before_target = len(df)
df = df[df["target_score"].notna()]
print(f"   Removed {before_target - len(df)} rows with missing target")

print(f"\n📊 Cleaning summary: {initial_count} → {len(df)} rows ({len(df)/initial_count*100:.1f}% retained)")

# ============================================================
# STEP 3: VALIDATION CHECKS
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: DATA VALIDATION")
print("=" * 60)

# 1. Schema check
print("\n🔹 Checking required columns...")
missing_cols = REQUIRED_COLUMNS - set(df.columns)
if missing_cols:
    fail(f"Missing required columns: {missing_cols}")
print("   ✅ All required columns present")

# 2. Minimum samples check
print(f"\n🔹 Checking minimum sample count (>= {MIN_SAMPLES})...")
if len(df) < MIN_SAMPLES:
    fail(f"Not enough samples ({len(df)} < {MIN_SAMPLES})")
print(f"   ✅ {len(df)} samples available")

# 3. Null check
print("\n🔹 Checking for excessive nulls...")
null_ratio = df.isna().mean()
bad_nulls = null_ratio[null_ratio > 0.01]
if not bad_nulls.empty:
    fail(f"Too many nulls detected:\n{bad_nulls}")
print("   ✅ Null values within acceptable range")

# 4. Duplicate leakage check
print(f"\n🔹 Checking duplicate leakage (<= {MAX_DUPLICATE_RATIO*100}%)...")
dup_ratio = df.duplicated("text_hash").mean()
print(f"   Duplicate ratio: {dup_ratio:.2%}")
if dup_ratio > MAX_DUPLICATE_RATIO:
    fail("Duplicate leakage too high")
print("   ✅ Duplicate ratio acceptable")

# 5. Label distribution
print(f"\n🔹 Checking label distribution (max class <= {MAX_SINGLE_CLASS_RATIO*100}%)...")
label_dist = df["target_score"].value_counts(normalize=True)
print(f"   Label distribution:\n{label_dist}")
if label_dist.max() > MAX_SINGLE_CLASS_RATIO:
    fail("Severe label imbalance detected")
if df["target_score"].nunique() < 2:
    fail("Only one target class present")
print("   ✅ Label distribution acceptable")

# 6. Score consistency
print(f"\n🔹 Checking feedback-score alignment (<= {MAX_FEEDBACK_MISMATCH*100}%)...")
bad_alignment = (
    ((df["user_feedback"] == "human") & (df["model_score"] > 0.8)) |
    ((df["user_feedback"] == "ai") & (df["model_score"] < 0.2))
).mean()
print(f"   Feedback-score mismatch: {bad_alignment:.2%}")
if bad_alignment > MAX_FEEDBACK_MISMATCH:
    fail("Too much disagreement between model_score and feedback")
print("   ✅ Feedback-score alignment acceptable")

# 7. Freshness check
print(f"\n🔹 Checking data freshness (<= {MAX_AVG_AGE_DAYS} days avg)...")
df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
now = datetime.now(timezone.utc)
avg_age_days = ((now - df["event_timestamp"]).dt.total_seconds() / 86400).mean()
print(f"   Average feedback age: {avg_age_days:.1f} days")
if avg_age_days > MAX_AVG_AGE_DAYS:
    warn(f"Data is older than {MAX_AVG_AGE_DAYS} days")
else:
    print("   ✅ Data freshness acceptable")

# 8. Target score range
print("\n🔹 Checking target_score range [0.0, 1.0]...")
if not df["target_score"].between(0.0, 1.0).all():
    fail("target_score outside expected range [0,1]")
print("   ✅ Target scores within valid range")

# 9. Drift check
print(f"\n🔹 Checking model score drift (<= {MAX_MEAN_SCORE_DRIFT})...")
current_mean = df["model_score"].mean()

if os.path.exists(HISTORICAL_MEAN_PATH):
    with open(HISTORICAL_MEAN_PATH, "r") as f:
        old_mean = float(f.read().strip())
    
    drift = abs(current_mean - old_mean)
    print(f"   Previous mean: {old_mean:.3f}")
    print(f"   Current mean: {current_mean:.3f}")
    print(f"   Drift: {drift:.3f}")
    
    if drift > MAX_MEAN_SCORE_DRIFT:
        warn(f"Model score drift is high ({drift:.3f})")
    else:
        print("   ✅ Drift within acceptable range")
else:
    print(f"   Current mean: {current_mean:.3f}")
    print("   ℹ️  No historical baseline found (first run)")

# Save current mean for next run
with open(HISTORICAL_MEAN_PATH, "w") as f:
    f.write(str(current_mean))

# ============================================================
# STEP 4: SAVE CLEANED DATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: SAVING CLEANED DATA")
print("=" * 60)

df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved {len(df)} clean samples to {OUTPUT_PATH}")

# Display sample
print("\n📋 Sample of cleaned data:")
print(df[["text", "model_score", "user_feedback", "target_score"]].head())

# ============================================================
# SUCCESS
# ============================================================

print("\n" + "=" * 60)
print("✅ DATA FETCH & VALIDATION COMPLETE — READY FOR RETRAINING")
print("=" * 60)
print(f"\n📊 Final Statistics:")
print(f"   Total samples: {len(df)}")
print(f"   Retention rate: {len(df)/initial_count*100:.1f}%")
print(f"   Average age: {avg_age_days:.1f} days")
print(f"   Mean model score: {current_mean:.3f}")
print(f"   Output: {OUTPUT_PATH}")