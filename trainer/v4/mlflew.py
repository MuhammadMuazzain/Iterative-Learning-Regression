import os
import hashlib
import pandas as pd
import numpy as np
import mlflow
from transformers import RobertaTokenizer
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =========================
# CONFIG (CHANGE ONLY THIS)
# =========================
CSV_PATH = "./v4-training-without-test.csv"
TEXT_COL = "text"
LABEL_COL = "label"

MODEL_NAME = os.getenv("MODEL_NAME", "roberta-large")
MAX_LENGTH = 512

EXPERIMENT_NAME = "dataset_audit_v4"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db")

# =========================
# UTILS
# =========================
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def log_basic_stats(df):
    mlflow.log_metric("num_rows", len(df))
    mlflow.log_metric("num_null_text", df[TEXT_COL].isna().sum())
    mlflow.log_metric("num_null_labels", df[LABEL_COL].isna().sum())

def log_label_stats(labels):
    labels = labels.astype(float)
    mlflow.log_metric("label_mean", labels.mean())
    mlflow.log_metric("label_std", labels.std())
    mlflow.log_metric("label_min", labels.min())
    mlflow.log_metric("label_max", labels.max())
    mlflow.log_metric("label_median", np.median(labels))

def log_token_stats(texts, tokenizer):
    lengths = [len(tokenizer(t)["input_ids"]) for t in texts]

    mlflow.log_metric("token_len_mean", np.mean(lengths))
    mlflow.log_metric("token_len_std", np.std(lengths))
    mlflow.log_metric("token_len_p95", np.percentile(lengths, 95))
    mlflow.log_metric("token_len_p99", np.percentile(lengths, 99))
    mlflow.log_metric("token_len_max", np.max(lengths))
    mlflow.log_metric(
        "pct_truncated",
        np.mean(np.array(lengths) > MAX_LENGTH)
    )

# =========================
# MAIN
# =========================
def run_dataset_audit(csv_path):
    csv_path = Path(csv_path)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"dataset_audit_{csv_path.stem}"):

        # ---- Dataset checksum (TAG) ----
        checksum = sha256_file(csv_path)
        mlflow.set_tag("dataset_checksum", checksum)
        mlflow.set_tag("dataset_file", csv_path.name)

        # ---- Load data ----
        df = pd.read_csv(csv_path)

        # ---- Basic sanity stats ----
        log_basic_stats(df)

        # ---- Label distribution ----
        log_label_stats(df[LABEL_COL])

        # ---- Token statistics ----
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
        log_token_stats(df[TEXT_COL].astype(str).tolist(), tokenizer)

        print("✅ Dataset audit logged to MLflow")
        print(f"🔐 Dataset checksum: {checksum}")

# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    run_dataset_audit(CSV_PATH)
