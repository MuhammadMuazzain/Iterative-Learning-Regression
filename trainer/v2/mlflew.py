import os
import mlflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CSV_PATH = "v2-training-without-test.csv"

# 👇 ALWAYS do this
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db"))
mlflow.set_experiment("dataset_registry")

with mlflow.start_run(run_name="dataset_v2-without-test"):
    mlflow.log_artifact(CSV_PATH, artifact_path="datasets")
