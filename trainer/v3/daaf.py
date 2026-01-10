import os
import mlflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db")
model_run_id = "db1e2839e4c3422790aca01a2eeefb48"
csv_path = "./v3-training-without-test.csv"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Activate existing run
with mlflow.start_run(run_id=model_run_id):
    mlflow.log_artifact(csv_path, artifact_path="dataset")
    print(f"✅ CSV logged to run {model_run_id}")
