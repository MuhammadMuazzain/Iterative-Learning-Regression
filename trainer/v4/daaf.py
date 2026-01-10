# Dataset as an artifact model run id is train experiment run, id, first run the mlflew.py: logs dataset to mlflow, then run training loop, v4_final.py
# then run this file, will attach the dataset as an artifact to the model run.

import os
import mlflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db")
model_run_id = "bf6fa44badde4e9b80412dce11457a0c"
csv_path = "./v4-training-without-test.csv"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Activate existing run
with mlflow.start_run(run_id=model_run_id):
    mlflow.log_artifact(csv_path, artifact_path="dataset")
    print(f"✅ CSV logged to run {model_run_id}")
