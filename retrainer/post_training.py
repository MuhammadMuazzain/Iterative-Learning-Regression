import os
import json
import shutil
import subprocess
import mlflow
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIG ----------------
EXPERIMENT_NAME = "roberta_lora_regression_continuous_v2"  # ✅ ADDED - Must match train.py
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

BEST_MODEL_PATH = "models/model.pt"  # Production model
CANDIDATE_MODEL_PATH = "models/candidate_model.pt"

BEST_METRICS_PATH = "models/model_metrics.json"  # Production metrics
CANDIDATE_METRICS_PATH = "models/candidate_metrics.json"

BACKUP_DIR = "models/backups"

# Metric to compare (lower is better for MAE/RMSE)
METRIC_KEY = "test_mae"  # ✅ Matches training script output

# Improvement threshold (only promote if at least X% better)
MIN_IMPROVEMENT = 0.02  # 2% improvement required

# ---------------- UTILS ----------------
def load_metrics(path):
    """Load metrics from JSON file"""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def save_metrics(path, metrics):
    """Save metrics to JSON file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

def backup_current_model():
    """Backup current production model before replacing"""
    if not os.path.exists(BEST_MODEL_PATH):
        return None
    
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"model_backup_{timestamp}.pt")
    
    shutil.copy(BEST_MODEL_PATH, backup_path)
    print(f"📦 Backed up current model to: {backup_path}")
    return backup_path

def dvc_track_and_push(path):
    """Track file with DVC and push to remote"""
    try:
        # Add to DVC
        subprocess.run(["dvc", "add", path], check=True)
        print(f"✅ DVC tracked: {path}")
        
        # Add .dvc file to Git
        dvc_file = f"{path}.dvc"
        gitignore_file = os.path.join(os.path.dirname(path), ".gitignore")
        
        # Add .dvc file
        subprocess.run(["git", "add", dvc_file], check=True)
        
        # Add .gitignore only if it exists
        if os.path.exists(gitignore_file):
            subprocess.run(["git", "add", gitignore_file], check=True)
        
        # Push to DVC remote
        subprocess.run(["dvc", "push"], check=True)
        print(f"✅ DVC pushed: {path}")
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  DVC operation failed: {e}")
        raise

def git_commit_and_push(message):
    """Commit and push to Git"""
    try:
        subprocess.run(["git", "commit", "-m", message], check=True)
        print(f"✅ Git committed: {message}")
        
        # Optional: auto-push to remote
        subprocess.run(["git", "push"], check=True)
        print(f"✅ Git pushed")
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Git operation failed: {e}")
        raise

def cleanup_candidate():
    """Remove candidate model after decision"""
    if os.path.exists(CANDIDATE_MODEL_PATH):
        os.remove(CANDIDATE_MODEL_PATH)
    if os.path.exists(CANDIDATE_METRICS_PATH):
        os.remove(CANDIDATE_METRICS_PATH)
    print("🧹 Cleaned up candidate files")

# ---------------- MAIN LOGIC ----------------
def main():
    print("=" * 60)
    print("POST-TRAINING MODEL PROMOTION")
    print("=" * 60)
    
    # Validate candidate exists
    if not os.path.exists(CANDIDATE_MODEL_PATH):
        print("❌ Candidate model not found. Did training complete?")
        return False
    
    if not os.path.exists(CANDIDATE_METRICS_PATH):
        print("❌ Candidate metrics missing. Training may have failed.")
        return False
    
    # Load metrics
    candidate_metrics = load_metrics(CANDIDATE_METRICS_PATH)
    best_metrics = load_metrics(BEST_METRICS_PATH)
    
    candidate_score = candidate_metrics.get(METRIC_KEY)
    if candidate_score is None:
        print(f"❌ Metric '{METRIC_KEY}' not found in candidate metrics")
        return False
    
    # First-time deployment (no existing model)
    is_first_deployment = best_metrics is None
    
    if is_first_deployment:
        best_score = float("inf")
        print("📍 First deployment - no existing model to compare")
    else:
        best_score = best_metrics.get(METRIC_KEY, float("inf"))
        print(f"📊 Current best {METRIC_KEY}: {best_score:.4f}")
    
    print(f"📊 Candidate {METRIC_KEY}: {candidate_score:.4f}")
    
    # ============================================================
    # ✅ FIXED: MLflow experiment setup
    # ============================================================
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Get or create experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print(f"📌 Created new MLflow experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        # Restore if deleted
        if experiment.lifecycle_stage != "active":
            mlflow.tracking.MlflowClient().restore_experiment(experiment_id)
            print(f"♻️ Restored deleted experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
    
    # Set experiment explicitly
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Start run with explicit experiment_id
    with mlflow.start_run(run_name="model_promotion_decision", experiment_id=experiment_id) as run:
        
        # Log metrics
        mlflow.log_metric("candidate_score", candidate_score)
        mlflow.log_metric("best_score_before", best_score)
        
        # Log candidate run_id if available
        if "run_id" in candidate_metrics:
            mlflow.log_param("candidate_run_id", candidate_metrics["run_id"])
        
        # Calculate improvement
        if best_score != float("inf"):
            improvement = (best_score - candidate_score) / best_score
            mlflow.log_metric("improvement_pct", improvement * 100)
            print(f"📈 Improvement: {improvement*100:.2f}%")
        else:
            improvement = 1.0  # First deployment always "improves"
        
        # ---------------- DECISION LOGIC ----------------
        should_promote = (
            candidate_score < best_score and 
            (is_first_deployment or improvement >= MIN_IMPROVEMENT)
        )
        
        if should_promote:
            print("\n" + "=" * 60)
            print("🏆 PROMOTING NEW MODEL TO PRODUCTION")
            print("=" * 60)
            
            # Backup current model (if exists)
            backup_path = backup_current_model()
            if backup_path:
                mlflow.log_param("backup_path", backup_path)
            
            # Promote candidate to production
            shutil.copy(CANDIDATE_MODEL_PATH, BEST_MODEL_PATH)
            save_metrics(BEST_METRICS_PATH, candidate_metrics)
            
            print(f"✅ Copied candidate → {BEST_MODEL_PATH}")
            print(f"✅ Updated metrics → {BEST_METRICS_PATH}")
            
            # Version with DVC
            try:
                dvc_track_and_push(BEST_MODEL_PATH)
                dvc_track_and_push(BEST_METRICS_PATH)
                
                # Commit to Git
                commit_msg = (
                    f"Promote model: {METRIC_KEY} improved "
                    f"{best_score:.4f} → {candidate_score:.4f} "
                    f"({improvement*100:.1f}% better)"
                )
                git_commit_and_push(commit_msg)
                
                mlflow.log_param("promotion_status", "success")
                mlflow.log_param("promoted", True)
                mlflow.log_metric("best_score_after", candidate_score)
                
                print("\n✅ Model successfully promoted and versioned!")
                
            except Exception as e:
                print(f"\n❌ Promotion failed during versioning: {e}")
                mlflow.log_param("promotion_status", "failed")
                mlflow.log_param("error", str(e))
                return False
            
            # Cleanup candidate
            cleanup_candidate()
            
            return True
        
        else:
            print("\n" + "=" * 60)
            print("❌ CANDIDATE MODEL REJECTED")
            print("=" * 60)
            
            if candidate_score >= best_score:
                print(f"   Reason: No improvement ({candidate_score:.4f} >= {best_score:.4f})")
            elif improvement < MIN_IMPROVEMENT:
                print(f"   Reason: Improvement too small ({improvement*100:.2f}% < {MIN_IMPROVEMENT*100}%)")
            
            mlflow.log_param("promotion_status", "rejected")
            mlflow.log_param("promoted", False)
            mlflow.log_metric("best_score_after", best_score)
            mlflow.log_param("rejection_reason", 
                           "no_improvement" if candidate_score >= best_score else "insufficient_improvement")
            
            # Cleanup candidate
            cleanup_candidate()
            
            print("\n💡 Suggestion: Try different hyperparameters or collect more data")
            
            return False

# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)