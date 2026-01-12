import subprocess
import sys

def run_step(script, description):
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print('='*60)

    # Use the SAME Python interpreter that started this script
    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"❌ {description} failed")
        sys.exit(1)

    print(f"✅ {description} complete")

if __name__ == "__main__":
    print("Pipeline Python:", sys.executable)

    run_step("fetch_validate_feedback.py", "Fetch & Validate Data")
    run_step("train.py", "Train Candidate Model")
    run_step("post_training.py", "Evaluate & Promote")

    print("\n🎉 Retraining pipeline complete!")
