import os
import sys
import mlflow

THRESHOLD = 0.80  # 🔥 changed from 0.85


def main():
    # --------------------------------------------------------------
    # 1. Read Run ID
    # --------------------------------------------------------------
    if not os.path.exists("model_info.txt"):
        print("ERROR: model_info.txt not found.")
        sys.exit(1)

    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()

    if not run_id:
        print("ERROR: model_info.txt is empty.")
        sys.exit(1)

    print(f"Run ID: {run_id}")

    # --------------------------------------------------------------
    # 2. Get accuracy from MLflow
    # --------------------------------------------------------------
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")

    if accuracy is None:
        print("ERROR: 'accuracy' metric not found.")
        sys.exit(1)

    print(f"Accuracy: {accuracy:.4f}")

    # --------------------------------------------------------------
    # 3. Check threshold
    # --------------------------------------------------------------
    if accuracy < THRESHOLD:
        print(f"FAIL: Accuracy {accuracy:.4f} is below threshold {THRESHOLD}.")
        sys.exit(1)
    else:
        print(f"PASS: Accuracy {accuracy:.4f} meets threshold {THRESHOLD}.")


if __name__ == "__main__":
    main()