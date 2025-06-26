import mlflow
import subprocess
import time
import webbrowser
from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    # Get the MLflow tracking URI
    tracking_uri = Client().active_stack.experiment_tracker.get_tracking_uri()
    print("MLflow Tracking URI:", tracking_uri)

    # Start MLflow UI
    subprocess.Popen(["mlflow", "ui", "--backend-store-uri", tracking_uri])

    # Optional: Open MLflow UI in browser after short delay
    time.sleep(7)
    webbrowser.open("http://127.0.0.1:5000")

    # Run your pipeline
    train_pipeline(data_path="data\\olist_customers_dataset.csv")
