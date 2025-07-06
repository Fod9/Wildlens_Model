from scripts.train import complete_training
from scripts.test import test_model
from scripts.dataviz import vizualize
import json
import os

def main():
    # Train the model
    model, history, real_val_ds = complete_training()

    # Test the model
    new_metrics = vizualize(model, real_val_ds)

    if test_model(model, new_metrics):
        os.makedirs("weights", exist_ok=True)
        save_path = "weights/wildlens_multiclassifier.keras"
        model.save(save_path)

        os.makedirs("metrics", exist_ok=True)
        save_metrics_path = "metrics/metrics.json"
        with open(save_metrics_path, "w") as f:
            json.dump(new_metrics, f)
    else:
        print("Model did not pass the test.")


if __name__ == "__main__":
    main()