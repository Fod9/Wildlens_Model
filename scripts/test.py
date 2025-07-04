import os.path
import json
from tensorflow.keras.models import load_model
import numpy as np

def test_prediction_output_shape():
    model = load_model("weights/wildlens_multiclassifier.keras")
    dummy_input = np.random.rand(1, 224, 224, 3)
    prediction = model.predict(dummy_input)
    return prediction.shape[1] >= 13

def test_model(model, new_metrics):
    if not test_prediction_output_shape():
        return False

    metrics_path = "metrics/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            old_metrics = json.load(f)
        return old_metrics.get("val_loss", float("inf")) >= new_metrics.get("val_loss", float("inf"))
    return True

