import os
from keras.models import load_model

def test_model_file_exists():
    assert os.path.exists("models/wildlens_multiclassifier.keras")


def test_model_loading():
    model = load_model("weights/wildlens_multiclassifier.keras")
    assert model is not None


