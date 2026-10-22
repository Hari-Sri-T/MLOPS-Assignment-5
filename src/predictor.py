"""Inference wrapper used by the Flask app."""
import io
import json
import os
import threading

import numpy as np
from PIL import Image

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model/model.keras")
CLASSES_PATH = os.getenv("CLASSES_PATH", "artifacts/model/classes.json")


class Predictor:
    """Loads the Keras model once (lazily) and classifies image bytes."""

    def __init__(self, model_path: str = MODEL_PATH, classes_path: str = CLASSES_PATH):
        self.model_path = model_path
        self.classes_path = classes_path
        self._model = None
        self._lock = threading.Lock()
        self.classes = ["cats", "dogs"]
        self.image_size = 160

    @property
    def ready(self) -> bool:
        if not all(os.path.isfile(path) for path in (self.model_path, self.classes_path)):
            return False
        self._load()
        return True

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import tensorflow as tf  # heavy import, done once

                    with open(self.classes_path, encoding="utf-8") as f:
                        meta = json.load(f)
                    self.classes = meta["classes"]
                    self.image_size = meta["image_size"]
                    self._model = tf.keras.models.load_model(self.model_path)
        return self._model

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((self.image_size, self.image_size))
        # Rescaling to [-1, 1] happens inside the model, so pass raw 0-255 pixels
        return np.expand_dims(np.asarray(img, dtype=np.float32), axis=0)

    def predict(self, image_bytes: bytes) -> dict:
        model = self._load()
        x = self.preprocess(image_bytes)
        p_second = float(np.asarray(model(x, training=False)).ravel()[0])  # P(class index 1)
        probs = {self.classes[0]: 1.0 - p_second, self.classes[1]: p_second}
        label = max(probs, key=probs.get)
        return {
            "label": label,
            "confidence": round(probs[label], 4),
            "probabilities": {k: round(v, 4) for k, v in probs.items()},
        }
