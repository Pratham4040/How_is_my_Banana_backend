from __future__ import annotations

import os
import numpy as np

from app.core.exceptions import PredictionError


class TensorFlowModel:
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._model = self._load_model(model_path)

    def _load_model(self, model_path: str):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        try:
            import tensorflow as tf

            tf.config.set_visible_devices([], "GPU")
            return tf.keras.models.load_model(model_path)
        except Exception as exc:
            raise PredictionError("Failed to load model") from exc

    def predict(self, batch: np.ndarray) -> np.ndarray:
        try:
            return self._model.predict(batch)
        except Exception as exc:
            raise PredictionError("Model prediction failed") from exc
