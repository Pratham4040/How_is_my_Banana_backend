from __future__ import annotations

import numpy as np

from app.core.exceptions import PredictionError
from app.domain.labels import get_label
from app.domain.types import ImageProcessorProtocol, ModelProtocol, PredictionResult


class BananaPredictor:
    def __init__(self, model: ModelProtocol, image_processor: ImageProcessorProtocol) -> None:
        self._model = model
        self._image_processor = image_processor

    def predict_from_bgr(self, img_bgr: np.ndarray) -> PredictionResult:
        try:
            processed = self._image_processor.preprocess(img_bgr)
            preds = self._model.predict(np.expand_dims(processed, 0))
            if preds is None or len(preds) == 0:
                raise PredictionError("Empty prediction output")
            class_id = int(np.argmax(preds))
            confidence = float(np.max(preds))
            label = get_label(class_id)
            return PredictionResult(class_id=class_id, label=label, confidence=confidence)
        except Exception as exc:
            raise PredictionError("Prediction failed") from exc
