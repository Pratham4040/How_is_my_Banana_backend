from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np


class ModelProtocol(Protocol):
    def predict(self, batch: np.ndarray) -> np.ndarray: ...


class ImageProcessorProtocol(Protocol):
    def decode_to_bgr(self, content: bytes) -> np.ndarray: ...

    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class PredictionResult:
    class_id: int
    label: str
    confidence: float


class PredictorProtocol(Protocol):
    def predict_from_bgr(self, img_bgr: np.ndarray) -> PredictionResult: ...
