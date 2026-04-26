from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf

from app.core.exceptions import ImageDecodeError, InvalidImageError


class ImageProcessor:
    def __init__(self, target_size: int) -> None:
        if target_size <= 0:
            raise ValueError("target_size must be positive")
        self._target_size = target_size

    def decode_to_bgr(self, content: bytes) -> np.ndarray:
        if not content:
            raise InvalidImageError("Empty image content")
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ImageDecodeError("Could not decode image")
        return img

    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr is None or img_bgr.size == 0:
            raise InvalidImageError("Invalid image array")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize(img_rgb, (self._target_size, self._target_size))
        scaled = resized / 255.0
        return scaled.numpy().astype("float32")
