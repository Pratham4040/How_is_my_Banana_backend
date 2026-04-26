from __future__ import annotations

from typing import List


CLASS_LABELS: List[str] = ["0", "2-1", "4-3", "5-6", "7-8"]


def get_label(class_id: int) -> str:
    if class_id < 0 or class_id >= len(CLASS_LABELS):
        raise ValueError("class_id out of range")
    return CLASS_LABELS[class_id]
