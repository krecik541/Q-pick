from dataclasses import dataclass
import numpy as np


@dataclass
class Detection:
    id: int
    bbox: list[int]
    confidence: float
    source: str
    area: int | None = None
    mask: np.ndarray | None = None
    crop: np.ndarray | None = None
    embedding: np.ndarray | None = None
    matches: list | None = None
    best_match: dict | None = None