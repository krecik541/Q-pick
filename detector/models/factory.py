# detectors/factory.py

from functools import lru_cache

from .grounding_dino import GroundingDINODetector
from .sam2_detector import SAM2Detector


@lru_cache(maxsize=None)
def create_detector(name: str):

    if name == "groundingdino":
        return GroundingDINODetector()

    if name == "sam2":
        return SAM2Detector(
            checkpoint="weights/sam2.1_hiera_small.pt",
            config="configs/sam2.1/sam2.1_hiera_s.yaml"
        )

    raise ValueError(f"Unknown detector: {name}")