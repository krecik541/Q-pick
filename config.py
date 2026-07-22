from pathlib import Path
import os

import torch


PROJECT_ROOT = Path(__file__).resolve().parent

REFERENCE_DIR = PROJECT_ROOT / "reference"
CACHE_DIR = PROJECT_ROOT / "cache"
HF_CACHE_DIR = CACHE_DIR / "huggingface"
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_DIR = PROJECT_ROOT / "results"
IMAGE_DIR = PROJECT_ROOT / "images"

DETECTOR_DIR = PROJECT_ROOT / "detector"
GROUNDINGDINO_DIR = DETECTOR_DIR / "GroundingDINO"
GROUNDINGDINO_CONFIG = DETECTOR_DIR / "models" / "GroundingDINO_SwinT_OGC.py"
GROUNDINGDINO_WEIGHTS = DETECTOR_DIR / "weights" / "groundingdino_swint_ogc.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_MODEL = "yolo11x.pt"
SIGLIP_MODEL = "google/siglip2-base-patch16-224"

CONFIDENCE_THRESHOLD = 0.75
TOP_K = 5

DETECTION_CAPTION = "product . package . bottle . box . shelf item"

CACHE_FILE = CACHE_DIR / "reference_embeddings.npz"
RESULTS_FILE = RESULTS_DIR / "results.json"
DEFAULT_OUTPUT_IMAGE = OUTPUT_DIR / "result.jpg"
DEFAULT_DEBUG_DIR = OUTPUT_DIR / "debug"

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR))


def ensure_runtime_dirs():
    for directory in (
        REFERENCE_DIR,
        CACHE_DIR,
        HF_CACHE_DIR,
        OUTPUT_DIR,
        RESULTS_DIR,
        IMAGE_DIR,
        GROUNDINGDINO_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
