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

# Global toggle to enable/disable saving/reading local caches (set False to disable caching)
ENABLE_CACHING = False

# Allow toggling per-class overrides via environment variable (useful for experiments)
USE_PER_CLASS = os.environ.get('USE_PER_CLASS', '1') == '1'

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

# Inference / postprocessing
NMS_TYPE = "wbf"  # one of: "nms", "soft-nms", "wbf"
NMS_IOU_THRESHOLD = 0.5
SOFT_NMS_SIGMA = 0.5

# Weighted Box Fusion parameters
WBF_IOU_THRESHOLD = 0.55
WBF_SCORE_THRESHOLD = 0.05

# Tiling / small object handling
TILE_INFERENCE = False
TILE_SIZE = 1024
TILE_OVERLAP = 0.2

# Embedding / matching thresholds (used by product localizer)
EMBED_MATCH_THRESHOLD = 0.60
EMBED_MARGIN_THRESHOLD = 0.08
EMBED_FINAL_SCORE_THRESHOLD = 0.55

# Crop refinement & scoring
CROP_PADDING = 0.10  # fraction to expand bbox when creating crop
MIN_CROP_AREA = 400  # px^2, ignore tiny crops
CROP_SHARPNESS_WEIGHT = 1.0
CROP_SIZE_WEIGHT = 0.8
CROP_ASPECT_WEIGHT = 0.5
CROP_EMBED_WEIGHT = 1.2
TOP_K_CROPS = 5

# If True, small crops below MIN_CROP_AREA are upscaled to a usable size
CROP_UPSCALE = True

# Scoring weights for final crop/product ranking
EMBED_WEIGHT = 2.0
MARGIN_WEIGHT = 1.0
CROP_QUALITY_WEIGHT = 1.0
COLOR_HIST_WEIGHT = 2.5

# Final score threshold to accept a detection (combined score)
FINAL_SCORE_THRESHOLD = 0.65

# Per-class overrides: map product identifier -> dict of weight/threshold overrides
# Example: give much higher color weight for 'lays_paprika'
PER_CLASS_CONFIG = {
    "lays_paprika": {
        "COLOR_HIST_WEIGHT": 4.0,
        "EMBED_WEIGHT": 2.0,
        "MARGIN_WEIGHT": 1.0,
        "CROP_QUALITY_WEIGHT": 1.0,
        "FINAL_SCORE_THRESHOLD": 0.60,
    },
    # add more product-specific overrides here as needed
}
