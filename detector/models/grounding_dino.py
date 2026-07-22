import json
import os

import numpy as np
import torch
import cv2
import uuid
from pathlib import Path

from detector.models.detection import Detection
from detector.models.base import BaseDetector
from config import DETECTION_CAPTION, DEVICE, GROUNDINGDINO_CONFIG, GROUNDINGDINO_WEIGHTS
from do_3_razy_sztuka.utils.logger import get_logger
from detector.GroundingDINO.groundingdino.util.inference import load_model, load_image, predict

class GroundingDINODetector(BaseDetector):
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Loading GroundingDINO...")

        self.model = load_model(
            model_config_path=str(GROUNDINGDINO_CONFIG),
            model_checkpoint_path=str(GROUNDINGDINO_WEIGHTS),
            device=DEVICE,
        )

        self.box_threshold = 0.3
        self.text_threshold = 0.25
        self.logger.info("GroundingDINO loaded")

    def detect(self, image):
        """
        image: np.array (BGR or RGB)
        returns: list of boxes
        """
        # GroundingDINO likes RGB
        if image.shape[-1] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Temporary save (GroundingDINO API expects a file path for load_image)
        temp_path = Path(f"temp_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(str(temp_path), image_rgb) # Note: Passing image_rgb here is better since GroundingDINO prefers it

        # safety check (Windows IO race fix)
        while not temp_path.exists():
            pass

        # Load the image once
        image_source, image_tensor = load_image(str(temp_path))

        # Predict - using global predict import directly
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=DETECTION_CAPTION,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=DEVICE,
        )

        # Safe cleanup of the temp file right after inference is done
        temp_path.unlink(missing_ok=True)

        h, w = image.shape[:2]
        detections = []

        for box, score in zip(boxes, logits):
            # 1. Unpack GroundingDINO's native [cx, cy, bw, bh] format
            cx, cy, bw, bh = box

            # 2. Convert center-relative format to top-left/bottom-right [x1, y1, x2, y2]
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            # 3. Scale up to absolute pixel coordinates and clip to image boundaries
            x1 = max(0, int(x1 * w))
            y1 = max(0, int(y1 * h))
            x2 = min(w, int(x2 * w))
            y2 = min(h, int(y2 * h))

            detections.append(
                Detection(
                    id=len(detections),
                    bbox=[x1, y1, x2, y2],
                    confidence=float(score),
                    source="groundingdino"
                )
            )

        for det in detections:
            det.crop = self.crop(image, det)

        return detections

    def crop(self, image, detection, padding=0.1):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = detection.bbox

        dx = int((x2 - x1) * padding)
        dy = int((y2 - y1) * padding)

        x1 = max(0, x1 - dx)
        y1 = max(0, y1 - dy)
        x2 = min(w, x2 + dx)
        y2 = min(h, y2 + dy)

        return image[y1:y2, x1:x2]

    def save_results(self, image, detections, output_dir="output"):

        output_dir = Path(output_dir)
        crops_dir = output_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for i, det in enumerate(detections):
            crop = det.crop if det.crop is not None else self.crop(image, det)

            filename = f"crop_{i:03d}.jpg"
            crop_path = crops_dir / filename

            cv2.imwrite(str(crop_path), crop)

            results.append({
                "id": i,
                "image": filename,
                "bbox": det.bbox,
                "confidence": float(det.confidence),
                "source": det.source,
            })

        json_path = output_dir / "detections.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Saved {len(results)} crops to {output_dir}")

        return {
            "json_path": str(json_path),
            "crops_dir": str(crops_dir),
            "count": len(results)
        }