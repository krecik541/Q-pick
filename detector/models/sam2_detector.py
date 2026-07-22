import json
from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from .base import BaseDetector
from .detection import Detection
from do_3_razy_sztuka.utils.logger import get_logger

class SAM2Detector(BaseDetector):

    def __init__(
        self,
        checkpoint,
        config,
        device="cpu"
    ):

        self.device = device

        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Loading SAM2...")

        self.model = build_sam2(
            config,
            checkpoint,
            device=device
        )

        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=500
        )
        self.logger.info("SAM2 loaded")

    def detect(self, image):

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = self.mask_generator.generate(image_rgb)

        H, W = image.shape[:2]

        detections = []

        max_area = H * W * 0.30

        for idx, m in enumerate(masks):

            area = m["area"]

            if area < 500:
                continue

            if area > max_area:
                continue

            x, y, w, h = m["bbox"]

            det = Detection(
                id=len(detections),

                bbox=[
                    int(x),
                    int(y),
                    int(x + w),
                    int(y + h)
                ],

                confidence=float(m["predicted_iou"]),

                source="sam2",
                
                area=area,

                mask=m["segmentation"]
            )

            detections.append(det)
        
        self.logger.info(f"SAM2 generated {len(masks)} masks")
        self.logger.info(f"After filtering: {len(detections)} detections")

        detections.sort(
            key=lambda d: d.confidence,
            reverse=True
        )

        filtered = []

        detections = sorted(
            detections,
            key=lambda d: d.confidence,
            reverse=True
        )

        for det in detections:

            keep = True

            for other in filtered:

                if self.bbox_iou(det.bbox, other.bbox) > 0.6:
                    keep = False
                    break

            if keep:
                filtered.append(det)

        self.logger.info(f"SAM2 generated {len(masks)} masks")
        self.logger.info(f"After NMS: {len(filtered)} detections")

        for det in filtered:
            det.crop = self.crop(image, det)
            if det.crop is None:
                filtered.remove(det)

        return filtered

    def crop(self, image, detection):

        x1, y1, x2, y2 = detection.bbox
        mask = detection.mask[y1:y2, x1:x2]

        coverage = mask.mean()

        if coverage < 0.25 or coverage > 0.95:
            return None

        mask_uint8 = mask.astype(np.uint8) * 255

        x, y, w, h = cv2.boundingRect(mask_uint8)

        crop = image[y:y+h, x:x+w]

        local_mask = mask[y:y+h, x:x+w]

        crop = crop.copy()
        crop[~local_mask] = (255,255,255)

        # crop[~mask] = 255

        crop = cv2.copyMakeBorder(
            crop,
            15,
            15,
            15,
            15,
            cv2.BORDER_CONSTANT,
            value=(255,255,255)
        )

        w = x2 - x1
        h = y2 - y1

        if min(w, h) < 20:
            return None

        return crop
    
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

    @staticmethod
    def bbox_iou(box1, box2):
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        inter = max(0, xB - xA) * max(0, yB - yA)

        area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
        area2 = (box2[2]-box2[0])*(box2[3]-box2[1])

        union = area1 + area2 - inter

        if union == 0:
            return 0

        return inter / union
    
    @staticmethod
    def nms(detections, iou_threshold=0.6):

        detections = sorted(
            detections,
            key=lambda d: d.confidence,
            reverse=True,
        )

        keep = []

        while detections:

            best = detections.pop(0)
            keep.append(best)

            detections = [
                d for d in detections
                if SAM2Detector.bbox_iou(best.bbox, d.bbox) < iou_threshold
            ]

        return keep