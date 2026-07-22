import argparse
import cv2
import os
from pathlib import Path

from config import DEFAULT_DEBUG_DIR, DEFAULT_OUTPUT_IMAGE, ensure_runtime_dirs, CONFIDENCE_THRESHOLD
from detector.models.factory import create_detector

from do_3_razy_sztuka.models.embedder import ProductEmbedder
from do_3_razy_sztuka.models.reference_db import ReferenceDatabase
from do_3_razy_sztuka.pipeline.product_localizer import ProductLocalizer
from do_3_razy_sztuka.pipeline.decision_engine import DecisionEngine
from do_3_razy_sztuka.utils.visualization import Visualizer


def main():
    ensure_runtime_dirs()

    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument("--query", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_IMAGE))
    parser.add_argument("--debug_dir", default=str(DEFAULT_DEBUG_DIR))
    parser.add_argument("--detector", default=str("sam2"))


    args = parser.parse_args()

    os.makedirs(args.debug_dir, exist_ok=True)

    # ----------------------------
    # 1. LOAD IMAGE
    # ----------------------------
    image = cv2.imread(args.image)

    # ----------------------------
    # 2. DETECTOR
    # ----------------------------
    print("[STEP] Running detector...")

    detector = create_detector(args.detector)

    detections = detector.detect(image)
    detections = list(filter(lambda d: d.crop is not None, detections))
    detections = detector.nms(detections, iou_threshold=0.6)

    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    save_result = detector.save_results(image, detections, output_dir=str(debug_dir))

    # ----------------------------
    # 3. LOCALIZER
    # ----------------------------
    print("[STEP] Running localizer...")

    embedder = ProductEmbedder()

    db = ReferenceDatabase(embedder)
    db.load()

    localizer = ProductLocalizer(embedder, db)

    detections = localizer.run(detections, target_product=args.query)

    # ----------------------------
    # 4. DECISION LAYER
    # ----------------------------
    engine = DecisionEngine(threshold=CONFIDENCE_THRESHOLD)

    decisions = engine.decide(detections, target_product=args.query)

    dec = []
    # dec = filter(lambda x: x["score"] >= CONFIDENCE_THRESHOLD, dec)
    # dec = filter(lambda x: x["detected_product"] == args.query, dec)

    best = engine.find_best(
        detections,
        target_product=args.query
    )

    # ----------------------------
    # 5. VISUALIZATION
    # ----------------------------
    vis = Visualizer()

    vis_img = vis.draw(image, decisions)

    cv2.imwrite(args.output, vis_img)

    # ----------------------------
    # 6. API RESPONSE STYLE OUTPUT
    # ----------------------------
    response = {
        "query": args.query,
        "best_match": best.best_match if best else None,
        "detections": decisions,
        "debug_dir": str(debug_dir),
        "crops_dir": save_result.get("crops_dir"),
        "output_image": args.output
    }

    print("\n======================")
    print("SYSTEM RESPONSE")
    print("======================")

    print(response)

    return response


if __name__ == "__main__":
    main()