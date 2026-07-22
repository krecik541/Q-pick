import cv2


class Visualizer:

    def draw(self, image, detections):
        return self.draw_results(image, detections)

    def draw_results(self, image, detections):
        """
        detections: Detection[] albo decyzje z DecisionEngine
        """

        img = image.copy()

        for det in detections:

            if isinstance(det, dict):
                bbox = det["bbox"]
                best_match = det.get("best_match") or det.get("match")
                confidence = float(det.get("confidence", 0.0))
                label = det.get("detected_product")
                score = float(det.get("score", 0.0))
            else:
                bbox = det.bbox
                best_match = det.best_match or {}
                confidence = float(det.confidence)
                label = best_match.get("product")
                score = float(best_match.get("score", 0.0))

            x1, y1, x2, y2 = bbox

            label = label or "unknown"

            text = f"{label} ({score:.2f}) [{confidence:.2f}]"

            # bbox
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # label background
            cv2.rectangle(
                img,
                (x1, y1 - 20),
                (x1 + 200, y1),
                (0, 255, 0),
                -1
            )

            cv2.putText(
                img,
                text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )

        return img