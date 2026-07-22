class DecisionEngine:

    def __init__(self, threshold=0.4):
        self.threshold = threshold

    def decide(self, detections, target_product=None):

        decisions = []

        for det in detections:

            best = det.best_match or {}

            product = best.get("product")
            score = float(best.get("score", 0.0))

            is_match = product is not None and score >= self.threshold
            
            decisions.append({
                "id": det.id,
                "bbox": det.bbox,
                "confidence": det.confidence,
                "detected_product": product,
                "score": score,
                "is_match": is_match,
                "target_match": (product == target_product) if target_product else None
            })

        return decisions

    def find_best(self, detections, target_product=None):

        best = None
        best_score = -1

        for det in detections:

            m = det.best_match or {}
            product = m.get("product")
            score = float(m.get("score", -1))

            if target_product and product != target_product:
                continue

            if score > best_score:
                best_score = score
                best = det

        return best