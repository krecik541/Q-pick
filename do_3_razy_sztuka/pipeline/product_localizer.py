from PIL import Image
import cv2
from collections import Counter

from do_3_razy_sztuka.models.matcher import ProductMatcher


class ProductLocalizer:

    def __init__(self, embedder, reference_db):

        self.embedder = embedder
        self.db = reference_db
        self.embeddings = reference_db.embeddings
        self.metadata = reference_db.metadata
        self.matcher = ProductMatcher()

    # -------------------------------------------------

    def run(self, detections, target_product=None):

        # jeśli użytkownik podał konkretny produkt,
        # pobieramy tylko jego prototyp
        prototype = None

        if target_product is not None:
            prototype = self.db.prototypes.get(target_product)

            if prototype is None:
                available_products = ", ".join(sorted(self.db.prototypes.keys()))
                raise ValueError(
                    f"Unknown target product '{target_product}'. "
                    f"Available products: {available_products}"
                )

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            if det.crop is None:
                raise ValueError(f"Detection {det.id} is missing crop data")

            image = Image.fromarray(
                cv2.cvtColor(det.crop, cv2.COLOR_BGR2RGB)
            )

            embedding = self.embedder.embed_image(image)

            det.embedding = embedding

            # ---------- tryb wyszukiwania jednego produktu ----------
            if prototype is not None:

                score = float(embedding @ prototype)

                if score < 0.60:
                    det.best_match = None
                else:
                    det.best_match = {
                        "product": target_product,
                        "score": score,
                    }

            # ---------- tryb klasyfikacji (opcjonalny) ----------
            else:

                scores = {
                    product: float(embedding @ proto)
                    for product, proto in self.db.prototypes.items()
                }

                ordered = sorted(
                    scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                best_product, best_score = ordered[0]

                second_score = ordered[1][1] if len(ordered) > 1 else -1.0

                margin = best_score - second_score

                if best_score < 0.60:
                    det.best_match = None
                    continue

                if margin < 0.08:
                    det.best_match = None
                    continue

                det.best_match = {
                    "product": best_product,
                    "score": best_score,
                    "margin": margin,
                    "scores": scores
                }


            w = x2 - x1
            h = y2 - y1
            if det.best_match is not None:
                print(
                    det.id,
                    w,
                    h,
                    w*h,
                    round(w/h,2),
                    round(det.best_match["score"],3)
                )
        # -----------------------------
        # wybór dominującego produktu
        # -----------------------------

        counter = Counter()

        for det in detections:

            if det.best_match is None:
                continue

            counter[det.best_match["product"]] += 1

        if counter:

            winner = counter.most_common(1)[0][0]

            for det in detections:

                if det.best_match is None:
                    continue

                if det.best_match["product"] != winner:
                    det.best_match = None

        detections = [
            d for d in detections
            if d.best_match and d.best_match["score"] > 0.55
        ]

        detections.sort(
            key=lambda d: d.best_match["score"],
            reverse=True,
        )

        # detections = detections[:10]

        return detections