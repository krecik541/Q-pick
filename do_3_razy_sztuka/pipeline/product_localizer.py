from PIL import Image
import cv2
from collections import Counter
from typing import List

from do_3_razy_sztuka.models.matcher import ProductMatcher
import config as cfg
import numpy as np


def box_iou(a, b):
    # a, b = (x1,y1,x2,y2)
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    areaB = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union


def wbf_merge(detections: List[object], iou_thr=0.55, score_thr=0.05):
    if not detections:
        return []

    # collect boxes and scores
    boxes = [tuple(d.bbox) for d in detections]
    scores = [getattr(d, 'confidence', getattr(d, 'score', 1.0)) for d in detections]
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    visited = [False] * len(boxes)
    merged = []

    for idx in order:
        if visited[idx]:
            continue
        if scores[idx] < score_thr:
            continue
        cluster = [idx]
        visited[idx] = True
        for j in order:
            if visited[j]:
                continue
            if scores[j] < score_thr:
                continue
            if box_iou(boxes[idx], boxes[j]) >= iou_thr:
                cluster.append(j)
                visited[j] = True

        # weighted average coordinates
        weights = [scores[k] for k in cluster]
        s = sum(weights) if sum(weights) > 0 else 1.0
        x1 = sum(boxes[k][0] * w for k, w in zip(cluster, weights)) / s
        y1 = sum(boxes[k][1] * w for k, w in zip(cluster, weights)) / s
        x2 = sum(boxes[k][2] * w for k, w in zip(cluster, weights)) / s
        y2 = sum(boxes[k][3] * w for k, w in zip(cluster, weights)) / s

        # choose representative detection (highest score) and update bbox/conf
        rep = max(cluster, key=lambda k: scores[k])
        det = detections[rep]
        det.bbox = (int(x1), int(y1), int(x2), int(y2))
        det.confidence = max(scores[k] for k in cluster)
        merged.append(det)

    return merged


def laplacian_variance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_quality_score(img):
    # img: BGR numpy array
    h, w = img.shape[:2]
    area = float(w * h)

    # sharpness normalized
    sharp = laplacian_variance(img)
    sharp_norm = min(1.0, sharp / 100.0)

    # size normalized (favor reasonably large crops)
    size_norm = min(1.0, area / 20000.0)

    # aspect ratio: prefer near-square or typical product shapes
    ar = (w / h) if h > 0 else 1.0
    aspect_norm = float(np.exp(-abs(ar - 1.0)))

    w_sh = cfg.CROP_SHARPNESS_WEIGHT
    w_sz = cfg.CROP_SIZE_WEIGHT
    w_ar = cfg.CROP_ASPECT_WEIGHT

    total = w_sh + w_sz + w_ar
    score = (w_sh * sharp_norm + w_sz * size_norm + w_ar * aspect_norm) / total
    return float(score)


def color_histogram(img):
    # compute H channel histogram with 16 bins, return normalized vector
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        hist = cv2.calcHist([h], [0], None, [16], [0, 180]).flatten()
        if hist.sum() > 0:
            hist = hist / hist.sum()
        return hist
    except Exception:
        return None


def _get_weights_for_product(product):
    # return tuple: (embed_w, margin_w, crop_w, color_w, final_threshold)
    embed_w = cfg.EMBED_WEIGHT
    margin_w = cfg.MARGIN_WEIGHT
    crop_w = cfg.CROP_QUALITY_WEIGHT
    color_w = cfg.COLOR_HIST_WEIGHT
    thresh = cfg.FINAL_SCORE_THRESHOLD

    try:
        if product and getattr(cfg, 'USE_PER_CLASS', True) and hasattr(cfg, 'PER_CLASS_CONFIG'):
            pc = cfg.PER_CLASS_CONFIG.get(product)
            if pc:
                embed_w = pc.get('EMBED_WEIGHT', embed_w)
                margin_w = pc.get('MARGIN_WEIGHT', margin_w)
                crop_w = pc.get('CROP_QUALITY_WEIGHT', crop_w)
                color_w = pc.get('COLOR_HIST_WEIGHT', color_w)
                thresh = pc.get('FINAL_SCORE_THRESHOLD', thresh)
    except Exception:
        pass

    return float(embed_w), float(margin_w), float(crop_w), float(color_w), float(thresh)


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

        # apply postprocessing merge (WBF) early to reduce duplicates
        if cfg.NMS_TYPE == 'wbf':
            detections = wbf_merge(detections, iou_thr=cfg.WBF_IOU_THRESHOLD, score_thr=cfg.WBF_SCORE_THRESHOLD)

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            if det.crop is None:
                raise ValueError(f"Detection {det.id} is missing crop data")

            # compute crop quality (sharpness / size / aspect)
            try:
                det.crop_score = crop_quality_score(det.crop)
            except Exception:
                det.crop_score = 0.0

            image = Image.fromarray(
                cv2.cvtColor(det.crop, cv2.COLOR_BGR2RGB)
            )

            embedding = self.embedder.embed_image(image)

            det.embedding = embedding
            # compute color histogram for crop
            det.color_hist = color_histogram(det.crop)

            # ---------- tryb wyszukiwania jednego produktu ----------
            if prototype is not None:
                score = float(embedding @ prototype)
                if score < cfg.EMBED_MATCH_THRESHOLD:
                    det.best_match = None
                else:
                    det.best_match = {
                        "product": target_product,
                        "score": score,
                        "embed_score": score,
                        "margin": None,
                        "color_score": color_score,
                    }
                # color similarity to target product
                color_score = 0.0
                prod_hist = self.db.product_histograms.get(target_product) if hasattr(self.db, 'product_histograms') else None
                if det.color_hist is not None and prod_hist is not None:
                    corr = cv2.compareHist(prod_hist.astype('float32'), det.color_hist.astype('float32'), cv2.HISTCMP_CORREL)
                    color_score = float((corr + 1.0) / 2.0)

                # combine embedding score with crop quality and color similarity into final_score
                embed_w, margin_w, crop_w, color_w, effective_thresh = _get_weights_for_product(target_product)
                if det.best_match is not None:
                    embed_score = max(0.0, det.best_match["score"])
                    margin_norm = 0.0
                    det.final_score = (
                        embed_w * embed_score +
                        margin_w * margin_norm +
                        crop_w * det.crop_score +
                        color_w * color_score
                    ) / (embed_w + margin_w + crop_w + color_w)
                else:
                    det.final_score = det.crop_score

            # ---------- tryb klasyfikacji (opcjonalny) ----------
            else:

                scores = {
                    product: float(embedding @ proto)
                    for product, proto in self.db.prototypes.items()
                }

                # compute color similarity for each product if histograms exist
                color_scores = {}
                for product in scores.keys():
                    color_scores[product] = 0.0
                    prod_hist = None
                    if hasattr(self.db, 'product_histograms'):
                        prod_hist = self.db.product_histograms.get(product)
                    if det.color_hist is not None and prod_hist is not None:
                        corr = cv2.compareHist(prod_hist.astype('float32'), det.color_hist.astype('float32'), cv2.HISTCMP_CORREL)
                        color_scores[product] = float((corr + 1.0) / 2.0)

                ordered = sorted(
                    scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                best_product, best_score = ordered[0]

                second_score = ordered[1][1] if len(ordered) > 1 else -1.0

                margin = best_score - second_score


                if best_score < cfg.EMBED_MATCH_THRESHOLD:
                    det.best_match = None
                    det.final_score = det.crop_score
                    continue

                if margin < cfg.EMBED_MARGIN_THRESHOLD:
                    det.best_match = None
                    det.final_score = det.crop_score
                    continue

                det.best_match = {
                    "product": best_product,
                    "score": best_score,
                    "margin": margin,
                    "scores": scores,
                    "color_score": color_scores.get(best_product, 0.0)
                }

                # combine embedding, margin, crop quality and color similarity into final_score
                embed_norm = max(0.0, best_score)
                margin_norm = max(0.0, margin)
                color_sc = color_scores.get(best_product, 0.0)

                # apply per-product overrides if present
                embed_w, margin_w, crop_w, color_w, effective_thresh = _get_weights_for_product(best_product)

                det.final_score = (
                    embed_w * embed_norm +
                    margin_w * margin_norm +
                    crop_w * det.crop_score +
                    color_w * color_sc
                ) / (embed_w + margin_w + crop_w + color_w)


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

        # filter using per-class threshold when available
        def _passes_threshold(d):
            if not d.best_match:
                return False
            prod = d.best_match.get('product')
            thr = cfg.FINAL_SCORE_THRESHOLD
            try:
                if prod and hasattr(cfg, 'PER_CLASS_CONFIG'):
                    thr = cfg.PER_CLASS_CONFIG.get(prod, {}).get('FINAL_SCORE_THRESHOLD', thr)
            except Exception:
                pass
            return float(getattr(d, 'final_score', d.best_match.get('score', 0.0))) > float(thr)

        detections = [d for d in detections if _passes_threshold(d)]

        detections.sort(
            key=lambda d: getattr(d, 'final_score', d.best_match.get("score", 0.0)),
            reverse=True,
        )

        # detections = detections[:10]

        return detections