from collections import defaultdict

import numpy as np


class ProductMatcher:

    def __init__(self, threshold=0.75):
        self.threshold = threshold

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def search_topk(
        self,
        embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        metadata: list,
        k: int = 10,
    ):
        """
        embedding             : (768,)
        candidate_embeddings  : (N,768)
        metadata              : list(dict)

        Returns top-k matches.
        """

        if candidate_embeddings.size == 0:
            return []

        scores = candidate_embeddings @ embedding

        order = np.argsort(scores)[::-1][:k]

        results = []

        for idx in order:

            meta = metadata[idx]

            results.append(
                {
                    "index": int(idx),
                    "product": meta["product"],
                    "image": meta["image"],
                    "score": float(scores[idx]),
                }
            )

        return results

    def vote(self, topk):

        votes = defaultdict(float)

        for i, result in enumerate(topk):

            weight = len(topk) - i

            votes[result["product"]] += result["score"] * weight
            
        averages = {
            product: float(np.mean(scores))
            for product, scores in votes.items()
        }

        best_product = max(
            averages,
            key=averages.get,
        )

        references = [
            r for r in topk
            if r["product"] == best_product
        ]

        return {
            "product": best_product,
            "score": averages[best_product],
            "references": references,
        }

    def match(
        self,
        embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        metadata: list,
        k: int = 10,
    ):
        """
        Full matching pipeline.

        embedding -> top-k -> voting -> threshold
        """

        topk = self.search_topk(
            embedding,
            candidate_embeddings,
            metadata,
            k=k,
        )

        if not topk:
            return {
                "product": None,
                "score": 0.0,
                "references": [],
                "topk": [],
            }

        result = self.vote(topk)

        if result["score"] < self.threshold:

            result["product"] = None

        result["topk"] = topk

        return result