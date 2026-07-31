from pathlib import Path
import numpy as np
from PIL import Image

from config import (
    REFERENCE_DIR,
    CACHE_DIR,
    CACHE_FILE,
    ENABLE_CACHING,
)
from do_3_razy_sztuka.utils.logger import get_logger


class ReferenceDatabase:

    def __init__(self, embedder):

        self.embedder = embedder

        self.logger = get_logger(self.__class__.__name__)

        self.embeddings = None
        self.metadata = None
        self.products = None
        self.prototypes = {}

    def _cache_has_embeddings(self):
        if not ENABLE_CACHING:
            return False

        if not CACHE_FILE.exists():
            return False

        try:
            with np.load(CACHE_FILE, allow_pickle=True) as data:
                embeddings = data["embeddings"]
                return embeddings.ndim == 2 and embeddings.shape[0] > 0
        except Exception:
            return False

    # -------------------------------------------------

    def build(self):
        if self._cache_has_embeddings():
            self.logger.info("Cache already exists.")
            # if caching is enabled and cache exists, nothing to build
            return

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        embeddings = []
        products = []
        image_names = []

        product_dirs = sorted(
            [d for d in REFERENCE_DIR.iterdir() if d.is_dir()]
        )

        self.logger.info(
            f"Found {len(product_dirs)} product folders."
        )

        for product_dir in product_dirs:

            product_name = product_dir.name

            images = list(product_dir.glob("*"))

            for img_path in images:

                if img_path.suffix.lower() not in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp",
                ]:
                    continue

                image = Image.open(img_path).convert("RGB")

                emb = self.embedder.embed_image(image)
                embeddings.append(emb)
                products.append(product_name)
                image_names.append(img_path.name)

                # compute simple HSV hue histogram for the image (H channel only)
                try:
                    import cv2
                    import numpy as _np
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is None:
                        img_bgr = _np.asarray(image)[:, :, ::-1].copy()
                    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
                    h = hsv[:, :, 0]
                    hist = cv2.calcHist([h], [0], None, [16], [0, 180]).flatten()
                    if hist.sum() > 0:
                        hist = hist / hist.sum()
                    else:
                        hist = hist
                except Exception:
                    hist = None

                # store per-image histograms in parallel list
                if 'image_hists' not in locals():
                    image_hists = []
                image_hists.append(hist)

        embeddings = np.asarray(embeddings)

        if embeddings.size == 0:
            raise ValueError(
                f"No valid reference images found in {REFERENCE_DIR}"
            )

        # always populate in-memory structures
        self.embeddings = embeddings
        self.products = np.asarray(products)
        self.metadata = [
            {"product": str(p), "image": str(i)} for p, i in zip(products, image_names)
        ]

        # compute prototypes
        self.prototypes = {}
        for product in np.unique(self.products):
            idx = self.products == product
            proto = self.embeddings[idx].mean(axis=0)
            proto /= np.linalg.norm(proto)
            self.prototypes[str(product)] = proto

        # compute per-product average histogram if available
        self.product_histograms = {}
        try:
            unique_products = np.unique(self.products)
            for up in unique_products:
                inds = np.where(self.products == up)[0]
                hlist = [image_hists[i] for i in inds if image_hists[i] is not None]
                if hlist:
                    avg = np.mean(hlist, axis=0)
                    if avg.sum() > 0:
                        avg = avg / avg.sum()
                    self.product_histograms[str(up)] = avg
                else:
                    self.product_histograms[str(up)] = None
        except Exception:
            self.product_histograms = {}

        # save to disk only if caching enabled
        if ENABLE_CACHING:
            # Save product histograms too (store as prod_names + arrays)
            prod_names = np.array(list(self.product_histograms.keys()))
            prod_hists = np.array([self.product_histograms[p] if self.product_histograms[p] is not None else np.zeros(16) for p in prod_names])

            np.savez(
                CACHE_FILE,
                embeddings=self.embeddings,
                products=np.array(products),
                images=np.array(image_names),
                product_names=prod_names,
                product_histograms=prod_hists,
            )

            self.logger.info(
                f"Saved {len(self.embeddings)} embeddings."
            )

    # -------------------------------------------------

    def load(self):

        if not self._cache_has_embeddings():
            self.logger.info(
                "Cache missing or empty, building it from references."
            )
            self.build()

        if ENABLE_CACHING:
            data = np.load(CACHE_FILE, allow_pickle=True)

            self.embeddings = data["embeddings"]

            products = data["products"]
            images = data["images"]
            self.products = np.asarray(products)

            self.prototypes = {}

            self.metadata = []

            for product, image in zip(products, images):
                self.metadata.append(
                    {
                        "product": str(product),
                        "image": str(image)
                    }
                )

            self.logger.info(
                f"Loaded {len(self.embeddings)} embeddings."
            )

            for product in np.unique(self.products):

                idx = self.products == product

                proto = self.embeddings[idx].mean(axis=0)

                proto /= np.linalg.norm(proto)

                self.prototypes[str(product)] = proto
            # load product histograms if present
            try:
                if 'product_names' in data and 'product_histograms' in data:
                    prod_names = data['product_names']
                    prod_hists = data['product_histograms']
                    for name, hist in zip(prod_names, prod_hists):
                        self.product_histograms[str(name)] = hist / (hist.sum() if hist.sum() > 0 else 1)
                else:
                    # fallback: compute from images on disk
                    self.product_histograms = {}
                    for entry in self.metadata:
                        pass
            except Exception:
                self.product_histograms = {}
        else:
            # if caching disabled, build() already populated in-memory structures
            if self.embeddings is None:
                raise RuntimeError("Reference embeddings not built and caching is disabled.")

            self.logger.info(f"Built {len(self.embeddings)} embeddings in-memory (caching disabled).")
