from pathlib import Path
import numpy as np
from PIL import Image

from config import (
    REFERENCE_DIR,
    CACHE_DIR,
    CACHE_FILE,
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

        embeddings = np.asarray(embeddings)

        if embeddings.size == 0:
            raise ValueError(
                f"No valid reference images found in {REFERENCE_DIR}"
            )

        np.savez(
            CACHE_FILE,
            embeddings=embeddings,
            products=np.array(products),
            images=np.array(image_names),
        )

        self.logger.info(
            f"Saved {len(embeddings)} embeddings."
        )

    # -------------------------------------------------

    def load(self):

        if not self._cache_has_embeddings():
            self.logger.info(
                "Cache missing or empty, building it from references."
            )
            self.build()

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
