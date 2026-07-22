import torch
import numpy as np
from PIL import Image
from functools import lru_cache
from transformers import AutoProcessor, SiglipVisionModel

from config import DEVICE, SIGLIP_MODEL, HF_CACHE_DIR
from do_3_razy_sztuka.utils.logger import get_logger


@lru_cache(maxsize=1)
def _load_siglip_components():
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL, cache_dir=HF_CACHE_DIR)
    model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL, cache_dir=HF_CACHE_DIR)
    return processor, model


class ProductEmbedder:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

        self.logger.info(f"Loading model: {SIGLIP_MODEL}")

        self.device = DEVICE

        self.processor, self.model = _load_siglip_components()

        self.model.to(self.device)
        self.model.eval()

        self.logger.info(f"Model loaded on {self.device}")

    def embed_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        inputs = self.processor(images=image, return_tensors="pt")

        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            emb = outputs.last_hidden_state[:, 0, :]

        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

        return emb.squeeze(0).cpu().numpy()

    def embed_batch(self, images):
        return np.array([self.embed_image(img) for img in images])