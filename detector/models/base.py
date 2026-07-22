from abc import ABC, abstractmethod
from .detection import Detection


class BaseDetector(ABC):

    @abstractmethod
    def detect(self, image) -> list[Detection]:
        pass

    @abstractmethod
    def save_results(self, image, detections, output_dir):
        pass