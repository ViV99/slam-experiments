from typing import Sequence
from abc import ABC, abstractmethod

import numpy as np
from cv2 import KeyPoint, ORB


class FeatureDetector(ABC):
    @abstractmethod
    def detect(self, img: np.ndarray, mask: np.ndarray = None) -> Sequence[KeyPoint]:
        raise NotImplementedError

    @abstractmethod
    def detect_and_compute(self, img: np.ndarray, mask: np.ndarray = None) -> tuple[Sequence[KeyPoint], np.ndarray]:
        raise NotImplementedError


class OrbFeatureDetector(FeatureDetector):
    def __init__(self, n_features: int = 500) -> None:
        self.orb = ORB.create(nfeatures=n_features)

    def detect(self, img: np.ndarray, mask: np.ndarray = None) -> Sequence[KeyPoint]:
        return self.orb.detect(img, mask)

    def detect_and_compute(self, img: np.ndarray, mask: np.ndarray = None) -> tuple[Sequence[KeyPoint], np.ndarray]:
        return self.orb.detectAndCompute(img, mask)
