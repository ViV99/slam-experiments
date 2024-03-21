from typing import Sequence
from abc import ABC, abstractmethod

import numpy as np
from cv2 import KeyPoint, ORB


class FeatureDetector(ABC):
    @abstractmethod
    def get_features(self, img: np.ndarray) -> tuple[Sequence[KeyPoint], np.ndarray]:
        raise NotImplementedError


class OrbFeatureDetector(FeatureDetector):
    def __init__(self, n_features: int = 500) -> None:
        self.orb = ORB.create(nfeatures=n_features)

    def get_features(self, img: np.ndarray) -> tuple[Sequence[KeyPoint], np.ndarray]:
        return self.orb.detectAndCompute(img, None)
