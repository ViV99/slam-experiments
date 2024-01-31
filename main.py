from abc import ABC, abstractmethod

import cv2
import numpy as np


class FeatureExtractor(ABC):
    @abstractmethod
    def get_features(self, img: np.ndarray):
        ...


class Orb2FeatureExtractor(FeatureExtractor):
    def __init__(self, n_features: int = 500) -> None:
        self.orb2 = cv2.ORB.create(nfeatures=n_features)

    def get_features(self, img: np.ndarray):
        keypoints = self.orb2.detect(img)
        descriptors = self.orb2.compute(img, keypoints)
        return keypoints, descriptors


def main():
    orb_extractor = Orb2FeatureExtractor()
    img = np.ndarray([])
    print(orb_extractor.get_features(img))


if __name__ == "__main__":
    main()