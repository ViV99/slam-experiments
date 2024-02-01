from typing import Sequence, Optional
from abc import ABC, abstractmethod
from operator import attrgetter

import cv2
import numpy as np
from cv2 import KeyPoint, DMatch, NORM_HAMMING


class FeatureExtractor(ABC):
    @abstractmethod
    def get_features(self, img: np.ndarray) -> tuple[Sequence[KeyPoint], np.ndarray]:
        raise NotImplementedError


class OrbFeatureExtractor(FeatureExtractor):
    def __init__(self, n_features: int = 500) -> None:
        self.orb2 = cv2.ORB.create(nfeatures=n_features)

    def get_features(self, img: np.ndarray) -> tuple[Sequence[KeyPoint], np.ndarray]:
        keypoints = self.orb2.detect(img)
        return self.orb2.compute(img, keypoints)


class FeatureMatcher(ABC):
    @abstractmethod
    def match_features(
        self, source_descriptors: np.ndarray, query_descriptors: np.ndarray
    ) -> Sequence:
        raise NotImplementedError

    @classmethod
    def draw_matches(
        cls,
        img1: np.ndarray,
        keypoints1: Sequence[KeyPoint],
        img2: np.ndarray,
        keypoints2: Sequence[KeyPoint],
        matches: Sequence[DMatch]
    ) -> None:
        matches_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
        cv2.imshow("Matches", matches_img)


class BruteForceFeatureMatcher(FeatureMatcher):
    def __init__(self, norm_type: int):
        self.bf = cv2.BFMatcher(normType=norm_type, crossCheck=False)

    def match_features(
        self, source_descriptors: np.ndarray, query_descriptors: np.ndarray, dist_threshold: Optional[float] = None
    ) -> Sequence[DMatch]:
        matches = self.bf.match(query_descriptors, source_descriptors)
        min_dist = min(matches, key=attrgetter("distance")).distance
        if dist_threshold:
            return [m for m in matches if m.distance < max(2 * min_dist, dist_threshold)]
        return matches


def main():
    orb_extractor = OrbFeatureExtractor(n_features=500)
    img1 = cv2.imread("1.png")
    img2 = cv2.imread("2.png")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb_extractor.get_features(img1)
    kp2, des2 = orb_extractor.get_features(img2)

    bf_matcher = BruteForceFeatureMatcher(norm_type=NORM_HAMMING)
    matches = bf_matcher.match_features(des1, des2, 30)
    bf_matcher.draw_matches(img1, kp1, img2, kp2, matches)
    cv2.waitKey()


if __name__ == "__main__":
    main()
