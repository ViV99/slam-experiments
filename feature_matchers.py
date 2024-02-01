from typing import Sequence, Optional
from operator import attrgetter
from abc import ABC, abstractmethod

import numpy as np
from cv2 import KeyPoint, DMatch, BFMatcher, drawMatches, imshow


class FeatureMatcher(ABC):
    @abstractmethod
    def match_features(
        self, source_descriptors: np.ndarray, query_descriptors: np.ndarray
    ) -> Sequence:
        raise NotImplementedError

    @classmethod
    def draw_matches(
        cls,
        source_img: np.ndarray,
        source_keypoints: Sequence[KeyPoint],
        query_img: np.ndarray,
        query_keypoints: Sequence[KeyPoint],
        matches: Sequence[DMatch]
    ) -> None:
        matches_img = drawMatches(query_img, query_keypoints, source_img, source_keypoints, matches, None)
        imshow("Matches", matches_img)


class BruteForceFeatureMatcher(FeatureMatcher):
    def __init__(self, norm_type: int):
        self.bf = BFMatcher(normType=norm_type)

    def match_features(
        self, source_descriptors: np.ndarray, query_descriptors: np.ndarray, dist_threshold: Optional[float] = None
    ) -> Sequence[DMatch]:
        matches = self.bf.match(query_descriptors, source_descriptors)
        min_dist = min(matches, key=attrgetter("distance")).distance
        if dist_threshold:
            return [m for m in matches if m.distance < max(2 * min_dist, dist_threshold)]
        return matches