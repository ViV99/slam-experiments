from __future__ import annotations

import yaml

import cv2
import numpy as np

from frontend import Frontend
from backend import Backend, Map
from primitives import Camera, Frame
from feature_detectors import OrbFeatureDetector, FeatureDetector
from feature_matchers import BruteForceFeatureMatcher, FeatureMatcher

from threading import Thread


class SLAM:
    def __init__(self, frontend: Frontend, backend: Backend = None):
        self.frontend = frontend
        self.backend = backend

    def start(self):
        pass

    def process(self, img: np.ndarray, timestamp: np.float64):
        frame = Frame.create_frame(img, timestamp)
        self.frontend.add_frame(frame)



def main():
    camera_matrix = np.array([[520.9, 0, 325.1],
                              [0, 521., 249.7],
                              [0, 0, 1]])
    orb_detector = OrbFeatureDetector(n_features=200)
    img1 = cv2.imread("1.png", flags=cv2.IMREAD_COLOR)
    img2 = cv2.imread("2.png", flags=cv2.IMREAD_COLOR)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    bf_matcher = BruteForceFeatureMatcher(norm_type=cv2.NORM_HAMMING)

    cam = Camera(521, 521, 325, 250)
    map = Map()
    frontend = Frontend(orb_detector, bf_matcher, map, cam)
    frame1 = Frame.create_frame(img1, np.float64(0.1))
    frame2 = Frame.create_frame(img1, np.float64(0.2))
    frame3 = Frame.create_frame(img1, np.float64(0.3))
    frame4 = Frame.create_frame(img2, np.float64(0.4))
    print(frontend.get_status())
    frontend.add_frame(frame1)
    print(frontend.get_status(), "1 added")
    frontend.add_frame(frame2)
    print(frontend.get_status(), "2 added")
    frontend.add_frame(frame3)
    print(frontend.get_status(), "3 added")
    print(frontend._current_frame.pose)
    frontend.add_frame(frame4)
    print(frontend.get_status())
    print(frontend._current_frame.pose)

     


if __name__ == "__main__":
    np.random.seed(228)
    main()
