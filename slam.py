from typing import Sequence, Optional
from threading import Thread
from queue import Queue

import cv2
import numpy as np
from jaxlie import SE3

from frontend import Frontend
from backend import Backend, Map
from primitives import Camera, Frame
from feature_detectors import OrbFeatureDetector, FeatureDetector
from feature_matchers import BruteForceFeatureMatcher, FeatureMatcher


class OrbSLAM:

    frontend: Frontend
    backend: Optional[Backend]
    message_queue: Optional[Queue]

    def __init__(self, camera: Camera):
        orb_detector = OrbFeatureDetector(n_features=200)
        bf_matcher = BruteForceFeatureMatcher(norm_type=cv2.NORM_HAMMING)
        self.frontend = Frontend(orb_detector, bf_matcher, camera)
        self.backend = None
        self.backend_thread = Thread(target=self.backend_runner)
        self.message_queue = None

    def get_last_pose(self) -> SE3:
        return self.frontend.get_last_frame().pose

    def start(self):
        if self.backend:
            self.backend_thread.start()

    def process(self, img: np.ndarray, timestamp: float):
        frame = Frame.create_frame(img, timestamp)
        self.frontend.add_frame(frame)

    def stop(self):
        if self.backend:
            self.backend_thread.join()

    def backend_runner(self):
        while True:
            primitive = self.message_queue.get()
            if isinstance(primitive, Frame):
                pass
            elif isinstance(primitive, Frame):
                pass


