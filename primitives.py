from __future__ import annotations

from typing import Optional, Any, Sequence
from threading import Lock

import cv2
import numpy as np
from jaxlie import SE3


class Camera:
    __slots__ = ("fx", "fy", "cx", "cy", "intrinsics")

    fx: float
    fy: float
    cx: float
    cy: float
    intrinsics: np.ndarray

    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

    def projection(self, pose: SE3) -> np.ndarray:
        return np.array((self.intrinsics @ pose.as_matrix()[:-1]))

    @staticmethod
    def world_to_camera(p_w: np.ndarray, pose: SE3) -> np.ndarray:
        """
        :param p_w: np.array[3,]
        :param pose: SE3
        :return: np.ndarray[3,]
        """
        return pose @ p_w

    @staticmethod
    def camera_to_world(p_c: np.ndarray, pose: SE3) -> np.ndarray:
        """
        :param p_c: np.array[3,]
        :param pose: SE3
        :return: np.ndarray[3,]
        """
        return pose.inverse() @ p_c

    def camera_to_pixel(self, p_c: Sequence[float]) -> np.ndarray:
        """
        :param p_c: np.array[3,]
        :return: np.ndarray[2,]
        """
        return np.array([
            self.fx * p_c[0] / p_c[2] + self.cx,
            self.fy * p_c[1] / p_c[2] + self.cy
        ], dtype=np.int32)

    def pixel_to_camera(self, p_p: Sequence[float], depth: float = 1) -> np.ndarray:
        """
        :param p_p: np.array[2,]
        :param depth: float
        :return: np.ndarray[3,]
        """
        return np.array([
            (p_p[0] - self.cx) * depth / self.fx,
            (p_p[1] - self.cy) * depth / self.fy,
            depth
        ])

    def pixel_to_world(self, p_p: Sequence[float], pose: SE3, depth: float = 1) -> np.ndarray:
        """
        :param p_p: np.array[2,]
        :param pose: SE3
        :param depth: float
        :return: np.ndarray[3,]
        """
        return self.camera_to_world(self.pixel_to_camera(p_p, depth), pose)

    def world_to_pixel(self, p_w: np.ndarray, pose: SE3) -> np.ndarray:
        """
        :param p_w: np.array[3,]
        :param pose: SE3
        :return: np.ndarray[2,]
        """
        return self.camera_to_pixel(self.world_to_camera(p_w, pose))


class Feature:
    __slots__ = (
        "frame", "keypoint", "map_point", "is_outlier", "descriptor"
    )

    frame: Frame
    keypoint: cv2.KeyPoint
    descriptor: Any
    map_point: Optional[MapPoint]
    is_outlier: bool

    def __init__(self, frame: Frame, kp: cv2.KeyPoint, descriptor):
        self.frame = frame
        self.keypoint = kp
        self.descriptor = descriptor
        self.is_outlier = False
        self.map_point = None

    @property
    def position(self):
        return np.array(self.keypoint.pt, dtype=np.int32)


class MapPoint:
    _factory_id: np.uint64 = 0

    __slots__ = (
        "id", "position", "is_outlier", "_mutex", "observations"
    )

    id: np.uint64
    position: np.ndarray
    is_outlier: bool
    _mutex: Lock
    observations: set[Feature]

    def __init__(self, id: np.uint64 = 0, position: np.ndarray = np.zeros((3,))):
        self.id = id
        self.position = position
        self.is_outlier = False
        self._mutex = Lock()
        self.observations = set()

    def set_position(self, position: np.ndarray):
        with self._mutex:
            self.position = position

    def add_observation(self, feature: Feature):
        with self._mutex:
            self.observations.add(feature)

    def remove_observation(self, feature: Feature):
        with self._mutex:
            if feature in self.observations:
                self.observations.remove(feature)
                feature.map_point = None

    def get_observations(self):
        with self._mutex:
            return self.observations

    @staticmethod
    def create_map_point(position: np.ndarray) -> MapPoint:
        map_point = MapPoint(id=MapPoint._factory_id)
        MapPoint._factory_id += 1
        return map_point


class Frame:
    _factory_id: np.uint64 = 0
    _keyframe_factory_id: np.uint64 = 0

    __slots__ = (
        "id", "keyframe_id", "is_keyframe", "time_stamp", "pose", "_mutex", "img", "features"
    )

    id: np.uint64
    keyframe_id: Optional[np.uint64]
    is_keyframe: bool
    time_stamp: float
    pose: SE3  # Tcw
    _mutex: Lock
    img: np.ndarray
    features: list[Feature]

    def __init__(self, id: np.uint64 = 0, time_stamp: float = 0, pose: SE3 = None, img: np.ndarray = None):
        self.id = id
        self.keyframe_id = None
        self.is_keyframe = False
        self.time_stamp = time_stamp
        self.pose = pose
        self._mutex = Lock()
        self.img = img
        self.features = []

    def set_pose(self, pose: SE3):
        with self._mutex:
            self.pose = pose

    def make_keyframe(self):
        with self._mutex:
            self.is_keyframe = True
            self.keyframe_id = Frame._keyframe_factory_id
            Frame._keyframe_factory_id += 1
            for feature in self.features:
                if feature.map_point:
                    feature.map_point.add_observation(feature)

    def get_descriptors(self) -> np.ndarray:
        with self._mutex:
            descriptors = []
            for feature in self.features:
                descriptors.append(feature.descriptor)
            return np.array(descriptors)

    @staticmethod
    def create_frame(img: np.ndarray, timestamp: float) -> Frame:
        frame = Frame(Frame._factory_id, timestamp, img=img)
        Frame._factory_id += 1
        return frame
