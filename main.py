from __future__ import annotations

import logging as logger
from typing import Sequence, Optional

import cv2
import numpy as np
from cv2 import NORM_HAMMING, KeyPoint, DMatch, findEssentialMat, findHomography, recoverPose, triangulatePoints
from jaxlie import SO3, SE3

from feature_exractors import OrbFeatureExtractor
from feature_matchers import BruteForceFeatureMatcher

from multiprocessing import Process, Queue, Lock


def pose_estimation_2d2d(
    source_keypoints: Sequence[KeyPoint],
    query_keypoints: Sequence[KeyPoint],
    matches: Sequence[DMatch],
    camera_matrix: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    source_pts, query_pts = [], []
    for match in matches:
        source_pts.append(source_keypoints[match.trainIdx].pt)
        query_pts.append(query_keypoints[match.queryIdx].pt)

    source_pts = np.array(source_pts)
    query_pts = np.array(query_pts)

    if camera_matrix is not None:
        essential_matrix, _ = findEssentialMat(source_pts, query_pts, cameraMatrix=camera_matrix)
        _, R, t, _ = recoverPose(essential_matrix, source_pts, query_pts, cameraMatrix=camera_matrix)
        return R, t
    else:
        raise NotImplementedError
        # homography_matrix = findHomography(source_pts, query_pts, method=RANSAC, ransacReprojThreshold=3)


def pixel2cam(p: Sequence[float], camera_matrix: np.ndarray):
    return np.array([
        (p[0] - camera_matrix[0, 2]) / camera_matrix[0, 0],
        (p[1] - camera_matrix[1, 2]) / camera_matrix[1, 1]
    ])


def triangulation(
    source_keypoints: Sequence[KeyPoint],
    query_keypoints: Sequence[KeyPoint],
    matches: Sequence[DMatch],
    R: np.ndarray,
    t: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    """
    Returns Nx3 triangulated points in world coordinates
    """
    T1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])
    T2 = R[:, [0, 1, 2, 0]]

    source_pts, query_pts = [], []
    for match in matches:
        source_pts.append(pixel2cam(source_keypoints[match.trainIdx].pt, camera_matrix))
        query_pts.append(pixel2cam(query_keypoints[match.queryIdx].pt, camera_matrix))

    source_pts = np.array(source_pts)
    query_pts = np.array(query_pts)

    pts_4d = triangulatePoints(T1, T2, source_pts.T, query_pts.T)

    pts = []
    for i in range(pts_4d.shape[1]):
        pts.append(pts_4d[:3, i] / pts_4d[3, i])

    return np.array(pts)


def get_color(depth: float) -> tuple[float, float, float]:
    up_th = 50
    low_th = 10
    th_range = up_th - low_th
    if depth > up_th:
        depth = up_th
    if depth < low_th:
        depth = low_th
    return 255 * depth / th_range, 0, 255 * (1 - depth / th_range)


class Frontend:
    def __init__(self):
        pass


class Backend:
    def __init__(self):
        pass


class Feature:
    __slots__ = (
        "frame", "position", "map_point", "is_outlier"
    )

    frame: Frame
    position: KeyPoint
    map_point: Optional[MapPoint]
    is_outlier: bool

    def __init__(self, frame: Frame, kp: KeyPoint):
        self.frame = frame
        self.position = kp
        self.is_outlier = False


class MapPoint:
    _factory_id: np.uint64 = 0

    __slots__ = (
        "id", "pos", "is_outlier", "observed_times", "pos_mutex", "obs_mutex", "observations"
    )

    id: np.uint64
    pos: np.ndarray
    is_outlier: bool
    observed_times: int
    pos_mutex: Lock
    obs_mutex: Lock
    observations: set[Feature]

    def __init__(self, id: np.uint64 = 0, position: np.ndarray = np.zeros((3,))):
        self.id = id
        self.pos = position
        self.is_outlier = False
        self.observed_times = 0
        self.pos_mutex = Lock()
        self.obs_mutex = Lock()
        self.observations = set()

    def get_pos(self) -> np.ndarray:
        with self.pos_mutex:
            return self.pos.copy()

    def set_pos(self, pos: np.ndarray):
        with self.pos_mutex:
            self.pos = pos

    def add_observation(self, feature: Feature):
        with self.obs_mutex:
            self.observations.add(feature)
            self.observed_times += 1

    def remove_observation(self, feature: Feature):
        with self.obs_mutex:
            if feature in self.observations:
                self.observations.remove(feature)
                feature.map_point = None
                self.observed_times -= 1

    def get_observations(self):
        with self.obs_mutex:
            return self.observations

    @staticmethod
    def create_map_point() -> MapPoint:
        map_point = MapPoint(id=MapPoint._factory_id)
        MapPoint._factory_id += 1
        return map_point


class Frame:
    _factory_id: np.uint64 = 0
    _keyframe_factory_id: np.uint64 = 0

    __slots__ = (
        "id", "keyframe_id", "is_keyframe", "time_stamp", "pose", "pose_mutex", "img", "features"
    )

    id: np.uint64
    keyframe_id: Optional[np.uint64]
    is_keyframe: bool
    time_stamp: np.float64
    pose: SE3  # Tcw
    pose_mutex: Lock
    img: np.ndarray
    features: list[Feature]

    def __init__(self, id: np.uint64 = 0, time_stamp: np.float64 = 0, pose: SE3 = None, img: np.ndarray = None):
        self.id = id
        self.keyframe_id = None
        self.is_keyframe = False
        self.time_stamp = time_stamp
        self.pose = pose
        self.pose_mutex = Lock()
        self.img = img
        self.features = []

    def get_pose(self) -> SE3:
        with self.pose_mutex:
            return self.pose

    def set_pose(self, pose: SE3):
        with self.pose_mutex:
            self.pose = pose

    def set_keyframe(self):
        self.is_keyframe = True
        self.keyframe_id = Frame._keyframe_factory_id
        Frame._keyframe_factory_id += 1

    @staticmethod
    def create_frame() -> Frame:
        frame = Frame(id=Frame._factory_id)
        Frame._factory_id += 1
        return frame


class Map:
    NUM_ACTIVE_KEYFRAMES = 7
    MIN_DIST_THRESHOLD = 0.2

    __slots__ = (
        "_landmarks", "_keyframes", "_active_landmarks", "_active_keyframes", "_data_mutex", "_current_frame"
    )

    _landmarks: dict[np.uint64, MapPoint]
    _keyframes: dict[np.uint64, Frame]
    _active_landmarks: dict[np.uint64, MapPoint]
    _active_keyframes: dict[np.uint64, Frame]
    _data_mutex: Lock
    _current_frame: Optional[Frame]

    def __init__(self):
        self._data_mutex = Lock()
        self._current_frame = None

    def insert_keyframe(self, frame: Frame):  # TODO: check diff with c++ implementation
        self._current_frame = frame
        self._keyframes[frame.keyframe_id] = frame
        self._active_keyframes[frame.keyframe_id] = frame

        if len(self._active_keyframes) > Map.NUM_ACTIVE_KEYFRAMES:
            self._remove_old_keyframe()

    def insert_map_point(self, map_point: MapPoint):
        self._landmarks[map_point.id] = map_point
        self._active_landmarks[map_point.id] = map_point

    def get_all_keyframes(self) -> dict[np.uint64, Frame]:
        with self._data_mutex:
            return self._keyframes

    def get_all_map_points(self) -> dict[np.uint64, MapPoint]:
        with self._data_mutex:
            return self._landmarks

    def get_active_keyframes(self) -> dict[np.uint64, Frame]:
        with self._data_mutex:
            return self._active_keyframes

    def get_active_map_points(self) -> dict[np.uint64, MapPoint]:
        with self._data_mutex:
            return self._active_landmarks

    def clean_map(self):
        cnt_removed = 0
        for l_id in list(self._active_landmarks.keys()):
            if self._active_landmarks[l_id].observed_times == 0:
                self._active_landmarks.pop(l_id)
                cnt_removed += 1
        logger.info(f"Removed {cnt_removed} active landmarks")

    def _remove_old_keyframe(self):
        """
        Find closest and farthest keyframe. Remove closest if it's too close to current frame, otherwise remove farthest
        """
        if self._current_frame is None:
            return

        max_dist, min_dist = 0, 1e9
        max_keyframe_id, min_keyframe_id = -1, -1

        twc = self._current_frame.pose.inverse()  # transform of current frame in world coordinates
        for kf_id, kf in self._active_keyframes.items():
            if kf_id == self._current_frame.id:
                continue

            dist = np.linalg.norm((kf.pose @ twc).log())
            if dist > max_dist:
                max_dist = dist
                max_keyframe_id = kf_id
            elif dist < min_dist:
                min_dist = dist
                min_keyframe_id = kf_id

        if min_dist < Map.MIN_DIST_THRESHOLD:
            keyframe_to_remove = self._keyframes[min_keyframe_id]
        else:
            keyframe_to_remove = self._keyframes[max_keyframe_id]

        logger.info(f"Removing keyframe: {keyframe_to_remove.keyframe_id}")
        self._active_keyframes.pop(keyframe_to_remove.keyframe_id)

        for feature in keyframe_to_remove.features:
            if feature.map_point:
                feature.map_point.remove_observation(feature)

        self.clean_map()




class SLAM:
    def __init__(self):
        pass


def main():
    camera_matrix = np.array([[520.9, 0, 325.1],
                              [0, 521., 249.7],
                              [0, 0, 1]])
    orb_extractor = OrbFeatureExtractor(n_features=500)
    img1 = cv2.imread("1.png", flags=cv2.IMREAD_COLOR)
    img2 = cv2.imread("2.png", flags=cv2.IMREAD_COLOR)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb_extractor.get_features(img1)
    kp2, des2 = orb_extractor.get_features(img2)
    bf_matcher = BruteForceFeatureMatcher(norm_type=NORM_HAMMING)
    matches = bf_matcher.match_features(des1, des2, 25)

    R, t = pose_estimation_2d2d(kp1, kp2, matches, camera_matrix)
    print(R, t)
    from sophuspy import SE3 as s_se3
    from jaxlie import SE3 as j_se3
    from jaxlie import SO3 as j_so3
    s_pose = s_se3(R, t)
    j_pose = j_se3.from_rotation_and_translation(rotation=j_so3.from_matrix(R), translation=t.flatten())

    points = triangulation(kp1, kp2, matches, R, t, camera_matrix)


if __name__ == "__main__":
    main()
