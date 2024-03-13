from __future__ import annotations
from typing import Sequence, Optional

import cv2
import numpy as np
from cv2 import NORM_HAMMING, KeyPoint, DMatch, findEssentialMat, findHomography, recoverPose, triangulatePoints
from sophuspy import SO3, SE3

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


class MapPoint:
    def __init__(self):
        pass


class Feature:
    __slots__ = (
        "frame", "position", "map_point", "is_outlier"
    )

    frame: Frame
    position: KeyPoint
    map_point: MapPoint
    is_outlier: bool

    def __init__(self, frame: Frame, kp: KeyPoint):
        self.frame = frame
        self.position = kp
        self.is_outlier = False


class Frame:
    _factory_id: np.uint64 = 0
    _keyframe_factory_id: np.uint64 = 0

    __slots__ = (
        "id", "keyframe_id", "is_keyframe", "time_stamp", "pose", "pose_mutex", "img", "features"
    )

    id: np.uint64
    keyframe_id: np.uint64
    is_keyframe: bool
    time_stamp: np.float64
    pose: SE3  # Tcw
    pose_mutex: Lock
    img: np.ndarray
    features: list[Feature]

    def __init__(self, id: np.uint64 = 0, time_stamp: np.float64 = 0, pose: SE3 = None, img: np.ndarray = None):
        self.id = id
        self.time_stamp = time_stamp
        self.pose = pose
        self.img = img
        self.pose_mutex = Lock()

    def get_pose(self) -> SE3:
        with self.pose_mutex:
            return self.pose

    def set_pose(self, pose: SE3):
        with self.pose_mutex:
            self.pose = pose

    def set_keyframe(self):
        self.is_keyframe = True
        Frame._keyframe_factory_id += 1

    @staticmethod
    def create_frame() -> Frame:
        frame = Frame(id=Frame._factory_id)
        Frame._factory_id += 1
        return frame


class Map:
    NUM_ACTIVE_KEYFRAMES = 7

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
        pass

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
        pass

    def _remove_old_keyframe(self):
        if self._current_frame is None:
            return

        max_dist, min_dist = 0, 1e9
        max_keyframe_id, min_keyframe_id = -1, -1

        twc = self._current_frame.pose.inverse()  # transform of current frame in world coordinates
        for kf_id, kf in self._active_keyframes.items():
            if kf.id == self._current_frame.id:
                continue

            dist = (kf.pose @ twc).log()
            a = SE3().normalize()


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
    print(s_pose.log())
    print(j_pose.log())

    points = triangulation(kp1, kp2, matches, R, t, camera_matrix)


if __name__ == "__main__":
    main()
