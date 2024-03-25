from __future__ import annotations

import logging as logger
from typing import Sequence, Optional
from enum import Enum

import cv2
import numpy as np
# from cv2 import NORM_HAMMING, KeyPoint, DMatch, findEssentialMat, findHomography, recoverPose, triangulatePoints
from jaxlie import SO3, SE3

from feature_detectors import OrbFeatureDetector, FeatureDetector
from feature_matchers import BruteForceFeatureMatcher

from multiprocessing import Process, Queue, Lock


def pose_estimation_2d2d(
    source_keypoints: Sequence[cv2.KeyPoint],
    query_keypoints: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    camera_matrix: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    source_pts, query_pts = [], []
    for match in matches:
        source_pts.append(source_keypoints[match.trainIdx].pt)
        query_pts.append(query_keypoints[match.queryIdx].pt)

    source_pts = np.array(source_pts)
    query_pts = np.array(query_pts)

    if camera_matrix is not None:
        essential_matrix, _ = cv2.findEssentialMat(source_pts, query_pts, cameraMatrix=camera_matrix)
        _, R, t, _ = cv2.recoverPose(essential_matrix, source_pts, query_pts, cameraMatrix=camera_matrix)
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
    source_keypoints: Sequence[cv2.KeyPoint],
    query_keypoints: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
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

    pts_4d = cv2.triangulatePoints(T1, T2, source_pts.T, query_pts.T)

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


class Viewer:
    pass


class Camera:
    __slots__ = ("fx", "fy", "cx", "cy", "pose", "pose_inv")

    fx: float
    fy: float
    cx: float
    cy: float
    pose: SE3
    pose_inv: SE3

    def __init__(self, fx: float, fy: float, cx: float, cy: float, pose: SE3):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.pose = pose
        self.pose_inv = pose.inverse()

    def K(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

    def world_to_camera(self, p_w: np.ndarray, t_c_w: SE3) -> np.ndarray:
        """
        :param p_w: np.array[3,]
        :param t_c_w: SE3
        :return: np.ndarray[3,]
        """
        return self.pose @ t_c_w @ p_w

    def camera_to_world(self, p_c: np.ndarray, t_c_w: SE3) -> np.ndarray:
        """
        :param p_c: np.array[3,]
        :param t_c_w: SE3
        :return: np.ndarray[3,]
        """
        return t_c_w.inverse() @ self.pose_inv @ p_c

    def camera_to_pixel(self, p_c: np.ndarray) -> np.ndarray:
        """
        :param p_c: np.array[3,]
        :return: np.ndarray[2,]
        """
        return np.array([
            self.fx * p_c[0] / p_c[2] + self.cx,
            self.fy * p_c[1] / p_c[2] + self.cy
        ])

    def pixel_to_camera(self, p_p: np.ndarray, depth: float = 1) -> np.ndarray:
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

    def pixel_to_world(self, p_p: np.ndarray, t_c_w: SE3, depth: float = 1) -> np.ndarray:
        """
        :param p_p: np.array[2,]
        :param t_c_w: SE3
        :param depth: float
        :return: np.ndarray[3,]
        """
        return self.camera_to_world(self.pixel_to_camera(p_p, depth), t_c_w)

    def world_to_pixel(self, p_w: np.ndarray, t_c_w: SE3) -> np.ndarray:
        """
        :param p_w: np.array[3,]
        :param t_c_w: SE3
        :return: np.ndarray[2,]
        """
        return self.camera_to_pixel(self.world_to_camera(p_w, t_c_w))


class Frontend:
    N_FEATURES = 200
    N_FEATURES_INIT = 100
    N_FEATURES_TRACKING = 50
    N_FEATURES_TRACKING_BAD = 20
    N_FEATURES_FOR_KEYFRAME = 80

    class Status(Enum):
        INITIALIZING = 0
        GOOD_TRACKING = 1
        BAD_TRACKING = 2
        LOST = 3

    __slots__ = (
        "_status", "_current_frame", "_last_frame", "_camera", "_map", "_backend", "_viewer", "_relative_motion",
        "_tracking_inliers", "_feature_detector", "n_features", "n_features_init", "n_features_good_tracking",
        "n_features_bad_tracking", "n_features_tracking_for_keyframe"
    )

    _status: Frontend.Status
    _current_frame: Optional[Frame]
    _last_frame: Optional[Frame]
    _camera: Camera
    _map: Map
    _backend: Backend
    _viewer: Optional[Viewer]
    _relative_motion: SE3
    _tracking_inliers: int
    _feature_detector: FeatureDetector

    n_features: int
    n_features_init: int
    n_features_tracking_good: int
    n_features_tracking_bad: int
    n_features_tracking_for_keyframe: int

    def __init__(
            self,
            feature_detector: FeatureDetector,
            n_features: int = 200,
            n_features_init: int = 100,
            viewer: Viewer = None
    ):
        self._feature_detector = feature_detector
        self._viewer = viewer
        self.n_features = n_features
        self.n_features_init = n_features_init
        self.n_features_good_tracking = 50
        self.n_features_bad_tracking = 20
        self.n_features_tracking_for_keyframe = 80
        self._current_frame = None
        self._last_frame = None

    def _init(self) -> bool:
        if self._init_map():
            self._status = Frontend.Status.GOOD_TRACKING
            if self._viewer:
                # self._viewer.add_current_frame(self._current_frame)
                # self._viewer.update_map()
                raise NotImplementedError()
            return True

        return False

    def _track(self):
        if self._last_frame:
            self._current_frame.set_pose(self._relative_motion @ self._last_frame.pose)

        self._track_last_frame()
        self._tracking_inliers = self._estimate_current_pose()

        if self._tracking_inliers > self.n_features_good_tracking:
            self._status = Frontend.Status.GOOD_TRACKING
        elif self._tracking_inliers > self.n_features_bad_tracking:
            self._status = Frontend.Status.BAD_TRACKING
        else:
            self._status = Frontend.Status.LOST

        self._insert_keyframe()

        self._relative_motion = self._current_frame.pose @ self._last_frame.pose.inverse()

        if self._viewer:
            # self._viewer.add_current_frame(self._current_frame)
            raise NotImplementedError()

    def _reset(self):
        pass

    def _track_last_frame(self):
        pass

    def _estimate_current_pose(self) -> int:
        pass

    def _insert_keyframe(self):
        pass

    def _detect_features(self) -> int:
        mask = np.full(self._current_frame.img.shape[:2], fill_value=255, dtype=np.uint8)
        for feature in self._current_frame.features:
            pt = np.array(feature.position.pt)
            mask = cv2.rectangle(
                mask,
                pt - np.array([10, 10]),
                pt + np.array([10, 10]),
                0,
                cv2.FILLED
            )
        keypoints = self._feature_detector.detect(self._current_frame.img, mask)
        cnt_detected = 0
        for kp in keypoints:
            self._current_frame.features.append(Feature(self._current_frame, kp))
            cnt_detected += 1

        logger.info(f"Detected {cnt_detected} new features")
        return cnt_detected


    def _init_map(self):
        poses = [self._camera.pose]
        cnt_init_landmarks = 0

        for feature in self._current_frame.features:
            point_camera = self._camera.pixel_to_camera(np.array(feature.position.pt), )


    def _triangulate_new_points(self) -> int:
        pass

    def _set_observations_for_keyframe(self):
        pass

    def add_frame(self, frame: Frame):
        self._current_frame = frame

        if self._status == Frontend.Status.INITIALIZING:
            self._init()
        elif self._status == Frontend.Status.GOOD_TRACKING or self._status == Frontend.Status.BAD_TRACKING:
            self._track()
        elif self._status == Frontend.Status.LOST:
            self._reset()

        self._last_frame = self._current_frame

    def set_map(self, map: Map):
        pass

    def set_backend(self, backend: Backend):
        pass

    # def set_viewer(self, viewer: Viewer):
    #     pass

    def get_status(self) -> Frontend.Status:
        return self._status




class Backend:
    def __init__(self):
        pass


class Feature:
    __slots__ = (
        "frame", "position", "map_point", "is_outlier"
    )

    frame: Frame
    position: cv2.KeyPoint
    map_point: Optional[MapPoint]
    is_outlier: bool

    def __init__(self, frame: Frame, kp: cv2.KeyPoint):
        self.frame = frame
        self.position = kp
        self.is_outlier = False
        self.map_point = None


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
    orb_detector = OrbFeatureDetector(n_features=500)
    img1 = cv2.imread("1.png", flags=cv2.IMREAD_COLOR)
    img2 = cv2.imread("2.png", flags=cv2.IMREAD_COLOR)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb_detector.detect_and_compute(img1)
    kp2, des2 = orb_detector.detect_and_compute(img2)
    bf_matcher = BruteForceFeatureMatcher(norm_type=cv2.NORM_HAMMING)
    matches = bf_matcher.match_features(des1, des2, 25)

    R, t = pose_estimation_2d2d(kp1, kp2, matches, camera_matrix)
    print(R, t)
    from sophuspy import SE3 as s_se3
    from jaxlie import SE3 as j_se3
    from jaxlie import SO3 as j_so3
    s_pose = s_se3(R, t)
    j_pose = j_se3.from_rotation_and_translation(rotation=j_so3.from_matrix(R), translation=t.flatten())
    pose1 = j_se3.from_rotation_and_translation(rotation=j_so3.from_matrix(R), translation=t.flatten())
    pose2 = j_se3.from_rotation_and_translation(rotation=j_so3.from_matrix(R - 1), translation=t.flatten() + 1)
    print(pose1 @ pose2 @ np.array([1, 2, 0]))
    # points = triangulation(kp1, kp2, matches, R, t, camera_matrix)


if __name__ == "__main__":
    main()
    # img = np.full((10, 10), fill_value=255, dtype=np.uint8)
    # img = cv2.rectangle(img, np.array([2, 2]), np.array([4, 4]), 0, cv2.FILLED)
    # print()
