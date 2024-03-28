from __future__ import annotations

import logging as logger
from typing import Sequence, Optional
from enum import Enum

import cv2
import g2o
import numpy as np
# from cv2 import NORM_HAMMING, KeyPoint, DMatch, findEssentialMat, findHomography, recoverPose, triangulatePoints
from jaxlie import SO3, SE3

from feature_detectors import OrbFeatureDetector, FeatureDetector
from feature_matchers import BruteForceFeatureMatcher, FeatureMatcher

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


def triangulation(
    source_keypoints: Sequence[cv2.KeyPoint],
    query_keypoints: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    camera: Camera,
) -> np.ndarray:
    """
    Returns Nx3 triangulated points in world coordinates
    """
    projection_source = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])
    projection_query = camera.projection

    source_pts, query_pts = [], []
    for match in matches:
        source_pts.append(camera.pixel_to_camera(np.array(source_keypoints[match.trainIdx].pt)))
        query_pts.append(camera.pixel_to_camera(np.array(query_keypoints[match.queryIdx].pt)))

    source_pts = np.array(source_pts)
    query_pts = np.array(query_pts)

    pts_4d = cv2.triangulatePoints(projection_source, projection_query, source_pts.T, query_pts.T)

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
    pose: SE3  # camera rotation and translation (extrinsics) - world to camera transform
    pose_inv: SE3

    def __init__(self, fx: float, fy: float, cx: float, cy: float, pose: SE3):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.pose = pose
        self.pose_inv = pose.inverse()

    @property
    def projection(self) -> np.ndarray:
        return self.intrinsics @ self.pose.as_matrix()[:-1]

    @property
    def intrinsics(self) -> np.ndarray:
        """
        :return: camera intrinsics matrix - camera to pixel transform
        """
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
    class Status(Enum):
        INITIALIZING = 0
        TRACKING = 1
        LOST = 2

    __slots__ = (
        "_status", "_current_frame", "_last_frame", "_camera", "_map", "_backend", "_viewer", "_relative_motion",
        "_tracking_inliers", "_feature_detector", "_feature_matcher", "n_features", "n_features_init",
        "n_features_tracking", "n_features_tracking_for_keyframe", "feature_radius"
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
    _feature_matcher: FeatureMatcher

    n_features: int
    n_features_init: int
    n_features_tracking: int
    n_features_tracking_for_keyframe: int
    feature_radius: int

    def __init__(
            self,
            feature_detector: FeatureDetector,
            feature_matcher: FeatureMatcher,
            n_features: int = 200,
            n_features_init: int = 100,
            feature_radius: int = 10,
            viewer: Viewer = None
    ):
        self._feature_detector = feature_detector
        self._feature_matcher = feature_matcher
        self._viewer = viewer
        self.n_features = n_features
        self.n_features_init = n_features_init
        self.n_features_tracking = 50
        self.n_features_tracking_for_keyframe = 80
        self.feature_radius = feature_radius
        self._current_frame = None
        self._last_frame = None

    def add_frame(self, frame: Frame):
        self._current_frame = frame

        if self._status == Frontend.Status.INITIALIZING:
            self._init()
        elif self._status == Frontend.Status.TRACKING:
            self._track()
        elif self._status == Frontend.Status.LOST:
            self._reset()

        self._last_frame = self._current_frame

    def _init(self):
        if self._init_map():
            self._status = Frontend.Status.TRACKING
            if self._viewer:
                # self._viewer.add_current_frame(self._current_frame)
                # self._viewer.update_map()
                raise NotImplementedError()

    def _track(self):
        if self._last_frame:
            self._current_frame.set_pose(self._relative_motion @ self._last_frame.pose)

        self._track_current_frame()
        self._tracking_inliers = self._estimate_current_pose()

        if self._tracking_inliers > self.n_features_tracking:
            self._status = Frontend.Status.TRACKING
        else:
            self._status = Frontend.Status.LOST

        self._insert_keyframe()

        self._relative_motion = self._current_frame.pose @ self._last_frame.pose.inverse()

        if self._viewer:
            # self._viewer.add_current_frame(self._current_frame)
            raise NotImplementedError()

    def _reset(self):
        pass

    def _track_current_frame(self):
        keypoints_last, descriptors_last = [], []
        keypoints_cur = []
        mask = np.full(self._current_frame.img.shape[:2], fill_value=0, dtype=np.uint8)
        for feature in self._last_frame.features:
            descriptors_last.append(feature.descriptor)
            pt = np.array(feature.position.pt)
            if feature.map_point:  # can use more accurate pixel coordinates
                point_pixel = self._camera.world_to_pixel(feature.map_point.pos, self._current_frame.pose)
                keypoints_last.append(pt)
                keypoints_cur.append(point_pixel)
            else:
                keypoints_last.append(pt)
                keypoints_cur.append(pt)

            mask = cv2.rectangle(
                mask,
                keypoints_cur[-1] - np.array([self.feature_radius, self.feature_radius]),
                keypoints_cur[-1] + np.array([self.feature_radius, self.feature_radius]),
                255,
                cv2.FILLED
            )
        keypoints_cur, descriptors_cur = self._feature_detector.detect_and_compute(self._current_frame.img, mask)
        matches = self._feature_matcher.match(np.array(descriptors_last), descriptors_cur)

        for match in matches:
            feature = Feature(self._current_frame, keypoints_cur[match.queryIdx], descriptors_cur[match.queryIdx])
            feature.map_point = self._last_frame.features[match.trainIdx].map_point
            self._current_frame.features.append(feature)

        logger.info("Found %s matches for features", len(matches))
        return len(matches)

    def _estimate_current_pose(self) -> int:
        solver = g2o.OptimizationAlgorithmLevenberg(g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3()))
        optimizer = g2o.SparseOptimizer()
        optimizer.set_algorithm(solver)
        optimizer.set_verbose(False)

        cam = g2o.CameraParameters(
            (self._camera.fx + self._camera.fy) / 2, np.array([self._camera.cx, self._camera.cy]), 0
        )
        cam.set_id(0)
        optimizer.add_parameter(cam)

        vertex_pose = g2o.VertexSE3Expmap()
        vertex_pose.set_id(0)
        vertex_pose.set_estimate(
            g2o.SE3Quat(self._current_frame.pose.rotation().as_matrix(), self._current_frame.pose.translation())
        )
        optimizer.add_vertex(vertex_pose)

        index = 1
        edges, features = [], []
        for feature in self._current_frame.features:
            if feature.map_point:
                features.append(feature)

                vertex_map_point = g2o.VertexPointXYZ()
                vertex_map_point.set_id(index)
                vertex_map_point.set_marginalized(True)
                vertex_map_point.set_estimate(feature.map_point.pos)
                optimizer.add_vertex(vertex_map_point)

                edge = g2o.EdgeSE3ProjectXYZ()
                edge.set_parameter_id(0, 0)
                edge.set_vertex(0, vertex_map_point)
                edge.set_vertex(1, vertex_pose)
                edge.set_measurement(np.array(feature.position.pt))
                edge.set_information(np.identity(2))
                edge.set_robust_kernel(g2o.RobustKernelHuber())
                edges.append(edge)
                optimizer.add_edge(edge)
                index += 1

        chi2_threshold = 5.991
        cnt_outliers = 0
        for iteration in range(4):
            cnt_outliers = 0
            vertex_pose.set_estimate(
                g2o.SE3Quat(self._current_frame.pose.rotation().as_matrix(), self._current_frame.pose.translation())
            )
            optimizer.initialize_optimization()
            optimizer.optimize(10)
            for i in range(len(edges)):
                if features[i].is_outlier:
                    edges[i].compute_error()
                if edges[i].chi2() > chi2_threshold:
                    features[i].is_outlier = True
                    edges[i].set_level(1)
                    cnt_outliers += 1
                else:
                    features[i].is_outlier = False
                    edges[i].set_level(0)
                if iteration == 2:
                    edges[i].set_robust_kernel(None)

        logger.info("Outlier/Inlier in pose estimation: %s / %s", cnt_outliers, len(features) - cnt_outliers)

        self._current_frame.set_pose(SE3.from_matrix(vertex_pose.estimate().matrix()))

        logger.info("Current Pose = %s", self._current_frame.pose)
        for feature in features:
            if feature.is_outlier:
                feature.map_point = None
                feature.is_outlier = False

        return len(features) - cnt_outliers

    def _insert_keyframe(self):
        pass

    def _detect_features(self) -> int:
        mask = np.full(self._current_frame.img.shape[:2], fill_value=255, dtype=np.uint8)
        for feature in self._current_frame.features:
            pt = np.array(feature.position.pt)
            mask = cv2.rectangle(
                mask,
                pt - np.array([self.feature_radius, self.feature_radius]),
                pt + np.array([self.feature_radius, self.feature_radius]),
                0,
                cv2.FILLED
            )
        keypoints, descriptors = self._feature_detector.detect_and_compute(self._current_frame.img, mask)
        cnt_detected = 0
        for i in range(len(keypoints)):
            self._current_frame.features.append(Feature(self._current_frame, keypoints[i], descriptors[i]))
            cnt_detected += 1

        logger.info("Detected %s new features", cnt_detected)
        return cnt_detected

    def _init_map(self):
        poses = [self._camera.pose]
        cnt_init_landmarks = 0

        for feature in self._current_frame.features:
            point_camera = self._camera.pixel_to_camera(np.array(feature.position.pt))
            triangulation()


    def _triangulate_new_points(self) -> int:
        pass

    def _set_observations_for_keyframe(self):
        pass

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
        "frame", "position", "map_point", "is_outlier", "descriptor"
    )

    frame: Frame
    position: cv2.KeyPoint
    # descriptor: None
    map_point: Optional[MapPoint]
    is_outlier: bool

    def __init__(self, frame: Frame, kp: cv2.KeyPoint, descriptor):
        self.frame = frame
        self.position = kp
        self.descriptor = descriptor
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
        logger.info("Removed %s active landmarks", cnt_removed)

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

        logger.info("Removing keyframe: %s", keyframe_to_remove.keyframe_id)
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
    matches = bf_matcher.match(des1, des2, 25)

    R, t = pose_estimation_2d2d(kp1, kp2, matches, camera_matrix)
    print(R, t)
    from sophuspy import SE3 as s_se3
    from jaxlie import SE3 as j_se3
    from jaxlie import SO3 as j_so3
    s_pose = s_se3(R, t)
    j_pose = j_se3.from_rotation_and_translation(rotation=j_so3.from_matrix(R), translation=t.flatten())
    pose1 = j_se3.from_rotation_and_translation(rotation=j_so3.from_matrix(R), translation=t.flatten())
    pose2 = j_se3.from_rotation_and_translation(rotation=j_so3.from_matrix(R - 1), translation=t.flatten() + 1)

    solver = g2o.OptimizationAlgorithmLevenberg(g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3()))
    optimizer = g2o.SparseOptimizer()
    optimizer.set_algorithm(solver)
    optimizer.set_verbose(True)

    intrinsics = np.array([
        [5, 0, 2],
        [0, 5, 2],
        [0, 0, 1]
    ])

    cam = g2o.CameraParameters(5, np.array([2, 2]), 0)
    cam.set_id(0)
    optimizer.add_parameter(cam)

    vertex_pose = g2o.VertexSE3Expmap()
    vertex_pose.set_id(0)
    vertex_pose.set_estimate(
        g2o.SE3Quat(pose1.rotation().as_matrix(), pose1.translation())
    )
    vertex_pose.set_fixed(False)
    optimizer.add_vertex(vertex_pose)
    edges = []
    for i in range(1, 11):
        vp = g2o.VertexPointXYZ()
        vp.set_id(i)
        vp.set_marginalized(True)
        vp.set_estimate([1, 2, 44] + np.random.randn(3))
        optimizer.add_vertex(vp)

        # edge = g2o.EdgeProjectXYZ2UV()
        edge = g2o.EdgeSE3ProjectXYZ()
        edge.set_parameter_id(0, 0)
        edge.set_vertex(0, vp)
        edge.set_vertex(1, vertex_pose)
        edge.set_measurement(np.random.randn(2))
        edge.set_information(np.identity(2))
        edge.set_robust_kernel(g2o.RobustKernelHuber())
        optimizer.add_edge(edge)
        edges.append(edge)

    print(vertex_pose.estimate().matrix())
    optimizer.initialize_optimization()
    optimizer.optimize(10)
    print(vertex_pose.estimate().matrix())
    print(edges[0].cam_project(np.array([1, 66, 3])))

    # for e in edges:
    #     print(e.measurement())
    #     e.compute_error()
    #     print(e.chi2())


    # points = triangulation(kp1, kp2, matches, R, t, camera_matrix)

if __name__ == "__main__":
    np.random.seed(228)
    main()
