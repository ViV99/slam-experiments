from __future__ import annotations

import logging as logger
from typing import Sequence, Optional, Any
from enum import Enum
from copy import deepcopy

import cv2
import g2o
import numpy as np
# from cv2 import NORM_HAMMING, KeyPoint, DMatch, findEssentialMat, findHomography, recoverPose, triangulatePoints
from jaxlie import SO3, SE3

from feature_detectors import OrbFeatureDetector, FeatureDetector
from feature_matchers import BruteForceFeatureMatcher, FeatureMatcher

from threading import Lock
from queue import Queue


def pose_estimation_2d2d(
    source_frame: Frame, query_frame: Frame, matches: Sequence[cv2.DMatch], camera: Camera = None,
) -> SE3:
    source_pts, query_pts = [], []
    for match in matches:
        source_pts.append(source_frame.features[match.trainIdx].position)
        query_pts.append(query_frame.features[match.queryIdx].position)

    source_pts = np.array(source_pts)
    query_pts = np.array(query_pts)

    if camera:
        essential_matrix, _ = cv2.findEssentialMat(source_pts, query_pts, cameraMatrix=camera.intrinsics)
        _, R, t, _ = cv2.recoverPose(essential_matrix, source_pts, query_pts, cameraMatrix=camera.intrinsics)
        return SE3.from_rotation_and_translation(SO3.from_matrix(R), t.flatten())
    else:
        raise NotImplementedError
        # homography_matrix = findHomography(source_pts, query_pts, method=RANSAC, ransacReprojThreshold=3)


def triangulation(
    source_frame: Frame, query_frame: Frame, matches: Sequence[cv2.DMatch], camera: Camera,
) -> tuple[np.ndarray, np.ndarray]:
    """
    :return: Nx3 triangulated points in world coordinates and corresponding Nx2 pixel coordinates.
    """
    projection_source = camera.projection(source_frame.pose)
    projection_query = camera.projection(query_frame.pose)

    source_pts, query_pts = [], []
    for match in matches:
        source_pts.append(camera.pixel_to_camera(source_frame.features[match.trainIdx].position)[:2])
        query_pts.append(camera.pixel_to_camera(query_frame.features[match.queryIdx].position)[:2])

    source_pts = np.array(source_pts)
    query_pts = np.array(query_pts)

    pts_4d = cv2.triangulatePoints(projection_source, projection_query, source_pts.T, query_pts.T)

    pts = []
    for i in range(pts_4d.shape[1]):
        pts.append(pts_4d[:3, i] / pts_4d[3, i])

    return np.array(pts), source_pts


def get_featured_detection_mask(
    shape: tuple[int, int],
    features: Sequence[Feature],
    radius: int,
    inner: bool = True,
    camera: Camera = None,  # if we want to use map_points
    pose: SE3 = None
) -> np.ndarray:
    mask = np.full(shape, fill_value=0 if inner else 255, dtype=np.uint8)
    for feature in features:
        pt, shift = feature.position, np.array([radius, radius])
        if camera and feature.map_point:  # can use more accurate pixel coordinates
            pt = camera.world_to_pixel(feature.map_point.position, pose)

        mask = cv2.rectangle(mask, pt - shift, pt + shift, 255 if inner else 0, cv2.FILLED)

    return mask


class Viewer:
    pass


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


class Frontend:
    class Status(Enum):
        INITIALIZING = 0
        TRACKING = 1

    __slots__ = (
        "_status", "_current_frame", "_last_frame", "_camera", "_map", "_backend", "_viewer", "_relative_motion",
        "_tracking_inliers", "_init_frame_cnt",  "_feature_detector", "_feature_matcher", "n_features",
        "n_features_init", "n_features_tracking", "n_features_tracking_for_keyframe", "feature_radius",
        "reprojection_threshold", "last_frame_refresh_rate"
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
    _init_frame_cnt: int
    _feature_detector: FeatureDetector
    _feature_matcher: FeatureMatcher

    n_features_tracking_for_keyframe: int
    feature_radius: int
    reprojection_threshold: float
    last_frame_refresh_rate: int

    def __init__(
        self,
        feature_detector: FeatureDetector,
        feature_matcher: FeatureMatcher,
        map: Map,
        camera: Camera,
        viewer: Viewer = None,
        backend: Backend = None,
        n_features_tracking_for_keyframe: int = 80,
        feature_radius: int = 10,
        reprojection_threshold: float = 2.,
        last_frame_refresh_rate: int = 10
    ):
        self._feature_detector = feature_detector
        self._feature_matcher = feature_matcher
        self._map = map
        self._camera = camera
        self._viewer = viewer
        self._backend = backend
        self.n_features_tracking_for_keyframe = n_features_tracking_for_keyframe
        self.feature_radius = feature_radius
        self.reprojection_threshold = reprojection_threshold
        self.last_frame_refresh_rate = last_frame_refresh_rate

        self._init_frame_cnt = 0
        self._status = Frontend.Status.INITIALIZING
        self._current_frame = None
        self._last_frame = None

    def get_status(self) -> Frontend.Status:
        return self._status

    def add_frame(self, frame: Frame):
        self._current_frame = frame

        if self._status == Frontend.Status.INITIALIZING:
            self._init()
        elif self._status == Frontend.Status.TRACKING:
            self._track()

        if self._status == Frontend.Status.TRACKING or self._init_frame_cnt > self.last_frame_refresh_rate:
            self._last_frame = self._current_frame

    def _init(self):
        """
        Try to triangulate new points with last frame. If triangulation succeed - stop INITIALIZING and start TRACKING.
        """
        self._detect_features(True)
        if self._last_frame is None:
            self._current_frame.pose = SE3.identity()
            self._last_frame = self._current_frame
            return

        self._init_frame_cnt += 1

        matches = self._match_features()
        if len(matches) == 0:
            return

        self._relative_motion = pose_estimation_2d2d(self._last_frame, self._current_frame, matches, self._camera)
        self._current_frame.pose = self._relative_motion @ self._last_frame.pose

        if self._triangulate_new_points(matches):
            self._status = Frontend.Status.TRACKING
            self._init_frame_cnt = 0
            self._last_frame.make_keyframe()
            self._map.insert_keyframe(self._last_frame)

            if self._backend:
                # self._backend.update_map()
                raise NotImplementedError()

            if self._viewer:
                # self._viewer.add_current_frame(self._current_frame)
                # self._viewer.update_map()
                raise NotImplementedError()

    def _track(self):
        self._track_current_frame()
        self._tracking_inliers = self._correct_current_pose()

        if self._tracking_inliers < self.n_features_tracking_for_keyframe:
            self._reinitialize_from_keyframe()

        if self._viewer:
            # self._viewer.add_current_frame(self._current_frame)
            raise NotImplementedError()

    def _track_current_frame(self):
        """
        Track features from last frame. Propagate map_points from last frame.
        """
        self._detect_features(False)

        matches = self._match_features()
        if len(matches) == 0:
            self._current_frame.pose = self._relative_motion @ self._last_frame.pose
            self._reinitialize_from_keyframe()
            return

        self._relative_motion = pose_estimation_2d2d(self._last_frame, self._current_frame, matches, self._camera)
        self._current_frame.pose = self._relative_motion @ self._last_frame.pose

        for match in matches:
            map_point = self._last_frame.features[match.trainIdx].map_point
            if map_point:
                self._current_frame.features[match.queryIdx].map_point = map_point

        logger.info("Found %s matches for features", len(matches))

    def _match_features(self):
        """
        Match features between last and current frames.
        """
        desc_last = self._last_frame.get_descriptors()
        desc_cur = self._current_frame.get_descriptors()
        return self._feature_matcher.match(desc_last, desc_cur)

    def _triangulate_new_points(self, matches: Sequence[cv2.DMatch]) -> bool:
        """
        Create map_points for new features.
        :param matches: all feature matches.
        """
        matches = [match for match in matches if self._last_frame.features[match.trainIdx].map_point is None]
        world_points, px_points = triangulation(
            self._last_frame, self._current_frame, matches, self._camera
        )
        if self._get_reprojection_error(world_points, px_points) < self.reprojection_threshold:
            for i, p_w in enumerate(world_points):
                if p_w[2] > 0:
                    map_point = MapPoint.create_map_point(p_w)
                    map_point.add_observation(self._last_frame.features[matches[i].trainIdx])
                    self._last_frame.features[matches[i].trainIdx].map_point = map_point
                    self._current_frame.features[matches[i].queryIdx].map_point = map_point
                    self._map.insert_map_point(map_point)
            return True

        return False

    def _get_reprojection_error(self, world_points: np.ndarray, px_points: np.ndarray) -> float:
        error = 0
        for i in range(len(world_points)):
            px_predicted = self._camera.world_to_pixel(world_points[i], self._last_frame.pose)
            error += np.linalg.norm(px_predicted - px_points[i])

        return error / len(world_points)

    def _reinitialize_from_keyframe(self):
        """
        Prepare current frame do become a key-Frame: find new features and start re-INITIALIZING.
        """
        self._status = self.Status.INITIALIZING
        self._detect_features(True)
        self._last_frame = self._current_frame

    def _detect_features(self, new: bool):
        """
        Track features in current frame.
        :param new: Track old features close to their last positions or find new features somewhere else.
        """
        mask = get_featured_detection_mask(
            self._current_frame.img.shape[:2],
            self._current_frame.features if new else self._last_frame.features,
            self.feature_radius,
            inner=not new,
            camera=None if new else self._camera,
            pose=None if new else self._last_frame.pose
        )

        keypoints, descriptors = self._feature_detector.detect_and_compute(self._current_frame.img, mask)
        cnt_detected = 0
        for i in range(len(keypoints)):
            self._current_frame.features.append(Feature(self._current_frame, keypoints[i], descriptors[i]))
            cnt_detected += 1

        logger.info("Detected %s %s features", cnt_detected, "new" if new else "old")

    def _correct_current_pose(self) -> int:
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
                vertex_map_point.set_estimate(feature.map_point.position)
                optimizer.add_vertex(vertex_map_point)

                edge = g2o.EdgeSE3ProjectXYZ()
                edge.set_parameter_id(0, 0)
                edge.set_vertex(0, vertex_map_point)
                edge.set_vertex(1, vertex_pose)
                edge.set_measurement(feature.position)
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


class Backend:
    def __init__(self):
        pass


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
    time_stamp: np.float64
    pose: SE3  # Tcw
    _mutex: Lock
    img: np.ndarray
    features: list[Feature]

    def __init__(self, id: np.uint64 = 0, time_stamp: np.float64 = 0, pose: SE3 = None, img: np.ndarray = None):
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
    def create_frame(img: np.ndarray, timestamp: np.float64) -> Frame:
        frame = Frame(Frame._factory_id, timestamp, img=img)
        Frame._factory_id += 1
        return frame


class Map:
    NUM_ACTIVE_KEYFRAMES = 7
    MIN_DIST_THRESHOLD = 0.2

    __slots__ = (
        "_landmarks", "_keyframes", "_active_landmarks", "_active_keyframes", "_current_frame"
    )

    _landmarks: dict[np.uint64, MapPoint]
    _keyframes: dict[np.uint64, Frame]
    _active_landmarks: dict[np.uint64, MapPoint]
    _active_keyframes: dict[np.uint64, Frame]
    _current_frame: Optional[Frame]

    def __init__(self):
        self._current_frame = None
        self._landmarks = {}
        self._keyframes = {}
        self._active_landmarks = {}
        self._active_keyframes = {}

    def insert_keyframe(self, frame: Frame):
        self._current_frame = frame
        self._keyframes[frame.keyframe_id] = frame
        self._active_keyframes[frame.keyframe_id] = frame

        if len(self._active_keyframes) > Map.NUM_ACTIVE_KEYFRAMES:
            self._remove_old_keyframe()

    def insert_map_point(self, map_point: MapPoint):
        self._landmarks[map_point.id] = map_point
        self._active_landmarks[map_point.id] = map_point

    def get_all_keyframes(self) -> dict[np.uint64, Frame]:
        return deepcopy(self._keyframes)

    def get_all_map_points(self) -> dict[np.uint64, MapPoint]:
        return deepcopy(self._landmarks)

    def get_active_keyframes(self) -> dict[np.uint64, Frame]:
        return deepcopy(self._active_keyframes)

    def get_active_map_points(self) -> dict[np.uint64, MapPoint]:
        return deepcopy(self._active_landmarks)

    def clean_map(self):
        cnt_removed = 0
        for l_id in list(self._active_landmarks.keys()):
            if len(self._active_landmarks[l_id].observations) == 0:
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
