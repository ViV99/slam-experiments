from __future__ import annotations

import logging as logger
from enum import Enum
from queue import Queue
from typing import Sequence, Optional


import cv2
import g2o
import numpy as np
from jaxlie import SE3

from feature_detectors import FeatureDetector
from feature_matchers import FeatureMatcher
from backend import Backend, Map
from primitives import Feature, MapPoint, Frame, Camera
from utils import pose_estimation_2d2d, get_featured_detection_mask, triangulation


class Viewer:
    pass


class Frontend:
    class Status(Enum):
        INITIALIZING = 0
        TRACKING = 1

    __slots__ = (
        "_status", "_current_frame", "_last_frame", "_camera", "_backend_queue", "_viewer", "_relative_motion",
        "_tracking_inliers", "_init_frame_cnt",  "_feature_detector", "_feature_matcher", "n_features",
        "n_features_init", "n_features_tracking", "n_features_tracking_for_keyframe", "feature_radius",
        "reprojection_threshold", "last_frame_refresh_rate"
    )

    _status: Frontend.Status
    _current_frame: Optional[Frame]
    _last_frame: Optional[Frame]
    _camera: Camera
    _backend_queue: Queue
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
        camera: Camera,
        viewer: Viewer = None,
        backend_queue: Queue = None,
        n_features_tracking_for_keyframe: int = 80,
        feature_radius: int = 10,
        reprojection_threshold: float = 2.,
        last_frame_refresh_rate: int = 10
    ):
        self._feature_detector = feature_detector
        self._feature_matcher = feature_matcher
        self._camera = camera
        self._viewer = viewer
        self._backend_queue = backend_queue
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

    def get_last_frame(self) -> Frame:
        return self._last_frame

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

            if self._backend_queue:
                self._backend_queue.put(self._last_frame)
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
                    if self._backend_queue:
                        self._backend_queue.put(map_point)
                        raise NotImplementedError()
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
