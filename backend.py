import logging as logger
from typing import Optional
from copy import deepcopy

import numpy as np

from primitives import MapPoint, Frame


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


class Backend:
    def __init__(self):
        pass
