from typing import Sequence

import cv2
import numpy as np
from jaxlie import SO3, SE3

from primitives import Frame, Feature, Camera


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