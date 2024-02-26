from typing import Sequence, Optional

import cv2
import g2o.g2opy as g2o
import jaxlie
import numpy as np
from cv2 import NORM_HAMMING, KeyPoint, DMatch, findEssentialMat, findHomography, recoverPose, decomposeHomographyMat, RANSAC
from jaxlie import SO3, SE3

from feature_exractors import OrbFeatureExtractor
from feature_matchers import BruteForceFeatureMatcher


def pose_estimation_2d2d(
    source_keypoints: Sequence[KeyPoint],
    query_keypoints: Sequence[KeyPoint],
    matches: Sequence[DMatch],
    camera_maxrix: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    source_pts, query_pts = [], []
    for match in matches:
        source_pts.append(source_keypoints[match.trainIdx].pt)
        query_pts.append(query_keypoints[match.queryIdx].pt)

    source_pts = np.array(source_pts)
    query_pts = np.array(query_pts)

    if camera_maxrix is not None:
        essential_matrix, _ = findEssentialMat(source_pts, query_pts, cameraMatrix=camera_maxrix)
        _, R, t, _ = recoverPose(essential_matrix, source_pts, query_pts, cameraMatrix=camera_maxrix)
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

    print(source_pts.shape, query_pts.shape)
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


def main():
    camera_matrix = np.array([[520.9, 0, 325.1],
                              [0, 521., 249.7],
                              [0, 0, 1]])
    solver = g2o.OptimizationAlgorithmLevenberg(g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3()))
    optimizer = g2o.SparseOptimizer()
    optimizer.set_algorithm(solver)
    optimizer.set_verbose(True)

    kp1, des1 = orb_extractor.get_features(img1)
    kp2, des2 = orb_extractor.get_features(img2)

                se3 = jaxlie.SE3(np.array(arr[3:10], dtype=np.float32))
                e.set_measurement(g2o.Isometry3d(se3.as_matrix()))
                info = np.zeros((6, 6))
                x, y = np.triu_indices_from(info)
                info[x, y] = np.array(arr[10:], dtype=np.float32)

                e.set_information()
                optimizer.add_edge(e)

    # orb_extractor = OrbFeatureExtractor(n_features=500)
    # img1 = cv2.imread("1.png", flags=cv2.IMREAD_COLOR)
    # img2 = cv2.imread("2.png", flags=cv2.IMREAD_COLOR)
    # # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #
    # kp1, des1 = orb_extractor.get_features(img1)
    # kp2, des2 = orb_extractor.get_features(img2)
    #
    # bf_matcher = BruteForceFeatureMatcher(norm_type=NORM_HAMMING)
    # matches = bf_matcher.match_features(des1, des2, 25)
    #
    # print(pose_estimation_2d2d(kp1, kp2, matches, camera_matrix))
    #
    # bf_matcher.draw_matches(img1, kp1, img2, kp2, matches)
    # cv2.waitKey()


if __name__ == "__main__":
    # main()
    a = np.random.randint(0, 5, size=(6, 6))
    print(a)
    x, y = np.triu_indices_from(a)
    arr = a[x, y]
    res = np.zeros((6, 6))
    res[x, y] = arr
    np.fill_diagonal(res, 0)
    res = res.T
    res[x, y] = arr
    print(res)
