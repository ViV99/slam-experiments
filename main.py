from typing import Sequence, Optional

import cv2
import g2o
import jaxlie
import numpy as np
from cv2 import NORM_HAMMING, KeyPoint, DMatch, findEssentialMat, findHomography, recoverPose, triangulatePoints
from jaxlie import SO3, SE3

from feature_exractors import OrbFeatureExtractor
from feature_matchers import BruteForceFeatureMatcher


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

    with open("sphere.g2o", "r") as f:
        vertex_cnt, edge_cnt = 0, 0
        for line in f.readlines():
            arr = line.split()

            if arr[0] == "VERTEX_SE3:QUAT":
                idx = int(arr[1])
                se3 = SE3(np.array(arr[2:], dtype=np.float32))

                v = g2o.VertexSE3()
                v.set_id(idx)
                v.set_estimate(g2o.Isometry3d(se3.as_matrix()))

                optimizer.add_vertex(v)
                if idx == 0:
                    v.set_fixed(True)
                vertex_cnt += 1

            elif arr[0] == "EDGE_SE3:QUAT":
                idx1, idx2 = int(arr[1]), int(arr[2])
                se3 = SE3(np.array(arr[3:10], dtype=np.float32))
                info = np.zeros((6, 6))
                i_u, j_u = np.triu_indices(6)
                i_l, j_l = np.tril_indices(6, k=-1)
                info[i_u, j_u] = arr[10:]
                info[i_l, j_l] = info[j_l, i_l]

                e = g2o.EdgeSE3()
                e.set_id(edge_cnt)
                e.set_vertex(0, optimizer.vertex(idx1))
                e.set_vertex(1, optimizer.vertex(idx2))
                e.set_measurement(g2o.Isometry3d(se3.as_matrix()))
                e.set_information(info)

                optimizer.add_edge(e)
                edge_cnt += 1

    print(f"Total: {vertex_cnt} vertices | {edge_cnt} edges")
    optimizer.initialize_optimization()
    optimizer.optimize(30)

    optimizer.save("result.g2o")



if __name__ == "__main__":
    main()
