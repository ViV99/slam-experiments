import os
import time
import yaml
import logging as logger
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
from jaxlie import SE3, SO3

from slam import OrbSLAM
from primitives import Camera


def main(settings_path: Path, path_to_image_folder: Path, path_to_times: Path, gt_path: Path):
    with open(settings_path) as f:
        settings = yaml.safe_load(f)

    gt_poses = load_gt_poses(gt_path)
    image_paths, timestamps = load_images(path_to_image_folder, path_to_times)

    camera = Camera(*settings["intrinsics"])
    slam = OrbSLAM(camera, gt_poses[0])
    slam.start()

    tracking_times = []

    logger.info("Start processing sequence ..."
                f"Images in the sequence: {len(image_paths)}")

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")
    translations = np.empty((0, 3))
    gt_translations = np.empty((0, 3))

    for idx in range(100):
        image = cv2.imread(str(image_paths[idx]), cv2.IMREAD_UNCHANGED)
        tframe = timestamps[idx]

        if image is None:
            logger.error(f"Failed to load image at {image_paths[idx]}")
            return 1

        t1 = time.time()
        slam.process(image, tframe)
        t2 = time.time()

        tracking_time = t2 - t1
        tracking_times.append(tracking_time)

        delta_frame_time = 0
        if idx < len(image_paths) - 1:
            delta_frame_time = timestamps[idx + 1] - tframe
        elif idx > 0:
            delta_frame_time = tframe - timestamps[idx - 1]

        # if tracking_time < delta_frame_time:
        #     time.sleep(delta_frame_time - tracking_time)
        pose = slam.get_last_pose()
        print(pose.translation(), gt_poses[idx].translation())
        translations = np.vstack((translations, pose.translation()))
        gt_translations = np.vstack((gt_translations, gt_poses[idx].translation()))


    ax.scatter(translations[:, 0], translations[:, 1], translations[:, 2], c="red", label="pred")
    ax.scatter(gt_translations[:, 0], gt_translations[:, 1], gt_translations[:, 2], c="blue", label="gt")
    plt.show()

    slam.stop()

    tracking_times = sorted(tracking_times)
    total_time = sum(tracking_times)
    print('-----')
    print('median tracking time: {0}'.format(tracking_times[len(image_paths) // 2]))
    print('mean tracking time: {0}'.format(total_time / len(image_paths)))

    return 0


def load_images(path_to_images: Path, path_to_times: Path) -> tuple[list[Path], list[float]]:
    image_paths = []
    timestamps = []
    time_dfs = pd.read_csv(path_to_times)
    for i in range(len(time_dfs)):
        timestamps.append(float(time_dfs.iloc[i, 0]) / 1e9)
        image_paths.append(path_to_images / time_dfs.iloc[i, 1])

    return image_paths, timestamps


def load_gt_poses(path_to_gt: Path) -> list[SE3]:
    poses = []
    gt_df = pd.read_csv(path_to_gt)
    for i in range(len(gt_df)):
        poses.append(SE3(
            wxyz_xyz=np.array([gt_df.iloc[i, 4], gt_df.iloc[i, 5], gt_df.iloc[i, 6], gt_df.iloc[i, 7],
                               gt_df.iloc[i, 1], gt_df.iloc[i, 2], gt_df.iloc[i, 3]])
        ))
    return poses


def save_trajectory(trajectory, filename):
    with open(filename, 'w') as traj_file:
        traj_file.writelines('{time} {r00} {r01} {r02} {t0} {r10} {r11} {r12} {t1} {r20} {r21} {r22} {t2}\n'.format(
            time=repr(t),
            r00=repr(r00),
            r01=repr(r01),
            r02=repr(r02),
            t0=repr(t0),
            r10=repr(r10),
            r11=repr(r11),
            r12=repr(r12),
            t1=repr(t1),
            r20=repr(r20),
            r21=repr(r21),
            r22=repr(r22),
            t2=repr(t2)
        ) for t, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 in trajectory)


if __name__ == '__main__':
    # print(cv2.imread("data/mav0/cam0/data/1403636579763555584.png"))
    main(
        Path("./config/orb.yaml"),
        Path("./data/mav0/cam0/data"),
        Path("./data/mav0/cam0/data.csv"),
        Path("./data/mav0/state_groundtruth_estimate0/data.csv")
    )
