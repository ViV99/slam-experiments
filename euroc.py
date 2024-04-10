import os
import time
import yaml
import logging as logger
from pathlib import Path

import cv2
import pandas as pd

from slam import OrbSLAM
from primitives import Camera


def main(settings_path: Path, path_to_image_folder: Path, path_to_times: Path):
    with open(settings_path) as f:
        settings = yaml.safe_load(f)

    image_paths, timestamps = load_images(path_to_image_folder, path_to_times)

    camera = Camera(*settings["intrinsics"])
    slam = OrbSLAM(camera)
    slam.start()

    tracking_times = []

    logger.info("Start processing sequence ..."
                f"Images in the sequence: {len(image_paths)}")

    for idx in range(len(image_paths)):
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

        print(slam.get_last_pose())

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
    main(Path("./config/orb.yaml"), Path("./data/mav0/cam0/data"), Path("./data/mav0/cam0/data.csv"))
