from main import Frontend, Frame


def main(vocab_path, settings_path, path_to_image_folder, path_to_times_file):



    return 0


def load_images(path_to_images, path_to_times_file):
    image_files = []
    timestamps = []
    with open(path_to_times_file) as times_file:
        for line in times_file:
            timestamps.append(float(line) / 1e9)
            image_files.append(os.path.join(path_to_images, "{0}.png".format(line.rstrip())))
    return image_files, timestamps


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
    if len(sys.argv) != 5:
        print('Usage: ./orbslam_mono_euroc path_to_vocabulary path_to_settings path_to_image_folder path_to_times_file')
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])