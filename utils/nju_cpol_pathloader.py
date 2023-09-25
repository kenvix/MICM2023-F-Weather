import os


def get_numpy_paths_of_dataset(data_dir, dim='dBZ'):
    """
    Get all the numpy files' path of the dataset.
    :param data_dir: The root directory of the dataset.
    :param dim: The dimension of the data.
    :param sub_dir: The sub directory of the dataset.
    :return: A list of numpy files' path.
    """
    data_dir = os.path.join(data_dir, dim)
    paths = {}
    for dir1 in os.listdir(data_dir):
        if dir1 not in paths:
            paths[dir1] = {}
        path_1 = os.path.join(data_dir, dir1)
        if os.path.isdir(path_1):
            for dir2 in os.listdir(path_1):
                if dir2 not in paths[dir1]:
                    path_2 = os.path.join(path_1, dir2)
                    dir3 = os.listdir(path_2)
                    paths[dir1][dir2] = [os.path.join(path_2, it) for it in dir3]
    return paths

def get_numpy_paths_of_dataset_label(data_dir, dim='kdp-rain'):
    """
    Get all the numpy files' path of the dataset.
    :param data_dir: The root directory of the dataset.
    :param dim: The dimension of the data.
    :param sub_dir: The sub directory of the dataset.
    :return: A list of numpy files' path.
    """
    data_dir = os.path.join(data_dir, dim)
    paths = {}
    path_1 = data_dir
    for dir2 in os.listdir(path_1):
        if dir2 not in paths:
            path_2 = os.path.join(path_1, dir2)
            dir3 = os.listdir(path_2)
            paths[dir2] = [os.path.join(path_2, it) for it in dir3]
    return paths
