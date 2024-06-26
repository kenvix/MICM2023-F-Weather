import multiprocessing
import os.path

import numpy as np
from loguru import logger

logger.info("Application Initialzing ...")

from utils import dataset_cpol_dir_test, dataset_base_dir
import matplotlib.pyplot as plt

def _entry():
    run()
def run():
    visual_in = dataset_cpol_dir_test
    visual_out = os.path.join(dataset_base_dir, 'visual')
    if not os.path.exists(visual_out):
        os.mkdir(visual_out)
    max_dir_item = 15

    for dir1 in os.listdir(visual_in):
        path_1 = os.path.join(visual_in, dir1)
        if dir1 == 'kdp-rain':
            continue
        for dir2 in os.listdir(path_1):
            # if dir2 == '1.0km':
            #     continue
            path_2 = os.path.join(path_1, dir2)
            count_3 = 0
            for dir3 in os.listdir(path_2):
                path_3 = os.path.join(path_2, dir3)
                if count_3 >= max_dir_item:
                    break
                for dir4 in os.listdir(path_3):
                    path_4 = os.path.join(path_3, dir4)
                    logger.info(f"Processing {path_4} ...")
                    data = np.load(path_4)
                    path_out_dir = os.path.join(visual_out, dir1, dir2, dir3)
                    if not os.path.exists(path_out_dir):
                        os.makedirs(path_out_dir)
                    path_out = os.path.join(path_out_dir, dir4 + ".svg")
                    if os.path.exists(path_out):
                        logger.info(f"Skip {path_out} ...")
                        continue

                    fig = plt.figure()
                    plt.imshow(data, cmap='Blues')
                    # plt.gca().invert_yaxis()
                    plt.colorbar()
                    fig.savefig(path_out)
                    plt.close()

                count_3 += 1

if __name__ == '__main__':
    _entry()
