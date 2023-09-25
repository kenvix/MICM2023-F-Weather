import os

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

# Load the .npy file
def _entry(dir='', save=None, format='png'):
    for dir_path, dir_names, filenames in os.walk(dir):
        for file in filenames:
            if str(file).endswith('.npy'):

                path_in = os.path.join(dir_path, file)
                if not os.path.exists(f'{path_in}.{format}'):
                    logger.info(path_in)
                    # Load the .npy file
                    data_groundtruth = np.load(path_in)

                    plt.imshow(data_groundtruth, cmap='Blues')
                    plt.title(path_in)

                    plt.colorbar()
                    if save is not None:
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)
                        plt.savefig(f'{path_in}.{format}')
                    else:
                        plt.show()
                    plt.close()


if __name__ == '__main__':
    from fire import Fire

    Fire(_entry)
