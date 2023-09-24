import os

import numpy as np
import matplotlib.pyplot as plt


# Load the .npy file
def _entry(dir='./dataset/test/unkn', save=None):
    for file in os.listdir(f'{dir}'):
        # Load the .npy file
        data_groundtruth = np.load(f'{dir}/{file}') - 0.2

        plt.imshow(data_groundtruth, cmap='Blues')
        plt.title(file)
        # plt.gca().invert_yaxis()
        plt.colorbar()
        if save is not None:
            if not os.path.exists(save):
                os.makedirs(save)
            plt.savefig(f'{save}/{file}.svg')
        else:
            plt.show()
        plt.close()


if __name__ == '__main__':
    from fire import Fire

    Fire(_entry)
