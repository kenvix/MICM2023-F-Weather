import os

import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
dir = './dataset/test/'

for file in os.listdir(f'{dir}/gt'):
    # Load the .npy file
    data_groundtruth = np.load(f'{dir}/gt/{file}')
    data_output = np.load(f'{dir}/out/{file}')

    plt.imshow(data_groundtruth, cmap='Blues')
    # plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()
    plt.imshow(data_output, cmap='Blues')
    # plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()
