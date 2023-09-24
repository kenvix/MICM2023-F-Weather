import os

import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
dir = './dataset/test/unkn'

for file in os.listdir(f'{dir}'):
    # Load the .npy file
    data_groundtruth = np.load(f'{dir}/{file}') - 0.2

    plt.imshow(data_groundtruth, cmap='Blues')
    plt.title(file)
    # plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()
