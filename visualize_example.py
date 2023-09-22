import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
data = np.load('./dataset/data_dir_000/frame_000.npy')

plt.imshow(data, cmap='Blues')
# plt.gca().invert_yaxis()
plt.colorbar()
plt.show()
