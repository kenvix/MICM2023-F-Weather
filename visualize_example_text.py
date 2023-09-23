import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
data = np.load('./dataset/data_dir_000/frame_000.npy')

fig, ax = plt.subplots()
im = ax.imshow(data, cmap='Blues')
# plt.gca().invert_yaxis()
for i in range(len(data)):
    for j in range(len(data[0])):
        plt.annotate(str(round(data[i][j], 2)), xy=(j, i), ha='center', va='center')

# 调整像素大小和字体大小
fig.set_size_inches(2 ^ 16, 2 ^ 16)
plt.rcParams.update({'font.size': 1})
plt.tight_layout()

# Add a colorbar to the image
cbar = ax.figure.colorbar(im, ax=ax)

plt.savefig('1.svg')
plt.show()
