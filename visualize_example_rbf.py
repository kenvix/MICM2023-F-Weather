import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

# Load the .npy file
data = np.load('./dataset/data_dir_000/frame_000.npy')

plt.imshow(data, cmap='Blues')
# plt.gca().invert_yaxis()
plt.colorbar()
plt.show()

# 找到缺失值的位置
missing_mask = (data == 0)

# 获取缺失值的坐标和数值
missing_coords = np.argwhere(missing_mask)
missing_values = data[missing_mask]

# 判断 0 值是否需要填充
for i in range(len(data)):
    for j in range(len(data[0])):
        if data[i][j] == 0 and ((i > 0 and data[i-1][j] != 0) or (i < len(data)-1 and data[i+1][j] != 0) or
                                (j > 0 and data[i][j-1] != 0) or (j < len(data[0])-1 and data[i][j+1] != 0)):
            missing_mask[i][j] = True
        else:
            missing_mask[i][j] = False

# 获取非缺失值的坐标和数值
non_missing_coords = np.argwhere(~missing_mask)
non_missing_values = data[~missing_mask]

# 使用 RBF 插值填充缺失值
rbf = Rbf(non_missing_coords[:, 0], non_missing_coords[:, 1], non_missing_values, function='multiquadric')
filled_values = rbf(missing_coords[:, 0], missing_coords[:, 1])

# 将填充后的数值放回原始数据中
data[missing_mask] = filled_values

plt.imshow(data, cmap='Blues')
# plt.gca().invert_yaxis()
plt.colorbar()
plt.show()
