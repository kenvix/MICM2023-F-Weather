import matplotlib.pyplot as plt
import numpy as np

# Load the image as a numpy array
img = plt.imread('./dataset/visualize-test.jpg')

# Print the shape of the loaded image
print(img.shape)

plt.imshow(img[:, :, 0], cmap='gray')
plt.show()
