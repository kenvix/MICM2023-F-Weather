import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

def poly_func2(x, b, c):
    return b * x + c

def poly_func3(x, a, b, c):
    return a * (x ** 2) + b * x + c


def _entry(dir='./dataset/test/', mode='corresponding'):
    x = []
    y = []
    out_files = os.listdir(f'{dir}/out')
    data_outputs = []

    idx = 0
    for file in os.listdir(f'{dir}/gt'):
        # Load the .npy file
        if mode == 'corresponding':
            data_groundtruth = np.load(f'{dir}/gt/{file}')
            data_output = np.load(f'{dir}/out/{file}')
        else:
            data_groundtruth = np.load(f'{dir}/gt/{file}')
            data_output = np.load(f'{dir}/out/{out_files[idx]}')
            data_outputs.append(data_output)

        data_groundtruth_vector = data_groundtruth.reshape(-1)
        data_output_vector = data_output.reshape(-1)

        y.append(data_groundtruth_vector.mean())
        x.append(data_output_vector.mean())

        plt.imshow(data_groundtruth, cmap='Blues')
        plt.title("gt-" + file)
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(data_output, cmap='Blues')
        plt.title("out-" + file)
        plt.colorbar()
        plt.show()
        plt.close()

        idx += 1

    print(x)
    print(y)
    param = None
    if idx <= 2:
        v = curve_fit(poly_func2, x, y)
        param = v[0]
        print(v)
    else:
        v = curve_fit(poly_func3, x, y)
        param = v[0]
        print(v)

    idx = 0
    for out in data_outputs:
        out = poly_func2(out, *param)

        plt.imshow(out, cmap='Blues')
        plt.colorbar()
        plt.title(f'fix-{out_files[idx]}')
        plt.show()
        plt.close()
        idx += 1

if __name__ == '__main__':
    from fire import Fire

    Fire(_entry)
