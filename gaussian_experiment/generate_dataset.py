import numpy as np
from matplotlib import pyplot as plt
import math

for i in range(5, 12):
    num_modes = i**2
    modes_per_row = i
    num_samples = 10000
    samples_per_class = int(num_samples/num_modes)
    result = np.empty((0, 2))

    for i in range(modes_per_row):
        for j in range(modes_per_row):
            pts = np.random.multivariate_normal(
                [i*2, j*2], [[.0025, 0], [0, .0025]], size=samples_per_class, check_valid='warn')
            print(pts.shape)
            print(np.std(pts, axis=0))
            result = np.concatenate((result, pts), axis=0)
    for i in range(modes_per_row):
        for j in range(modes_per_row):
            if len(result) >= num_samples:
                break
            pts = np.random.multivariate_normal(
                [i*2, j*2], [[.0025, 0], [0, .0025]], size=1, check_valid='warn')
            print(pts.shape)
            print(np.std(pts, axis=0))
            result = np.concatenate((result, pts), axis=0)
    print(len(result))
    np.save('%d_modes_gaussian_ptp' % num_modes, np.ptp(result))
    np.save('%d_modes_gaussian_min' % num_modes, np.min(result))
    result = 2.*(result - np.min(result))/np.ptp(result)-1
    np.save('%d_modes_gaussian_10000' % num_modes, result)
