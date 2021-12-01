# Reference to page 6 in 3.2 histogram-based
from skimage import exposure
from skimage import data
from skimage import io, data, img_as_float
# import Image as Img
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def bincount(my_list, minlength=0):
    length = max(minlength, max(my_list))
    count = Counter(my_list)
    return np.array([count[x] for x in range(length)])


# 1. n-dimensional joint histogram
def eval_hist(my_data: np.ndarray, bins=(64, 64)):

    n_dim = len(bins)
    n_range = [[int(255/bins[i]) * j for j in range(bins[i]+1)]
               for i in range(n_dim)]
    nbin = np.empty(n_dim, int)
    for i in range(n_dim):
        nbin[i] = len(n_range[i])

    Ncount = tuple(
        np.searchsorted(n_range[i], my_data[:, i], side='right')
        for i in range(n_dim)
    )

    xy = np.ravel_multi_index(Ncount, nbin)

    hist = bincount(xy, minlength=nbin.prod())

    hist = hist.reshape(nbin)

    print(hist)
    return hist, n_range


my_data = data.moon()
hist, edges = eval_hist(my_data, bins=(64, 64))
print(len(edges[0]), hist.shape)
plt.pcolormesh(edges[0], edges[1], hist.T, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')
plt.show()
