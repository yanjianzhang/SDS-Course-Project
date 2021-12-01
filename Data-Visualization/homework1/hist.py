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


# # 1. n-dimensional joint histogram
# def eval_hist(my_data: np.ndarray, bins=(64, 64)):

#     n_dim = len(bins)
#     n_range = [[int(255/bins[i]) * j for j in range(bins[i]+1)]
#                for i in range(n_dim)]
#     nbin = np.empty(n_dim, int)
#     for i in range(n_dim):
#         nbin[i] = len(n_range[i])

#     Ncount = tuple(
#         np.searchsorted(n_range[i], my_data[:, i], side='right')
#         for i in range(n_dim)
#     )

#     xy = np.ravel_multi_index(Ncount, nbin)

#     hist = bincount(xy, minlength=nbin.prod())

#     hist = hist.reshape(nbin)

#     print(hist)
#     return hist, n_range


# my_data = data.moon()
# hist, edges = eval_hist(my_data, bins=(64, 64))
# print(len(edges[0]), hist.shape)
# plt.pcolormesh(edges[0], edges[1], hist.T, cmap='Blues')
# cb = plt.colorbar()
# cb.set_label('counts in bin')
# plt.show()


# # 2. Contrast stretching
# def linear(value):
#     r1, s1 = (0.375*255, 0.125*255)
#     r2, s2 = (0.625*255, 0.875*255)
#     # print(value)
#     if value <= r1:
#         return float(s1)/r1*value
#     elif value <= r2:
#         return s1 + float(s2-s1)/(r2-r1)*(value-r1)
#     else:
#         return s2 + float(255.0-s2)/(255.0-r2)*(value-r2)


# image = data.moon()
# io.imsave("img.jpg", image)
# dtype = image.dtype.type
# img_rescale = dtype(list(
#     [[linear(v) for v in item] for item in image]))
# io.imsave("img_rescale.jpg", img_rescale)


# 3.
# def hist_equal(image, bins=256):
#     hist = np.bincount(image.reshape((-1)), minlength=bins)
#     uni_hist = (bins - 1) * (np.cumsum(hist)/float(hist.size))
#     height, width = image.shape
#     new_img = np.zeros(image.shape)
#     # print(uni_hist)
#     # exit(886)
#     for i in range(height):
#         for j in range(width):
#             new_img[i][j] = uni_hist[image[i][j]]
#     return new_img


# def efficien_compute(old_hist, removed_pixels, pixels, bins=256):
#     hist_remove = np.bincount(removed_pixels.reshape((-1)), minlength=bins)
#     hist_extend = np.bincount(pixels.reshape((-1)), minlength=bins)
#     new_list = old_hist
#     for i in range(old_hist.shape[0]):
#         new_list[i] = old_hist[i] - hist_remove[i] + hist_extend[i]
#     return new_list


# def local_equal(image, bins=256, filter=128, unit_step=1):

#     steps = int((image.shape[0] - filter)/unit_step)
#     # print(steps)
#     # exit(88)
#     steps2hist = {(i, j): np.array([0])
#                   for i in range(steps) for j in range(steps)}

#     hist_init = np.bincount(
#         image[0:filter, 0:filter].reshape((-1)), minlength=bins)
#     steps2hist[(0, 0)] = hist_init
#     for i in range(0, steps, unit_step):
#         for j in range(0, steps, unit_step):
#             if (i-1, j) in steps2hist and steps2hist[(i-1, j)].any():
#                 new_hist = efficien_compute(
#                     steps2hist[(i-unit_step, j)], image[i, j:j+filter], image[i+filter, j:j+filter])
#                 steps2hist[(i, j)] = new_hist
#                 sum_count = sum(new_hist)
#                 # print(new_hist)
#                 # print("sum", sum_count)
#                 # exit(886)

#                 mean_hist = np.mean(
#                     [i*new_hist[i]/sum_count for i in range(len(new_hist))])
#                 print(mean_hist)
#                 if mean_hist > 0.4 and mean_hist < 0.5:
#                     # print("in", mean_hist)
#                     # uni_hist = (bins - 1) * \
#                     #     (np.cumsum(new_hist)/float(new_hist.size))
#                     img_cdf = new_hist.cumsum()
#                     img_cdf = img_cdf / float(img_cdf[-1])
#                     out = np.interp(
#                         image.flat[i:i+filter, j:j+filter], mean_hist, cdf)
#                     for m in range(filter):
#                         for n in range(filter):
#                             np.interp(
#                             image[i+m][j+n]=out[m][n]
#                 # continue
#             if (i, j-1) in steps2hist and steps2hist[(i, j-1)].any():
#                 new_hist=efficien_compute(
#                     steps2hist[(i, j-1)], image[i:i+filter, j], image[i:i+filter, j+filter])
#                 steps2hist[(i, j)]=new_hist
#                 sum_count=sum(new_hist)
#                 uni_hist=(bins - 1) *
#                     (np.cumsum(new_hist)/float(new_hist.size))
#                 # print(uni_hist)
#                 # exit(886)
#                 mean_hist=np.mean(
#                     [i*new_hist[i]/sum_count for i in range(len(new_hist))])
#                 print(mean_hist)
#                 if mean_hist > 0.4 and mean_hist < 0.5:
#                     # print("in", mean_hist)
#                     # uni_hist = (bins - 1) * \
#                     #     (np.cumsum(new_hist)/float(new_hist.size))
#                     img_cdf=new_hist.cumsum()
#                     img_cdf=img_cdf / float(img_cdf[-1])
#                     out=np.interp(
#                         image.flat[i:i+filter, j:j+filter], mean_hist, cdf)
#                     for m in range(filter):
#                         for n in range(filter):
#                             np.interp(
#                             image[i+m][j+n]=out[m][n]
#     return image


# image = data.moon()

# # uniformed_image = hist_equal(image)
# # io.imsave("img_u.jpg", uniformed_image)
# # plt.imshow(uniformed_image, plt.cm.gray)
# local_uniformed_image = local_equal(image)
# print(local_uniformed_image)
# io.imsave("img_u.jpg", local_uniformed_image)
# plt.imshow(local_uniformed_image, plt.cm.gray)
# plt.show()
