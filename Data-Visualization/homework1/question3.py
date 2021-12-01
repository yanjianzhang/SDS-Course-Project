from skimage import exposure
from skimage import data
from skimage import io, data, img_as_float
# import Image as Img
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import copy
from tqdm import tqdm, trange


# 3.
def hist_equal(image, bins=256):
    size = image.shape[0]
    hist = bincount(image.reshape((-1)), minlength=bins)
    img_cdf = hist.cumsum()
    img_cdf = img_cdf / float(img_cdf[-1])
    out = np.interp(
        image.reshape(-1), range(256), img_cdf)
    out = np.array([int(x * 256) for x in out])
    out = out.reshape(size, size)
    new_img = copy.deepcopy(image)
    for i in range(size):
        for j in range(size):
            new_img[i][j] = out[i][j]
    return new_img

def bincount(my_list, minlength=0):
    length = max(minlength, max(my_list))
    count = Counter(my_list)
    return np.array([count[x] for x in range(length)])

def efficien_compute(old_hist, removed_pixels, extend_pixels, bins=256):
    hist_remove = bincount(removed_pixels.reshape(-1), minlength=bins)
    hist_extend = bincount(extend_pixels.reshape(-1), minlength=bins)

    new_list = copy.deepcopy(old_hist)
    new_list = new_list - hist_remove + hist_extend

    return new_list


def local_equal(image, bins=256, filter=64, uni_step = 5):
    origin_image = copy.deepcopy(image) # keep an original image unchanged
    steps = int((image.shape[0] - filter))
    steps2hist = {(i, j): np.array([0])
                  for i in range(steps) for j in range(0, steps, uni_step)}

    hist_init = bincount(
        origin_image[0:filter, 0:filter].reshape(-1), minlength=bins)
    steps2hist[(0, 0)] = hist_init
    local_process = tqdm(range(int(steps*steps/(uni_step*uni_step))))
    for i in range(0, steps, uni_step):
        for j in range(0, steps, uni_step):
            new_hist = np.array([0]) # initial as empty array
            if (i-uni_step, j) in steps2hist and steps2hist[(i-uni_step, j)].any():
                new_hist = efficien_compute(
                    steps2hist[(i-uni_step, j)], removed_pixels = origin_image[i-uni_step:i, j:j+filter], extend_pixels = origin_image[i+filter-uni_step:i+filter, j:j+filter])
            elif (i, j - uni_step) in steps2hist and steps2hist[(i, j - uni_step)].any():
                new_hist = efficien_compute(
                    steps2hist[(i, j - uni_step)], origin_image[i:i + filter, j - uni_step: j],
                    origin_image[i:i + filter, j + filter - uni_step:j + filter])
            if new_hist.any():
                steps2hist[(i, j)] = new_hist
                sum_count = sum(new_hist)
                mean_hist = np.mean(
                    [i*new_hist[i]/sum_count for i in range(len(new_hist))])
                if mean_hist > 0.4 and mean_hist < 0.5:
                    img_cdf = new_hist.cumsum()
                    img_cdf = img_cdf / float(img_cdf[-1])
                    out = np.interp(
                        origin_image[i:i + filter, j:j + filter].reshape(-1), range(256), img_cdf)
                    out = np.array([float(x*256) for x in out])
                    out = out.reshape(filter, filter)
                    for m in range(filter):
                        for n in range(filter):
                            image[i+m][j+n]=out[m][n]

            local_process.update(1)

    return image


image=data.moon()
print("Processing the global histogram equalization")
uniformed_image = hist_equal(image)
io.imsave("img_global.jpg", uniformed_image)
plt.imshow(uniformed_image, plt.cm.gray)
plt.show()

print("Processing the local histogram equalization")
local_uniformed_image=local_equal(image, filter=64, uni_step= 5)
io.imsave("img_local.jpg", local_uniformed_image)
plt.imshow(local_uniformed_image, plt.cm.gray)
plt.show()
