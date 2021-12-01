from skimage import exposure
from skimage import data
from skimage import io, data, img_as_float
# import Image as Img
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


# 2. Contrast stretching
def linear(value):
    r1, s1 = (0.375*255, 0.125*255)
    r2, s2 = (0.625*255, 0.875*255)
    # print(value)
    if value <= r1:
        return float(s1)/r1*value
    elif value <= r2:
        return s1 + float(s2-s1)/(r2-r1)*(value-r1)
    else:
        return s2 + float(255.0-s2)/(255.0-r2)*(value-r2)


image = data.moon()
io.imsave("img.jpg", image)
dtype = image.dtype.type
img_rescale = dtype(list(
    [[linear(v) for v in item] for item in image]))
io.imsave("img_rescale.jpg", img_rescale)
plt.imshow(img_rescale, plt.cm.gray)
plt.show()
