from skimage import exposure
from skimage import data
from skimage import io, data, img_as_float
# import Image as Img
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import copy
from skimage.color import rgb2gray
from skimage.color import rgb2lab, lab2rgb
from skimage.util import img_as_ubyte


image = io.imread("./image/earth.jpg")

dtype = image.dtype.type
# print(dtype)


image = rgb2gray(image)
height, width = image.shape

io.imsave("./image/earth_grey.jpg", img_as_ubyte(image))


image = io.imread("./image/earth_grey.jpg")


pseudoimage = np.zeros((height, width, 3), dtype='float64')

for i in range(height):
    for j in range(width):
        pseudoimage[i][j][0] = 0.0-255.0*image[i][j]
        pseudoimage[i][j][1] = 127.0-255.0*image[i][j]
        pseudoimage[i][j][2] = 255.0-255.0*image[i][j]


pseudoimage = dtype(pseudoimage)

io.imsave("./image/earth_pseudo.jpg", pseudoimage)
plt.show()
