from skimage import exposure
from skimage import data
from skimage import io, data, img_as_float
# import Image as Img
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import copy
from skimage.color import rgb2gray


def smooth(image, N=9):
    operator = np.array([[1./(N*N)] * N] * N)
    new_image = copy.deepcopy(image)
    np_image = np.array(image)
    for i in range(int((N-1)/2), len(image)-int((N-1)/2)):
        for j in range(int((N-1)/2), len(image[0])-int((N-1)/2)):
            new_image[i][j] = np.sum(
                np_image[i-int((N-1)/2): i+int((N+1)/2), j-int((N-1)/2): j+int((N+1)/2)] * operator)
            # new_image[i][j] = np.sum(np_image[i-1: i+2, j-1: j+2] * operator)
    image_max = np.max(new_image)
    image_min = np.min(new_image)
    new_image = [[(p-image_min)/(image_max-image_min)*255
                  for p in ps] for ps in new_image]
    # print(new_image)
    return new_image


def sharp(image):
    operator = np.array(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    np_image = np.array(image)
    filter_image = np.zeros(np_image.shape)
    for i in range(1, len(image)-1):
        for j in range(1, len(image[0])-1):
            filter_image[i][j] = 0.1 * np.sum(np_image[i-1: i+2, j-1: j+2] * operator)
    new_image = np_image + filter_image
    image_max = np.max(new_image)
    image_min = np.min(new_image)
    new_image = [[(p-image_min)/(image_max-image_min)*255
                  for p in ps] for ps in new_image]
    return filter_image, new_image

# image = data.moon()
image = io.imread("./image/pattern.jpg")
dtype = image.dtype.type
print(image)
image =  rgb2gray(image)
image = dtype(list([int(i*255) for i in j] for j in image))



print(image)
dtype = image.dtype.type
io.imsave("./image/moon_gray.jpg", image)
plt.subplot(2, 2, 1)
plt.imshow(image, plt.cm.gray)
plt.axis('off')
plt.title('Origin Image')

image_smooth = smooth(image)
image_smooth = dtype(image_smooth)
io.imsave("./image/img_smooth.jpg", image_smooth)
plt.subplot(2, 2, 2)
plt.imshow(image_smooth, plt.cm.gray)
plt.axis('off')
plt.title('Smooth Image')

image = io.imread("./image/moon_gray.jpg")
dtype = image.dtype.type
io.imsave("./image/img.jpg", image)
image_edge, image_sharp = sharp(image)
image_sharp = dtype(image_sharp)
image_edge = dtype(image_edge)
io.imsave("./image/img_sharp.jpg", image_sharp)

plt.subplot(2, 2, 3)
plt.imshow(image_edge, plt.cm.gray)
plt.axis('off')
plt.title('Sharp Filter Image')

plt.subplot(2, 2, 4)
plt.imshow(image_sharp, plt.cm.gray)
plt.axis('off')
plt.title('Sharp Image')
plt.show()
