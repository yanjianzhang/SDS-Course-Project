from pylab import show
from pylab import title
from pylab import plot, scatter
from pylab import array
from pylab import imshow
import matplotlib
from PIL import Image
from numpy import tile, array, zeros, multiply, sqrt, divide, ones
from interpolation import linearInter
from locally_affine import locally_affine
from point_extract import point_extract
import numpy as np
import sys

sys.path.append('\\')


# matplotlib.rcParams['savefig.dpi'] = 500  # 图片像素
# matplotlib.rcParams['figure.dpi'] = 500  # 分辨率

# 初始化标记点并可视化
animal, person = point_extract()
im2 = array(Image.open('./images/photo.jpg'))
# imshow(im2)
# x = [x for x, y in person]
# y = [y for x, y in person]
# plot(x, y, 'r*')
# title('Points of person')
# show()

# 初始化标记点并可视化
im = array(Image.open('./images/ape.png'))
imshow(im)
x = [x for x, y in animal]
y = [y for x, y in animal]
plot(x, y, 'r*')
title('Points of ape')
show()

# 调用仿射变换函数；output图片中（x,y)点在input中对应的坐标为xx(x,y),yy(x,y)
xx, yy = locally_affine(person, animal, im.shape, 2)

# 避免超出范围
xx = np.minimum(xx, (im2.shape[0]-1)*ones(xx.shape))
yy = np.minimum(yy, (im2.shape[1]-1)*ones(yy.shape))

# 初始化output图片
outIm = zeros([im.shape[0], im.shape[1], 3])

# 线性内插，生成output图片

print(im2.shape)
for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        outIm[i, j] = linearInter([xx[i, j], yy[i, j]], im2)

# 可视化output图片
imshow(outIm.round()/256)
title('After transformation')
show()
