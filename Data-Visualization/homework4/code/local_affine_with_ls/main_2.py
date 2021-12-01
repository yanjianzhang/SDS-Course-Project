import numpy as np
import sys

sys.path.append('\\')
from point_extract_interactive import point_extract_interative
from locally_affine_with_ls import locally_affine
from interpolation import linearInter

from numpy import tile, array, zeros, multiply, sqrt, divide, ones
from PIL import Image
from pylab import imshow
from pylab import array
from pylab import plot, scatter
from pylab import title
from pylab import show


## 初始化标记点并可视化
_, person = point_extract_interative('photo.jpg')
_, animal = point_extract_interative('ape.png')

colors = "bgrcmykw"
im2 = array(Image.open('./photo.jpg'))
imshow(im2)
for i in range(len(person)):
    x = [x for x, y in person[i]]
    y = [y for x, y in person[i]]
    plot(x, y, '*',c=colors[i])
title('Points of person')
show()

im = array(Image.open('./ape.png'))
imshow(im)
for i in range(len(animal)):
    x = [x for x, y in animal[i]]
    y = [y for x, y in animal[i]]
    plot(x, y, '*',c=colors[i])
title('Points of ape')
show()

## 调用仿射变换函数；output图片中（x,y)点在input中对应的坐标为xx(x,y),yy(x,y)
xx, yy = locally_affine(person, animal, im.shape,2)

## 初始化output图片
outIm = zeros([im.shape[0], im.shape[1], 3])

## 线性内插，生成output图片
for i in range(im.shape[1]):
    for j in range(im.shape[0]):
        outIm[j,i]=linearInter([yy[i,j],xx[i,j]],im2)

#可视化output图片
imshow(outIm.round()/256)
title('After transformation')
show()
