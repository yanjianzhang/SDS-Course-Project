from pylab import show
from pylab import title
from pylab import plot, scatter
from pylab import array
from pylab import imshow
import matplotlib
from PIL import Image
from numpy import tile, array, zeros, multiply, sqrt, divide
from point_extract import point_extract
import numpy as np
import cv2
import sys
sys.path.append('\\')


def locally_affine(lOri, lTarget, sTarget, e):
    '''
    :param lOri: 人（input/Original）图像的标记点[(x,y),...]
    :param lTarget: 狒狒（output/Target）图像的标记点[(x,y),...]
    :param sTarget: 狒狒图像尺寸（output图像尺寸）
    :param e: 可调参数e
    :return: 平移情形下对应原图像的各坐标
    '''

    # 记录output图像尺寸
    sx, sy = sTarget[0], sTarget[1]
    n = len(lOri)  # 标记点个数

    # 初始化：output图片中（x,y)点在input中对应的坐标为xx(x,y),yy(x,y)
    xx = tile(array(range(sx)), (sy, 1)).transpose()
    yy = tile(array(range(sy)), (sx, 1))

    # 受标记点格式影响，将标记点xy逆序
    lOri = [(b, a) for a, b in lOri]
    lTarget = [(b, a) for a, b in lTarget]

    # 记录变换S，距离D，中间变换G
    S = zeros([sx, sy, 2])
    D = zeros([sx, sy, n])
    Gx = zeros([sx, sy, n])  # x分量
    Gy = zeros([sx, sy, n])  # y分量

    # 采用数值的方法统一in range和out of range的表达式
    epsilon = 1e-5

    # 对标记点迭代
    for i in range(n):
        # 原始图片坐标
        ox, oy = lOri[i]
        # 目标图片坐标
        tx, ty = lTarget[i]
        # update g
        g_bx = ox - tx
        g_by = oy - ty
        # 计算距离
        D[:, :, i] = np.power(
            sqrt(multiply(xx - tx, xx - tx) + multiply(yy - ty, yy - ty) + epsilon), -e)
        # 计算中间变换
        Gx[:, :, i] = xx + g_bx
        Gy[:, :, i] = yy + g_by
        # 对加权后中间变换进行累加
        S[:, :, 0] += multiply(Gx[:, :, i], D[:, :, i])
        S[:, :, 1] += multiply(Gy[:, :, i], D[:, :, i])

    # 归一化
    return S[:, :, 0] / np.sum(D, axis=2), S[:, :, 1] / np.sum(D, axis=2)


if __name__ == '__main__':

    matplotlib.rcParams['savefig.dpi'] = 1000  # 图片像素
    matplotlib.rcParams['figure.dpi'] = 1000  # 分辨率
    # 获取标记点
    animal, person = point_extract()
    # output image
    im = array(Image.open('./ape.png'))
    # 获取变换矩阵
    xx, yy = locally_affine(person, animal, im.shape, 1)
    x = xx.flatten().tolist()
    y = yy.flatten().tolist()
    # 可视化原图片
    im = array(Image.open('./photo.jpg'))
    imshow(im)
    # 可视化变换矩阵
    scatter(y, x, marker='.', s=1, alpha=0.2, c=[
            float(i)/len(x) for i in range(len(x))])
    show()

