import numpy as np
import cv2
import sys
import scipy

sys.path.append('\\')
from numpy import tile, array, zeros, multiply, sqrt, divide, linalg
from convexHull import convexHull
from PIL import Image
import matplotlib
from pylab import imshow
from pylab import array
from pylab import plot, scatter
from pylab import title
from pylab import show


def locally_affine(dOri, dTarget, sTarget, e):
    '''
    :param dOri: 人（input/Original）图像的标记点[(x,y),...]
    :param dTarget: 狒狒（output/Target）图像的标记点[(x,y),...]
    :param sTarget: 狒狒图像尺寸（output图像尺寸）
    :param e: 可调参数e
    :return: 平移情形下对应原图像的各坐标
    '''

    colors = "bgrcmykw"

    # 记录output图像尺寸
    sx, sy = sTarget[0], sTarget[1]
    n = len(dOri)  # 标记点组数

    # 初始化：output图片中（x,y)点在input中对应的坐标为xx(x,y),yy(x,y)
    xx = tile(array(range(sx)), (sy, 1)).transpose()
    yy = tile(array(range(sy)), (sx, 1))

    ## 记录变换S，距离D，中间变换G
    S = zeros([sx, sy, 2])
    D = zeros([sx, sy, n])
    Gx = zeros([sx, sy, n])  # x分量
    Gy = zeros([sx, sy, n])  # y分量

    # 采用数值的方法统一in range和out of range的表达式
    epsilon = 1e-5
    im = array(Image.open('photo.jpg'))
    imshow(im)
    # 对仿射变换组进行迭代
    for i in range(n):
        # 原始图片坐标初始化
        l = len(dOri[i])
        pOri = dOri[i]
        pTarget = dTarget[i]
        X = zeros([2 * l, 6])
        Y = zeros([2 * l, 1])
        for j in range(l):

            X[2 * j , :] = [pTarget[j][0], pTarget[j][1], 0, 0, 1, 0]
            X[2 * j + 1, :] = [0, 0,pTarget[j][0], pTarget[j][1], 0, 1]
            Y[2 * j, 0] = pOri[j][0]
            Y[2 * j+1, 0] = pOri[j][1]

        # 最小二乘计算仿射变换参数 [m1,m2,m3,m4,t1,t2] (XtX)-1Xtb
        beta = linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)
        y_hat = X.dot(beta).reshape([l,2])

        # 原始图片对应在target图片中的像素点
        x = [x for x, y in y_hat]
        y = [y for x, y in y_hat]
        plot(x, y, '*',c=colors[i])
        title('Points of ape after transform')

        # 计算距离 采用convexHull算法
        newpoints = np.array([[i,j] for i in range(sx) for j in range(sy)])
        D[:, :, i] = np.power(np.maximum(np.array(convexHull(np.array(pTarget),newpoints)),0).reshape([sx,sy])+epsilon,-e)

        # 计算中间变换
        for k in range(sx):
            for j in range(sy):
                Gx[k, j, i] = np.array([k, j, 0, 0, 1, 0]).dot(beta)
                Gy[k,j,i]=np.array([0, 0, k, j, 0, 1]).dot(beta)

        # 对加权后中间变换进行累加
        S[:, :, 0] += multiply(Gx[:, :, i], D[:, :, i])
        S[:, :, 1] += multiply(Gy[:, :, i], D[:, :, i])

    show()
    # 归一化
    xx=S[:, :, 0] / np.sum(D, axis=2)
    yy=S[:, :, 1] / np.sum(D, axis=2)
    x = xx.flatten().tolist()
    y = yy.flatten().tolist()

    # 可视化原图片
    im = array(Image.open('./photo.jpg'))
    imshow(im)
    # 可视化变换矩阵
    scatter(x, y, marker='.', s=1, alpha=0.2, c=[float(i) / len(x) for i in range(len(x))])
    title('Transform Matrix')
    show()
    return xx, yy

if __name__ == '__main__':
    matplotlib.rcParams['savefig.dpi'] = 200  # 图片像素
    matplotlib.rcParams['figure.dpi'] = 200  # 分辨率
    # 获取标记点
    person = {0: [(79.0151515151515, 120.98917748917745), (90.09740259740258, 115.4480519480519),
         (101.17965367965368, 115.10173160173156), (106.37445887445887, 118.91125541125538),
         (95.63852813852813, 122.0281385281385)]
      }
    animal = {0: [(62.73809523809521, 26.44372294372289), (75.55194805194805, 25.05844155844153),
         (98.75541125541125, 25.05844155844153), (112.26190476190476, 37.525974025973966),
         (85.94155844155841, 41.33549783549779)]
     }
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
    scatter(x, y, marker='.', s=1, alpha=0.2, c=[float(i) / len(x) for i in range(len(x))])
    title('Transform Matrix')
    show()
