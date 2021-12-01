import numpy as np
import cv2


def linearInter(X_, img_human):
    """
    :param X_: 狒狒图像的像素 Y=(i, j) 经局部仿射后对应到人脸图像的坐标 X_=(x, y)
    :param img_human: 人脸图
    :return: 线性插值后得到的人脸图上的灰度值f(x, y)
    """
    x, y = X_[0], X_[1]
    # 人脸图中与X_相邻的四个点
    f_y, c_y = int(np.floor(y)), int(np.ceil(y))  # 对y向下/向上取整
    f_x, c_x = int(np.floor(x)), int(np.ceil(x))  # 对x向下/向上取整
    p_2, p_3 = img_human[(f_x, f_y)], img_human[(f_x, c_y)]
    p_0, p_1 = img_human[(c_x, f_y)], img_human[(c_x, c_y)]
    r = y - f_y
    s = c_x - x
    ret = (1 - s) * (1 - r) * p_0 + (1 - s) * r * p_1 + s * (1 - r) * p_2 + s * r * p_3

    return ret


# img_human = cv2.imread('../img_input/img1.jpg', cv2.IMREAD_GRAYSCALE)
# print(linearInter((2.45, 3.23), img_human))
