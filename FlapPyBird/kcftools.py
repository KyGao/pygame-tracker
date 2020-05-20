# -*- coding: utf-8 -*

import numpy as np
import cv2


def fftd(img, backwards=False):
    """
    傅里叶变换工具
    :param img: 输入图像
    :param backwards: 默认为False
    :return: 傅里叶变换结果
    """
    return cv2.dft(np.float32(img), flags=((cv2.DFT_INVERSE | cv2.DFT_SCALE) if backwards else cv2.DFT_COMPLEX_OUTPUT))


def real(img):
    """
    返回图像实部
    :param img: 输入图像
    :return: 返回实部
    """
    return img[:, :, 0]


def imag(img):
    """
    返回图像虚部
    :param img: 输入图像
    :return: 返回虚部
    """
    return img[:, :, 1]


def complexMultiplication(a, b):
    """
    复数乘法
    :param a: 复数a
    :param b: 负数b
    :return: 返回乘法结果
    """
    res = np.zeros(a.shape, a.dtype)

    res[:, :, 0] = a[:, :, 0] * b[:, :, 0] - a[:, :, 1] * b[:, :, 1]
    res[:, :, 1] = a[:, :, 0] * b[:, :, 1] + a[:, :, 1] * b[:, :, 0]
    return res


def complexDivision(a, b):
    """
    复数除法
    :param a: 负数a
    :param b: 负数b
    :return: 返回除法结果
    """
    res = np.zeros(a.shape, a.dtype)
    divisor = 1. / (b[:, :, 0] ** 2 + b[:, :, 1] ** 2)

    res[:, :, 0] = (a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1]) * divisor
    res[:, :, 1] = (a[:, :, 1] * b[:, :, 0] + a[:, :, 0] * b[:, :, 1]) * divisor
    return res


def rearrange(img):
    """
    实现图像区域的分块移动
    即将  [[0, 1],[2, 3]] 变为 [[3, 2], [1, 0]]
    :param img: 输入图像
    :return: 返回移动后的图像
    """
    assert (img.ndim == 2)
    new_img = np.zeros(img.shape, img.dtype)
    xh, yh = img.shape[1] // 2, img.shape[0] // 2
    new_img[0:yh, 0:xh], new_img[yh:img.shape[0], xh:img.shape[1]] = img[yh:img.shape[0], xh:img.shape[1]], img[0:yh,
                                                                                                            0:xh]
    new_img[0:yh, xh:img.shape[1]], new_img[yh:img.shape[0], 0:xh] = img[yh:img.shape[0], 0:xh], img[0:yh,
                                                                                                 xh:img.shape[1]]
    return new_img


def x2(rect):
    """
    取矩形框的下边界
    :param rect: 矩形框[x, y, height, width]
    :return: 返回下边界值
    """
    return rect[0] + rect[2]


def y2(rect):
    """
    取矩形框的右边界
    :param rect: 矩形框[x, y, height, width]
    :return: 返回右边界值
    """
    return rect[1] + rect[3]


def limit(rect, limit):
    """
    把矩形框限制在limit之内
    :param rect: 矩形框
    :param limit: 限制框
    :return: 返回被限制之后的矩形框
    """
    if rect[0] + rect[2] > limit[0] + limit[2]:
        rect[2] = limit[0] + limit[2] - rect[0]
    if rect[1] + rect[3] > limit[1] + limit[3]:
        rect[3] = limit[1] + limit[3] - rect[1]
    if rect[0] < limit[0]:
        rect[2] -= (limit[0] - rect[0])
        rect[0] = limit[0]
    if rect[1] < limit[1]:
        rect[3] -= (limit[1] - rect[1])
        rect[1] = limit[1]
    if rect[2] < 0:
        rect[2] = 0
    if rect[3] < 0:
        rect[3] = 0
    return rect


def getBorder(original, limited):
    """
    获取超出limit框的部分
    :param original: 原始框
    :param limited: limit框
    :return: 返回原始框超出limit框的部分
    """
    res = [0, 0, 0, 0]
    res[0] = limited[0] - original[0]
    res[1] = limited[1] - original[1]
    res[2] = x2(original) - x2(limited)
    res[3] = y2(original) - y2(limited)
    assert (np.all(np.array(res) >= 0))
    return res


def subwindow(img, window, borderType=cv2.BORDER_CONSTANT):
    """
    获取真实的窗口大小
    :param img: 图像
    :param window: 输入窗口
    :param borderType: 边界填充类型
    :return: 返回修正之后的真实窗口大小
    """
    cutWindow = [x for x in window]
    limit(cutWindow, [0, 0, img.shape[1], img.shape[0]])  # modify cutWindow
    assert (cutWindow[2] > 0 and cutWindow[3] > 0)
    border = getBorder(window, cutWindow)
    res = img[cutWindow[1]:cutWindow[1] + cutWindow[3], cutWindow[0]:cutWindow[0] + cutWindow[2]]

    if border != [0, 0, 0, 0]:  # 如果有超出则对cutWindow进行边界填充
        res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], borderType)
    return res
