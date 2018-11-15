#-*- coding: utf-8 -*-
import numpy as np
import cv2

'''
        自定义函数进行边缘检测
    1 使用medianBlur()作为模糊函数,去除彩色图像的噪声
    2 灰度化
    3 使用Laplacian()作为边缘检测函数,产生边缘线条
    4 归一化 并进行边缘和背景的黑白转换(之前是背景黑,边缘白,乘以原图像将边缘变黑)

'''

def strokeEdge(src,  blurKsize = 7, edgeKsize = 5):
    '''

    :param src:          原图像  BGR彩色空间
    :param blurKsize:    模糊滤波器kernel size    k<3  不进行模糊操作
    :param edgeKsize:    边缘检测kernel size
    :return:             dst


    '''
    if blurKsize >= 3:
        ##中值模糊
        blurredSrc = cv2.medianBlur(src, blurKsize)
        #灰度化
        gray = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        #灰度化
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    cv2.imshow('gray',gray)
   ###边缘检测
    laplacian = cv2.Laplacian(gray, cv2.CV_8U, gray, ksize= edgeKsize)
    cv2.imshow('Laplacian',laplacian)
   ##归一化   转换背景
    normalizeInverse = (1.0/255)*(255- gray)
    cv2.imshow('normalizeInverse', normalizeInverse)
   ##分离通道
    channels = cv2.split(src)

    ##计算后的结果与每个通道相乘
    for channel in channels:
        ##这里是点乘  即对应原素相乘
        channel[:] = channel * normalizeInverse

    cv2.imshow('B',channels[0])
    #合并通道
    return  cv2.merge(channels)


img = cv2.imread('1.jpg',cv2.IMREAD_COLOR)
dst = strokeEdge(img)
cv2.imshow('dst',dst)
cv2.waitKey()
cv2.destroyAllWindows()