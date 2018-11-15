#-*- coding: utf-8 -*-
import cv2
import numpy as np
'''
        Canny边缘检测
        1 使用高斯滤波器进行去除图像噪声
        2 计算图像梯度
        3 非极大值抑制
        4 滞后阈值即在检测边缘上使用双阈值去除假阳性
        5 分析所有边缘与其之间的连接,以保留真正的边缘消除不明显的边缘

'''
img = cv2.imread('1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(gray, (3,3), 0)
canny = cv2.Canny(img,20,100)

cv2.imshow('canny',canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

