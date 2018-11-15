#-*- coding:utf-8 -*-
import cv2
import numpy as np

'''
    使用numpy索引制定图片通道,改变相应像素值,CV加载彩色图像为BGR

'''

img = cv2.imread('1.jpg')
print img.shape
print img.size
print img.dtype
img[:,:,2] = 0   ###0,1,2 = B, R, G
cv2.imshow('img',img)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()





























