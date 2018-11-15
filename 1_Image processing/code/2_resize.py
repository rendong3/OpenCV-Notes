#-*- coding:utf-8 -*-
import cv2
import numpy as np

'''
        几何变换

'''
# #resize
# img = cv2.imread('1.jpg')
#
# ##不指定具体尺寸, 按照比例进行resize
# # res = cv2.resize(img,None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
#
# ###直接给定缩放尺寸
# res = cv2.resize(img, (640,480),interpolation=cv2.INTER_LINEAR)
#
# while(1):
#     cv2.imshow('img',img)
#     cv2.imshow('res',res)
#
#     k = cv2.waitKey(1) & 0xff
#     if k == 27 :
#         break
#
# cv2.destroyWindow()

##rotation

img = cv2.imread('1.jpg')

rows, cols, ch = img.shape

###给定旋转矩阵 2*3
M  = cv2.getRotationMatrix2D((rows/2,cols/2), 30, 0.8)

dst = cv2.warpAffine(img, M, (2*rows,2*cols))

cv2.imshow('img',img)
cv2.imshow('res',dst)
cv2.waitKey(0)















