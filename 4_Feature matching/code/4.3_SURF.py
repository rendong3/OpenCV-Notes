# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np

'''
        created on 09:05:10 2018-11-22
        @author ren_dong
        
        使用快速Hessian算法和SURF来提取和检测特征

'''
img = cv2.imread('chess.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#创建一个SURF对象
surf = cv2.xfeatures2d.SURF_create(20000)
#SURF算法使用Hessian算法计算关键点,并且在关键点周围区域计算特征向量,该函数返回关键点的信息和描述符
keypoints, descriptor = surf.detectAndCompute(gray, None)
print descriptor.shape
print len(keypoints)

img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints, flags=4, color=(51, 163, 236))

cv2.imshow('SURF', img)

cv2.waitKey()
cv2.destroyAllWindows()