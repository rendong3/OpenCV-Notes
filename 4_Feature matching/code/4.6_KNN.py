# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np

'''
        created on 09:05:10 2018-11-22
        @author ren_dong

        使用KNN即K近邻算法进行特征匹配
        ORB算法

'''

img1 = cv2.imread('orb1.jpg',0)  #测试图像img1
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('orb2.jpg',0)  ##训练图像img2
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = False)
#对每个匹配选择两个最佳的匹配
matches = bf.knnMatch(des1, des2, k=2)

print(type(matches), len(matches), matches[0])

# 获取img1中的第一个描述符即[0][]在img2中最匹配即[0][0]的一个描述符  距离最小
dMatch0 = matches[0][0]

# 获取img1中的第一个描述符在img2中次匹配的一个描述符  距离次之
dMatch1 = matches[0][1]
print('knnMatches', dMatch0.distance, dMatch0.queryIdx, dMatch0.trainIdx)
print('knnMatches', dMatch1.distance, dMatch1.queryIdx, dMatch1.trainIdx)
# 将不满足的最近邻的匹配之间距离比率大于设定的阈值匹配剔除。

img3 = None
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, img3, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
img3 = cv2.resize(img3,(1000, 400))
cv2.imshow('KNN',img3)
cv2.waitKey()
cv2.destroyAllWindows()

