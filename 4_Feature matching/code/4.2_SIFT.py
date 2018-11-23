# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np

'''
        created on 08:05:10 2018-11-20
        @author ren_dong

        使用DoG和SIFT进行特征提取和描述

        cv2.SIFT.detectAndCompute(image, mask[, descriptors[, useProvidedKeypoints]]) → keypoints, descriptors
        
        cv2.drawKeypoints(image, keypoints[, outImage[, color[, flags]]]) → outImage
        
        首先创建了一个SIFT对象，SIFT对象会使用DoG检测关键点，并且对每个关键点周围区域计算特征向量。
        detectAndCompute()函数会返回关键点信息(每一个元素都是一个对象，有兴趣的可以看一下OpenCV源码)和关键点的描述符。
        然后，我们在图像上绘制关键点，并显示出来。

'''
img = cv2.imread('chess.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#创建sift对象
sift = cv2.xfeatures2d.SIFT_create()

#进行检测和计算  返回特征点信息和描述符
keypoints , descriptor = sift.detectAndCompute(gray, None)
#keypoints：特征点集合list，向量内每一个元素是一个KeyPoint对象，包含了特征点的各种属性信息；



#绘制关键点
img = cv2.drawKeypoints(img, keypoints=keypoints, outImage=img, color= (51, 163, 236), flags= cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#sift得到的图像为128维的特征向量集

print len(keypoints)
print descriptor.shape

cv2.imshow('sift_keypoints',img)
cv2.waitKey()
cv2.destroyAllWindows()