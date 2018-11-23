#-*- coding:utf-8 -*-
import os
import cv2
import numpy as np
'''
        created on 08:05:10 2018-11-20
        @author ren_dong
        
        使用cornerHarris进行角点检测
        
        cv2.cornerHarris(src, blockSize, ksize, k[, dst[, borderType]]) → dst


'''
img = cv2.imread('chess.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray= np.float32(gray)

#第三个参数定义了角点检测的敏感度,其值必须是介于3和31之间的奇数
##dst为函数返回的浮点值图像,其中包含角点检测结果
#第二个参数值决定了标记角点的记号大小,参数值越小,记号越小
dst = cv2.cornerHarris(gray, 4, 23, 0.04)
print dst.shape
#在原图进行标记   阈值
img[dst > 0.01*dst.max()] = [0, 0, 255]

while(True):

    cv2.imshow('Harris', img)
    if cv2.waitKey(100) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
