# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 16:09:48 2018

@author: lenovo
"""

'''
FAST角点检测
'''
import cv2

'''1、加载图片'''
img1 = cv2.imread('chess.jpg')
img1 = cv2.resize(img1,dsize=(600,400))
image1 = img1.copy()


'''2、提取特征点'''
#创建一个FAST对象，传入阈值t  可以处理RGB色彩空间图像
fast = cv2.FastFeatureDetector_create(threshold=50)
keypoints1 = fast.detect(image1,None)
#在图像上绘制关键点
image1 = cv2.drawKeypoints(image=image1,keypoints = keypoints1,outImage=image1,color=(255,0,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#输出默认参数
print("Threshold: ", fast.getThreshold())
print("nonmaxSuppression: ", fast.getNonmaxSuppression())
print("neighborhood: ", fast.getType())
print("Total Keypoints with nonmaxSuppression: ", len(keypoints1))

#显示图像
cv2.imshow('fast_keypoints1',image1)
cv2.waitKey(20)

#关闭非极大值抑制
fast.setNonmaxSuppression(0)
keypoints1 = fast.detect(image1,None)
print("Total Keypoints without nonmaxSuppression: ", len(keypoints1))
image1 = cv2.drawKeypoints(image=image1,keypoints = keypoints1,outImage=image1,color=(255,0,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('fast_keypoints1 nms',image1)

cv2.waitKey(0)
cv2.destroyAllWindows()