#-*_ coding:utf-8 -*-
import cv2
import numpy as np
'''
        created on Tues jan 08:28:51 2018
        @author: ren_dong
        
        contour detection
        cv2.findContours()    寻找轮廓
        cv2.drawContours()    绘制轮廓


'''
#加载图像img
img = cv2.imread('person.jpg')
cv2.imshow('origin', img)
'''

灰度化处理,注意必须调用cv2.cvtColor(),
如果直接使用cv2.imread('1.jpg',0),会提示图像深度不对,不符合cv2.CV_8U


'''
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)

#调用cv2.threshold()进行简单阈值化,由灰度图像得到二值化图像
#  输入图像必须为单通道8位或32位浮点型
ret, thresh = cv2.threshold(gray, 127, 255, 0)
cv2.imshow('thresh', thresh)

#调用cv2.findContours()寻找轮廓,返回修改后的图像,轮廓以及他们的层次
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('image', image)


print('contours[0]:',contours[0])
print('len(contours):',len(contours))
print('hierarchy.shape:',hierarchy.shape)
print('hierarchy:',hierarchy)

#调用cv2.drawContours()在原图上绘制轮廓
img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
cv2.imshow('contours', img)



cv2.waitKey()
cv2.destroyAllWindows()