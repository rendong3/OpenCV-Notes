#-*- coding:utf-8 -*-
import cv2
import numpy as np
'''
        created on Tues jan 13:57:30 2018
        @author:ren_dong
        
                Hough变换检测直线
            
                cv2.HoughLine()
                cv2.HoughLines()
                
                
'''
#载入图片img
img = cv2.imread('line.jpg')

#灰度化  --> gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

##中值滤波去除噪声  图像平滑
gray = cv2.medianBlur(gray, ksize=3)

#Canny边缘检测
edges = cv2.Canny(gray, 50, 120)


#最小直线长度, 更短的直线会被消除
minLineLength = 10

#最大线段间隙,一条线段的间隙长度大于这个值会被认为是两条分开的线段
maxLineGap = 5

##Hough 变换检测直线
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength, maxLineGap)

for i in range(len(lines)):
    for x1, y1, x2, y2 in lines[i]:
        #给定两点  在原始图片绘制线段
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


cv2.imshow('edges', edges)
cv2.imshow('lines', img)


cv2.waitKey()
cv2.destroyAllWindows()