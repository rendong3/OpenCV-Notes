#-*- coding:utf-8 -*-
import cv2
import numpy as np
'''
       created on Tues jan 14:37:10 2018
        @author:ren_dong
        
                Hough变换检测圆
            
                cv2.HoughCricles()


'''
#载入图片img
img = cv2.imread('plant.jpg')

#灰度化处理 --> gray image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

##中值滤波  图像平滑
gray = cv2.medianBlur(gray, 7)

##Hough变换检测圆
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 200,
                           param1=200, param2=20, minRadius= 0, maxRadius=0 )

circles = np.uint16(np.around(circles))


##绘制圆
for i in circles[0,:]:
    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 2)


cv2.imshow('HoughCircles',img)


cv2.waitKey()
cv2.destroyAllWindows()
