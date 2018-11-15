#-*- coding:utf-8 -*-
import cv2
import numpy as np
'''
        created on Tues jan 09:36:30 2018
        @author:ren_dong

        cv2.boundingRect()          边界框即直边界矩形
        cv2.minAreaRect()           最小矩形区域即旋转的边界矩形
        cv2.minEnclosingCircle()    最小闭圆

'''
#载入图像img
img = cv2.pyrDown(cv2.imread('star.jpg', cv2.IMREAD_UNCHANGED))

#灰度化
gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

#二值化
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

#寻找轮廓
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print('hierarchy[0]:', hierarchy[0])
print('len(contours):', len(contours))
print('hierarchy.shape:', hierarchy.shape)

#遍历每一个轮廓
for C in contours:
    #计算边界框坐标
    x, y, w, h = cv2.boundingRect(C)

    ##在Img图像上绘制矩形框,颜色为green, 线宽为2
    cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)

    #计算包围目标的最小矩形区域
    rect = cv2.minAreaRect(C)

    #计算最小矩形的坐标
    box = cv2.boxPoints(rect)

    #坐标变为整数
    box = np.int0(box)

    #绘制矩形框轮廓  颜色为red  线宽为3
    cv2.drawContours(img, [box], 0, (0,0,255),3)

    #最小闭圆的圆心和半径
    (x,y),radius = cv2.minEnclosingCircle(C)

    #转换为整型
    center = (int(x), int(y))
    radius = int(radius)

    #绘制最小闭圆
    img = cv2.circle(img,center, radius, (255, 0, 0), 2)



cv2.drawContours(img,contours, -1, (0, 0, 255), 1)
cv2.imshow('contours',img)


cv2.waitKey()
cv2.destroyAllWindows()