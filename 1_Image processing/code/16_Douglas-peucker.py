#-*- coding:utf-8 -*-
import cv2
import numpy as np
'''
        created on Tues jan 10:49:30 2018
        @author:ren_dong
            
        凸轮廓和Douglas-Peucker算法
        
        cv2.approxPloyDP()
        CV2.arcLength()
        cv2.convexHull()

'''
#读入图像img
img = cv2.pyrDown(cv2.imread('arc.jpg', cv2.IMREAD_COLOR))

#resize
#img = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_CUBIC)

#创建空白图像,用来绘制多边形轮廓
curve = np.zeros(img.shape,np.uint8)

#灰度变换
gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

#使用定制kernel 进行中值滤波,去除一些噪声
kernel = np.ones((3,3),np.float32) / 9
#这里的-1表示目标图像和原图像具有同样的深度,比如cv2.CV_8U
gray = cv2.filter2D(gray,-1,kernel)

#阈值化   gray image --> binary image
# 输入图像必须为单通道8位或32位浮点型
# 这里使用cv2.THRESH_BINARY_INV  实现效果像素>125 设置为0(黑)  否则设置为255(白)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

#寻找轮廓  返回修改后的图像, 图像轮廓  以及他们的层次
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


##假设contours[0]为最大轮廓
cnt = contours[0]
max_area = cv2.contourArea(cnt)

##遍历contours, 和原始设置比较,获得最大轮廓区域
for i in contours:
    if cv2.contourArea(i) > max_area:
        cnt = i
        max_area = cv2.contourArea(cnt)
print('max_area:',max_area)

##获得轮廓周长
epsilon = 0.01 * cv2.arcLength(cnt, True)

#计算得到近似的多边形框
approx = cv2.approxPolyDP(cnt, epsilon, True)

#得到凸包
hull = cv2.convexHull(cnt)



print('contours', len(contours), type(contours))
print('cnt.shape', cnt.shape, type(cnt))
print('approx.shape', approx.shape, type(approx))
print('hull.shape', hull.shape, type(hull))



#在原图像得到原始的轮廓
cv2.drawContours(img, contours, -1, (255, 0 , 0),2)

#在空白图像中得到最大轮廓, 多边形轮廓, 凸包轮廓
cv2.drawContours(curve, [cnt], -1, (0, 0, 255), 1)  #
cv2.drawContours(curve, [hull], -1, (0, 255, 0), 2)  ##绿色多边形轮廓  线宽2
cv2.drawContours(curve, [approx], -1, (255, 0, 0), 3)  ##蓝色凸包轮廓  线宽3


cv2.imshow('contours',img)
cv2.imshow('all',curve)


cv2.waitKey()
cv2.destroyAllWindows()