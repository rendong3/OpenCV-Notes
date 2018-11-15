# -*- coding:utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

##使用matplotlib显示图像

##彩色图像使用OpenCV加载时是BGR模式，但是matplotlib是RGB模式。
'''
img = cv2.imread('000028.jpg', 0)
plt.imshow(img, cmap = 'gray', interpolation= 'bicubic')
plt.xticks([]),plt.yticks([])
plt.show()

'''

#使用opencv显示图像并保存


img = cv2.imread('000028.jpg', 3)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)     ##创建一个窗口，调整窗口大小
cv2.imshow('image', img)
k = cv2.waitKey(0)
if k == ord('q'):                               #wait for q key to exit
    cv2.destroyAllWindows()
elif k == ord('s'):                             #wait for 's' to save and exit
    cv2.imwrite('1.jpg', img)
    img1 = cv2.imread('1.jpg', 0)
    cv2.imshow('gray', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

