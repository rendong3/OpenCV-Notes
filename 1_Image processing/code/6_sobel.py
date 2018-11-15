#-*- coding: utf-8 -*-
'''
        sobel算子   uint8         8维无符号数
                    cv2.cv_16s    16维有符号数

'''

import cv2

img = cv2.imread('1.jpg', 0)

x = cv2.Sobel(img,cv2.CV_16S,1,0)    #对x方向进行求导1,0
y = cv2.Sobel(img,cv2.CV_16S,0,1)    #对y方向进行求导0,1,选用cv_16s  避免图像显示不全

absX = cv2.convertScaleAbs(x)        ##对x,和y方向进行还原uint8,用来显示图片
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX,0.5,absY,0.5,0)

cv2.imshow("absX", absX)
cv2.imshow("absY", absY)

cv2.imshow("Result", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()