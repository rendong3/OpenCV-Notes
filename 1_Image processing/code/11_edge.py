#coding=utf-8
import cv2
import numpy
'''
检测边缘
形态学检测边缘的原理很简单，在膨胀时，图像中的物体会想周围“扩张”；
腐蚀时，图像中的物体会“收缩”。比较这两幅图像，由于其变化的区域只发生在边缘。
所以这时将两幅图像相减，得到的就是图像中物体的边缘

'''

image = cv2.imread("1.jpg",0)
#构造一个3×3的结构元素
element = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
dilate = cv2.dilate(image, element)
erode = cv2.erode(image, element)

#将两幅图像相减获得边，第一个参数是膨胀后的图像，第二个参数是腐蚀后的图像
result = cv2.absdiff(dilate,erode)
#上面得到的结果是灰度图，将其二值化以便更清楚的观察结果
retval, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY)
#反色，即对二值图每个像素取反
result = cv2.bitwise_not(result)
#显示图像
cv2.imshow("result",result)
cv2.waitKey(0)
cv2.destroyAllWindows()
