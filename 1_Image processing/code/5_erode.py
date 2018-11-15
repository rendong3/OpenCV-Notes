# coding=utf-8
import cv2
import numpy as np

'''
    腐蚀和膨胀
    腐蚀和膨胀的处理很简单，只需设置好结构元素，然后分别调用cv2.erode(...)和cv2.dilate(...)函数即可，
    其中第一个参数是需要处理的图像，第二个是结构元素。返回处理好的图像。
'''
img = cv2.imread('1.jpg', 0)
# OpenCV定义的结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# 腐蚀图像
eroded = cv2.erode(img, kernel)
# 显示腐蚀后的图像
cv2.imshow("Eroded Image", eroded)

# 膨胀图像
dilated = cv2.dilate(img, kernel)
# 显示膨胀后的图像
cv2.imshow("Dilated Image", dilated)
# 原图像
cv2.imshow("Origin", img)

# NumPy定义的结构元素
NpKernel = np.uint8(np.ones((3, 3)))
Nperoded = cv2.erode(img, NpKernel)
# 显示腐蚀后的图像
cv2.imshow("Eroded by NumPy kernel", Nperoded)

cv2.waitKey(0)
cv2.destroyAllWindows()
