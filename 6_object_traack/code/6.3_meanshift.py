#-*- coding:utf-8 -*-
import numpy as np
import  cv2
'''
    @author 2019-1-25 20:25
    Meanshift 均值漂移
    
    标记感兴趣区域-->HSV空间 -->计算直方图+归一化 --> 反向投影 --> meanshift -->绘框并显示

'''

#标记初始感兴趣的区域
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
r,h,c,w = 10, 200, 10, 20
track_window = (c,r,w,h)

roi = frame[r:r+h, c:c+w]
#BGR ---> HSV
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#构建mask,给定array上下界
mask = cv2.inRange(hsv_roi, np.array((100., 30, 32.)),np.array((180, 120, 255)))

#计算直方图
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])

#线性直方图归一化
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

#设定meanshift迭代停止条件
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)

        #对ROI进行meanshift 给定window和停止条件
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        #重新计算window 然后在原图绘制矩形框
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255, 2)
        cv2.imshow('img2', img2)

        k = cv2.waitKey(60) &0xff
        if k == 27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()






















