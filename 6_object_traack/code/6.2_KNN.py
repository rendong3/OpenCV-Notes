#-*- coding: utf-8 -*-

import cv2
import numpy as np
'''
    背景分割器KNN实现运动检测
    @author 2018-1-23   18:53  

    KNN -->阈值化去除阴影-->膨胀去粗白色斑点-->寻找轮廓-->绘框并显示
'''
bs = cv2.createBackgroundSubtractorKNN(detectShadows = True)
camera = cv2.VideoCapture('1.mp4')

while True:
    ret, frame = camera.read()
    fgmask = bs.apply(frame)
    th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 2)

    image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) > 1600:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,0), 2)

    cv2.imshow("mog", fgmask)
    cv2.imshow("thresh", th)
    cv2.imshow("detection", frame)

    if cv2.waitKey(30) & 0xff == 27:
        break
    camera.release()
    cv2.destroyAllWindows()