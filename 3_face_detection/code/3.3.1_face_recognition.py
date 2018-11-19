# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
'''

       created on  10:24:27 2018-11-18

       @author:ren_dong
        
       人脸识别  调用摄像头进行个人头像数据采集,作为数据库
       
       只是在原来检测人脸目标框的基础上,添加了resize()函数进行图像的resize,
       然后调用imwrite()函数进行保存到指定路径下
        
'''

def Generate_name():

    # data = input('data:')
    #
    # name = input('input your name:')
    #
    # path = os.path.join(data, name)
    #
    # if os.path.isdir(path):
    #     os.remove(path)
    #     os.removedirs(path)

    #
    # os.mkdir(path)

    face_cascade = cv2.CascadeClassifier('./cascade/haarcascade_frontalface_default.xml')

    camera = cv2.VideoCapture(0)
    cv2.namedWindow('myself')
    count  = 1

    while(True):

        ret, frame = camera.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = cv2.CascadeClassifier.detectMultiScale(face_cascade, gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                f = cv2.resize(gray[y:y+h, x:x+w], (92, 112))

                cv2.imwrite('./data/ren_dong/%s.pgm' % str(count), f)

                count += 1

            cv2.imshow('myself', frame)

            if cv2.waitKey(35) & 0xff == ord('q'):
                break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    Generate_name()