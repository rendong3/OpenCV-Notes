# -*- coding:utf-8 -*-
import cv2
import numpy as np

'''

       created on  19:18:27 2018-11-16

       @author:ren_dong

       haar级联分类器实现视频中的人脸检测
    
       打开摄像头，读取帧，检测帧中的人脸，扫描检测到的人脸中的眼睛，对人脸绘制蓝色的矩形框，对人眼绘制绿色的矩形框

       
       cv2.CascadeClassifier([filename]) → <CascadeClassifier object>
        
       cv2.CascadeClassifier.detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]


'''


def DynamicDetect():

    # 创建一个级联分类器 加载一个 .xml 分类器文件. 它既可以是Haar特征也可以是LBP特征的分类器.
    face_cascade = cv2.CascadeClassifier('./cascade/haarcascade_frontalface_default.xml')

    eye_cascade = cv2.CascadeClassifier('./cascade/haarcascade_eye.xml')

    # 打开摄像头
    camera = cv2.VideoCapture(0)
    cv2.namedWindow('Dynamic')

    while (True):
        # 读取一帧图像
        ret, frame = camera.read()
        # 判断图片读取成功？
        #rect()函数会返回两个值,第一个值是布尔值,用来表明是否成功读取帧,第二个为帧本身
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 人脸检测
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                # 在原图像上绘制矩形
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                # 眼睛检测
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40, 40))
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (ex + x, ey + y), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

            cv2.imshow('Dynamic', frame)
            # 如果按下q键则退出
            if cv2.waitKey(100) & 0xff == ord('q'):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # filename = './image/img23.jpg'
    # StaticDetect(filename)
    DynamicDetect()

